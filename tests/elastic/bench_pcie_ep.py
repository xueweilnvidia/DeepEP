import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import count_bytes, per_token_cast_back, per_token_cast_to_fp8
from deep_ep.utils.testing import bench_kineto


TensorOrFP8 = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def parse_int_list(value: str) -> List[int]:
    values = [int(item) for item in value.split(',') if item]
    if not values:
        raise argparse.ArgumentTypeError('expected a comma-separated list of integers')
    return values


def make_remote_only_topk(num_tokens: int,
                          num_topk: int,
                          num_experts: int,
                          rank_idx: int,
                          num_ranks: int,
                          remote_fanout: int) -> torch.Tensor:
    """
    Construct a deterministic, balanced routing pattern with no local destinations.

    Every token visits ``remote_fanout`` remote ranks. Additional top-k lanes
    visit a different expert on the same remote rank.
    """
    assert num_ranks > 1
    assert num_experts % num_ranks == 0
    assert 1 <= remote_fanout <= num_ranks - 1
    assert num_topk >= remote_fanout
    assert num_topk % remote_fanout == 0

    num_experts_per_rank = num_experts // num_ranks
    num_copies_per_peer = num_topk // remote_fanout
    assert num_copies_per_peer <= num_experts_per_rank, \
        'not enough experts per rank to keep every top-k selection unique'

    token_idx = torch.arange(num_tokens, dtype=torch.int64, device='cuda').unsqueeze(1)
    topk_lane = torch.arange(num_topk, dtype=torch.int64, device='cuda').unsqueeze(0)
    dst_rank_idx = (rank_idx + 1 + topk_lane % remote_fanout) % num_ranks
    copy_idx = topk_lane // remote_fanout
    local_expert_idx = (token_idx * num_copies_per_peer + copy_idx) % num_experts_per_rank
    topk_idx = dst_rank_idx * num_experts_per_rank + local_expert_idx

    # A token cannot choose the same expert twice, and no destination may be local.
    assert torch.all(torch.sort(topk_idx, dim=1).values[:, 1:] !=
                     torch.sort(topk_idx, dim=1).values[:, :-1])
    assert torch.all(topk_idx // num_experts_per_rank != rank_idx)
    return topk_idx.to(deep_ep.topk_idx_t)


def make_case(num_tokens: int,
              hidden: int,
              num_topk: int,
              num_experts: int,
              rank_idx: int,
              num_ranks: int,
              remote_fanout: int,
              dtype: str,
              with_topk_weights: bool) -> Tuple[TensorOrFP8, torch.Tensor, Optional[torch.Tensor]]:
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x = per_token_cast_to_fp8(x_bf16) if dtype == 'fp8' else x_bf16
    topk_idx = make_remote_only_topk(
        num_tokens, num_topk, num_experts, rank_idx, num_ranks, remote_fanout)
    topk_weights = None
    if with_topk_weights:
        topk_weights = torch.full(
            (num_tokens, num_topk), 1.0 / num_topk,
            dtype=torch.float, device='cuda')
    return x, topk_idx, topk_weights


def gather_dict(local: Dict) -> List[Dict]:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return gathered


def summarize_profile(name: str,
                      local_main_s: float,
                      local_epilogue_s: float,
                      local_useful_bytes: int) -> Dict:
    gathered = gather_dict({
        'main_s': local_main_s,
        'epilogue_s': local_epilogue_s,
        'useful_bytes': local_useful_bytes,
    })
    max_main_s = max(item['main_s'] for item in gathered)
    max_epilogue_s = max(item['epilogue_s'] for item in gathered)
    total_useful_bytes = sum(item['useful_bytes'] for item in gathered)
    return {
        f'{name}_main_us_max': max_main_s * 1e6,
        f'{name}_main_us_mean': (
            sum(item['main_s'] for item in gathered) / len(gathered) * 1e6),
        f'{name}_epilogue_us_max': max_epilogue_s * 1e6,
        f'{name}_aggregate_useful_gbps': (
            total_useful_bytes / max_main_s / 1e9 if max_main_s > 0 else 0.0),
        f'{name}_per_rank_useful_gbps': (
            total_useful_bytes / len(gathered) / max_main_s / 1e9
            if max_main_s > 0 else 0.0),
    }


def run_correctness_smoke(buffer: deep_ep.ElasticBuffer,
                          args: argparse.Namespace,
                          dtype: str,
                          num_sms: int) -> None:
    num_tokens = min(args.tokens)
    x, topk_idx, topk_weights = make_case(
        num_tokens, args.hidden, args.num_topk, args.num_experts,
        buffer.rank_idx, buffer.num_ranks, args.remote_fanout,
        dtype, bool(args.with_topk_weights))
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_max_tokens_per_rank=args.num_max_tokens,
        num_experts=args.num_experts,
        expert_alignment=1,
        num_sms=num_sms,
        num_qps=0,
        do_handle_copy=False,
        do_cpu_sync=True,
        do_expand=False,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
    actual_num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    expected_num_recv_tokens = num_tokens * args.remote_fanout
    assert actual_num_recv_tokens == expected_num_recv_tokens, \
        f'{actual_num_recv_tokens=} != {expected_num_recv_tokens=}'
    assert recv_x[0].shape[0] == expected_num_recv_tokens \
        if isinstance(recv_x, tuple) else recv_x.shape[0] == expected_num_recv_tokens
    assert recv_topk_idx is not None
    assert (recv_topk_idx >= 0).sum().item() == num_tokens * args.num_topk

    recv_src_rank = (
        handle.recv_src_metadata[:, 0] // args.num_max_tokens) % buffer.num_ranks
    assert torch.all(recv_src_rank != buffer.rank_idx)

    recv_x_bf16 = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    combine_x = recv_x_bf16.clone()
    combined_x, combined_topk_weights, _ = buffer.combine(
        x=combine_x,
        topk_weights=recv_topk_weights,
        handle=handle,
        num_sms=num_sms,
        num_qps=0,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    assert combined_x.shape == (num_tokens, args.hidden)
    assert torch.isfinite(combined_x).all()
    if topk_weights is not None:
        assert torch.equal(combined_topk_weights, topk_weights)
    else:
        assert combined_topk_weights is None
    buffer.barrier(use_comm_stream=False, with_cpu_sync=True)


def profile_case(buffer: deep_ep.ElasticBuffer,
                 args: argparse.Namespace,
                 dtype: str,
                 num_tokens: int,
                 requested_num_sms: int) -> Dict:
    num_sms = (
        buffer.get_theoretical_num_sms(args.num_experts, args.num_topk)
        if requested_num_sms == 0 else requested_num_sms)
    assert num_sms <= torch.cuda.get_device_properties('cuda').multi_processor_count

    x, topk_idx, topk_weights = make_case(
        num_tokens, args.hidden, args.num_topk, args.num_experts,
        buffer.rank_idx, buffer.num_ranks, args.remote_fanout,
        dtype, bool(args.with_topk_weights))
    dispatch_args = dict(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_max_tokens_per_rank=args.num_max_tokens,
        num_experts=args.num_experts,
        expert_alignment=1,
        num_sms=num_sms,
        num_qps=0,
        do_handle_copy=False,
        do_cpu_sync=False,
        do_expand=False,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
    actual_num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    expected_num_recv_tokens = num_tokens * args.remote_fanout
    assert actual_num_recv_tokens == expected_num_recv_tokens

    input_payload_bytes_per_token = count_bytes(x, topk_idx, topk_weights) // num_tokens
    local_dispatch_bytes = actual_num_recv_tokens * input_payload_bytes_per_token
    recv_counts = gather_dict({'num_recv_tokens': actual_num_recv_tokens})
    row: Dict = {
        'dtype': dtype,
        'num_ranks': buffer.num_ranks,
        'num_tokens_per_rank': num_tokens,
        'num_max_tokens_per_rank': args.num_max_tokens,
        'hidden': args.hidden,
        'num_topk': args.num_topk,
        'num_experts': args.num_experts,
        'num_sms_requested': requested_num_sms,
        'num_sms': num_sms,
        'num_qps_argument': 0,
        'theoretical_num_qps_unused': buffer.get_theoretical_num_qps(num_sms),
        'with_topk_weights': bool(args.with_topk_weights),
        'remote_only_routing': True,
        'remote_fanout': args.remote_fanout,
        'num_recv_tokens_min': min(
            item['num_recv_tokens'] for item in recv_counts),
        'num_recv_tokens_max': max(
            item['num_recv_tokens'] for item in recv_counts),
    }

    if args.profile in ('all', 'dispatch'):
        dispatch_main_s, dispatch_copy_s = bench_kineto(
            lambda: buffer.dispatch(**dispatch_args),
            kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
            num_tests=args.num_tests,
            flush_l2=bool(args.flush_l2),
            barrier_comm_profiling=True,
            barrier=buffer.barrier)
        row.update(summarize_profile(
            'dispatch', dispatch_main_s, dispatch_copy_s, local_dispatch_bytes))

    if args.profile in ('all', 'cached'):
        cached_dispatch_args = dict(
            x=x,
            topk_weights=topk_weights,
            handle=handle,
            num_sms=num_sms,
            num_qps=0,
            do_expand=False,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
        )
        cached_main_s, cached_copy_s = bench_kineto(
            lambda: buffer.dispatch(**cached_dispatch_args),
            kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
            num_tests=args.num_tests,
            flush_l2=bool(args.flush_l2),
            barrier_comm_profiling=True,
            barrier=buffer.barrier)
        row.update(summarize_profile(
            'cached_dispatch', cached_main_s, cached_copy_s, local_dispatch_bytes))

    if args.profile in ('all', 'combine'):
        num_allocated_recv_tokens = (
            recv_x[0].shape[0] if isinstance(recv_x, tuple) else recv_x.shape[0])
        combine_x = torch.randn(
            (num_allocated_recv_tokens, args.hidden),
            dtype=torch.bfloat16, device='cuda')
        combine_payload_bytes_per_token = (
            combine_x[0].numel() * combine_x.element_size() +
            (args.num_topk * torch.tensor([], dtype=torch.float).element_size()
             if topk_weights is not None else 0))
        local_combine_bytes = actual_num_recv_tokens * combine_payload_bytes_per_token
        combine_args = dict(
            x=combine_x,
            topk_weights=recv_topk_weights,
            handle=handle,
            num_sms=num_sms,
            num_qps=0,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
        )
        combine_main_s, combine_epilogue_s = bench_kineto(
            lambda: buffer.combine(**combine_args),
            kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
            num_tests=args.num_tests,
            flush_l2=bool(args.flush_l2),
            barrier_comm_profiling=True,
            barrier=buffer.barrier)
        row.update(summarize_profile(
            'combine', combine_main_s, combine_epilogue_s, local_combine_bytes))

    buffer.barrier(use_comm_stream=False, with_cpu_sync=True)
    return row


@torch.inference_mode()
def worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    _, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    assert num_ranks == num_local_ranks, 'this benchmark only supports one node'
    assert os.getenv('EP_DISABLE_GIN') == '1', \
        'set EP_DISABLE_GIN=1 to guarantee a PCIe-only run'
    assert 1 <= args.remote_fanout <= num_ranks - 1
    assert args.num_topk >= args.remote_fanout
    assert args.num_topk % args.remote_fanout == 0
    assert args.num_max_tokens >= max(args.tokens)

    dtypes = ('bf16', 'fp8') if args.dtype == 'both' else (args.dtype,)
    for dtype in dtypes:
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=args.num_max_tokens,
            hidden=args.hidden,
            num_topk=args.num_topk,
            use_fp8_dispatch=dtype == 'fp8',
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            num_allocated_qps=0,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.num_gpu_timeout_secs,
            num_cpu_timeout_secs=args.num_cpu_timeout_secs)
        assert buffer.get_logical_domain_size() == (1, num_ranks)

        resolved_sms = [
            buffer.get_theoretical_num_sms(args.num_experts, args.num_topk)
            if value == 0 else value
            for value in args.num_sms
        ]
        if args.check:
            run_correctness_smoke(buffer, args, dtype, resolved_sms[0])
            if buffer.rank_idx == 0:
                print(json.dumps({
                    'event': 'correctness_smoke_passed',
                    'dtype': dtype,
                    'num_ranks': num_ranks,
                    'num_tokens': min(args.tokens),
                    'num_sms': resolved_sms[0],
                }, sort_keys=True), flush=True)

        for num_tokens in args.tokens:
            for requested_num_sms in args.num_sms:
                row = profile_case(
                    buffer, args, dtype, num_tokens, requested_num_sms)
                if buffer.rank_idx == 0:
                    print(f'RESULT_JSON {json.dumps(row, sort_keys=True)}', flush=True)
        buffer.destroy()

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark DeepEP direct EP kernels over PCIe-only CUDA P2P')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--tokens', type=parse_int_list, default=[4096],
                        help='comma-separated token counts')
    parser.add_argument('--num-max-tokens', type=int, default=8192)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--remote-fanout', type=int, default=0,
                        help='number of remote peers per token; 0 means all remote ranks')
    parser.add_argument('--num-sms', type=parse_int_list, default=[0],
                        help='comma-separated SM counts; 0 adds the automatic baseline')
    parser.add_argument('--dtype', choices=('bf16', 'fp8', 'both'), default='bf16')
    parser.add_argument('--profile', choices=('all', 'dispatch', 'cached', 'combine'),
                        default='all')
    parser.add_argument('--with-topk-weights', type=int, choices=(0, 1), default=1)
    parser.add_argument('--num-tests', type=int, default=10)
    parser.add_argument('--flush-l2', type=int, choices=(0, 1), default=1)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100)
    args = parser.parse_args()

    if os.getenv('EP_DISABLE_GIN') != '1':
        parser.error('EP_DISABLE_GIN=1 is required for a PCIe-only run')
    if args.num_processes < 2:
        parser.error('--num-processes must be at least 2')
    if args.num_experts % args.num_processes != 0:
        parser.error('--num-experts must be divisible by --num-processes')
    args.remote_fanout = (
        args.num_processes - 1 if args.remote_fanout == 0 else args.remote_fanout)
    if not 1 <= args.remote_fanout < args.num_processes:
        parser.error('--remote-fanout must be in [1, num_processes - 1]')
    if args.num_topk < args.remote_fanout:
        parser.error('--num-topk must be at least --remote-fanout')
    if args.num_topk % args.remote_fanout != 0:
        parser.error('--num-topk must be divisible by --remote-fanout')

    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    main()
