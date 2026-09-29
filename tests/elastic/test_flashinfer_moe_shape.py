#!/usr/bin/env python3
"""DeepEP dispatch/NVFP4 fused-MoE/combine test for the SM120 MoE shape.

The source FlashInfer shape is:

* 6,045 input tokens per GPU (48,360 routed tokens at top-k 8)
* hidden size 6,144
* intermediate size 2,304
* 128 experts in total
* top-k 8

For EP=2 this gives 64 local experts per GPU; for EP=4 it gives 32; for EP=8
it gives 16. DeepEP
dispatches BF16 activations. Each rank compacts the local expert IDs,
quantizes the received activations and its local expert weights to NVFP4, runs
``flashinfer.fused_moe.cutlass_fused_moe``, and combines the weighted local
expert outputs back to the source rank. Routing is deterministically load
balanced: every token selects the same number of experts on every rank, and
global expert assignment counts differ by at most one.

Examples:

    # EP=2: 64 experts/GPU, 6,045 tokens/GPU
    CUDA_VISIBLE_DEVICES=0,1 python tests/elastic/test_flashinfer_moe_shape.py \
        --num-processes 2

    # EP=4: 32 experts/GPU, 6,045 tokens/GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/elastic/test_flashinfer_moe_shape.py \
        --num-processes 4

    # EP=8: 16 experts/GPU, 6,045 tokens/GPU, copy-engine PCIe send
    python tests/elastic/test_flashinfer_moe_shape.py --num-processes 8 --pcie-mode ce

The script configures ``EP_DISABLE_GIN=1`` and ``NCCL_LSA_TEAM_SIZE`` itself.
``--pcie-mode`` selects the PCIe send path (``sm``: SM kernels, ``ce``:
``EP_PCIE_CE=1``, ``shm``: ``EP_PCIE_SHM=1``); by default the environment is
left untouched.
It is intentionally standalone and does not import ``tests/elastic/test_ep.py``.
Use ``--communication-only`` to retain the lightweight identity-expert path.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist

import deep_ep


SEQUENCE_LENGTH = 6045
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2304
TOTAL_EXPERTS = 128
TOP_K = 8
DEFAULT_NUM_SMS = 8
DEFAULT_EXPERT_ALIGNMENT = 128
DEFAULT_WARMUP_ITERATIONS = 2
DEFAULT_BENCHMARK_ITERATIONS = 5
DEFAULT_AUTOTUNE_CACHE = Path('/tmp/flashinfer_nvfp4_moe_sm120_autotune_cache.json')


@dataclass(frozen=True)
class TestConfig:
    num_processes: int
    sequence_length: int
    hidden_size: int
    intermediate_size: int
    total_experts: int
    top_k: int
    num_sms: int
    expert_alignment: int
    warmup_iterations: int
    benchmark_iterations: int
    seed: int
    master_addr: str
    master_port: int
    skip_correctness: bool
    communication_only: bool
    cache_only: bool
    autotune_cache: Path
    pcie_mode: Optional[str]

    @property
    def local_experts(self) -> int:
        return self.total_experts // self.num_processes

    @property
    def routed_tokens_per_rank(self) -> int:
        return self.sequence_length * self.top_k

    @property
    def local_top_k(self) -> int:
        return self.top_k // self.num_processes


@dataclass(frozen=True)
class TimingResult:
    name: str
    max_rank_latencies_ms: tuple[float, ...]
    # Maximum per-rank received BF16 bytes, used for the bandwidth column
    num_bytes: int = 0

    @property
    def average_ms(self) -> float:
        return statistics.fmean(self.max_rank_latencies_ms)


@dataclass
class NVFP4MoEState:
    cutlass_fused_moe: Callable
    fp4_quantize: Callable
    activation_type: object
    w1_q: torch.Tensor
    w2_q: torch.Tensor
    quant_scales: list[torch.Tensor]
    input_encode_scale: torch.Tensor
    output: torch.Tensor


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--intermediate-size', type=int, default=INTERMEDIATE_SIZE)
    parser.add_argument('--total-experts', type=int, default=TOTAL_EXPERTS)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--num-sms', type=int, default=DEFAULT_NUM_SMS)
    parser.add_argument('--expert-alignment', type=int, default=DEFAULT_EXPERT_ALIGNMENT)
    parser.add_argument('--warmup-iterations', type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument('--benchmark-iterations', type=int, default=DEFAULT_BENCHMARK_ITERATIONS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=8362)
    parser.add_argument('--skip-correctness', action='store_true')
    parser.add_argument(
        '--communication-only',
        action='store_true',
        help='skip FlashInfer and use weighted identity experts',
    )
    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='load FlashInfer autotune results but do not tune cache misses',
    )
    parser.add_argument(
        '--autotune-cache',
        type=Path,
        default=DEFAULT_AUTOTUNE_CACHE,
        help='base path for per-rank FlashInfer autotune cache files',
    )
    parser.add_argument(
        '--pcie-mode',
        choices=('sm', 'ce', 'shm'),
        default=None,
        help='PCIe send path (default: keep EP_PCIE_CE / EP_PCIE_SHM from the environment)',
    )
    return TestConfig(**vars(parser.parse_args()))


def validate_config(cfg: TestConfig) -> None:
    positive_fields = {
        'sequence length': cfg.sequence_length,
        'hidden size': cfg.hidden_size,
        'intermediate size': cfg.intermediate_size,
        'total experts': cfg.total_experts,
        'top-k': cfg.top_k,
        'number of SMs': cfg.num_sms,
        'expert alignment': cfg.expert_alignment,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive, got {value}')
    if cfg.total_experts % cfg.num_processes != 0:
        raise ValueError('total experts must be divisible by the EP size')
    if cfg.top_k > cfg.total_experts:
        raise ValueError('top-k cannot exceed the total number of experts')
    if cfg.top_k % cfg.num_processes != 0:
        raise ValueError('top-k must be divisible by the EP size for exact rank balance')
    if cfg.top_k // cfg.num_processes > cfg.local_experts:
        raise ValueError('a token cannot select more unique local experts than exist')
    if cfg.warmup_iterations < 0 or cfg.benchmark_iterations <= 0:
        raise ValueError('warmups must be non-negative and benchmark iterations positive')

    # The requested FlashInfer mapping always has 128 total experts.
    if cfg.total_experts == TOTAL_EXPERTS:
        expected_local_experts = {2: 64, 4: 32, 8: 16}[cfg.num_processes]
        if cfg.local_experts != expected_local_experts:
            raise AssertionError(
                f'expected {expected_local_experts} local experts, got {cfg.local_experts}'
            )


def initialize_distributed(local_rank: int, cfg: TestConfig) -> dist.ProcessGroup:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{cfg.master_addr}:{cfg.master_port}',
        world_size=cfg.num_processes,
        rank=local_rank,
        device_id=torch.device(f'cuda:{local_rank}'),
    )
    return dist.new_group(list(range(cfg.num_processes)))


def make_balanced_routing(local_rank: int, cfg: TestConfig, device: torch.device):
    """Create deterministic routing balanced across ranks and local experts.

    Each token selects ``top_k / EP`` experts from every rank. Within each
    destination rank, assignments follow one continuous round-robin stream
    across all source ranks. Therefore every destination rank receives exactly
    the same number of routed assignments and global per-expert counts differ
    by at most one (the best possible when the total is not divisible).
    """
    experts_per_destination = cfg.top_k // cfg.num_processes
    assignments_per_source_destination = (
        cfg.sequence_length * experts_per_destination
    )
    token_indices = torch.arange(
        cfg.sequence_length, dtype=torch.int64, device=device
    ).unsqueeze(1)
    slot_indices = torch.arange(
        experts_per_destination, dtype=torch.int64, device=device
    ).unsqueeze(0)

    expert_id_blocks = []
    for destination_rank in range(cfg.num_processes):
        # Continue the local-expert round robin where the preceding source rank
        # stopped, which minimizes aggregate imbalance across all sources.
        source_offset = local_rank * assignments_per_source_destination
        local_expert_ids = (
            source_offset + token_indices * experts_per_destination + slot_indices
        ) % cfg.local_experts
        expert_id_blocks.append(
            local_expert_ids + destination_rank * cfg.local_experts
        )

    expert_ids = torch.cat(expert_id_blocks, dim=1).to(deep_ep.topk_idx_t)
    routing_weights = torch.full(
        (cfg.sequence_length, cfg.top_k),
        1.0 / cfg.top_k,
        dtype=torch.float32,
        device=device,
    )
    return expert_ids, routing_weights


def make_inputs(local_rank: int, cfg: TestConfig):
    """Create random BF16 activations and deterministic balanced routing."""
    device = torch.device(f'cuda:{local_rank}')
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + local_rank)

    x = torch.randn(
        cfg.sequence_length,
        cfg.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    expert_ids, routing_weights = make_balanced_routing(local_rank, cfg, device)
    return x, expert_ids, routing_weights


def global_encode_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Return the global NVFP4 encode scale used by FlashInfer."""
    amax = tensor.abs().amax().to(torch.float32)
    return torch.where(amax > 0, (448.0 * 6.0) / amax, torch.ones_like(amax))


def quantize_expert_weights(fp4_quantize: Callable, weights: torch.Tensor):
    """Quantize each expert independently, matching the source SM120 test."""
    packed_experts = []
    scale_experts = []
    global_scales = []
    for expert_weights in weights:
        encode_scale = global_encode_scale(expert_weights)
        packed, block_scales = fp4_quantize(expert_weights, encode_scale)
        packed_experts.append(packed)
        scale_experts.append(block_scales)
        global_scales.append(encode_scale)
    return (
        torch.stack(packed_experts).contiguous(),
        torch.stack(scale_experts).contiguous(),
        torch.stack(global_scales).to(torch.float32).contiguous(),
    )


def get_rank_autotune_cache(cfg: TestConfig, local_rank: int) -> Path:
    """Use one file per process so concurrent autotuning cannot corrupt a cache."""
    base = cfg.autotune_cache.expanduser().resolve()
    suffix = base.suffix or '.json'
    stem = base.stem if base.suffix else base.name
    return base.with_name(
        f'{stem}.ep{cfg.num_processes}.rank{local_rank}{suffix}'
    )


def load_flashinfer_nvfp4():
    """Import and validate FlashInfer's dedicated SM120 fused-MoE backend."""
    try:
        import flashinfer
        from flashinfer import fp4_quantize
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import (
            ActivationType,
            get_cutlass_fused_moe_module,
        )
    except Exception as exc:
        raise RuntimeError(
            'FlashInfer NVFP4 imports failed; install flashinfer-python[cu13] '
            f'in this environment: {exc}'
        ) from exc

    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            f'FlashInfer NVFP4 path requires SM120, got capability {capability}'
        )
    module = get_cutlass_fused_moe_module('120')
    if not hasattr(module, 'cutlass_fused_moe'):
        raise RuntimeError('fused_moe_120 does not export cutlass_fused_moe')
    if not hasattr(flashinfer, 'autotune'):
        raise RuntimeError('this FlashInfer build does not expose flashinfer.autotune')
    return flashinfer, fp4_quantize, cutlass_fused_moe, ActivationType


def prepare_nvfp4_moe(
    local_rank: int,
    cfg: TestConfig,
    num_received: int,
) -> tuple[object, NVFP4MoEState, Path, float]:
    """Create and quantize the rank-local experts once before benchmarking."""
    flashinfer, fp4_quantize, cutlass_fused_moe, activation_type = (
        load_flashinfer_nvfp4()
    )
    device = torch.device(f'cuda:{local_rank}')
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 1000 + local_rank)
    started = time.perf_counter()

    # Materialize one BF16 weight tensor at a time to reduce setup peak memory.
    w1 = (
        torch.randn(
            cfg.local_experts,
            2 * cfg.intermediate_size,
            cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    w1_q, w1_sf, w1_gs = quantize_expert_weights(fp4_quantize, w1)
    del w1

    w2 = (
        torch.randn(
            cfg.local_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    w2_q, w2_sf, w2_gs = quantize_expert_weights(fp4_quantize, w2)
    del w2

    input_encode_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    fc2_encode_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    quant_scales = [
        input_encode_scale,
        w1_sf.view(torch.int32),
        1.0 / (input_encode_scale * w1_gs),
        fc2_encode_scale,
        w2_sf.view(torch.int32),
        1.0 / (fc2_encode_scale * w2_gs),
    ]
    output = torch.empty(
        num_received,
        cfg.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    cache_path = get_rank_autotune_cache(cfg, local_rank)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    state = NVFP4MoEState(
        cutlass_fused_moe=cutlass_fused_moe,
        fp4_quantize=fp4_quantize,
        activation_type=activation_type,
        w1_q=w1_q,
        w2_q=w2_q,
        quant_scales=quant_scales,
        input_encode_scale=input_encode_scale,
        output=output,
    )
    return flashinfer, state, cache_path, setup_seconds


def compact_local_routing(
    recv_expert_ids: torch.Tensor,
    recv_routing_weights: torch.Tensor,
    num_received: int,
    cfg: TestConfig,
    validate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove expert slots owned by other ranks before the local fused MoE."""
    ids = recv_expert_ids[:num_received]
    weights = recv_routing_weights[:num_received]
    valid = ids >= 0
    if validate:
        counts = valid.sum(dim=1)
        if not bool((counts == cfg.local_top_k).all()):
            raise AssertionError(
                'each dispatched row must contain exactly '
                f'{cfg.local_top_k} local experts'
            )
    local_ids = ids[valid].reshape(num_received, cfg.local_top_k)
    local_weights = weights[valid].reshape(num_received, cfg.local_top_k)
    return local_ids.to(torch.int32).contiguous(), local_weights.contiguous()


def quantize_dispatched_activation(
    state: NVFP4MoEState,
    recv_x: torch.Tensor,
    num_received: int,
):
    return state.fp4_quantize(
        recv_x[:num_received], state.input_encode_scale
    )


def run_nvfp4_moe(
    state: NVFP4MoEState,
    x_q: torch.Tensor,
    x_sf: torch.Tensor,
    local_expert_ids: torch.Tensor,
    local_routing_weights: torch.Tensor,
    cfg: TestConfig,
) -> torch.Tensor:
    """Run the local experts and return their weighted partial contribution."""
    state.cutlass_fused_moe(
        x_q,
        local_expert_ids,
        local_routing_weights,
        state.w1_q.view(torch.long),
        state.w2_q.view(torch.long),
        torch.bfloat16,
        quant_scales=state.quant_scales,
        input_sf=x_sf,
        output=state.output,
        activation_type=state.activation_type.Swiglu,
        tune_max_num_tokens=max(8192, state.output.shape[0]),
    )
    return state.output


def check_routing_balance(expert_ids, cfg: TestConfig) -> tuple[int, int]:
    """Verify exact rank balance and optimal global expert balance."""
    destination_ranks = expert_ids // cfg.local_experts
    local_rank_counts = torch.bincount(
        destination_ranks.flatten(), minlength=cfg.num_processes
    )
    expected_rank_count = cfg.routed_tokens_per_rank // cfg.num_processes
    if not bool((local_rank_counts == expected_rank_count).all()):
        raise AssertionError(
            f'rank {dist.get_rank()}: destination assignment counts are not balanced: '
            f'{local_rank_counts.tolist()}'
        )

    global_expert_counts = torch.bincount(
        expert_ids.flatten().to(torch.int64), minlength=cfg.total_experts
    )
    dist.all_reduce(global_expert_counts, op=dist.ReduceOp.SUM)
    minimum_count = int(global_expert_counts.min().item())
    maximum_count = int(global_expert_counts.max().item())
    if maximum_count - minimum_count > 1:
        raise AssertionError(
            f'global expert assignment counts differ by more than one: '
            f'min={minimum_count}, max={maximum_count}'
        )
    return minimum_count, maximum_count


def dispatch_once(buffer: deep_ep.ElasticBuffer, x, expert_ids, routing_weights, cfg: TestConfig):
    return buffer.dispatch(
        x=x,
        topk_idx=expert_ids,
        topk_weights=routing_weights,
        num_max_tokens_per_rank=cfg.sequence_length,
        num_experts=cfg.total_experts,
        expert_alignment=cfg.expert_alignment,
        num_sms=cfg.num_sms,
        num_qps=0,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
        do_handle_copy=True,
        do_cpu_sync=True,
    )


def make_identity_expert_output(recv_x, recv_expert_ids, recv_routing_weights):
    """Emulate local identity experts plus their routing-weight reduction.

    Non-expanded DeepEP dispatch produces one token row per destination rank,
    not one row per expert.  The local fused MoE is responsible for running all
    local experts and reducing their weighted outputs into that row before
    DeepEP combine sends it back to the source rank.
    """
    local_weight_sum = recv_routing_weights.masked_fill(
        recv_expert_ids < 0, 0
    ).sum(dim=1, dtype=torch.float32)
    return (recv_x.float() * local_weight_sum.unsqueeze(1)).to(torch.bfloat16)


def combine_once(buffer: deep_ep.ElasticBuffer, local_expert_output, recv_routing_weights, handle, cfg: TestConfig):
    return buffer.combine(
        x=local_expert_output,
        handle=handle,
        topk_weights=recv_routing_weights,
        num_sms=cfg.num_sms,
        num_qps=0,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )


def gather_inputs(x, expert_ids, routing_weights, cfg: TestConfig):
    """Gather source tensors so each rank can build its own dispatch reference."""
    gathered_x = [torch.empty_like(x) for _ in range(cfg.num_processes)]
    gathered_expert_ids = [torch.empty_like(expert_ids) for _ in range(cfg.num_processes)]
    gathered_routing_weights = [torch.empty_like(routing_weights) for _ in range(cfg.num_processes)]
    dist.all_gather(gathered_x, x)
    dist.all_gather(gathered_expert_ids, expert_ids)
    dist.all_gather(gathered_routing_weights, routing_weights)
    return gathered_x, gathered_expert_ids, gathered_routing_weights


def build_dispatch_reference(local_rank: int, x, expert_ids, routing_weights, cfg: TestConfig):
    """Build the expected tokens sent to the experts owned by ``local_rank``."""
    gathered_x, gathered_ids, gathered_weights = gather_inputs(
        x, expert_ids, routing_weights, cfg
    )
    expert_begin = local_rank * cfg.local_experts
    expert_end = expert_begin + cfg.local_experts

    ref_x = []
    ref_ids = []
    ref_weights = []
    ref_source_indices = []
    for source_rank, (source_x, source_ids, source_weights) in enumerate(
        zip(gathered_x, gathered_ids, gathered_weights)
    ):
        local_expert_mask = (source_ids >= expert_begin) & (source_ids < expert_end)
        token_mask = local_expert_mask.any(dim=1)
        selected_ids = source_ids[token_mask] - expert_begin
        selected_ids.masked_fill_(~local_expert_mask[token_mask], -1)

        ref_x.append(source_x[token_mask])
        ref_ids.append(selected_ids)
        ref_weights.append(source_weights[token_mask])
        source_token_indices = torch.arange(
            cfg.sequence_length, device=x.device, dtype=torch.int64
        )[token_mask]
        ref_source_indices.append(source_rank * cfg.sequence_length + source_token_indices)

    return (
        torch.cat(ref_x),
        torch.cat(ref_ids),
        torch.cat(ref_weights),
        torch.cat(ref_source_indices),
    )


def check_dispatch(
    local_rank: int,
    x,
    expert_ids,
    routing_weights,
    recv_x,
    recv_expert_ids,
    recv_routing_weights,
    handle,
    cfg: TestConfig,
) -> int:
    ref_x, ref_ids, ref_weights, ref_source_indices = build_dispatch_reference(
        local_rank, x, expert_ids, routing_weights, cfg
    )
    num_received = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    if num_received != ref_x.shape[0]:
        raise AssertionError(f'rank {local_rank}: received {num_received}, expected {ref_x.shape[0]}')
    if len(handle.num_recv_tokens_per_expert_list) != cfg.local_experts:
        raise AssertionError(
            f'rank {local_rank}: DeepEP exposed '
            f'{len(handle.num_recv_tokens_per_expert_list)} local experts, '
            f'expected {cfg.local_experts}'
        )

    actual_source_indices = handle.recv_src_metadata[:num_received, 0].to(torch.int64)
    actual_order = torch.argsort(actual_source_indices)
    reference_order = torch.argsort(ref_source_indices)
    if not torch.equal(actual_source_indices[actual_order], ref_source_indices[reference_order]):
        raise AssertionError(f'rank {local_rank}: dispatched source-token indices differ')

    actual_x = recv_x[:num_received][actual_order]
    actual_ids = recv_expert_ids[:num_received][actual_order]
    actual_weights = recv_routing_weights[:num_received][actual_order]
    expected_x = ref_x[reference_order]
    expected_ids = ref_ids[reference_order]
    expected_weights = ref_weights[reference_order]
    if not torch.equal(actual_x, expected_x):
        raise AssertionError(f'rank {local_rank}: dispatched BF16 activations differ')
    if not torch.equal(actual_ids, expected_ids):
        raise AssertionError(f'rank {local_rank}: dispatched expert IDs differ')

    # Weight slots for experts on other ranks are ignored by DeepEP and need not
    # be initialized, so compare only slots owned by this rank.
    valid_weight_mask = expected_ids >= 0
    if not torch.equal(actual_weights[valid_weight_mask], expected_weights[valid_weight_mask]):
        raise AssertionError(f'rank {local_rank}: dispatched routing weights differ')
    return num_received


def check_identity_combine(x, routing_weights, combined_x, combined_weights, local_rank: int) -> None:
    """Check that weighted identity expert outputs combine back to the input."""
    if not torch.equal(combined_weights, routing_weights):
        raise AssertionError(f'rank {local_rank}: combined routing weights differ')

    # Every dispatched expert returns its input unchanged. Since routing weights
    # are normalized, their weighted sum should reconstruct x. The reduction is
    # BF16, so use a small numerical tolerance instead of bitwise equality.
    torch.testing.assert_close(
        combined_x.float(),
        x.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=lambda message: f'rank {local_rank}: combine correctness failed: {message}',
    )


def check_nvfp4_pipeline(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    local_expert_output: torch.Tensor,
    combined_x: torch.Tensor,
    combined_weights: torch.Tensor,
    local_rank: int,
) -> None:
    """Smoke-check the real distributed MoE path without duplicating all weights."""
    if combined_x.shape != x.shape:
        raise AssertionError(
            f'rank {local_rank}: combined shape {tuple(combined_x.shape)} '
            f'differs from input {tuple(x.shape)}'
        )
    if not torch.equal(combined_weights, routing_weights):
        raise AssertionError(f'rank {local_rank}: combined routing weights differ')
    for name, tensor in (
        ('local NVFP4 MoE output', local_expert_output),
        ('combined NVFP4 MoE output', combined_x),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f'rank {local_rank}: {name} contains NaN or Inf')
        if int(torch.count_nonzero(tensor)) == 0:
            raise AssertionError(f'rank {local_rank}: {name} is entirely zero')


def benchmark(
    name: str,
    operation: Callable,
    cfg: TestConfig,
    num_bytes: int = 0,
) -> tuple[object, TimingResult]:
    result: Optional[object] = None
    for _ in range(cfg.warmup_iterations):
        result = operation()
    torch.cuda.synchronize()

    local_latencies_ms = []
    for _ in range(cfg.benchmark_iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        local_latencies_ms.append(float(start.elapsed_time(end)))

    latency_tensor = torch.tensor(local_latencies_ms, dtype=torch.float64, device='cuda')
    gathered_latencies = [torch.empty_like(latency_tensor) for _ in range(cfg.num_processes)]
    dist.all_gather(gathered_latencies, latency_tensor)
    per_iteration_max = torch.stack(gathered_latencies).max(dim=0).values.cpu().tolist()
    num_bytes_tensor = torch.tensor(num_bytes, dtype=torch.int64, device='cuda')
    dist.all_reduce(num_bytes_tensor, op=dist.ReduceOp.MAX)
    timing = TimingResult(
        name=name,
        max_rank_latencies_ms=tuple(per_iteration_max),
        num_bytes=int(num_bytes_tensor.item()),
    )
    if dist.get_rank() == 0:
        bandwidth = (
            f', {timing.num_bytes / timing.average_ms / 1e6:.2f} GB/s'
            if timing.num_bytes else ''
        )
        print(
            f'{name}: max-rank avg={timing.average_ms:.3f} ms, '
            f'min={min(per_iteration_max):.3f} ms, '
            f'max={max(per_iteration_max):.3f} ms{bandwidth}, '
            f'iterations={cfg.benchmark_iterations}',
            flush=True,
        )
    return result, timing


def print_timing_summary(timings: list[TimingResult]) -> None:
    if dist.get_rank() != 0:
        return
    print('\nFull MoE stage timing (maximum rank per iteration)')
    print(f'  {"stage":<28} {"avg ms":>10} {"min ms":>10} {"max ms":>10} {"GB/s":>8}')
    for timing in timings:
        values = timing.max_rank_latencies_ms
        bandwidth = (
            f'{timing.num_bytes / timing.average_ms / 1e6:>8.2f}'
            if timing.num_bytes else f'{"-":>8}'
        )
        print(
            f'  {timing.name:<28} {timing.average_ms:>10.3f} '
            f'{min(values):>10.3f} {max(values):>10.3f} {bandwidth}'
        )


def run_full_nvfp4_pipeline(
    buffer: deep_ep.ElasticBuffer,
    state: NVFP4MoEState,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    cfg: TestConfig,
):
    dispatch_result = dispatch_once(
        buffer, x, expert_ids, routing_weights, cfg
    )
    recv_x, recv_ids, recv_weights, handle, _ = dispatch_result
    num_received = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    local_ids, local_weights = compact_local_routing(
        recv_ids, recv_weights, num_received, cfg, validate=False
    )
    x_q, x_sf = quantize_dispatched_activation(state, recv_x, num_received)
    local_output = run_nvfp4_moe(
        state, x_q, x_sf, local_ids, local_weights, cfg
    )
    combined_x, combined_weights, event = combine_once(
        buffer, local_output, recv_weights, handle, cfg
    )
    return combined_x, combined_weights, event


@torch.inference_mode()
def run_rank(local_rank: int, cfg: TestConfig) -> None:
    group = initialize_distributed(local_rank, cfg)
    buffer: Optional[deep_ep.ElasticBuffer] = None
    try:
        x, expert_ids, routing_weights = make_inputs(local_rank, cfg)
        min_expert_count, max_expert_count = check_routing_balance(expert_ids, cfg)
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=cfg.sequence_length,
            hidden=cfg.hidden_size,
            num_topk=cfg.top_k,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
        )
        logical_scaleout, logical_scaleup = buffer.get_logical_domain_size()
        if (logical_scaleout, logical_scaleup) != (1, cfg.num_processes):
            raise AssertionError(
                f'rank {local_rank}: expected logical domain 1 x {cfg.num_processes}, '
                f'got {logical_scaleout} x {logical_scaleup}'
            )

        dispatch_result = dispatch_once(buffer, x, expert_ids, routing_weights, cfg)
        recv_x, recv_ids, recv_weights, handle, _ = dispatch_result
        num_received = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        if not cfg.skip_correctness:
            num_received = check_dispatch(
                local_rank,
                x,
                expert_ids,
                routing_weights,
                recv_x,
                recv_ids,
                recv_weights,
                handle,
                cfg,
            )

        if cfg.communication_only:
            local_expert_output = make_identity_expert_output(
                recv_x, recv_ids, recv_weights
            )
            combine_result = combine_once(
                buffer, local_expert_output, recv_weights, handle, cfg
            )
            combined_x, combined_weights, _ = combine_result
            if not cfg.skip_correctness:
                check_identity_combine(
                    x, routing_weights, combined_x, combined_weights, local_rank
                )

            print(
                f'rank {local_rank}: input_tokens={cfg.sequence_length}, '
                f'received_tokens={num_received}, local_experts={cfg.local_experts}, '
                f'mode=communication-only, '
                f'correctness={"skipped" if cfg.skip_correctness else "PASS"}',
                flush=True,
            )
            timings = []
            dispatch_result, dispatch_timing = benchmark(
                'dispatch',
                lambda: dispatch_once(
                    buffer, x, expert_ids, routing_weights, cfg
                ),
                cfg,
                num_received * cfg.hidden_size * 2,
            )
            timings.append(dispatch_timing)
            recv_x, recv_ids, recv_weights, handle, _ = dispatch_result
            local_expert_output = make_identity_expert_output(
                recv_x, recv_ids, recv_weights
            )
            _, combine_timing = benchmark(
                'combine',
                lambda: combine_once(
                    buffer, local_expert_output, recv_weights, handle, cfg
                ),
                cfg,
                num_received * cfg.hidden_size * 2,
            )
            timings.append(combine_timing)
            print_timing_summary(timings)
        else:
            flashinfer, moe_state, cache_path, setup_seconds = prepare_nvfp4_moe(
                local_rank, cfg, num_received
            )
            max_setup_seconds = torch.tensor(
                setup_seconds, dtype=torch.float64, device=x.device
            )
            dist.all_reduce(max_setup_seconds, op=dist.ReduceOp.MAX)
            if local_rank == 0:
                print(
                    f'FlashInfer {getattr(flashinfer, "__version__", "unknown")}: '
                    f'local weight preparation '
                    f'max-rank={float(max_setup_seconds):.3f} s (one-time)',
                    flush=True,
                )
            print(
                f'rank {local_rank}: FlashInfer autotune cache={cache_path}',
                flush=True,
            )

            # Each process owns a separate cache file. This avoids concurrent
            # writers while retaining the same tactics on subsequent runs.
            with flashinfer.autotune(
                not cfg.cache_only, cache=str(cache_path)
            ):
                local_ids, local_weights = compact_local_routing(
                    recv_ids, recv_weights, num_received, cfg, validate=True
                )
                x_q, x_sf = quantize_dispatched_activation(
                    moe_state, recv_x, num_received
                )
                local_expert_output = run_nvfp4_moe(
                    moe_state,
                    x_q,
                    x_sf,
                    local_ids,
                    local_weights,
                    cfg,
                )
                combined_x, combined_weights, _ = combine_once(
                    buffer, local_expert_output, recv_weights, handle, cfg
                )
                if not cfg.skip_correctness:
                    check_nvfp4_pipeline(
                        x,
                        routing_weights,
                        local_expert_output,
                        combined_x,
                        combined_weights,
                        local_rank,
                    )

                print(
                    f'rank {local_rank}: input_tokens={cfg.sequence_length}, '
                    f'received_tokens={num_received}, '
                    f'local_experts={cfg.local_experts}, '
                    f'local_top_k={cfg.local_top_k}, mode=NVFP4, '
                    f'correctness={"skipped" if cfg.skip_correctness else "PASS"}',
                    flush=True,
                )

                timings = []
                dispatch_result, timing = benchmark(
                    '1. dispatch (BF16)',
                    lambda: dispatch_once(
                        buffer, x, expert_ids, routing_weights, cfg
                    ),
                    cfg,
                    num_received * cfg.hidden_size * 2,
                )
                timings.append(timing)
                recv_x, recv_ids, recv_weights, handle, _ = dispatch_result
                num_received = int(
                    handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
                )

                routing_result, timing = benchmark(
                    '2. compact local routing',
                    lambda: compact_local_routing(
                        recv_ids, recv_weights, num_received, cfg, validate=False
                    ),
                    cfg,
                )
                timings.append(timing)
                local_ids, local_weights = routing_result

                quantized_result, timing = benchmark(
                    '3. NVFP4 activation quantize',
                    lambda: quantize_dispatched_activation(
                        moe_state, recv_x, num_received
                    ),
                    cfg,
                )
                timings.append(timing)
                x_q, x_sf = quantized_result

                local_expert_output, timing = benchmark(
                    '4. NVFP4 fused MoE',
                    lambda: run_nvfp4_moe(
                        moe_state,
                        x_q,
                        x_sf,
                        local_ids,
                        local_weights,
                        cfg,
                    ),
                    cfg,
                )
                timings.append(timing)

                _, timing = benchmark(
                    '5. combine (BF16)',
                    lambda: combine_once(
                        buffer,
                        local_expert_output,
                        recv_weights,
                        handle,
                        cfg,
                    ),
                    cfg,
                    num_received * cfg.hidden_size * 2,
                )
                timings.append(timing)

                _, timing = benchmark(
                    '6. end-to-end pipeline',
                    lambda: run_full_nvfp4_pipeline(
                        buffer,
                        moe_state,
                        x,
                        expert_ids,
                        routing_weights,
                        cfg,
                    ),
                    cfg,
                )
                timings.append(timing)
                print_timing_summary(timings)

        dist.barrier()
        if local_rank == 0:
            print(
                f'global expert assignment balance: min={min_expert_count}, '
                f'max={max_expert_count}',
                flush=True,
            )
            mode = 'communication-only' if cfg.communication_only else 'full NVFP4 MoE'
            print(
                f'PASS: DeepEP {mode} pipeline completed.',
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def main() -> None:
    cfg = parse_args()
    validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if torch.cuda.device_count() < cfg.num_processes:
        raise RuntimeError(
            f'need {cfg.num_processes} visible GPUs, found {torch.cuda.device_count()}'
        )

    # Pure-PCIe DeepEP configuration. Set before child processes are spawned so
    # every rank observes the same values during NCCL/DeepEP initialization.
    os.environ['EP_DISABLE_GIN'] = '1'
    os.environ['NCCL_LSA_TEAM_SIZE'] = str(cfg.num_processes)
    if cfg.pcie_mode is not None:
        os.environ['EP_PCIE_CE'] = '1' if cfg.pcie_mode == 'ce' else '0'
        os.environ['EP_PCIE_SHM'] = '1' if cfg.pcie_mode == 'shm' else '0'
    # Keep JIT-generated sources and binaries off the slower workspace mount.
    # FlashInfer reads this variable during its first import in each child.
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE', '/tmp/flashinfer-deepep')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch-extensions-deepep')

    print('DeepEP FlashInfer-equivalent MoE shape')
    print(f'  EP size:                 {cfg.num_processes}')
    print(f'  sequence length / rank:  {cfg.sequence_length}')
    print(f'  routed tokens / rank:    {cfg.routed_tokens_per_rank}')
    print(f'  hidden size:             {cfg.hidden_size}')
    print(f'  intermediate size:       {cfg.intermediate_size}')
    print(f'  total experts:           {cfg.total_experts}')
    print(f'  local experts / rank:    {cfg.local_experts}')
    print(f'  top-k:                   {cfg.top_k}')
    print(f'  experts selected/rank:   {cfg.top_k // cfg.num_processes} per token')
    print(f'  routing:                 deterministic balanced round robin')
    print(f'  routing weight:          {1.0 / cfg.top_k:.3f} per selected expert')
    print(f'  activation transport:    BF16')
    print(
        '  local expert compute:    '
        f'{"weighted identity" if cfg.communication_only else "FlashInfer SM120 NVFP4 fused MoE"}'
    )
    if not cfg.communication_only:
        print(f'  autotune cache base:      {cfg.autotune_cache}')
        print(f'  autotune cache-only:      {cfg.cache_only}')
    print(f'  EP_DISABLE_GIN:          {os.environ["EP_DISABLE_GIN"]}')
    print(f'  NCCL_LSA_TEAM_SIZE:      {os.environ["NCCL_LSA_TEAM_SIZE"]}')
    print(f'  EP_PCIE_CE:              {os.environ.get("EP_PCIE_CE", "0")}')
    print(f'  EP_PCIE_SHM:             {os.environ.get("EP_PCIE_SHM", "0")}')
    if not cfg.communication_only:
        print(f'  FlashInfer JIT base:      {os.environ["FLASHINFER_WORKSPACE_BASE"]}')
    torch.multiprocessing.spawn(run_rank, args=(cfg, ), nprocs=cfg.num_processes)


if __name__ == '__main__':
    main()
