import argparse
import json
from typing import List

import torch


def parse_int_list(value: str) -> List[int]:
    values = [int(item) for item in value.split(',') if item]
    if not values:
        raise argparse.ArgumentTypeError('expected a comma-separated list of integers')
    return values


@torch.inference_mode()
def benchmark_copy(src_rank: int,
                   dst_rank: int,
                   size_mib: int,
                   warmups: int,
                   num_tests: int,
                   check: bool) -> dict:
    assert torch.cuda.can_device_access_peer(src_rank, dst_rank)
    num_elements = size_mib * 1024 * 1024 // torch.empty(
        (), dtype=torch.uint8).element_size()
    src = torch.full(
        (num_elements,), src_rank + 1, dtype=torch.uint8,
        device=f'cuda:{src_rank}')
    dst = torch.zeros(
        (num_elements,), dtype=torch.uint8, device=f'cuda:{dst_rank}')

    with torch.cuda.device(dst_rank):
        stream = torch.cuda.current_stream()
        for _ in range(warmups):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(num_tests):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
        end.synchronize()
        elapsed_s = start.elapsed_time(end) / 1e3 / num_tests

    if check:
        assert torch.all(dst == src_rank + 1)

    num_bytes = src.numel() * src.element_size()
    return {
        'src_rank': src_rank,
        'dst_rank': dst_rank,
        'size_mib': size_mib,
        'time_us': elapsed_s * 1e6,
        'gbps': num_bytes / elapsed_s / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure one-way CUDA peer copies between every visible GPU pair')
    parser.add_argument('--sizes-mib', type=parse_int_list, default=[16, 64, 256])
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    num_devices = torch.cuda.device_count()
    if num_devices < 2:
        parser.error('at least two visible CUDA devices are required')

    for size_mib in args.sizes_mib:
        for src_rank in range(num_devices):
            for dst_rank in range(num_devices):
                if src_rank == dst_rank:
                    continue
                if not torch.cuda.can_device_access_peer(src_rank, dst_rank):
                    parser.error(
                        f'CUDA peer access is unavailable for {src_rank} -> {dst_rank}')
                result = benchmark_copy(
                    src_rank, dst_rank, size_mib,
                    args.warmups, args.num_tests, args.check)
                print(f'RESULT_JSON {json.dumps(result, sort_keys=True)}',
                      flush=True)


if __name__ == '__main__':
    main()
