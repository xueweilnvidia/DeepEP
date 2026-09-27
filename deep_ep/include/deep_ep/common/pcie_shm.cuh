#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::pcie_shm {

// PCIe host-memory send (`EP_PCIE_SHM=1`): every rank owns a pinned host segment on its GPU's NUMA node, laid out
// exactly like its GPU buffer (plus per-`[src rank][slot]` arrival flags at `flags_offset`). For peers whose
// peer-to-peer path goes through the CPU, the sender writes a token into the receiver's host segment and then
// publishes the slot's flag (value = epoch), the receiver's pull warps copy arrived tokens into its GPU buffer
// within the same kernel, so that host writes and reads are pipelined. No CPU synchronization is involved.
struct Args {
    // Device array of `[rank]` host segment base pointers, `nullptr` if disabled
    void* const* segments;
    int64_t flags_offset;
    // Bit `r` set: rank `r` is reached via host memory (otherwise via direct peer-to-peer stores)
    uint32_t far_mask;
};

__forceinline__ __device__ __host__ bool is_far(const Args& args, const int& rank_idx) {
    return (args.far_mask >> rank_idx) & 1u;
}

#ifdef __CUDACC__

// The address in `rank_idx`'s host segment of `ptr`, which is an address inside the (identically laid out) local
// GPU buffer starting at `buffer`
__forceinline__ __device__ void* get_host_ptr(const Args& args, const int& rank_idx, const void* buffer, const void* ptr) {
    return math::advance_ptr(args.segments[rank_idx],
                             static_cast<const int8_t*>(ptr) - static_cast<const int8_t*>(buffer));
}

// The arrival flag of `[src_rank_idx][slot_idx]` in `rank_idx`'s host segment
__forceinline__ __device__ int* get_flag_ptr(const Args& args, const int& rank_idx,
                                             const int& src_rank_idx, const int& slot_idx,
                                             const int& num_max_tokens_per_rank) {
    return math::advance_ptr<int>(args.segments[rank_idx], args.flags_offset) +
           static_cast<int64_t>(src_rank_idx) * num_max_tokens_per_rank + slot_idx;
}

// Warp-cooperative copy of `num_bytes` (a multiple of 16) from host memory into GPU memory
__forceinline__ __device__ void warp_copy_from_host(void* dst, const void* src, const int& num_bytes, const int& lane_idx) {
    // NOTES: host reads are slow under concurrent host writes, keep as many reads in flight as possible
    constexpr int kUnroll = 16;
    const auto src_vec = static_cast<const int4*>(src);
    const auto dst_vec = static_cast<int4*>(dst);
    const int num_vecs = num_bytes / static_cast<int>(sizeof(int4));
    for (int i = lane_idx; i < num_vecs; i += 32 * kUnroll) {
        int4 values[kUnroll];
        #pragma unroll
        for (int k = 0; k < kUnroll; ++ k) {
            if (i + k * 32 < num_vecs) {
                asm volatile("ld.volatile.global.v4.s32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(values[k].x), "=r"(values[k].y), "=r"(values[k].z), "=r"(values[k].w)
                             : "l"(src_vec + i + k * 32));
            }
        }
        #pragma unroll
        for (int k = 0; k < kUnroll; ++ k) {
            if (i + k * 32 < num_vecs)
                dst_vec[i + k * 32] = values[k];
        }
    }
}

// Pull warps: copy the tokens arrived from far ranks (`[src rank][slot]`, slot-contiguous) from this rank's host
// segment into the identically laid out local GPU buffer; lane `r` holds the number of tokens to receive from rank `r`
// NOTES: chunks of 32 slots are ordered by slot range first, as tokens arrive in slot order from all ranks
template <int kNumGlobalPullWarps, int64_t kNumTimeoutCycles>
__forceinline__ __device__ void pull(const Args& args, const int& count,
                                     void* buffer, const int64_t& num_bytes_per_rank, const int& num_token_bytes,
                                     const int& num_max_tokens_per_rank,
                                     const int& rank_idx, const int& epoch,
                                     const int& global_pull_warp_idx, const int& lane_idx) {
    const int num_chunks = math::ceil_div(count, 32);
    int level = 0, level_base = 0;
    for (int chunk_idx = global_pull_warp_idx; ; chunk_idx += kNumGlobalPullWarps) {
        uint32_t level_mask;
        while ((level_mask = __ballot_sync(0xffffffff, num_chunks > level)) != 0 and
               chunk_idx >= level_base + __popc(level_mask)) {
            level_base += __popc(level_mask);
            ++ level;
        }
        if (level_mask == 0)
            break;
        for (int k = 0; k < chunk_idx - level_base; ++ k)
            level_mask &= level_mask - 1;
        const int src_rank_idx = __ffs(level_mask) - 1;
        const int slot_start = level * 32;
        const int num_slots = min(32, __shfl_sync(0xffffffff, count, src_rank_idx) - slot_start);

        // Wait arrivals: the chunk's flags are contiguous, poll them with one coalesced volatile load per warp
        // NOTES: like NCCL's SHM transport, the data is then read with volatile loads (no stale L1), without fences
        const auto flag_ptr = get_flag_ptr(args, rank_idx, src_rank_idx, slot_start + lane_idx, num_max_tokens_per_rank);
        const auto start_clock = clock64();
        while (not __all_sync(0xffffffff, lane_idx >= num_slots or ptx::ld_volatile<int>(flag_ptr) == epoch)) {
            if (clock64() - start_clock > kNumTimeoutCycles) {
                if (lane_idx < num_slots)
                    printf("DeepEP PCIe host-memory receive timeout, rank: %d, src: %d, slot: %d, flag: %d, expected: %d\n",
                           rank_idx, src_rank_idx, slot_start + lane_idx, ptx::ld_volatile<int>(flag_ptr), epoch);
                asm volatile("trap;");
            }
        }

        // The slots are contiguous, copy them at once
        const auto dst_ptr = math::advance_ptr(buffer, num_bytes_per_rank * src_rank_idx + static_cast<int64_t>(num_token_bytes) * slot_start);
        warp_copy_from_host(dst_ptr, get_host_ptr(args, rank_idx, buffer, dst_ptr), num_slots * num_token_bytes, lane_idx);
    }
}

// Arrival flags of the tokens a lane has TMA-stored into host memory, published in batches (like NCCL's SHM transport:
// one system fence, then relaxed flag stores), so that host writes do not need to complete token by token
// NOTES: the senders are not the bottleneck, larger batches delay the receivers (measured 8 tokens: -14%)
struct PendingFlags {
    static constexpr int kNumMaxPending = 1;
    int* ptrs[kNumMaxPending];
    int num_pending = 0;

    __forceinline__ __device__ void add(int* ptr) {
        ptrs[num_pending ++] = ptr;
    }

    // NOTES: must be called by the whole warp (the TMA stores may be issued by other lanes)
    __forceinline__ __device__ void publish(const int& epoch, const bool& only_if_full = false) {
        if (__any_sync(0xffffffff, only_if_full ? num_pending == kNumMaxPending : num_pending > 0)) {
            // Complete the TMA stores, and order them before the flag stores
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            ptx::tma_store_fence_global();
            __syncwarp();
            if (num_pending > 0) {
                ptx::fence_acq_rel_sys();
                #pragma unroll
                for (int i = 0; i < kNumMaxPending; ++ i) {
                    if (i < num_pending)
                        ptx::st_relaxed_sys(ptrs[i], epoch);
                }
            }
            num_pending = 0;
        }
        __syncwarp();
    }
};

#endif

}  // namespace deep_ep::elastic::pcie_shm
