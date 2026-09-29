#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/pcie_shm.cuh>
#include <deep_ep/common/ptx.cuh>

#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          int kNumRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          bool kStagedSend,
          bool kPcieShm, int kNumPullWarps,
          int kNumThreads = (kNumWarps + kNumPullWarps) * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_impl(nv_bfloat16* x,
             float* topk_weights,
             int* src_metadata, int* psum_num_recv_tokens_per_scaleup_rank,
             const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* buffer, void* workspace,
             const int rank_idx,
             int num_reduced_tokens,
             void* staging,
             const pcie_shm::Args shm,
             const topk_idx_t* combined_topk_idx,
             const int* dst_buffer_slot_idx,
             const int num_combined_tokens) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto raw_warp_idx = ptx::get_warp_idx();
    const bool is_pull_warp = raw_warp_idx >= kNumWarps;
    const auto warp_idx = (raw_warp_idx + rank_idx) % kNumWarps;
    const auto lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Staged-send mode (PCIe copy engines): the token received at dispatch slot `(src rank, slot)` is written into a
    // local staging buffer at the same position (`[src rank][slot]`), the host then copies each `[src rank]` block
    // into the source rank's landing buffer, and the reduce epilogue locates tokens via `dst_buffer_slot_idx`
    EP_STATIC_ASSERT(not kStagedSend or (kIsScaleupNVLink and not kDoExpandedSend), "Invalid staged-send configuration");

    // Host-memory mode (see `pcie_shm::Args`): the same `[src rank][slot]` layout as the staged mode, but written
    // directly into the source ranks' GPU buffers (near ranks) or host segments (far ranks, then pulled by the
    // source ranks' pull warps)
    EP_STATIC_ASSERT(not kPcieShm or (kIsScaleupNVLink and not kDoExpandedSend and
                                      not kStagedSend and kNumRanks <= 32 and kNumPullWarps > 0),
                     "Invalid host-memory configuration");
    EP_STATIC_ASSERT(kPcieShm or kNumPullWarps == 0, "Pull warps are only for the host-memory mode");

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);

    // Buffer layouts
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);
    const auto send_buffer = layout::BufferLayout<false>(
        token_layout, kNumRanks,
        kNumMaxTokensPerRank * (kDoExpandedSend ? kNumTopk : 1),
        recv_buffer.get_buffer_end_ptr());
    const auto staging_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, staging);
    const auto shm_staged_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);

    // Init TMA
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (not is_pull_warp) {
        if (ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        __syncwarp();
    }

    // Expanding send mode must not be backward
    if constexpr (kDoExpandedSend)
        EP_DEVICE_ASSERT(topk_weights == nullptr);

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Full barrier to ensure the remote buffer is available
    // NOTES: the host-memory epoch is read before the first barrier, and only updated by SM 0 after the last one
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const int shm_epoch = kPcieShm ? ptx::ld_volatile<int>(workspace_layout.get_pcie_shm_epoch_ptr()) + 1 : 0;
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Do TMA writes into the remote buffers
    int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, kNumSMs * kNumWarps);
    const int token_start_idx = num_tokens_per_warp * global_warp_idx;
    const int token_end_idx = is_pull_warp ? 0 : min(token_start_idx + num_tokens_per_warp, num_reduced_tokens);
    // NOTES: in the host-memory mode, lane 0 publishes the flags of the host-memory stores in batches
    pcie_shm::PendingFlags shm_pending_flags;
    const auto wait_buffer_release = []() {
        if constexpr (kPcieShm) {
            ptx::tma_store_wait_read();
        } else {
            ptx::tma_store_wait();
        }
    };
    // NOTES: the host-memory mode visits tokens ordered by `(slot, source rank)` and interleaved among warps, so that
    // each source rank receives its tokens in slot order (as the pull warps expect), concurrent writes spread over all
    // source ranks, and warps are balanced even with skewed per-rank token counts
    int shm_count = 0, shm_psum_start = 0;
    if constexpr (kPcieShm) {
        if (lane_idx < kNumRanks) {
            shm_psum_start = lane_idx > 0 ? __ldg(psum_num_recv_tokens_per_scaleup_rank + lane_idx - 1) : 0;
            shm_count = __ldg(psum_num_recv_tokens_per_scaleup_rank + lane_idx) - shm_psum_start;
        }
    }
    int shm_level = 0, shm_level_base = 0;
    for (int iter = 0; ; ++ iter) {
        int i;
        if constexpr (kPcieShm) {
            if (is_pull_warp)
                break;
            const int q = global_warp_idx + iter * kNumSMs * kNumWarps;
            uint32_t level_mask;
            while ((level_mask = __ballot_sync(0xffffffff, shm_count > shm_level)) != 0 and
                   q >= shm_level_base + __popc(level_mask)) {
                shm_level_base += __popc(level_mask);
                ++ shm_level;
            }
            if (level_mask == 0)
                break;
            for (int k = 0; k < q - shm_level_base; ++ k)
                level_mask &= level_mask - 1;
            i = __shfl_sync(0xffffffff, shm_psum_start, __ffs(level_mask) - 1) + shm_level;
        } else {
            i = token_start_idx + iter;
            if (i >= token_end_idx)
                break;
        }
        if constexpr (kPcieShm)
            shm_pending_flags.publish(shm_epoch, true);

        // The master slot index during dispatch
        constexpr int kMetadataStride = 2 + kNumTopk;
        const int src_token_idx = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;
        const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);
        const int src_rank_idx = src_rank_topk_idx / kNumTopk;
        const int src_topk_idx = src_rank_topk_idx % kNumTopk;

        // Directly to the remote or via RDMA
        const bool nvlink_bypass = gin.is_nvlink_accessible<team_t>(src_rank_idx);
        layout::TokenLayout master_token_buffer = [=]() {
            // Host-memory mode: `[this rank][slot]` of the source rank's GPU buffer (near) or host segment (far)
            if constexpr (kPcieShm) {
                const int slot_idx = i - (src_rank_idx > 0 ? __ldg(psum_num_recv_tokens_per_scaleup_rank + src_rank_idx - 1) : 0);
                auto token_buffer = shm_staged_buffer.get_rank_buffer(rank_idx).get_token_buffer(slot_idx);
                token_buffer.set_base_ptr(pcie_shm::is_far(shm, src_rank_idx) ?
                    pcie_shm::get_host_ptr(shm, src_rank_idx, buffer, token_buffer.get_base_ptr()) :
                    gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
                return token_buffer;
            }

            // Local staging (tokens are received in `[src rank][slot]` order at dispatch)
            if constexpr (kStagedSend) {
                const int slot_idx = i - (src_rank_idx > 0 ? __ldg(psum_num_recv_tokens_per_scaleup_rank + src_rank_idx - 1) : 0);
                return staging_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(slot_idx);
            }

            // NVLink bypass
            if (nvlink_bypass) {
                auto token_buffer = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx).get_token_buffer(src_token_idx);
                token_buffer.set_base_ptr(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));
                return token_buffer;
            }

            // Use RDMA
            return send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx);
        }();

        // Hidden requirements
        EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);

        // Read source indices for expand mode
        int stored_topk_slot_idx = -1;
        if constexpr (kUseExpandedLayout) {
            if (lane_idx < kNumTopk)
                stored_topk_slot_idx = __ldg(src_metadata + i * kMetadataStride + (2 + lane_idx));
            __syncwarp();
        }

        // 3 cases:
        //  - no expand + no reduce, or expand + no reduce
        //  - expand + reduce
        //  - expand + send all
        auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
        auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
        if (no_local_reduce) {
            int token_idx_in_tensor = i;
            if constexpr (kUseExpandedLayout)
                token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

            // No reduce
            if (ptx::elect_one_sync()) {
                const auto load_ptr =
                    math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                wait_buffer_release();
                ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                ptx::tma_store_commit();
            }
            __syncwarp();
        } else if constexpr (kAllowMultipleReduction) {
            // Do local reduction
            // Sort valid top-k indices to front
            int topk_slot_idx[kNumTopk];
            compute_topk_slots(
                topk_slot_idx, reduce_valid_mask,
                [=](const int& idx) {
                    return ptx::exchange(stored_topk_slot_idx, idx);
                }
            );

            // Reduce into shared memory
            constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
            combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                /* Get source base */ [=](const int& slot_idx) {
                    return math::advance_ptr<combine_vec_t>(
                        x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                },
                /* Wait buffer release */ [=]() {
                    wait_buffer_release();
                    __syncwarp();
                }
            );
            ptx::tma_store_fence();
            __syncwarp();

            // Issue TMA stores
            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                ptx::tma_store_commit();
            }
            __syncwarp();
        } else {
            // No local reduction, send all data (expanded send)
            #pragma unroll
            for (int k = 0; k < kNumTopk; ++ k) {
                const auto slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                if (slot_idx >= 0) {
                    const auto src_token_ptr = math::advance_ptr<int4>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                    const auto token_buffer = recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx);
                    if (ptx::elect_one_sync()) {
                        // Load
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), src_token_ptr, mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        if (nvlink_bypass) {
                            // Write into the same position
                            ptx::tma_store_1d(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx),
                                              tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            ptx::tma_store_commit();
                        } else {
                            // Write to the RDMA send buffer
                            const auto send_token_buffer =
                                send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx * kNumTopk + k);
                            ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);
                            ptx::tma_store_commit();
                            ptx::tma_store_wait();

                            // Issue RDMA
                            gin.put<team_t>(token_buffer.get_base_ptr(), send_token_buffer.get_base_ptr(),
                                            kNumHiddenBytes, src_rank_idx);
                        }
                    }
                    __syncwarp();
                }
            }
        }

        // Write topk weights
        if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
            float value = 0;
            if constexpr (kUseExpandedLayout) {
                if (stored_topk_slot_idx >= 0)
                    value = __ldg(topk_weights + stored_topk_slot_idx);
            } else {
                value = __ldg(topk_weights + (i * kNumTopk + lane_idx));
            }
            master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;
        }
        __syncwarp();

        // Host-memory mode: the slot's flag is published once the stores of far ranks complete (in batches)
        if constexpr (kPcieShm) {
            if (pcie_shm::is_far(shm, src_rank_idx) and lane_idx == 0) {
                const int slot_idx = i - (src_rank_idx > 0 ? __ldg(psum_num_recv_tokens_per_scaleup_rank + src_rank_idx - 1) : 0);
                shm_pending_flags.add(pcie_shm::get_flag_ptr(shm, src_rank_idx, rank_idx, slot_idx, kNumMaxTokensPerRank));
            }
        }

        // Wait send buffer's TMA store and issue RDMA send
        // NOTES: `kDoExpandedSend` mode has already issued
        if (not kDoExpandedSend and not nvlink_bypass and ptx::elect_one_sync()) {
            ptx::tma_store_wait();
            const auto dst_ptr = recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)
                .get_token_buffer(src_token_idx).get_base_ptr();
            gin.put<team_t>(dst_ptr, master_token_buffer.get_base_ptr(),
                            master_token_buffer.get_num_bytes<false>(), src_rank_idx);
        }
    }

    if constexpr (kPcieShm)
        shm_pending_flags.publish(shm_epoch);

    // Pull warps: copy the tokens returned by far ranks from the local host segment
    if constexpr (kPcieShm) {
        if (is_pull_warp) {
            // The number of tokens sent to each far rank at dispatch (slots are dense), from the master slot indices,
            // scanned by all pull warps and reduced in the workspace (epoch-tagged, so that no cleaning is needed)
            constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
            constexpr int kNumGlobalPullWarps = kNumPullWarps * kNumSMs;
            const int global_pull_warp_idx = (raw_warp_idx - kNumWarps) * kNumSMs + sm_idx;
            int max_slots[kNumRanks] = {};
            for (int i = global_pull_warp_idx * 32 + lane_idx; i < num_combined_tokens * kNumTopk; i += kNumGlobalPullWarps * 32) {
                const auto expert_idx = static_cast<int>(__ldg(combined_topk_idx + i));
                const auto encoded_slot_idx = __ldg(dst_buffer_slot_idx + i);
                if (expert_idx >= 0 and encoded_slot_idx >= 0) {
                    #pragma unroll
                    for (int r = 0; r < kNumRanks; ++ r) {
                        if (r == expert_idx / kNumExpertsPerRank)
                            max_slots[r] = max(max_slots[r], encoded_slot_idx - rank_idx * kNumMaxTokensPerRank + 1);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < kNumRanks; ++ r) {
                const int value = __reduce_max_sync(0xffffffff, max_slots[r]);
                if (lane_idx == r and value > 0) {
                    atomicMax(reinterpret_cast<unsigned long long*>(workspace_layout.get_pcie_shm_combine_count_ptr(r)),
                              (static_cast<unsigned long long>(shm_epoch) << 32) | static_cast<uint32_t>(value));
                }
            }
            __syncwarp();

            // Wait for all pull warps
            if (lane_idx == 0) {
                __threadfence();
                atomicAdd(workspace_layout.get_pcie_shm_combine_arrival_ptr(), 1);
                while (ptx::ld_volatile<int>(workspace_layout.get_pcie_shm_combine_arrival_ptr()) != kNumGlobalPullWarps);
                __threadfence();
            }
            __syncwarp();
            int count = 0;
            if (lane_idx < kNumRanks and pcie_shm::is_far(shm, lane_idx)) {
                const auto value = ptx::ld_volatile<uint64_t>(workspace_layout.get_pcie_shm_combine_count_ptr(lane_idx));
                count = static_cast<int>(value >> 32) == shm_epoch ? static_cast<int>(value & 0xffffffffull) : 0;
            }

            pcie_shm::pull<kNumPullWarps * kNumSMs, kNumTimeoutCycles>(
                shm, count, buffer, shm_staged_buffer.get_num_bytes_per_rank(), token_layout.get_num_bytes<false>(),
                kNumMaxTokensPerRank, rank_idx, shm_epoch, global_pull_warp_idx, lane_idx);
        }
    }

    // Final barrier to ensure data arrival
    if constexpr (kStagedSend) {
        // Data arrival is signaled by the host-issued copies, only drain local stores here
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
        cooperative_groups::this_grid().sync();
    } else {
        comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                          kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kCombineTag1, true, true, false>(
            gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
    }

    // Finish the host-memory epoch
    if constexpr (kPcieShm) {
        if (sm_idx == 0 and thread_idx == 0) {
            *workspace_layout.get_pcie_shm_combine_arrival_ptr() = 0;
            *workspace_layout.get_pcie_shm_epoch_ptr() = shm_epoch;
        }
    }
}

}  // deep_ep::elastic
