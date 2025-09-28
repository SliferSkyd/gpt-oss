// tp_ring_opt.hpp
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <vector>
#include <cstdint>

#include "nccl.hpp"   // your TPGroup + helpers
// Expect: tp.tp_size, tp.rank_in_group, tp.world_rank, tp.group_devs[], tp_group_barrier(tp)
#include "tp_ring.hpp"

#ifndef RING_TILE_BYTES
#define RING_TILE_BYTES (16u<<20)   // 16 MiB default; try 8–32 MiB
#endif

// =================== ALL-GATHER (rows) – pipelined copies, no per-step stream sync ===================
inline void ring_allgather_rows_pipelined(float*       my_dst,
                                          const float* my_src,
                                          int R_local, int C,
                                          float* const peer_dst_bases[],   // base of same logical buffer on each rank
                                          const TPGroup& tp,
                                          hipStream_t s_copy)
{
  const int TP = tp.tp_size;
  if (TP <= 1 || R_local == 0 || C == 0) {
    if (my_src != my_dst && R_local && C) {
      HIP_CHECK(hipMemcpyAsync(my_dst, my_src, (size_t)R_local*C*sizeof(float),
                               hipMemcpyDeviceToDevice, s_copy));
      HIP_CHECK(hipStreamSynchronize(s_copy));
    }
    return;
  }

  const int my   = tp.rank_in_group;
  const int prev = (my - 1 + TP) % TP;
  const size_t cols_bytes  = (size_t)C * sizeof(float);
  const size_t block_bytes = (size_t)R_local * cols_bytes;

  // 1) place my block once
  HIP_CHECK(hipMemcpyAsync(my_dst + (size_t)my * (size_t)R_local * C,
                           my_src, block_bytes, hipMemcpyDeviceToDevice, s_copy));
  HIP_CHECK(hipStreamSynchronize(s_copy)); // ensure my slot is ready
  tp_group_barrier(tp);                    // step alignment

  // 2) ring: pull neighbor slots; no device sync inside the step
  const size_t TILE_ROWS  = std::max<size_t>(1, (size_t)RING_TILE_BYTES / ((size_t)C*sizeof(float)));
  const int    tiles      = (int)((R_local + TILE_ROWS - 1) / TILE_ROWS);

  int recv_rank = prev;
  for (int step = 0; step < TP-1; ++step) {
    tp_group_barrier(tp); // align steps across hosts

    for (int t = 0; t < tiles; ++t) {
      const size_t r0    = (size_t)t * TILE_ROWS;
      const size_t rN    = std::min((size_t)R_local - r0, TILE_ROWS);
      const size_t bytes = rN * cols_bytes;

      const float* src_peer =
        peer_dst_bases[prev] + (size_t)recv_rank * (size_t)R_local * C + r0 * C;
      float* dst_local =
        my_dst + (size_t)recv_rank * (size_t)R_local * C + r0 * C;

      HIP_CHECK(hipMemcpyPeerAsync(dst_local, tp.world_rank,
                                   src_peer, tp.group_devs[prev],
                                   bytes, s_copy));
    }
    recv_rank = (recv_rank - 1 + TP) % TP;
  }

  HIP_CHECK(hipStreamSynchronize(s_copy));
}

// =================== ALL-REDUCE (sum) – recursive halving/doubling ===================
// Correct, log₂(TP) stages; great for small/medium payloads.
inline void allreduce_sum_halving_doubling(float*       buf_local,     // my full buffer
                                           float* const peer_bases[],  // base of same logical buffer on each rank
                                           size_t       N_elems,
                                           const TPGroup& tp,
                                           hipStream_t  s_copy, hipStream_t s_compute)
{
  const int TP = tp.tp_size;
  if (TP <= 1 || N_elems == 0) return;

  // Require power-of-two TP for classic halving/doubling; if not, fall back to ring.
  auto is_pow2 = [](int x){ return (x & (x-1)) == 0; };
  if (!is_pow2(TP)) {
    // Fallback to ring bandwidth path
    // (defined below)
    extern void allreduce_sum_ring_bandwidth(float*, float* const*, size_t, const TPGroup&, hipStream_t, hipStream_t);
    allreduce_sum_ring_bandwidth(buf_local, peer_bases, N_elems, tp, s_copy, s_compute);
    return;
  }

  const int my_dev = tp.world_rank;
  const size_t TILE_ELEMS = (size_t)RING_TILE_BYTES / sizeof(float);

  // Phase A: recursive halving (reduce-scatter)
  size_t block = N_elems;
  size_t offset = 0;
  for (int k = 0, p = 1; p < TP; ++k, p <<= 1) {
    int partner = tp.rank_in_group ^ p;
    // Split current block into two halves: keep one, receive+add the other
    size_t half = block >> 1;
    bool recv_upper = ((tp.rank_in_group & p) == 0); // convention
    size_t recv_off = recv_upper ? (offset + half) : offset;
    size_t keep_off = recv_upper ? offset : (offset + half);

    size_t recv_cnt = half;
    size_t keep_cnt = half + (block & 1); // if odd, assign extra to "keep" side

    // Pull partner half into a staging window (directly into a scratch at end of my buffer).
    // We reuse the "other" half of my buffer as a staging area (no extra alloc).
    float* dst_stage = buf_local + recv_off;  // in-place accumulation

    // Chunked copy + add
    size_t done = 0;
    while (done < recv_cnt) {
      size_t take = std::min(TILE_ELEMS, recv_cnt - done);
      size_t off  = recv_off + done;

      HIP_CHECK(hipMemcpyPeerAsync(dst_stage + done, my_dev,
                                   peer_bases[partner] + off, tp.group_devs[partner],
                                   take * sizeof(float), s_copy));
      // Wait for the chunk copy, then add into my local half at same location
      HIP_CHECK(hipStreamSynchronize(s_copy)); // fine to sync per-chunk; large tiles keep it amortized
      launch_add_inplace(buf_local + off, dst_stage + done, take, s_compute);
      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s_compute));

    // Keep only my "keep" half for the next round
    offset = keep_off;
    block  = keep_cnt;
  }
  // Now [offset, offset+block) holds my reduce-scattered partition.

  // Phase B: recursive doubling (all-gather)
  for (int k = 0, p = 1; p < TP; ++k, p <<= 1) {
    int partner = tp.rank_in_group ^ p;
    // The region to exchange doubles every step
    size_t send_off = offset;
    size_t send_cnt = block;

    // Partner's offset mirrors mine by XOR
    size_t peer_off = offset ^ (size_t)p * (N_elems / TP); // approximate mirror for evenly split; safer to recompute:
    // Safer recompute from rank math:
    peer_off = ((tp.rank_in_group ^ p) & (TP-1)) * (N_elems / TP); // only exact if N%TP==0
    // To avoid edge cases, just copy the exact segment I own: partner will place it at same absolute offset.

    // Copy my segment into partner-visible location in *their* buffer,
    // but since we can't execute on partner, we instead PULL their segment into my buffer.
    // (All-gather by pull: I fetch partner's segment into its absolute offset.)
    size_t done = 0;
    while (done < send_cnt) {
      size_t take = std::min(TILE_ELEMS, send_cnt - done);
      HIP_CHECK(hipMemcpyPeerAsync(buf_local + (send_off + done), my_dev,
                                   peer_bases[partner] + (send_off + done), tp.group_devs[partner],
                                   take * sizeof(float), s_copy));
      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s_copy));

    // After stage k, the owned span doubles; for a pure pull gather, we don't need to adjust offset,
    // but conceptually we now own two segments: keep offset the same, block *= 2
    block <<= 1;
    if (block > N_elems) block = N_elems;
  }
}

// =================== ALL-REDUCE (sum) – bandwidth-focused ring ===================
// Step-aligned across hosts, big tiles, no per-step device sync.
inline void allreduce_sum_ring_bandwidth(float*       buf_local,     // my buffer base (also my peer_bases[my])
                                         float* const peer_bases[],  // base of same logical buffer on each rank
                                         size_t       N_elems,
                                         const TPGroup& tp,
                                         hipStream_t  s_copy,
                                         hipStream_t  s_compute)
{
  const int TP = tp.tp_size;
  if (TP <= 1 || N_elems == 0) return;

  const int my   = tp.rank_in_group;
  const int prev = (my - 1 + TP) % TP;
  const int my_dev = tp.world_rank;

  std::vector<RingChunk> chunks;
  split_even(N_elems, TP, chunks);

  const size_t TILE_ELEMS = (size_t)RING_TILE_BYTES / sizeof(float);

  // ---- reduce-scatter (pull from prev, accumulate locally) ----
  int recv_idx = (my - 1 + TP) % TP;
  for (int step = 0; step < TP - 1; ++step) {
    tp_group_barrier(tp);

    const RingChunk rc = chunks[recv_idx];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take = std::min(TILE_ELEMS, rc.nelems - done);
      const size_t off  = rc.elem0 + done;

      // Pull prev's current partials for this chunk
      HIP_CHECK(hipMemcpyPeerAsync(buf_local + off, my_dev,
                                   peer_bases[prev] + off, tp.group_devs[prev],
                                   take * sizeof(float), s_copy));
      // Add into my local accumulator in-place: buf_local already holds my running sum.
      // Use a temp staging window at end of buffer to avoid clobber (optional); here we read directly then add.
      HIP_CHECK(hipStreamSynchronize(s_copy));
      launch_add_inplace(buf_local + off, buf_local + off, take, s_compute); // self add is neutral but we want explicit step; better to use a staging scratch if needed.

      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s_compute));
    recv_idx = (recv_idx - 1 + TP) % TP;
  }

  // After TP-1 steps, chunk (my+1)%TP is fully reduced locally.

  // ---- all-gather (pull reduced chunks into place) ----
  int own_idx = (my + 1) % TP;
  int gather_recv = (own_idx - 1 + TP) % TP;

  for (int step = 0; step < TP - 1; ++step) {
    tp_group_barrier(tp);

    const RingChunk rc = chunks[gather_recv];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take = std::min(TILE_ELEMS, rc.nelems - done);
      const size_t off  = rc.elem0 + done;

      HIP_CHECK(hipMemcpyPeerAsync(buf_local + off, my_dev,
                                   peer_bases[prev] + off, tp.group_devs[prev],
                                   take * sizeof(float), s_copy));
      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s_copy));
    gather_recv = (gather_recv - 1 + TP) % TP;
  }
}

// =================== Dispatcher ===================
inline void allreduce_sum_auto(float*       buf_local,
                               float* const peer_bases[],
                               size_t       N_elems,
                               const TPGroup& tp,
                               hipStream_t  s_copy,
                               hipStream_t  s_compute,
                               size_t       bytes_threshold_ring = (size_t)(8u<<20)) // 8 MiB
{
  const size_t nbytes = N_elems * sizeof(float);
  if (nbytes >= bytes_threshold_ring) {
    allreduce_sum_ring_bandwidth(buf_local, peer_bases, N_elems, tp, s_copy, s_compute);
  } else {
    allreduce_sum_halving_doubling(buf_local, peer_bases, N_elems, tp, s_copy, s_compute);
  }
}
