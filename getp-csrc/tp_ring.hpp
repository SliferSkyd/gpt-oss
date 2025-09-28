// tp_ring.hpp
#pragma once
#include "nccl.hpp"

struct RingChunk {
  size_t elem0;   // starting element index in the *global logical* array
  size_t nelems;  // number of elements in this chunk
};

// Split [0..N) into TP nearly-equal chunks
inline void split_even(size_t N, int TP, std::vector<RingChunk>& out) {
  out.resize(TP);
  size_t base = N / TP, rem = N % TP, off = 0;
  for (int r = 0; r < TP; ++r) {
    size_t n = base + (r < rem ? 1 : 0);
    out[r] = {off, n};
    off += n;
  }
}

inline void ring_allgather_rows(float*       my_dst,
                                const float* my_src,
                                int R_local, int C,
                                float* const peer_dst_bases[],   // size >= TP
                                const TPGroup& tp, hipStream_t s)
{
  const int TP = tp.tp_size;
  if (R_local == 0 || C == 0) return;

  const int my   = tp.rank_in_group;
  const int prev = (my - 1 + TP) % TP;
  const size_t cols_bytes  = (size_t)C * sizeof(float);
  const size_t block_bytes = (size_t)R_local * cols_bytes;

  // --- put my rows into my slot [my] once
  HIP_CHECK(hipMemcpyAsync(my_dst + (size_t)my * (size_t)R_local * C,
                           my_src, block_bytes, hipMemcpyDeviceToDevice, s));

  // **NEW:** ensure self-copy finished on the device before anyone reads it
  HIP_CHECK(hipStreamSynchronize(s));
  tp_group_barrier(tp); // align before step 0 (everyone's [my] is ready)

  // Tile ~4 MiB
  const size_t TILE_ROWS = std::max<size_t>(1, (4u<<20) / ((size_t)C*sizeof(float)));
  const int tiles = (int)((R_local + TILE_ROWS - 1)/TILE_ROWS);

  int recv_rank = prev;
  for (int step = 0; step < TP-1; ++step) {
    tp_group_barrier(tp); // keep phases aligned

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
                                     bytes, s));
    }
    HIP_CHECK(hipStreamSynchronize(s));   // lockstep across steps
    recv_rank = (recv_rank - 1 + TP) % TP;
  }
}

inline void ring_allgather_rows_bf16(__hip_bfloat16*       my_dst,
                                     const __hip_bfloat16* my_src,
                                     int R_local, int C,
                                     __hip_bfloat16* const peer_dst_bases[], // size >= TP
                                     const TPGroup& tp, hipStream_t s)
{
  const int TP = tp.tp_size;
  if (R_local == 0 || C == 0 || TP <= 0) return;

  const int my   = tp.rank_in_group;
  const int prev = (my - 1 + TP) % TP;

  const size_t cols_bytes  = (size_t)C * sizeof(__hip_bfloat16);
  const size_t block_bytes = (size_t)R_local * cols_bytes;

  // Put my rows into my slot [my] once
  HIP_CHECK(hipMemcpyAsync(my_dst + (size_t)my * (size_t)R_local * C,
                           my_src, block_bytes, hipMemcpyDeviceToDevice, s));

  // Ensure self-copy finished before peers read it
  HIP_CHECK(hipStreamSynchronize(s));
  tp_group_barrier(tp); // align before step 0

  // Tile ~4 MiB
  const size_t TILE_ROWS = std::max<size_t>(1, (4u<<20) / ((size_t)C * sizeof(__hip_bfloat16)));
  const int tiles = (int)((R_local + TILE_ROWS - 1) / TILE_ROWS);

  int recv_rank = prev;
  for (int step = 0; step < TP - 1; ++step) {
    tp_group_barrier(tp); // keep phases aligned

    for (int t = 0; t < tiles; ++t) {
      const size_t r0    = (size_t)t * TILE_ROWS;
      const size_t rN    = std::min((size_t)R_local - r0, TILE_ROWS);
      const size_t bytes = rN * cols_bytes;

      const __hip_bfloat16* src_peer =
          peer_dst_bases[prev] + (size_t)recv_rank * (size_t)R_local * C + r0 * C;
      __hip_bfloat16* dst_local =
          my_dst + (size_t)recv_rank * (size_t)R_local * C + r0 * C;

      HIP_CHECK(hipMemcpyPeerAsync(dst_local, tp.world_rank,
                                   src_peer, tp.group_devs[prev],
                                   bytes, s));
    }
    HIP_CHECK(hipStreamSynchronize(s)); // lockstep across steps
    recv_rank = (recv_rank - 1 + TP) % TP;
  }
}


__global__ void vec_add_inplace(float* __restrict__ dst,
                                const float* __restrict__ src,
                                size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) dst[i] += src[i];
}

inline void launch_add_inplace(float* dst, const float* src, size_t n, hipStream_t s) {
  if (n == 0) return;
  const int T = 256;
  int G = (int)std::min<size_t>((n + T - 1) / T, 65535);
  vec_add_inplace<<<G, T, 0, s>>>(dst, src, n);
  HIP_CHECK(hipGetLastError());
}

// peer_bases[r] must be the base pointer to the SAME logical buffer on rank r
// (group-local index 0..TP-1). On the caller side you can fill this from
// gpu_transformers[group_base + r]->state.expert_output_partial_g.
inline void ring_allreduce_sum(float*           buf_local,     // my buffer base
                               float* const     peer_bases[],  // size >= tp.tp_size
                               size_t           N_elems,       // total floats
                               float* d_recv,
                               const TPGroup&   tp,
                               hipStream_t      s)
{
  const int TP   = tp.tp_size;
  if (TP <= 1 || N_elems == 0) return;

  int my_dev = tp.world_rank;
  const int my = tp.rank_in_group;
  const int prev = (my - 1 + TP) % TP;

  // Split into nearly-equal chunks
  std::vector<RingChunk> chunks;
  split_even(N_elems, TP, chunks);

  // ----------------------
  // Phase 1: reduce-scatter
  // ----------------------
  // Canonical ring (pull from prev). After TP-1 steps, the chunk fully reduced on
  // this rank is idx = (my + 1) % TP.
  int recv_idx = (my - 1 + TP) % TP;
  const int TILE_ELEMS = RING_TILE_BYTES / sizeof(float);
  for (int step = 0; step < TP - 1; ++step) {
    tp_group_barrier(tp); // align steps across ranks (cheap CPU barrier)

    const RingChunk rc = chunks[recv_idx];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take = min(TILE_ELEMS, rc.nelems - done);
      const size_t off  = rc.elem0 + done;
      const size_t bytes= take * sizeof(float);

      HIP_CHECK(hipMemcpyPeerAsync(d_recv, my_dev,
                                    peer_bases[prev] + off, tp.group_devs[prev],
                                    bytes, s));
      // Accumulate into my local storage for this chunk
      launch_add_inplace(buf_local + off, d_recv, take, s);
      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s)); // keep phases in lockstep
    recv_idx = (recv_idx - 1 + TP) % TP;
  }

  // ----------------------
  // Phase 2: all-gather
  // ----------------------
  const int own_idx = (my + 1) % TP; // fully reduced chunk index held locally now
  int gather_recv = (own_idx - 1 + TP) % TP;

  for (int step = 0; step < TP - 1; ++step) {
    tp_group_barrier(tp);

    const RingChunk rc = chunks[gather_recv];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take = min(TILE_ELEMS, rc.nelems - done);
      const size_t off  = rc.elem0 + done;
      const size_t bytes= take * sizeof(float);

      // Pull the ready reduced chunk tile from prev into the correct slot of my buf
      HIP_CHECK(hipMemcpyPeerAsync(buf_local + off, my_dev,
                                    peer_bases[prev] + off, tp.group_devs[prev],
                                    bytes, s));
      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s));
    gather_recv = (gather_recv - 1 + TP) % TP;
  }
}


#ifndef RING_TILE_BYTES
#define RING_TILE_BYTES (4u<<20)  // 4 MiB tiles (works well on BF16)
#endif

// ---- BF16 add (tile-wise), fp32 math then round-to-bf16 ----
__global__ void vec_add_inplace_bf16(__hip_bfloat16* __restrict__ dst,
                                     const __hip_bfloat16* __restrict__ src,
                                     size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    float a = __bfloat162float(dst[i]);
    float b = __bfloat162float(src[i]);
    dst[i] = __float2bfloat16(a + b);
  }
}

inline void launch_add_inplace_bf16(__hip_bfloat16* dst,
                                    const __hip_bfloat16* src,
                                    size_t n,
                                    hipStream_t s)
{
  if (n == 0) return;
  const int T = 256;
  int G = (int)std::min<size_t>((n + T - 1) / T, 65535);
  vec_add_inplace_bf16<<<G, T, 0, s>>>(dst, src, n);
  HIP_CHECK(hipGetLastError());
}

// ---------------- Ring AllReduce (BF16): reduce-scatter + all-gather ----------------
//
//  - peer_bases[r] must be the base pointer to the *same logical buffer* on rank r.
//  - buf_local is the local instance of that buffer (same size and layout).
//  - d_recv is a BF16 scratch tile, capacity >= RING_TILE_BYTES.
//
// After the call, buf_local contains the elementwise sum across all ranks.
//
inline void ring_allreduce_sum_bf16(__hip_bfloat16*        buf_local,    // my buffer base (BF16)
                                    __hip_bfloat16* const  peer_bases[], // size >= tp.tp_size (BF16)
                                    size_t                 N_elems,      // total BF16 elements
                                    __hip_bfloat16*        d_recv,       // scratch recv tile (BF16)
                                    const TPGroup&         tp,
                                    hipStream_t            s)
{
  const int P = tp.tp_size;
  if (P <= 1 || N_elems == 0) return;

  const int my   = tp.rank_in_group;
  const int prev = (my - 1 + P) % P;

  // Split into nearly-equal contiguous chunks across the logical array
  std::vector<RingChunk> chunks;
  split_even(N_elems, P, chunks);

  const size_t TILE_ELEMS = RING_TILE_BYTES / sizeof(__hip_bfloat16);

  // ----------------------
  // Phase 1: reduce-scatter (pull from prev; accumulate in place)
  // After P-1 steps, rank r holds reduced chunk idx = (r + 1) % P.
  // ----------------------
  int recv_idx = (my - 1 + P) % P;             // step 0 target
  for (int s_step = 0; s_step < P - 1; ++s_step) {
    tp_group_barrier(tp);                       // align steps across ranks

    const RingChunk rc = chunks[recv_idx];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take  = std::min(TILE_ELEMS, rc.nelems - done);
      const size_t off   = rc.elem0 + done;
      const size_t bytes = take * sizeof(__hip_bfloat16);

      // Pull the tile for chunk recv_idx from PREV (prev finished this exact chunk in prior step)
      HIP_CHECK(hipMemcpyPeerAsync(
          d_recv, tp.world_rank,
          peer_bases[prev] + off, tp.group_devs[prev],
          bytes, s));

      // Accumulate into my local copy at the same offset
      launch_add_inplace_bf16(buf_local + off, d_recv, take, s);

      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s));         // ensure my buffer is updated before next step
    recv_idx = (recv_idx - 1 + P) % P;          // next chunk comes in descending order
  }

  // ----------------------
  // Phase 2: all-gather (pull reduced chunks from prev into final positions)
  // At this point, rank r fully owns chunk idx 'own_idx' = (r + 1) % P.
  // We now rotate reduced chunks so everyone gets all of them.
  // ----------------------
  int own_idx     = (my + 1) % P;
  int gather_recv = (own_idx - 1 + P) % P;     // first reduced chunk I still need

  for (int s_step = 0; s_step < P - 1; ++s_step) {
    tp_group_barrier(tp);                      // align before each gather hop

    const RingChunk rc = chunks[gather_recv];
    size_t done = 0;
    while (done < rc.nelems) {
      const size_t take  = std::min(TILE_ELEMS, rc.nelems - done);
      const size_t off   = rc.elem0 + done;
      const size_t bytes = take * sizeof(__hip_bfloat16);

      // Pull the ready reduced chunk tile from PREV into my final location
      HIP_CHECK(hipMemcpyPeerAsync(
          buf_local + off, tp.world_rank,
          peer_bases[prev] + off, tp.group_devs[prev],
          bytes, s));

      done += take;
    }
    HIP_CHECK(hipStreamSynchronize(s));        // ensure the tile is visible to my next peer
    gather_recv = (gather_recv - 1 + P) % P;
  }
}
