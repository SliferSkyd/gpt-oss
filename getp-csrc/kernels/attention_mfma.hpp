#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <float.h>
#include <stdint.h>

#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif
#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-6f
#endif

#ifndef KV_BF16_KEEP_TOKENS
#define KV_BF16_KEEP_TOKENS 4
#endif

// 64 + padding to break LDS bank conflicts
constexpr int PADDED_HEAD_DIM = 72;

// 8-lane subgroup reductions (width=8)
__device__ inline float subgroup8_sum(float x)
{
  x += __shfl_xor(x, 4, 8);
  x += __shfl_xor(x, 2, 8);
  x += __shfl_xor(x, 1, 8);
  return x;
}

// ====================================================================================
// Single-warp (64 lanes) kernel: one KV head per block, processes its 8 query heads
// Streaming softmax: O = α*O + w*V_t, l = α*l + w, with per-head (m,l) state in lanes li==0
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64, y=1)   — 8 heads × 8 lanes/head
// ====================================================================================

__global__ __launch_bounds__(64, 8) void fused_attention_kernel_1warp8q_fast(
    float *__restrict__ output, const float *__restrict__ q,
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, const float * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int tile_t)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;
  const int lane = threadIdx.x; // 0..63
  const int hloc = lane >> 3;   // 0..7 (which Q head)
  const int li = lane & 7;      // 0..7 (lane-in-head)
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  int pos = seq_lengths[b];
  int t_start = (use_sw && ((layer_idx & 1) == 0)) ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;

  const int kv_dim = D * KVH;
  const __hip_bfloat16 *__restrict__ K0 =
      key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
  const __hip_bfloat16 *__restrict__ V0 =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

  // Pre-scale q by 1/sqrt(D) once
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      qseg[s] = qh[li + 8 * s] * inv;
  }

  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16 *sK = smem;
  __hip_bfloat16 *sV = sK + (size_t)tile_t * PADDED_HEAD_DIM;

  // streaming softmax state (per head, stored on li==0 lane)
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // vec load helper
  struct u128
  {
    uint4 v;
  };
  auto ld8 = [](__hip_bfloat16 const *p)
  { u128 r; r.v=*reinterpret_cast<uint4 const*>(p); return r; };

  for (int base = 0; base < n_steps; base += tile_t)
  {
    const int cur = min(tile_t, n_steps - base);

    // stage K/V
    const int vec8 = D / 8; // 8 bf16 = 16B
    const int tot = cur * vec8;
    for (int e = lane; e < tot; e += 64)
    {
      int tloc = e / vec8;
      int i8 = (e - tloc * vec8) * 8;
      int t_abs = t_start + base + tloc;
      int tw = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      *reinterpret_cast<u128 *>(sK + (size_t)tloc * PADDED_HEAD_DIM + i8) =
          ld8(K0 + (size_t)tw * kv_dim + i8);
      *reinterpret_cast<u128 *>(sV + (size_t)tloc * PADDED_HEAD_DIM + i8) =
          ld8(V0 + (size_t)tw * kv_dim + i8);
    }
    __syncthreads();

// tokens in this tile
#pragma unroll 1
    for (int t = 0; t < cur; ++t)
    {
      const __hip_bfloat16 *__restrict__ krow = sK + (size_t)t * PADDED_HEAD_DIM;
      const __hip_bfloat16 *__restrict__ vrow = sV + (size_t)t * PADDED_HEAD_DIM;

      // dot for this head across 64 dims split over 8 lanes
      float part = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        float k = __bfloat162float(krow[li + 8 * s]);
        part = fmaf(qseg[s], k, part); // q pre-scaled
      }
      // subgroup-8 sum
      part += __shfl_xor(part, 4, 8);
      part += __shfl_xor(part, 2, 8);
      part += __shfl_xor(part, 1, 8);

      // leader does alpha/w; broadcast to subgroup
      float alpha = 1.f, w = 0.f;
      if (li == 0)
      {
        float s = part; // already scaled
        float m_new = fmaxf(m, s);
        alpha = __expf(m - m_new);
        w = __expf(s - m_new);
        l = l * alpha + w;
        m = m_new;
      }
      alpha = __shfl(alpha, (hloc << 3), 64);
      w = __shfl(w, (hloc << 3), 64);

// numerator update for this lane's 8 dims
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        float v = __bfloat162float(vrow[li + 8 * s]);
        out8[s] = fmaf(w, v, out8[s] * alpha);
      }
    }
    __syncthreads();
  }

  // sink merge
  float alpha_sink = 1.f, l_final = 0.f;
  if (li == 0)
  {
    float ss = __bfloat162float(sinks[h]);
    float m_new = fmaxf(m, ss);
    float alpha = __expf(m - m_new);
    float e = __expf(ss - m_new);
    l = l * alpha + e;
    alpha_sink = alpha;
    l_final = l;
  }
  alpha_sink = __shfl(alpha_sink, (hloc << 3), 64);
  l_final = __shfl(l_final, (hloc << 3), 64);

  // store
  float *__restrict__ oh = output + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    int d = li + 8 * s;
    oh[d] = (out8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
  }
}

// =====================
// Flash-Decoding Phase 1: per-split partials (numerator + (m,l))
// Computes, for each (b, kv_h, split), the per-head unnormalized numerator vector n_s
//   and the streaming-softmax state (m_s, l_s). No sink is applied here.
// Grid:  blockDim=(64), gridDim=(KVH, B, max_splits)
// One block handles 8 query heads that share the same KV head (GQA=8, D=64).
// =====================
__global__ __launch_bounds__(64, 8) void flashdecoding_phase1_kernel_1warp8q(
    float *__restrict__ partial_out, // [Z, B, H, D]  (Z = max_splits)
    float *__restrict__ partial_m,   // [Z, B, H]
    float *__restrict__ partial_l,   // [Z, B, H]
    const float *__restrict__ q,     // [B, H, D]
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int split_t // requested split length (tokens per split)
)
{
  const int kv_h = blockIdx.x, b = blockIdx.y, split_idx = blockIdx.z;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  // Lane layout: 64 threads = 8 heads * 8 lanes each (D=64 split across the 8 lanes)
  const int lane = threadIdx.x; // 0..63
  const int hloc = lane >> 3;   // which of the 8 Q heads in this block
  const int li = lane & 7;      // lane-in-head
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // Decoding position & sliding-window start
  const int pos = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1) == 0)) ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1; // number of tokens to attend (inclusive of pos)

  // This split covers tokens [s_base, s_base + s_len)
  const int s_base = split_idx * split_t;
  if (s_base >= n_steps)
    return; // inactive split
  const int s_len = min(split_t, n_steps - s_base);

  // K/V pointers. Layout is time-major with stride kv_dim across time.
  const int kv_dim = D * KVH;
  const __hip_bfloat16 *__restrict__ K0 =
      key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
  const __hip_bfloat16 *__restrict__ V0 =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

  // Pre-scale q by 1/sqrt(D) once
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      qseg[s] = qh[li + 8 * s] * inv;
  }

  // Shared mem for staging this split's [s_len x PADDED_HEAD_DIM] (K and V)
  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16 *sK = smem;
  __hip_bfloat16 *sV = sK + (size_t)split_t * PADDED_HEAD_DIM; // reserve max; we'll use s_len subset

  // Streaming softmax state for this head inside the split
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // 16B vector load helper for bf16x8
  struct u128
  {
    uint4 v;
  };
  auto ld8 = [](__hip_bfloat16 const *p)
  { u128 r; r.v=*reinterpret_cast<uint4 const*>(p); return r; };

  // Stage K/V for this split into shared memory (coalesced bf16x8)
  {
    const int vec8 = D / 8; // 8 bf16 = 16B
    const int tot = s_len * vec8;
    for (int e = lane; e < tot; e += 64)
    {
      const int tloc = e / vec8;
      const int i8 = (e - tloc * vec8) * 8;
      const int t_abs = t_start + s_base + tloc;
      const int tw = ((layer_idx & 1) == 0) ? (t_abs % (use_sw ? SW_WINDOW : max(L, 1))) : t_abs;
      *reinterpret_cast<u128 *>(sK + (size_t)tloc * PADDED_HEAD_DIM + i8) = ld8(K0 + (size_t)tw * kv_dim + i8);
      *reinterpret_cast<u128 *>(sV + (size_t)tloc * PADDED_HEAD_DIM + i8) = ld8(V0 + (size_t)tw * kv_dim + i8);
    }
    __syncthreads();
  }

// Compute attention over this split (like FlashAttention streaming inside a block)
#pragma unroll 1
  for (int t = 0; t < s_len; ++t)
  {
    const __hip_bfloat16 *__restrict__ krow = sK + (size_t)t * PADDED_HEAD_DIM;
    const __hip_bfloat16 *__restrict__ vrow = sV + (size_t)t * PADDED_HEAD_DIM;

    // partial dot on 8-lane subgroup
    float part = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float k = __bfloat162float(krow[li + 8 * s]);
      part = fmaf(qseg[s], k, part);
    }
    // subgroup-8 sum (width=8)
    part += __shfl_xor(part, 4, 8);
    part += __shfl_xor(part, 2, 8);
    part += __shfl_xor(part, 1, 8);

    // leader lane does softmax update; broadcast within the 8-lane subgroup
    float alpha = 1.f, w = 0.f;
    if (li == 0)
    {
      float s = part; // already 1/sqrt(D) scaled
      float m_new = fmaxf(m, s);
      alpha = __expf(m - m_new);
      w = __expf(s - m_new);
      l = l * alpha + w;
      m = m_new;
    }
    alpha = __shfl(alpha, (hloc << 3), 64);
    w = __shfl(w, (hloc << 3), 64);

// numerator update for this lane's 8 dims
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float vv = __bfloat162float(vrow[li + 8 * s]);
      out8[s] = fmaf(w, vv, out8[s] * alpha);
    }
  }

  // Write partials for this split (per head)
  float *__restrict__ pout = partial_out + ((size_t)split_idx * B * H + (size_t)b * H + (size_t)h) * D;
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    const int d = li + 8 * s;
    pout[d] = out8[s]; // unnormalized numerator n_s (w.r.t m_s)
  }
  if (li == 0)
  {
    float *pm = partial_m + (size_t)split_idx * B * H + (size_t)b * H + (size_t)h;
    float *pl = partial_l + (size_t)split_idx * B * H + (size_t)b * H + (size_t)h;
    *pm = m;
    *pl = l;
  }
}

// =====================
// Flash-Decoding Phase 2: final reduction across splits
// Consumes partial_out / partial_(m,l) across splits and performs a streaming
// reduction across splits to produce the final output. Applies the sink at the end.
// Grid:  blockDim=(64), gridDim=(KVH, B, 1)
// =====================
__global__ __launch_bounds__(64, 8) void flashdecoding_reduce_splits_kernel_1warp8q(
    float *__restrict__ output,               // [B, H, D]
    const float *__restrict__ partial_out,    // [Z, B, H, D]
    const float *__restrict__ partial_m,      // [Z, B, H]
    const float *__restrict__ partial_l,      // [Z, B, H]
    const __hip_bfloat16 *__restrict__ sinks, // [H] (optional scalar sink scores per head)
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    int max_splits, // Z used in phase 1
    int split_t)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  const int lane = threadIdx.x; // 0..63
  const int hloc = lane >> 3;   // which of 8 Q heads in this block
  const int li = lane & 7;      // lane-in-head
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // How many splits are valid for this (b,h)
  const int pos = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1) == 0)) ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;
  const int n_valid = (n_steps + split_t - 1) / split_t;

  // Streaming reduction across splits: (m,l,out8)
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (int sidx = 0; sidx < n_valid; ++sidx)
  {
    const size_t base_h = (size_t)sidx * B * H + (size_t)b * H + (size_t)h;

    // Load this split's state
    float ms = partial_m[base_h];
    float ls = partial_l[base_h];

    // Combine (m,l) in a numerically stable streaming fashion
    float m_new = fmaxf(m, ms);
    float alpha = __expf(m - m_new); // scales existing accumulator
    float beta = __expf(ms - m_new); // scales incoming split

    // Accumulate numerator vector
    const float *__restrict__ nin = partial_out + base_h * D;
#pragma unroll
    for (int t = 0; t < 8; ++t)
    {
      const int d = li + 8 * t;
      float add = nin[d] * beta;
      out8[t] = out8[t] * alpha + add;
    }
    // Accumulate denominator
    l = l * alpha + ls * beta;
    m = m_new;
  }

  // Sink merge AFTER all splits: affects normalization only (and scales numerator if m increases)
  float alpha_sink = 1.f, l_final = l;
  if (li == 0)
  {
    float ss = sinks ? __bfloat162float(sinks[h]) : -INFINITY; // if no sinks, choose -inf to no-op
    float m_new = fmaxf(m, ss);
    float a = __expf(m - m_new);
    float e = __expf(ss - m_new);
    l_final = l * a + e;
    alpha_sink = a;
    m = m_new;
  }
  alpha_sink = __shfl(alpha_sink, (hloc << 3), 64);
  l_final = __shfl(l_final, (hloc << 3), 64);

  // Write final normalized output
  float *__restrict__ oh = output + (size_t)b * H * D + (size_t)h * D;
  const float inv_den = 1.f / (l_final + SOFTMAX_EPS);
#pragma unroll
  for (int t = 0; t < 8; ++t)
  {
    const int d = li + 8 * t;
    oh[d] = (out8[t] * alpha_sink) * inv_den;
  }
}

// ======================================================================================
// Fused Flash-Decoding (single kernel, multi-warp, no partial global writes)
// - Parallelizes across splits inside each block (multiple warps per (b, kv_h)).
// - Each warp streams over its assigned splits, producing per-head (m,l,out) locally.
// - A fast block-wide reduction merges the warps' partials using log-sum-exp logic.
// - Avoids the Phase-1/Phase-2 round-trip to global memory (big bandwidth + launch cost).
// - No LDS staging of K/V: loads are coalesced across the 8-lane subgroup (li=0..7).
// - Includes softmax "phi" short-circuit to skip tiny weights (fewer expf).
// Assumptions: D==64, GQA==8, warpSize==64 (AMD).
// ======================================================================================

#ifndef FD_MAX_WARPS_PER_CTA
#define FD_MAX_WARPS_PER_CTA 4 // Tune: {2,4,6,8}. 4 is a good default for MI2xx/MI3xx.
#endif

// Short-circuit for small weights: if s <= m - PHI, then w≈0 (skip expf). Keep conservative default.
#ifndef SOFTMAX_PHI
#define SOFTMAX_PHI 8.0f
#endif

#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-9f
#endif

// Merge two streaming-softmax states (m1,l1,n1) and (m2,l2,n2):
// Returns (m_new, l_new) and scales (alpha, beta) such that:
//   n_new = alpha * n1 + beta * n2
//   l_new = alpha * l1 + beta * l2
__device__ inline void merge_softmax_states(float m1, float l1, float m2, float l2,
                                            float &m_out, float &l_out, float &alpha, float &beta)
{
  float m_new = fmaxf(m1, m2);
  float a = __expf(m1 - m_new);
  float b = __expf(m2 - m_new);
  m_out = m_new;
  l_out = l1 * a + l2 * b;
  alpha = a;
  beta = b;
}

// Kernel mapping:
// - blockIdx = (kv_h, b, 0)
// - blockDim.x = 64 * W, where W = number of warps per block (<= FD_MAX_WARPS_PER_CTA)
// - Each warp processes one split at a time: split indices s = warp_id, warp_id+W, ...
// - Within a warp: 64 lanes = 8 heads (hloc) * 8 lanes per head (li), D=64 per head.
// - No shared K/V. Each subgroup-8 (li=0..7) performs coalesced strided loads.
__global__ __launch_bounds__(64 * FD_MAX_WARPS_PER_CTA, 2) void flashdecoding_fused_multiwarp_1warp8q_old(
    float *__restrict__ output,  // [B, H, D]
    const float *__restrict__ q, // [B, H, D], fp32
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, // [H], optional (can be nullptr)
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int split_t // tokens per split (tile over sequence)
)
{
  // Block coords
  const int kv_h = blockIdx.x;
  const int b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  // Thread coords
  const int tid = threadIdx.x;
  const int warp_id = tid >> 6;  // 0..(W-1)
  const int lane = tid & 63;     // 0..63
  const int hloc = lane >> 3;    // 0..7 (which of 8 query heads mapped to this warp)
  const int li = lane & 7;       // 0..7 lane-in-head
  const int W = blockDim.x >> 6; // runtime warps/CTA

  // Head index for this lane
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // Effective sequence region for this layer (sliding window if enabled on even layers)
  const int pos = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1) == 0)) ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;
  if (n_steps <= 0)
    return;

  // Number of splits for this (b, kv_h)
  const int n_splits = (n_steps + split_t - 1) / split_t;

  // Base K/V pointer setup
  const int kv_dim = D * KVH;
  const __hip_bfloat16 *__restrict__ K0 =
      key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
  const __hip_bfloat16 *__restrict__ V0 =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

  // Load & scale q once (1/sqrt(D))
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      qseg[s] = qh[li + 8 * s] * inv;
  }

  // Per-warp local accumulation across its subset of splits
  float m_loc = -INFINITY;
  float l_loc = 0.f;
  float out8_loc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Iterate over splits assigned to this warp
  for (int sidx = warp_id; sidx < n_splits; sidx += W)
  {
    const int s_base = sidx * split_t;
    const int s_len = min(split_t, n_steps - s_base);

    // Streaming softmax within this split
    float m_s = -INFINITY, l_s = 0.f;
    float out8_s[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll 1
    for (int t = 0; t < s_len; ++t)
    {
      const int t_abs = t_start + s_base + t;
      const int tw = ((layer_idx & 1) == 0) ? (t_abs % (use_sw ? SW_WINDOW : max(L, 1))) : t_abs;

      // Dot-product partial for this head across its 8 lanes (subgroup-8 reduction)
      float part = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        const int d = li + 8 * s;
        float k = __bfloat162float(K0[(size_t)tw * kv_dim + d]);
        part = fmaf(qseg[s], k, part);
      }
      // subgroup-8 sum (width=8)
      part += __shfl_xor(part, 4, 8);
      part += __shfl_xor(part, 2, 8);
      part += __shfl_xor(part, 1, 8);

      // Leader lane computes softmax update; add phi short-circuit
      float alpha = 1.f, w = 0.f;
      if (li == 0)
      {
        const float s = part; // score already scaled
        const float dm = s - m_s;
        if (dm <= -SOFTMAX_PHI)
        {
          // Tiny contribution; keep state unchanged and skip expf
          alpha = 1.f;
          w = 0.f;
        }
        else
        {
          const float m_new = fmaxf(m_s, s);
          alpha = __expf(m_s - m_new);
          w = __expf(s - m_new);
          l_s = l_s * alpha + w;
          m_s = m_new;
        }
      }
      alpha = __shfl(alpha, (hloc << 3), 64); // broadcast to subgroup
      w = __shfl(w, (hloc << 3), 64);

// Numerator update (8 dims for this lane)
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        const int d = li + 8 * s;
        float vv = __bfloat162float(V0[(size_t)tw * kv_dim + d]);
        out8_s[s] = fmaf(w, vv, out8_s[s] * alpha);
      }
    } // split tokens

    // Merge this split into warp-local accumulator using log-sum-exp combine
    {
      float m_new, l_new, alpha, beta;
      merge_softmax_states(m_loc, l_loc, m_s, l_s, m_new, l_new, alpha, beta);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        out8_loc[s] = out8_loc[s] * alpha + out8_s[s] * beta;
      }
      m_loc = m_new;
      l_loc = l_new;
    }
  } // splits loop

  // ===== Block-wide reduction across warps (merge per-warp (m,l,out)) =====
  // Use dynamic shared memory as a compact reduction buffer.
  extern __shared__ float redbuf[];
  float *sm_out = redbuf;                    // [W, 64, 8] floats
  float *sm_m = sm_out + (size_t)W * 64 * 8; // [W, 8] floats (per head)
  float *sm_l = sm_m + (size_t)W * 8;        // [W, 8] floats (per head)

// Write this warp's vector to smem
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    sm_out[(size_t)warp_id * 64 * 8 + (size_t)lane * 8 + s] = out8_loc[s];
  }
  // Leader of each subgroup writes (m,l) per head
  if (li == 0)
  {
    sm_m[warp_id * 8 + hloc] = m_loc;
    sm_l[warp_id * 8 + hloc] = l_loc;
  }
  __syncthreads();

  // Let warp 0 do the final merge (W is small; this is cheap and avoids extra syncs)
  if (warp_id == 0)
  {
    // Start from warp 0's state
    float m_fin = sm_m[0 * 8 + hloc];
    float l_fin = sm_l[0 * 8 + hloc];
    float out8_fin[8];
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      out8_fin[s] = sm_out[(size_t)0 * 64 * 8 + (size_t)lane * 8 + s];
    }
    // Merge remaining warps
    for (int w = 1; w < W; ++w)
    {
      float mw = sm_m[w * 8 + hloc];
      float lw = sm_l[w * 8 + hloc];
      float m_new, l_new, alpha, beta;
      merge_softmax_states(m_fin, l_fin, mw, lw, m_new, l_new, alpha, beta);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        const float add = sm_out[(size_t)w * 64 * 8 + (size_t)lane * 8 + s] * beta;
        out8_fin[s] = out8_fin[s] * alpha + add;
      }
      m_fin = m_new;
      l_fin = l_new;
    }

    // Sink merge (post-reduction) — normalization only; scale numerator if m increases
    float alpha_sink = 1.f, l_final = l_fin;
    if (li == 0)
    {
      float ss = sinks ? __bfloat162float(sinks[h]) : -INFINITY;
      float m_new = fmaxf(m_fin, ss);
      float a = __expf(m_fin - m_new);
      float e = __expf(ss - m_new);
      l_final = l_fin * a + e;
      alpha_sink = a;
      m_fin = m_new;
    }
    alpha_sink = __shfl(alpha_sink, (hloc << 3), 64);
    l_final = __shfl(l_final, (hloc << 3), 64);

    // Write final normalized output
    float *__restrict__ oh = output + (size_t)b * H * D + (size_t)h * D;
    const float inv_den = 1.f / (l_final + SOFTMAX_EPS);
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      const int d = li + 8 * s;
      oh[d] = (out8_fin[s] * alpha_sink) * inv_den;
    }
  }
}

// ======================================================================================
// Fused Flash-Decoding (multi-warp, fast-merge, NO K/V staging to LDS)
// - Each warp streams a disjoint time "split" directly from global K/V (coalesced).
// - Keeps per-warp streaming softmax state (m,l,out).
// - A *single* fast block-wide merge in shared memory (warp 0) combines partials
//   with log-sum-exp and writes final normalized outputs.
// - Includes softmax-phi short-circuit to skip tiny expf().
// Assumptions: D==64, GQA==8, warpSize==64 (AMD).
// ======================================================================================

#ifndef SOFTMAX_PHI
#define SOFTMAX_PHI 20.0f
#endif

// bf16x8 (16B) vector helper (used for type-safe aligned accesses if needed later)
struct __align__(16) u128_bf16x8
{
  uint4 v;
};

// ---------------------------------
// Kernel
// ---------------------------------
__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q(
    float *__restrict__ output, const float *__restrict__ q,
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, const float * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  // lane & warp identifiers (AMD ROCm: warpSize==64)
  const int tix = threadIdx.x;
  const int lane = tix & 63;          // 0..63
  const int wid = tix >> 6;           // warp id within block
  const int nwarps = blockDim.x >> 6; // warps per block

  const int hloc = lane >> 3; // 0..7  (which Q head inside this warp)
  const int li = lane & 7;    // 0..7  (lane-in-head)
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // sequence bounds (supports sliding-window KV ring for even layers)
  const int pos = seq_lengths[b];
  const bool do_sw = (use_sw && ((layer_idx & 1) == 0));
  const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;

  const int kv_dim = D * KVH;
  const __hip_bfloat16 *__restrict__ K0 =
      key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
  const __hip_bfloat16 *__restrict__ V0 =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

  // Pre-scale q by 1/sqrt(D) once (each subgroup lane holds 8 scalars)
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      qseg[s] = qh[li + 8 * s] * inv;
  }

  // --------------------------------------
  // Partition time dimension into splits
  // --------------------------------------
  // Each warp processes a contiguous split: [t_begin, t_end)
  const int t_begin = (n_steps * wid) / nwarps;
  const int t_end = (n_steps * (wid + 1)) / nwarps;

  // --------------------------------------
  // Per-warp streaming softmax over its split (no LDS staging)
  // --------------------------------------
  float m = -INFINITY, l = 0.f;             // streaming state per head (owned by li==0)
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // numerator accumulator for this lane's 8 dims

  for (int tloc = t_begin; tloc < t_end; ++tloc)
  {
    const int t_abs = t_start + tloc;
    const int tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;

    // dot across 64 dims split over 8 lanes; memory access is coalesced across lanes
    float part = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float k = __bfloat162float(K0[(size_t)tw * kv_dim + (li + 8 * s)]);
      part = fmaf(qseg[s], k, part);
    }

    // subgroup-8 reduction to sum partials from the 8 lanes of this head
    part += __shfl_xor(part, 4, 8);
    part += __shfl_xor(part, 2, 8);
    part += __shfl_xor(part, 1, 8);

    // leader updates streaming softmax with "phi" short-circuit
    float alpha = 1.f, w = 0.f;
    if (li == 0)
    {
      const float s = part; // already scaled by 1/sqrt(D)
      const float mNew = fmaxf(m, s);
      alpha = __expf(m - mNew);  // <= 1
      const float dm = s - mNew; // <= 0
      // Skip very small contributions: exp(dm) ~ 0 when dm <= -SOFTMAX_PHI
      w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;
      l = l * alpha + w;
      m = mNew;
    }
    // broadcast alpha,w within the 8-lane subgroup (this head)
    alpha = __shfl(alpha, (hloc << 3), 64);
    w = __shfl(w, (hloc << 3), 64);

// numerator update for this lane's 8 dims (directly from global V)
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float v = __bfloat162float(V0[(size_t)tw * kv_dim + (li + 8 * s)]);
      out8[s] = fmaf(w, v, out8[s] * alpha);
    }
  }

  // --------------------------------------
  // Fast block-wide merge: write partials to a tiny red buffer in LDS
  // --------------------------------------
  extern __shared__ float redbuf[];
  float *red_m = redbuf;               // [nwarps][8]
  float *red_l = red_m + nwarps * 8;   // [nwarps][8]
  float *red_out = red_l + nwarps * 8; // [nwarps][64][8] (lane-major × 8 scalars)

  if (li == 0)
  {
    red_m[wid * 8 + hloc] = m;
    red_l[wid * 8 + hloc] = l;
  }
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    red_out[((wid * 64 + lane) * 8 + s)] = out8[s];
  }
  __syncthreads();

  // --------------------------------------
  // Leader warp merges all warps' partial softmax states per head and writes output
  // --------------------------------------
  if ((wid == 0))
  {
    const int my_h = hloc;
    const int my_li = li;

    // 1) Find m_max across warps for this head
    float m_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      float mw = red_m[w * 8 + my_h];
      m_max = fmaxf(m_max, mw);
    }

    // 2) Accumulate scaled l and numerator for this lane's 8 dims
    float l_sum = 0.f;
    float num8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      const float mw = red_m[w * 8 + my_h];
      const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);
      l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);

      const int base_idx = ((w * 64 + lane) * 8);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]);
      }
    }

    // 3) Merge sink (denominator only) and write out
    float alpha_sink = 1.f, l_final = l_sum;
    if (my_li == 0)
    {
      const float ss = __bfloat162float(sinks[kv_h * gqa + my_h]); // per head
      const float m_new = fmaxf(m_max, ss);
      const float alpha = __expf(m_max - m_new);
      const float e = __expf(ss - m_new);
      l_final = l_sum * alpha + e;
      alpha_sink = alpha;
    }
    // Broadcast denominator pieces to the subgroup of this head
    alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);
    l_final = __shfl(l_final, (my_h << 3), 64);

    // 4) Store final, fully-normalized output for this lane's 8 dims
    float *__restrict__ oh = output + (size_t)b * H * D + (size_t)(kv_h * gqa + my_h) * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      const int d = my_li + 8 * s;
      oh[d] = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
    }
  }
}




// ---------------------------------
// Kernel (bf16 output)
// ---------------------------------
__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16out(
    __hip_bfloat16 *__restrict__ output, const float *__restrict__ q,
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, const float * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  // lane & warp identifiers (AMD ROCm: warpSize==64)
  const int tix = threadIdx.x;
  const int lane = tix & 63;          // 0..63
  const int wid = tix >> 6;           // warp id within block
  const int nwarps = blockDim.x >> 6; // warps per block

  const int hloc = lane >> 3; // 0..7  (which Q head inside this warp)
  const int li = lane & 7;    // 0..7  (lane-in-head)
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // sequence bounds (supports sliding-window KV ring for even layers)
  const int pos = seq_lengths[b];
  const bool do_sw = (use_sw && ((layer_idx & 1) == 0));
  const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;

  const int kv_dim = D * KVH;
  const __hip_bfloat16 *__restrict__ K0 =
      key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
  const __hip_bfloat16 *__restrict__ V0 =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

  // Pre-scale q by 1/sqrt(D) once (each subgroup lane holds 8 scalars)
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
      qseg[s] = qh[li + 8 * s] * inv;
  }

  // --------------------------------------
  // Partition time dimension into splits
  // --------------------------------------
  // Each warp processes a contiguous split: [t_begin, t_end)
  const int t_begin = (n_steps * wid) / nwarps;
  const int t_end = (n_steps * (wid + 1)) / nwarps;

  // --------------------------------------
  // Per-warp streaming softmax over its split (no LDS staging)
  // --------------------------------------
  float m = -INFINITY, l = 0.f;             // streaming state per head (owned by li==0)
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // numerator accumulator for this lane's 8 dims

  for (int tloc = t_begin; tloc < t_end; ++tloc)
  {
    const int t_abs = t_start + tloc;
    const int tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;

    // dot across 64 dims split over 8 lanes; memory access is coalesced across lanes
    float part = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float k = __bfloat162float(K0[(size_t)tw * kv_dim + (li + 8 * s)]);
      part = fmaf(qseg[s], k, part);
    }

    // subgroup-8 reduction to sum partials from the 8 lanes of this head
    part += __shfl_xor(part, 4, 8);
    part += __shfl_xor(part, 2, 8);
    part += __shfl_xor(part, 1, 8);

    // leader updates streaming softmax with "phi" short-circuit
    float alpha = 1.f, w = 0.f;
    if (li == 0)
    {
      const float s = part; // already scaled by 1/sqrt(D)
      const float mNew = fmaxf(m, s);
      alpha = __expf(m - mNew);  // <= 1
      const float dm = s - mNew; // <= 0
      // Skip very small contributions: exp(dm) ~ 0 when dm <= -SOFTMAX_PHI
      w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;
      l = l * alpha + w;
      m = mNew;
    }
    // broadcast alpha,w within the 8-lane subgroup (this head)
    alpha = __shfl(alpha, (hloc << 3), 64);
    w = __shfl(w, (hloc << 3), 64);

    // numerator update for this lane's 8 dims (directly from global V)
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      float v = __bfloat162float(V0[(size_t)tw * kv_dim + (li + 8 * s)]);
      out8[s] = fmaf(w, v, out8[s] * alpha);
    }
  }

  // --------------------------------------
  // Fast block-wide merge: write partials to a tiny red buffer in LDS
  // --------------------------------------
  extern __shared__ float redbuf[];
  float *red_m = redbuf;               // [nwarps][8]
  float *red_l = red_m + nwarps * 8;   // [nwarps][8]
  float *red_out = red_l + nwarps * 8; // [nwarps][64][8] (lane-major × 8 scalars)

  if (li == 0)
  {
    red_m[wid * 8 + hloc] = m;
    red_l[wid * 8 + hloc] = l;
  }
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    red_out[((wid * 64 + lane) * 8 + s)] = out8[s];
  }
  __syncthreads();

  // --------------------------------------
  // Leader warp merges all warps' partial softmax states per head and writes output (bf16)
  // --------------------------------------
  if ((wid == 0))
  {
    const int my_h = hloc;
    const int my_li = li;

    // 1) Find m_max across warps for this head
    float m_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      float mw = red_m[w * 8 + my_h];
      m_max = fmaxf(m_max, mw);
    }

    // 2) Accumulate scaled l and numerator for this lane's 8 dims
    float l_sum = 0.f;
    float num8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      const float mw = red_m[w * 8 + my_h];
      const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);
      l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);

      const int base_idx = ((w * 64 + lane) * 8);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]);
      }
    }

    // 3) Merge sink (denominator only) and write out
    float alpha_sink = 1.f, l_final = l_sum;
    if (my_li == 0)
    {
      const float ss = __bfloat162float(sinks[kv_h * gqa + my_h]); // per head
      const float m_new = fmaxf(m_max, ss);
      const float alpha = __expf(m_max - m_new);
      const float e = __expf(ss - m_new);
      l_final = l_sum * alpha + e;
      alpha_sink = alpha;
    }
    // Broadcast denominator pieces to the subgroup of this head
    alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);
    l_final = __shfl(l_final, (my_h << 3), 64);

    // 4) Store final, fully-normalized output for this lane's 8 dims (bf16)
    __hip_bfloat16 *__restrict__ oh =
        output + (size_t)b * H * D + (size_t)(kv_h * gqa + my_h) * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      const int d = my_li + 8 * s;
      const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
      oh[d] = __float2bfloat16(val);
    }
  }
}


// ---------------------------------
// Kernel body (bf16 q input, bf16 output)
// ---------------------------------
template <bool kUseBf16Kv>
__device__ inline void flashdecoding_fused_fastmerge_nostage_1warp8q_body(
    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,
    const int8_t *__restrict__ key_cache,
    const int8_t *__restrict__ value_cache,
    const __half *__restrict__ key_scales,
    const __half *__restrict__ value_scales,
    const __hip_bfloat16 *__restrict__ key_cache_recent_bf16,
    const __hip_bfloat16 *__restrict__ value_cache_recent_bf16,
    const int *__restrict__ kv_cache_recent_pos,
    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset,
    int /*tile_t_unused*/)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b >= B || kv_h >= KVH || D != 64)
    return;

  // lane & warp identifiers (AMD ROCm: warpSize==64)
  const int tix = threadIdx.x;
  const int lane = tix & 63;          // 0..63
  const int wid = tix >> 6;           // warp id within block
  const int nwarps = blockDim.x >> 6; // warps per block

  const int hloc = lane >> 3; // 0..7  (which Q head inside this warp)
  const int li   = lane & 7;  // 0..7  (lane-in-head)
  const int gqa  = 8;
  const int h    = kv_h * gqa + hloc;
  if (h >= H)
    return;

  // sequence bounds (supports sliding-window KV ring for even layers)
  const int pos = seq_lengths[b];
  const bool do_sw = (use_sw && ((layer_idx & 1) == 0));
  const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;
  const int n_steps = pos - t_start + 1;
  if (n_steps <= 0)
    return;

  const int kv_dim = D * KVH;
  const size_t base_kv = (size_t)b * batch_kv_stride + layer_kv_offset;
  const int8_t *__restrict__ K0 =
      key_cache + base_kv + (size_t)kv_h * D;
  const int8_t *__restrict__ V0 =
      value_cache + base_kv + (size_t)kv_h * D;
  const __half *__restrict__ KS =
      key_scales + (size_t)b * batch_scale_stride + layer_scale_offset;
  const __half *__restrict__ VS =
      value_scales + (size_t)b * batch_scale_stride + layer_scale_offset;

#if KV_BF16_KEEP_TOKENS > 0
  const bool bf16_case = ((layer_idx & 1) == 1) && (pos < KV_BF16_KEEP_TOKENS);
  if (kUseBf16Kv)
  {
    if (!bf16_case || !key_cache_recent_bf16 || !value_cache_recent_bf16 || !kv_cache_recent_pos)
      return;
  }
  else
  {
    if (bf16_case && key_cache_recent_bf16 && value_cache_recent_bf16 && kv_cache_recent_pos)
      return;
  }

  const size_t recent_slots = KV_BF16_KEEP_TOKENS;
  const size_t recent_pos_base = ((size_t)b * (size_t)L + (size_t)layer_idx) * recent_slots;
  const size_t recent_elem_base = recent_pos_base * (size_t)kv_dim;
#else
  if (kUseBf16Kv)
    return;
  (void)key_cache_recent_bf16;
  (void)value_cache_recent_bf16;
  (void)kv_cache_recent_pos;
#endif

  if (kUseBf16Kv)
  {
    (void)K0;
    (void)V0;
    (void)KS;
    (void)VS;
  }

  // Pre-scale q by 1/sqrt(D) once (each subgroup lane holds 8 scalars)
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const __hip_bfloat16 *__restrict__ qh =
        q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      // q is bf16 now: convert to float, then apply scaling
      qseg[s] = __bfloat162float(qh[li + 8 * s]) * inv;
    }
  }

  // --------------------------------------
  // Partition time dimension into splits
  // --------------------------------------
  // Each warp processes a contiguous split: [t_begin, t_end)
  const int t_begin = (n_steps * wid) / nwarps;
  const int t_end   = (n_steps * (wid + 1)) / nwarps;

  // --------------------------------------
  // Per-warp streaming softmax over its split (no LDS staging)
  // --------------------------------------
  float m = -INFINITY, l = 0.f;             // streaming state per head (owned by li==0)
  float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // numerator accumulator for this lane's 8 dims

  for (int tloc = t_begin; tloc < t_end; ++tloc)
  {
    const int t_abs = t_start + tloc;

    float part = 0.f;
    int tw = 0;
    float scale_v = 0.f;
#if KV_BF16_KEEP_TOKENS > 0
    const __hip_bfloat16 *V_recent_head = nullptr;
    if (kUseBf16Kv)
    {
      const size_t slot = (size_t)t_abs;
      if (slot >= recent_slots)
        return;
      if (kv_cache_recent_pos[recent_pos_base + slot] != t_abs)
        return;
      const size_t elem_offset = recent_elem_base + slot * (size_t)kv_dim + (size_t)kv_h * D;
      const __hip_bfloat16 *K_recent_head = key_cache_recent_bf16 + elem_offset;
      V_recent_head = value_cache_recent_bf16 + elem_offset;
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        const float k = __bfloat162float(K_recent_head[li + 8 * s]);
        part = fmaf(qseg[s], k, part);
      }
    }
    else
#endif
    {
      tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;
      const float scale_k = __half2float(KS[tw]);
      scale_v = __half2float(VS[tw]);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        const int8_t k_q = K0[(size_t)tw * kv_dim + (li + 8 * s)];
        const float k = static_cast<float>(k_q) * scale_k;
        part = fmaf(qseg[s], k, part);
      }
    }

#if KV_BF16_KEEP_TOKENS > 0
    if (kUseBf16Kv)
    {
      (void)tw;
      (void)scale_v;
    }
#endif

    // subgroup-8 reduction to sum partials from the 8 lanes of this head
    part += __shfl_xor(part, 4, 8);
    part += __shfl_xor(part, 2, 8);
    part += __shfl_xor(part, 1, 8);

    // leader updates streaming softmax with "phi" short-circuit
    float alpha = 1.f, w = 0.f;
    if (li == 0)
    {
      const float s = part; // already scaled by 1/sqrt(D)
      const float mNew = fmaxf(m, s);
      alpha = __expf(m - mNew);  // <= 1
      const float dm = s - mNew; // <= 0
      // Skip very small contributions: exp(dm) ~ 0 when dm <= -SOFTMAX_PHI
      w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;
      l = l * alpha + w;
      m = mNew;
    }
    // broadcast alpha,w within the 8-lane subgroup (this head)
    alpha = __shfl(alpha, (hloc << 3), 64);
    w     = __shfl(w,     (hloc << 3), 64);

    // numerator update for this lane's 8 dims (directly from global/cache V)
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
#if KV_BF16_KEEP_TOKENS > 0
      if (kUseBf16Kv)
      {
        const float v = __bfloat162float(V_recent_head[li + 8 * s]);
        out8[s] = fmaf(w, v, out8[s] * alpha);
      }
      else
#endif
      {
        const int8_t v_q = V0[(size_t)tw * kv_dim + (li + 8 * s)];
        const float v = static_cast<float>(v_q) * scale_v;
        out8[s] = fmaf(w, v, out8[s] * alpha);
      }
    }
  }

  // --------------------------------------
  // Fast block-wide merge: write partials to a tiny red buffer in LDS
  // --------------------------------------
  extern __shared__ float redbuf[];
  float *red_m   = redbuf;               // [nwarps][8]
  float *red_l   = red_m   + nwarps * 8; // [nwarps][8]
  float *red_out = red_l   + nwarps * 8; // [nwarps][64][8] (lane-major × 8 scalars)

  if (li == 0)
  {
    red_m[wid * 8 + hloc] = m;
    red_l[wid * 8 + hloc] = l;
  }
#pragma unroll
  for (int s = 0; s < 8; ++s)
  {
    red_out[((wid * 64 + lane) * 8 + s)] = out8[s];
  }
  __syncthreads();

  // --------------------------------------
  // Leader warp merges all warps' partial softmax states per head and writes output (bf16)
  // --------------------------------------
  if ((wid == 0))
  {
    const int my_h  = hloc;
    const int my_li = li;

    // 1) Find m_max across warps for this head
    float m_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      float mw = red_m[w * 8 + my_h];
      m_max = fmaxf(m_max, mw);
    }

    // 2) Accumulate scaled l and numerator for this lane's 8 dims
    float l_sum = 0.f;
    float num8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
    for (int w = 0; w < nwarps; ++w)
    {
      const float mw = red_m[w * 8 + my_h];
      const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);
      l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);

      const int base_idx = ((w * 64 + lane) * 8);
#pragma unroll
      for (int s = 0; s < 8; ++s)
      {
        num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]);
      }
    }

    // 3) Merge sink (denominator only) and write out
    float alpha_sink = 1.f, l_final = l_sum;
    if (my_li == 0)
    {
      const float ss = __bfloat162float(sinks[kv_h * gqa + my_h]); // per head
      const float m_new = fmaxf(m_max, ss);
      const float alpha = __expf(m_max - m_new);
      const float e     = __expf(ss - m_new);
      l_final   = l_sum * alpha + e;
      alpha_sink = alpha;
    }
    // Broadcast denominator pieces to the subgroup of this head
    alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);
    l_final    = __shfl(l_final,    (my_h << 3), 64);

    // 4) Store final, fully-normalized output for this lane's 8 dims (bf16)
    __hip_bfloat16 *__restrict__ oh =
        output + (size_t)b * H * D + (size_t)(kv_h * gqa + my_h) * D;
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
      const int d   = my_li + 8 * s;
      const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
      oh[d] = __float2bfloat16(val);
    }
  }
}

__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16q_bf16out(
    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,
    const int8_t *__restrict__ key_cache,
    const int8_t *__restrict__ value_cache,
    const __half *__restrict__ key_scales,
    const __half *__restrict__ value_scales,
    const __hip_bfloat16 *__restrict__ key_cache_recent_bf16,
    const __hip_bfloat16 *__restrict__ value_cache_recent_bf16,
    const int *__restrict__ kv_cache_recent_pos,
    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 *mask,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int seq_len,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset,
    int tile_t_unused)
{
  flashdecoding_fused_fastmerge_nostage_1warp8q_body<false>(
      output, q, key_cache, value_cache, key_scales, value_scales,
      key_cache_recent_bf16, value_cache_recent_bf16, kv_cache_recent_pos,
      sinks, mask, seq_lengths,
      B, H, KVH, D, seq_len,
      L, layer_idx, use_sw,
      batch_kv_stride, layer_kv_offset,
      batch_scale_stride, layer_scale_offset,
      tile_t_unused);
}

#if KV_BF16_KEEP_TOKENS > 0
__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16q_bf16out_bf16kv(
    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,
    const int8_t *__restrict__ key_cache,
    const int8_t *__restrict__ value_cache,
    const __half *__restrict__ key_scales,
    const __half *__restrict__ value_scales,
    const __hip_bfloat16 *__restrict__ key_cache_recent_bf16,
    const __hip_bfloat16 *__restrict__ value_cache_recent_bf16,
    const int *__restrict__ kv_cache_recent_pos,
    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 *mask,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int seq_len,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset,
    int tile_t_unused)
{
  flashdecoding_fused_fastmerge_nostage_1warp8q_body<true>(
      output, q, key_cache, value_cache, key_scales, value_scales,
      key_cache_recent_bf16, value_cache_recent_bf16, kv_cache_recent_pos,
      sinks, mask, seq_lengths,
      B, H, KVH, D, seq_len,
      L, layer_idx, use_sw,
      batch_kv_stride, layer_kv_offset,
      batch_scale_stride, layer_scale_offset,
      tile_t_unused);
}
#endif



// ---------------------------------
// Kernel (bf16 q input, bf16 output)
// ---------------------------------
__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16q_bf16out_full_bf16kv(
    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)
{
    const int kv_h = blockIdx.x, b = blockIdx.y;
    if (b >= B || kv_h >= KVH || D != 64)
        return;

    // lane & warp identifiers (AMD ROCm: warpSize==64)
    const int tix = threadIdx.x;
    const int lane = tix & 63;          // 0..63
    const int wid = tix >> 6;           // warp id within block
    const int nwarps = blockDim.x >> 6; // warps per block

    const int hloc = lane >> 3; // 0..7  (which Q head inside this warp)
    const int li = lane & 7;    // 0..7  (lane-in-head)
    const int gqa = 8;
    const int h = kv_h * gqa + hloc;
    if (h >= H)
        return;

    // sequence bounds (supports sliding-window KV ring for even layers)
    const int pos = seq_lengths[b];
    const bool do_sw = (use_sw && ((layer_idx & 1) == 0));
    const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;
    const int n_steps = pos - t_start + 1;

    const int kv_dim = D * KVH;
    const __hip_bfloat16 *__restrict__ K0 =
        key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
    const __hip_bfloat16 *__restrict__ V0 =
        value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;

    // Pre-scale q by 1/sqrt(D) once (each subgroup lane holds 8 scalars)
    float qseg[8];
    {
        const float inv = rsqrtf(64.f);
        const __hip_bfloat16 *__restrict__ qh =
            q + (size_t)b * H * D + (size_t)h * D;
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            // q is bf16 now: convert to float, then apply scaling
            qseg[s] = __bfloat162float(qh[li + 8 * s]) * inv;
        }
    }

    // --------------------------------------
    // Partition time dimension into splits
    // --------------------------------------
    // Each warp processes a contiguous split: [t_begin, t_end)
    const int t_begin = (n_steps * wid) / nwarps;
    const int t_end = (n_steps * (wid + 1)) / nwarps;

    // --------------------------------------
    // Per-warp streaming softmax over its split (no LDS staging)
    // --------------------------------------
    float m = -INFINITY, l = 0.f;             // streaming state per head (owned by li==0)
    float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // numerator accumulator for this lane's 8 dims

    for (int tloc = t_begin; tloc < t_end; ++tloc)
    {
        const int t_abs = t_start + tloc;
        const int tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;

        // dot across 64 dims split over 8 lanes; memory access is coalesced across lanes
        float part = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            float k = __bfloat162float(K0[(size_t)tw * kv_dim + (li + 8 * s)]);
            part = fmaf(qseg[s], k, part);
        }

        // subgroup-8 reduction to sum partials from the 8 lanes of this head
        part += __shfl_xor(part, 4, 8);
        part += __shfl_xor(part, 2, 8);
        part += __shfl_xor(part, 1, 8);

        // leader updates streaming softmax with "phi" short-circuit
        float alpha = 1.f, w = 0.f;
        if (li == 0)
        {
            const float s = part; // already scaled by 1/sqrt(D)
            const float mNew = fmaxf(m, s);
            alpha = __expf(m - mNew);  // <= 1
            const float dm = s - mNew; // <= 0
            // Skip very small contributions: exp(dm) ~ 0 when dm <= -SOFTMAX_PHI
            w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;
            l = l * alpha + w;
            m = mNew;
        }
        // broadcast alpha,w within the 8-lane subgroup (this head)
        alpha = __shfl(alpha, (hloc << 3), 64);
        w = __shfl(w, (hloc << 3), 64);

        // numerator update for this lane's 8 dims (directly from global V)
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            float v = __bfloat162float(V0[(size_t)tw * kv_dim + (li + 8 * s)]);
            out8[s] = fmaf(w, v, out8[s] * alpha);
        }
    }

    // --------------------------------------
    // Fast block-wide merge: write partials to a tiny red buffer in LDS
    // --------------------------------------
    extern __shared__ float redbuf[];
    float *red_m = redbuf;               // [nwarps][8]
    float *red_l = red_m + nwarps * 8;   // [nwarps][8]
    float *red_out = red_l + nwarps * 8; // [nwarps][64][8] (lane-major × 8 scalars)

    if (li == 0)
    {
        red_m[wid * 8 + hloc] = m;
        red_l[wid * 8 + hloc] = l;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s)
    {
        red_out[((wid * 64 + lane) * 8 + s)] = out8[s];
    }
    __syncthreads();

    // --------------------------------------
    // Leader warp merges all warps' partial softmax states per head and writes output (bf16)
    // --------------------------------------
    if ((wid == 0))
    {
        const int my_h = hloc;
        const int my_li = li;

        // 1) Find m_max across warps for this head
        float m_max = -INFINITY;
#pragma unroll
        for (int w = 0; w < nwarps; ++w)
        {
            float mw = red_m[w * 8 + my_h];
            m_max = fmaxf(m_max, mw);
        }

        // 2) Accumulate scaled l and numerator for this lane's 8 dims
        float l_sum = 0.f;
        float num8[8] = {0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int w = 0; w < nwarps; ++w)
        {
            const float mw = red_m[w * 8 + my_h];
            const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);
            l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);

            const int base_idx = ((w * 64 + lane) * 8);
#pragma unroll
            for (int s = 0; s < 8; ++s)
            {
                num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]);
            }
        }

        // 3) Merge sink (denominator only) and write out
        float alpha_sink = 1.f, l_final = l_sum;
        if (my_li == 0)
        {
            const float ss = __bfloat162float(sinks[kv_h * gqa + my_h]); // per head
            const float m_new = fmaxf(m_max, ss);
            const float alpha = __expf(m_max - m_new);
            const float e = __expf(ss - m_new);
            l_final = l_sum * alpha + e;
            alpha_sink = alpha;
        }
        // Broadcast denominator pieces to the subgroup of this head
        alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);
        l_final = __shfl(l_final, (my_h << 3), 64);

        // 4) Store final, fully-normalized output for this lane's 8 dims (bf16)
        __hip_bfloat16 *__restrict__ oh =
            output + (size_t)b * H * D + (size_t)(kv_h * gqa + my_h) * D;
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            const int d = my_li + 8 * s;
            const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
            oh[d] = __float2bfloat16(val);
        }
    }
}









// ----------------------------------------------


// "Optimized" kernel (start with same baseline).


// Copy of flash_decoding_kernel under new name.


// You can safely modify/tune this one.


// ----------------------------------------------


__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16q_bf16out_full_bf16kv_okay(


    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,


    const __hip_bfloat16 *__restrict__ key_cache,


    const __hip_bfloat16 *__restrict__ value_cache,


    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 * /*mask*/,


    const int *__restrict__ seq_lengths,


    int B, int H, int KVH, int D, int /*seq_len*/,


    int L, int layer_idx, bool use_sw,


    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)


{


    const int kv_h = blockIdx.x, b = blockIdx.y;


    if (b >= B || kv_h >= KVH || D != 64)


        return;





    const int tix = threadIdx.x;


    const int lane = tix & 63;


    const int wid = tix >> 6;


    const int nwarps = blockDim.x >> 6;





    const int hloc = lane >> 3; // 0..7


    const int li = lane & 7;    // 0..7, a thread now processes a contiguous chunk


    const int gqa = 8;


    const int h = kv_h * gqa + hloc;


    if (h >= H)


        return;





    // CHANGED: Each thread is responsible for a contiguous chunk of 8 dimensions (16 bytes)


    const int d_start = li * 8;





    const int pos = seq_lengths[b];


    const bool do_sw = (use_sw && ((layer_idx & 1) == 0));


    const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;


    const int n_steps = pos - t_start + 1;





    const int kv_dim = D * KVH;


    const __hip_bfloat16 *__restrict__ K0 =


        key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;


    const __hip_bfloat16 *__restrict__ V0 =


        value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;





    float qseg[8];


    {


        const float inv = rsqrtf(64.f);


        const __hip_bfloat16 *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;





        // CHANGED: Perform a single 128-bit vectorized load for Q


        const uint4 q_chunk = *reinterpret_cast<const uint4 *>(qh + d_start);


        const __hip_bfloat16* q_bf16 = reinterpret_cast<const __hip_bfloat16*>(&q_chunk);


        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            qseg[s] = __bfloat162float(q_bf16[s]) * inv;


        }


    }





    const int t_begin = (n_steps * wid) / nwarps;


    const int t_end   = (n_steps * (wid + 1)) / nwarps;





    float m = -INFINITY, l = 0.f;


    float out8[8] = {0,0,0,0,0,0,0,0};





    for (int tloc = t_begin; tloc < t_end; ++tloc) {


        const int t_abs = t_start + tloc;


        const int tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;





        // --- Q.K^T calculation ---


        // CHANGED: Perform a single 128-bit vectorized load for K


        const size_t k_offset = (size_t)tw * kv_dim + d_start;


        const uint4 k_chunk = *reinterpret_cast<const uint4 *>(K0 + k_offset);


        const __hip_bfloat16* k_bf16 = reinterpret_cast<const __hip_bfloat16*>(&k_chunk);


        


        float part = 0.f;


        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            // Unpack and compute dot product


            float k = __bfloat162float(k_bf16[s]);


            part = fmaf(qseg[s], k, part);


        }





        // Intra-warp reduction logic is IDENTICAL, because we still need to sum


        // the `part` values from the 8 threads (li=0..7) in the same hloc group.


        part += __shfl_xor(part, 4, 8);


        part += __shfl_xor(part, 2, 8);


        part += __shfl_xor(part, 1, 8);





        float alpha = 1.f, w = 0.f;


        if (li == 0) {


            const float s = part;


            const float mNew = fmaxf(m, s);


            alpha = __expf(m - mNew);


            const float dm = s - mNew;


            w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;


            l = l * alpha + w;


            m = mNew;


        }


        alpha = __shfl(alpha, (hloc << 3), 64);


        w     = __shfl(w,     (hloc << 3), 64);





        // --- S.V calculation ---


        // CHANGED: Perform a single 128-bit vectorized load for V


        const size_t v_offset = (size_t)tw * kv_dim + d_start;


        const uint4 v_chunk = *reinterpret_cast<const uint4 *>(V0 + v_offset);


        const __hip_bfloat16* v_bf16 = reinterpret_cast<const __hip_bfloat16*>(&v_chunk);





        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            float v = __bfloat162float(v_bf16[s]);


            out8[s] = fmaf(w, v, out8[s] * alpha);


        }


    }





    // --- Inter-warp reduction (using the original, faster single-warp method) ---


    extern __shared__ float redbuf[];


    float *red_m   = redbuf;


    float *red_l   = red_m + nwarps * 8;


    float *red_out = red_l + nwarps * 8;





    if (li == 0) {


        red_m[wid * 8 + hloc] = m;


        red_l[wid * 8 + hloc] = l;


    }


    #pragma unroll


    for (int s = 0; s < 8; ++s) {


        red_out[((wid * 64 + ((hloc<<3)|li)) * 8 + s)] = out8[s];


    }


    __syncthreads();





    if ((wid == 0)) {


        const int my_h  = hloc;


        const int my_li = li;





        float m_max = -INFINITY;


        #pragma unroll


        for (int w = 0; w < nwarps; ++w) { m_max = fmaxf(m_max, red_m[w * 8 + my_h]); }





        float l_sum = 0.f;


        float num8[8] = {0,0,0,0,0,0,0,0};





        #pragma unroll


        for (int w = 0; w < nwarps; ++w) {


            const float mw = red_m[w * 8 + my_h];


            const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);


            l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);





            const int base_idx = ((w * 64 + ((my_h<<3)|my_li)) * 8);


            #pragma unroll


            for (int s = 0; s < 8; ++s) { num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]); }


        }





        float alpha_sink = 1.f, l_final = l_sum;


        if (my_li == 0) {


            const float ss = __bfloat162float(sinks[kv_h * 8 + my_h]);


            const float m_new = fmaxf(m_max, ss);


            const float alpha2 = __expf(m_max - m_new);


            const float e = __expf(ss - m_new);


            l_final = l_sum * alpha2 + e;


            alpha_sink = alpha2;


        }


        alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);


        l_final    = __shfl(l_final,    (my_h << 3), 64);





        __hip_bfloat16 *__restrict__ oh =


            output + (size_t)b * H * D + (size_t)(kv_h * 8 + my_h) * D;


        


        // CHANGED: Pack the final output and perform a single 128-bit vectorized store


        __hip_bfloat16 out_bf16[8];


        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);


            out_bf16[s] = __float2bfloat16(val);


        }


        *reinterpret_cast<uint4*>(oh + d_start) = *reinterpret_cast<uint4*>(out_bf16);


    }


}








__global__ __launch_bounds__(256, 4) void flashdecoding_fused_fastmerge_nostage_1warp8q_bf16q_bf16out_full_bf16kv_db(


    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,


    const __hip_bfloat16 *__restrict__ key_cache,


    const __hip_bfloat16 *__restrict__ value_cache,


    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 * /*mask*/,


    const int *__restrict__ seq_lengths,


    int B, int H, int KVH, int D, int /*seq_len*/,


    int L, int layer_idx, bool use_sw,


    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)


{


    // --- Block/Thread setup (Không thay đổi) ---


    const int kv_h = blockIdx.x, b = blockIdx.y;


    if (b >= B || kv_h >= KVH || D != 64)


        return;





    const int tix = threadIdx.x;


    const int lane = tix & 63;


    const int wid = tix >> 6;


    const int nwarps = blockDim.x >> 6;





    const int hloc = lane >> 3;


    const int li = lane & 7;


    const int gqa = 8;


    const int h = kv_h * gqa + hloc;


    if (h >= H)


        return;





    const int d_start = li * 8;





    const int pos = seq_lengths[b];


    const bool do_sw = (use_sw && ((layer_idx & 1) == 0));


    const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;


    const int n_steps = pos - t_start + 1;





    const int kv_dim = D * KVH;


    const __hip_bfloat16 *__restrict__ K0 =


        key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;


    const __hip_bfloat16 *__restrict__ V0 =


        value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;





    float qseg[8];


    {


        const float inv = rsqrtf(64.f);


        const __hip_bfloat16 *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;


        const uint4 q_chunk = *reinterpret_cast<const uint4 *>(qh + d_start);


        const __hip_bfloat16* q_bf16 = reinterpret_cast<const __hip_bfloat16*>(&q_chunk);


        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            qseg[s] = __bfloat162float(q_bf16[s]) * inv;


        }


    }





    const int t_begin = (n_steps * wid) / nwarps;


    const int t_end   = (n_steps * (wid + 1)) / nwarps;





    float m = -INFINITY, l = 0.f;


    float out8[8] = {0,0,0,0,0,0,0,0};





    // ==================== BEGIN: SOFTWARE PIPELINE IMPLEMENTATION ====================


    if (t_begin < t_end) {


        uint4 k_chunk_pipe, v_chunk_pipe;





        // --- 1. PROLOGUE: Nạp trước dữ liệu cho vòng lặp đầu tiên (t_begin) ---


        {


            const int t_abs0 = t_start + t_begin;


            const int tw0 = do_sw ? (t_abs0 % SW_WINDOW) : t_abs0;


            const size_t offset0 = (size_t)tw0 * kv_dim + d_start;


            k_chunk_pipe = *reinterpret_cast<const uint4 *>(K0 + offset0);


            v_chunk_pipe = *reinterpret_cast<const uint4 *>(V0 + offset0);


        }





        // --- 2. MAIN PIPELINED LOOP: Chạy từ đầu đến gần cuối ---


        for (int tloc = t_begin; tloc < t_end - 1; ++tloc) {


            // Dữ liệu hiện tại (current) cho tloc đã có trong ..._pipe


            const uint4 k_chunk_current = k_chunk_pipe;


            const uint4 v_chunk_current = v_chunk_pipe;





            // Tải trước (prefetch) dữ liệu cho vòng lặp tiếp theo (tloc + 1)


            {


                const int t_abs_next = t_start + tloc + 1;


                const int tw_next = do_sw ? (t_abs_next % SW_WINDOW) : t_abs_next;


                const size_t offset_next = (size_t)tw_next * kv_dim + d_start;


                k_chunk_pipe = *reinterpret_cast<const uint4 *>(K0 + offset_next);


                v_chunk_pipe = *reinterpret_cast<const uint4 *>(V0 + offset_next);


            }


            


            // --- COMPUTE: Thực hiện tính toán trên dữ liệu của vòng lặp hiện tại ---


            const __hip_bfloat16* k_bf16 = reinterpret_cast<const __hip_bfloat16*>(&k_chunk_current);


            float part = 0.f;


            #pragma unroll


            for (int s = 0; s < 8; ++s) {


                part = fmaf(qseg[s], __bfloat162float(k_bf16[s]), part);


            }





            part += __shfl_xor(part, 4, 8);


            part += __shfl_xor(part, 2, 8);


            part += __shfl_xor(part, 1, 8);





            float alpha = 1.f, w = 0.f;


            if (li == 0) {


                const float s = part;


                const float mNew = fmaxf(m, s);


                alpha = __expf(m - mNew);


                const float dm = s - mNew;


                w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;


                l = l * alpha + w;


                m = mNew;


            }


            alpha = __shfl(alpha, (hloc << 3), 64);


            w     = __shfl(w,     (hloc << 3), 64);


            


            const __hip_bfloat16* v_bf16 = reinterpret_cast<const __hip_bfloat16*>(&v_chunk_current);


            #pragma unroll


            for (int s = 0; s < 8; ++s) {


                out8[s] = fmaf(w, __bfloat162float(v_bf16[s]), out8[s] * alpha);


            }


            // --- END COMPUTE ---


        }





        // --- 3. EPILOGUE: Xử lý vòng lặp cuối cùng (t_end - 1) ---


        // Dữ liệu cho vòng lặp cuối đã nằm sẵn trong ..._pipe


        {


            // --- COMPUTE: Mã tính toán được lặp lại y hệt ---


            const __hip_bfloat16* k_bf16 = reinterpret_cast<const __hip_bfloat16*>(&k_chunk_pipe);


            float part = 0.f;


            #pragma unroll


            for (int s = 0; s < 8; ++s) {


                part = fmaf(qseg[s], __bfloat162float(k_bf16[s]), part);


            }





            part += __shfl_xor(part, 4, 8);


            part += __shfl_xor(part, 2, 8);


            part += __shfl_xor(part, 1, 8);





            float alpha = 1.f, w = 0.f;


            if (li == 0) {


                const float s = part;


                const float mNew = fmaxf(m, s);


                alpha = __expf(m - mNew);


                const float dm = s - mNew;


                w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;


                l = l * alpha + w;


                m = mNew;


            }


            alpha = __shfl(alpha, (hloc << 3), 64);


            w     = __shfl(w,     (hloc << 3), 64);





            const __hip_bfloat16* v_bf16 = reinterpret_cast<const __hip_bfloat16*>(&v_chunk_pipe);


            #pragma unroll


            for (int s = 0; s < 8; ++s) {


                out8[s] = fmaf(w, __bfloat162float(v_bf16[s]), out8[s] * alpha);


            }


            // --- END COMPUTE ---


        }


    }


    // ==================== END: SOFTWARE PIPELINE IMPLEMENTATION ====================





    // --- Final reduction (Không thay đổi, vẫn dùng single-warp) ---


    extern __shared__ float redbuf[];


    //... (phần còn lại của kernel giữ nguyên)


    float *red_m   = redbuf;


    float *red_l   = red_m + nwarps * 8;


    float *red_out = red_l + nwarps * 8;





    if (li == 0) {


        red_m[wid * 8 + hloc] = m;


        red_l[wid * 8 + hloc] = l;


    }


    #pragma unroll


    for (int s = 0; s < 8; ++s) {


        red_out[((wid * 64 + ((hloc<<3)|li)) * 8 + s)] = out8[s];


    }


    __syncthreads();





    if ((wid == 0)) {


        const int my_h  = hloc;


        const int my_li = li;





        float m_max = -INFINITY;


        #pragma unroll


        for (int w = 0; w < nwarps; ++w) { m_max = fmaxf(m_max, red_m[w * 8 + my_h]); }





        float l_sum = 0.f;


        float num8[8] = {0,0,0,0,0,0,0,0};





        #pragma unroll


        for (int w = 0; w < nwarps; ++w) {


            const float mw = red_m[w * 8 + my_h];


            const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);


            l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);





            const int base_idx = ((w * 64 + ((my_h<<3)|my_li)) * 8);


            #pragma unroll


            for (int s = 0; s < 8; ++s) { num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]); }


        }





        float alpha_sink = 1.f, l_final = l_sum;


        if (my_li == 0) {


            const float ss = __bfloat162float(sinks[kv_h * 8 + my_h]);


            const float m_new = fmaxf(m_max, ss);


            const float alpha2 = __expf(m_max - m_new);


            const float e = __expf(ss - m_new);


            l_final = l_sum * alpha2 + e;


            alpha_sink = alpha2;


        }


        alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);


        l_final    = __shfl(l_final,    (my_h << 3), 64);





        __hip_bfloat16 *__restrict__ oh =


            output + (size_t)b * H * D + (size_t)(kv_h * 8 + my_h) * D;


        


        __hip_bfloat16 out_bf16[8];


        #pragma unroll


        for (int s = 0; s < 8; ++s) {


            const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);


            out_bf16[s] = __float2bfloat16(val);


        }


        *reinterpret_cast<uint4*>(oh + d_start) = *reinterpret_cast<uint4*>(out_bf16);


    }


}


__global__ void flash_decoding_kernel_opt_old(
    __hip_bfloat16 *__restrict__ output, const __hip_bfloat16 *__restrict__ q,
    const __hip_bfloat16 *__restrict__ key_cache,
    const __hip_bfloat16 *__restrict__ value_cache,
    const __hip_bfloat16 *__restrict__ sinks, const __hip_bfloat16 * /*mask*/,
    const int *__restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int /*tile_t_unused*/)
{
    // --- Block/Thread/Warp setup (Không thay đổi) ---
    const int kv_h = blockIdx.x, b = blockIdx.y;
    if (b >= B || kv_h >= KVH || D != 64)
        return;
    const int tix = threadIdx.x;
    const int lane = tix & 63;
    const int wid = tix >> 6;
    const int nwarps = blockDim.x >> 6;
    const int hloc = lane >> 3;
    const int li = lane & 7;
    const int gqa = H / KVH;
    const int h = kv_h * gqa + hloc;
    if (h >= H)
        return;
    const int d_start = li * 8;
    // --- Tính toán n_steps và các con trỏ (Không thay đổi) ---
    const int pos = seq_lengths[b];
    const bool do_sw = (use_sw && ((layer_idx & 1) == 0));
    const int t_start = do_sw ? max(0, pos - (SW_WINDOW - 1)) : 0;
    const int n_steps = pos - t_start + 1;
    const int kv_dim = D * KVH;
    const __hip_bfloat16 *__restrict__ K0 =
        key_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
    const __hip_bfloat16 *__restrict__ V0 =
        value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * D;
    // --- Tải vector Q (Không thay đổi) ---
    float qseg[8];
    {
        const float inv = 1.0f / sqrtf(64.0f);
        const __hip_bfloat16 *__restrict__ qh = q + (size_t)b * H * D + (size_t)h * D;
        const uint4 q_chunk = *reinterpret_cast<const uint4 *>(qh + d_start);
        const __hip_bfloat16 *q_bf16 = reinterpret_cast<const __hip_bfloat16 *>(&q_chunk);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            qseg[s] = __bfloat162float(q_bf16[s]) * inv;
        }
    }
    
    // --- Khởi tạo các biến tích lũy (Không thay đổi) ---
    float m = -INFINITY, l = 0.f;
    float out8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // --- Vòng lặp chính - THAY ĐỔI TỪ TĨNH SANG ĐỘNG ---
    const int CHUNK_SIZE = 32; 
    extern __shared__ int s_dyn_mem[];
    int* s_next_tloc_ptr = s_dyn_mem;
    
    if (tix == 0) {
        *s_next_tloc_ptr = 0;
    }
    __syncthreads();
    while (true) {
        int my_chunk_start_tloc;
        if (lane == 0) {
            my_chunk_start_tloc = atomicAdd(s_next_tloc_ptr, CHUNK_SIZE);
        }
        my_chunk_start_tloc = __shfl(my_chunk_start_tloc, 0, 64);
        if (my_chunk_start_tloc >= n_steps) {
            break;
        }
        
        const int t_end_for_chunk = min(my_chunk_start_tloc + CHUNK_SIZE, n_steps);
        for (int tloc = my_chunk_start_tloc; tloc < t_end_for_chunk; ++tloc) {
            const int t_abs = t_start + tloc;
            const int tw = do_sw ? (t_abs % SW_WINDOW) : t_abs;
            // Q.K^T
            const size_t k_offset = (size_t)tw * kv_dim + d_start;
            const uint4 k_chunk = *reinterpret_cast<const uint4 *>(K0 + k_offset);
            const __hip_bfloat16 *k_bf16 = reinterpret_cast<const __hip_bfloat16 *>(&k_chunk);
            float part = 0.f;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                part = fmaf(qseg[s], __bfloat162float(k_bf16[s]), part);
            }
            part += __shfl_xor(part, 4, 8);
            part += __shfl_xor(part, 2, 8);
            part += __shfl_xor(part, 1, 8);
            float alpha = 1.f, w = 0.f;
            if (li == 0) {
                const float s = part;
                const float mNew = fmaxf(m, s);
                alpha = __expf(m - mNew);
                const float dm = s - mNew;
                w = (dm > -SOFTMAX_PHI) ? __expf(dm) : 0.f;
                l = l * alpha + w;
                m = mNew;
            }
            alpha = __shfl(alpha, (hloc << 3), 64);
            w = __shfl(w, (hloc << 3), 64);
            // S.V
            const size_t v_offset = (size_t)tw * kv_dim + d_start;
            const uint4 v_chunk = *reinterpret_cast<const uint4 *>(V0 + v_offset);
            const __hip_bfloat16 *v_bf16 = reinterpret_cast<const __hip_bfloat16 *>(&v_chunk);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                out8[s] = fmaf(w, __bfloat162float(v_bf16[s]), out8[s] * alpha);
            }
        }
    }
    // --- Inter-warp reduction ---
    float *redbuf = (float*)(s_dyn_mem + 1);
    float *red_m = redbuf;
    float *red_l = red_m + nwarps * 8;
    float *red_out = red_l + nwarps * 8;
    if (li == 0) {
        red_m[wid * 8 + hloc] = m;
        red_l[wid * 8 + hloc] = l;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        red_out[((wid * 64 + ((hloc << 3) | li)) * 8 + s)] = out8[s];
    }
    __syncthreads();
    if (wid == 0) {
        const int my_h = hloc;
        const int my_li = li;
        float m_max = -INFINITY;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            m_max = fmaxf(m_max, red_m[w * 8 + my_h]);
        }
        float l_sum = 0.f;
        float num8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            const float mw = red_m[w * 8 + my_h];
            const float scale = (mw == -INFINITY) ? 0.f : __expf(mw - m_max);
            l_sum = fmaf(red_l[w * 8 + my_h], scale, l_sum);
            const int base_idx = ((w * 64 + ((my_h << 3) | my_li)) * 8);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                num8[s] = fmaf(red_out[base_idx + s], scale, num8[s]);
            }
        }
        float alpha_sink = 1.f, l_final = l_sum;
        if (my_li == 0) {
            const float ss = __bfloat162float(sinks[kv_h * 8 + my_h]);
            const float m_new = fmaxf(m_max, ss);
            const float alpha2 = __expf(m_max - m_new);
            const float e = __expf(ss - m_new);
            l_final = l_sum * alpha2 + e;
            alpha_sink = alpha2;
        }
        alpha_sink = __shfl(alpha_sink, (my_h << 3), 64);
        l_final = __shfl(l_final, (my_h << 3), 64);
        __hip_bfloat16 *__restrict__ oh = output + (size_t)b * H * D + (size_t)h * D;
        __hip_bfloat16 out_bf16[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float val = (num8[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
            out_bf16[s] = __float2bfloat16(val);
        }
        *reinterpret_cast<uint4 *>(oh + d_start) = *reinterpret_cast<uint4 *>(out_bf16);
    }
}