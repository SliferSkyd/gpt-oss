#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <float.h>
#include <stdint.h>

#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif
#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-6f
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
