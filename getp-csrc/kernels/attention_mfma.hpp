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
__device__ inline float subgroup8_sum(float x) {
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
__global__ __launch_bounds__(64, 8)
void fused_attention_kernel_1warp8q(
    float * __restrict__ output,                  // [B, H, 64]
    const float * __restrict__ q,                 // [B, H, 64] fp32
    const __hip_bfloat16 * __restrict__ key_cache,// [..] bf16
    const __hip_bfloat16 * __restrict__ value_cache,
    const __hip_bfloat16 * __restrict__ sinks,    // [H] bf16 (layer-offset)
    const float * __restrict__ /*mask*/,          // unused
    const int * __restrict__ seq_lengths,         // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int /*seq_len*/, int n_layers, int layer_idx, bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t)
{
  const int kv_h = blockIdx.x;
  const int b    = blockIdx.y;
  if (b>=batch_size || kv_h>=n_kv_heads) return;
  if (head_dim != 64) return;

  // lane mapping
  const int lane = threadIdx.x;        // 0..63
  const int h_local = lane >> 3;       // 0..7 (which query head within KV group)
  const int li      = lane & 7;        // 0..7 (lane-in-head)
  const int gqa_ratio = 8;
  const int h = kv_h * gqa_ratio + h_local;
  if (h >= n_heads) return;

  // sliding window positions
  const int pos = seq_lengths[b];
  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0))
    t_start = max(0, pos - (SW_WINDOW - 1));
  const int n_steps = pos - t_start + 1;

  // base pointers
  const int kv_dim = head_dim * n_kv_heads;
  const __hip_bfloat16 * __restrict__ k_head_base =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset + (size_t)kv_h * head_dim;

  // preload q segment (8 elements of this head distributed across li)
  float q_seg[8];
  {
    const float * __restrict__ q_head =
        q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    #pragma unroll
    for (int s=0; s<8; ++s) q_seg[s] = q_head[li + 8*s];
  }

  // LDS for K/V tiles (bf16), vectorized & padded to avoid bank conflicts
  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16* sK = smem;
  __hip_bfloat16* sV = sK + (size_t)tile_t * PADDED_HEAD_DIM;

  // per-head streaming softmax state:
  float m = -INFINITY;  // only valid on li==0 lanes (head leaders)
  float l = 0.f;

  // per-lane accumulator for 8 dims of this head (kept in registers)
  float out_vec[8] = {0,0,0,0,0,0,0,0};
  const float inv_sqrt_d = rsqrtf(64.f);

  // cooperative staging config
  const int threads_total = blockDim.x;           // 64
  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // Stage K/V into LDS: [cur, PADDED_HEAD_DIM] (row-major)
    // Use 16B (8×bf16) vector loads
    const int vec8 = head_dim / 8;               // 8
    const int elems_vec8 = cur * vec8;
    for (int e8 = lane; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;
      const int i8   = (e8 - tloc * vec8) * 8;
      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      *reinterpret_cast<u128*>(sK + (size_t)tloc * PADDED_HEAD_DIM + i8) =
        *reinterpret_cast<const u128*>(k_head_base + (size_t)tw * kv_dim + i8);
      *reinterpret_cast<u128*>(sV + (size_t)tloc * PADDED_HEAD_DIM + i8) =
        *reinterpret_cast<const u128*>(v_head_base + (size_t)tw * kv_dim + i8);
    }
    __syncthreads();

    // Process tokens in this tile — true streaming softmax (no chunk buffering)
    #pragma unroll 1
    for (int t = 0; t < cur; ++t) {
      // dot = <q,h , k_t> over 64 dims distributed across the 8 lanes of this head
      float part = 0.f;
      #pragma unroll
      for (int s=0; s<8; ++s) {
        float k = __bfloat162float(sK[(size_t)t * PADDED_HEAD_DIM + (li + 8*s)]);
        part = fmaf(q_seg[s], k, part);
      }
      // reduce within subgroup (8 lanes per head)
      float dot = subgroup8_sum(part);
      float s = 0.f;
      if (li == 0) s = dot * inv_sqrt_d;

      // online softmax update on the head-leader (li==0)
      float alpha = 1.f, w = 0.f;
      if (li == 0) {
        float m_new = fmaxf(m, s);
        alpha = __expf(m - m_new);
        w     = __expf(s - m_new);
        l     = l * alpha + w;
        m     = m_new;
      }
      // broadcast alpha & w to 8-lane subgroup
      alpha = __shfl(alpha, (h_local<<3), 64);  // from leader of this head
      w     = __shfl(w,     (h_local<<3), 64);

      // numerator update for this lane's 8 dims: O = alpha*O + w*V_t
      #pragma unroll
      for (int s=0; s<8; ++s) {
        float v = __bfloat162float(sV[(size_t)t * PADDED_HEAD_DIM + (li + 8*s)]);
        out_vec[s] = fmaf(w, v, out_vec[s] * alpha);
      }
    }
    __syncthreads();
  }

  // Sink merge: apply sink to (m,l) and scale O by alpha_sink
  float alpha_sink = 1.f, l_final = 0.f;
  if (li == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, (h_local<<3), 64);
  l_final    = __shfl(l_final,    (h_local<<3), 64);

  // write back normalized
  float* __restrict__ out_head =
      output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  #pragma unroll
  for (int s=0; s<8; ++s) {
    const int d = li + 8*s;
    out_head[d] = (out_vec[s] * alpha_sink) / (l_final + SOFTMAX_EPS);
  }
}


__global__ __launch_bounds__(64, 8)
void fused_attention_kernel_1warp8q_fast(
    float * __restrict__ output, const float * __restrict__ q,
    const __hip_bfloat16 * __restrict__ key_cache,
    const __hip_bfloat16 * __restrict__ value_cache,
    const __hip_bfloat16 * __restrict__ sinks, const float* /*mask*/,
    const int * __restrict__ seq_lengths,
    int B, int H, int KVH, int D, int /*seq_len*/,
    int L, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int tile_t)
{
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b>=B || kv_h>=KVH || D!=64) return;
  const int lane = threadIdx.x;               // 0..63
  const int hloc = lane >> 3;                 // 0..7 (which Q head)
  const int li   = lane & 7;                  // 0..7 (lane-in-head)
  const int gqa  = 8;
  const int h = kv_h*gqa + hloc; if (h>=H) return;

  int pos = seq_lengths[b];
  int t_start = (use_sw && ((layer_idx & 1)==0)) ? max(0, pos-(SW_WINDOW-1)) : 0;
  const int n_steps = pos - t_start + 1;

  const int kv_dim = D * KVH;
  const __hip_bfloat16* __restrict__ K0 =
      key_cache   + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;
  const __hip_bfloat16* __restrict__ V0 =
      value_cache + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;

  // Pre-scale q by 1/sqrt(D) once
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float* __restrict__ qh = q + (size_t)b*H*D + (size_t)h*D;
    #pragma unroll
    for (int s=0; s<8; ++s) qseg[s] = qh[li + 8*s] * inv;
  }

  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16* sK = smem;
  __hip_bfloat16* sV = sK + (size_t)tile_t * PADDED_HEAD_DIM;

  // streaming softmax state (per head, stored on li==0 lane)
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0,0,0,0,0,0,0,0};

  // vec load helper
  struct u128 { uint4 v; };
  auto ld8 = [](__hip_bfloat16 const* p){ u128 r; r.v=*reinterpret_cast<uint4 const*>(p); return r; };

  for (int base=0; base<n_steps; base+=tile_t) {
    const int cur = min(tile_t, n_steps-base);

    // stage K/V
    const int vec8 = D/8;                 // 8 bf16 = 16B
    const int tot  = cur * vec8;
    for (int e=lane; e<tot; e+=64) {
      int tloc = e / vec8;
      int i8   = (e - tloc*vec8) * 8;
      int t_abs = t_start + base + tloc;
      int tw = ((layer_idx & 1)==0) ? (t_abs % SW_WINDOW) : t_abs;
      *reinterpret_cast<u128*>(sK + (size_t)tloc*PADDED_HEAD_DIM + i8) =
        ld8(K0 + (size_t)tw*kv_dim + i8);
      *reinterpret_cast<u128*>(sV + (size_t)tloc*PADDED_HEAD_DIM + i8) =
        ld8(V0 + (size_t)tw*kv_dim + i8);
    }
    __syncthreads();

    // tokens in this tile
    #pragma unroll 1
    for (int t=0; t<cur; ++t) {
      const __hip_bfloat16* __restrict__ krow = sK + (size_t)t*PADDED_HEAD_DIM;
      const __hip_bfloat16* __restrict__ vrow = sV + (size_t)t*PADDED_HEAD_DIM;

      // dot for this head across 64 dims split over 8 lanes
      float part = 0.f;
      #pragma unroll
      for (int s=0; s<8; ++s) {
        float k = __bfloat162float(krow[li + 8*s]);
        part = fmaf(qseg[s], k, part);      // q pre-scaled
      }
      // subgroup-8 sum
      part += __shfl_xor(part, 4, 8);
      part += __shfl_xor(part, 2, 8);
      part += __shfl_xor(part, 1, 8);

      // leader does alpha/w; broadcast to subgroup
      float alpha = 1.f, w = 0.f;
      if (li==0) {
        float s = part;                      // already scaled
        float m_new = fmaxf(m, s);
        alpha = __expf(m - m_new);
        w     = __expf(s - m_new);
        l     = l * alpha + w;
        m     = m_new;
      }
      alpha = __shfl(alpha, (hloc<<3), 64);
      w     = __shfl(w,     (hloc<<3), 64);

      // numerator update for this lane's 8 dims
      #pragma unroll
      for (int s=0; s<8; ++s) {
        float v = __bfloat162float(vrow[li + 8*s]);
        out8[s] = fmaf(w, v, out8[s] * alpha);
      }
    }
    __syncthreads();
  }

  // sink merge
  float alpha_sink = 1.f, l_final = 0.f;
  if (li==0) {
    float ss = __bfloat162float(sinks[h]);
    float m_new = fmaxf(m, ss);
    float alpha = __expf(m - m_new);
    float e     = __expf(ss - m_new);
    l           = l * alpha + e;
    alpha_sink  = alpha;
    l_final     = l;
  }
  alpha_sink = __shfl(alpha_sink, (hloc<<3), 64);
  l_final    = __shfl(l_final,    (hloc<<3), 64);

  // store
  float* __restrict__ oh = output + (size_t)b*H*D + (size_t)h*D;
  #pragma unroll
  for (int s=0; s<8; ++s) {
    int d = li + 8*s;
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
__global__ __launch_bounds__(64, 8)
void flashdecoding_phase1_kernel_1warp8q(
    float * __restrict__ partial_out,   // [Z, B, H, D]  (Z = max_splits)
    float * __restrict__ partial_m,     // [Z, B, H]
    float * __restrict__ partial_l,     // [Z, B, H]
    const float * __restrict__ q,       // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,
    const __hip_bfloat16 * __restrict__ value_cache,
    const int * __restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int split_t                                    // requested split length (tokens per split)
){
  const int kv_h = blockIdx.x, b = blockIdx.y, split_idx = blockIdx.z;
  if (b>=B || kv_h>=KVH || D!=64) return;

  // Lane layout: 64 threads = 8 heads * 8 lanes each (D=64 split across the 8 lanes)
  const int lane = threadIdx.x;               // 0..63
  const int hloc = lane >> 3;                 // which of the 8 Q heads in this block
  const int li   = lane & 7;                  // lane-in-head
  const int gqa  = 8;
  const int h = kv_h*gqa + hloc; if (h>=H) return;

  // Decoding position & sliding-window start
  const int pos = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1)==0)) ? max(0, pos-(SW_WINDOW-1)) : 0;
  const int n_steps = pos - t_start + 1;                   // number of tokens to attend (inclusive of pos)

  // This split covers tokens [s_base, s_base + s_len)
  const int s_base = split_idx * split_t;
  if (s_base >= n_steps) return;                            // inactive split
  const int s_len  = min(split_t, n_steps - s_base);

  // K/V pointers. Layout is time-major with stride kv_dim across time.
  const int kv_dim = D * KVH;
  const __hip_bfloat16* __restrict__ K0 =
      key_cache   + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;
  const __hip_bfloat16* __restrict__ V0 =
      value_cache + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;

  // Pre-scale q by 1/sqrt(D) once
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float* __restrict__ qh = q + (size_t)b*H*D + (size_t)h*D;
    #pragma unroll
    for (int s=0; s<8; ++s) qseg[s] = qh[li + 8*s] * inv;
  }

  // Shared mem for staging this split's [s_len x PADDED_HEAD_DIM] (K and V)
  extern __shared__ __hip_bfloat16 smem[];
  __hip_bfloat16* sK = smem;
  __hip_bfloat16* sV = sK + (size_t)split_t * PADDED_HEAD_DIM;  // reserve max; we'll use s_len subset

  // Streaming softmax state for this head inside the split
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0,0,0,0,0,0,0,0};

  // 16B vector load helper for bf16x8
  struct u128 { uint4 v; };
  auto ld8 = [](__hip_bfloat16 const* p){ u128 r; r.v=*reinterpret_cast<uint4 const*>(p); return r; };

  // Stage K/V for this split into shared memory (coalesced bf16x8)
  {
    const int vec8 = D/8;                  // 8 bf16 = 16B
    const int tot  = s_len * vec8;
    for (int e=lane; e<tot; e+=64) {
      const int tloc = e / vec8;
      const int i8   = (e - tloc*vec8) * 8;
      const int t_abs = t_start + s_base + tloc;
      const int tw = ((layer_idx & 1)==0) ? (t_abs % (use_sw ? SW_WINDOW : max(L,1))) : t_abs;
      *reinterpret_cast<u128*>(sK + (size_t)tloc*PADDED_HEAD_DIM + i8) = ld8(K0 + (size_t)tw*kv_dim + i8);
      *reinterpret_cast<u128*>(sV + (size_t)tloc*PADDED_HEAD_DIM + i8) = ld8(V0 + (size_t)tw*kv_dim + i8);
    }
    __syncthreads();
  }

  // Compute attention over this split (like FlashAttention streaming inside a block)
  #pragma unroll 1
  for (int t=0; t<s_len; ++t) {
    const __hip_bfloat16* __restrict__ krow = sK + (size_t)t*PADDED_HEAD_DIM;
    const __hip_bfloat16* __restrict__ vrow = sV + (size_t)t*PADDED_HEAD_DIM;

    // partial dot on 8-lane subgroup
    float part = 0.f;
    #pragma unroll
    for (int s=0; s<8; ++s) {
      float k = __bfloat162float(krow[li + 8*s]);
      part = fmaf(qseg[s], k, part);
    }
    // subgroup-8 sum (width=8)
    part += __shfl_xor(part, 4, 8);
    part += __shfl_xor(part, 2, 8);
    part += __shfl_xor(part, 1, 8);

    // leader lane does softmax update; broadcast within the 8-lane subgroup
    float alpha = 1.f, w = 0.f;
    if (li==0) {
      float s = part;                      // already 1/sqrt(D) scaled
      float m_new = fmaxf(m, s);
      alpha = __expf(m - m_new);
      w     = __expf(s - m_new);
      l     = l * alpha + w;
      m     = m_new;
    }
    alpha = __shfl(alpha, (hloc<<3), 64);
    w     = __shfl(w,     (hloc<<3), 64);

    // numerator update for this lane's 8 dims
    #pragma unroll
    for (int s=0; s<8; ++s) {
      float vv = __bfloat162float(vrow[li + 8*s]);
      out8[s] = fmaf(w, vv, out8[s] * alpha);
    }
  }

  // Write partials for this split (per head)
  float* __restrict__ pout = partial_out + ((size_t)split_idx*B*H + (size_t)b*H + (size_t)h) * D;
  #pragma unroll
  for (int s=0; s<8; ++s) {
    const int d = li + 8*s;
    pout[d] = out8[s];                       // unnormalized numerator n_s (w.r.t m_s)
  }
  if (li==0) {
    float* pm = partial_m + (size_t)split_idx*B*H + (size_t)b*H + (size_t)h;
    float* pl = partial_l + (size_t)split_idx*B*H + (size_t)b*H + (size_t)h;
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
__global__ __launch_bounds__(64, 8)
void flashdecoding_reduce_splits_kernel_1warp8q(
    float * __restrict__ output,                // [B, H, D]
    const float * __restrict__ partial_out,     // [Z, B, H, D]
    const float * __restrict__ partial_m,       // [Z, B, H]
    const float * __restrict__ partial_l,       // [Z, B, H]
    const __hip_bfloat16 * __restrict__ sinks,  // [H] (optional scalar sink scores per head)
    const int * __restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    int max_splits,                              // Z used in phase 1
    int split_t
){
  const int kv_h = blockIdx.x, b = blockIdx.y;
  if (b>=B || kv_h>=KVH || D!=64) return;

  const int lane = threadIdx.x;               // 0..63
  const int hloc = lane >> 3;                 // which of 8 Q heads in this block
  const int li   = lane & 7;                  // lane-in-head
  const int gqa  = 8;
  const int h = kv_h*gqa + hloc; if (h>=H) return;

  // How many splits are valid for this (b,h)
  const int pos = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1)==0)) ? max(0, pos-(SW_WINDOW-1)) : 0;
  const int n_steps = pos - t_start + 1;
  const int n_valid = (n_steps + split_t - 1) / split_t;

  // Streaming reduction across splits: (m,l,out8)
  float m = -INFINITY, l = 0.f;
  float out8[8] = {0,0,0,0,0,0,0,0};

  for (int sidx=0; sidx<n_valid; ++sidx) {
    const size_t base_h = (size_t)sidx*B*H + (size_t)b*H + (size_t)h;

    // Load this split's state
    float ms = partial_m[base_h];
    float ls = partial_l[base_h];

    // Combine (m,l) in a numerically stable streaming fashion
    float m_new = fmaxf(m, ms);
    float alpha = __expf(m - m_new);    // scales existing accumulator
    float beta  = __expf(ms - m_new);   // scales incoming split

    // Accumulate numerator vector
    const float* __restrict__ nin = partial_out + base_h * D;
    #pragma unroll
    for (int t=0; t<8; ++t) {
      const int d = li + 8*t;
      float add = nin[d] * beta;
      out8[t] = out8[t]*alpha + add;
    }
    // Accumulate denominator
    l = l*alpha + ls*beta;
    m = m_new;
  }

  // Sink merge AFTER all splits: affects normalization only (and scales numerator if m increases)
  float alpha_sink = 1.f, l_final = l;
  if (li==0) {
    float ss = sinks ? __bfloat162float(sinks[h]) : -INFINITY;  // if no sinks, choose -inf to no-op
    float m_new = fmaxf(m, ss);
    float a = __expf(m - m_new);
    float e = __expf(ss - m_new);
    l_final = l * a + e;
    alpha_sink = a;
    m = m_new;
  }
  alpha_sink = __shfl(alpha_sink, (hloc<<3), 64);
  l_final    = __shfl(l_final,    (hloc<<3), 64);

  // Write final normalized output
  float* __restrict__ oh = output + (size_t)b*H*D + (size_t)h*D;
  const float inv_den = 1.f / (l_final + SOFTMAX_EPS);
  #pragma unroll
  for (int t=0; t<8; ++t) {
    const int d = li + 8*t;
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
#define FD_MAX_WARPS_PER_CTA 4   // Tune: {2,4,6,8}. 4 is a good default for MI2xx/MI3xx.
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
                                            float &m_out, float &l_out, float &alpha, float &beta) {
  float m_new = fmaxf(m1, m2);
  float a = __expf(m1 - m_new);
  float b = __expf(m2 - m_new);
  m_out = m_new;
  l_out = l1 * a + l2 * b;
  alpha = a;
  beta  = b;
}

// Kernel mapping:
// - blockIdx = (kv_h, b, 0)
// - blockDim.x = 64 * W, where W = number of warps per block (<= FD_MAX_WARPS_PER_CTA)
// - Each warp processes one split at a time: split indices s = warp_id, warp_id+W, ...
// - Within a warp: 64 lanes = 8 heads (hloc) * 8 lanes per head (li), D=64 per head.
// - No shared K/V. Each subgroup-8 (li=0..7) performs coalesced strided loads.
__global__ __launch_bounds__(64 * FD_MAX_WARPS_PER_CTA, 2)
void flashdecoding_fused_multiwarp_1warp8q(
    float * __restrict__ output,               // [B, H, D]
    const float * __restrict__ q,              // [B, H, D], fp32
    const __hip_bfloat16 * __restrict__ key_cache,
    const __hip_bfloat16 * __restrict__ value_cache,
    const __hip_bfloat16 * __restrict__ sinks, // [H], optional (can be nullptr)
    const int * __restrict__ seq_lengths,
    int B, int H, int KVH, int D, int L,
    int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int split_t                                  // tokens per split (tile over sequence)
){
  // Block coords
  const int kv_h = blockIdx.x;
  const int b    = blockIdx.y;
  if (b>=B || kv_h>=KVH || D!=64) return;

  // Thread coords
  const int tid     = threadIdx.x;
  const int warp_id = tid >> 6;          // 0..(W-1)
  const int lane    = tid & 63;          // 0..63
  const int hloc    = lane >> 3;         // 0..7 (which of 8 query heads mapped to this warp)
  const int li      = lane & 7;          // 0..7 lane-in-head
  const int W       = blockDim.x >> 6;   // runtime warps/CTA

  // Head index for this lane
  const int gqa = 8;
  const int h = kv_h * gqa + hloc;
  if (h >= H) return;

  // Effective sequence region for this layer (sliding window if enabled on even layers)
  const int pos     = seq_lengths[b];
  const int t_start = (use_sw && ((layer_idx & 1)==0)) ? max(0, pos-(SW_WINDOW-1)) : 0;
  const int n_steps = pos - t_start + 1;
  if (n_steps <= 0) return;

  // Number of splits for this (b, kv_h)
  const int n_splits = (n_steps + split_t - 1) / split_t;

  // Base K/V pointer setup
  const int kv_dim = D * KVH;
  const __hip_bfloat16* __restrict__ K0 =
      key_cache   + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;
  const __hip_bfloat16* __restrict__ V0 =
      value_cache + (size_t)b*batch_kv_stride + layer_kv_offset + (size_t)kv_h*D;

  // Load & scale q once (1/sqrt(D))
  float qseg[8];
  {
    const float inv = rsqrtf(64.f);
    const float * __restrict__ qh = q + (size_t)b*H*D + (size_t)h*D;
    #pragma unroll
    for (int s=0; s<8; ++s) qseg[s] = qh[li + 8*s] * inv;
  }

  // Per-warp local accumulation across its subset of splits
  float m_loc = -INFINITY;
  float l_loc = 0.f;
  float out8_loc[8] = {0,0,0,0,0,0,0,0};

  // Iterate over splits assigned to this warp
  for (int sidx = warp_id; sidx < n_splits; sidx += W) {
    const int s_base = sidx * split_t;
    const int s_len  = min(split_t, n_steps - s_base);

    // Streaming softmax within this split
    float m_s = -INFINITY, l_s = 0.f;
    float out8_s[8] = {0,0,0,0,0,0,0,0};

    #pragma unroll 1
    for (int t=0; t<s_len; ++t) {
      const int t_abs = t_start + s_base + t;
      const int tw    = ((layer_idx & 1)==0) ? (t_abs % (use_sw ? SW_WINDOW : max(L,1))) : t_abs;

      // Dot-product partial for this head across its 8 lanes (subgroup-8 reduction)
      float part = 0.f;
      #pragma unroll
      for (int s=0; s<8; ++s) {
        const int d = li + 8*s;
        float k = __bfloat162float(K0[(size_t)tw*kv_dim + d]);
        part = fmaf(qseg[s], k, part);
      }
      // subgroup-8 sum (width=8)
      part += __shfl_xor(part, 4, 8);
      part += __shfl_xor(part, 2, 8);
      part += __shfl_xor(part, 1, 8);

      // Leader lane computes softmax update; add phi short-circuit
      float alpha = 1.f, w = 0.f;
      if (li==0) {
        const float s = part;                 // score already scaled
        const float dm = s - m_s;
        if (dm <= -SOFTMAX_PHI) {
          // Tiny contribution; keep state unchanged and skip expf
          alpha = 1.f; w = 0.f;
        } else {
          const float m_new = fmaxf(m_s, s);
          alpha = __expf(m_s - m_new);
          w     = __expf(s   - m_new);
          l_s   = l_s * alpha + w;
          m_s   = m_new;
        }
      }
      alpha = __shfl(alpha, (hloc<<3), 64);   // broadcast to subgroup
      w     = __shfl(w,     (hloc<<3), 64);

      // Numerator update (8 dims for this lane)
      #pragma unroll
      for (int s=0; s<8; ++s) {
        const int d = li + 8*s;
        float vv = __bfloat162float(V0[(size_t)tw*kv_dim + d]);
        out8_s[s] = fmaf(w, vv, out8_s[s] * alpha);
      }
    } // split tokens

    // Merge this split into warp-local accumulator using log-sum-exp combine
    {
      float m_new, l_new, alpha, beta;
      merge_softmax_states(m_loc, l_loc, m_s, l_s, m_new, l_new, alpha, beta);
      #pragma unroll
      for (int s=0; s<8; ++s) {
        out8_loc[s] = out8_loc[s] * alpha + out8_s[s] * beta;
      }
      m_loc = m_new;
      l_loc = l_new;
    }
  } // splits loop

  // ===== Block-wide reduction across warps (merge per-warp (m,l,out)) =====
  // Use dynamic shared memory as a compact reduction buffer.
  extern __shared__ float redbuf[];
  float* sm_out = redbuf;                                        // [W, 64, 8] floats
  float* sm_m   = sm_out + (size_t)W * 64 * 8;                   // [W, 8] floats (per head)
  float* sm_l   = sm_m   + (size_t)W * 8;                        // [W, 8] floats (per head)

  // Write this warp's vector to smem
  #pragma unroll
  for (int s=0; s<8; ++s) {
    sm_out[(size_t)warp_id*64*8 + (size_t)lane*8 + s] = out8_loc[s];
  }
  // Leader of each subgroup writes (m,l) per head
  if (li==0) {
    sm_m[warp_id*8 + hloc] = m_loc;
    sm_l[warp_id*8 + hloc] = l_loc;
  }
  __syncthreads();

  // Let warp 0 do the final merge (W is small; this is cheap and avoids extra syncs)
  if (warp_id == 0) {
    // Start from warp 0's state
    float m_fin = sm_m[0*8 + hloc];
    float l_fin = sm_l[0*8 + hloc];
    float out8_fin[8];
    #pragma unroll
    for (int s=0; s<8; ++s) {
      out8_fin[s] = sm_out[(size_t)0*64*8 + (size_t)lane*8 + s];
    }
    // Merge remaining warps
    for (int w=1; w<W; ++w) {
      float mw = sm_m[w*8 + hloc];
      float lw = sm_l[w*8 + hloc];
      float m_new, l_new, alpha, beta;
      merge_softmax_states(m_fin, l_fin, mw, lw, m_new, l_new, alpha, beta);
      #pragma unroll
      for (int s=0; s<8; ++s) {
        const float add = sm_out[(size_t)w*64*8 + (size_t)lane*8 + s] * beta;
        out8_fin[s] = out8_fin[s] * alpha + add;
      }
      m_fin = m_new;
      l_fin = l_new;
    }

    // Sink merge (post-reduction) — normalization only; scale numerator if m increases
    float alpha_sink = 1.f, l_final = l_fin;
    if (li==0) {
      float ss = sinks ? __bfloat162float(sinks[h]) : -INFINITY;
      float m_new = fmaxf(m_fin, ss);
      float a = __expf(m_fin - m_new);
      float e = __expf(ss    - m_new);
      l_final = l_fin * a + e;
      alpha_sink = a;
      m_fin = m_new;
    }
    alpha_sink = __shfl(alpha_sink, (hloc<<3), 64);
    l_final    = __shfl(l_final,    (hloc<<3), 64);

    // Write final normalized output
    float* __restrict__ oh = output + (size_t)b*H*D + (size_t)h*D;
    const float inv_den = 1.f / (l_final + SOFTMAX_EPS);
    #pragma unroll
    for (int s=0; s<8; ++s) {
      const int d = li + 8*s;
      oh[d] = (out8_fin[s] * alpha_sink) * inv_den;
    }
  }
}


static size_t get_device_max_dyn_shmem() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  int basic = 0;
  hipError_t st_basic = hipDeviceGetAttribute(
      &basic, hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
  if (st_basic != hipSuccess || basic <= 0) {
    // Fallback if the query ever fails: assume 64 KiB (typical default)
    basic = 64 * 1024;
  }

  // On NVIDIA (HIP on CUDA), an "optin" cap may exist. Guard it so ROCm compiles.
  // Only prefer it if it’s larger than the basic per-block limit.
#if defined(__HIP_PLATFORM_NVIDIA__) && defined(hipDeviceAttributeMaxSharedMemoryPerBlockOptin)
  int optin = 0;
  hipError_t st_optin = hipDeviceGetAttribute(
      &optin, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, dev);
  if (st_optin == hipSuccess && optin > basic) {
    return static_cast<size_t>(optin);
  }
#endif

  return static_cast<size_t>(basic);
}

