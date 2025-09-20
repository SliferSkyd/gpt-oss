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

