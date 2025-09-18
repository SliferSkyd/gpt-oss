#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cfloat>

// === Tunables ===
#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif

// Softmax shift (phi). Start conservatively; you can tighten later.
#ifndef SOFTMAX_PHI
#define SOFTMAX_PHI 8.0f
#endif

// If any score - phi exceeds this, we fallback to exact max (avoid exp overflow / bad scaling)
#ifndef SOFTMAX_PHI_OVERFLOW_GUARD
#define SOFTMAX_PHI_OVERFLOW_GUARD 80.0f
#endif

#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-9f
#endif

// ===== warp/block reductions (HIP-safe) =====
__device__ inline float warpReduceMax(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off));
    return v;
}
__device__ inline float warpReduceSum(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}
__device__ inline float blockReduceMax(float v, float *shared_mem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1); // warp id
    v = warpReduceMax(v);
    if (lane == 0) shared_mem[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = shared_mem[lane];
    __syncthreads();
    out = warpReduceMax(out);
    if (lane == 0 && wid == 0) shared_mem[0] = out;
    __syncthreads();
    return shared_mem[0];
}
__device__ inline float blockReduceSum(float v, float *shared_mem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1);
    v = warpReduceSum(v);
    if (lane == 0) shared_mem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = shared_mem[lane];
    __syncthreads();
    out = warpReduceSum(out);
    if (lane == 0 && wid == 0) shared_mem[0] = out;
    __syncthreads();
    return shared_mem[0];
}

// ================================================================
// Fused Attention (decode) — Tiled K/V + Streaming Softmax (bf16 KV)
// - Packs kv_mul heads per block for GQA reuse
// - Vectorized 16B loads (8×bf16) → unpack → float tiles in LDS
// - Sliding-window on even layers (mod SW_WINDOW)
// - Keeps SOFTMAX_PHI as a define (not used in streaming path)
// - Interface matches your current callsite arguments
// ================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <float.h>
#include <stdint.h>

#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif
#ifndef SOFTMAX_PHI
#define SOFTMAX_PHI 0.0f
#endif
#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-6f
#endif

// Enable the fast online/streaming softmax
#ifndef SOFTMAX_USE_ONLINE
#define SOFTMAX_USE_ONLINE 1
#endif

// ----------------- Warp helpers -----------------
__device__ inline float warp_reduce_sum(float v) {
  for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

// ----------------- 16B bf16 loader + unpack -----------------
struct u128 { uint4 v; }; // 16 bytes

__device__ inline u128 gload_bf16x8(const __hip_bfloat16 *ptr) {
  u128 out;
  out.v = *reinterpret_cast<const uint4*>(ptr); // expect 16B alignment
  return out;
}

__device__ inline void unpack_bf16x8_to_f32(
    const u128 &src,
    float &f0,float &f1,float &f2,float &f3,
    float &f4,float &f5,float &f6,float &f7)
{
  const uint32_t *p32 = reinterpret_cast<const uint32_t*>(&src.v);
  uint32_t w0 = p32[0], w1 = p32[1], w2 = p32[2], w3 = p32[3];
  uint16_t b0 =  w0        & 0xFFFFu;
  uint16_t b1 = (w0 >> 16) & 0xFFFFu;
  uint16_t b2 =  w1        & 0xFFFFu;
  uint16_t b3 = (w1 >> 16) & 0xFFFFu;
  uint16_t b4 =  w2        & 0xFFFFu;
  uint16_t b5 = (w2 >> 16) & 0xFFFFu;
  uint16_t b6 =  w3        & 0xFFFFu;
  uint16_t b7 = (w3 >> 16) & 0xFFFFu;

  f0 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b0));
  f1 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b1));
  f2 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b2));
  f3 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b3));
  f4 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b4));
  f5 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b5));
  f6 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b6));
  f7 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&b7));
}

// ================================================================
// Kernel: packs kv_mul heads per block for K/V tile reuse
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * head_dim * sizeof(float)
// ================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel( // <-- keep your original symbol name
    float * __restrict__ output,                    // [B, H, D]
    const float * __restrict__ q,                   // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,  // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache,// [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,      // [H] (bf16, layer-offset)
    const float * __restrict__ mask,                // unused
    const int * __restrict__ seq_lengths,           // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t                                       // cooperative tile length
) {
  const int kv_h  = blockIdx.x;
  const int b     = blockIdx.y;
  const int warp  = threadIdx.y;           // 0..kv_mul-1
  const int lane  = threadIdx.x;           // 0..63
  const int kv_mul = blockDim.y;
  const int h     = kv_h * kv_mul + warp;

  if (b >= batch_size || kv_h >= n_kv_heads || warp >= kv_mul || h >= n_heads)
    return;

  // decode step for this batch
  const int pos = seq_lengths[b];

  // sliding window on even layers
  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1; // core tokens (sink handled after)

  const int kv_dim    = head_dim * n_kv_heads;
  const int gqa_ratio = n_heads / n_kv_heads;
  // sanity: ensure mapping is consistent
  if ((h / gqa_ratio) != kv_h) return;

  // base pointers (b, layer)
  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  // per-kv head base (avoid recomputing kv_h * head_dim)
  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * head_dim;

  // query lane value
  const float * __restrict__ q_head =
      q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  const float qi = (lane < head_dim) ? q_head[lane] : 0.0f;

  // shared tiles (float)
  extern __shared__ float shared_mem[];
  float *sK = shared_mem;
  float *sV = shared_mem + (size_t)tile_t * head_dim;
  const int width = head_dim;

#if SOFTMAX_USE_ONLINE
  float out_i = 0.0f;         // running output dim for this lane
  float m = -INFINITY;    // running max for logits
  float l = 0.0f;             // running denominator
#endif

  // 2D cooperative loader indexing
  const int threads_total = blockDim.x * blockDim.y;          // 64 * kv_mul
  const int tid2D         = threadIdx.y * blockDim.x + lane;  // 0..threads_total-1

  // Tile over time
  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // Vectorized part: 8×bf16 per 16B transaction
    const int vec8 = head_dim / 8;
    const int elems_vec8 = cur * vec8;

    for (int e8 = tid2D; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;              // 0..cur-1
      const int i8   = (e8 - tloc * vec8) * 8; // starting dim (multiple of 8)

      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;

      const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + i8;
      const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + i8;

      // 1×16B global loads
      u128 k128 = gload_bf16x8(k_ptr);
      u128 v128 = gload_bf16x8(v_ptr);

      float kf0,kf1,kf2,kf3,kf4,kf5,kf6,kf7;
      float vf0,vf1,vf2,vf3,vf4,vf5,vf6,vf7;
      unpack_bf16x8_to_f32(k128, kf0,kf1,kf2,kf3,kf4,kf5,kf6,kf7);
      unpack_bf16x8_to_f32(v128, vf0,vf1,vf2,vf3,vf4,vf5,vf6,vf7);

      float * __restrict__ k_row = sK + (size_t)tloc * width;
      float * __restrict__ v_row = sV + (size_t)tloc * width;

      // contiguous shared writes
      k_row[i8+0]=kf0; k_row[i8+1]=kf1; k_row[i8+2]=kf2; k_row[i8+3]=kf3;
      k_row[i8+4]=kf4; k_row[i8+5]=kf5; k_row[i8+6]=kf6; k_row[i8+7]=kf7;

      v_row[i8+0]=vf0; v_row[i8+1]=vf1; v_row[i8+2]=vf2; v_row[i8+3]=vf3;
      v_row[i8+4]=vf4; v_row[i8+5]=vf5; v_row[i8+6]=vf6; v_row[i8+7]=vf7;
    }

    // Tail if head_dim % 8 != 0
    const int tail = head_dim - vec8 * 8;
    if (tail) {
      const int elems_tail = cur * tail;
      for (int et = tid2D; et < elems_tail; et += threads_total) {
        const int tloc = et / tail;
        const int io   = (et - tloc * tail) + vec8 * 8;

        const int t_abs = t_start + base + tloc;
        const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;

        const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + io;
        const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + io;

        float kf = __bfloat162float(*k_ptr);
        float vf = __bfloat162float(*v_ptr);
        sK[(size_t)tloc * width + io] = kf;
        sV[(size_t)tloc * width + io] = vf;
      }
    }

    __syncthreads();

    // ---- Consume the tile with streaming softmax ----
#if SOFTMAX_USE_ONLINE
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int t = 0; t < cur; ++t) {
      float part = (lane < head_dim) ? (qi * sK[(size_t)t * width + lane]) : 0.0f;
      part = warp_reduce_sum(part);          // dot(q, k_t)

      float e = 0.f, alpha = 0.f;
      if (lane == 0) {
        float s     = part * inv_sqrt_d;
        float m_new = fmaxf(m, s);
        alpha       = __expf(m - m_new);
        e           = __expf(s - m_new);
        l           = l * alpha + e;
        m           = m_new;
      }
      e     = __shfl(e, 0);
      alpha = __shfl(alpha, 0);

      if (lane < head_dim) {
        float v = sV[(size_t)t * width + lane];
        out_i = fmaf(e, v, alpha * out_i);   // out = alpha*out + e*v
      }
    }
#else
    // (Optional) two-pass fixed-phi path, not shown to keep code compact
    // You can keep your old two-pass if needed.
#endif

    __syncthreads();
  }

  // ---- Sink update & write out ----
#if SOFTMAX_USE_ONLINE
  float alpha_sink = 0.f, l_final = 0.f;
  if (lane == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    m            = m_new;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, 0);
  l_final    = __shfl(l_final, 0);

  if (lane < head_dim) {
    float out_norm = (alpha_sink * out_i) / (l_final + SOFTMAX_EPS);
    float * __restrict__ out_head =
        output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    out_head[lane] = out_norm;
  }
#else
  // If using fixed-phi path, normalize accordingly (omitted)
#endif
}



// How many K/V pairs to process in the unrolled loop.
// This is the core of the optimization. A value of 4 is a good starting point.
#define ATTN_UNROLL_FACTOR 8


// ================================================================
// Kernel: packs kv_mul heads per block for K/V tile reuse
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * head_dim * sizeof(float)
// ================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized( // <-- Symbol name remains the same
    float * __restrict__ output,               // [B, H, D]
    const float * __restrict__ q,               // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,  // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache,// [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,      // [H] (bf16, layer-offset)
    const float * __restrict__ mask,               // unused
    const int * __restrict__ seq_lengths,           // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t                                     // cooperative tile length
) {
  const int kv_h  = blockIdx.x;
  const int b     = blockIdx.y;
  const int warp  = threadIdx.y;      // 0..kv_mul-1
  const int lane  = threadIdx.x;      // 0..63
  const int kv_mul = blockDim.y;
  const int h     = kv_h * kv_mul + warp;

  if (b >= batch_size || kv_h >= n_kv_heads || warp >= kv_mul || h >= n_heads)
    return;

  // decode step for this batch
  const int pos = seq_lengths[b];

  // sliding window on even layers
  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1; // core tokens (sink handled after)

  const int kv_dim    = head_dim * n_kv_heads;
  const int gqa_ratio = n_heads / n_kv_heads;
  // sanity: ensure mapping is consistent
  if ((h / gqa_ratio) != kv_h) return;

  // base pointers (b, layer)
  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  // per-kv head base (avoid recomputing kv_h * head_dim)
  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * head_dim;

  // query lane value
  const float * __restrict__ q_head =
      q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  const float qi = (lane < head_dim) ? q_head[lane] : 0.0f;

  // shared tiles (float)
  extern __shared__ float shared_mem[];
  float *sK = shared_mem;
  float *sV = shared_mem + (size_t)tile_t * head_dim;
  const int width = head_dim;

#if SOFTMAX_USE_ONLINE
  float out_i = 0.0f;     // running output dim for this lane
  float m = -INFINITY; // running max for logits
  float l = 0.0f;       // running denominator
#endif

  // 2D cooperative loader indexing
  const int threads_total = blockDim.x * blockDim.y;       // 64 * kv_mul
  const int tid2D         = threadIdx.y * blockDim.x + lane; // 0..threads_total-1

  // Tile over time
  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // [UNCHANGED] Cooperative loading from global to shared memory
    const int vec8 = head_dim / 8;
    const int elems_vec8 = cur * vec8;
    for (int e8 = tid2D; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;
      const int i8   = (e8 - tloc * vec8) * 8;
      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + i8;
      const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + i8;
      u128 k128 = gload_bf16x8(k_ptr);
      u128 v128 = gload_bf16x8(v_ptr);
      float kf0,kf1,kf2,kf3,kf4,kf5,kf6,kf7;
      float vf0,vf1,vf2,vf3,vf4,vf5,vf6,vf7;
      unpack_bf16x8_to_f32(k128, kf0,kf1,kf2,kf3,kf4,kf5,kf6,kf7);
      unpack_bf16x8_to_f32(v128, vf0,vf1,vf2,vf3,vf4,vf5,vf6,vf7);
      float * __restrict__ k_row = sK + (size_t)tloc * width;
      float * __restrict__ v_row = sV + (size_t)tloc * width;
      k_row[i8+0]=kf0; k_row[i8+1]=kf1; k_row[i8+2]=kf2; k_row[i8+3]=kf3;
      k_row[i8+4]=kf4; k_row[i8+5]=kf5; k_row[i8+6]=kf6; k_row[i8+7]=kf7;
      v_row[i8+0]=vf0; v_row[i8+1]=vf1; v_row[i8+2]=vf2; v_row[i8+3]=vf3;
      v_row[i8+4]=vf4; v_row[i8+5]=vf5; v_row[i8+6]=vf6; v_row[i8+7]=vf7;
    }
    const int tail = head_dim - vec8 * 8;
    if (tail) {
      const int elems_tail = cur * tail;
      for (int et = tid2D; et < elems_tail; et += threads_total) {
        const int tloc = et / tail;
        const int io   = (et - tloc * tail) + vec8 * 8;
        const int t_abs = t_start + base + tloc;
        const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
        const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + io;
        const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + io;
        sK[(size_t)tloc * width + io] = __bfloat162float(*k_ptr);
        sV[(size_t)tloc * width + io] = __bfloat162float(*v_ptr);
      }
    }
    __syncthreads();

    // ---- [OPTIMIZED] Consume the tile with unrolled streaming softmax ----
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    int t = 0;

    // Main unrolled loop to process 4 K/V pairs at a time
    for (; t < cur - (ATTN_UNROLL_FACTOR - 1); t += ATTN_UNROLL_FACTOR) {
        float k0_i = sK[(size_t)(t+0) * width + lane];
        float k1_i = sK[(size_t)(t+1) * width + lane];
        float k2_i = sK[(size_t)(t+2) * width + lane];
        float k3_i = sK[(size_t)(t+3) * width + lane];

        float v0_i = sV[(size_t)(t+0) * width + lane];
        float v1_i = sV[(size_t)(t+1) * width + lane];
        float v2_i = sV[(size_t)(t+2) * width + lane];
        float v3_i = sV[(size_t)(t+3) * width + lane];

        // Compute 4 scores in parallel across the warp
        float s0 = warp_reduce_sum(qi * k0_i);
        float s1 = warp_reduce_sum(qi * k1_i);
        float s2 = warp_reduce_sum(qi * k2_i);
        float s3 = warp_reduce_sum(qi * k3_i);
        
        // Lane 0 serially updates the softmax state for the 4 scores
        float e0, a0, e1, a1, e2, a2, e3, a3;
        if (lane == 0) {
            s0 *= inv_sqrt_d; float m_new0 = fmaxf(m, s0); a0 = __expf(m - m_new0); e0 = __expf(s0 - m_new0); l = l * a0 + e0; m = m_new0;
            s1 *= inv_sqrt_d; float m_new1 = fmaxf(m, s1); a1 = __expf(m - m_new1); e1 = __expf(s1 - m_new1); l = l * a1 + e1; m = m_new1;
            s2 *= inv_sqrt_d; float m_new2 = fmaxf(m, s2); a2 = __expf(m - m_new2); e2 = __expf(s2 - m_new2); l = l * a2 + e2; m = m_new2;
            s3 *= inv_sqrt_d; float m_new3 = fmaxf(m, s3); a3 = __expf(m - m_new3); e3 = __expf(s3 - m_new3); l = l * a3 + e3; m = m_new3;
        }

        // Broadcast update factors from lane 0 to all other lanes
        e0 = __shfl(e0, 0); a0 = __shfl(a0, 0);
        e1 = __shfl(e1, 0); a1 = __shfl(a1, 0);
        e2 = __shfl(e2, 0); a2 = __shfl(a2, 0);
        e3 = __shfl(e3, 0); a3 = __shfl(a3, 0);

        // All lanes update their output accumulator
        if (lane < head_dim) {
            out_i = fmaf(e3, v3_i, a3 * fmaf(e2, v2_i, a2 * fmaf(e1, v1_i, a1 * fmaf(e0, v0_i, a0 * out_i))));
        }
    }
    
    // Tail loop for remaining K/V pairs if cur is not divisible by 4
    for (; t < cur; ++t) {
        float part = (lane < head_dim) ? (qi * sK[(size_t)t * width + lane]) : 0.0f;
        part = warp_reduce_sum(part);

        float e = 0.f, alpha = 0.f;
        if (lane == 0) {
            float s     = part * inv_sqrt_d;
            float m_new = fmaxf(m, s);
            alpha       = __expf(m - m_new);
            e           = __expf(s - m_new);
            l           = l * alpha + e;
            m           = m_new;
        }
        e     = __shfl(e, 0);
        alpha = __shfl(alpha, 0);

        if (lane < head_dim) {
            float v = sV[(size_t)t * width + lane];
            out_i = fmaf(e, v, alpha * out_i);
        }
    }

    __syncthreads();
  }

  // ---- [UNCHANGED] Sink update & write out ----
#if SOFTMAX_USE_ONLINE
  float alpha_sink = 0.f, l_final = 0.f;
  if (lane == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    m            = m_new;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, 0);
  l_final    = __shfl(l_final, 0);

  if (lane < head_dim) {
    float out_norm = (alpha_sink * out_i) / (l_final + SOFTMAX_EPS);
    float * __restrict__ out_head =
        output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    out_head[lane] = out_norm;
  }
#endif
}

// ================================================================
// Kernel: packs kv_mul heads per block for K/V tile reuse
// - Optimization 1: Parallel online softmax update within the warp.
// - Optimization 2: Uses bfloat16 for K/V tiles in shared memory to halve usage.
//
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * head_dim * sizeof(__hip_bfloat16)
// ================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized_1(
    float * __restrict__ output,              // [B, H, D]
    const float * __restrict__ q,              // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,   // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache, // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,     // [H] (bf16, layer-offset)
    const float * __restrict__ mask,              // unused
    const int * __restrict__ seq_lengths,         // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t
) {
  const int kv_h  = blockIdx.x;
  const int b     = blockIdx.y;
  const int warp  = threadIdx.y;      // 0..kv_mul-1
  const int lane  = threadIdx.x;      // 0..63
  const int kv_mul = blockDim.y;
  const int h     = kv_h * kv_mul + warp;

  if (b >= batch_size || kv_h >= n_kv_heads || warp >= kv_mul || h >= n_heads)
    return;

  const int pos = seq_lengths[b];

  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1;

  const int kv_dim    = head_dim * n_kv_heads;
  const int gqa_ratio = n_heads / n_kv_heads;
  if ((h / gqa_ratio) != kv_h) return;

  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * head_dim;

  const float * __restrict__ q_head =
      q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  const float qi = (lane < head_dim) ? q_head[lane] : 0.0f;

  // OPTIMIZATION 2: Use bfloat16 for shared memory tiles
  extern __shared__ __hip_bfloat16 shared_mem_bf16[];
  __hip_bfloat16 *sK = shared_mem_bf16;
  __hip_bfloat16 *sV = shared_mem_bf16 + (size_t)tile_t * head_dim;
  const int width = head_dim;

  float out_i = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int threads_total = blockDim.x * blockDim.y;
  const int tid2D         = threadIdx.y * blockDim.x + lane;

  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // Cooperative loading from global to shared memory (bfloat16)
    const int vec8 = head_dim / 8;
    const int elems_vec8 = cur * vec8;
    for (int e8 = tid2D; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;
      const int i8   = (e8 - tloc * vec8) * 8;
      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + i8;
      const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + i8;

      // Directly store 8xbf16 values (16 bytes)
      *reinterpret_cast<u128*>(sK + (size_t)tloc * width + i8) = gload_bf16x8(k_ptr);
      *reinterpret_cast<u128*>(sV + (size_t)tloc * width + i8) = gload_bf16x8(v_ptr);
    }
    const int tail = head_dim - vec8 * 8;
    if (tail) {
      const int elems_tail = cur * tail;
      for (int et = tid2D; et < elems_tail; et += threads_total) {
        const int tloc = et / tail;
        const int io   = (et - tloc * tail) + vec8 * 8;
        const int t_abs = t_start + base + tloc;
        const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
        sK[(size_t)tloc * width + io] = k_head_base[(size_t)tw * kv_dim + io];
        sV[(size_t)tloc * width + io] = v_head_base[(size_t)tw * kv_dim + io];
      }
    }
    __syncthreads();

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    int t = 0;

    for (; t < cur - (ATTN_UNROLL_FACTOR - 1); t += ATTN_UNROLL_FACTOR) {
      // Load 4 K and V vectors, converting to float32 on the fly
      float k0_i = __bfloat162float(sK[(size_t)(t+0)*width+lane]);
      float k1_i = __bfloat162float(sK[(size_t)(t+1)*width+lane]);
      float k2_i = __bfloat162float(sK[(size_t)(t+2)*width+lane]);
      float k3_i = __bfloat162float(sK[(size_t)(t+3)*width+lane]);

      float v0_i = __bfloat162float(sV[(size_t)(t+0)*width+lane]);
      float v1_i = __bfloat162float(sV[(size_t)(t+1)*width+lane]);
      float v2_i = __bfloat162float(sV[(size_t)(t+2)*width+lane]);
      float v3_i = __bfloat162float(sV[(size_t)(t+3)*width+lane]);

      // Compute 4 scores in parallel
      float s0 = warp_reduce_sum(qi * k0_i);
      float s1 = warp_reduce_sum(qi * k1_i);
      float s2 = warp_reduce_sum(qi * k2_i);
      float s3 = warp_reduce_sum(qi * k3_i);

      // OPTIMIZATION 1: Parallel softmax update
      float p0, p1, p2, p3, alpha, beta;
      if (lane == 0) {
        s0 *= inv_sqrt_d; s1 *= inv_sqrt_d; s2 *= inv_sqrt_d; s3 *= inv_sqrt_d;

        float m_tile = s0;
        m_tile = fmaxf(m_tile, s1);
        m_tile = fmaxf(m_tile, s2);
        m_tile = fmaxf(m_tile, s3);

        float m_old = m;
        m = fmaxf(m_old, m_tile);
        alpha = __expf(m_old - m);
        beta  = __expf(m_tile - m);

        p0 = __expf(s0 - m_tile);
        p1 = __expf(s1 - m_tile);
        p2 = __expf(s2 - m_tile);
        p3 = __expf(s3 - m_tile);
        
        l = l * alpha + (p0 + p1 + p2 + p3) * beta;
      }

      alpha = __shfl(alpha, 0); beta  = __shfl(beta, 0);
      p0 = __shfl(p0, 0); p1 = __shfl(p1, 0);
      p2 = __shfl(p2, 0); p3 = __shfl(p3, 0);

      if (lane < head_dim) {
        out_i = out_i * alpha + (
            fmaf(p0, v0_i, fmaf(p1, v1_i, fmaf(p2, v2_i, p3 * v3_i)))
        ) * beta;
      }
    }

    // Tail loop for remaining K/V pairs
    for (; t < cur; ++t) {
      float k_i = __bfloat162float(sK[(size_t)t * width + lane]);
      float part = (lane < head_dim) ? (qi * k_i) : 0.0f;
      part = warp_reduce_sum(part);

      float e = 0.f, alpha = 0.f;
      if (lane == 0) {
        float s     = part * inv_sqrt_d;
        float m_new = fmaxf(m, s);
        alpha       = __expf(m - m_new);
        e           = __expf(s - m_new);
        l           = l * alpha + e;
        m           = m_new;
      }
      e     = __shfl(e, 0);
      alpha = __shfl(alpha, 0);

      if (lane < head_dim) {
        float v_i = __bfloat162float(sV[(size_t)t * width + lane]);
        out_i = fmaf(e, v_i, alpha * out_i);
      }
    }
    __syncthreads();
  }

  // Sink update & write out
  float alpha_sink = 0.f, l_final = 0.f;
  if (lane == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, 0);
  l_final    = __shfl(l_final, 0);

  if (lane < head_dim) {
    float out_norm = (alpha_sink * out_i) / (l_final + SOFTMAX_EPS);
    float * __restrict__ out_head =
        output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    out_head[lane] = out_norm;
  }
}

// ================================================================
// Kernel: packs kv_mul heads per block for K/V tile reuse
// - Optimization 1: Parallel online softmax update within the warp.
// - Optimization 2: Uses bfloat16 for K/V tiles in shared memory to halve usage.
// - Optimization 3: Increased unroll factor to 8 for higher ILP.
//
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * head_dim * sizeof(__hip_bfloat16)
// ================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized_2(
    float * __restrict__ output,              // [B, H, D]
    const float * __restrict__ q,              // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,   // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache, // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,     // [H] (bf16, layer-offset)
    const float * __restrict__ mask,              // unused
    const int * __restrict__ seq_lengths,         // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t
) {
  const int kv_h  = blockIdx.x;
  const int b     = blockIdx.y;
  const int warp  = threadIdx.y;      // 0..kv_mul-1
  const int lane  = threadIdx.x;      // 0..63
  const int kv_mul = blockDim.y;
  const int h     = kv_h * kv_mul + warp;

  if (b >= batch_size || kv_h >= n_kv_heads || warp >= kv_mul || h >= n_heads)
    return;

  const int pos = seq_lengths[b];

  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1;

  const int kv_dim    = head_dim * n_kv_heads;
  const int gqa_ratio = n_heads / n_kv_heads;
  if ((h / gqa_ratio) != kv_h) return;

  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * head_dim;

  const float * __restrict__ q_head =
      q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  // Since head_dim (64) == warpSize, the conditional is not strictly needed for the active lanes.
  const float qi = q_head[lane];

  // OPTIMIZATION 2: Use bfloat16 for shared memory tiles
  extern __shared__ __hip_bfloat16 shared_mem_bf16[];
  __hip_bfloat16 *sK = shared_mem_bf16;
  __hip_bfloat16 *sV = shared_mem_bf16 + (size_t)tile_t * head_dim;
  const int width = head_dim;

  float out_i = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int threads_total = blockDim.x * blockDim.y;
  const int tid2D         = threadIdx.y * blockDim.x + lane;

  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // Cooperative loading from global to shared memory (bfloat16)
    const int vec8 = head_dim / 8;
    const int elems_vec8 = cur * vec8;
    for (int e8 = tid2D; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;
      const int i8   = (e8 - tloc * vec8) * 8;
      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + i8;
      const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + i8;

      *reinterpret_cast<u128*>(sK + (size_t)tloc * width + i8) = gload_bf16x8(k_ptr);
      *reinterpret_cast<u128*>(sV + (size_t)tloc * width + i8) = gload_bf16x8(v_ptr);
    }
    const int tail = head_dim - vec8 * 8;
    if (tail) {
      const int elems_tail = cur * tail;
      for (int et = tid2D; et < elems_tail; et += threads_total) {
        const int tloc = et / tail;
        const int io   = (et - tloc * tail) + vec8 * 8;
        const int t_abs = t_start + base + tloc;
        const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
        sK[(size_t)tloc * width + io] = k_head_base[(size_t)tw * kv_dim + io];
        sV[(size_t)tloc * width + io] = v_head_base[(size_t)tw * kv_dim + io];
      }
    }
    __syncthreads();

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    int t = 0;

    for (; t < cur - (ATTN_UNROLL_FACTOR - 1); t += ATTN_UNROLL_FACTOR) {
      float k0_i = __bfloat162float(sK[(size_t)(t+0)*width+lane]); float v0_i = __bfloat162float(sV[(size_t)(t+0)*width+lane]);
      float k1_i = __bfloat162float(sK[(size_t)(t+1)*width+lane]); float v1_i = __bfloat162float(sV[(size_t)(t+1)*width+lane]);
      float k2_i = __bfloat162float(sK[(size_t)(t+2)*width+lane]); float v2_i = __bfloat162float(sV[(size_t)(t+2)*width+lane]);
      float k3_i = __bfloat162float(sK[(size_t)(t+3)*width+lane]); float v3_i = __bfloat162float(sV[(size_t)(t+3)*width+lane]);
      float k4_i = __bfloat162float(sK[(size_t)(t+4)*width+lane]); float v4_i = __bfloat162float(sV[(size_t)(t+4)*width+lane]);
      float k5_i = __bfloat162float(sK[(size_t)(t+5)*width+lane]); float v5_i = __bfloat162float(sV[(size_t)(t+5)*width+lane]);
      float k6_i = __bfloat162float(sK[(size_t)(t+6)*width+lane]); float v6_i = __bfloat162float(sV[(size_t)(t+6)*width+lane]);
      float k7_i = __bfloat162float(sK[(size_t)(t+7)*width+lane]); float v7_i = __bfloat162float(sV[(size_t)(t+7)*width+lane]);

      float s0 = warp_reduce_sum(qi * k0_i); float s1 = warp_reduce_sum(qi * k1_i);
      float s2 = warp_reduce_sum(qi * k2_i); float s3 = warp_reduce_sum(qi * k3_i);
      float s4 = warp_reduce_sum(qi * k4_i); float s5 = warp_reduce_sum(qi * k5_i);
      float s6 = warp_reduce_sum(qi * k6_i); float s7 = warp_reduce_sum(qi * k7_i);

      float p0, p1, p2, p3, p4, p5, p6, p7, alpha, beta;
      if (lane == 0) {
        s0 *= inv_sqrt_d; s1 *= inv_sqrt_d; s2 *= inv_sqrt_d; s3 *= inv_sqrt_d;
        s4 *= inv_sqrt_d; s5 *= inv_sqrt_d; s6 *= inv_sqrt_d; s7 *= inv_sqrt_d;

        float m_tile = s0;
        m_tile = fmaxf(m_tile, s1); m_tile = fmaxf(m_tile, s2); m_tile = fmaxf(m_tile, s3);
        m_tile = fmaxf(m_tile, s4); m_tile = fmaxf(m_tile, s5); m_tile = fmaxf(m_tile, s6);
        m_tile = fmaxf(m_tile, s7);

        float m_old = m;
        m = fmaxf(m_old, m_tile);
        alpha = __expf(m_old - m);
        beta  = __expf(m_tile - m);

        p0 = __expf(s0 - m_tile); p1 = __expf(s1 - m_tile);
        p2 = __expf(s2 - m_tile); p3 = __expf(s3 - m_tile);
        p4 = __expf(s4 - m_tile); p5 = __expf(s5 - m_tile);
        p6 = __expf(s6 - m_tile); p7 = __expf(s7 - m_tile);
        
        l = l * alpha + (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * beta;
      }

      alpha = __shfl(alpha, 0); beta  = __shfl(beta, 0);
      p0 = __shfl(p0, 0); p1 = __shfl(p1, 0); p2 = __shfl(p2, 0); p3 = __shfl(p3, 0);
      p4 = __shfl(p4, 0); p5 = __shfl(p5, 0); p6 = __shfl(p6, 0); p7 = __shfl(p7, 0);

      out_i = out_i * alpha + (
          fmaf(p0, v0_i, fmaf(p1, v1_i, fmaf(p2, v2_i, fmaf(p3, v3_i, 
          fmaf(p4, v4_i, fmaf(p5, v5_i, fmaf(p6, v6_i, p7 * v7_i)))))))
      ) * beta;
    }

    // Tail loop for remaining K/V pairs
    for (; t < cur; ++t) {
      float k_i = __bfloat162float(sK[(size_t)t * width + lane]);
      float part = qi * k_i;
      part = warp_reduce_sum(part);

      float e = 0.f, alpha = 0.f;
      if (lane == 0) {
        float s     = part * inv_sqrt_d;
        float m_new = fmaxf(m, s);
        alpha       = __expf(m - m_new);
        e           = __expf(s - m_new);
        l           = l * alpha + e;
        m           = m_new;
      }
      e     = __shfl(e, 0);
      alpha = __shfl(alpha, 0);

      float v_i = __bfloat162float(sV[(size_t)t * width + lane]);
      out_i = fmaf(e, v_i, alpha * out_i);
    }
    __syncthreads();
  }

  // Sink update & write out
  float alpha_sink = 0.f, l_final = 0.f;
  if (lane == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, 0);
  l_final    = __shfl(l_final, 0);

  if (lane < head_dim) {
    float out_norm = (alpha_sink * out_i) / (l_final + SOFTMAX_EPS);
    float * __restrict__ out_head =
        output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    out_head[lane] = out_norm;
  }
}

// ====================================================================================
// Kernel implementing Idea 1 (Loop Padding) & Idea 3 (Bank Conflict Avoidance)
//
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16)
// ====================================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized_2(
    float * __restrict__ output,              // [B, H, D]
    const float * __restrict__ q,              // [B, H, D]
    const __hip_bfloat16 * __restrict__ key_cache,   // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache, // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,     // [H] (bf16, layer-offset)
    const float * __restrict__ mask,              // unused
    const int * __restrict__ seq_lengths,       // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int tile_t
) {
  const int kv_h  = blockIdx.x;
  const int b     = blockIdx.y;
  const int warp  = threadIdx.y;    // 0..kv_mul-1
  const int lane  = threadIdx.x;    // 0..63
  const int kv_mul = blockDim.y;
  const int h     = kv_h * kv_mul + warp;

  if (b >= batch_size || kv_h >= n_kv_heads || warp >= kv_mul || h >= n_heads)
    return;

  const int pos = seq_lengths[b];

  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1;

  const int kv_dim    = head_dim * n_kv_heads;
  const int gqa_ratio = n_heads / n_kv_heads;
  if ((h / gqa_ratio) != kv_h) return;

  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * head_dim;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * head_dim;

  const float * __restrict__ q_head =
      q + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
  const float qi = q_head[lane];

  // IDEA 3: Add padding to shared memory width to avoid bank conflicts.
  // 64 (head_dim) + 8 (padding) = 72. Accessing s[t*72+lane] and s[t*72+lane+32]
  // will now fall into different memory banks.
  // NOTE: The host must launch the kernel with increased shared memory:
  // shmem = 2 * tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16)
  constexpr int PADDED_HEAD_DIM = 72;

  extern __shared__ __hip_bfloat16 shared_mem_bf16[];
  __hip_bfloat16 *sK = shared_mem_bf16;
  __hip_bfloat16 *sV = shared_mem_bf16 + (size_t)tile_t * PADDED_HEAD_DIM;
  
  float out_i = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int threads_total = blockDim.x * blockDim.y;
  const int tid2D         = threadIdx.y * blockDim.x + lane;

  for (int base = 0; base < n_steps; base += tile_t) {
    const int cur = min(tile_t, n_steps - base);

    // Cooperative loading from global to shared memory (bfloat16)
    // Since head_dim (64) is a multiple of 8, the tail-loading loop is removed.
    const int vec8 = head_dim / 8;
    const int elems_vec8 = cur * vec8;
    for (int e8 = tid2D; e8 < elems_vec8; e8 += threads_total) {
      const int tloc = e8 / vec8;
      const int i8   = (e8 - tloc * vec8) * 8;
      const int t_abs = t_start + base + tloc;
      const int tw    = ((layer_idx & 1) == 0) ? (t_abs % SW_WINDOW) : t_abs;
      const __hip_bfloat16 *k_ptr = k_head_base + (size_t)tw * kv_dim + i8;
      const __hip_bfloat16 *v_ptr = v_head_base + (size_t)tw * kv_dim + i8;

      *reinterpret_cast<u128*>(sK + (size_t)tloc * PADDED_HEAD_DIM + i8) = gload_bf16x8(k_ptr);
      *reinterpret_cast<u128*>(sV + (size_t)tloc * PADDED_HEAD_DIM + i8) = gload_bf16x8(v_ptr);
    }
    __syncthreads();

    const float inv_sqrt_d = rsqrtf((float)head_dim);

    // IDEA 1: The computation loop is now unified. The separate tail loop is removed.
    // Padding is handled inside by masking scores for out-of-bounds steps to -inf.
    for (int t = 0; t < cur; t += ATTN_UNROLL_FACTOR) {
      float k0_i = __bfloat162float(sK[(size_t)(t+0)*PADDED_HEAD_DIM+lane]); float v0_i = __bfloat162float(sV[(size_t)(t+0)*PADDED_HEAD_DIM+lane]);
      float k1_i = __bfloat162float(sK[(size_t)(t+1)*PADDED_HEAD_DIM+lane]); float v1_i = __bfloat162float(sV[(size_t)(t+1)*PADDED_HEAD_DIM+lane]);
      float k2_i = __bfloat162float(sK[(size_t)(t+2)*PADDED_HEAD_DIM+lane]); float v2_i = __bfloat162float(sV[(size_t)(t+2)*PADDED_HEAD_DIM+lane]);
      float k3_i = __bfloat162float(sK[(size_t)(t+3)*PADDED_HEAD_DIM+lane]); float v3_i = __bfloat162float(sV[(size_t)(t+3)*PADDED_HEAD_DIM+lane]);
      float k4_i = __bfloat162float(sK[(size_t)(t+4)*PADDED_HEAD_DIM+lane]); float v4_i = __bfloat162float(sV[(size_t)(t+4)*PADDED_HEAD_DIM+lane]);
      float k5_i = __bfloat162float(sK[(size_t)(t+5)*PADDED_HEAD_DIM+lane]); float v5_i = __bfloat162float(sV[(size_t)(t+5)*PADDED_HEAD_DIM+lane]);
      float k6_i = __bfloat162float(sK[(size_t)(t+6)*PADDED_HEAD_DIM+lane]); float v6_i = __bfloat162float(sV[(size_t)(t+6)*PADDED_HEAD_DIM+lane]);
      float k7_i = __bfloat162float(sK[(size_t)(t+7)*PADDED_HEAD_DIM+lane]); float v7_i = __bfloat162float(sV[(size_t)(t+7)*PADDED_HEAD_DIM+lane]);

      float s0 = warp_reduce_sum(qi * k0_i); float s1 = warp_reduce_sum(qi * k1_i);
      float s2 = warp_reduce_sum(qi * k2_i); float s3 = warp_reduce_sum(qi * k3_i);
      float s4 = warp_reduce_sum(qi * k4_i); float s5 = warp_reduce_sum(qi * k5_i);
      float s6 = warp_reduce_sum(qi * k6_i); float s7 = warp_reduce_sum(qi * k7_i);

      float p0, p1, p2, p3, p4, p5, p6, p7, alpha, beta;
      if (lane == 0) {
        // IDEA 1: Mask scores from padded steps before softmax.
        s0 = ((t + 0) < cur) ? s0 * inv_sqrt_d : -INFINITY;
        s1 = ((t + 1) < cur) ? s1 * inv_sqrt_d : -INFINITY;
        s2 = ((t + 2) < cur) ? s2 * inv_sqrt_d : -INFINITY;
        s3 = ((t + 3) < cur) ? s3 * inv_sqrt_d : -INFINITY;
        s4 = ((t + 4) < cur) ? s4 * inv_sqrt_d : -INFINITY;
        s5 = ((t + 5) < cur) ? s5 * inv_sqrt_d : -INFINITY;
        s6 = ((t + 6) < cur) ? s6 * inv_sqrt_d : -INFINITY;
        s7 = ((t + 7) < cur) ? s7 * inv_sqrt_d : -INFINITY;

        float m_tile = s0;
        m_tile = fmaxf(m_tile, s1); m_tile = fmaxf(m_tile, s2); m_tile = fmaxf(m_tile, s3);
        m_tile = fmaxf(m_tile, s4); m_tile = fmaxf(m_tile, s5); m_tile = fmaxf(m_tile, s6);
        m_tile = fmaxf(m_tile, s7);

        float m_old = m;
        m = fmaxf(m_old, m_tile);
        alpha = __expf(m_old - m);
        beta  = __expf(m_tile - m);

        p0 = __expf(s0 - m_tile); p1 = __expf(s1 - m_tile);
        p2 = __expf(s2 - m_tile); p3 = __expf(s3 - m_tile);
        p4 = __expf(s4 - m_tile); p5 = __expf(s5 - m_tile);
        p6 = __expf(s6 - m_tile); p7 = __expf(s7 - m_tile);
        
        l = l * alpha + (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * beta;
      }

      alpha = __shfl(alpha, 0); beta  = __shfl(beta, 0);
      p0 = __shfl(p0, 0); p1 = __shfl(p1, 0); p2 = __shfl(p2, 0); p3 = __shfl(p3, 0);
      p4 = __shfl(p4, 0); p5 = __shfl(p5, 0); p6 = __shfl(p6, 0); p7 = __shfl(p7, 0);

      out_i = out_i * alpha + (
            fmaf(p0, v0_i, fmaf(p1, v1_i, fmaf(p2, v2_i, fmaf(p3, v3_i, 
            fmaf(p4, v4_i, fmaf(p5, v5_i, fmaf(p6, v6_i, p7 * v7_i)))))))
      ) * beta;
    }
    __syncthreads();
  }

  // Sink update & write out
  float alpha_sink = 0.f, l_final = 0.f;
  if (lane == 0) {
    float s_sink = __bfloat162float(sinks[h]);
    float m_new  = fmaxf(m, s_sink);
    float alpha  = __expf(m - m_new);
    float e      = __expf(s_sink - m_new);
    l            = l * alpha + e;
    alpha_sink   = alpha;
    l_final      = l;
  }
  alpha_sink = __shfl(alpha_sink, 0);
  l_final    = __shfl(l_final, 0);

  if (lane < head_dim) {
    float out_norm = (alpha_sink * out_i) / (l_final + 1e-6f);
    float * __restrict__ out_head =
        output + (size_t)b * n_heads * head_dim + (size_t)h * head_dim;
    out_head[lane] = out_norm;
  }
}

// ================================================================
// Fused Attention (decode) with Matrix Cores (MFMA, gfx90a+)
// - Processes up to 16 Q rows (GQA heads) × 16 time steps per tile
// - Uses MFMA 16x16x16 bf16->f32 for: S = Q_tile * K_tile^T  and  O += P_tile * V_tile
// - Online (streaming) softmax across time tiles
// - Works with bf16 K/V caches; Q is fp32 scaled by 1/sqrt(d)
// - Grid:  (x = n_kv_heads, y = batch_size)
// - Block: (x = 64 lanes, y = 1 wave)
// Shared memory requirement (bytes):
//   2*(16*16)*4 + (48*head_dim + 256)*2 + 64*4  =  2304 + 2*(48*D+256) + 256
//   For D=64: ~8960 bytes
// ================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <float.h>

#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif
#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-6f
#endif

// --------- Small vector aliases (match matmul.hpp style) ---------
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4))); // 4×i16 bit-cast

// --------- Lane helpers (1 wave = 64 lanes) ---------
__device__ inline int lane_row(int lane)   { return lane & 15; }   // 0..15 (column inside 16x16 tile)
__device__ inline int lane_group(int lane) { return lane >> 4; }   // 0..3  (which 4-row group)

// --------- bf16 bit helpers ---------
__device__ inline uint16_t hipbf16_to_bits(__hip_bfloat16 x) {
  return *reinterpret_cast<uint16_t*>(&x);
}
__device__ inline uint16_t f32_to_bf16_bits(float x) {
  __hip_bfloat16 t = __float2bfloat16(x);
  return *reinterpret_cast<uint16_t*>(&t);
}

// --------- MFMA wrapper (gfx90a bf16->f32 accumulate) ---------
__device__ inline f32x4 mfma_16x16x16_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x4 c_vec) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
}

// --------- Build A/B k-slices for the MFMA micro-tile ---------
template<int WM>
__device__ inline bf16x4 make_a_vec_k(const uint16_t* __restrict__ sA,
                                      int ldA, int aRowBase, int kOff, int lane)
{
  const int r   = aRowBase + (lane & 15); // we map lane_row to "k index selector" in 16
  const int grp = lane_group(lane);       // choose which of 4 rows this lane writes
  bf16x4 v;
  #pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = sA[(grp * 4 + i) * ldA + (kOff + r)];
  return v;
}

template<int WN>
__device__ inline bf16x4 make_b_vec_k(const uint16_t* __restrict__ sB,
                                      int ldB, int bColBase, int kOff, int lane)
{
  const int col = bColBase + lane_row(lane); // 0..15 which column of the 16x16 tile
  const int grp = lane_group(lane);          // selects 4 contiguous k's
  bf16x4 v;
  #pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = sB[col * ldB + (kOff + grp * 4 + i)];
  return v;
}

// --------- Store an MFMA 16x16 f32x4 register tile into float smem (row-major) ---------
__device__ inline void store_reg_tile_to_smem(const f32x4& acc,
                                              float* __restrict__ sTile,
                                              int ld, int lane,
                                              int valid_cols, int valid_rows)
{
  const int col     = lane_row(lane);         // column 0..15
  const int rowBase = lane_group(lane) * 4;   // rows [rowBase..rowBase+3]

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = rowBase + i;
    if (r < valid_rows && col < valid_cols) {
      sTile[r * ld + col] = acc[i];
    } else {
      // Pad to -inf so masked steps don't contribute in softmax
      if (r < 16 && col < 16) sTile[r * ld + col] = -INFINITY;
    }
  }
}

// ================================================
// Fused Attention (decode) with Matrix Cores (MFMA)
// gfx90a+, bf16 K/V, fp32 Q/out, streaming softmax
// ================================================
extern "C" __global__ __launch_bounds__(64, 2)
void fused_attention_kernel_mfma_complete(
    float * __restrict__ output,               // [B, H, D]
    const float * __restrict__ q,              // [B, H, D] fp32
    const __hip_bfloat16 * __restrict__ key_cache,   // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache, // [B, L, T, KV] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,       // [H] (bf16)
    const float * __restrict__ /*mask*/,             // unused (causal)
    const int * __restrict__ seq_lengths,            // [B]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window,
    size_t batch_kv_stride, size_t layer_kv_offset,
    int /*tile_t_ignored*/)
{
  constexpr int TILE_T = 16;
  const int b    = blockIdx.y;
  const int kv_h = blockIdx.x;
  const int lane = threadIdx.x; // 0..63
  if (b >= batch_size || kv_h >= n_kv_heads) return;

  const int gqa_ratio = n_heads / n_kv_heads; // e.g., 8
  const int h0        = kv_h * gqa_ratio;
  const int rows_this = gqa_ratio > 16 ? 16 : gqa_ratio;

  const int pos = seq_lengths[b];
  int t_start = 0;
  if (use_sliding_window && ((layer_idx & 1) == 0)) {
    t_start = max(0, pos - (SW_WINDOW - 1));
  }
  const int n_steps = pos - t_start + 1;

  const int D       = head_dim;
  const int kv_dim  = D * n_kv_heads;

  const __hip_bfloat16 * __restrict__ k_base_layer =
      key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
  const __hip_bfloat16 * __restrict__ v_base_layer =
      value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;

  const __hip_bfloat16 * __restrict__ k_head_base = k_base_layer + (size_t)kv_h * D;
  const __hip_bfloat16 * __restrict__ v_head_base = v_base_layer + (size_t)kv_h * D;

  const float inv_sqrt_d = rsqrtf((float)D);

  // ---- shared memory layout ----
  extern __shared__ uint8_t smem_raw[];
  uint16_t* sQ_u16   = reinterpret_cast<uint16_t*>(smem_raw);                     // [16, D]
  uint16_t* sKB_u16  = sQ_u16  + (size_t)16 * D;                                  // [16, D]
  uint16_t* sVB_u16  = sKB_u16 + (size_t)16 * D;                                  // [D, 16]
  float*    sS_f32   = reinterpret_cast<float*>(sVB_u16 + (size_t)D * 16);        // [16,16]
  float*    sP_f32   = sS_f32  + 16 * 16;                                         // [16,16]
  uint16_t* sP_u16   = reinterpret_cast<uint16_t*>(sP_f32 + 16 * 16);             // [16,16]
  float*    m_row    = reinterpret_cast<float*>(sP_u16 + 16 * 16);                // [16]
  float*    l_row    = m_row + 16;                                                // [16]
  float*    alpha_rw = l_row + 16;                                                // [16]
  float*    beta_rw  = alpha_rw + 16;                                             // [16]
  float*    alpha_sink = beta_rw + 16;                                            // [16]

  if (lane < 16) {
    m_row[lane]     = -INFINITY;
    l_row[lane]     = 0.f;
    alpha_rw[lane]  = 1.f;
    beta_rw[lane]   = 0.f;
    alpha_sink[lane]= 1.f;
  }
  __syncthreads();

  // ---- Q tile (16 rows × D) -> bf16 in LDS ----
  if (lane < D) {
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
      float qv = 0.f;
      if (r < rows_this) {
        const int h = h0 + r;
        if (h < n_heads) {
          const float* __restrict__ q_head = q + (size_t)b * n_heads * D + (size_t)h * D;
          qv = q_head[lane] * inv_sqrt_d;
        }
      }
      sQ_u16[(size_t)r * D + lane] = f32_to_bf16_bits(qv);
    }
  }
  __syncthreads();

  // ---- per-column accumulators for O (as plain scalars) ----
  float o_acc0[4] = {0.f,0.f,0.f,0.f};
  float o_acc1[4] = {0.f,0.f,0.f,0.f};
  float o_acc2[4] = {0.f,0.f,0.f,0.f};
  float o_acc3[4] = {0.f,0.f,0.f,0.f};

  for (int base = 0; base < n_steps; base += TILE_T) {
    const int cur = min(TILE_T, n_steps - base);

    // ---- load K/V tiles into LDS ----
    #pragma unroll
    for (int t = 0; t < TILE_T; ++t) {
      const bool valid_t = (t < cur);
      int t_abs = t_start + base + t;
      if (use_sliding_window && ((layer_idx & 1) == 0)) t_abs = t_abs % SW_WINDOW;
      if (lane < D) {
        uint16_t k_bits = 0u, v_bits = 0u;
        if (valid_t) {
          const __hip_bfloat16 *k_ptr = k_head_base + (size_t)t_abs * kv_dim + lane;
          const __hip_bfloat16 *v_ptr = v_head_base + (size_t)t_abs * kv_dim + lane;
          k_bits = hipbf16_to_bits(*k_ptr);
          v_bits = hipbf16_to_bits(*v_ptr);
        }
        sKB_u16[(size_t)t * D + lane]      = k_bits;              // K: [cur,D], ld=D
        sVB_u16[(size_t)lane * TILE_T + t] = v_bits;              // V: [D,cur], ld=16
      }
    }
    __syncthreads();

    // ---- S = Q(16×D) * K^T(D×cur) via MFMA ----
    f32x4 accS = {0.f,0.f,0.f,0.f};
    #pragma unroll
    for (int kk = 0; kk < 1024; kk += 16) {
      if (kk >= D) break;
      const bf16x4 avec = make_a_vec_k<16>(sQ_u16,  D, 0, kk, lane);
      const bf16x4 bvec = make_b_vec_k<16>(sKB_u16, D, 0, kk, lane);
      accS = mfma_16x16x16_bf16(avec, bvec, accS);
    }
    store_reg_tile_to_smem(accS, sS_f32, 16, lane, cur, rows_this);
    __syncthreads();

    // ---- streaming softmax over time for 16 rows ----
    if ((lane & 15) == 0) {
      const int rowBase = lane_group(lane) * 4;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = rowBase + i;
        if (r >= rows_this) continue;

        float mt = -INFINITY;
        #pragma unroll
        for (int c = 0; c < TILE_T; ++c) if (c < cur) mt = fmaxf(mt, sS_f32[r * 16 + c]);

        float l_tile = 0.f;
        #pragma unroll
        for (int c = 0; c < TILE_T; ++c) {
          float p = (c < cur) ? __expf(sS_f32[r * 16 + c] - mt) : 0.f;
          sP_f32[r * 16 + c] = p;
          l_tile += p;
        }

        const float m_old = m_row[r];
        const float m_new = fmaxf(m_old, mt);
        const float a     = __expf(m_old - m_new);
        const float b2    = __expf(mt    - m_new);
        m_row[r] = m_new;
        l_row[r] = l_row[r] * a + l_tile * b2;
        alpha_rw[r] = a;
        beta_rw[r]  = b2;
      }
    }
    __syncthreads();

    // ---- cast P' to bf16 for MFMA ----
    for (int idx = lane; idx < 16 * 16; idx += 64) sP_u16[idx] = f32_to_bf16_bits(sP_f32[idx]);
    __syncthreads();

    // ---- O += P'(16×cur) * V(cur×D) via MFMA; merge with (alpha,beta) ----
    #pragma unroll
    for (int n0 = 0; n0 < 1024; n0 += 16) {
      if (n0 >= D) break;

      f32x4 accO = {0.f,0.f,0.f,0.f};
      const bf16x4 avecP = make_a_vec_k<16>(sP_u16, 16, 0, 0, lane);
      const bf16x4 bvecV = make_b_vec_k<16>(sVB_u16, 16, n0, 0, lane);
      accO = mfma_16x16x16_bf16(avecP, bvecV, accO);

      const int rowBase = lane_group(lane) * 4;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = rowBase + i;
        float a = 1.f, b2 = 0.f;
        if (r < rows_this) { a = alpha_rw[r]; b2 = beta_rw[r]; }

        float* dstp = nullptr;
        if      (n0 == 0)  dstp = &o_acc0[i];
        else if (n0 == 16) dstp = &o_acc1[i];
        else if (n0 == 32) dstp = &o_acc2[i];
        else               dstp = &o_acc3[i];

        *dstp = a * (*dstp) + b2 * accO[i];
      }
    }
    __syncthreads();
  }

  // ---- sink merge ----
  if ((lane & 15) == 0) {
    const int rowBase = lane_group(lane) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = rowBase + i;
      if (r >= rows_this) continue;
      const int h = h0 + r;
      float s_sink = __bfloat162float(sinks[h]);
      const float m_old = m_row[r];
      const float m_new = fmaxf(m_old, s_sink);
      const float a     = __expf(m_old - m_new);
      const float e     = __expf(s_sink - m_new);
      l_row[r]      = l_row[r] * a + e;
      alpha_sink[r] = a;
      m_row[r]      = m_new;
    }
  }
  __syncthreads();

  // ---- write out normalized O ----
  #pragma unroll
  for (int n0 = 0; n0 < 1024; n0 += 16) {
    if (n0 >= D) break;
    const int col = n0 + lane_row(lane);
    const int rowBase = lane_group(lane) * 4;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = rowBase + i;
      if (r >= rows_this || col >= D) continue;

      const float aS   = alpha_sink[r];
      const float denom= l_row[r] + SOFTMAX_EPS;
      float val = (n0 == 0  ? o_acc0[i]
                 : (n0 == 16 ? o_acc1[i]
                 : (n0 == 32 ? o_acc2[i]
                             : o_acc3[i])));
      val = (aS * val) / denom;

      const int h = h0 + r;
      float * __restrict__ out_head =
          output + (size_t)b * n_heads * D + (size_t)h * D;
      out_head[col] = val;
    }
  }
}



// ==============================
// Fused Attention (decode) — HIP (wave64) optimized (bf16 in LDS)
// - LDS width padding: width = head_dim + 1 (64->65) to remove bank conflicts
// - Keep K/V as bf16 in LDS (convert-on-read) → halve LDS traffic & shmem
// - Vectorized global loads: bf16x8 (16B) with scalar tail
// - Hoist q scale once: q_scaled = qi * 0.125f  (1/sqrt(64))
// - All reductions via wave shuffles
// - tile_t auto-capped to keep shmem ≲ 64KB/block (bf16 LDS)
// ==============================

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <float.h>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif
#ifndef ATTN_UNROLL_FACTOR
#define ATTN_UNROLL_FACTOR 4
#endif
#ifndef SOFTMAX_EPS
#define SOFTMAX_EPS 1e-6f
#endif

__device__ inline float warp_reduce_sum_f32(float v) {
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}




// === KV cache update kernel (write __hip_bfloat16) ===
__global__ void update_kv_cache_kernel(__hip_bfloat16 *key_cache, __hip_bfloat16 *value_cache,
                                       const float *k, const float *v,
                                       const int *seq_lengths, int batch_size,
                                       int n_layers, int layer_idx, int seq_len,
                                       int kv_dim, size_t batch_kv_stride, size_t layer_kv_offset)
{
    size_t batch_idx = blockIdx.x;
    size_t dim_idx   = 1LL * blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= (size_t)batch_size || dim_idx >= (size_t)kv_dim)
        return;

    int pos = seq_lengths[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

    const size_t base = (size_t)batch_idx * batch_kv_stride + layer_kv_offset;
    const int row = ((layer_idx & 1) ? pos : (pos % SW_WINDOW));
    const size_t cache_idx = base + (size_t)row * (size_t)kv_dim + (size_t)dim_idx;

    key_cache[cache_idx]   = __float2bfloat16(k[1LL*batch_idx * kv_dim + dim_idx]);
    value_cache[cache_idx] = __float2bfloat16(v[1LL*batch_idx * kv_dim + dim_idx]);
}