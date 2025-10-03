#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
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

#ifndef KV_BF16_KEEP_TOKENS
#define KV_BF16_KEEP_TOKENS 16
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
#include <hip/hip_fp16.h>
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

// ====================================================================================
// Kernel implementing Idea 1 (Loop Padding) & Idea 3 (Bank Conflict Avoidance)
//
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16)
// ====================================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized(
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
    // NOTE: Must check bounds to avoid reading uninitialized shared memory
    for (int t = 0; t < cur; t += ATTN_UNROLL_FACTOR) {
      // Safe loads with bounds checking - use 0 for out-of-bounds elements
      float k0_i = (t+0 < cur) ? __bfloat162float(sK[(size_t)(t+0)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k1_i = (t+1 < cur) ? __bfloat162float(sK[(size_t)(t+1)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k2_i = (t+2 < cur) ? __bfloat162float(sK[(size_t)(t+2)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k3_i = (t+3 < cur) ? __bfloat162float(sK[(size_t)(t+3)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k4_i = (t+4 < cur) ? __bfloat162float(sK[(size_t)(t+4)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k5_i = (t+5 < cur) ? __bfloat162float(sK[(size_t)(t+5)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k6_i = (t+6 < cur) ? __bfloat162float(sK[(size_t)(t+6)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float k7_i = (t+7 < cur) ? __bfloat162float(sK[(size_t)(t+7)*PADDED_HEAD_DIM+lane]) : 0.0f;

      float v0_i = (t+0 < cur) ? __bfloat162float(sV[(size_t)(t+0)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v1_i = (t+1 < cur) ? __bfloat162float(sV[(size_t)(t+1)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v2_i = (t+2 < cur) ? __bfloat162float(sV[(size_t)(t+2)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v3_i = (t+3 < cur) ? __bfloat162float(sV[(size_t)(t+3)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v4_i = (t+4 < cur) ? __bfloat162float(sV[(size_t)(t+4)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v5_i = (t+5 < cur) ? __bfloat162float(sV[(size_t)(t+5)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v6_i = (t+6 < cur) ? __bfloat162float(sV[(size_t)(t+6)*PADDED_HEAD_DIM+lane]) : 0.0f;
      float v7_i = (t+7 < cur) ? __bfloat162float(sV[(size_t)(t+7)*PADDED_HEAD_DIM+lane]) : 0.0f;

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





// === KV cache quantization kernel (write int8 with per-row scales) ===
template <bool kUseBf16Cache>
__device__ inline void quantize_kv_cache_kernel_impl(
    int8_t *key_cache, int8_t *value_cache,
    __half *key_scales, __half *value_scales,
    __hip_bfloat16 *key_cache_recent_bf16,
    __hip_bfloat16 *value_cache_recent_bf16,
    int *recent_positions,
    const __hip_bfloat16 *k, const __hip_bfloat16 *v,
    const int *seq_lengths, int batch_size,
    int n_layers, int layer_idx, int seq_len,
    int kv_dim,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset)
{
    const size_t batch_idx = blockIdx.x;
    if (batch_idx >= (size_t)batch_size)
        return;

    const int pos = seq_lengths[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

#if KV_BF16_KEEP_TOKENS > 0
    const bool bf16_case = ((layer_idx & 1) == 1) && (pos < KV_BF16_KEEP_TOKENS);
    if (kUseBf16Cache)
    {
        if (!bf16_case || !key_cache_recent_bf16 || !value_cache_recent_bf16 || !recent_positions)
            return;
    }
    else
    {
        if (bf16_case && key_cache_recent_bf16 && value_cache_recent_bf16 && recent_positions)
            return;
    }
#else
    if (kUseBf16Cache)
        return;
#endif

    const size_t base_elem = (size_t)batch_idx * batch_kv_stride + layer_kv_offset;
    const size_t base_scale = (size_t)batch_idx * batch_scale_stride + layer_scale_offset;
    const int row = ((layer_idx & 1) ? pos : (pos % SW_WINDOW));
    const size_t cache_offset = base_elem + (size_t)row * (size_t)kv_dim;
    const size_t scale_offset = base_scale + (size_t)row;

    float local_max_k = 0.f;
    float local_max_v = 0.f;

    for (int dim = threadIdx.x; dim < kv_dim; dim += blockDim.x)
    {
        const size_t idx = (size_t)batch_idx * (size_t)kv_dim + (size_t)dim;
        const float k_val = fabsf(__bfloat162float(k[idx]));
        const float v_val = fabsf(__bfloat162float(v[idx]));
        local_max_k = fmaxf(local_max_k, k_val);
        local_max_v = fmaxf(local_max_v, v_val);
    }

    __shared__ float shared_max_k[THREADS_PER_BLOCK];
    __shared__ float shared_max_v[THREADS_PER_BLOCK];

    shared_max_k[threadIdx.x] = local_max_k;
    shared_max_v[threadIdx.x] = local_max_v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_max_k[threadIdx.x] = fmaxf(shared_max_k[threadIdx.x], shared_max_k[threadIdx.x + stride]);
            shared_max_v[threadIdx.x] = fmaxf(shared_max_v[threadIdx.x], shared_max_v[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float max_k = shared_max_k[0];
    const float max_v = shared_max_v[0];
    const float safe_k = fmaxf(max_k, 1e-6f);
    const float safe_v = fmaxf(max_v, 1e-6f);
    const float inv_scale_k = 127.f / safe_k;
    const float inv_scale_v = 127.f / safe_v;
    const float scale_k = safe_k / 127.f;
    const float scale_v = safe_v / 127.f;

    if (threadIdx.x == 0)
    {
        key_scales[scale_offset] = __float2half(scale_k);
        value_scales[scale_offset] = __float2half(scale_v);
    }
    __syncthreads();

    for (int dim = threadIdx.x; dim < kv_dim; dim += blockDim.x)
    {
        const size_t idx = (size_t)batch_idx * (size_t)kv_dim + (size_t)dim;
        const float k_val = __bfloat162float(k[idx]) * inv_scale_k;
        const float v_val = __bfloat162float(v[idx]) * inv_scale_v;

        int qk = __float2int_rn(k_val);
        int qv = __float2int_rn(v_val);

        if (qk > 127) qk = 127;
        if (qk < -127) qk = -127;
        if (qv > 127) qv = 127;
        if (qv < -127) qv = -127;

        key_cache[cache_offset + dim] = static_cast<int8_t>(qk);
        value_cache[cache_offset + dim] = static_cast<int8_t>(qv);
    }

#if KV_BF16_KEEP_TOKENS > 0
    if (kUseBf16Cache)
    {
        const size_t slots = (size_t)KV_BF16_KEEP_TOKENS;
        const size_t token_slot = (size_t)pos;
        const size_t recent_pos_base = ((size_t)batch_idx * (size_t)n_layers + (size_t)layer_idx) * slots;
        const size_t recent_elem_base = recent_pos_base * (size_t)kv_dim;

        if (threadIdx.x == 0)
        {
            recent_positions[recent_pos_base + token_slot] = pos;
        }
        __syncthreads();

        for (int dim = threadIdx.x; dim < kv_dim; dim += blockDim.x)
        {
            const size_t src_idx = (size_t)batch_idx * (size_t)kv_dim + (size_t)dim;
            const size_t dst = recent_elem_base + token_slot * (size_t)kv_dim + (size_t)dim;
            key_cache_recent_bf16[dst] = k[src_idx];
            value_cache_recent_bf16[dst] = v[src_idx];
        }
    }
#else
    (void)kUseBf16Cache;
    (void)key_cache_recent_bf16;
    (void)value_cache_recent_bf16;
    (void)recent_positions;
    (void)n_layers;
#endif
}

__global__ void quantize_kv_cache_kernel_int8(
    int8_t *key_cache, int8_t *value_cache,
    __half *key_scales, __half *value_scales,
    __hip_bfloat16 *key_cache_recent_bf16,
    __hip_bfloat16 *value_cache_recent_bf16,
    int *recent_positions,
    const __hip_bfloat16 *k, const __hip_bfloat16 *v,
    const int *seq_lengths, int batch_size,
    int n_layers, int layer_idx, int seq_len,
    int kv_dim,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset)
{
    quantize_kv_cache_kernel_impl<false>(
        key_cache, value_cache,
        key_scales, value_scales,
        key_cache_recent_bf16, value_cache_recent_bf16,
        recent_positions,
        k, v, seq_lengths, batch_size,
        n_layers, layer_idx, seq_len,
        kv_dim,
        batch_kv_stride, layer_kv_offset,
        batch_scale_stride, layer_scale_offset);
}

__global__ void quantize_kv_cache_kernel(
    int8_t *key_cache, int8_t *value_cache,
    __half *key_scales, __half *value_scales,
    __hip_bfloat16 *key_cache_recent_bf16,
    __hip_bfloat16 *value_cache_recent_bf16,
    int *recent_positions,
    const __hip_bfloat16 *k, const __hip_bfloat16 *v,
    const int *seq_lengths, int batch_size,
    int n_layers, int layer_idx, int seq_len,
    int kv_dim,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset)
{
    quantize_kv_cache_kernel_impl<false>(
        key_cache, value_cache,
        key_scales, value_scales,
        key_cache_recent_bf16, value_cache_recent_bf16,
        recent_positions,
        k, v, seq_lengths, batch_size,
        n_layers, layer_idx, seq_len,
        kv_dim,
        batch_kv_stride, layer_kv_offset,
        batch_scale_stride, layer_scale_offset);
}

#if KV_BF16_KEEP_TOKENS > 0
__global__ void quantize_kv_cache_kernel_bf16(
    int8_t *key_cache, int8_t *value_cache,
    __half *key_scales, __half *value_scales,
    __hip_bfloat16 *key_cache_recent_bf16,
    __hip_bfloat16 *value_cache_recent_bf16,
    int *recent_positions,
    const __hip_bfloat16 *k, const __hip_bfloat16 *v,
    const int *seq_lengths, int batch_size,
    int n_layers, int layer_idx, int seq_len,
    int kv_dim,
    size_t batch_kv_stride, size_t layer_kv_offset,
    size_t batch_scale_stride, size_t layer_scale_offset)
{
    quantize_kv_cache_kernel_impl<true>(
        key_cache, value_cache,
        key_scales, value_scales,
        key_cache_recent_bf16, value_cache_recent_bf16,
        recent_positions,
        k, v, seq_lengths, batch_size,
        n_layers, layer_idx, seq_len,
        kv_dim,
        batch_kv_stride, layer_kv_offset,
        batch_scale_stride, layer_scale_offset);
}
#endif



// === KV cache update kernel (write __hip_bfloat16) ===
__global__ void update_kv_cache_kernel(__hip_bfloat16 *key_cache, __hip_bfloat16 *value_cache,
                                       const __hip_bfloat16 *k, const __hip_bfloat16 *v,
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

    key_cache[cache_idx]   = k[1LL*batch_idx * kv_dim + dim_idx];
    value_cache[cache_idx] = v[1LL*batch_idx * kv_dim + dim_idx];
}