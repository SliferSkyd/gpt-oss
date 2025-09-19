#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "matmul.hpp"

// --- MODIFIED FUSED ATTENTION KERNEL (2-D mapping: lanes x warps) ---
#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif

#define ATTN_UNROLL_FACTOR 8

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
__device__ inline float blockReduceMax(float v, float *smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1); // warp id
    v = warpReduceMax(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = smem[lane];
    __syncthreads();
    out = warpReduceMax(out);
    if (lane == 0 && wid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}
__device__ inline float blockReduceSum(float v, float *smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1);
    v = warpReduceSum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = smem[lane];
    __syncthreads();
    out = warpReduceSum(out);
    if (lane == 0 && wid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

__global__ void fused_attention_kernel(
    float * __restrict__ output,                    // [batch, n_heads, head_dim]
    const float * __restrict__ q,                   // [batch, n_heads * head_dim]
    const __hip_bfloat16 * __restrict__ key_cache,  // [batch, n_layers, seq_len, kv_dim] (bf16)
    const __hip_bfloat16 * __restrict__ value_cache,// [batch, n_layers, seq_len, kv_dim] (bf16)
    const __hip_bfloat16 * __restrict__ sinks,      // [n_heads] (already layer-offset on host)
    const float * __restrict__ mask,                // [seq_len, seq_len]
    const int * __restrict__ positions,             // [batch]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window, size_t batch_kv_stride, size_t layer_kv_offset)
{
    extern __shared__ float s_sh[];  // layout later

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    if (b >= batch_size || h >= n_heads) return;

    // Wave bookkeeping
    const int WARP  = warpSize;                            // 64 on AMD, 32 on NV
    const int WARPS = (blockDim.x + WARP - 1) / WARP;
    const int lane  = tid & (WARP - 1);
    const int wid   = tid >> (__ffs(WARP) - 1);

    // Pointers / shapes
    const int pos        = positions[b];
    const int gqa_ratio  = n_heads / n_kv_heads;
    const int kv_h       = h / gqa_ratio;
    const int kv_dim     = head_dim * n_kv_heads;

    const float *q_head = q + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;
    const __hip_bfloat16 *k_base = key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
    const __hip_bfloat16 *v_base = value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;
    float *out_head =
        output + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;

    // Windowing + capacities
    const bool apply_window = use_sliding_window && ((layer_idx & 1) == 0);
    const int  win_start    = apply_window ? max(0, pos - (SW_WINDOW - 1)) : 0;
    const int  win_core_len = pos - win_start + 1;                  // tokens before sink
    const int  att_cap      = apply_window ? (SW_WINDOW + 1) : seq_len; // <= host reserve
    float *s_att      = s_sh;                                       // [att_cap]
    float *s_partials = s_att + att_cap;                            // [WARPS * head_dim]
    float *s_reduce   = s_partials + WARPS * head_dim;              // [WARPS]
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    // Fast path: one lane per head-dim element
    const float q_lane = (lane < head_dim) ? q_head[lane] : 0.0f;

    // ---- Pass 1: compute attention scores into s_att[0..win_core_len-1]
    for (int w = wid; w < win_core_len; w += WARPS) {
        const int t = win_start + w;
        const __hip_bfloat16 *k_vec = k_base + 1LL * ((layer_idx & 1) ? t : t % SW_WINDOW) * kv_dim
                                      + 1LL * kv_h * head_dim;

        float prod = 0.f;
        if (lane < head_dim) {
            const float kf = __bfloat162float(k_vec[lane]);
            prod = q_lane * kf;
        }
        float dot = warpReduceSum(prod);  // 64-lane sum
        if (lane == 0) {
            float acc = (double)dot * inv_sqrt_d;
            if (apply_window) {
                acc += mask[1LL * pos * seq_len + t];
            }
            s_att[w] = acc;
        }
    }
    __syncthreads();

    // Append sink (if any) after the core window
    int softmax_len = win_core_len;
    if (pos + 1 < seq_len) {
        if (tid == 0) s_att[softmax_len] = __bfloat162float(sinks[h]);
        softmax_len += 1;
    }
    __syncthreads();

    // ---- Softmax over s_att[0..softmax_len-1]
    float tmax = -INFINITY;
    for (int i = tid; i < softmax_len; i += blockDim.x) tmax = fmaxf(tmax, s_att[i]);
    const float max_val = blockReduceMax(tmax, s_reduce);

    float tsum = 0.f;
    for (int i = tid; i < softmax_len; i += blockDim.x) {
        float v = expf(s_att[i] - max_val);
        s_att[i] = v;
        tsum += v;
    }
    const float sum_val = blockReduceSum(tsum, s_reduce);
    const float inv_sum = 1.f / (sum_val + 1e-9f);

    for (int i = tid; i < softmax_len; i += blockDim.x) s_att[i] *= inv_sum;
    __syncthreads();

    // ---- Pass 2: V-weighted sum over the core window (exclude optional sink)
    if (lane < head_dim) {
        float partial = 0.f;
        for (int w = wid; w < win_core_len; w += WARPS) {
            const int t = win_start + w;
            const __hip_bfloat16 *v_vec = v_base + 1LL * ((layer_idx & 1) ? t : t % SW_WINDOW) * kv_dim
                                          + 1LL * kv_h * head_dim;
            const float vf = __bfloat162float(v_vec[lane]);
            partial += (double)s_att[w] * vf;
        }
        s_partials[wid * head_dim + lane] = partial;
    }
    __syncthreads();

    // Cross-warp reduce the partial output vectors (one lane per dim)
    if (wid == 0 && lane < head_dim) {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += s_partials[w * head_dim + lane];
        out_head[lane] = acc;
    }
}


// === KV cache update kernel (write __hip_bfloat16) ===
__global__ void update_kv_cache_kernel(__hip_bfloat16 *key_cache, __hip_bfloat16 *value_cache,
                                       const float *k, const float *v,
                                       const int *positions, int batch_size,
                                       int n_layers, int layer_idx, int seq_len,
                                       int kv_dim, size_t batch_kv_stride, size_t layer_kv_offset)
{
    size_t batch_idx = blockIdx.x;
    size_t dim_idx = 1LL * blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= (size_t)batch_size || dim_idx >= (size_t)kv_dim)
        return;

    int pos = positions[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

    const size_t base = (size_t)batch_idx * batch_kv_stride + layer_kv_offset;
    const int row = ((layer_idx & 1) ? pos : (pos % SW_WINDOW));
    const size_t cache_idx = base + (size_t)row * (size_t)kv_dim + (size_t)dim_idx;

    key_cache[cache_idx]   = __float2bfloat16(k[1LL*batch_idx * kv_dim + dim_idx]);
    value_cache[cache_idx] = __float2bfloat16(v[1LL*batch_idx * kv_dim + dim_idx]);
}


// Stores the accumulator tile back to global memory and performs the fusion steps.
template<bool Interior>
__device__ inline void store_and_fuse_tile(
    float* __restrict__ x, // In/Out buffer
    const __hip_bfloat16* __restrict__ bias,
    const f32x4& acc,
    int M, int N,
    int m0, int n0,
    int wave_m, int wave_n,
    int lane)
{
    const int rowBase = m0 + wave_m * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n * WN + lane_row(lane);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if constexpr (!Interior) {
            if (row >= M || col >= N) continue;
        }

        // Fused operations: add bias and residual
        float Cvalue = acc[i];
        Cvalue += __bfloat162float(bias[col]);
        Cvalue += x[(size_t)row * N + col];
        x[(size_t)row * N + col] = Cvalue;
    }
}



__global__ __launch_bounds__(64 * WAVES_PER_BLOCK, 2)
void fused_output_projection_kernel_optimized(
    float* __restrict__ C,                         // [M,N] in/out (residual + out)
    const float* __restrict__ A,               // [M,K] fp32
    const __hip_bfloat16* __restrict__ Wbf16,     // [N,K] bf16 row-major
    const __hip_bfloat16* __restrict__ bias,                // [N]  fp32
    int M, int K, int N)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x;               // 0..63
    const int wave   = threadIdx.y;               // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // LDS ping–pong:
    // sA0,sA1: [BLOCK_M x (BLOCK_K+pad)] row-major (bf16 bits)
    // sB0,sB1: [(BLOCK_K+pad) x BLOCK_N] col-major (bf16 bits)
    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K + PAD_K_MC;          // leading dim in bf16 elems
    const int ldB = BLOCK_K + PAD_K_MC;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m0 + BLOCK_M;

    // Alignment guards (decide paths once per block)
    const bool alignedA  = (((uintptr_t)A     & 0x7)==0) && ((K & 1)==0); // 8B & even
    const bool use128bW  = (((uintptr_t)Wbf16 & 0xF)==0) && ((K & 7)==0); // 16B & K%8==0

    // Preload kBase = 0
    if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);

    if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);
    else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    // Ping–pong pointers
    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    // Per-wave MFMA bases
    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    // Split K into main slabs of BLOCK_K and one possible tail
    const int Kmain = (K / BLOCK_K) * BLOCK_K;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BLOCK_K) {
        const int kNext = k0 + BLOCK_K;

        // Prefetch next main slab
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);

            if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
            else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        // Consume current tiles
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }

        __syncthreads();
        if (kNext < Kmain) {
            // swap
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    // Tail slab (0 < K - Kmain < BLOCK_K)
    // Tail slab (0 < K - Kmain < BLOCK_K)
    if (has_tail) {
        // load the tail into next*
        if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);
        else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

        // <<< fix: consume the tail you just loaded
        currA = nextA;
        currB = nextB;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }


    // Stores
    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) store_and_fuse_tile<true >(C, bias, acc, M, N, m0, n0, wave_m, wave_n, lane);
    else          store_and_fuse_tile<false>(C, bias, acc, M, N, m0, n0, wave_m, wave_n, lane);
}



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


// ====================================================================================
// Kernel implementing Idea 1 (Loop Padding) & Idea 3 (Bank Conflict Avoidance)
//
// Grid: (x=n_kv_heads, y=batch_size)
// Block: (x=64 lanes, y=kv_mul warps), total = 64*kv_mul threads
// Shared: 2 * tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16)
// ====================================================================================
__global__ __launch_bounds__(512)
void fused_attention_kernel_optimized_3(
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



