#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "matmul.hpp"
#include "../memory/fp8_e4m3.hpp"  // FP8 E4M3 conversion utilities

// --- MODIFIED FUSED ATTENTION KERNEL (2-D mapping: lanes x warps) ---
#ifndef SW_WINDOW
#define SW_WINDOW 128
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
    const uint8_t * __restrict__ key_cache,         // [batch, n_layers, seq_len, kv_dim] (FP8 E4M3)
    const __hip_bfloat16 * __restrict__ key_cache_bf16, // [batch, n_layers, BF16_KEY_TOKENS, kv_dim] (BF16)
    const uint8_t * __restrict__ value_cache,       // [batch, n_layers, seq_len, kv_dim] (FP8 E4M3)
    const __hip_bfloat16 * __restrict__ value_cache_bf16, // [batch, n_layers, BF16_VALUE_TOKENS, kv_dim] (BF16)
    const float * __restrict__ key_cache_scales,    // [n_layers] - FP8 scale factors
    const float * __restrict__ value_cache_scales,  // [n_layers] - FP8 scale factors
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
    const uint8_t *k_base = key_cache   + (size_t)b * batch_kv_stride + layer_kv_offset;
    const __hip_bfloat16 *k_bf16_base = key_cache_bf16 +
        1LL * b * n_layers * BF16_KEY_TOKENS * kv_dim +
        1LL * layer_idx * BF16_KEY_TOKENS * kv_dim;
    const uint8_t *v_base = value_cache + (size_t)b * batch_kv_stride + layer_kv_offset;
    const __hip_bfloat16 *v_bf16_base = value_cache_bf16 +
        1LL * b * n_layers * BF16_VALUE_TOKENS * kv_dim +
        1LL * layer_idx * BF16_VALUE_TOKENS * kv_dim;
    float *out_head =
        output + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;

    // FP8 scale factors for this layer
    const float key_scale = key_cache_scales[layer_idx];
    const float value_scale = value_cache_scales[layer_idx];

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
        const int cache_pos = (layer_idx & 1) ? t : t % SW_WINDOW;

        float prod = 0.f;
        if (lane < head_dim) {
            float kf;
            // Use bf16 for first BF16_KEY_TOKENS tokens (higher precision)
            if (cache_pos < BF16_KEY_TOKENS) {
                const __hip_bfloat16 *k_vec_bf16 = k_bf16_base +
                    1LL * cache_pos * kv_dim + 1LL * kv_h * head_dim;
                kf = __bfloat162float(k_vec_bf16[lane]);
            } else {
                // Use FP8 for remaining tokens
                const uint8_t *k_vec = k_base + 1LL * cache_pos * kv_dim
                                              + 1LL * kv_h * head_dim;
                kf = fp8_e4m3::fp8_e4m3_to_float(k_vec[lane]) * key_scale;
            }
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
            const int cache_pos = (layer_idx & 1) ? t : t % SW_WINDOW;

            float vf;
            // Use bf16 for first BF16_VALUE_TOKENS tokens (higher precision)
            if (cache_pos < BF16_VALUE_TOKENS) {
                const __hip_bfloat16 *v_vec_bf16 = v_bf16_base +
                    1LL * cache_pos * kv_dim + 1LL * kv_h * head_dim;
                vf = __bfloat162float(v_vec_bf16[lane]);
            } else {
                // Use FP8 for remaining tokens
                const uint8_t *v_vec = v_base + 1LL * cache_pos * kv_dim
                                              + 1LL * kv_h * head_dim;
                vf = fp8_e4m3::fp8_e4m3_to_float(v_vec[lane]) * value_scale;
            }
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


// === Dynamic FP8 scale computation kernel ===
__global__ void compute_kv_cache_scales_kernel(
    float *key_scale_out, float *value_scale_out,
    const float *k, const float *v,
    int batch_size, int kv_dim, int layer_idx)
{
    extern __shared__ float shared_mem[];

    const int tid = threadIdx.x;
    const int num_elements = batch_size * kv_dim;

    // Find max absolute value for keys
    float local_key_max = 0.0f;
    float local_value_max = 0.0f;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < num_elements; idx += gridDim.x * blockDim.x) {
        local_key_max = fmaxf(local_key_max, fabsf(k[idx]));
        local_value_max = fmaxf(local_value_max, fabsf(v[idx]));
    }

    // Reduce within block
    shared_mem[tid] = local_key_max;
    shared_mem[tid + blockDim.x] = local_value_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
            shared_mem[tid + blockDim.x] = fmaxf(shared_mem[tid + blockDim.x], shared_mem[tid + stride + blockDim.x]);
        }
        __syncthreads();
    }

    // Write result (first thread only)
    if (tid == 0) {
        float key_max = shared_mem[0];
        float value_max = shared_mem[blockDim.x];

        // Compute scales to fit in FP8 E4M3 range (with 5% margin)
        const float target_range = fp8_e4m3::FP8_E4M3_MAX_VALUE * 0.95f;

        // Update scale using atomic max to handle multiple blocks
        float key_scale = (key_max > 0) ? (key_max / target_range) : 1.0f;
        float value_scale = (value_max > 0) ? (value_max / target_range) : 1.0f;

        atomicMax((int*)&key_scale_out[layer_idx], __float_as_int(key_scale));
        atomicMax((int*)&value_scale_out[layer_idx], __float_as_int(value_scale));
    }
}

// === KV cache update kernel (hybrid BF16/FP8 for keys and values) ===
__global__ void update_kv_cache_kernel(uint8_t *key_cache, __hip_bfloat16 *key_cache_bf16,
                                       uint8_t *value_cache, __hip_bfloat16 *value_cache_bf16,
                                       const float *k, const float *v,
                                       float *key_cache_scales, float *value_cache_scales,
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

    const int cache_pos = ((layer_idx & 1) ? pos : (pos % SW_WINDOW));

    // Key value to store
    float key_val = k[1LL*batch_idx * kv_dim + dim_idx];

    // Store keys: first BF16_KEY_TOKENS in bf16, rest in fp8
    if (cache_pos < BF16_KEY_TOKENS) {
        // Store in bf16 for higher precision
        const size_t bf16_idx = 1LL * batch_idx * n_layers * BF16_KEY_TOKENS * kv_dim +
                               1LL * layer_idx * BF16_KEY_TOKENS * kv_dim +
                               1LL * cache_pos * kv_dim + dim_idx;
        key_cache_bf16[bf16_idx] = __float2bfloat16(key_val);
    } else {
        // Store in fp8 for memory efficiency
        const size_t base = (size_t)batch_idx * batch_kv_stride + layer_kv_offset;
        const size_t cache_idx = base + (size_t)cache_pos * (size_t)kv_dim + (size_t)dim_idx;
        float key_inv_scale = 1.0f / key_cache_scales[layer_idx];
        key_cache[cache_idx] = fp8_e4m3::float_to_fp8_e4m3(key_val * key_inv_scale, true);
    }

    // Value value to store
    float value_val = v[1LL*batch_idx * kv_dim + dim_idx];

    // Store values: first BF16_VALUE_TOKENS in bf16, rest in fp8
    if (cache_pos < BF16_VALUE_TOKENS) {
        // Store in bf16 for higher precision
        const size_t bf16_idx = 1LL * batch_idx * n_layers * BF16_VALUE_TOKENS * kv_dim +
                               1LL * layer_idx * BF16_VALUE_TOKENS * kv_dim +
                               1LL * cache_pos * kv_dim + dim_idx;
        value_cache_bf16[bf16_idx] = __float2bfloat16(value_val);
    } else {
        // Store in fp8 for memory efficiency
        const size_t base = (size_t)batch_idx * batch_kv_stride + layer_kv_offset;
        const size_t cache_idx = base + (size_t)cache_pos * (size_t)kv_dim + (size_t)dim_idx;
        float value_inv_scale = 1.0f / value_cache_scales[layer_idx];
        value_cache[cache_idx] = fp8_e4m3::float_to_fp8_e4m3(value_val * value_inv_scale, true);
    }
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
