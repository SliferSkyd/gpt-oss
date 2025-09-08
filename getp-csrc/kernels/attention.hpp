#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/paged_attention.hpp"
#include "matmul.hpp"

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
    float * __restrict__ output,               // [batch, n_heads, head_dim]
    const float * __restrict__ q,              // [batch, n_heads * head_dim]
    const float * __restrict__ key_cache,   // [batch, n_layers, seq_len, kv_dim]
    const float * __restrict__ value_cache, // [batch, n_layers, seq_len, kv_dim]
    const __hip_bfloat16 * __restrict__ sinks,       // [n_heads] (already layer-offset on host)
    const float * __restrict__ mask,                 // [seq_len, seq_len]
    const int * __restrict__ positions,              // [batch]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window)
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
    const float *k_base =
        key_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    const float *v_base =
        value_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
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

    // === Fast path: map one lane per head-dim element (designed for head_dim == warpSize) ===
    // Falls back correctly when head_dim < warpSize by masking lanes; for head_dim > warpSize,
    // add a lane-strided loop over 'i += WARP' (omitted here since you asked about 64).
    const float q_lane = (lane < head_dim) ? q_head[lane] : 0.0f;

    // ---- Pass 1: compute attention scores into s_att[0..win_core_len-1] (all warps participate)
    for (int w = wid; w < win_core_len; w += WARPS) {
        const int t = win_start + w;
        const float *k_vec = k_base + 1LL * t * kv_dim + 1LL * kv_h * head_dim;

        float prod = 0.f;
        if (lane < head_dim) {
            prod = (float)q_lane * (k_vec[lane]);
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
    // Every warp accumulates a partial output vector for its token stripe.
    if (lane < head_dim) {
        float partial = 0.f;
        for (int w = wid; w < win_core_len; w += WARPS) {
            const int t = win_start + w;
            const float *v_vec = v_base + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
            partial += (double)s_att[w] * (v_vec[lane]);
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


__global__ void fused_rmsnorm_qkv_rope_kvcache_kernel(
    // Outputs
    float *__restrict__ q_out,
    float *__restrict__ k_out,
    float *__restrict__ v_out,
    __hip_bfloat16 *__restrict__ key_cache,
    __hip_bfloat16 *__restrict__ value_cache,
    // Inputs
    const float *__restrict__ x_in,
    const __hip_bfloat16 *__restrict__ rms_w,
    const __hip_bfloat16 *__restrict__ w_qkv,
    const __hip_bfloat16 *__restrict__ b_qkv,
    const float *__restrict__ cos_vals,
    const float *__restrict__ sin_vals,
    const int *__restrict__ positions,
    // Config
    int hidden_dim,
    int qkv_dim,
    int q_dim,
    int k_dim,
    int v_dim,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads,
    int n_layers,
    int layer_idx,
    int seq_len)
{
    // Each thread block processes one item in the batch
    int batch_idx = blockIdx.x;
    // Each thread calculates one output dimension (for Q, K, or V)
    int out_dim_idx = threadIdx.x;

    // Shared memory for RMSNorm and holding the input vector `x`
    extern __shared__ float s_mem[];
    float *s_x = s_mem;                    // size: hidden_dim
    float *s_rms_sum = &s_mem[hidden_dim]; // size: 1 (or blockDim.x for reduction)

    // --- 1. Load input `x` to shared memory and start RMSNorm ---
    const float *x = x_in + batch_idx * hidden_dim;

    // Parallel load into shared memory
    for (int i = out_dim_idx; i < hidden_dim; i += blockDim.x)
    {
        s_x[i] = x[i];
    }

    // Calculate sum of squares for RMSNorm in parallel
    float ss = 0.0f;
    for (int i = out_dim_idx; i < hidden_dim; i += blockDim.x)
    {
        ss += s_x[i] * s_x[i];
    }

    // Reduction for sum of squares
    // Note: Using a single shared memory location for simplicity. For larger blocks,
    // a parallel reduction tree in shared memory would be more efficient.
    if (threadIdx.x == 0)
        s_rms_sum[0] = 0.0f;
    __syncthreads();
    atomicAdd(s_rms_sum, ss);
    __syncthreads();

    // Finalize RMSNorm scaling factor
    if (threadIdx.x == 0)
    {
        float mean_ss = s_rms_sum[0] / hidden_dim;
        s_rms_sum[0] = 1.0f / sqrtf(mean_ss + 1e-5f);
    }
    __syncthreads();
    float rms_scale = s_rms_sum[0];

    // --- 2. Matmul, Bias, RoPE, and KV Cache Update ---
    // Each thread computes one element of the QKV output vector
    for (int i = out_dim_idx; i < qkv_dim; i += blockDim.x)
    {
        float val = 0.0f;
        // Perform dot product (Matmul)
        for (int j = 0; j < hidden_dim; ++j)
        {
            // Apply RMSNorm on the fly
            float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
            val += normalized_x * __bfloat162float(w_qkv[i * hidden_dim + j]);
        }

        // Add bias
        val += __bfloat162float(b_qkv[i]);

        // --- Split, Apply RoPE, and Write to Output ---
        int pos = positions[batch_idx];

        if (i < q_dim)
        { // This is a Q dimension
            int head_idx = i / head_dim;
            int h_dim_idx = i % head_dim;
            int half = head_dim / 2;

            if (h_dim_idx < half)
            {
                float q1 = val;
                // Fetch the corresponding q2 value for RoPE
                // This requires a temporary buffer or a more complex indexing scheme.
                // For simplicity here, we assume a way to get q2. A real implementation
                // might re-compute or use shared memory.
                // This part is complex to fuse perfectly without restructuring data.
                // We'll apply RoPE after a sync, reading from an intermediate register stage.
                // Let's calculate the other pair part.
                int i2 = i + half;
                float val2 = 0.0f;
                for (int j = 0; j < hidden_dim; ++j)
                {
                    float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
                    val2 += normalized_x * __bfloat162float(w_qkv[i2 * hidden_dim + j]);
                }
                val2 += __bfloat162float(b_qkv[i2]);

                float c = cos_vals[pos * half + h_dim_idx];
                float s = sin_vals[pos * half + h_dim_idx];

                q_out[batch_idx * q_dim + i] = q1 * c - val2 * s;
                q_out[batch_idx * q_dim + i2] = val2 * c + q1 * s;
            }
        }
        else if (i < q_dim + k_dim)
        { // This is a K dimension
            int k_idx = i - q_dim;
            int head_idx = k_idx / head_dim;
            int h_dim_idx = k_idx % head_dim;
            int half = head_dim / 2;

            if (h_dim_idx < half)
            {
                float k1 = val;
                int i2 = i + half;
                float val2 = 0.0f;
                for (int j = 0; j < hidden_dim; ++j)
                {
                    float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
                    val2 += normalized_x * __bfloat162float(w_qkv[i2 * hidden_dim + j]);
                }
                val2 += __bfloat162float(b_qkv[i2]);

                float c = cos_vals[pos * half + h_dim_idx];
                float s = sin_vals[pos * half + h_dim_idx];

                float final_k1 = k1 * c - val2 * s;
                float final_k2 = val2 * c + k1 * s;

                // Write to k_out
                k_out[batch_idx * k_dim + k_idx] = final_k1;
                k_out[batch_idx * k_dim + k_idx + half] = final_k2;

                // Write to key_cache
                size_t k_cache_idx1 = (size_t)batch_idx * n_layers * seq_len * k_dim +
                                      (size_t)layer_idx * seq_len * k_dim + (size_t)pos * k_dim + k_idx;
                size_t k_cache_idx2 = k_cache_idx1 + half;
                key_cache[k_cache_idx1] = __float2bfloat16(final_k1);
                key_cache[k_cache_idx2] = __float2bfloat16(final_k2);
            }
        }
        else
        { // This is a V dimension
            int v_idx = i - q_dim - k_dim;
            // Write to v_out
            v_out[batch_idx * v_dim + v_idx] = val;

            // Write to value_cache
            size_t v_cache_idx = (size_t)batch_idx * n_layers * seq_len * v_dim +
                                 (size_t)layer_idx * seq_len * v_dim + (size_t)pos * v_dim + v_idx;
            value_cache[v_cache_idx] = __float2bfloat16(val);
        }
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

// NEW: KV cache update kernel with BF16 quantization
__global__ void update_kv_cache_kernel(float *key_cache, float *value_cache,
                                       const float *k, const float *v,
                                       const int *positions, int batch_size,
                                       int n_layers, int layer_idx, int seq_len,
                                       int kv_dim)
{
    size_t batch_idx = blockIdx.x;
    size_t dim_idx = 1LL * blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || dim_idx >= kv_dim)
        return;

    int pos = positions[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

    // Update key cache - convert FP32 to BF16 for memory efficiency
    size_t k_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    key_cache[k_cache_idx] = (k[1LL*batch_idx * kv_dim + dim_idx]);

    // Update value cache - convert FP32 to BF16 for memory efficiency
    size_t v_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    value_cache[v_cache_idx] = (v[1LL*batch_idx * kv_dim + dim_idx]);
}

// Paged KV cache update (FP32). Writes one token row into the current page.
// Layout per page: [PAGE_SIZE, kv_dim] for K, then [PAGE_SIZE, kv_dim] for V.
__global__ void paged_update_kv_cache_kernel_fp32(
    float **block_table,          // [batch * PAGES_PER_SEQ], device pointers to pages
    const float *k,               // [batch, kv_dim]
    const float *v,               // [batch, kv_dim]
    const int   *positions,       // [batch]
    int batch_size,
    int kv_dim)
{
    const int b      = blockIdx.x;
    const int d_idx  = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || d_idx >= kv_dim) return;

    const int pos       = positions[b];
    const int page_idx  = pos / PAGE_SIZE;
    const int page_off  = pos % PAGE_SIZE;

    // Resolve page base for this (batch slot, page)
    float *page_base = block_table[b * PAGES_PER_SEQ + page_idx];
    if (page_base == nullptr) return; // nothing to write (page not allocated yet)

    // Row bases inside the page
    float *k_row = page_base + 1LL * page_off * kv_dim;
    float *v_row = page_base + 1LL * PAGE_SIZE * kv_dim + 1LL * page_off * kv_dim;

    // Source rows from current minibatch K/V
    const float *k_src = k + 1LL * b * kv_dim;
    const float *v_src = v + 1LL * b * kv_dim;

    k_row[d_idx] = k_src[d_idx];
    v_row[d_idx] = v_src[d_idx];
}

// Fused attention using paged KV cache (FP32).
// Page layout: [K: PAGE_SIZE*kv_dim floats][V: PAGE_SIZE*kv_dim floats]
__global__ void fused_attention_kernel_paged(
    float * __restrict__ output,               // [batch, n_heads, head_dim]
    const float * __restrict__ q,              // [batch, n_heads * head_dim]
    float * const * __restrict__ block_table,  // [batch * PAGES_PER_SEQ] -> page ptrs
    const __hip_bfloat16 * __restrict__ sinks, // [n_heads] (layer-offset applied by host)
    const float * __restrict__ mask,           // [seq_len, seq_len]
    const int * __restrict__ positions,        // [batch]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int kv_dim, int seq_len,
    bool use_sliding_window)
{
    extern __shared__ float s_sh[];

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    if (b >= batch_size || h >= n_heads) return;

    // Wave bookkeeping
    const int WARP  = warpSize;                       // 64 on AMD
    const int WARPS = (blockDim.x + WARP - 1) / WARP;
    const int lane  = tid & (WARP - 1);
    const int wid   = tid >> (__ffs(WARP) - 1);

    // Shapes / mapping
    const int pos        = positions[b];
    const int gqa_ratio  = n_heads / n_kv_heads;
    const int kv_h       = h / gqa_ratio;

    const float *q_head  = q + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;
    float *out_head      = output + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;

    // Sliding window setup
    const bool apply_window = use_sliding_window && ((/*layer parity handled at call site*/true));
    const int  win_start    = apply_window ? max(0, pos - (SW_WINDOW - 1)) : 0;
    const int  win_core_len = pos - win_start + 1;                // tokens before (optional) sink
    const int  att_cap      = apply_window ? (SW_WINDOW + 1) : seq_len;

    float *s_att      = s_sh;                          // [att_cap]
    float *s_partials = s_att + att_cap;               // [WARPS * head_dim]
    float *s_reduce   = s_partials + WARPS * head_dim; // [WARPS]
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    // One lane per head-dim element (fast path when head_dim == warpSize)
    const float q_lane = (lane < head_dim) ? q_head[lane] : 0.0f;

    // ---- Pass 1: compute attention scores into s_att[0..win_core_len-1]
    for (int w = wid; w < win_core_len; w += WARPS) {
        const int t        = win_start + w;
        const int page_idx = t / PAGE_SIZE;
        const int page_off = t % PAGE_SIZE;

        float *page = block_table[b * PAGES_PER_SEQ + page_idx];
        float acc = -INFINITY; // default if page missing

        if (page != nullptr) {
            // K row for this token + kv head
            const float *k_vec = page + 1LL * page_off * kv_dim + 1LL * kv_h * head_dim;

            float prod = 0.f;
            if (lane < head_dim) prod = q_lane * k_vec[lane];
            const float dot = warpReduceSum(prod);

            if (lane == 0) {
                acc = dot * inv_sqrt_d;
                if (apply_window) acc += mask[1LL * pos * seq_len + t];
                s_att[w] = acc;
            }
        }
        if (page == nullptr && lane == 0) {
            s_att[w] = acc; // -INF
        }
    }
    __syncthreads();

    // Append sink (if any)
    int softmax_len = win_core_len;
    if (pos + 1 < seq_len) {
        if (tid == 0) s_att[softmax_len] = __bfloat162float(sinks[h]);
        softmax_len += 1;
    }
    __syncthreads();

    // ---- Softmax
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
            const int t        = win_start + w;
            const int page_idx = t / PAGE_SIZE;
            const int page_off = t % PAGE_SIZE;

            float *page = block_table[b * PAGES_PER_SEQ + page_idx];
            if (page == nullptr) continue;

            const float *v_vec = page
                + 1LL * PAGE_SIZE * kv_dim
                + 1LL * page_off * kv_dim
                + 1LL * kv_h * head_dim;

            partial += s_att[w] * v_vec[lane];
        }
        s_partials[wid * head_dim + lane] = partial;
    }
    __syncthreads();

    // Cross-warp reduce (one lane per dim)
    if (wid == 0 && lane < head_dim) {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += s_partials[w * head_dim + lane];
        out_head[lane] = acc;
    }
}
