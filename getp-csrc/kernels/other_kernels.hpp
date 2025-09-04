#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void accumulate_kernel(float *a, const float *b, float factor,
                                  int batch_size, int size)
{
    size_t idx = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = batch_size * size;

    if (idx < total_size)
    {
        a[idx] += b[idx] * factor;
    }
}

// NEW: Kernel to add bias to matrix multiplication result with bfloat16 bias
__global__ void add_bias_kernel(float *output, const __hip_bfloat16 *bias, int batch_size, int size)
{
    size_t idx = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size)
    {
        int dim_idx = idx % size;
        // Convert bfloat16 bias to fp32 on-the-fly
        float bias_fp32 = __bfloat162float(bias[dim_idx]);
        output[idx] += bias_fp32;
    }
}

// NEW: KV cache update kernel with BF16 quantization
__global__ void update_kv_cache_kernel(__hip_bfloat16 *key_cache, __hip_bfloat16 *value_cache,
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
    key_cache[k_cache_idx] = __float2bfloat16(k[1LL*batch_idx * kv_dim + dim_idx]);

    // Update value cache - convert FP32 to BF16 for memory efficiency
    size_t v_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    value_cache[v_cache_idx] = __float2bfloat16(v[1LL*batch_idx * kv_dim + dim_idx]);
}


__global__ void copy_embeddings_kernel(float *output, const __hip_bfloat16 *embeddings,
                                       const int *tokens, int batch_size, int hidden_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / hidden_dim;
    size_t dim_idx = idx % hidden_dim;

    if (batch_idx >= batch_size || dim_idx >= hidden_dim)
        return;
    int token = tokens[batch_idx];
    if (token < 0)
        return; // Safety check for invalid tokens

    // Convert bfloat16 embedding to fp32 on-the-fly
    float embedding_fp32 = __bfloat162float(embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}

static __device__ __forceinline__ float bf16_to_f32(uint16_t x) {
    // Reinterpret BF16 as the high 16 bits of FP32
    uint32_t u = ((uint32_t)x) << 16;
    return __int_as_float((int)u);
}

template<int VEC=4>
__global__ void add_bias_kernel_vec(
    float* __restrict__ output,                 // [batch_size, size] row-major
    const __hip_bfloat16* __restrict__ bias,    // [size] (bf16)
    int size)
{
    // Each block = one row (batch element)
    const int row = blockIdx.x;
    float* __restrict__ row_out = output + (size_t)row * size;

    // Vectorized loop: process VEC columns per iteration
    int d = threadIdx.x * VEC;
    const int stride = blockDim.x * VEC;

    // Main vectorized body (VEC==4): load two u32 (4 bf16), convert via shifts
    for (; d + (VEC-1) < size; d += stride)
    {
        // Unaligned loads are okay on modern AMD GPUs; use memcpy to be safe wrt aliasing
        uint32_t p0, p1;
        __builtin_memcpy(&p0, reinterpret_cast<const uint32_t*>(bias + d), sizeof(uint32_t));
        __builtin_memcpy(&p1, reinterpret_cast<const uint32_t*>(bias + d + 2), sizeof(uint32_t));

        // p0 holds bias[d+0] (lo16) and bias[d+1] (hi16)
        // p1 holds bias[d+2] (lo16) and bias[d+3] (hi16)
        uint16_t b0 = (uint16_t)( p0        & 0xFFFFu);
        uint16_t b1 = (uint16_t)((p0 >> 16) & 0xFFFFu);
        uint16_t b2 = (uint16_t)( p1        & 0xFFFFu);
        uint16_t b3 = (uint16_t)((p1 >> 16) & 0xFFFFu);

        float f0 = bf16_to_f32(b0);
        float f1 = bf16_to_f32(b1);
        float f2 = bf16_to_f32(b2);
        float f3 = bf16_to_f32(b3);

        // Coalesced stores
        row_out[d+0] += f0;
        row_out[d+1] += f1;
        row_out[d+2] += f2;
        row_out[d+3] += f3;
    }

    // Scalar cleanup for remaining columns in this thread's lane
    for (; d < size; d += stride) {
        // Single element path
        uint16_t b16;
        __builtin_memcpy(&b16, reinterpret_cast<const uint16_t*>(bias + d), sizeof(uint16_t));
        row_out[d] += bf16_to_f32(b16);
    }
}

// Convenience launcher: chooses a good block size and launches one block per row
inline void add_bias_launch(float* output,
                            const __hip_bfloat16* bias,
                            int batch_size,
                            int size,
                            hipStream_t stream = 0)
{
    // 256 threads → each thread handles 4 cols per iteration (1KB per warp per iter)
    constexpr int THREADS = 256;
    dim3 grid(batch_size);
    dim3 block(THREADS);
    add_bias_kernel_vec<4><<<grid, block, 0, stream>>>(output, bias, size);
}
