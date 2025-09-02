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

// Include FP8 conversion utilities
#include "../config.hpp"
#if USE_FP8_KV_CACHE
#include "../memory/fp8_e4m3.hpp"
#endif

// KV cache update kernel with configurable FP8/BF16 quantization
#if USE_FP8_KV_CACHE
__global__ void update_kv_cache_kernel(
    uint8_t *key_cache, uint8_t *value_cache,
    __hip_bfloat16 *key_cache_bf16, __hip_bfloat16 *value_cache_bf16,
    const float *k, const float *v,
    float *kv_scale,  // Per-layer scales [n_layers * 2]
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

    float k_val = k[1LL * batch_idx * kv_dim + dim_idx];
    float v_val = v[1LL * batch_idx * kv_dim + dim_idx];
    
    // Mixed precision: keep first few tokens in BF16 for accuracy
    if (pos < FP8_MIXED_PRECISION_TOKENS) {
        // Store in BF16 cache
        size_t k_bf16_idx = batch_idx * n_layers * FP8_MIXED_PRECISION_TOKENS * kv_dim +
                           layer_idx * FP8_MIXED_PRECISION_TOKENS * kv_dim + 
                           pos * kv_dim + dim_idx;
        size_t v_bf16_idx = k_bf16_idx;  // Same structure for value cache
        
        key_cache_bf16[k_bf16_idx] = __float2bfloat16(k_val);
        value_cache_bf16[v_bf16_idx] = __float2bfloat16(v_val);
    }
    
    // Always store in FP8 cache for long-range context
    size_t cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    
    // Get per-layer scales
    float k_scale = kv_scale[layer_idx * 2];
    float v_scale = kv_scale[layer_idx * 2 + 1];
    
    // Quantize to FP8 with scaling
    key_cache[cache_idx] = fp8_e4m3::float_to_fp8_e4m3(k_val * k_scale, true);
    value_cache[cache_idx] = fp8_e4m3::float_to_fp8_e4m3(v_val * v_scale, true);
}
#else
// Original BF16 version
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
#endif


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