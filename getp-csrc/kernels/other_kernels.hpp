#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void accumulate_kernel(float *a, const float *b, float factor,
                                  int batch_size, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * size;

    if (idx < total_size)
    {
        a[idx] += b[idx] * factor;
    }
}

// NEW: Kernel to add bias to matrix multiplication result with bfloat16 bias
__global__ void add_bias_kernel(float *output, const __hip_bfloat16 *bias, int batch_size, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size)
    {
        int dim_idx = idx % size;
        // Convert bfloat16 bias to fp32 on-the-fly
        float bias_fp32 = __bfloat162float(bias[dim_idx]);
        output[idx] += bias_fp32;
    }
}

// NEW: KV cache update kernel
__global__ void update_kv_cache_kernel(float *key_cache, float *value_cache,
                                       const float *k, const float *v,
                                       const int *positions, int batch_size,
                                       int n_layers, int layer_idx, int seq_len,
                                       int kv_dim)
{
    int batch_idx = blockIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || dim_idx >= kv_dim)
        return;

    int pos = positions[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

    // Update key cache
    int k_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    key_cache[k_cache_idx] = k[batch_idx * kv_dim + dim_idx];

    // Update value cache
    int v_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    value_cache[v_cache_idx] = v[batch_idx * kv_dim + dim_idx];
}


__global__ void copy_embeddings_kernel(float *output, const __hip_bfloat16 *embeddings,
                                       const int *tokens, int batch_size, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_dim;
    int dim_idx = idx % hidden_dim;

    if (batch_idx >= batch_size || dim_idx >= hidden_dim)
        return;
    int token = tokens[batch_idx];
    if (token < 0)
        return; // Safety check for invalid tokens

    // Convert bfloat16 embedding to fp32 on-the-fly
    float embedding_fp32 = __bfloat162float(embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}