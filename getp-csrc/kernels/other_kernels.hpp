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
        // Use double precision for the accumulation
        double a_val = (double)a[idx];
        double b_val = (double)b[idx];
        double factor_val = (double)factor;
        a[idx] = (float)(a_val + b_val * factor_val);
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

// NEW: KV cache update kernel with BF16 quantization and slot_active check
__global__ void update_kv_cache_kernel(float *key_cache, float *value_cache,
                                       const float *k, const float *v,
                                       const int *positions, const bool *slot_active,
                                       int batch_size, int n_layers, int layer_idx,
                                       int seq_len, int kv_dim)
{
    size_t batch_idx = blockIdx.x;
    size_t dim_idx = 1LL * blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || dim_idx >= kv_dim)
        return;
    
    // Skip inactive slots
    if (!slot_active[batch_idx])
        return;

    int pos = positions[batch_idx];
    if (pos >= seq_len || pos < 0)
        return; // Safety check

    // Update key cache
    size_t k_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    key_cache[k_cache_idx] = (k[1LL*batch_idx * kv_dim + dim_idx]);

    // Update value cache
    size_t v_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    value_cache[v_cache_idx] = (v[1LL*batch_idx * kv_dim + dim_idx]);
}


// Kernel to clear KV cache for a specific layer of a slot
__global__ void clear_kv_cache_layer_kernel(float *key_cache, float *value_cache,
                                            int slot_idx, int layer_idx, int seq_len, 
                                            int kv_dim, int n_layers)
{
    size_t pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos_idx >= seq_len || dim_idx >= kv_dim)
        return;
    
    // Clear key cache for this slot and layer
    size_t k_cache_idx = slot_idx * n_layers * seq_len * kv_dim +
                        layer_idx * seq_len * kv_dim + 
                        pos_idx * kv_dim + dim_idx;
    key_cache[k_cache_idx] = 0.0f;
    
    // Clear value cache for this slot and layer
    size_t v_cache_idx = slot_idx * n_layers * seq_len * kv_dim +
                        layer_idx * seq_len * kv_dim + 
                        pos_idx * kv_dim + dim_idx;
    value_cache[v_cache_idx] = 0.0f;
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

// Version with slot_active check for continuous batching
__global__ void copy_embeddings_kernel_with_active(float *output, const __hip_bfloat16 *embeddings,
                                                   const int *tokens, const bool *slot_active,
                                                   int batch_size, int hidden_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / hidden_dim;
    size_t dim_idx = idx % hidden_dim;

    if (batch_idx >= batch_size || dim_idx >= hidden_dim)
        return;
    
    // Skip inactive slots - set output to 0
    if (!slot_active[batch_idx]) {
        output[idx] = 0.0f;
        return;
    }
    
    int token = tokens[batch_idx];
    if (token < 0) {
        output[idx] = 0.0f; // Set to 0 for invalid tokens
        return;
    }

    // Convert bfloat16 embedding to fp32 on-the-fly
    float embedding_fp32 = __bfloat162float(embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}