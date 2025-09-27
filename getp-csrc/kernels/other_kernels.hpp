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


__global__ void axpy_inplace_b_bf16(
    float *__restrict__ a,
    const __hip_bfloat16 *__restrict__ b,  // <<< b in BF16
    float factor,
    int batch_size, int size)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_size = (size_t)batch_size * (size_t)size;

    if (idx < total_size)
    {
        const float bv = __bfloat162float(b[idx]); // BF16 -> FP32
        a[idx] += bv * factor;                     // accumulate in FP32
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


// NEW: Kernel to add bias to matrix multiplication result with bfloat16 bias
__global__ void add_bias_kernel(float *output, const float *bias, int batch_size, int size)
{
    size_t idx = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size)
    {
        int dim_idx = idx % size;
        // Convert bfloat16 bias to fp32 on-the-fly
        float bias_fp32 = (bias[dim_idx]);
        output[idx] += bias_fp32;
    }
}
__global__ void scale_bf16_kernel(__hip_bfloat16 *dst, const __hip_bfloat16 *src,
                                  float s, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __bfloat162float(src[i]);
  v *= s;
  dst[i] = __float2bfloat16(v);
}


// Elementwise add for int arrays: dst[i] += src[i]
__global__ void add_int_arrays_kernel(int* __restrict__ dst,
                                      const int* __restrict__ src,
                                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void copy_embeddings_kernel(float *output, const float *embeddings,
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
    float embedding_fp32 = (embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}

__global__ void copy_embeddings_kernel_with_active(float *output, const float *embeddings,
                                       const int *tokens, const bool *slot_active, 
                                       int batch_size, int hidden_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / hidden_dim;
    size_t dim_idx = idx % hidden_dim;

    if (batch_idx >= batch_size || dim_idx >= hidden_dim)
        return;
    
    // Check if slot is active, if not, set output to zero
    if (!slot_active[batch_idx]) {
        output[idx] = 0.0f;
        return;
    }
    
    int token = tokens[batch_idx];
    if (token < 0)
        return; // Safety check for invalid tokens

    // Convert bfloat16 embedding to fp32 on-the-fly
    float embedding_fp32 = (embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}


// Zero-out the KV cache region for a single logical slot across all layers/positions.
// Layout matches update_kv_cache_kernel's likely addressing:
// (((layer * batch_size + slot) * max_seq_len) + pos) * kv_dim + d
__global__ void clear_kv_cache_for_slot_kernel(
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int batch_size,
    int n_layers,
    int max_seq_len,
    int kv_dim,
    int slot)
{
    int layer = blockIdx.x;     // [0, n_layers)
    int pos   = blockIdx.y;     // [0, max_seq_len)
    int tid   = threadIdx.x;

    // Sanity guards (cheap, avoids accidental OOB if launched oddly)
    if (layer >= n_layers || pos >= max_seq_len || slot >= batch_size) return;

    const size_t base = ( ( (size_t)layer * batch_size + (size_t)slot ) * max_seq_len + (size_t)pos ) * (size_t)kv_dim;

    // Stride over kv_dim
    for (int d = tid; d < kv_dim; d += blockDim.x) {
        key_cache  [base + d] = 0.0f;
        value_cache[base + d] = 0.0f;
    }
}
