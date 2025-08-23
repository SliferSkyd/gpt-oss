#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void softmax_kernel(float *x, int batch_size, int size)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    float *batch_x = x + batch_idx * size;

    __shared__ float shared_max[THREADS_PER_BLOCK];
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    // Find max value
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x)
    {
        max_val = fmaxf(max_val, batch_x[i]);
    }
    shared_max[tid] = max_val;
    __syncthreads();

    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    max_val = shared_max[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_x[i] = expf(batch_x[i] - max_val);
        sum += batch_x[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    sum = shared_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_x[i] /= sum;
    }
}



__global__ void softmax_kernel_variable_len(float *x, const int *positions,
                                            int batch_size, int n_heads, int max_seq_len)
{
    // Each block processes one head for one batch item
    int batch_idx = blockIdx.x / n_heads;
    int head_idx = blockIdx.x % n_heads;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    int pos = positions[batch_idx];
    int size = pos + 2; // Real size including the attention sink

    float *batch_head_x = x + (batch_idx * n_heads + head_idx) * max_seq_len;

    __shared__ float shared_max[THREADS_PER_BLOCK];
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    // Find max value in the valid range
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x)
    {
        max_val = fmaxf(max_val, batch_head_x[i]);
    }
    shared_max[tid] = max_val;
    __syncthreads();

    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // Compute exp and sum over the valid range
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_head_x[i] = expf(batch_head_x[i] - max_val);
        sum += batch_head_x[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    sum = shared_sum[0];
    __syncthreads();

    // Normalize over the valid range
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_head_x[i] /= sum;
    }
}
