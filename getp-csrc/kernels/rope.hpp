#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void apply_rotary_emb_kernel(float *x, const float *cos_vals, const float *sin_vals,
                                        const int *positions, int batch_size,
                                        int n_heads, int head_dim)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int half = head_dim / 2;
    if (dim_idx >= half)
        return;

    int pos = positions[batch_idx];
    float *x_batch = x + batch_idx * n_heads * head_dim;
    const float *cos_pos = cos_vals + pos * half;
    const float *sin_pos = sin_vals + pos * half;

    float x1 = x_batch[head_idx * head_dim + dim_idx];
    float x2 = x_batch[head_idx * head_dim + half + dim_idx];

    float c = cos_pos[dim_idx];
    float s = sin_pos[dim_idx];

    x_batch[head_idx * head_dim + dim_idx] = x1 * c - x2 * s;
    x_batch[head_idx * head_dim + half + dim_idx] = x2 * c + x1 * s;
}
