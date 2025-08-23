#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void attention_scores_kernel(float *att, const float *q, const float *key_cache,
                                        const float *mask, const int *positions,
                                        int batch_size, int n_heads, int head_dim,
                                        int seq_len, int n_layers, int layer_idx,
                                        bool use_sliding_window)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int pos = positions[batch_idx];
    if (t > pos)
        return;

    int kv_dim = head_dim * (n_heads / 8); // Assuming GQA with 4:1 ratio
    int kv_head = head_idx / 8;

    const float *q_head = q + batch_idx * n_heads * head_dim + head_idx * head_dim;
    const float *k_head = key_cache + batch_idx * n_layers * seq_len * kv_dim +
                          layer_idx * seq_len * kv_dim + t * kv_dim + kv_head * head_dim;

    float score = 0.0f;
    for (int i = 0; i < head_dim; i++)
    {
        score += q_head[i] * k_head[i];
    }
    score /= sqrtf((float)head_dim);

    // Apply sliding window mask if enabled
    if (use_sliding_window && (layer_idx % 2 == 0))
    {
        score += mask[pos * seq_len + t];
    }

    att[batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
}

__global__ void attention_weighted_sum_kernel(float *output, const float *att,
                                              const float *value_cache, const int *positions,
                                              int batch_size, int n_heads, int head_dim,
                                              int seq_len, int n_layers, int layer_idx)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads || dim_idx >= head_dim)
        return;

    int pos = positions[batch_idx];
    int kv_dim = head_dim * (n_heads / 8);
    int kv_head = head_idx / 8;

    const float *att_head = att + batch_idx * n_heads * seq_len + head_idx * seq_len;
    float *out_head = output + batch_idx * n_heads * head_dim + head_idx * head_dim;

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const float *v_head = value_cache + batch_idx * n_layers * seq_len * kv_dim +
                              layer_idx * seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
        sum += att_head[t] * v_head[dim_idx];
    }
    out_head[dim_idx] = sum;
}

__global__ void add_sinks_kernel(float *att, const __hip_bfloat16 *sinks, const int *positions,
                                 int seq_len, int n_heads)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int pos = positions[batch_idx];
    if (pos + 1 < seq_len)
    {
        // Index for the sink is at pos + 1
        int sink_att_idx = batch_idx * n_heads * seq_len + head_idx * seq_len + (pos + 1);
        // Convert bfloat16 sink to fp32 on-the-fly
        float sink_fp32 = __bfloat162float(sinks[head_idx]);
        att[sink_att_idx] = sink_fp32;
    }
}



