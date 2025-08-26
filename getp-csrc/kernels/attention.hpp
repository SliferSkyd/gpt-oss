#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

// Original attention kernels (kept for backward compatibility)
__global__ void attention_scores_kernel(float *att, const float *q, const __hip_bfloat16 *key_cache,
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

    const float *q_head = q + 1LL*batch_idx * n_heads * head_dim + head_idx * head_dim;
    const __hip_bfloat16 *k_head = key_cache + 1LL*batch_idx * n_layers * seq_len * kv_dim +
                                    1LL*layer_idx * seq_len * kv_dim + 1LL*t * kv_dim + kv_head * head_dim;

    float score = 0.0f;
    for (int i = 0; i < head_dim; i++)
    {
        // Convert BF16 key to FP32 for computation
        float k_val = __bfloat162float(k_head[i]);
        score += q_head[i] * k_val;
    }
    score /= sqrtf((float)head_dim);

    // Apply sliding window mask if enabled
    if (use_sliding_window && (layer_idx % 2 == 0))
    {
        score += mask[pos * seq_len + t];
    }

    att[1LL*batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
}

__global__ void attention_weighted_sum_kernel(float *output, const float *att,
                                              const __hip_bfloat16 *value_cache, const int *positions,
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

    const float *att_head = att + 1LL * batch_idx * n_heads * seq_len + 1LL * head_idx * seq_len;
    float *out_head = output + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const __hip_bfloat16 *v_head = value_cache + 1LL*batch_idx * n_layers * seq_len * kv_dim +
                                        1LL*layer_idx * seq_len * kv_dim + 1LL*t * kv_dim + kv_head * head_dim;
        // Convert BF16 value to FP32 for computation
        float v_val = __bfloat162float(v_head[dim_idx]);
        sum += att_head[t] * v_val;
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

// NEW: Paged attention kernels
__global__ void paged_attention_scores_kernel(float *att, const float *q, 
                                              __hip_bfloat16 **block_table, const float *mask,
                                              const int *positions,
                                              int batch_size, int n_heads, int head_dim, int kv_dim,
                                              int seq_len, bool use_sliding_window)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int token_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int pos = positions[batch_idx];
    int actual_seq_len = pos + 1;
    if (token_idx >= actual_seq_len)
        return;

    // Calculate which page and offset within page this token belongs to
    int page_idx = token_idx / PAGE_SIZE;
    int page_offset = token_idx % PAGE_SIZE;
    
    // Get physical block ID from block table
    const __hip_bfloat16* physical_block = block_table[batch_idx * PAGES_PER_SEQ + page_idx];
    if (physical_block == nullptr) {
        printf("Error: Attempting to write to unallocated page (batch %d, page %d)\n", batch_idx, page_idx);
        return;
    }

    int kv_head = head_idx / 8; // Assuming GQA with 8:1 ratio
    
    // Calculate query pointer
    const float *q_head = q + 1LL*batch_idx * n_heads * head_dim + head_idx * head_dim;
    
    // Calculate key pointer in paged cache
    // Layout: [physical_block][layer][page_offset][kv_head][head_dim]
    const __hip_bfloat16 *k_head = physical_block + 
        1LL*page_offset * kv_dim +
        1LL*kv_head * head_dim;

    // Compute attention score
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++)
    {
        float k_val = __bfloat162float(k_head[i]);
        score += q_head[i] * k_val;
    }
    score /= sqrtf((float)head_dim);

    // Apply sliding window mask if enabled
    if (use_sliding_window)
    {
        score += mask[pos * seq_len + token_idx];
    }

    // Store attention score
    att[1LL*batch_idx * n_heads * seq_len + head_idx * seq_len + token_idx] = score;
}

__global__ void paged_attention_weighted_sum_kernel(float *output, const float *att,
                                                    __hip_bfloat16 **block_table,
                                                    const int *positions,
                                                    int batch_size, int n_heads, int head_dim,
                                                    int kv_dim, int seq_len)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads || dim_idx >= head_dim)
        return;

    int pos = positions[batch_idx];
    int actual_seq_len = pos + 1;
    int kv_head = head_idx / 8;

    // const float *att_head = att + 1LL * batch_idx * n_heads * actual_seq_len + 1LL * head_idx * actual_seq_len;
    // Corrected line in paged_attention_weighted_sum_kernel
    const float *att_head = att + 1LL * batch_idx * n_heads * seq_len + 1LL * head_idx * seq_len;
    float *out_head = output + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;

    float sum = 0.0f;
    
    // Iterate through all tokens in the sequence
    for (int token_idx = 0; token_idx < actual_seq_len; token_idx++)
    {
        // Calculate which page and offset within page this token belongs to
        int page_idx = token_idx / PAGE_SIZE;
        int page_offset = token_idx % PAGE_SIZE;
        
        // Get physical block ID from block table
        __hip_bfloat16* physical_block = block_table[batch_idx * PAGES_PER_SEQ + page_idx];
        if (physical_block == nullptr) {
            printf("Error: Attempting to write to unallocated page (batch %d, page %d)\n", batch_idx, page_idx);
            return;
        }
        
        // Calculate value pointer in paged cache
        const __hip_bfloat16 *v_head = physical_block + 1LL * PAGE_SIZE * kv_dim +
            1LL*page_offset * kv_dim +
            1LL*kv_head * head_dim;
        
        // Accumulate weighted value
        float v_val = __bfloat162float(v_head[dim_idx]);
        sum += att_head[token_idx] * v_val;
    }
    
    out_head[dim_idx] = sum;
}

// NEW: Paged KV cache update kernel
__global__ void paged_update_kv_cache_kernel(__hip_bfloat16 **block_table,
                                             const float *k, const float *v,
                                             const int *positions,
                                             int batch_size, int kv_dim)
{
    int batch_idx = blockIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || dim_idx >= kv_dim)
        return;

    int pos = positions[batch_idx];
    
    // Calculate which page and offset for the current position
    int page_idx = pos / PAGE_SIZE;
    int page_offset = pos % PAGE_SIZE;
    
    // Get physical block ID from block table
    __hip_bfloat16* physical_block = block_table[batch_idx * PAGES_PER_SEQ + page_idx];
    if (physical_block == nullptr) {
        printf("Error: Attempting to write to unallocated page (batch %d, page %d)\n", batch_idx, page_idx);
        return;
    }

    // Update key cache
    size_t cache_idx = 1LL*page_offset * kv_dim + dim_idx;
    
    physical_block[cache_idx] = __float2bfloat16(k[1LL*batch_idx * kv_dim + dim_idx]);
    physical_block[cache_idx + PAGE_SIZE * kv_dim] = __float2bfloat16(v[1LL*batch_idx * kv_dim + dim_idx]);
}



