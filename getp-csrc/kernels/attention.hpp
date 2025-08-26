#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"



__global__ void attention_scores_shared_mem_kernel(
    float *att, const float *q, const __hip_bfloat16 *key_cache,
    const float *mask, const int *positions,
    int batch_size, int n_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window)
{
    // Khai báo mảng trong shared memory. head_dim = 64
    __shared__ float q_shared[64];

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    int thread_in_block = threadIdx.z;

    // --- Tải Q vào Shared Memory ---
    // Mỗi thread trong block (32 threads) sẽ tải 2 phần tử của Q
    const float *q_head_global = q + 1LL * batch_idx * n_heads * head_dim + head_idx * head_dim;
    if (thread_in_block < 32) {
        q_shared[thread_in_block] = q_head_global[thread_in_block];
        q_shared[thread_in_block + 32] = q_head_global[thread_in_block + 32];
    }
    
    // Đợi tất cả thread trong block tải xong
    __syncthreads();

    // --- Các bước còn lại gần như giữ nguyên ---
    if (batch_idx >= batch_size || head_idx >= n_heads) return;
    int pos = positions[batch_idx];
    if (t > pos) return;

    int kv_dim = head_dim * (n_heads / 8);
    int kv_head = head_idx / 8;
    const __hip_bfloat16 *k_head = key_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim +
                                    1LL * layer_idx * seq_len * kv_dim + 1LL * t * kv_dim + kv_head * head_dim;

    // Thay vì dùng q_ptr, ta dùng con trỏ tới shared memory
    const float4 *q_ptr_shared = reinterpret_cast<const float4 *>(q_shared);
    
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int i = 0; i < head_dim / 4; i++)
    {
        float4 q_vec = q_ptr_shared[i];
        
        float k_vals[4];
        k_vals[0] = __bfloat162float(k_head[i * 4 + 0]);
        k_vals[1] = __bfloat162float(k_head[i * 4 + 1]);
        k_vals[2] = __bfloat162float(k_head[i * 4 + 2]);
        k_vals[3] = __bfloat162float(k_head[i * 4 + 3]);
        
        s0 += q_vec.x * k_vals[0];
        s1 += q_vec.y * k_vals[1];
        s2 += q_vec.z * k_vals[2];
        s3 += q_vec.w * k_vals[3];
    }

    float score = (s0 + s1) + (s2 + s3);
    score /= 8.0f;
    if (use_sliding_window && layer_idx % 2 == 0) {
        score += mask[pos * seq_len + t];
    }
    att[1LL * batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
}

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



