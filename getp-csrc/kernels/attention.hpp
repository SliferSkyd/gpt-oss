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
    const __hip_bfloat16 *k_head = key_cache + 
                                    1LL * batch_idx * n_layers * seq_len * kv_dim +
                                    1LL * layer_idx * seq_len * kv_dim + 
                                    1LL * t * kv_dim + 
                                    kv_head * head_dim;

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






/**
 * @brief Fused attention kernel for AMD GPUs using HIP.
 *
 * This single kernel performs four sequential operations from the original attention mechanism:
 * 1.  **Attention Score Calculation**: Computes dot-product scores between queries (Q) and keys (K).
 * 2.  **Add Sinks**: Adds special "sink" values to the attention scores.
 * 3.  **Softmax**: Applies a numerically stable softmax to the scores to get attention weights.
 * 4.  **Weighted Sum**: Computes the weighted sum of values (V) using the attention weights.
 *
 * This fusion minimizes kernel launch overhead and leverages fast on-chip shared memory
 * for intermediate data (attention scores), significantly reducing global memory traffic.
 *
 * @param output The output buffer for the attention head [batch_size, n_heads, head_dim].
 * @param q The query vectors [batch_size, n_heads * head_dim].
 * @param key_cache The key cache [batch_size, n_layers, seq_len, kv_dim].
 * @param value_cache The value cache [batch_size, n_layers, seq_len, kv_dim].
 * @param sinks The attention sink values [n_layers, n_heads].
 * @param mask The sliding window attention mask [seq_len, seq_len].
 * @param positions The current sequence position for each item in the batch [batch_size].
 * @param batch_size The number of sequences being processed in parallel.
 * @param n_heads The number of attention heads.
 * @param n_kv_heads The number of key/value heads (for Grouped-Query Attention).
 * @param head_dim The dimension of each attention head.
 * @param seq_len The maximum sequence length of the model.
 * @param n_layers The total number of layers in the model.
 * @param layer_idx The index of the current transformer layer.
 * @param use_sliding_window A flag to enable the sliding window attention mask.
 */
__global__ void fused_attention_kernel(
    float *output,              // Output: [batch, n_heads, head_dim]
    const float *q,             // Input Q: [batch, n_heads * head_dim]
    const __hip_bfloat16 *key_cache,   // K Cache: [batch, n_layers, seq_len, kv_dim]
    const __hip_bfloat16 *value_cache, // V Cache: [batch, n_layers, seq_len, kv_dim]
    const __hip_bfloat16 *sinks,       // Sinks: [n_layers, n_heads]
    const float *mask,                 // Mask: [seq_len, seq_len]
    const int *positions,              // Positions: [batch]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window)
{
    // --- Shared Memory Declaration ---
    // Use dynamic shared memory allocated at launch time.
    extern __shared__ float s_data[];
    float* s_q = s_data;                               // For the query vector of the current head. Size: head_dim
    float* s_att = (float*)&s_q[head_dim];             // For attention scores. Size: seq_len
    float* s_reduce = (float*)&s_att[seq_len];         // For block-wide reductions in softmax. Size: blockDim.x

    // --- Thread & Block Identification ---
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Early exit for padding blocks in the grid
    if (batch_idx >= batch_size || head_idx >= n_heads) return;

    const int pos = positions[batch_idx];

    // --- GQA (Grouped-Query Attention) Setup ---
    const int gqa_ratio = n_heads / n_kv_heads;
    const int kv_head = head_idx / gqa_ratio;
    const int kv_dim = head_dim * n_kv_heads;

    // --- Set up Global Memory Pointers ---
    const float* q_head_ptr = q + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;
    const __hip_bfloat16* k_cache_layer_ptr = key_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    float* output_ptr = output + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;

    // --- Step 1: Load Query (Q) into Shared Memory ---
    // All threads in the block cooperate to load the query vector.
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = q_head_ptr[i];
    }
    __syncthreads();

    // --- Step 2: Calculate Attention Scores (Q * K^T) ---
    const float inv_sqrt_head_dim = rsqrtf((float)head_dim);
    // Parallelize the score calculation over the sequence length `t`.
    for (int t = tid; t <= pos; t += blockDim.x) {
        const __hip_bfloat16* k_vec = k_cache_layer_ptr + 1LL * t * kv_dim + 1LL * kv_head * head_dim;
        float score = 0.0f;
        
        // Dot product between shared Q and global K
        for (int i = 0; i < head_dim; i++) {
            score += s_q[i] * __bfloat162float(k_vec[i]);
        }
        score *= inv_sqrt_head_dim;

        // Optionally apply sliding window mask
        if (use_sliding_window && (layer_idx % 2 == 0)) {
            score += mask[1LL * pos * seq_len + t];
        }
        s_att[t] = score;
    }
    __syncthreads();

    // --- Step 3: Add Sinks ---
    int softmax_len = pos + 1;
    if (pos + 1 < seq_len) {
        if (tid == 0) { // Only one thread needs to write the sink value
            s_att[pos + 1] = __bfloat162float(sinks[head_idx]);
        }
        softmax_len++; // The sink increases the effective sequence length for softmax
    }
    __syncthreads();

    // --- Step 4: In-place Softmax in Shared Memory ---
    // 4.1: Find max value for numerical stability (parallel reduction)
    float max_val = -INFINITY;
    for (int t = tid; t < softmax_len; t += blockDim.x) {
        max_val = fmaxf(max_val, s_att[t]);
    }
    s_reduce[tid] = max_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + offset]);
        __syncthreads();
    }
    max_val = s_reduce[0];
    __syncthreads();

    // 4.2: Compute exp(score - max) and sum the results (parallel reduction)
    float sum_val = 0.0f;
    for (int t = tid; t < softmax_len; t += blockDim.x) {
        float val = expf(s_att[t] - max_val);
        s_att[t] = val; // Store intermediate result back to shared memory
        sum_val += val;
    }
    s_reduce[tid] = sum_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) s_reduce[tid] += s_reduce[tid + offset];
        __syncthreads();
    }
    sum_val = s_reduce[0];
    __syncthreads();

    // 4.3: Normalize to get final attention weights
    const float inv_sum_val = 1.0f / (sum_val + 1e-9f); // Add epsilon for safety
    for (int t = tid; t < softmax_len; t += blockDim.x) {
        s_att[t] *= inv_sum_val;
    }
    __syncthreads();

    // --- Step 5: Weighted Sum of Values (Att * V) ---
    const __hip_bfloat16* v_cache_layer_ptr = value_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;

    // Each thread computes one dimension of the final output vector.
    if (tid < head_dim) {
        float weighted_sum = 0.0f;
        // The sum only includes actual tokens, not sinks.
        for (int t = 0; t <= pos; t++) {
            const __hip_bfloat16* v_vec = v_cache_layer_ptr + 1LL * t * kv_dim + 1LL * kv_head * head_dim;
            weighted_sum += s_att[t] * __bfloat162float(v_vec[tid]);
        }
        output_ptr[tid] = weighted_sum;
    }
}

