#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "matmul.hpp"

__global__ void attention_scores_shared_mem_kernel(
    float *att, const float *q, const float *key_cache,
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
    if (thread_in_block < 32)
    {
        q_shared[thread_in_block] = q_head_global[thread_in_block];
        q_shared[thread_in_block + 32] = q_head_global[thread_in_block + 32];
    }

    // Đợi tất cả thread trong block tải xong
    __syncthreads();

    // --- Các bước còn lại gần như giữ nguyên ---
    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;
    int pos = positions[batch_idx];
    if (t > pos)
        return;

    int kv_dim = head_dim * (n_heads / 8);
    int kv_head = head_idx / 8;
    const float *k_head = key_cache +
                                   1LL * batch_idx * n_layers * seq_len * kv_dim +
                                   1LL * layer_idx * seq_len * kv_dim +
                                   1LL * t * kv_dim +
                                   kv_head * head_dim;

    // Thay vì dùng q_ptr, ta dùng con trỏ tới shared memory
    const float4 *q_ptr_shared = reinterpret_cast<const float4 *>(q_shared);

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < head_dim / 4; i++)
    {
        float4 q_vec = q_ptr_shared[i];

        float k_vals[4];
        k_vals[0] = (k_head[i * 4 + 0]);
        k_vals[1] = (k_head[i * 4 + 1]);
        k_vals[2] = (k_head[i * 4 + 2]);
        k_vals[3] = (k_head[i * 4 + 3]);

        s0 += (double)q_vec.x * k_vals[0];
        s1 += (double)q_vec.y * k_vals[1];
        s2 += (double)q_vec.z * k_vals[2];
        s3 += (double)q_vec.w * k_vals[3];
    }

    double score = (s0 + s1) + (s2 + s3);
    score /= 8.0;
    if (use_sliding_window && layer_idx % 2 == 0)
    {
        score += mask[pos * seq_len + t];
    }
    att[1LL * batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
}

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

    const float *q_head = q + 1LL * batch_idx * n_heads * head_dim + head_idx * head_dim;
    const float *k_head = key_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim +
                                   1LL * layer_idx * seq_len * kv_dim + 1LL * t * kv_dim + kv_head * head_dim;

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

    att[1LL * batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
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

    const float *att_head = att + 1LL * batch_idx * n_heads * seq_len + 1LL * head_idx * seq_len;
    float *out_head = output + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;

    double sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const float *v_head = value_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim +
                                       1LL * layer_idx * seq_len * kv_dim + 1LL * t * kv_dim + kv_head * head_dim;
        // Convert BF16 value to FP32 for computation
        float v_val = (v_head[dim_idx]);
        sum += (double)att_head[t] * v_val;
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
__global__ void fused_attention_kernel_old(
    float *output,                     // Output: [batch, n_heads, head_dim]
    const float *q,                    // Input Q: [batch, n_heads * head_dim]
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
    float *s_q = s_data;                        // For the query vector of the current head. Size: head_dim
    float *s_att = (float *)&s_q[head_dim];     // For attention scores. Size: seq_len
    float *s_reduce = (float *)&s_att[seq_len]; // For block-wide reductions in softmax. Size: blockDim.x

    // --- Thread & Block Identification ---
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Early exit for padding blocks in the grid
    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    const int pos = positions[batch_idx];

    // --- GQA (Grouped-Query Attention) Setup ---
    const int gqa_ratio = n_heads / n_kv_heads;
    const int kv_head = head_idx / gqa_ratio;
    const int kv_dim = head_dim * n_kv_heads;

    // --- Set up Global Memory Pointers ---
    const float *q_head_ptr = q + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;
    const __hip_bfloat16 *k_cache_layer_ptr = key_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    float *output_ptr = output + 1LL * batch_idx * n_heads * head_dim + 1LL * head_idx * head_dim;

    // --- Step 1: Load Query (Q) into Shared Memory ---
    // All threads in the block cooperate to load the query vector.
    for (int i = tid; i < head_dim; i += blockDim.x)
    {
        s_q[i] = q_head_ptr[i];
    }
    __syncthreads();

    // --- Step 2: Calculate Attention Scores (Q * K^T) ---
    const float inv_sqrt_head_dim = rsqrtf((float)head_dim);
    // Parallelize the score calculation over the sequence length `t`.
    for (int t = tid; t <= pos; t += blockDim.x)
    {
        const __hip_bfloat16 *k_vec = k_cache_layer_ptr + 1LL * t * kv_dim + 1LL * kv_head * head_dim;
        float score = 0.0f;

        // Dot product between shared Q and global K
        for (int i = 0; i < head_dim; i++)
        {
            score += s_q[i] * __bfloat162float(k_vec[i]);
        }
        score *= inv_sqrt_head_dim;

        // Optionally apply sliding window mask
        if (use_sliding_window && (layer_idx % 2 == 0))
        {
            score += mask[1LL * pos * seq_len + t];
        }
        s_att[t] = score;
    }
    __syncthreads();

    // --- Step 3: Add Sinks ---
    int softmax_len = pos + 1;
    if (pos + 1 < seq_len)
    {
        if (tid == 0)
        { // Only one thread needs to write the sink value
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
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + offset]);
        __syncthreads();
    }
    max_val = s_reduce[0];
    __syncthreads();

    // 4.2: Compute exp(score - max) and sum the results (parallel reduction)
    float sum_val = 0.0f;
    for (int t = tid; t < softmax_len; t += blockDim.x)
    {
        float val = expf(s_att[t] - max_val);
        s_att[t] = val; // Store intermediate result back to shared memory
        sum_val += val;
    }
    s_reduce[tid] = sum_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            s_reduce[tid] += s_reduce[tid + offset];
        __syncthreads();
    }
    sum_val = s_reduce[0];
    __syncthreads();

    // 4.3: Normalize to get final attention weights
    const float inv_sum_val = 1.0f / (sum_val + 1e-9f); // Add epsilon for safety
    for (int t = tid; t < softmax_len; t += blockDim.x)
    {
        s_att[t] *= inv_sum_val;
    }
    __syncthreads();

    // --- Step 5: Weighted Sum of Values (Att * V) ---
    const __hip_bfloat16 *v_cache_layer_ptr = value_cache + 1LL * batch_idx * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;

    // Each thread computes one dimension of the final output vector.
    if (tid < head_dim)
    {
        float weighted_sum = 0.0f;
        // The sum only includes actual tokens, not sinks.
        for (int t = 0; t <= pos; t++)
        {
            const __hip_bfloat16 *v_vec = v_cache_layer_ptr + 1LL * t * kv_dim + 1LL * kv_head * head_dim;
            weighted_sum += s_att[t] * __bfloat162float(v_vec[tid]);
        }
        output_ptr[tid] = weighted_sum;
    }
}
// --- HELPER STRUCTS AND FUNCTIONS FOR EFFICIENT REDUCTIONS (FROM sgemv.cpp) ---

// --- MODIFIED FUSED ATTENTION KERNEL (2-D mapping: lanes x warps) ---
#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif

// ===== warp/block reductions (HIP-safe) =====
__device__ inline float warpReduceMax(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down(v, off));
    return v;
}
__device__ inline float warpReduceSum(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}
__device__ inline float blockReduceMax(float v, float *smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1); // warp id
    v = warpReduceMax(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = smem[lane];
    __syncthreads();
    out = warpReduceMax(out);
    if (lane == 0 && wid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}
__device__ inline float blockReduceSum(float v, float *smem) {
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> (__ffs(warpSize) - 1);
    v = warpReduceSum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) out = smem[lane];
    __syncthreads();
    out = warpReduceSum(out);
    if (lane == 0 && wid == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}

__global__ void fused_attention_kernel(
    float * __restrict__ output,               // [batch, n_heads, head_dim]
    const float * __restrict__ q,              // [batch, n_heads * head_dim]
    const float * __restrict__ key_cache,   // [batch, n_layers, seq_len, kv_dim]
    const float * __restrict__ value_cache, // [batch, n_layers, seq_len, kv_dim]
    const __hip_bfloat16 * __restrict__ sinks,       // [n_heads] (already layer-offset on host)
    const float * __restrict__ mask,                 // [seq_len, seq_len]
    const int * __restrict__ positions,              // [batch]
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window)
{
    extern __shared__ float s_sh[];  // layout later

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    if (b >= batch_size || h >= n_heads) return;

    // Wave bookkeeping
    const int WARP  = warpSize;                            // 64 on AMD, 32 on NV
    const int WARPS = (blockDim.x + WARP - 1) / WARP;
    const int lane  = tid & (WARP - 1);
    const int wid   = tid >> (__ffs(WARP) - 1);

    // Pointers / shapes
    const int pos        = positions[b];
    const int gqa_ratio  = n_heads / n_kv_heads;
    const int kv_h       = h / gqa_ratio;
    const int kv_dim     = head_dim * n_kv_heads;

    const float *q_head = q + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;
    const float *k_base =
        key_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    const float *v_base =
        value_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    float *out_head =
        output + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;

    // Windowing + capacities
    const bool apply_window = use_sliding_window && ((layer_idx & 1) == 0);
    const int  win_start    = apply_window ? max(0, pos - (SW_WINDOW - 1)) : 0;
    const int  win_core_len = pos - win_start + 1;                  // tokens before sink
    const int  att_cap      = apply_window ? (SW_WINDOW + 1) : seq_len; // <= host reserve
    float *s_att      = s_sh;                                       // [att_cap]
    float *s_partials = s_att + att_cap;                            // [WARPS * head_dim]
    float *s_reduce   = s_partials + WARPS * head_dim;              // [WARPS]
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    // === Fast path: map one lane per head-dim element (designed for head_dim == warpSize) ===
    // Falls back correctly when head_dim < warpSize by masking lanes; for head_dim > warpSize,
    // add a lane-strided loop over 'i += WARP' (omitted here since you asked about 64).
    const float q_lane = (lane < head_dim) ? q_head[lane] : 0.0f;

    // ---- Pass 1: compute attention scores into s_att[0..win_core_len-1] (all warps participate)
    for (int w = wid; w < win_core_len; w += WARPS) {
        const int t = win_start + w;
        const float *k_vec = k_base + 1LL * t * kv_dim + 1LL * kv_h * head_dim;

        float prod = 0.f;
        if (lane < head_dim) {
            prod = (float)q_lane * (k_vec[lane]);
        }
        float dot = warpReduceSum(prod);  // 64-lane sum
        if (lane == 0) {
            float acc = (double)dot * inv_sqrt_d;
            if (apply_window) {
                acc += mask[1LL * pos * seq_len + t];
            }
            s_att[w] = acc;
        }
    }
    __syncthreads();

    // Append sink (if any) after the core window
    int softmax_len = win_core_len;
    if (pos + 1 < seq_len) {
        if (tid == 0) s_att[softmax_len] = __bfloat162float(sinks[h]);
        softmax_len += 1;
    }
    __syncthreads();

    // ---- Softmax over s_att[0..softmax_len-1]
    float tmax = -INFINITY;
    for (int i = tid; i < softmax_len; i += blockDim.x) tmax = fmaxf(tmax, s_att[i]);
    const float max_val = blockReduceMax(tmax, s_reduce);

    float tsum = 0.f;
    for (int i = tid; i < softmax_len; i += blockDim.x) {
        float v = expf(s_att[i] - max_val);
        s_att[i] = v;
        tsum += v;
    }
    const float sum_val = blockReduceSum(tsum, s_reduce);
    const float inv_sum = 1.f / (sum_val + 1e-9f);

    for (int i = tid; i < softmax_len; i += blockDim.x) s_att[i] *= inv_sum;
    __syncthreads();

    // ---- Pass 2: V-weighted sum over the core window (exclude optional sink)
    // Every warp accumulates a partial output vector for its token stripe.
    if (lane < head_dim) {
        float partial = 0.f;
        for (int w = wid; w < win_core_len; w += WARPS) {
            const int t = win_start + w;
            const float *v_vec = v_base + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
            partial += (double)s_att[w] * (v_vec[lane]);
        }
        s_partials[wid * head_dim + lane] = partial;
    }
    __syncthreads();

    // Cross-warp reduce the partial output vectors (one lane per dim)
    if (wid == 0 && lane < head_dim) {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += s_partials[w * head_dim + lane];
        out_head[lane] = acc;
    }
}


// ============================================================================
// FUSED "MEGA" KERNEL
// ============================================================================
// This single kernel performs the following sequence of operations:
// 1. RMS Normalization
// 2. QKV Matrix Multiplication
// 3. Add QKV Bias
// 4. Apply Rotary Position Embedding (RoPE) to Q and K
// 5. Update the KV Cache
// This fusion eliminates multiple kernel launches and intermediate global memory
// writes (to `t` and `qkv` buffers), significantly improving performance.
__global__ void fused_rmsnorm_qkv_rope_kvcache_kernel(
    // Outputs
    float *__restrict__ q_out,
    float *__restrict__ k_out,
    float *__restrict__ v_out,
    __hip_bfloat16 *__restrict__ key_cache,
    __hip_bfloat16 *__restrict__ value_cache,
    // Inputs
    const float *__restrict__ x_in,
    const __hip_bfloat16 *__restrict__ rms_w,
    const __hip_bfloat16 *__restrict__ w_qkv,
    const __hip_bfloat16 *__restrict__ b_qkv,
    const float *__restrict__ cos_vals,
    const float *__restrict__ sin_vals,
    const int *__restrict__ positions,
    // Config
    int hidden_dim,
    int qkv_dim,
    int q_dim,
    int k_dim,
    int v_dim,
    int head_dim,
    int n_attn_heads,
    int n_kv_heads,
    int n_layers,
    int layer_idx,
    int seq_len)
{
    // Each thread block processes one item in the batch
    int batch_idx = blockIdx.x;
    // Each thread calculates one output dimension (for Q, K, or V)
    int out_dim_idx = threadIdx.x;

    // Shared memory for RMSNorm and holding the input vector `x`
    extern __shared__ float s_mem[];
    float *s_x = s_mem;                    // size: hidden_dim
    float *s_rms_sum = &s_mem[hidden_dim]; // size: 1 (or blockDim.x for reduction)

    // --- 1. Load input `x` to shared memory and start RMSNorm ---
    const float *x = x_in + batch_idx * hidden_dim;

    // Parallel load into shared memory
    for (int i = out_dim_idx; i < hidden_dim; i += blockDim.x)
    {
        s_x[i] = x[i];
    }

    // Calculate sum of squares for RMSNorm in parallel
    float ss = 0.0f;
    for (int i = out_dim_idx; i < hidden_dim; i += blockDim.x)
    {
        ss += s_x[i] * s_x[i];
    }

    // Reduction for sum of squares
    // Note: Using a single shared memory location for simplicity. For larger blocks,
    // a parallel reduction tree in shared memory would be more efficient.
    if (threadIdx.x == 0)
        s_rms_sum[0] = 0.0f;
    __syncthreads();
    atomicAdd(s_rms_sum, ss);
    __syncthreads();

    // Finalize RMSNorm scaling factor
    if (threadIdx.x == 0)
    {
        float mean_ss = s_rms_sum[0] / hidden_dim;
        s_rms_sum[0] = 1.0f / sqrtf(mean_ss + 1e-5f);
    }
    __syncthreads();
    float rms_scale = s_rms_sum[0];

    // --- 2. Matmul, Bias, RoPE, and KV Cache Update ---
    // Each thread computes one element of the QKV output vector
    for (int i = out_dim_idx; i < qkv_dim; i += blockDim.x)
    {
        float val = 0.0f;
        // Perform dot product (Matmul)
        for (int j = 0; j < hidden_dim; ++j)
        {
            // Apply RMSNorm on the fly
            float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
            val += normalized_x * __bfloat162float(w_qkv[i * hidden_dim + j]);
        }

        // Add bias
        val += __bfloat162float(b_qkv[i]);

        // --- Split, Apply RoPE, and Write to Output ---
        int pos = positions[batch_idx];

        if (i < q_dim)
        { // This is a Q dimension
            int head_idx = i / head_dim;
            int h_dim_idx = i % head_dim;
            int half = head_dim / 2;

            if (h_dim_idx < half)
            {
                float q1 = val;
                // Fetch the corresponding q2 value for RoPE
                // This requires a temporary buffer or a more complex indexing scheme.
                // For simplicity here, we assume a way to get q2. A real implementation
                // might re-compute or use shared memory.
                // This part is complex to fuse perfectly without restructuring data.
                // We'll apply RoPE after a sync, reading from an intermediate register stage.
                // Let's calculate the other pair part.
                int i2 = i + half;
                float val2 = 0.0f;
                for (int j = 0; j < hidden_dim; ++j)
                {
                    float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
                    val2 += normalized_x * __bfloat162float(w_qkv[i2 * hidden_dim + j]);
                }
                val2 += __bfloat162float(b_qkv[i2]);

                float c = cos_vals[pos * half + h_dim_idx];
                float s = sin_vals[pos * half + h_dim_idx];

                q_out[batch_idx * q_dim + i] = q1 * c - val2 * s;
                q_out[batch_idx * q_dim + i2] = val2 * c + q1 * s;
            }
        }
        else if (i < q_dim + k_dim)
        { // This is a K dimension
            int k_idx = i - q_dim;
            int head_idx = k_idx / head_dim;
            int h_dim_idx = k_idx % head_dim;
            int half = head_dim / 2;

            if (h_dim_idx < half)
            {
                float k1 = val;
                int i2 = i + half;
                float val2 = 0.0f;
                for (int j = 0; j < hidden_dim; ++j)
                {
                    float normalized_x = __bfloat162float(rms_w[j]) * (s_x[j] * rms_scale);
                    val2 += normalized_x * __bfloat162float(w_qkv[i2 * hidden_dim + j]);
                }
                val2 += __bfloat162float(b_qkv[i2]);

                float c = cos_vals[pos * half + h_dim_idx];
                float s = sin_vals[pos * half + h_dim_idx];

                float final_k1 = k1 * c - val2 * s;
                float final_k2 = val2 * c + k1 * s;

                // Write to k_out
                k_out[batch_idx * k_dim + k_idx] = final_k1;
                k_out[batch_idx * k_dim + k_idx + half] = final_k2;

                // Write to key_cache
                size_t k_cache_idx1 = (size_t)batch_idx * n_layers * seq_len * k_dim +
                                      (size_t)layer_idx * seq_len * k_dim + (size_t)pos * k_dim + k_idx;
                size_t k_cache_idx2 = k_cache_idx1 + half;
                key_cache[k_cache_idx1] = __float2bfloat16(final_k1);
                key_cache[k_cache_idx2] = __float2bfloat16(final_k2);
            }
        }
        else
        { // This is a V dimension
            int v_idx = i - q_dim - k_dim;
            // Write to v_out
            v_out[batch_idx * v_dim + v_idx] = val;

            // Write to value_cache
            size_t v_cache_idx = (size_t)batch_idx * n_layers * seq_len * v_dim +
                                 (size_t)layer_idx * seq_len * v_dim + (size_t)pos * v_dim + v_idx;
            value_cache[v_cache_idx] = __float2bfloat16(val);
        }
    }
}

// ============================================================================
// START: FUSED KERNEL FOR OUTPUT PROJECTION
// This kernel replaces three separate operations: matmul, add_bias, and accumulate.
// It performs a matrix multiplication, adds a bias vector, and adds the result
// to the residual stream `x` in a single pass.
// ============================================================================
#define TILE_DIM 32 // Defines the size of the tiles processed by each thread block.
__global__ void fused_output_projection_kernel(
    float *__restrict__ x,                     // Residual input, and final output [M, N]
    const float *__restrict__ input,           // Input from attention layers [M, K]
    const __hip_bfloat16 *__restrict__ weight, // Projection weights [N, K]
    const __hip_bfloat16 *__restrict__ bias,   // Projection bias [N]
    int M,                                     // Batch size
    int K,                                     // Attention output dimension
    int N                                      // Hidden dimension
)
{
    // Shared memory for tiles. No padding is needed with this new approach.
    __shared__ float input_tile[TILE_DIM][TILE_DIM];
    __shared__ float weight_tile[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // 1. Load a tile of the input matrix into shared memory (coalesced).
        // This part remains unchanged and correct.
        int input_col = t * TILE_DIM + tx;
        if (row < M && input_col < K)
        {
            input_tile[ty][tx] = input[row * K + input_col];
        }
        else
        {
            input_tile[ty][tx] = 0.0f;
        }

        // 2. CORRECTED: Coalesced load of weight matrix WITH on-the-fly transpose.
        int weight_load_row = bx * TILE_DIM + ty;
        int weight_load_col = t * TILE_DIM + tx;
        if (weight_load_row < N && weight_load_col < K)
        {
            // The source access `weight[...][...]` is coalesced.
            // The destination `weight_tile[tx][ty]` stores the data in a transposed layout.
            weight_tile[tx][ty] = __bfloat162float(weight[weight_load_row * K + weight_load_col]);
        }
        else
        {
            weight_tile[tx][ty] = 0.0f;
        }

        __syncthreads();

        // 3. CORRECTED & EFFICIENT: Multiply tiles from shared memory.
        for (int k = 0; k < TILE_DIM; ++k)
        {
            // This now performs the correct dot product: input[row] dot W[col].
            // The access to `weight_tile[k][tx]` is a conflict-free row-wise read
            // because of the on-the-fly transpose during loading.
            Cvalue += input_tile[ty][k] * weight_tile[k][tx];
        }

        __syncthreads();
    }

    // Fusion step remains the same.
    if (row < M && col < N)
    {
        Cvalue += __bfloat162float(bias[col]);
        Cvalue += x[row * N + col];
        x[row * N + col] = Cvalue;
    }
}



#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// =================================================================================================
// ## MFMA Configuration and Helpers
//
// This section defines the constants and helper functions necessary for the MFMA-based kernel.
// The configuration is tuned for a block size of 64x64, processed by 16 wavefronts.
// =================================================================================================

// --- Store and Fuse Results ---

// Stores the accumulator tile back to global memory and performs the fusion steps.
template<bool Interior>
__device__ inline void store_and_fuse_tile(
    float* __restrict__ x, // In/Out buffer
    const float* __restrict__ bias,
    const f32x4& acc,
    int M, int N,
    int m0, int n0,
    int wave_m, int wave_n,
    int lane)
{
    const int rowBase = m0 + wave_m * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n * WN + lane_row(lane);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if constexpr (!Interior) {
            if (row >= M || col >= N) continue;
        }

        // Fused operations: add bias and residual
        float Cvalue = acc[i];
        Cvalue += (bias[col]);
        Cvalue += x[(size_t)row * N + col];
        x[(size_t)row * N + col] = Cvalue;
    }
}

// =================================================================================================
// ## Optimized Fused Kernel (MFMA)
//
// This kernel calculates: x = (input @ weight^T) + bias + x
// - Uses MFMA instructions for the matmul.
// - Employs double-buffering in shared memory to hide data-loading latency.
// - Integrates bias and residual addition in the final store operation.
// =================================================================================================
__global__ void fused_output_projection_kernel_optimized(
    float *__restrict__ x,                     // Residual input [M,N], and final output [M,N]
    const float *__restrict__ input,           // Input from attention layers [M, K]
    const float *__restrict__ weight, // Projection weights [N, K]
    const float *__restrict__ bias,   // Projection bias [N]
    int M,                                     // Batch size
    int K,                                     // Attention output dimension
    int N                                      // Hidden dimension
) {
    // --- Block and Thread Identification ---
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x; // 0..63
    const int wave   = threadIdx.y; // 0..15
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;
    
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    // --- Shared Memory for Double-Buffered Tiles ---
    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    // --- Per-lane Accumulators ---
    f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // --- Pre-load first k-slice into shared memory ---
    {
        // Load input (A) tile: FP32 -> BF16, store row-major
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            float val = (gm < M && gk < K) ? input[(size_t)gm * K + gk] : 0.0f;
            sA0[idx] = f32_to_bf16_bits(val);
        }
        // Load weight (B) tile: BF16, store transposed (column-major)
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K; // Column in block (0..BLOCK_N-1)
            const int r = idx % BLOCK_K; // Row in block (0..BLOCK_K-1)
            const int gn = n0 + c;
            const int gk = r;
            float val = (gk < K && gn < N) ? weight[(size_t)gn * K + gk] : 0.0f;
            sB0[c * BLOCK_K + r] = f32_to_bf16_bits(val);
        }
    }
    __syncthreads();

    // --- Main Loop: Pipe-lined computation and data loading ---
    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Pre-fetch next tiles while computing on current tiles
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            // Load next input (A) tile
            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                float val = (gm < M && gk < K) ? input[(size_t)gm * K + gk] : 0.0f;
                nextA[idx] = f32_to_bf16_bits(val);
            }
            // Load next weight (B) tile
            for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K;
                const int r = idx % BLOCK_K;
                const int gn = n0 + c;
                const int gk = kBase + r;
                float val = (gk < K && gn < N) ? weight[(size_t)gn * K + gk] : 0.0f;
                nextB[c * BLOCK_K + r] = f32_to_bf16_bits(val);
            }
        }

        // --- MFMA Computation ---
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;
        bf16x4 avec = make_a_vec(currA, BLOCK_K, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, BLOCK_K, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();

        // Swap shared memory buffers for next iteration
        auto* tmpA = currA; currA = nextA; nextA = tmpA;
        auto* tmpB = currB; currB = nextB; nextB = tmpB;
    }

    // --- Store Results and Fuse Operations ---
    const bool interior = (m0 + BLOCK_M <= M && n0 + BLOCK_N <= N);
    if (interior) {
        store_and_fuse_tile<true>(x, bias, acc, M, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_and_fuse_tile<false>(x, bias, acc, M, N, m0, n0, wave_m, wave_n, lane);
    }
}


// =================================================================================================
// ## Updated Kernel Launcher
// =================================================================================================
extern "C" void launch_output_projection_optimized_kernel(
    const float* input1, const float* input2, float* output,
    int size1, int size2, int size3, hipStream_t stream = 0)
{
    int M = size1;
    int N = size2;
    int K = size3;
    
    const float* weight = reinterpret_cast<const float*>(input2);
    // Bias is located immediately after the weight matrix in memory
    const float* bias = weight + (size_t)N * K;
    
    // --- MFMA Launch Configuration ---
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 blockDim(LANE_PER_WAVE, WAVES_PER_BLOCK);
    
    // Shared memory: 2 buffers for A [M,K] tiles, 2 for B [K,N] tiles
    size_t shared_mem_bytes = (2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);
    
    // The `output` buffer serves as both input (for residual) and output
    hipLaunchKernelGGL(fused_output_projection_kernel_optimized, 
                       gridDim, 
                       blockDim, 
                       shared_mem_bytes, 
                       stream,
                       output, // `x` in the kernel
                       input1, // `input` in the kernel
                       weight, 
                       bias, 
                       M, K, N);
}
