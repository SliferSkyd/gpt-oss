// TODO: Modify this file to optimize end-to-end throughput with HIP GPU acceleration

#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include "../include/utils.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#ifndef GETP_RUN
#define GETP_RUN

// BATCH_SIZE can be increased for higher throughput
#define BATCH_SIZE 32
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

// Define precision mode: 0=FP32, 1=FP16, 2=BF16
#ifndef PRECISION_MODE
#define PRECISION_MODE 2  // Default to BF16
#endif

// Type definitions based on precision mode
#if PRECISION_MODE == 1
    typedef __half compute_t;
    typedef __half2 compute2_t;
    #define COMPUTE_EPSILON 1e-4f
#elif PRECISION_MODE == 2
    typedef __hip_bfloat16 compute_t;
    typedef __hip_bfloat162 compute2_t;
    #define COMPUTE_EPSILON 1e-4f
#else
    typedef float compute_t;
    typedef float2 compute2_t;
    #define COMPUTE_EPSILON 1e-5f
#endif

// CPU variables (used only in warmup)
float *cos_vals_cpu, *sin_vals_cpu;
int **prompt_tokens;
int *current_tokens;
bool *finished;
int *positions;
int *prompt_lens;

// GPU variables - using compute_t for reduced precision
compute_t *d_cos_vals, *d_sin_vals;
compute_t *d_x, *d_t, *d_tb, *d_tb2, *d_qkv, *d_q, *d_k, *d_v;
compute_t *d_key_cache, *d_value_cache, *d_att;
float *d_logits;  // Keep logits in FP32 for accuracy
compute_t *d_router_score, *d_topk_v, *d_mlp1_out, *d_gate, *d_up, *d_gate_up, *d_e_agg;
compute_t *d_mask, *d_expert_input_buffer;
int *d_topk_i, *d_current_tokens, *d_positions;
compute_t *d_temp_buffer;
compute_t *d_token_embedding_table;

// GPU weight pointers - using compute_t
compute_t *d_rms_attn_w, *d_rms_ffn_w, *d_rms_out_w;
compute_t *d_w_qkv, *d_b_qkv, *d_w_o, *d_b_o;
compute_t *d_w_router, *d_b_router;
compute_t *d_w_mlp1, *d_b_mlp1, *d_w_mlp2, *d_b_mlp2;
compute_t *d_out_w;

// Conversion helpers
__device__ __forceinline__ float to_float(compute_t x) {
#if PRECISION_MODE == 1
    return __half2float(x);
#elif PRECISION_MODE == 2
    return __bfloat162float(x);
#else
    return x;
#endif
}

__device__ __forceinline__ compute_t from_float(float x) {
#if PRECISION_MODE == 1
    return __float2half(x);
#elif PRECISION_MODE == 2
    return __float2bfloat16(x);
#else
    return x;
#endif
}

__host__ compute_t host_from_float(float x) {
#if PRECISION_MODE == 1
    return __float2half(x);
#elif PRECISION_MODE == 2  
    return __float2bfloat16(x);
#else
    return x;
#endif
}

// HIP error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU warmup functions (same as before)
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                             float scaling_factor,
                                             float initial_context_length,
                                             float ntk_beta, float ntk_alpha,
                                             float *concentration_out,
                                             float *inv_freq_out) {
    int d_half = head_dim / 2;
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f) {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;
        float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
        float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
        assert(0 < low && low < high && high < d_half - 1);

        for (int i = 0; i < d_half; i++) {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];
            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;
            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    } else {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++) {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }
    *concentration_out = concentration;
    free(freq);
}

void compute_cos_sin_getp(int pos, float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha, float *cos_out, float *sin_out) {
    int d_half = head_dim / 2;
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                         initial_context_length, ntk_beta,
                                         ntk_alpha, &concentration, inv_freq);

    for (int j = 0; j < d_half; j++) {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }
    free(inv_freq);
}

// GPU kernels
__global__ void rmsnorm_kernel(compute_t *output, const compute_t *input, const compute_t *weight,
                              int batch_size, int size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const compute_t *x = input + batch_idx * size;
    compute_t *o = output + batch_idx * size;
    
    // Shared memory for reduction - use float for accuracy
    __shared__ float shared_ss[THREADS_PER_BLOCK];
    
    // Calculate sum of squares
    float ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = to_float(x[i]);
        ss += val * val;
    }
    shared_ss[tid] = ss;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_ss[tid] += shared_ss[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        ss = shared_ss[0] / size;
        ss += COMPUTE_EPSILON;
        ss = 1.0f / sqrtf(ss);
        shared_ss[0] = ss;
    }
    __syncthreads();
    
    ss = shared_ss[0];
    
    // Normalize and scale
    for (int i = tid; i < size; i += blockDim.x) {
        o[i] = from_float(to_float(weight[i]) * (ss * to_float(x[i])));
    }
}

__global__ void softmax_kernel(compute_t *x, int batch_size, int size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    compute_t *batch_x = x + batch_idx * size;
    
    __shared__ float shared_max[THREADS_PER_BLOCK];
    __shared__ float shared_sum[THREADS_PER_BLOCK];
    
    // Find max value
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, to_float(batch_x[i]));
    }
    shared_max[tid] = max_val;
    __syncthreads();
    
    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = expf(to_float(batch_x[i]) - max_val);
        batch_x[i] = from_float(val);
        sum += val;
    }
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    sum = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < size; i += blockDim.x) {
        batch_x[i] = from_float(to_float(batch_x[i]) / sum);
    }
}

__global__ void matmul_kernel(compute_t *output, const compute_t *input, const compute_t *weight,
                             int batch_size, int input_dim, int output_dim) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    const compute_t *x = input + batch_idx * input_dim;
    compute_t *out = output + batch_idx * output_dim;
    
    __shared__ float shared_val[THREADS_PER_BLOCK];
    
    float val = 0.0f;
    for (int i = tid; i < input_dim; i += blockDim.x) {
        val += to_float(weight[out_idx * input_dim + i]) * to_float(x[i]);
    }
    shared_val[tid] = val;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_val[tid] += shared_val[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[out_idx] = from_float(shared_val[0]);
    }
}

__global__ void accumulate_kernel(compute_t *a, const compute_t *b, float factor,
                                 int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * size;
    
    if (idx < total_size) {
        a[idx] = from_float(to_float(a[idx]) + to_float(b[idx]) * factor);
    }
}

// NEW: Kernel to add bias to matrix multiplication result
__global__ void add_bias_kernel(compute_t *output, const compute_t *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size) {
        int dim_idx = idx % size;
        output[idx] = from_float(to_float(output[idx]) + to_float(bias[dim_idx]));
    }
}

__global__ void apply_rotary_emb_kernel(compute_t *x, const compute_t *cos_vals, const compute_t *sin_vals,
                                       const int *positions, int batch_size,
                                       int n_heads, int head_dim) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= n_heads) return;
    
    int half = head_dim / 2;
    if (dim_idx >= half) return;
    
    int pos = positions[batch_idx];
    compute_t *x_batch = x + batch_idx * n_heads * head_dim;
    const compute_t *cos_pos = cos_vals + pos * half;
    const compute_t *sin_pos = sin_vals + pos * half;
    
    float x1 = to_float(x_batch[head_idx * head_dim + dim_idx]);
    float x2 = to_float(x_batch[head_idx * head_dim + half + dim_idx]);
    
    float c = to_float(cos_pos[dim_idx]);
    float s = to_float(sin_pos[dim_idx]);
    
    x_batch[head_idx * head_dim + dim_idx] = from_float(x1 * c - x2 * s);
    x_batch[head_idx * head_dim + half + dim_idx] = from_float(x2 * c + x1 * s);
}

// NEW: KV cache update kernel
__global__ void update_kv_cache_kernel(compute_t *key_cache, compute_t *value_cache,
                                      const compute_t *k, const compute_t *v, 
                                      const int *positions, int batch_size,
                                      int n_layers, int layer_idx, int seq_len,
                                      int kv_dim) {
    int batch_idx = blockIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || dim_idx >= kv_dim) return;
    
    int pos = positions[batch_idx];
    // CRITICAL FIX: Add comprehensive bounds checking
    if (pos < 0 || pos >= seq_len || layer_idx < 0 || layer_idx >= n_layers) return;
    
    // Calculate indices with bounds checking
    size_t k_cache_idx = (size_t)batch_idx * n_layers * seq_len * kv_dim + 
                        (size_t)layer_idx * seq_len * kv_dim + 
                        (size_t)pos * kv_dim + dim_idx;
    size_t v_cache_idx = k_cache_idx; // Same calculation for value cache
    size_t input_idx = (size_t)batch_idx * kv_dim + dim_idx;
    
    // Update key cache
    key_cache[k_cache_idx] = k[input_idx];
    
    // Update value cache  
    value_cache[v_cache_idx] = v[input_idx];
}

__global__ void attention_scores_kernel(compute_t *att, const compute_t *q, const compute_t *key_cache,
                                       const compute_t *mask, const int *positions,
                                       int batch_size, int n_heads, int head_dim,
                                       int seq_len, int n_layers, int layer_idx,
                                       bool use_sliding_window) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch_idx >= batch_size || head_idx >= n_heads) return;
    
    int pos = positions[batch_idx];
    if (t > pos || t >= seq_len || pos < 0 || pos >= seq_len) return;
    
    // CRITICAL FIX: Add bounds checking for layer and head indices
    if (layer_idx < 0 || layer_idx >= n_layers) return;
    
    int kv_dim = head_dim * (n_heads / 8); // Assuming GQA with 4:1 ratio
    int kv_head = head_idx / 8;
    
    // CRITICAL FIX: Add bounds checking for kv_head
    if (kv_head * head_dim + head_dim > kv_dim) return;
    
    const compute_t *q_head = q + (size_t)batch_idx * n_heads * head_dim + (size_t)head_idx * head_dim;
    const compute_t *k_head = key_cache + (size_t)batch_idx * n_layers * seq_len * kv_dim +
                         (size_t)layer_idx * seq_len * kv_dim + (size_t)t * kv_dim + (size_t)kv_head * head_dim;
    
    float score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        score += to_float(q_head[i]) * to_float(k_head[i]);
    }
    score /= sqrtf((float)head_dim);
    
    // Apply sliding window mask if enabled
    if (use_sliding_window && mask) {
        size_t mask_idx = (size_t)pos * seq_len + t;
        score += to_float(mask[mask_idx]);
    }
    
    size_t att_idx = (size_t)batch_idx * n_heads * seq_len + (size_t)head_idx * seq_len + t;
    att[att_idx] = from_float(score);
}

__global__ void attention_weighted_sum_kernel(compute_t *output, const compute_t *att,
                                             const compute_t *value_cache, const int *positions,
                                             int batch_size, int n_heads, int head_dim,
                                             int seq_len, int n_layers, int layer_idx) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= n_heads || dim_idx >= head_dim) return;
    
    int pos = positions[batch_idx];
    int kv_dim = head_dim * (n_heads / 8);
    int kv_head = head_idx / 8;
    // int kv_head = head_idx / 4;
    
    const compute_t *att_head = att + batch_idx * n_heads * seq_len + head_idx * seq_len;
    compute_t *out_head = output + batch_idx * n_heads * head_dim + head_idx * head_dim;
    
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const compute_t *v_head = value_cache + batch_idx * n_layers * seq_len * kv_dim +
                             layer_idx * seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
        sum += to_float(att_head[t]) * to_float(v_head[dim_idx]);
    }
    out_head[dim_idx] = from_float(sum);
}

__global__ void swiglu_kernel(compute_t *gate, compute_t *up, compute_t *output,
                             int batch_size, int intermediate_dim, float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / intermediate_dim;
    int dim_idx = idx % intermediate_dim;
    
    if (batch_idx >= batch_size || dim_idx >= intermediate_dim) return;
    
    const float alpha = 1.702f;
    float gate_val = to_float(gate[idx]);
    float up_val = to_float(up[idx]);
    
    // Clamping
    gate_val = fminf(fmaxf(gate_val, -limit), limit);
    up_val = fminf(fmaxf(up_val, -limit), limit);
    
    // SiLU activation
    gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));
    gate_val *= (up_val + 1.0f);
    
    output[idx] = from_float(gate_val);
}

__global__ void split_gate_up_kernel(compute_t *gate, compute_t *up, const compute_t *mlp1_out,
                                     const compute_t *bias, int batch_size, int intermediate_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / intermediate_dim;
    int dim_idx = idx % intermediate_dim;
    
    if (batch_idx >= batch_size || dim_idx >= intermediate_dim) return;
    
    int mlp1_idx = batch_idx * 2 * intermediate_dim;
    gate[idx] = from_float(to_float(mlp1_out[mlp1_idx + 2 * dim_idx]) + to_float(bias[2 * dim_idx]));
    up[idx] = from_float(to_float(mlp1_out[mlp1_idx + 2 * dim_idx + 1]) + to_float(bias[2 * dim_idx + 1]));
}

__global__ void topk_kernel(compute_t *topk_values, int *topk_indices, const compute_t *scores,
                           int batch_size, int n_experts, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const compute_t *batch_scores = scores + batch_idx * n_experts;
    compute_t *batch_topk_v = topk_values + batch_idx * k;
    int *batch_topk_i = topk_indices + batch_idx * k;
    
    // Simple selection sort for top-k (works well for small k)
    for (int i = 0; i < k; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int j = 0; j < n_experts; j++) {
            bool already_selected = false;
            for (int prev = 0; prev < i; prev++) {
                if (batch_topk_i[prev] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && to_float(batch_scores[j]) > max_val) {
                max_val = to_float(batch_scores[j]);
                max_idx = j;
            }
        }
        
        batch_topk_v[i] = from_float(max_val);
        batch_topk_i[i] = max_idx;
    }
}

__global__ void copy_embeddings_kernel(compute_t *output, const compute_t *embeddings,
                                      const int *tokens, int batch_size, int hidden_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_dim;
    int dim_idx = idx % hidden_dim;
    
    if (batch_idx >= batch_size || dim_idx >= hidden_dim) return;
    
    int token = tokens[batch_idx];
    // CRITICAL FIX: Add bounds checking for token values
    if (token < 0 || token >= vocab_size) {
        // Set to zero for invalid tokens instead of crashing
        output[idx] = from_float(0.0f);
        return;
    }
    
    output[idx] = embeddings[token * hidden_dim + dim_idx];
}

// Add bounds checking to matmul kernel
__global__ void matmul_kernel_safe(compute_t *output, const compute_t *input, const compute_t *weight,
                                  int batch_size, int input_dim, int output_dim) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    const compute_t *x = input + batch_idx * input_dim;
    compute_t *out = output + batch_idx * output_dim;
    
    __shared__ float shared_val[THREADS_PER_BLOCK];
    
    float val = 0.0f;
    for (int i = tid; i < input_dim; i += blockDim.x) {
        val += to_float(weight[out_idx * input_dim + i]) * to_float(x[i]);
    }
    
    if (tid < THREADS_PER_BLOCK) {
        shared_val[tid] = val;
    }
    __syncthreads();
    
    // Reduction with bounds checking
    for (int stride = min(blockDim.x, THREADS_PER_BLOCK) / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < THREADS_PER_BLOCK) {
            shared_val[tid] += shared_val[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[out_idx] = from_float(shared_val[0]);
    }
}

// Memory allocation functions
void malloc_gpu_run_state(RunState *s, Config *p) {
    int kv_dim = p->head_dim * p->n_kv_heads;
    
    // Initialize pointers to NULL
    d_mask = NULL;
    
    // FIXED: Correct memory size calculations
    size_t batch_hidden_elements = BATCH_SIZE * p->hidden_dim;
    size_t batch_qkv_elements = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads);
    size_t kv_cache_elements = BATCH_SIZE * p->n_layers * p->seq_len * kv_dim;
    
    printf("Allocating GPU memory: batch_size=%d, hidden_dim=%d, seq_len=%d\n", 
           BATCH_SIZE, p->hidden_dim, p->seq_len);
    printf("KV cache size per batch: %zu MB\n", (kv_cache_elements * sizeof(compute_t)) / (1024 * 1024));
    
    // Check for potential overflow in size calculations
    if (kv_cache_elements > SIZE_MAX / sizeof(compute_t)) {
        // fprintf(stderr, "Error: KV cache size calculation overflow\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate GPU memory with corrected size calculations
    HIP_CHECK(hipMalloc((void**)&d_x, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_t, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_tb, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_tb2, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_qkv, batch_qkv_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_q, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_k, BATCH_SIZE * kv_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_v, BATCH_SIZE * kv_dim * sizeof(compute_t)));
    
    // KV cache allocation - this is usually the largest allocation
    HIP_CHECK(hipMalloc((void**)&d_key_cache, kv_cache_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_value_cache, kv_cache_elements * sizeof(compute_t)));
    
    HIP_CHECK(hipMalloc((void**)&d_att, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_logits, BATCH_SIZE * p->vocab_size * sizeof(float))); // Keep FP32 for logits
    HIP_CHECK(hipMalloc((void**)&d_router_score, BATCH_SIZE * p->n_experts * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_topk_v, BATCH_SIZE * p->experts_per_token * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_topk_i, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&d_mlp1_out, BATCH_SIZE * 2 * p->intermediate_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_gate, BATCH_SIZE * p->intermediate_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_up, BATCH_SIZE * p->intermediate_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_gate_up, BATCH_SIZE * p->intermediate_dim * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_e_agg, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&d_positions, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&d_cos_vals, (p->head_dim / 2) * p->seq_len * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_sin_vals, (p->head_dim / 2) * p->seq_len * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_expert_input_buffer, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_temp_buffer, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMalloc((void**)&d_token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(compute_t)));

    // Weight allocations are now handled in copy_weights_to_gpu function
    
    // Initialize all allocated memory to zero with corrected sizes
    HIP_CHECK(hipMemset(d_x, 0, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_t, 0, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_tb, 0, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_tb2, 0, batch_hidden_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_qkv, 0, batch_qkv_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_q, 0, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_k, 0, BATCH_SIZE * kv_dim * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_v, 0, BATCH_SIZE * kv_dim * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_key_cache, 0, kv_cache_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_value_cache, 0, kv_cache_elements * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_att, 0, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(compute_t)));
    HIP_CHECK(hipMemset(d_logits, 0, BATCH_SIZE * p->vocab_size * sizeof(float)));

    if (p->sliding_window > 0) {
        size_t mask_size = p->seq_len * p->seq_len * sizeof(compute_t);
        HIP_CHECK(hipMalloc((void**)&d_mask, mask_size));
        
        // Initialize mask on GPU if needed
        compute_t *h_mask = (compute_t*)malloc(mask_size);
        if (!h_mask) {
            // fprintf(stderr, "Failed to allocate host memory for mask\n");
            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < p->seq_len; i++) {
            for (int j = 0; j < p->seq_len; j++) {
                h_mask[i * p->seq_len + j] = host_from_float((i - j >= p->sliding_window) ? -INFINITY : 0.0f);
            }
        }
        HIP_CHECK(hipMemcpy(d_mask, h_mask, mask_size, hipMemcpyHostToDevice));
        free(h_mask);
    }
    
    printf("GPU memory allocation completed successfully\n");
}

// Kernel to convert FP32 to compute_t on GPU
__global__ void convert_fp32_to_compute_kernel(compute_t *output, const float *input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = from_float(input[idx]);
    }
}

compute_t* copy_and_convert_weights(const float *h_src, size_t num_elements) {
    // Allocate GPU memory for weights in compute_t format
    compute_t *d_dest;
    HIP_CHECK(hipMalloc((void**)&d_dest, num_elements * sizeof(compute_t)));
    
#if PRECISION_MODE != 0
    // For FP16/BF16, convert on CPU first, then copy
    compute_t *h_converted = (compute_t*)malloc(num_elements * sizeof(compute_t));
    
    // Convert weights from FP32 to compute_t on CPU
    for (size_t i = 0; i < num_elements; i++) {
        h_converted[i] = host_from_float(h_src[i]);
    }
    
    // Copy converted weights to GPU
    HIP_CHECK(hipMemcpy(d_dest, h_converted, num_elements * sizeof(compute_t), hipMemcpyHostToDevice));
    
    free(h_converted);
#else
    // For FP32, direct copy (assuming compute_t is float when PRECISION_MODE == 0)
    HIP_CHECK(hipMemcpy(d_dest, h_src, num_elements * sizeof(float), hipMemcpyHostToDevice));
#endif
    
    return d_dest;
}

void copy_weights_to_gpu(Transformer *transformer) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    
    printf("Copying weights to GPU (Precision mode: %s)...\n", 
           PRECISION_MODE == 1 ? "FP16" : (PRECISION_MODE == 2 ? "BF16" : "FP32"));
    
    // Copy normalization weights - allocate and convert
    d_rms_attn_w = copy_and_convert_weights(w->rms_attn_w, p->n_layers * p->hidden_dim);
    d_rms_ffn_w = copy_and_convert_weights(w->rms_ffn_w, p->n_layers * p->hidden_dim);
    d_rms_out_w = copy_and_convert_weights(w->rms_out_w, p->hidden_dim);
    
    // Copy attention weights - allocate and convert
    int qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    d_w_qkv = copy_and_convert_weights(w->w_qkv, qkv_size);
    d_b_qkv = copy_and_convert_weights(w->b_qkv, p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim);
    
    int attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    d_w_o = copy_and_convert_weights(w->w_o, attn_out_size);
    d_b_o = copy_and_convert_weights(w->b_o, p->n_layers * p->hidden_dim);
    
    // Copy MoE weights - allocate and convert
    d_w_router = copy_and_convert_weights(w->w_router, p->n_layers * p->hidden_dim * p->n_experts);
    d_b_router = copy_and_convert_weights(w->b_router, p->n_layers * p->n_experts);
    
    int mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    d_w_mlp1 = copy_and_convert_weights(w->w_mlp1, mlp1_size);
    d_b_mlp1 = copy_and_convert_weights(w->b_mlp1, p->n_layers * p->n_experts * (2 * p->intermediate_dim));
    
    int mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    d_w_mlp2 = copy_and_convert_weights(w->w_mlp2, mlp2_size);
    d_b_mlp2 = copy_and_convert_weights(w->b_mlp2, p->n_layers * p->n_experts * p->hidden_dim);
    
    // Copy output weights - allocate and convert
    d_out_w = copy_and_convert_weights(w->out, p->hidden_dim * p->vocab_size);
    
    printf("Weight copying completed successfully\n");
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
    Config *p = &transformer->config;
    
    // CRITICAL FIX: Initialize HIP device first
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSynchronize());
    
    // Allocate GPU memory first
    malloc_gpu_run_state(&transformer->state, p);
    
    // CRITICAL FIX: Copy all weights to GPU BEFORE using them
    copy_weights_to_gpu(transformer);
    
    // Synchronize after weight copying
    HIP_CHECK(hipDeviceSynchronize());
    
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    
    // CPU allocation for RoPE values (used in warmup only)
    cos_vals_cpu = (float*)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));
    sin_vals_cpu = (float*)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));
    
    for (int pos = 0; pos < p->seq_len; ++pos) {
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                           p->initial_context_length, ntk_beta, ntk_alpha,
                           cos_vals_cpu + (pos * p->head_dim / 2),
                           sin_vals_cpu + (pos * p->head_dim / 2));
    }
    
    // Copy RoPE values to GPU (now that GPU memory is allocated)
    // These are special cases where we already have allocated GPU memory
#if PRECISION_MODE != 0
    // For FP16/BF16, convert on CPU first, then copy
    compute_t *h_cos_converted = (compute_t*)malloc((p->head_dim / 2) * p->seq_len * sizeof(compute_t));
    compute_t *h_sin_converted = (compute_t*)malloc((p->head_dim / 2) * p->seq_len * sizeof(compute_t));
    
    for (int i = 0; i < (p->head_dim / 2) * p->seq_len; i++) {
        h_cos_converted[i] = host_from_float(cos_vals_cpu[i]);
        h_sin_converted[i] = host_from_float(sin_vals_cpu[i]);
    }
    
    HIP_CHECK(hipMemcpy(d_cos_vals, h_cos_converted, (p->head_dim / 2) * p->seq_len * sizeof(compute_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sin_vals, h_sin_converted, (p->head_dim / 2) * p->seq_len * sizeof(compute_t), hipMemcpyHostToDevice));
    
    free(h_cos_converted);
    free(h_sin_converted);
    
    // Convert token embedding table
    compute_t *h_embed_converted = (compute_t*)malloc(p->vocab_size * p->hidden_dim * sizeof(compute_t));
    for (int i = 0; i < p->vocab_size * p->hidden_dim; i++) {
        h_embed_converted[i] = host_from_float(transformer->weights.token_embedding_table[i]);
    }
    HIP_CHECK(hipMemcpy(d_token_embedding_table, h_embed_converted, p->vocab_size * p->hidden_dim * sizeof(compute_t), hipMemcpyHostToDevice));
    free(h_embed_converted);
#else
    // For FP32, direct copy
    HIP_CHECK(hipMemcpy(d_cos_vals, cos_vals_cpu, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sin_vals, sin_vals_cpu, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_token_embedding_table, transformer->weights.token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(float), hipMemcpyHostToDevice));
#endif

    // CPU allocations for batch management
    prompt_tokens = (int**)malloc(BATCH_SIZE * sizeof(int*));
    current_tokens = (int*)malloc(BATCH_SIZE * sizeof(int));
    for (int b = 0; b < BATCH_SIZE; b++) {
        prompt_tokens[b] = (int*)malloc((p->seq_len + 3) * sizeof(int));
    }
    finished = (bool*)malloc(BATCH_SIZE * sizeof(bool));
    positions = (int*)malloc(BATCH_SIZE * sizeof(int));
    prompt_lens = (int*)malloc(BATCH_SIZE * sizeof(int));
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
    // Free GPU memory
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_t));
    HIP_CHECK(hipFree(d_tb));
    HIP_CHECK(hipFree(d_tb2));
    HIP_CHECK(hipFree(d_qkv));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_key_cache));
    HIP_CHECK(hipFree(d_value_cache));
    HIP_CHECK(hipFree(d_att));
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_router_score));
    HIP_CHECK(hipFree(d_topk_v));
    HIP_CHECK(hipFree(d_topk_i));
    HIP_CHECK(hipFree(d_mlp1_out));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_gate_up));
    HIP_CHECK(hipFree(d_e_agg));
    HIP_CHECK(hipFree(d_current_tokens));
    HIP_CHECK(hipFree(d_positions));
    HIP_CHECK(hipFree(d_cos_vals));
    HIP_CHECK(hipFree(d_sin_vals));
    HIP_CHECK(hipFree(d_expert_input_buffer));
    HIP_CHECK(hipFree(d_temp_buffer));
    HIP_CHECK(hipFree(d_token_embedding_table));
    if (d_mask) HIP_CHECK(hipFree(d_mask));
    
    // Free GPU weight memory
    HIP_CHECK(hipFree(d_rms_attn_w));
    HIP_CHECK(hipFree(d_rms_ffn_w));
    HIP_CHECK(hipFree(d_rms_out_w));
    HIP_CHECK(hipFree(d_w_qkv));
    HIP_CHECK(hipFree(d_b_qkv));
    HIP_CHECK(hipFree(d_w_o));
    HIP_CHECK(hipFree(d_b_o));
    HIP_CHECK(hipFree(d_w_router));
    HIP_CHECK(hipFree(d_b_router));
    HIP_CHECK(hipFree(d_w_mlp1));
    HIP_CHECK(hipFree(d_b_mlp1));
    HIP_CHECK(hipFree(d_w_mlp2));
    HIP_CHECK(hipFree(d_b_mlp2));
    HIP_CHECK(hipFree(d_out_w));
    
    // Free CPU memory
    free(cos_vals_cpu);
    free(sin_vals_cpu);
    for (int b = 0; b < BATCH_SIZE; b++) {
        free(prompt_tokens[b]);
    }
    free(prompt_tokens);
    free(current_tokens);
    free(finished);
    free(positions);
    free(prompt_lens);
}

// GPU-accelerated neural network functions
void attention_gpu(Transformer *transformer, int layer_idx, int batch_size) {
    Config *p = &transformer->config;
    
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    // fprintf(stderr, "Running attention layer %d with batch size %d\n", layer_idx, batch_size);
    // RMSNorm - FIXED: Use GPU weight pointer
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    rmsnorm_kernel<<<norm_grid, norm_block>>>(
        d_t, d_x, d_rms_attn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());

    // fprintf(stderr, "RMSNorm completed for layer %d\n", layer_idx);
    
    // QKV projection using safer matmul kernel - FIXED: Use GPU weight pointer
    int qkv_out_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
    dim3 matmul_grid(batch_size, (qkv_out_dim + 31) / 32);
    dim3 matmul_block(32, min(32, THREADS_PER_BLOCK / 32));
    int qkv_weight_offset = layer_idx * hidden_dim * qkv_out_dim;
    matmul_kernel_safe<<<matmul_grid, matmul_block>>>(
        d_qkv, d_t, d_w_qkv + qkv_weight_offset, batch_size, hidden_dim, qkv_out_dim);
    HIP_CHECK(hipGetLastError());

    // fprintf(stderr, "QKV projection completed for layer %d\n", layer_idx);
    
    // Add bias - FIXED: Use GPU bias pointer and proper kernel
    int qkv_bias_offset = layer_idx * qkv_out_dim;
    dim3 bias_grid((batch_size * qkv_out_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
        d_qkv, d_b_qkv + qkv_bias_offset, batch_size, qkv_out_dim);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "Bias addition completed for layer %d\n", layer_idx);


    // Copy Q, K, V from qkv buffer - SIMPLIFIED AND FIXED
    int q_size = p->n_attn_heads * head_dim;
    int k_size = p->n_kv_heads * head_dim;
    int v_size = p->n_kv_heads * head_dim;
    

    // fprintf(stderr, "Copying Q, K, V for layer %d\n", layer_idx);
    // Copy Q: shape [batch_size, n_attn_heads * head_dim]
    for (int b = 0; b < batch_size; b++) {
        compute_t *src = d_qkv + b * qkv_out_dim;
        compute_t *dst = d_q + b * q_size;
        HIP_CHECK(hipMemcpy(dst, src, q_size * sizeof(compute_t), hipMemcpyDeviceToDevice));
    }

    // fprintf(stderr, "Q copy completed for layer %d\n", layer_idx);
    
    // Copy K: shape [batch_size, n_kv_heads * head_dim] 
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < batch_size; b++) {
        compute_t *src = d_qkv + b * qkv_out_dim + k_offset;
        compute_t *dst = d_k + b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(compute_t), hipMemcpyDeviceToDevice));
    }

    // fprintf(stderr, "K copy completed for layer %d\n", layer_idx);

    
    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < batch_size; b++) {
        compute_t *src = d_qkv + b * qkv_out_dim + v_offset;
        compute_t *dst = d_v + b * v_size;
        HIP_CHECK(hipMemcpy(dst, src, v_size * sizeof(compute_t), hipMemcpyDeviceToDevice));
    }
    
    // fprintf(stderr, "V copy completed for layer %d\n", layer_idx);
    // Apply rotary embeddings
    dim3 rope_grid(batch_size, p->n_attn_heads);
    dim3 rope_block(head_dim / 2);
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        d_q, d_cos_vals, d_sin_vals, d_positions, batch_size, p->n_attn_heads, head_dim);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "Rotary embeddings applied for layer %d\n", layer_idx);
    rope_grid.y = p->n_kv_heads;
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        d_k, d_cos_vals, d_sin_vals, d_positions, batch_size, p->n_kv_heads, head_dim);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "Rotary embeddings applied to K for layer %d\n", layer_idx);
    // Update KV cache - NEW: Proper GPU kernel
    dim3 kv_grid(batch_size, (kv_dim + 31) / 32);
    dim3 kv_block(1, 32);
    update_kv_cache_kernel<<<kv_grid, kv_block>>>(
        d_key_cache, d_value_cache, d_k, d_v, d_positions, batch_size,
        p->n_layers, layer_idx, p->seq_len, kv_dim);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "KV cache updated for layer %d\n", layer_idx);
    // Compute attention scores
    dim3 att_grid(batch_size, p->n_attn_heads, (p->seq_len + 31) / 32);
    dim3 att_block(1, 1, 32);
    attention_scores_kernel<<<att_grid, att_block>>>(
        d_att, d_q, d_key_cache, d_mask, d_positions, batch_size, p->n_attn_heads,
        head_dim, p->seq_len, p->n_layers, layer_idx, p->sliding_window > 0);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "Attention scores computed for layer %d\n", layer_idx);
    // Softmax attention weights
    dim3 soft_grid(batch_size * p->n_attn_heads);
    softmax_kernel<<<soft_grid, norm_block>>>(d_att, batch_size * p->n_attn_heads, p->seq_len);
    HIP_CHECK(hipGetLastError());
    
    // fprintf(stderr, "Softmax applied to attention weights for layer %d\n", layer_idx);
    // Weighted sum of values
    dim3 wsum_grid(batch_size, p->n_attn_heads);
    dim3 wsum_block(head_dim);
    attention_weighted_sum_kernel<<<wsum_grid, wsum_block>>>(
        d_tb, d_att, d_value_cache, d_positions, batch_size, p->n_attn_heads,
        head_dim, p->seq_len, p->n_layers, layer_idx);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Weighted sum of values computed for layer %d\n", layer_idx);
    
    // Output projection - FIXED: Use GPU weight pointer
    dim3 out_grid(batch_size, (hidden_dim + 31) / 32);
    int attn_out_offset = layer_idx * (head_dim * p->n_attn_heads) * hidden_dim;
    matmul_kernel<<<out_grid, matmul_block>>>(
        d_tb2, d_tb, d_w_o + attn_out_offset, batch_size, head_dim * p->n_attn_heads, hidden_dim);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Output projection completed for layer %d\n", layer_idx);
    
    // Add bias and residual connection - FIXED: Use GPU bias pointer
    int attn_bias_offset = layer_idx * hidden_dim;
    add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
        d_tb2, d_b_o + attn_bias_offset, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Bias added to output for layer %d\n", layer_idx);
    
    accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        d_x, d_tb2, 1.0f, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Residual connection added for layer %d\n", layer_idx);
}

// NEW: Improved MoE implementation with better expert routing
// Mixed precision matmul kernel that outputs FP32 (for classifier)
__global__ void matmul_kernel_fp32_output(float *output, const compute_t *input, const compute_t *weight,
                                         int batch_size, int input_dim, int output_dim) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    const compute_t *x = input + batch_idx * input_dim;
    float *out = output + batch_idx * output_dim;
    
    __shared__ float shared_val[THREADS_PER_BLOCK];
    
    float val = 0.0f;
    for (int i = tid; i < input_dim; i += blockDim.x) {
        val += to_float(weight[out_idx * input_dim + i]) * to_float(x[i]);
    }
    shared_val[tid] = val;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_val[tid] += shared_val[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[out_idx] = shared_val[0];  // Output stays in FP32
    }
}

__global__ void expert_routing_kernel(compute_t *expert_weights, const int *topk_indices, 
                                     const compute_t *topk_values, int batch_size, 
                                     int experts_per_token, int n_experts) {
    int batch_idx = blockIdx.x;
    int expert_slot = blockIdx.y;
    
    if (batch_idx >= batch_size || expert_slot >= experts_per_token) return;
    
    int expert_id = topk_indices[batch_idx * experts_per_token + expert_slot];
    compute_t weight = topk_values[batch_idx * experts_per_token + expert_slot];
    
    expert_weights[batch_idx * n_experts + expert_id] = weight;
}

// NEW KERNEL for correct MoE aggregation
__global__ void aggregate_expert_output_kernel(compute_t* d_e_agg, const compute_t* expert_output,
                                             const int* topk_indices, const compute_t* topk_values,
                                             int current_expert_id, int batch_size, int hidden_dim,
                                             int experts_per_token) {
    // Each thread handles one dimension of one token in the batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_dim) return;

    int b = idx / hidden_dim; // Get the batch index for this thread

    // Check if the current expert (current_expert_id) was selected for this token (b)
    for (int k = 0; k < experts_per_token; ++k) {
        int expert_slot_idx = b * experts_per_token + k;
        if (topk_indices[expert_slot_idx] == current_expert_id) {
            // This token uses this expert. Add the weighted output to the aggregation buffer.
            float weight = to_float(topk_values[expert_slot_idx]);
            d_e_agg[idx] = from_float(to_float(d_e_agg[idx]) + weight * to_float(expert_output[idx]));
            
            // Since top-k indices are unique for a token, we can stop after finding the match
            break; 
        }
    }
}

void moe_gpu(Transformer *transformer, int layer_idx, int batch_size) {
    Config *p = &transformer->config;
    
    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    // fprintf(stderr, "Running MoE layer %d with batch size %d\n", layer_idx, batch_size);
    // FFN RMSNorm - CORRECT
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    rmsnorm_kernel<<<norm_grid, norm_block>>>(
        d_t, d_x, d_rms_ffn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "RMSNorm completed for MoE layer %d\n", layer_idx);
    
    // Router computation & bias - CORRECT
    int router_weight_offset = layer_idx * hidden_dim * n_experts;
    dim3 router_grid(batch_size, (n_experts + 31) / 32);
    dim3 router_block(32, 32);
    matmul_kernel<<<router_grid, router_block>>>(
        d_router_score, d_t, d_w_router + router_weight_offset, batch_size, hidden_dim, n_experts);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Router score computed for MoE layer %d\n", layer_idx);
    
    int router_bias_offset = layer_idx * n_experts;
    dim3 bias_grid((batch_size * n_experts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
        d_router_score, d_b_router + router_bias_offset, batch_size, n_experts);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Bias added to router score for MoE layer %d\n", layer_idx);
    
    // Top-k expert selection - CORRECT
    dim3 topk_grid(batch_size);
    topk_kernel<<<topk_grid, 1>>>(d_topk_v, d_topk_i, d_router_score, batch_size, n_experts, p->experts_per_token);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Top-k experts selected for MoE layer %d\n", layer_idx);
    
    // Softmax on top-k values to get weights - CORRECT
    dim3 topk_soft_grid(batch_size);
    softmax_kernel<<<topk_soft_grid, norm_block>>>(d_topk_v, batch_size, p->experts_per_token);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Softmax applied to top-k values for MoE layer %d\n", layer_idx);
    
    // Initialize expert aggregation buffer - CORRECT
    HIP_CHECK(hipMemset(d_e_agg, 0, batch_size * hidden_dim * sizeof(compute_t)));
    // fprintf(stderr, "Expert aggregation buffer initialized for MoE layer %d\n", layer_idx);
    
    // --- REWRITTEN EXPERT PROCESSING AND AGGREGATION ---
    dim3 expert_agg_grid((batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Process each expert and aggregate its output correctly
    for (int expert_id = 0; expert_id < n_experts; expert_id++) {
        // Calculate weight and bias offsets for this specific expert
        int mlp1_weight_offset = (layer_idx * n_experts + expert_id) * (2 * intermediate_dim) * hidden_dim;
        int mlp1_bias_offset = (layer_idx * n_experts + expert_id) * (2 * intermediate_dim);
        int mlp2_weight_offset = (layer_idx * n_experts + expert_id) * hidden_dim * intermediate_dim;
        int mlp2_bias_offset = (layer_idx * n_experts + expert_id) * hidden_dim;
        
        // --- Step 1: Calculate the output of the current expert for the ENTIRE batch ---
        // First MLP layer (gate/up projections)
        dim3 mlp1_grid(batch_size, (2 * intermediate_dim + 31) / 32);
        matmul_kernel<<<mlp1_grid, router_block>>>(
            d_mlp1_out, d_t, d_w_mlp1 + mlp1_weight_offset, batch_size, hidden_dim, 2 * intermediate_dim);
        HIP_CHECK(hipGetLastError());
        
        // Split into gate and up, add bias
        dim3 split_grid((batch_size * intermediate_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        split_gate_up_kernel<<<split_grid, THREADS_PER_BLOCK>>>(
            d_gate, d_up, d_mlp1_out, d_b_mlp1 + mlp1_bias_offset, batch_size, intermediate_dim);
        HIP_CHECK(hipGetLastError());
        
        // SwiGLU activation
        swiglu_kernel<<<split_grid, THREADS_PER_BLOCK>>>(
            d_gate, d_up, d_gate_up, batch_size, intermediate_dim, p->swiglu_limit);
        HIP_CHECK(hipGetLastError());
        
        // Second MLP layer (down projection) -> output stored in d_tb2
        dim3 mlp2_grid(batch_size, (hidden_dim + 31) / 32);
        matmul_kernel<<<mlp2_grid, router_block>>>(
            d_tb2, d_gate_up, d_w_mlp2 + mlp2_weight_offset, batch_size, intermediate_dim, hidden_dim);
        HIP_CHECK(hipGetLastError());
        
        // Add bias
        add_bias_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_tb2, d_b_mlp2 + mlp2_bias_offset, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
        
        // --- Step 2: CORRECTLY aggregate the expert's output using the new kernel ---
        // This kernel selectively adds the expert's output (d_tb2) to the final buffer (d_e_agg)
        // only for the tokens that chose this expert, scaled by the correct weight.
        aggregate_expert_output_kernel<<<expert_agg_grid, norm_block>>>(
            d_e_agg, d_tb2, d_topk_i, d_topk_v, expert_id, batch_size, hidden_dim, p->experts_per_token);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
// fprintf(stderr, "All experts processed successfully for layer %d\n", layer_idx);
    // fprintf(stderr, "Expert outputs aggregated for MoE layer %d\n", layer_idx);
    
    // Residual connection - CORRECT
    accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        d_x, d_e_agg, 1.0f, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "Residual connection added for MoE layer %d\n", layer_idx);
}

float *forward_batch_gpu(Transformer *transformer, int *tokens, int batch_size) {
    // fprintf(stderr, "[DEBUG] Forward pass on GPU for batch size %d\n", batch_size);
    Config *p = &transformer->config;
    
    int hidden_dim = p->hidden_dim;
    
    // CRITICAL FIX: Validate input parameters
    if (!tokens || batch_size <= 0 || batch_size > BATCH_SIZE) {
        // fprintf(stderr, "Error: Invalid input parameters to forward_batch_gpu\n");
        return NULL;
    }
    
    // CRITICAL FIX: Check if GPU memory has been allocated
    if (!d_current_tokens || !d_positions || !d_x || !d_token_embedding_table) {
        // fprintf(stderr, "Error: GPU memory not properly allocated\n");
        return NULL;
    }
    
    // Copy tokens to GPU
    HIP_CHECK(hipMemcpy(d_current_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_positions, positions, batch_size * sizeof(int), hipMemcpyHostToDevice));

    // fprintf(stderr, "[DEBUG] Tokens copied to GPU, proceeding with forward pass\n");
    
    // Copy token embeddings
    dim3 embed_grid((batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    copy_embeddings_kernel<<<embed_grid, THREADS_PER_BLOCK>>>(
        d_x, d_token_embedding_table, d_current_tokens, batch_size, hidden_dim, p->vocab_size);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "[DEBUG] Token embeddings copied to GPU\n");
    // Forward through all layers - UNCOMMENTED: All layers now enabled
    for (int l = 0; l < p->n_layers; l++) {
        attention_gpu(transformer, l, batch_size);
        moe_gpu(transformer, l, batch_size);
    }
    
    // fprintf(stderr, "[DEBUG] Forward pass through all layers completed\n");
    // Final RMSNorm - UNCOMMENTED: Now enabled with GPU weight pointer
    dim3 final_norm_grid(batch_size);
    dim3 final_norm_block(THREADS_PER_BLOCK);
    rmsnorm_kernel<<<final_norm_grid, final_norm_block>>>(
        d_x, d_x, d_rms_out_w, batch_size, hidden_dim);
    HIP_CHECK(hipGetLastError());

    // fprintf(stderr, "[DEBUG] Final RMSNorm applied\n");
    
    // Classifier - UNCOMMENTED: Now enabled with GPU weight pointer
    // Use mixed precision kernel for final output
    dim3 cls_grid(batch_size, (p->vocab_size + 31) / 32);
    dim3 cls_block(32, 32);
    matmul_kernel_fp32_output<<<cls_grid, cls_block>>>(
        d_logits, d_x, d_out_w, batch_size, hidden_dim, p->vocab_size);
    HIP_CHECK(hipGetLastError());
    // fprintf(stderr, "[DEBUG] Classifier output computed\n");
    
    // Copy logits back to CPU (you might want to keep this on GPU for sampling)
    static float *h_logits = nullptr;
    if (!h_logits) {
        h_logits = (float*)malloc(batch_size * p->vocab_size * sizeof(float));
    }
    // fprintf(stderr, "[DEBUG] Copying logits back to CPU\n");
    HIP_CHECK(hipMemcpy(h_logits, d_logits, batch_size * p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    
    return h_logits;
}

long long batched_generate_gpu(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, Requests *requests) {
    Config *p = &transformer->config;
    long long total_tokens_generated = 0;
    // fprintf(stderr, "[DEBUG] Starting batched generation with %d requests\n", requests->num_reqs);
    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE) {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                     ? (requests->num_reqs - req_start)
                                     : BATCH_SIZE;

        // fprintf(stderr, "[DEBUG] Processing batch starting at request %d with size %d\n", req_start, current_batch_size);
        // Initialize batch
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            // Encode prompt
            encode(tokenizer, input_seq, 1, 0, prompt_tokens[b],
                   &prompt_lens[b], p->initial_context_length);

            if (prompt_lens[b] < 1) {
                // fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                prompt_lens[b] = 1;
                prompt_tokens[b][0] = 1; // BOS token
            }

            // Initialize sequence state
            positions[b] = 0;
            finished[b] = false;
            current_tokens[b] = prompt_tokens[b][0];
        }

        // Generation loop
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;
        
        for (int step = 0; step < max_steps && alive > 0; step++) {
            // Forward pass on GPU
            float *logits = forward_batch_gpu(transformer, current_tokens, current_batch_size);

            // Sample next tokens (on CPU for now, could be moved to GPU)
            for (int b = 0; b < current_batch_size; b++) {
                if (finished[b]) continue;

                int req_idx = req_start + b;
                int pos = positions[b];
                float *logits_b = logits + b * p->vocab_size;

                int next_token;
                if (pos < prompt_lens[b] - 1) {
                    // Still processing prompt
                    next_token = prompt_tokens[b][pos + 1];
                } else {
                    // Generate new token
                    next_token = sample(sampler, logits_b);

                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1);
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                    fprintf(stderr, "[DEBUG] Generated token %d for request %d at position %d\n",
                           next_token, req_idx, gen_pos);
                }

                // Check for termination
                if (next_token == 1 || pos >= max_steps - 1) { // BOS token or max length
                    --alive;
                    finished[b] = true;
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1) + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }

                // Update for next iteration
                positions[b]++;
                current_tokens[b] = next_token;
            }
        }

        // Print results
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            int *output_tokens = get_tok_gen_ptr(requests, req_idx);

            // Print the original prompt string
            safe_printf(input_seq);
            printf("!");
            
            // Decode and print generated tokens
            int last_prompt_token = prompt_tokens[b][prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0; ; ++i) {
                int token = output_tokens[i];
                if (token == -1) break;

                const char *piece = decode_piece(tokenizer, prev_token, token);
                safe_printf(piece);
                prev_token = token;
            }
            printf("\n");
        }
        fflush(stdout);
    }

    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
    // fprintf(stderr, "[DEBUG] Starting inference with %d requests\n", requests->num_reqs);
    return batched_generate_gpu(transformer, tokenizer, sampler, requests);
}

#endif // GETP_RUN