#include "../include/gpu_kernels.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdio>

#define WARP_SIZE 64  // AMD GPUs use 64-thread wavefronts
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// External global variables for RoPE
extern float *gpu_cos_vals;
extern float *gpu_sin_vals;

// ================= Helper Functions =================

__device__ void warp_reduce_sum(float& val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
}

__device__ void warp_reduce_max(float& val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
}

// ================= BLAS Kernels =================

// Optimized tiled matrix multiplication kernel
__global__ void matmul_batch_tiled_kernel(bf16 *xout, bf16 *x, bf16 *w, 
                                          int batch_size, int n, int d) {
    __shared__ float tile_x[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ float tile_w[TILE_SIZE][TILE_SIZE + 1];
    
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (batch >= batch_size) return;
    
    float sum = 0.0f;
    bf16 *x_batch = x + batch * n;
    
    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        int x_idx = threadIdx.y * n + t * TILE_SIZE + threadIdx.x;
        int w_idx = (t * TILE_SIZE + threadIdx.y) + col * n;
        
        if (threadIdx.y < TILE_SIZE && t * TILE_SIZE + threadIdx.x < n) {
            tile_x[threadIdx.y][threadIdx.x] = bf16_to_float(x_batch[t * TILE_SIZE + threadIdx.x]);
        } else {
            tile_x[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < d && t * TILE_SIZE + threadIdx.y < n) {
            tile_w[threadIdx.y][threadIdx.x] = bf16_to_float(w[col * n + t * TILE_SIZE + threadIdx.y]);
        } else {
            tile_w[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < 1 && col < d) {  // Only one row per batch item
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += tile_x[0][k] * tile_w[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < 1 && col < d) {
        xout[batch * d + col] = float_to_bf16(sum);
    }
}

void matmul_batch_gpu(bf16 *xout, bf16 *x, bf16 *w, int batch_size, int n, int d) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((d + TILE_SIZE - 1) / TILE_SIZE, 1, batch_size);
    matmul_batch_tiled_kernel<<<grid, block>>>(xout, x, w, batch_size, n, d);
    CHECK_HIP(hipGetLastError());
}

// ================= RMSNorm Kernel =================

__global__ void rmsnorm_batch_kernel(bf16 *o, bf16 *x, bf16 *weight, 
                                     int batch_size, int hidden_dim) {
    extern __shared__ float shared_sum[];
    
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    
    if (batch >= batch_size) return;
    
    bf16 *x_batch = x + batch * hidden_dim;
    bf16 *o_batch = o + batch * hidden_dim;
    
    // Calculate sum of squares
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = bf16_to_float(x_batch[i]);
        thread_sum += val * val;
    }
    
    // Reduce within warp
    warp_reduce_sum(thread_sum);
    
    // Write warp sums to shared memory
    if (lane == 0) {
        shared_sum[tid / WARP_SIZE] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < blockDim.x / WARP_SIZE) {
        thread_sum = shared_sum[tid];
    } else {
        thread_sum = 0.0f;
    }
    warp_reduce_sum(thread_sum);
    
    // Broadcast RMS normalization factor
    if (tid == 0) {
        shared_sum[0] = 1.0f / sqrtf(thread_sum / hidden_dim + 1e-5f);
    }
    __syncthreads();
    
    float rms_scale = shared_sum[0];
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = bf16_to_float(x_batch[i]) * rms_scale;
        o_batch[i] = float_to_bf16(val * bf16_to_float(weight[i]));
    }
}

void rmsnorm_batch_gpu(bf16 *o, bf16 *x, bf16 *weight, int batch_size, int hidden_dim) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size);
    int shared_mem_size = (BLOCK_SIZE / WARP_SIZE + 1) * sizeof(float);
    rmsnorm_batch_kernel<<<grid, block, shared_mem_size>>>(o, x, weight, batch_size, hidden_dim);
    CHECK_HIP(hipGetLastError());
}

// ================= RoPE Kernel =================

__global__ void rope_batch_kernel(bf16 *q, bf16 *k, float *cos_vals, float *sin_vals,
                                  int *positions, int batch_size, int head_dim,
                                  int n_attn_heads, int n_kv_heads) {
    int batch = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size) return;
    
    int pos = positions[batch];
    int half_dim = head_dim / 2;
    
    // Apply RoPE to Q
    if (tid < n_attn_heads * half_dim) {
        int h = tid / half_dim;
        int i = tid % half_dim;
        
        bf16 *q_batch = q + batch * n_attn_heads * head_dim;
        float cos_val = cos_vals[pos * half_dim + i];
        float sin_val = sin_vals[pos * half_dim + i];
        
        int idx0 = h * head_dim + i;
        int idx1 = h * head_dim + i + half_dim;
        
        float q0 = bf16_to_float(q_batch[idx0]);
        float q1 = bf16_to_float(q_batch[idx1]);
        
        float rotated_q0 = q0 * cos_val - q1 * sin_val;
        float rotated_q1 = q0 * sin_val + q1 * cos_val;
        
        q_batch[idx0] = float_to_bf16(rotated_q0);
        q_batch[idx1] = float_to_bf16(rotated_q1);
    }
    
    // Apply RoPE to K
    if (tid < n_kv_heads * half_dim) {
        int h = tid / half_dim;
        int i = tid % half_dim;
        
        bf16 *k_batch = k + batch * n_kv_heads * head_dim;
        float cos_val = cos_vals[pos * half_dim + i];
        float sin_val = sin_vals[pos * half_dim + i];
        
        int idx0 = h * head_dim + i;
        int idx1 = h * head_dim + i + half_dim;
        
        float k0 = bf16_to_float(k_batch[idx0]);
        float k1 = bf16_to_float(k_batch[idx1]);
        
        float rotated_k0 = k0 * cos_val - k1 * sin_val;
        float rotated_k1 = k0 * sin_val + k1 * cos_val;
        
        k_batch[idx0] = float_to_bf16(rotated_k0);
        k_batch[idx1] = float_to_bf16(rotated_k1);
    }
}

void rope_batch_gpu(bf16 *q, bf16 *k, float *cos_vals, float *sin_vals,
                   int *positions, int batch_size, int head_dim,
                   int n_attn_heads, int n_kv_heads) {
    int max_heads = (n_attn_heads > n_kv_heads) ? n_attn_heads : n_kv_heads;
    dim3 block(BLOCK_SIZE);
    dim3 grid((max_heads * head_dim / 2 + block.x - 1) / block.x, batch_size);
    rope_batch_kernel<<<grid, block>>>(q, k, cos_vals, sin_vals, positions,
                                       batch_size, head_dim, n_attn_heads, n_kv_heads);
    CHECK_HIP(hipGetLastError());
}

// ================= Softmax Kernel =================

__global__ void softmax_batch_kernel(bf16 *x, int batch_size, int n_heads, 
                                     int seq_len, int *positions) {
    extern __shared__ float shared_data[];
    
    int batch = blockIdx.y;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= n_heads) return;
    
    int pos = positions[batch];
    bf16 *att = x + batch * n_heads * seq_len + head * seq_len;
    
    // Find max value in parallel
    float max_val = -INFINITY;
    for (int i = tid; i <= pos; i += blockDim.x) {
        float val = bf16_to_float(att[i]);
        max_val = fmaxf(max_val, val);
    }
    
    // Reduce max across threads
    warp_reduce_max(max_val);
    if (tid % WARP_SIZE == 0) {
        shared_data[tid / WARP_SIZE] = max_val;
    }
    __syncthreads();
    
    if (tid < blockDim.x / WARP_SIZE) {
        max_val = shared_data[tid];
    } else {
        max_val = -INFINITY;
    }
    warp_reduce_max(max_val);
    
    if (tid == 0) {
        shared_data[0] = max_val;
    }
    __syncthreads();
    max_val = shared_data[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i <= pos; i += blockDim.x) {
        float val = expf(bf16_to_float(att[i]) - max_val);
        att[i] = float_to_bf16(val);
        sum += val;
    }
    
    // Reduce sum
    warp_reduce_sum(sum);
    if (tid % WARP_SIZE == 0) {
        shared_data[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    
    if (tid < blockDim.x / WARP_SIZE) {
        sum = shared_data[tid];
    } else {
        sum = 0.0f;
    }
    warp_reduce_sum(sum);
    
    if (tid == 0) {
        shared_data[0] = 1.0f / sum;
    }
    __syncthreads();
    float inv_sum = shared_data[0];
    
    // Normalize
    for (int i = tid; i <= pos; i += blockDim.x) {
        att[i] = float_to_bf16(bf16_to_float(att[i]) * inv_sum);
    }
}

void softmax_batch_gpu(bf16 *x, int batch_size, int n_heads, int seq_len, int *positions) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(n_heads, batch_size);
    int shared_mem_size = (BLOCK_SIZE / WARP_SIZE + 1) * sizeof(float);
    softmax_batch_kernel<<<grid, block, shared_mem_size>>>(x, batch_size, n_heads, seq_len, positions);
    CHECK_HIP(hipGetLastError());
}

// ================= SwiGLU Kernel =================

__global__ void swiglu_batch_kernel(bf16 *gate, float swiglu_limit, 
                                    int batch_size, int intermediate_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= intermediate_dim) return;
    
    int global_idx = batch * intermediate_dim + idx;
    float val = bf16_to_float(gate[global_idx]);
    
    // Clamp
    val = fminf(fmaxf(val, -swiglu_limit), swiglu_limit);
    
    // SiLU activation: x * sigmoid(x)
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    val = val * sigmoid_val;
    
    gate[global_idx] = float_to_bf16(val);
}

void swiglu_batch_gpu(bf16 *gate, float swiglu_limit, int batch_size, int intermediate_dim) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((intermediate_dim + block.x - 1) / block.x, batch_size);
    swiglu_batch_kernel<<<grid, block>>>(gate, swiglu_limit, batch_size, intermediate_dim);
    CHECK_HIP(hipGetLastError());
}

// ================= Attention Kernels =================

__global__ void add_bias_residual_kernel(bf16 *x, bf16 *input, bf16 *bias,
                                        int batch_size, int hidden_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= hidden_dim) return;
    
    int global_idx = batch * hidden_dim + idx;
    float val = bf16_to_float(input[global_idx]) + 
                bf16_to_float(bias[idx]) +
                bf16_to_float(x[global_idx]);
    x[global_idx] = float_to_bf16(val);
}

// ================= Attention Component Kernels =================

__global__ void qkv_projection_kernel(bf16 *qkv_out, bf16 *q, bf16 *k, bf16 *v,
                                      bf16 *b_qkv, int batch_size, 
                                      int n_attn_heads, int n_kv_heads, int head_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size) return;
    
    int qkv_dim = (n_attn_heads + 2 * n_kv_heads) * head_dim;
    if (idx >= qkv_dim) return;
    
    int batch_offset = batch * qkv_dim;
    
    // Add bias to QKV
    float val = bf16_to_float(qkv_out[batch_offset + idx]) + bf16_to_float(b_qkv[idx]);
    qkv_out[batch_offset + idx] = float_to_bf16(val);
    
    // Split into Q, K, V
    int q_dim = n_attn_heads * head_dim;
    int k_dim = n_kv_heads * head_dim;
    
    if (idx < q_dim) {
        // Copy to Q
        q[batch * q_dim + idx] = qkv_out[batch_offset + idx];
    } else if (idx < q_dim + k_dim) {
        // Copy to K
        k[batch * k_dim + (idx - q_dim)] = qkv_out[batch_offset + idx];
    } else {
        // Copy to V
        v[batch * k_dim + (idx - q_dim - k_dim)] = qkv_out[batch_offset + idx];
    }
}

__global__ void sdpa_kernel(bf16 *att, bf16 *q, bf16 *k_cache, bf16 *v_cache,
                           bf16 *output, int *positions, bf16 *mask,
                           int batch_size, int n_heads, int n_kv_heads,
                           int head_dim, int seq_len, int layer_offset) {
    extern __shared__ float shared_mem[];
    
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= n_heads) return;
    
    int pos = positions[batch];
    int kv_head = head / (n_heads / n_kv_heads);
    
    float *att_scores = shared_mem;
    
    // Compute attention scores
    for (int t = tid; t <= pos; t += blockDim.x) {
        float score = 0.0f;
        
        for (int i = 0; i < head_dim; i++) {
            float q_val = bf16_to_float(q[batch * n_heads * head_dim + head * head_dim + i]);
            float k_val = bf16_to_float(k_cache[batch * seq_len * n_kv_heads * head_dim + 
                                              layer_offset + t * n_kv_heads * head_dim + 
                                              kv_head * head_dim + i]);
            score += q_val * k_val;
        }
        
        score /= sqrtf((float)head_dim);
        
        // Apply mask if needed
        if (mask != nullptr) {
            score += bf16_to_float(mask[pos * seq_len + t]);
        }
        
        att_scores[t] = score;
    }
    __syncthreads();
    
    // Softmax
    float max_score = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        max_score = fmaxf(max_score, att_scores[t]);
    }
    warp_reduce_max(max_score);
    
    if (tid == 0) {
        shared_mem[blockDim.x] = max_score;
    }
    __syncthreads();
    max_score = shared_mem[blockDim.x];
    
    float sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        att_scores[t] = expf(att_scores[t] - max_score);
        sum += att_scores[t];
    }
    warp_reduce_sum(sum);
    
    if (tid == 0) {
        shared_mem[blockDim.x + 1] = sum;
    }
    __syncthreads();
    sum = shared_mem[blockDim.x + 1];
    
    for (int t = tid; t <= pos; t += blockDim.x) {
        att_scores[t] /= sum;
    }
    __syncthreads();
    
    // Weighted sum of values
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        
        for (int t = 0; t <= pos; t++) {
            float v_val = bf16_to_float(v_cache[batch * seq_len * n_kv_heads * head_dim + 
                                               layer_offset + t * n_kv_heads * head_dim + 
                                               kv_head * head_dim + i]);
            acc += att_scores[t] * v_val;
        }
        
        output[batch * n_heads * head_dim + head * head_dim + i] = float_to_bf16(acc);
    }
}

void attention_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p, 
                        unsigned long long l, int batch_size, int *positions) {
    int qkv_dim = (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    
    // RMSNorm
    rmsnorm_batch_gpu(gs->t, gs->x, gw->rms_attn_w + l * p->hidden_dim, 
                     batch_size, p->hidden_dim);
    
    // QKV projection
    matmul_batch_gpu(gs->qkv, gs->t, 
                    gw->w_qkv + l * p->hidden_dim * qkv_dim,
                    batch_size, p->hidden_dim, qkv_dim);
    
    // Add bias and split QKV
    dim3 block1(BLOCK_SIZE);
    dim3 grid1((qkv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    qkv_projection_kernel<<<grid1, block1>>>(gs->qkv, gs->q, gs->k, gs->v,
                                            gw->b_qkv + l * qkv_dim,
                                            batch_size, p->n_attn_heads, 
                                            p->n_kv_heads, p->head_dim);
    CHECK_HIP(hipGetLastError());
    
    // Apply RoPE if cos/sin values are available
    if (gpu_cos_vals != nullptr && gpu_sin_vals != nullptr) {
        rope_batch_gpu(gs->q, gs->k, gpu_cos_vals, gpu_sin_vals, positions, 
                       batch_size, p->head_dim, p->n_attn_heads, p->n_kv_heads);
    }
    
    // Update KV cache
    int kv_dim = p->n_kv_heads * p->head_dim;
    for (int b = 0; b < batch_size; b++) {
        int pos = positions[b];
        CHECK_HIP(hipMemcpy(gs->key_cache + b * p->n_layers * p->seq_len * kv_dim + 
                           l * p->seq_len * kv_dim + pos * kv_dim,
                           gs->k + b * kv_dim,
                           kv_dim * sizeof(bf16), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(gs->value_cache + b * p->n_layers * p->seq_len * kv_dim + 
                           l * p->seq_len * kv_dim + pos * kv_dim,
                           gs->v + b * kv_dim,
                           kv_dim * sizeof(bf16), hipMemcpyDeviceToDevice));
    }
    
    // SDPA
    dim3 block2(128);
    dim3 grid2(1, p->n_attn_heads, batch_size);
    int shared_size = (p->seq_len + 2) * sizeof(float);
    sdpa_kernel<<<grid2, block2, shared_size>>>(gs->att, gs->q, gs->key_cache, gs->value_cache,
                                                gs->tb, positions, nullptr,
                                                batch_size, p->n_attn_heads, p->n_kv_heads,
                                                p->head_dim, p->seq_len, l * p->seq_len * kv_dim);
    CHECK_HIP(hipGetLastError());
    
    // Output projection
    matmul_batch_gpu(gs->tb2, gs->tb, 
                    gw->w_o + l * p->hidden_dim * p->n_attn_heads * p->head_dim,
                    batch_size, p->n_attn_heads * p->head_dim, p->hidden_dim);
    
    // Add bias and residual using a kernel
    dim3 block3(BLOCK_SIZE);
    dim3 grid3((p->hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    add_bias_residual_kernel<<<grid3, block3>>>(gs->x, gs->tb2, gw->b_o + l * p->hidden_dim,
                                               batch_size, p->hidden_dim);
    CHECK_HIP(hipGetLastError());
}

// ================= MoE Kernels =================

__global__ void add_bias_kernel(bf16 *scores, bf16 *bias, int batch_size, int n_experts) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= n_experts) return;
    
    int global_idx = batch * n_experts + idx;
    float val = bf16_to_float(scores[global_idx]) + bf16_to_float(bias[idx]);
    scores[global_idx] = float_to_bf16(val);
}

__global__ void accumulate_experts_kernel(bf16 *e_agg, bf16 *expert_out, 
                                         float weight, int batch_idx, 
                                         int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;
    
    int agg_idx = batch_idx * hidden_dim + idx;
    float val = bf16_to_float(e_agg[agg_idx]) + 
                weight * bf16_to_float(expert_out[idx]);
    e_agg[agg_idx] = float_to_bf16(val);
}

__global__ void residual_add_kernel(bf16 *x, bf16 *e_agg, int batch_size, int hidden_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= hidden_dim) return;
    
    int global_idx = batch * hidden_dim + idx;
    float val = bf16_to_float(x[global_idx]) + bf16_to_float(e_agg[global_idx]);
    x[global_idx] = float_to_bf16(val);
}

__global__ void topk_kernel(bf16 *scores, bf16 *topk_values, int *topk_indices,
                           int batch_size, int n_experts, int k) {
    extern __shared__ float shared_data[];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    bf16 *batch_scores = scores + batch * n_experts;
    bf16 *batch_topk_v = topk_values + batch * k;
    int *batch_topk_i = topk_indices + batch * k;
    
    // Simple selection sort for top-k (can be optimized with heap)
    for (int i = 0; i < k; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int j = 0; j < n_experts; j++) {
            float val = bf16_to_float(batch_scores[j]);
            if (val > max_val) {
                bool already_selected = false;
                for (int prev = 0; prev < i; prev++) {
                    if (batch_topk_i[prev] == j) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected) {
                    max_val = val;
                    max_idx = j;
                }
            }
        }
        
        batch_topk_v[i] = float_to_bf16(max_val);
        batch_topk_i[i] = max_idx;
    }
    
    // Softmax normalization of topk values
    float max_val = bf16_to_float(batch_topk_v[0]);
    float sum = 0.0f;
    
    for (int i = 0; i < k; i++) {
        float val = expf(bf16_to_float(batch_topk_v[i]) - max_val);
        batch_topk_v[i] = float_to_bf16(val);
        sum += val;
    }
    
    for (int i = 0; i < k; i++) {
        batch_topk_v[i] = float_to_bf16(bf16_to_float(batch_topk_v[i]) / sum);
    }
}

__global__ void mlp_gate_up_kernel(bf16 *output, bf16 *input, bf16 *weight, bf16 *bias,
                                   float swiglu_limit, int batch_size, 
                                   int hidden_dim, int intermediate_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= 2 * intermediate_dim) return;
    
    // Compute matrix multiplication for gate_up projection
    float sum = bf16_to_float(bias[idx]);
    
    for (int i = 0; i < hidden_dim; i++) {
        sum += bf16_to_float(input[batch * hidden_dim + i]) * 
               bf16_to_float(weight[idx * hidden_dim + i]);
    }
    
    // Apply SwiGLU to gate part (first half)
    if (idx < intermediate_dim) {
        sum = fminf(fmaxf(sum, -swiglu_limit), swiglu_limit);
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        sum = sum * sigmoid_val;
    }
    
    output[batch * 2 * intermediate_dim + idx] = float_to_bf16(sum);
}

__global__ void mlp_down_kernel(bf16 *output, bf16 *gate, bf16 *up, bf16 *weight, bf16 *bias,
                               int batch_size, int hidden_dim, int intermediate_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= hidden_dim) return;
    
    float sum = bf16_to_float(bias[idx]);
    
    for (int i = 0; i < intermediate_dim; i++) {
        float gate_val = bf16_to_float(gate[batch * intermediate_dim + i]);
        float up_val = bf16_to_float(up[batch * intermediate_dim + i]);
        sum += (gate_val * up_val) * bf16_to_float(weight[idx * intermediate_dim + i]);
    }
    
    output[batch * hidden_dim + idx] = float_to_bf16(sum);
}

void moe_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p,
                  unsigned long long l, int batch_size) {
    // RMSNorm
    rmsnorm_batch_gpu(gs->t, gs->x, gw->rms_ffn_w + l * p->hidden_dim,
                     batch_size, p->hidden_dim);
    
    // Router scores
    matmul_batch_gpu(gs->router_score, gs->t, 
                    gw->w_router + l * p->hidden_dim * p->n_experts,
                    batch_size, p->hidden_dim, p->n_experts);
    
    // Add bias to router scores
    dim3 block1(BLOCK_SIZE);
    dim3 grid1((p->n_experts + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    add_bias_kernel<<<grid1, block1>>>(gs->router_score, gw->b_router + l * p->n_experts,
                                       batch_size, p->n_experts);
    CHECK_HIP(hipGetLastError());
    
    // Top-k selection
    dim3 block2(1);
    dim3 grid2(batch_size);
    topk_kernel<<<grid2, block2>>>(gs->router_score, gs->topk_v, gs->topk_i,
                                   batch_size, p->n_experts, p->experts_per_token);
    CHECK_HIP(hipGetLastError());
    
    // Initialize aggregation buffer
    CHECK_HIP(hipMemset(gs->e_agg, 0, batch_size * p->hidden_dim * sizeof(bf16)));
    
    // Process each expert
    for (int expert_id = 0; expert_id < p->n_experts; expert_id++) {
        // Count tokens for this expert
        int token_count = 0;
        int token_indices[BATCH_SIZE];
        float token_weights[BATCH_SIZE];
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < p->experts_per_token; i++) {
                if (gs->topk_i[b * p->experts_per_token + i] == expert_id) {
                    token_indices[token_count] = b;
                    token_weights[token_count] = bf16_to_float(gs->topk_v[b * p->experts_per_token + i]);
                    token_count++;
                }
            }
        }
        
        if (token_count == 0) continue;
        
        // Process tokens through expert MLP
        bf16 *w_mlp1 = gw->w_mlp1 + (l * p->n_experts + expert_id) * 2 * p->intermediate_dim * p->hidden_dim;
        bf16 *b_mlp1 = gw->b_mlp1 + (l * p->n_experts + expert_id) * 2 * p->intermediate_dim;
        bf16 *w_mlp2 = gw->w_mlp2 + (l * p->n_experts + expert_id) * p->hidden_dim * p->intermediate_dim;
        bf16 *b_mlp2 = gw->b_mlp2 + (l * p->n_experts + expert_id) * p->hidden_dim;
        
        // Gate-up projection
        dim3 block3(BLOCK_SIZE);
        dim3 grid3((2 * p->intermediate_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, token_count);
        mlp_gate_up_kernel<<<grid3, block3>>>(gs->mlp1_out, gs->t, w_mlp1, b_mlp1,
                                             p->swiglu_limit, token_count, 
                                             p->hidden_dim, p->intermediate_dim);
        CHECK_HIP(hipGetLastError());
        
        // Split gate and up
        for (int i = 0; i < token_count; i++) {
            CHECK_HIP(hipMemcpy(gs->gate + i * p->intermediate_dim,
                              gs->mlp1_out + i * 2 * p->intermediate_dim,
                              p->intermediate_dim * sizeof(bf16), hipMemcpyDeviceToDevice));
            CHECK_HIP(hipMemcpy(gs->up + i * p->intermediate_dim,
                              gs->mlp1_out + i * 2 * p->intermediate_dim + p->intermediate_dim,
                              p->intermediate_dim * sizeof(bf16), hipMemcpyDeviceToDevice));
        }
        
        // Down projection
        dim3 grid4((p->hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, token_count);
        mlp_down_kernel<<<grid4, block3>>>(gs->tb2, gs->gate, gs->up, w_mlp2, b_mlp2,
                                          token_count, p->hidden_dim, p->intermediate_dim);
        CHECK_HIP(hipGetLastError());
        
        // Accumulate weighted results
        for (int i = 0; i < token_count; i++) {
            int batch_idx = token_indices[i];
            float weight = token_weights[i];
            
            dim3 block_acc(BLOCK_SIZE);
            dim3 grid_acc((p->hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
            accumulate_experts_kernel<<<grid_acc, block_acc>>>(gs->e_agg, 
                                                              gs->tb2 + i * p->hidden_dim,
                                                              weight, batch_idx, p->hidden_dim);
        }
        CHECK_HIP(hipGetLastError());
    }
    
    // Residual connection
    dim3 block_res(BLOCK_SIZE);
    dim3 grid_res((p->hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    residual_add_kernel<<<grid_res, block_res>>>(gs->x, gs->e_agg, batch_size, p->hidden_dim);
    CHECK_HIP(hipGetLastError());
}

// ================= Forward Pass =================

__global__ void embedding_lookup_kernel(bf16 *output, bf16 *embedding_table,
                                       int *tokens, int batch_size, int hidden_dim) {
    int batch = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || idx >= hidden_dim) return;
    
    int token = tokens[batch];
    output[batch * hidden_dim + idx] = embedding_table[token * hidden_dim + idx];
}

bf16* forward_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p,
                       int *tokens_cpu, int batch_size, int *positions_cpu) {
    // Allocate and copy tokens/positions to GPU
    int *tokens_gpu, *positions_gpu;
    CHECK_HIP(hipMalloc(&tokens_gpu, batch_size * sizeof(int)));
    CHECK_HIP(hipMalloc(&positions_gpu, batch_size * sizeof(int)));
    CHECK_HIP(hipMemcpy(tokens_gpu, tokens_cpu, batch_size * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(positions_gpu, positions_cpu, batch_size * sizeof(int), hipMemcpyHostToDevice));
    
    // Token embedding lookup
    dim3 block(BLOCK_SIZE);
    dim3 grid((p->hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);
    embedding_lookup_kernel<<<grid, block>>>(gs->x, gw->token_embedding_table,
                                            tokens_gpu, batch_size, p->hidden_dim);
    CHECK_HIP(hipGetLastError());
    
    // Process through all layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_batch_gpu(gw, gs, p, l, batch_size, positions_gpu);
        moe_batch_gpu(gw, gs, p, l, batch_size);
    }
    
    // Final RMSNorm
    rmsnorm_batch_gpu(gs->x, gs->x, gw->rms_out_w, batch_size, p->hidden_dim);
    
    // Classifier
    matmul_batch_gpu(gs->logits, gs->x, gw->out, batch_size, p->hidden_dim, p->vocab_size);
    
    // Cleanup
    CHECK_HIP(hipFree(tokens_gpu));
    CHECK_HIP(hipFree(positions_gpu));
    
    return gs->logits;
}