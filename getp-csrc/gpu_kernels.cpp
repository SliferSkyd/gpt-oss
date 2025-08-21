#include "gpu_kernels.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d - %s\n", \
                __FILE__, __LINE__, hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// CUDA Kernels Implementation
// ============================================================================

// Token embedding lookup kernel
__global__ void embedding_lookup_kernel(const bf16* embedding_table,
                                       const int* token_ids,
                                       float* output,
                                       int batch_size,
                                       int hidden_dim) {
    int batch_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx < batch_size && dim_idx < hidden_dim) {
        int token_id = token_ids[batch_idx];
        bf16 value = embedding_table[token_id * hidden_dim + dim_idx];
        output[batch_idx * hidden_dim + dim_idx] = __bfloat162float(value);
    }
}

// RMSNorm kernel
__global__ void rmsnorm_kernel(const float* input,
                              const bf16* weight,
                              float* output,
                              int batch_size,
                              int hidden_dim,
                              float eps) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* input_row = input + batch_idx * hidden_dim;
    float* output_row = output + batch_idx * hidden_dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = input_row[i];
        sum_sq += val * val;
    }
    
    // Reduce within block
    shared_mem[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared_mem[0] / hidden_dim + eps);
    float scale = 1.0f / rms;
    
    // Apply normalization and weight
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = input_row[i] * scale;
        float w = __bfloat162float(weight[i]);
        output_row[i] = normalized * w;
    }
}

// SwiGLU activation kernel
__device__ inline float swiglu_activation(float x, float limit) {
    if (x > limit) return x;
    if (x < -limit) return 0.0f;
    return x / (1.0f + expf(-x));
}

// Expert MLP forward kernel
__global__ void expert_mlp_kernel(const float* input,
                                 const bf16* w_mlp1,
                                 const bf16* b_mlp1,
                                 const bf16* w_mlp2,
                                 const bf16* b_mlp2,
                                 float* output,
                                 int num_tokens,
                                 int hidden_dim,
                                 int intermediate_dim,
                                 float swiglu_limit) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    extern __shared__ float shared_mem[];
    float* gate = shared_mem;
    float* up = shared_mem + intermediate_dim;
    
    // First linear: input -> [gate, up]
    // This is a simplified version - real implementation needs proper GEMM
    if (dim_idx < intermediate_dim) {
        float gate_val = __bfloat162float(b_mlp1[dim_idx]);
        float up_val = __bfloat162float(b_mlp1[intermediate_dim + dim_idx]);
        
        for (int h = 0; h < hidden_dim; h++) {
            float in_val = input[token_idx * hidden_dim + h];
            gate_val += in_val * __bfloat162float(w_mlp1[dim_idx * hidden_dim + h]);
            up_val += in_val * __bfloat162float(w_mlp1[(intermediate_dim + dim_idx) * hidden_dim + h]);
        }
        
        // Apply SwiGLU activation
        gate[dim_idx] = swiglu_activation(gate_val, swiglu_limit);
        up[dim_idx] = up_val;
    }
    
    __syncthreads();
    
    // Second linear: gate * up -> output
    if (dim_idx < hidden_dim) {
        float out_val = __bfloat162float(b_mlp2[dim_idx]);
        
        for (int i = 0; i < intermediate_dim; i++) {
            float gated = gate[i] * up[i];
            out_val += gated * __bfloat162float(w_mlp2[dim_idx * intermediate_dim + i]);
        }
        
        output[token_idx * hidden_dim + dim_idx] = out_val;
    }
}

// Top-K selection kernel for router
__global__ void topk_selection_kernel(const float* scores,
                                     int* indices,
                                     float* values,
                                     int batch_size,
                                     int n_experts,
                                     int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_scores = scores + batch_idx * n_experts;
    int* batch_indices = indices + batch_idx * k;
    float* batch_values = values + batch_idx * k;
    
    // Simple top-k selection (can be optimized with better algorithms)
    for (int i = 0; i < k; i++) {
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int e = 0; e < n_experts; e++) {
            bool already_selected = false;
            for (int j = 0; j < i; j++) {
                if (batch_indices[j] == e) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && batch_scores[e] > max_val) {
                max_val = batch_scores[e];
                max_idx = e;
            }
        }
        
        batch_indices[i] = max_idx;
        batch_values[i] = max_val;
    }
    
    // Normalize weights using softmax
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        batch_values[i] = expf(batch_values[i]);
        sum += batch_values[i];
    }
    for (int i = 0; i < k; i++) {
        batch_values[i] /= sum;
    }
}

// ============================================================================
// Host Functions Implementation
// ============================================================================

void gpu_embedding_lookup(const bf16* embedding_table,
                         const int* token_ids,
                         float* output,
                         int batch_size,
                         int hidden_dim,
                         int vocab_size,
                         hipStream_t stream) {
    dim3 grid(batch_size, (hidden_dim + 255) / 256);
    dim3 block(256);
    
    hipLaunchKernelGGL(embedding_lookup_kernel, grid, block, 0, stream,
                      embedding_table, token_ids, output, batch_size, hidden_dim);
}

void gpu_rmsnorm(const float* input,
                const bf16* weight,
                float* output,
                int batch_size,
                int hidden_dim,
                float eps,
                hipStream_t stream) {
    dim3 grid(batch_size);
    dim3 block(256);
    size_t shared_mem_size = block.x * sizeof(float);
    
    hipLaunchKernelGGL(rmsnorm_kernel, grid, block, shared_mem_size, stream,
                      input, weight, output, batch_size, hidden_dim, eps);
}

void gpu_moe_router(const float* input,
                   const bf16* w_router,
                   const bf16* b_router,
                   float* router_logits,
                   float* router_probs,
                   int* expert_indices,
                   float* expert_weights,
                   int batch_size,
                   int hidden_dim,
                   int n_experts,
                   int experts_per_token,
                   hipStream_t stream) {
    // Step 1: Compute router logits (simplified - needs proper GEMM)
    // This is a placeholder - implement with hipBLAS or custom kernel
    
    // Step 2: Top-K selection
    dim3 grid(batch_size);
    dim3 block(32);
    
    hipLaunchKernelGGL(topk_selection_kernel, grid, block, 0, stream,
                      router_logits, expert_indices, expert_weights,
                      batch_size, n_experts, experts_per_token);
}

void gpu_expert_mlp(const float* input,
                   const bf16* w_mlp1,
                   const bf16* b_mlp1,
                   const bf16* w_mlp2,
                   const bf16* b_mlp2,
                   bf16* mlp1_out,
                   bf16* gate,
                   bf16* up,
                   float* output,
                   int num_tokens,
                   int hidden_dim,
                   int intermediate_dim,
                   float swiglu_limit,
                   int expert_id,
                   int gpu_id,
                   hipStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(256);
    size_t shared_mem_size = 2 * intermediate_dim * sizeof(float);
    
    hipLaunchKernelGGL(expert_mlp_kernel, grid, block, shared_mem_size, stream,
                      input, w_mlp1, b_mlp1, w_mlp2, b_mlp2, output,
                      num_tokens, hidden_dim, intermediate_dim, swiglu_limit);
}

// ============================================================================
// Full Layer Implementation
// ============================================================================

void gpu_attention_layer(GPURunState& state,
                        const GPUModelWeights::LayerWeights& weights,
                        int layer_idx,
                        int batch_size,
                        const Config* config,
                        hipStream_t stream) {
    // RMSNorm
    gpu_rmsnorm(state.x, weights.rms_attn_w, state.residual,
               batch_size, config->hidden_dim, 1e-5f, stream);
    
    // QKV projection
    gpu_qkv_projection(state.residual, weights.w_qkv, weights.b_qkv,
                      state.q, state.k, state.v,
                      batch_size, config->hidden_dim,
                      config->n_attn_heads, config->n_kv_heads,
                      config->head_dim, stream);
    
    // RoPE
    gpu_rope(state.q, state.k, state.positions,
            batch_size, config->n_attn_heads, config->n_kv_heads,
            config->head_dim, config->rope_theta,
            config->rope_scaling_factor, stream);
    
    // Multi-head attention
    gpu_multi_head_attention(state.q, state.k, state.v,
                           state.key_cache, state.value_cache,
                           state.key_cache, state.value_cache,
                           state.positions, state.att_output,
                           batch_size, config->n_attn_heads,
                           config->n_kv_heads, config->head_dim,
                           config->seq_len, config->sliding_window,
                           layer_idx, stream);
    
    // Output projection
    gpu_output_projection(state.att_output, weights.w_o, weights.b_o,
                        state.att_output, batch_size, config->hidden_dim,
                        config->n_attn_heads, config->head_dim, stream);
    
    // Residual connection
    gpu_residual_add(state.x, state.att_output, state.x,
                    batch_size, config->hidden_dim, stream);
}

void gpu_moe_layer(GPURunState& state,
                  const GPUModelWeights::LayerWeights& weights,
                  GPUMemoryManager& memory_mgr,
                  int layer_idx,
                  int batch_size,
                  const Config* config,
                  hipStream_t stream) {
    // RMSNorm
    gpu_rmsnorm(state.x, weights.rms_ffn_w, state.residual,
               batch_size, config->hidden_dim, 1e-5f, stream);
    
    // Router
    gpu_moe_router(state.residual, weights.w_router, weights.b_router,
                  state.router_logits, state.router_probs,
                  state.expert_indices, state.expert_weights,
                  batch_size, config->hidden_dim,
                  config->n_experts, config->experts_per_token, stream);
    
    // Dispatch to experts
    gpu_expert_dispatch(state.residual, state.expert_indices,
                       state.expert_weights, state.expert_input,
                       batch_size, config->hidden_dim,
                       config->experts_per_token, stream);
    
    // Run expert MLPs (distributed across GPUs)
    // This is simplified - real implementation needs multi-GPU coordination
    for (int e = 0; e < config->experts_per_token; e++) {
        // Get expert weights from appropriate GPU
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        int expert_gpu = weights.device_id; // Simplified
        
        gpu_expert_mlp(state.expert_input + e * batch_size * config->hidden_dim,
                      weights.w_mlp1, weights.b_mlp1,
                      weights.w_mlp2, weights.b_mlp2,
                      state.mlp1_out, state.gate, state.up,
                      state.expert_output + e * batch_size * config->hidden_dim,
                      batch_size, config->hidden_dim,
                      config->intermediate_dim, config->swiglu_limit,
                      e, expert_gpu, stream);
    }
    
    // Combine expert outputs
    gpu_expert_combine(state.expert_output, state.expert_indices,
                      state.expert_weights, state.mlp_output,
                      batch_size, config->hidden_dim,
                      config->experts_per_token, stream);
    
    // Residual connection
    gpu_residual_add(state.x, state.mlp_output, state.x,
                    batch_size, config->hidden_dim, stream);
}

// ============================================================================
// Main Forward Pass
// ============================================================================

float* gpu_forward_batch(GPUModelWeights& weights,
                        GPURunState& state,
                        GPUMemoryManager& memory_mgr,
                        const int* tokens,
                        int batch_size,
                        const Config* config) {
    hipStream_t stream = 0;
    
    // Token embedding lookup
    gpu_embedding_lookup(weights.token_embedding_table, tokens, state.x,
                        batch_size, config->hidden_dim, config->vocab_size, stream);
    
    // Forward through layers
    for (int l = 0; l < config->n_layers; l++) {
        // Set device for this layer
        memory_mgr.set_device(weights.layers[l].device_id);
        
        // Attention
        gpu_attention_layer(state, weights.layers[l], l, batch_size, config, stream);
        
        // MoE
        gpu_moe_layer(state, weights.layers[l], memory_mgr, l, batch_size, config, stream);
    }
    
    // Final RMSNorm
    gpu_rmsnorm(state.x, weights.rms_out_w, state.x,
               batch_size, config->hidden_dim, 1e-5f, stream);
    
    // Compute logits
    gpu_compute_logits(state.x, weights.out, state.logits,
                      batch_size, config->hidden_dim, config->vocab_size, stream);
    
    // Synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    
    return state.logits;
}

// ============================================================================
// Validation and Testing
// ============================================================================

bool validate_kernel_output(const float* gpu_output,
                           const float* cpu_reference,
                           size_t num_elements,
                           float tolerance,
                           const char* kernel_name) {
    float* host_output = new float[num_elements];
    HIP_CHECK(hipMemcpy(host_output, gpu_output, 
                       num_elements * sizeof(float), hipMemcpyDeviceToHost));
    
    float max_error = 0.0f;
    int num_errors = 0;
    
    for (size_t i = 0; i < num_elements; i++) {
        float error = std::abs(host_output[i] - cpu_reference[i]);
        max_error = std::max(max_error, error);
        
        if (error > tolerance) {
            num_errors++;
            if (num_errors <= 5) {  // Print first 5 errors
                printf("[%s] Mismatch at %zu: GPU=%.6f, CPU=%.6f, error=%.6f\n",
                       kernel_name ? kernel_name : "Unknown",
                       i, host_output[i], cpu_reference[i], error);
            }
        }
    }
    
    if (num_errors > 0) {
        printf("[%s] Validation FAILED: %d/%zu elements differ (max error: %.6f)\n",
               kernel_name ? kernel_name : "Unknown",
               num_errors, num_elements, max_error);
    } else {
        printf("[%s] Validation PASSED (max error: %.6f)\n",
               kernel_name ? kernel_name : "Unknown", max_error);
    }
    
    delete[] host_output;
    return num_errors == 0;
}

void benchmark_kernel(void (*kernel_func)(void*),
                     void* params,
                     const char* kernel_name,
                     int warmup_iters,
                     int bench_iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        kernel_func(params);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        kernel_func(params);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    printf("[%s] Average time: %.3f ms\n", 
           kernel_name ? kernel_name : "Unknown",
           milliseconds / bench_iters);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}