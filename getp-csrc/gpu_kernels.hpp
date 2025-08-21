#pragma once

#include "gpu_memory.hpp"
#include <hip/hip_runtime.h>

// GPU Kernel declarations for GPT-OSS model

// ============================================================================
// Embedding and Position Kernels
// ============================================================================

// Token embedding lookup
void gpu_embedding_lookup(const bf16* embedding_table,
                         const int* token_ids,
                         float* output,
                         int batch_size,
                         int hidden_dim,
                         int vocab_size,
                         hipStream_t stream = 0);

// ============================================================================
// Normalization Kernels
// ============================================================================

// RMSNorm kernel
void gpu_rmsnorm(const float* input,
                 const bf16* weight,
                 float* output,
                 int batch_size,
                 int hidden_dim,
                 float eps = 1e-5f,
                 hipStream_t stream = 0);

// ============================================================================
// Attention Kernels
// ============================================================================

// QKV projection
void gpu_qkv_projection(const float* input,
                       const bf16* w_qkv,
                       const bf16* b_qkv,
                       float* q,
                       float* k,
                       float* v,
                       int batch_size,
                       int hidden_dim,
                       int n_heads,
                       int n_kv_heads,
                       int head_dim,
                       hipStream_t stream = 0);

// RoPE (Rotary Position Embedding)
void gpu_rope(float* q,
             float* k,
             const int* positions,
             int batch_size,
             int n_heads,
             int n_kv_heads,
             int head_dim,
             float rope_theta,
             float rope_scaling_factor,
             hipStream_t stream = 0);

// Multi-head attention
void gpu_multi_head_attention(const float* q,
                             const float* k,
                             const float* v,
                             const bf16* k_cache,
                             const bf16* v_cache,
                             bf16* k_cache_out,
                             bf16* v_cache_out,
                             const int* positions,
                             float* output,
                             int batch_size,
                             int n_heads,
                             int n_kv_heads,
                             int head_dim,
                             int seq_len,
                             int sliding_window,
                             int layer_idx,
                             hipStream_t stream = 0);

// Output projection
void gpu_output_projection(const float* input,
                          const bf16* w_o,
                          const bf16* b_o,
                          float* output,
                          int batch_size,
                          int hidden_dim,
                          int n_heads,
                          int head_dim,
                          hipStream_t stream = 0);

// ============================================================================
// MoE (Mixture of Experts) Kernels
// ============================================================================

// Router computation
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
                   hipStream_t stream = 0);

// Expert dispatch - gather inputs for selected experts
void gpu_expert_dispatch(const float* input,
                        const int* expert_indices,
                        const float* expert_weights,
                        float* expert_input,
                        int batch_size,
                        int hidden_dim,
                        int experts_per_token,
                        hipStream_t stream = 0);

// Expert MLP computation (for a batch of experts)
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
                   hipStream_t stream = 0);

// Expert combine - aggregate expert outputs
void gpu_expert_combine(const float* expert_output,
                       const int* expert_indices,
                       const float* expert_weights,
                       float* output,
                       int batch_size,
                       int hidden_dim,
                       int experts_per_token,
                       hipStream_t stream = 0);

// ============================================================================
// Final Output Kernels
// ============================================================================

// Final logits computation
void gpu_compute_logits(const float* input,
                       const bf16* w_out,
                       float* logits,
                       int batch_size,
                       int hidden_dim,
                       int vocab_size,
                       hipStream_t stream = 0);

// ============================================================================
// Utility Kernels
// ============================================================================

// Residual addition
void gpu_residual_add(const float* input1,
                     const float* input2,
                     float* output,
                     int batch_size,
                     int hidden_dim,
                     hipStream_t stream = 0);

// Memory copy with optional type conversion
void gpu_copy_with_conversion(const void* src,
                             void* dst,
                             size_t num_elements,
                             bool src_is_bf16,
                             bool dst_is_bf16,
                             hipStream_t stream = 0);

// ============================================================================
// Kernel Testing and Validation
// ============================================================================

// Compare GPU output with CPU reference
bool validate_kernel_output(const float* gpu_output,
                           const float* cpu_reference,
                           size_t num_elements,
                           float tolerance = 1e-3f,
                           const char* kernel_name = nullptr);

// Benchmark kernel performance
void benchmark_kernel(void (*kernel_func)(void*),
                     void* params,
                     const char* kernel_name,
                     int warmup_iters = 10,
                     int bench_iters = 100);

// ============================================================================
// Full Layer Functions (combining multiple kernels)
// ============================================================================

// Complete attention layer
void gpu_attention_layer(GPURunState& state,
                        const GPUModelWeights::LayerWeights& weights,
                        int layer_idx,
                        int batch_size,
                        const Config* config,
                        hipStream_t stream = 0);

// Complete MoE layer
void gpu_moe_layer(GPURunState& state,
                  const GPUModelWeights::LayerWeights& weights,
                  GPUMemoryManager& memory_mgr,
                  int layer_idx,
                  int batch_size,
                  const Config* config,
                  hipStream_t stream = 0);

// ============================================================================
// Pipeline Functions
// ============================================================================

// Forward pass for a batch
float* gpu_forward_batch(GPUModelWeights& weights,
                        GPURunState& state,
                        GPUMemoryManager& memory_mgr,
                        const int* tokens,
                        int batch_size,
                        const Config* config);