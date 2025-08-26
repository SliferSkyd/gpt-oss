#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include "config.hpp"
#include "utils.hpp"
#include "kernels/attention.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/matmul.hpp"
#include "kernels/softmax.hpp"
#include "kernels/other_kernels.hpp"
#include "kernels/moe.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/rope.hpp"
#include "memory/mxfp4.hpp"
#include "memory/multi_gpu.hpp"


#ifndef GETP_RUN
#define GETP_RUN

static MultiGPUSystem *multi_gpu_system;

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    multi_gpu_system = (MultiGPUSystem *)malloc(sizeof(MultiGPUSystem));
    build_multi_gpu_system(multi_gpu_system, transformer);
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;

    auto gpu_transformer_0 = multi_gpu_system->gpu_transformers[0];
    for (int pos = 0; pos < MAX_SEQ_LEN; ++pos)
    {
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                             p->initial_context_length, ntk_beta, ntk_alpha,
                             gpu_transformer_0->cpu_buffers.cos_vals + (pos * p->head_dim / 2),
                             gpu_transformer_0->cpu_buffers.sin_vals + (pos * p->head_dim / 2));
    }

    for (int i = 0; i < NUM_GPUS; i++)
    {        
        auto local_gpu_transformer = multi_gpu_system->gpu_transformers[i];
        HIP_CHECK(hipMemcpy(local_gpu_transformer->state.cos_vals, gpu_transformer_0->cpu_buffers.cos_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(local_gpu_transformer->state.sin_vals, gpu_transformer_0->cpu_buffers.sin_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
    }
    
    printf("Multi-GPU system initialized successfully with %d GPUs\n", NUM_GPUS);
}


void finish(Transformer *transformer, Tokenizer *tokenizer)
{
    // Free multi-GPU system safely
    if (multi_gpu_system) {
        free_multi_gpu_system(multi_gpu_system);
        free(multi_gpu_system);
        multi_gpu_system = NULL;
    }
}

// GPU-accelerated neural network functions
void attention_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size)
{
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    
    Timer rms_norm_timer("RMSNorm_attention", true);
    Timer matmul_timer("MatMul_attention", true);
    Timer add_bias_timer("AddBias_attention", true);
    Timer apply_rope_timer("ApplyRoPE_attention", true);
    Timer update_kv_cache_timer("UpdateKVCache_attention", true);
    Timer attention_scores_kernel_timer("AttentionScoresKernel_attention", true);
    Timer add_sinks_kernel_timer("AddSinksKernel_attention", true);
    Timer softmax_kernel_timer("SoftmaxKernel_attention", true);
    Timer matmul_kernel_simple_timer("MatMulKernelSimple_attention", true);
    Timer attention_weighted_sum_kernel_timer("AttentionWeightedSumKernel_attention", true);
    Timer accumulate_kernel_timer("AccumulateKernel_attention", true);

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;

    // RMSNorm - FIXED: Use GPU weight pointer
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<norm_grid, norm_block>>>(
            s->t, s->x, w->rms_attn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    dim3 matmul_grid(batch_size, ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + 31) / 32);
    dim3 matmul_block(32, min(32, THREADS_PER_BLOCK / 32));
    int qkv_weight_offset = layer_idx * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    // Define block and grid dimensions
    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + block_dim.y - 1) / block_dim.y; // Ceiling division
    {
        TIME_SCOPE(matmul_timer);
        // QKV projection using safer matmul kernel - FIXED: Use GPU weight pointer
        matmul(
            s->qkv,
            s->t,
            w->w_qkv + qkv_weight_offset,
            batch_size,
            hidden_dim,
            (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipGetLastError());
    // Add bias - FIXED: Use GPU bias pointer and proper kernel
    int qkv_bias_offset = layer_idx * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    dim3 bias_grid((1LL*batch_size * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    {
        TIME_SCOPE(add_bias_timer);
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            s->qkv, w->b_qkv + qkv_bias_offset, batch_size, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Copy Q, K, V from qkv buffer - SIMPLIFIED AND FIXED
    int q_size = p->n_attn_heads * head_dim;
    int k_size = p->n_kv_heads * head_dim;
    int v_size = p->n_kv_heads * head_dim;

        // Copy Q: shape [batch_size, n_attn_heads * head_dim]
    for (int b = 0; b < batch_size; b++)
    {
        float *src = s->qkv + 1LL*b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *dst = s->q + 1LL*b * q_size;
        HIP_CHECK(hipMemcpy(dst, src, q_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy K: shape [batch_size, n_kv_heads * head_dim]
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < batch_size; b++)
    {
        float *src = s->qkv + 1LL*b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + k_offset;
        float *dst = s->k + 1LL*b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < batch_size; b++)
    {
        float *src = s->qkv + 1LL*b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + v_offset;
        float *dst = s->v + 1LL*b * v_size;
        HIP_CHECK(hipMemcpy(dst, src, v_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Apply rotary embeddings
    dim3 rope_grid(batch_size, p->n_attn_heads);
    dim3 rope_block(head_dim / 2);
    {
        TIME_SCOPE(apply_rope_timer);
        apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
            s->q, s->cos_vals, s->sin_vals, s->positions, batch_size, p->n_attn_heads, head_dim);
        HIP_CHECK(hipGetLastError());
    }

    rope_grid.y = p->n_kv_heads;
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        s->k, s->cos_vals, s->sin_vals, s->positions, batch_size, p->n_kv_heads, head_dim);
    HIP_CHECK(hipGetLastError());

    // Update KV cache - NEW: Proper GPU kernel
    dim3 kv_grid(batch_size, (kv_dim + 31) / 32);
    dim3 kv_block(1, 32);
    {
        TIME_SCOPE(update_kv_cache_timer);
        update_kv_cache_kernel<<<kv_grid, kv_block>>>(
            s->key_cache, s->value_cache, s->k, s->v, s->positions, batch_size,
            p->n_layers, layer_idx, MAX_SEQ_LEN, kv_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Compute attention scores
    dim3 att_grid(batch_size, p->n_attn_heads, (MAX_SEQ_LEN + 31) / 32);
    dim3 att_block(1, 1, 32);
    {
        TIME_SCOPE(attention_scores_kernel_timer);
        attention_scores_kernel<<<att_grid, att_block>>>(
            s->att, s->q, s->key_cache, s->mask, s->positions, batch_size, p->n_attn_heads,
            head_dim, MAX_SEQ_LEN, p->n_layers, layer_idx, p->sliding_window > 0);
        HIP_CHECK(hipGetLastError());
    }

    dim3 sink_grid(batch_size, p->n_attn_heads);
    dim3 sink_block(1);
    {
        TIME_SCOPE(add_sinks_kernel_timer);
        add_sinks_kernel<<<sink_grid, sink_block>>>(
            s->att, w->attn_sinks + layer_idx * p->n_attn_heads, s->positions,
            MAX_SEQ_LEN, p->n_attn_heads);
        HIP_CHECK(hipGetLastError());
    }

    // Softmax attention weights
    dim3 soft_grid(batch_size * p->n_attn_heads);
    dim3 soft_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(softmax_kernel_timer);
        softmax_kernel_variable_len<<<soft_grid, soft_block>>>(
            s->att, s->positions, batch_size, p->n_attn_heads, MAX_SEQ_LEN);
        HIP_CHECK(hipGetLastError());
    }

    // Weighted sum of values
    dim3 wsum_grid(batch_size, p->n_attn_heads);
    dim3 wsum_block(head_dim);
    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        attention_weighted_sum_kernel<<<wsum_grid, wsum_block>>>(
            s->tb, s->att, s->value_cache, s->positions, batch_size, p->n_attn_heads,
            head_dim, MAX_SEQ_LEN, p->n_layers, layer_idx);
        HIP_CHECK(hipGetLastError());
    }
    // Output projection - FIXED: Use GPU weight pointer

    grid_dim.y = (hidden_dim + block_dim.y - 1) / block_dim.y; // Ceiling division

    int attn_out_offset = layer_idx * (head_dim * p->n_attn_heads) * hidden_dim;

    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        // Launch the simple kernel
        matmul(
            s->tb2, s->tb, w->w_o + attn_out_offset, batch_size, head_dim * p->n_attn_heads, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Add bias and residual connection - FIXED: Use GPU bias pointer
    int attn_bias_offset = layer_idx * hidden_dim;
    {
        TIME_SCOPE(accumulate_kernel_timer);
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            s->tb2, w->b_o + attn_bias_offset, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    {
        TIME_SCOPE(accumulate_kernel_timer);
        accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            s->x, s->tb2, 1.0f, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
}

void moe_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size)
{
    // Timer declarations (assuming they are defined elsewhere)
    Timer rms_norm_timer("RMSNorm_moe", true);
    Timer matmul_kernel_simple_timer("MatMulKernelSimple_moe", true);
    Timer add_bias_timer("AddBias_moe", true);
    Timer topk_kernel_timer("TopKKernel_moe", true);
    Timer softmax_kernel_timer("SoftmaxKernel_moe", true);
    Timer gather_expert_inputs_kernel_timer("GatherExpertInputsKernel_moe", true);
    Timer expert_agg_kernel_timer("ExpertAggKernel_moe", true);
    Timer scatter_expert_outputs_kernel_timer("ScatterExpertOutputsKernel_moe", true);
    Timer accumulate_kernel_timer("AccumulateKernel_moe", true);
    Timer split_gate_up_kernel_timer("SplitGateUpKernel_moe", true);
    Timer swiglu_kernel_timer("SwigluKernel_moe", true);

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    
    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    int experts_per_token = p->experts_per_token;

    // FFN RMSNorm
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<norm_grid, norm_block>>>(s->t, s->x, w->rms_ffn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
    }
    HIP_CHECK(hipGetLastError());

    // Router computation & bias
    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        matmul(s->router_score, s->t, w->w_router + layer_idx * hidden_dim * n_experts, batch_size, hidden_dim, n_experts);
    }
    {
        TIME_SCOPE(add_bias_timer);
        add_bias_kernel<<<(batch_size * n_experts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            s->router_score, w->b_router + layer_idx * n_experts, batch_size, n_experts);
    }

    // Top-k expert selection
    {
        TIME_SCOPE(topk_kernel_timer);
        topk_kernel<<<batch_size, 1>>>(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, experts_per_token);
    }

    // Softmax on top-k values to get weights
    {
        TIME_SCOPE(softmax_kernel_timer);
        softmax_kernel<<<batch_size, norm_block>>>(s->topk_v, batch_size, experts_per_token);
    }
    HIP_CHECK(hipGetLastError());

    // Initialize final aggregation buffer
    HIP_CHECK(hipMemset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float)));

    // --- 🚀 OPTIMIZED TWO-STAGE GATHER ALGORITHM 🚀 ---
    // Use pre-allocated persistent buffers instead of dynamic allocation

    // **FIX:** Declare host-side arrays and total_tokens here, before the timed scope
    int h_expert_counts[n_experts];
    int h_expert_offsets[n_experts];
    int total_tokens = 0;

    {
        TIME_SCOPE(gather_expert_inputs_kernel_timer); // Timing the entire gather operation

        // === STAGE 1: COUNT TOKENS PER EXPERT ===
        HIP_CHECK(hipMemset(s->d_expert_counts, 0, n_experts * sizeof(int)));
        dim3 count_grid((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<count_grid, THREADS_PER_BLOCK>>>(
            s->topk_i, s->d_expert_counts, batch_size, experts_per_token);
        HIP_CHECK(hipGetLastError());

        // === CALCULATE OFFSETS (PREFIX SUM ON CPU) ===
        HIP_CHECK(hipMemcpy(h_expert_counts, s->d_expert_counts, n_experts * sizeof(int), hipMemcpyDeviceToHost));

        for (int i = 0; i < n_experts; ++i)
        {
            h_expert_offsets[i] = total_tokens;
            total_tokens += h_expert_counts[i];
        }
        HIP_CHECK(hipMemcpy(s->d_expert_offsets, h_expert_offsets, n_experts * sizeof(int), hipMemcpyHostToDevice));

        // === STAGE 2: PERMUTE INPUTS WITH COALESCED COPY ===
        HIP_CHECK(hipMemset(s->d_expert_write_idx, 0, n_experts * sizeof(int)));
        dim3 permute_grid(batch_size); // One block per token
        dim3 permute_block(256);       // Block size for efficient copying
        int shared_mem_size = experts_per_token * sizeof(int); // For destination_indices
        permute_expert_inputs_kernel<<<permute_grid, permute_block, shared_mem_size>>>(
            s->t, s->topk_i, s->topk_v, s->d_expert_offsets, s->d_expert_write_idx,
            batch_size, hidden_dim, experts_per_token,
            s->expert_input_buffer, s->expert_indices, s->expert_weights);
        HIP_CHECK(hipGetLastError());
    } // End of gather timer scope

    // --- 2. COMPUTE STEP (EXPERT MLPS) ---
    // Loop through each expert to run its MLP on its dedicated slice of the compact buffer.
    for (int expert_id = 0; expert_id < n_experts; expert_id++)
    {
        int h_batch_count = h_expert_counts[expert_id];
        if (h_batch_count == 0)
        {
            continue; // Skip experts with no tokens
        }

        // Calculate pointers to this expert's slice of the data
        int expert_offset = h_expert_offsets[expert_id];
        float *expert_input_ptr = s->expert_input_buffer + expert_offset * hidden_dim;
        float *mlp1_out_ptr = s->mlp1_out + expert_offset * 2 * intermediate_dim;
        float *gate_ptr = s->gate + expert_offset * intermediate_dim;
        float *up_ptr = s->up + expert_offset * intermediate_dim;
        float *gate_up_ptr = s->gate_up + expert_offset * intermediate_dim;
        float *expert_output_ptr = s->expert_output_buffer + expert_offset * hidden_dim;

        // MLP1 (Gate/Up projections)
        {
           TIME_SCOPE(matmul_kernel_simple_timer);
            // Calculate offset for this expert's MXFP4 weights
            size_t expert_offset = (layer_idx * n_experts + expert_id) * (2 * intermediate_dim) * hidden_dim;
            size_t expert_packed_offset = expert_offset / 2;  // 2 FP4 values per byte
            size_t expert_scale_offset = expert_offset / MXFP4_BLOCK_SIZE;
            size_t total_elements = (2 * intermediate_dim) * hidden_dim;
            
            matmul_mxfp4(mlp1_out_ptr, expert_input_ptr,
                        w->w_mlp1_mxfp4 + expert_packed_offset,
                        w->w_mlp1_scales + expert_scale_offset,
                        h_batch_count, hidden_dim, 2 * intermediate_dim,
                        total_elements);
        }

        // Split, add bias, and apply SwiGLU
        dim3 expert_grid((h_batch_count * intermediate_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        {
            TIME_SCOPE(split_gate_up_kernel_timer);
            split_gate_up_kernel<<<expert_grid, THREADS_PER_BLOCK>>>(gate_ptr, up_ptr, mlp1_out_ptr,
                                                                     w->b_mlp1 + (layer_idx * n_experts + expert_id) * (2 * intermediate_dim), h_batch_count, intermediate_dim);
        }
        {
            TIME_SCOPE(swiglu_kernel_timer);
            swiglu_kernel<<<expert_grid, THREADS_PER_BLOCK>>>(gate_ptr, up_ptr, gate_up_ptr,
                                                              h_batch_count, intermediate_dim, p->swiglu_limit);
        }

        // MLP2 (Down projection)
        {
             TIME_SCOPE(matmul_kernel_simple_timer);
            // Calculate offset for this expert's MXFP4 weights
            size_t expert_offset_mlp2 = (layer_idx * n_experts + expert_id) * hidden_dim * intermediate_dim;
            size_t expert_packed_offset_mlp2 = expert_offset_mlp2 / 2;  // 2 FP4 values per byte
            size_t expert_scale_offset_mlp2 = expert_offset_mlp2 / MXFP4_BLOCK_SIZE;
            size_t total_elements_mlp2 = hidden_dim * intermediate_dim;
            
            matmul_mxfp4(expert_output_ptr, gate_up_ptr,
                        w->w_mlp2_mxfp4 + expert_packed_offset_mlp2,
                        w->w_mlp2_scales + expert_scale_offset_mlp2,
                        h_batch_count, intermediate_dim, hidden_dim,
                        total_elements_mlp2);
        }

        // Add MLP2 bias
        {
            TIME_SCOPE(add_bias_timer);
            dim3 bias_grid((1LL*h_batch_count * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
                expert_output_ptr, w->b_mlp2 + (layer_idx * n_experts + expert_id) * hidden_dim, h_batch_count, hidden_dim);
            HIP_CHECK(hipGetLastError());
        }
    }

    // --- 3. SCATTER STEP ---
    // The scatter kernel works as before, but we launch it over the total number of processed tokens.
    if (total_tokens > 0)
    {
        dim3 scatter_grid((total_tokens * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        {
            TIME_SCOPE(scatter_expert_outputs_kernel_timer);
            scatter_expert_outputs_kernel<<<scatter_grid, THREADS_PER_BLOCK>>>(
                s->e_agg, s->expert_output_buffer, s->expert_indices, s->expert_weights,
                total_tokens, hidden_dim);
            HIP_CHECK(hipGetLastError());
        }
    }

    // --- FINAL RESIDUAL CONNECTION ---
    {
        TIME_SCOPE(accumulate_kernel_timer);
        accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            s->x, s->e_agg, 1.0f, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    // No need to free buffers - they're persistent and reused across calls
}

float *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
{
    Timer copy_embed_timer("copy_embeddings_forward", true);
    Timer rms_norm_timer("RMSNorm_forward", true);
    Timer matmul_kernel_simple_timer("MatMulKernelSimple_forward", true);
    
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;

    int hidden_dim = p->hidden_dim;

    // Copy tokens to GPU
    HIP_CHECK(hipMemcpy(s->current_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s->positions, cpu_buf->positions, batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Copy token embeddings
    dim3 embed_grid((batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    {
        TIME_SCOPE(copy_embed_timer);
        copy_embeddings_kernel<<<embed_grid, THREADS_PER_BLOCK>>>(
            s->x, w->token_embedding_table, s->current_tokens, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    // Forward through all layers - UNCOMMENTED: All layers now enabled
    for (int l = 0; l < p->n_layers; l++)
    {
        attention_gpu(gpu_t, l, batch_size);
        moe_gpu(gpu_t, l, batch_size);
    }
    // Final RMSNorm - UNCOMMENTED: Now enabled with GPU weight pointer
    dim3 final_norm_grid(batch_size);
    dim3 final_norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<final_norm_grid, final_norm_block>>>(
            s->x, s->x, w->rms_out_w, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = (p->vocab_size + block_dim.y - 1) / block_dim.y; // Ceiling division

    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        // Launch the simple kernel
        matmul(
            s->logits, s->x, w->out, batch_size, hidden_dim, p->vocab_size);
    }
    // Copy logits back to CPU (you might want to keep this on GPU for sampling)
    float *h_logits = nullptr;
    if (!h_logits)
    {
        h_logits = (float *)malloc(batch_size * p->vocab_size * sizeof(float));
    }
    HIP_CHECK(hipMemcpy(h_logits, s->logits, batch_size * p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return h_logits;
}

long long continuous_batching_inference(GPUTransformer *gpu_t, Tokenizer *tokenizer,
                                       Sampler *sampler, Requests *requests)
{
    Config *p = &gpu_t->config;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
    GPURunState *state = &gpu_t->state;
    long long total_tokens_generated = 0;
    
    // Initialize
    cpu_buf->next_request_idx = 0;
    
    // Initialize all slots to inactive
    for (int slot = 0; slot < BATCH_SIZE; slot++) {
        cpu_buf->slot_active_cpu[slot] = false;
        cpu_buf->request_mapping_cpu[slot] = -1;
        cpu_buf->seq_lengths_cpu[slot] = 0;
        cpu_buf->positions[slot] = 0;
        cpu_buf->finished[slot] = true;
    }
    
    // Fill initial batch with first requests
    for (int slot = 0; slot < BATCH_SIZE && cpu_buf->next_request_idx < requests->num_reqs; slot++) {
        int req_idx = cpu_buf->next_request_idx++;
        const char *input_seq = get_str_req_ptr(requests, req_idx);
        
        // Encode prompt
        encode(tokenizer, input_seq, -1, -1, cpu_buf->prompt_tokens[slot],
               &cpu_buf->prompt_lens[slot], p->initial_context_length);
        
        if (cpu_buf->prompt_lens[slot] < 1) {
            fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
            cpu_buf->prompt_lens[slot] = 1;
            cpu_buf->prompt_tokens[slot][0] = 1; // BOS token
        }
        
        // Initialize slot
        cpu_buf->request_mapping_cpu[slot] = req_idx;
        cpu_buf->slot_active_cpu[slot] = true;
        cpu_buf->seq_lengths_cpu[slot] = 0;
        cpu_buf->positions[slot] = 0;
        cpu_buf->finished[slot] = false;
        cpu_buf->current_tokens[slot] = cpu_buf->prompt_tokens[slot][0];
    }
    
    // Copy initial state to GPU
    HIP_CHECK(hipMemcpy(state->slot_active, cpu_buf->slot_active_cpu, 
                       BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state->request_mapping, cpu_buf->request_mapping_cpu,
                       BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state->seq_lengths, cpu_buf->seq_lengths_cpu,
                       BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
    
    // Main generation loop
    int max_steps = requests->max_seq_len;
    bool has_active_slots = true;
    
    while (has_active_slots) {
        // Forward pass on all slots (inactive ones will be skipped internally)
        float *logits = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, BATCH_SIZE);
        
        // Process each slot
        has_active_slots = false;
        for (int slot = 0; slot < BATCH_SIZE; slot++) {
            if (!cpu_buf->slot_active_cpu[slot]) continue;
            
            has_active_slots = true;
            int req_idx = cpu_buf->request_mapping_cpu[slot];
            int pos = cpu_buf->positions[slot];
            float *logits_slot = logits + slot * p->vocab_size;
            
            int next_token;
            // Advance position first  
            pos++;
            
            if (pos < cpu_buf->prompt_lens[slot]) {
                // Still processing prompt - force next prompt token
                next_token = cpu_buf->prompt_tokens[slot][pos];
            } else {
                // Generate new token
                next_token = sample(sampler, logits_slot);
                
                // Save generated token
                int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                int gen_pos = pos - cpu_buf->prompt_lens[slot];
                if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                    output_tokens[gen_pos] = next_token;
                    total_tokens_generated++;
                }
            }
            
            // Check for completion
            bool completed = (next_token == 199999 || next_token == 200002 || 
                            pos >= max_steps - 1 || pos >= MAX_SEQ_LEN - 2);
            
            if (completed) {
                if (next_token == 199999 || next_token == 200002){
                    fprintf(stderr, "Request %d received EOS token %d at position %d\n", req_idx, next_token, pos);
                }
                // fprintf(stderr, "Request %d completed at position %d with token %d\n", req_idx, pos, next_token);
                // Mark end of generation
                int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                int gen_pos = pos - cpu_buf->prompt_lens[slot] + 1;
                if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                    output_tokens[gen_pos] = -1; // End marker
                }
                
                // Try to assign a new request to this slot
                if (cpu_buf->next_request_idx < requests->num_reqs) {
                    // Get next request
                    req_idx = cpu_buf->next_request_idx++;
                    const char *input_seq = get_str_req_ptr(requests, req_idx);
                    
                    // Encode prompt
                    encode(tokenizer, input_seq, -1, -1, cpu_buf->prompt_tokens[slot],
                           &cpu_buf->prompt_lens[slot], p->initial_context_length);
                    
                    if (cpu_buf->prompt_lens[slot] < 1) {
                        fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                        cpu_buf->prompt_lens[slot] = 1;
                        cpu_buf->prompt_tokens[slot][0] = 1; // BOS token
                    }
                    
                    // Reinitialize slot with new request
                    cpu_buf->request_mapping_cpu[slot] = req_idx;
                    cpu_buf->positions[slot] = 0;
                    cpu_buf->seq_lengths_cpu[slot] = 0;
                    cpu_buf->current_tokens[slot] = cpu_buf->prompt_tokens[slot][0];
                    // Slot remains active
                } else {
                    // No more requests, deactivate slot
                    cpu_buf->slot_active_cpu[slot] = false;
                    cpu_buf->request_mapping_cpu[slot] = -1;
                }
            } else {
                // Continue generation
                cpu_buf->positions[slot] = pos;
                cpu_buf->seq_lengths_cpu[slot]++;
                cpu_buf->current_tokens[slot] = next_token;
            }
        }
        
        // Update GPU state for next iteration
        HIP_CHECK(hipMemcpy(state->slot_active, cpu_buf->slot_active_cpu,
                           BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->seq_lengths, cpu_buf->seq_lengths_cpu,
                           BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->positions, cpu_buf->positions,
                           BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
    }
    
    // Print results for all requests
    for (int req_idx = 0; req_idx < requests->num_reqs; req_idx++) {
        const char *input_seq = get_str_req_ptr(requests, req_idx);
        int *output_tokens = get_tok_gen_ptr(requests, req_idx);
        
        // Print the original prompt string
        safe_printf(input_seq);
        printf("!");
        
        // Find last token of prompt for context
        int prompt_len = strlen(input_seq);
        int *temp_tokens = (int *)malloc((MAX_SEQ_LEN + 3) * sizeof(int));
        int temp_len;
        encode(tokenizer, input_seq, -1, -1, temp_tokens, &temp_len, p->initial_context_length);
        int last_prompt_token = temp_tokens[temp_len - 1];
        free(temp_tokens);
        
        // Decode and print generated tokens
        int prev_token = last_prompt_token;
        for (int i = 0; i < max_steps; ++i) {
            int token = output_tokens[i];
            if (token == -1) break;
            
            const char *piece = decode_piece(tokenizer, prev_token, token);
            safe_printf(piece);
            prev_token = token;
        }
        printf("\n");
    }
    fflush(stdout);
    
    return total_tokens_generated;
}

long long batched_generate_gpu(GPUTransformer *gpu_t, Tokenizer *tokenizer,
                               Sampler *sampler, Requests *requests)
{

    Config *p = &gpu_t->config;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
    long long total_tokens_generated = 0;

    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE)
    {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                     ? (requests->num_reqs - req_start)
                                     : BATCH_SIZE;

        // Initialize batch
        for (int b = 0; b < current_batch_size; b++)
        {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            // Encode prompt
            encode(tokenizer, input_seq, -1, -1, cpu_buf->prompt_tokens[b],
                   &cpu_buf->prompt_lens[b], p->initial_context_length);

            if (cpu_buf->prompt_lens[b] < 1)
            {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                cpu_buf->prompt_lens[b] = 1;
                cpu_buf->prompt_tokens[b][0] = 1; // BOS token
            }

            // Initialize sequence state
            cpu_buf->positions[b] = 0;
            cpu_buf->finished[b] = false;
            cpu_buf->current_tokens[b] = cpu_buf->prompt_tokens[b][0];
        }

        // Generation loop
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;

        for (int step = 0; step < max_steps && alive > 0; step++)
        {
            // Forward pass on GPU
            float *logits = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, current_batch_size);

            // Sample next tokens (on CPU for now, could be moved to GPU)
            for (int b = 0; b < current_batch_size; b++)
            {
                if (cpu_buf->finished[b])
                    continue;

                int req_idx = req_start + b;
                int pos = cpu_buf->positions[b];
                float *logits_b = logits + b * p->vocab_size;

                int next_token;
                // Advance position first
                pos++;
                
                if (pos < cpu_buf->prompt_lens[b])
                {
                    // Still processing prompt - force next prompt token
                    next_token = cpu_buf->prompt_tokens[b][pos];
                }
                else
                {
                    // Generate new token
                    next_token = sample(sampler, logits_b);

                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - cpu_buf->prompt_lens[b];
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }

                // Check for termination
                if (next_token == 199999 || next_token == 200002 || pos >= max_steps - 1)
                {
                    --alive;
                    cpu_buf->finished[b] = true;
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - cpu_buf->prompt_lens[b] + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }

                // Update for next iteration
                cpu_buf->positions[b] = pos;
                cpu_buf->current_tokens[b] = next_token;
            }
        }

        // Print results
        for (int b = 0; b < current_batch_size; b++)
        {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            int *output_tokens = get_tok_gen_ptr(requests, req_idx);

            // Print the original prompt string
            safe_printf(input_seq);
            printf("!");

            // Decode and print generated tokens
            int last_prompt_token = cpu_buf->prompt_tokens[b][cpu_buf->prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0; i < max_steps; ++i)
            {
                int token = output_tokens[i];
                if (token == -1)
                    break;

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


// Multi-GPU implementation functions
void *gpu_worker_thread(void *args)
{
    GPUWorkerArgs *worker_args = (GPUWorkerArgs *)args;
    
    fprintf(stderr, "GPU %d worker thread started\n", worker_args->gpu_id);
    
    // Set this thread to use the assigned GPU - Add error handling
    hipError_t set_device_err = hipSetDevice(worker_args->gpu_id);
    if (set_device_err != hipSuccess)
    {
        fprintf(stderr, "Error: Failed to set GPU device %d in worker thread: %s\n", 
                worker_args->gpu_id, hipGetErrorString(set_device_err));
        worker_args->tokens_generated = 0;
        return NULL;
    }
    
    // Synchronize to ensure device is properly set
    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess)
    {
        fprintf(stderr, "Error: Failed to synchronize GPU device %d: %s\n", 
                worker_args->gpu_id, hipGetErrorString(sync_err));
        worker_args->tokens_generated = 0;
        return NULL;
    }
    
    // Verify we're on the correct device
    int current_device;
    hipError_t get_device_err = hipGetDevice(&current_device);
    if (get_device_err != hipSuccess || current_device != worker_args->gpu_id)
    {
        fprintf(stderr, "Error: Device verification failed for GPU %d (current: %d)\n", 
                worker_args->gpu_id, current_device);
        worker_args->tokens_generated = 0;
        return NULL;
    }
    
    // Initialize HIP context for this GPU in this thread
    hipError_t ctx_err = hipFree(0);  // This forces context creation
    if (ctx_err != hipSuccess)
    {
        fprintf(stderr, "Error: Failed to initialize HIP context for GPU %d: %s\n", 
                worker_args->gpu_id, hipGetErrorString(ctx_err));
        worker_args->tokens_generated = 0;
        return NULL;
    }
    
    fprintf(stderr, "GPU %d worker thread: device context initialized\n", worker_args->gpu_id);
    // Create a subset Requests structure for this GPU
    Requests gpu_requests;
    gpu_requests.num_reqs = worker_args->num_requests;
    gpu_requests.max_seq_len = worker_args->requests->max_seq_len;
    gpu_requests.max_token_len = worker_args->requests->max_token_len;

    // Calculate correct offsets for subset of requests this GPU should process
    size_t str_offset = worker_args->start_req_idx * worker_args->requests->max_token_len * (worker_args->requests->max_seq_len + 1);
    size_t tok_offset = worker_args->start_req_idx * (worker_args->requests->max_seq_len + 1);

    gpu_requests.str_reqs = worker_args->requests->str_reqs + str_offset;
    gpu_requests.tok_gens = worker_args->requests->tok_gens + tok_offset;

    fprintf(stderr, "GPU %d worker: Processing %d requests (from idx %d)\n", 
            worker_args->gpu_id, gpu_requests.num_reqs, worker_args->start_req_idx);
    fprintf(stderr, "GPU %d worker: str_offset=%zu, tok_offset=%zu\n", 
            worker_args->gpu_id, str_offset, tok_offset);

    // Run inference on this GPU's subset of requests
    long long tokens_generated = continuous_batching_inference(
        worker_args->gpu_transformer,
        worker_args->tokenizer,
        worker_args->sampler,
        &gpu_requests);

    worker_args->tokens_generated = tokens_generated;

    // Final synchronization before thread exit
    hipError_t final_sync = hipDeviceSynchronize();
    if (final_sync != hipSuccess)
    {
        fprintf(stderr, "Warning: Final sync failed for GPU %d: %s\n", 
                worker_args->gpu_id, hipGetErrorString(final_sync));
    }

    fprintf(stderr, "GPU %d worker: Generated %lld tokens, exiting cleanly\n", 
            worker_args->gpu_id, tokens_generated);

    return NULL;
}

long long multi_gpu_inference(MultiGPUSystem *multi_gpu, Tokenizer *tokenizer,
                              Sampler *sampler, Requests *requests)
{

    if (!multi_gpu->initialized)
    {
        fprintf(stderr, "Error: MultiGPUSystem not initialized\n");
        return 0;
    }

    int available_gpus = get_available_gpu_count();
    
    // Only use GPUs that were actually initialized
    int usable_gpus = 0;
    for (int i = 0; i < available_gpus && i < NUM_GPUS; i++)
    {
        if (multi_gpu->gpu_transformers[i] != NULL)
        {
            usable_gpus++;
        }
        else
        {
            fprintf(stderr, "Warning: GPU %d not initialized, skipping\n", i);
            break; // Stop at first uninitialized GPU
        }
    }
    
    if (usable_gpus == 0)
    {
        fprintf(stderr, "Error: No usable GPUs found\n");
        return 0;
    }
    
    fprintf(stderr, "Using %d out of %d available GPUs\n", usable_gpus, available_gpus);

    // Calculate how to distribute requests across usable GPUs
    int requests_per_gpu = requests->num_reqs / usable_gpus;
    int remaining_requests = requests->num_reqs % usable_gpus;

    fprintf(stderr,"Distributing %d requests across %d GPUs (%d base + remainder)\n",
           requests->num_reqs, usable_gpus, requests_per_gpu);

    // Set up worker arguments for each usable GPU
    int current_start_idx = 0;
    for (int gpu_id = 0; gpu_id < usable_gpus; gpu_id++)
    {
        GPUWorkerArgs *args = &multi_gpu->worker_args[gpu_id];

        args->gpu_id = gpu_id;
        args->gpu_transformer = multi_gpu->gpu_transformers[gpu_id];
        args->tokenizer = tokenizer;
        args->sampler = sampler;
        args->requests = requests;
        args->start_req_idx = current_start_idx;

        // Give extra requests to first few GPUs if there's a remainder
        args->num_requests = requests_per_gpu + (gpu_id < remaining_requests ? 1 : 0);
        args->tokens_generated = 0;

        current_start_idx += args->num_requests;

        fprintf(stderr,"GPU %d: will process requests %d to %d (%d total)\n",
               gpu_id, args->start_req_idx,
               args->start_req_idx + args->num_requests - 1,
               args->num_requests);
    }

    // Launch worker threads
    fprintf(stderr,"Launching worker threads...\n");
    for (int gpu_id = 0; gpu_id < usable_gpus; gpu_id++)
    {
        int result = pthread_create(&multi_gpu->worker_threads[gpu_id],
                                    NULL,
                                    gpu_worker_thread,
                                    &multi_gpu->worker_args[gpu_id]);
        if (result != 0)
        {
            fprintf(stderr, "Error creating thread for GPU %d: %d\n", gpu_id, result);
            // Clean up already created threads
            for (int i = 0; i < gpu_id; i++)
            {
                pthread_cancel(multi_gpu->worker_threads[i]);
            }
            return 0;
        }
        
        // Small delay to avoid race conditions during thread startup
        usleep(100000); // 100ms delay between thread launches
        fprintf(stderr, "Thread for GPU %d created successfully\n", gpu_id);
    }

    // Wait for all threads to complete and collect results
    long long total_tokens_generated = 0;
    fprintf(stderr,"Waiting for worker threads to complete...\n");

    for (int gpu_id = 0; gpu_id < usable_gpus; gpu_id++)
    {
        void *thread_result;
        int result = pthread_join(multi_gpu->worker_threads[gpu_id], &thread_result);

        if (result != 0)
        {
            fprintf(stderr, "Error joining thread for GPU %d: %d\n", gpu_id, result);
        }
        else
        {
            total_tokens_generated += multi_gpu->worker_args[gpu_id].tokens_generated;
            fprintf(stderr, "GPU %d completed: %lld tokens generated\n",
                   gpu_id, multi_gpu->worker_args[gpu_id].tokens_generated);
        }
    }

    fprintf(stderr,"Multi-GPU inference completed: %lld total tokens generated\n", total_tokens_generated);
    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    // Use continuous batching for better throughput
    return multi_gpu_inference(multi_gpu_system, tokenizer, sampler, requests);
}

/*
 * REFACTORING NOTES:
 * 
 * This file has been partially refactored to group global variables into structs
 * similar to the CPU version in run.cpp. The main changes include:
 * 
 * 1. Created GPUTransformerWeights struct to hold all GPU weight pointers
 * 2. Created GPURunState struct to hold all GPU activation buffers
 * 3. Created CPUBuffers struct to hold CPU-side buffers
 * 4. Created GPUTransformer struct to tie everything together
 * 5. Added proper allocation and deallocation functions for each struct
 * 
 * TODO: The actual kernel calls throughout the file still reference the old
 * global variables (like d_x, d_t, etc.). These need to be updated to use
 * the struct members (e.g., gpu_state->x, gpu_state->t, etc.) and the
 * functions need to be updated to accept the struct parameters.
 * 
 * This refactoring provides better memory management, cleaner code organization,
 * and follows the same pattern as the CPU implementation.
 */

#endif // GETP_RUN