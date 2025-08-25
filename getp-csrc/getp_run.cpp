// TODO: Modify this file to optimize end-to-end throughput with HIP GPU acceleration

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
#include "config.hpp"
#include "utils.hpp"
#include "batch_manager.hpp"
#include "kernels/attention.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/matmul.hpp"
#include "kernels/softmax.hpp"
#include "kernels/other_kernels.hpp"
#include "kernels/moe.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/rope.hpp"
#include "memory/mxfp4.hpp"


#ifndef GETP_RUN
#define GETP_RUN



// GPU Transformer Weights struct - stores all model weights on GPU in bfloat16 format
typedef struct {
  // Embedding weights
  __hip_bfloat16 *token_embedding_table; // (vocab_size, hidden_dim)
  
  // RMSNorm weights  
  __hip_bfloat16 *rms_attn_w; // (n_layers, hidden_dim)
  __hip_bfloat16 *rms_ffn_w;  // (n_layers, hidden_dim)
  __hip_bfloat16 *rms_out_w;  // (hidden_dim,)
  
  // Attention weights
  __hip_bfloat16 *w_qkv;      // (n_layers, head_dim * (n_attn_heads + 2 * n_kv_heads), hidden_dim)
  __hip_bfloat16 *b_qkv;      // (n_layers, head_dim * (n_attn_heads + 2 * n_kv_heads))
  __hip_bfloat16 *w_o;        // (n_layers, hidden_dim, head_dim * n_attn_heads)
  __hip_bfloat16 *b_o;        // (n_layers, hidden_dim)
  __hip_bfloat16 *attn_sinks; // (n_layers, n_attn_heads)
  
  // MoE router weights
  __hip_bfloat16 *w_router;   // (n_layers, hidden_dim, n_experts)
  __hip_bfloat16 *b_router;   // (n_layers, n_experts)
  
  // MoE weights now use MXFP4 quantization
  uint8_t *w_mlp1_mxfp4, *w_mlp2_mxfp4;  // Packed MXFP4 indices
  float *w_mlp1_scales, *w_mlp2_scales;   // MXFP4 block scales
  __hip_bfloat16 *b_mlp1, *b_mlp2;        // Biases remain in bfloat16
  __hip_bfloat16 *out_w;
  // Output weights
  __hip_bfloat16 *out;        // (vocab_size, hidden_dim)
} GPUTransformerWeights;

// GPU Run State struct - stores all activation buffers on GPU
typedef struct {
  // Basic activation buffers
  float *x;                    // activation at current time stamp (batch_size, hidden_dim)
  float *t;                    // residual branch buffer (batch_size, hidden_dim)
  float *tb;                   // temp buffer (batch_size, head_dim * n_attn_heads)
  float *tb2;                  // temp buffer (batch_size, hidden_dim)
  float *temp_buffer;          // general purpose temp buffer (batch_size, hidden_dim)
  
  // Attention buffers
  float *qkv;                  // QKV buffer (batch_size, head_dim * (n_attn_heads + 2 * n_kv_heads))
  float *q;                    // query buffer (batch_size, n_attn_heads * head_dim)
  float *k;                    // key buffer (batch_size, n_kv_heads * head_dim)
  float *v;                    // value buffer (batch_size, n_kv_heads * head_dim)
  float *att;                  // attention scores (batch_size, n_attn_heads, seq_len)
  float *mask;                 // attention mask (seq_len, seq_len)
  
  // Legacy KV cache - kept for compatibility (now using BF16 for 50% memory reduction)
  __hip_bfloat16 *key_cache;   // (batch_size, n_layers, seq_len, kv_dim)
  __hip_bfloat16 *value_cache; // (batch_size, n_layers, seq_len, kv_dim)
  
  // NEW: Paged KV cache - replaces the above for better memory efficiency
  __hip_bfloat16 *paged_key_cache;   // (MAX_BLOCKS, n_layers, PAGE_SIZE, kv_dim)  
  __hip_bfloat16 *paged_value_cache; // (MAX_BLOCKS, n_layers, PAGE_SIZE, kv_dim)
  int *block_table;                  // (BATCH_SIZE, MAX_PAGES_PER_SEQ) - maps logical pages to physical blocks
  
  // RoPE buffers
  float *cos_vals;             // (head_dim/2, seq_len)
  float *sin_vals;             // (head_dim/2, seq_len)
  
  // MoE buffers
  float *router_score;         // router scores (batch_size, n_experts)
  float *topk_v;               // top-k expert weights (batch_size, experts_per_token)
  int *topk_i;                 // top-k expert indices (batch_size, experts_per_token)
  float *mlp1_out;             // MLP1 output (batch_size * experts_per_token, 2 * intermediate_dim)
  float *gate;                 // gate values (batch_size * experts_per_token, intermediate_dim)
  float *up;                   // up projection (batch_size * experts_per_token, intermediate_dim)
  float *gate_up;              // gate * up (batch_size * experts_per_token, intermediate_dim)
  float *e_agg;                // expert aggregation buffer (batch_size, hidden_dim)
  float *expert_input_buffer;  // input buffer for experts (batch_size * experts_per_token, hidden_dim)
  float *expert_output_buffer; // output buffer for experts (batch_size * experts_per_token, hidden_dim)
  int *expert_indices;         // expert indices for scatter/gather (batch_size * experts_per_token)
  float *expert_weights;       // expert weights for scatter/gather (batch_size * experts_per_token)
  int *batch_count;            // batch count for expert processing
  
  // Token and position buffers
  int *current_tokens;         // current tokens (batch_size)
  int *positions;              // current positions (batch_size)
  
  // Output buffer
  float *logits;               // output logits (batch_size, vocab_size)
  
  // Continuous batching fields
  int *seq_lengths;            // Current sequence length for each slot [BATCH_SIZE]
  bool *slot_active;           // Whether slot is processing a request [BATCH_SIZE]
  int *request_mapping;        // Maps batch slot -> request index in Requests [BATCH_SIZE]
  
  // Paged attention runtime flag
  bool use_paged_attention;    // Whether to use paged attention or legacy linear cache
} GPURunState;

// CPU buffers for warmup and host-side operations
typedef struct {
  float *cos_vals;             // RoPE cosine values (CPU)
  float *sin_vals;             // RoPE sine values (CPU)
  int **prompt_tokens;         // prompt tokens for each sequence
  int *current_tokens;         // current tokens (CPU copy)
  bool *finished;              // finished flags for each sequence
  int *positions;              // current positions (CPU copy)
  int *prompt_lens;            // prompt lengths
  
  // Continuous batching fields
  int next_request_idx;        // Index of next unprocessed request (0 to num_reqs-1)
  bool *slot_active_cpu;       // CPU mirror of slot_active for quick access
  int *seq_lengths_cpu;        // CPU mirror of seq_lengths
  int *request_mapping_cpu;   // CPU mirror of request_mapping
} CPUBuffers;

// Main GPU Transformer struct
typedef struct {
  Config config;                    // model configuration
  GPUTransformerWeights weights;     // GPU weights
  GPURunState state;                // GPU run state buffers
  CPUBuffers cpu_buffers;           // CPU buffers for host operations
  PagedAttentionManager *paged_attn_mgr; // Paged attention memory manager
} GPUTransformer;

// Global variables for direct access in batched_generate_gpu
static GPUTransformer* gpu_transformer;

// Memory allocation functions
void malloc_gpu_run_state(GPURunState *s, Config *p, bool use_paged_attention)
{
    int kv_dim = p->head_dim * p->n_kv_heads;
    int expert_per_token = p->experts_per_token;

    // Initialize pointers to NULL
    s->mask = NULL;
    s->paged_key_cache = NULL;
    s->paged_value_cache = NULL;
    s->block_table = NULL;
    s->use_paged_attention = use_paged_attention;

    // Check memory requirements and print for debugging
    size_t total_memory = 0;
    size_t batch_hidden = BATCH_SIZE * p->hidden_dim * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);
    
    if (use_paged_attention) {
        // For paged attention, allocate KV cache in blocks  
        printf("Allocating GPU memory with PAGED ATTENTION: batch_size=%d, hidden_dim=%d\n",
               BATCH_SIZE, p->hidden_dim);
        printf("Paged KV cache: %d blocks, %d tokens per page\n", MAX_BLOCKS, PAGE_SIZE);
        
        // Allocate paged KV cache
        size_t paged_kv_cache_size = MAX_BLOCKS * p->n_layers * PAGE_SIZE * kv_dim * sizeof(__hip_bfloat16);
        HIP_CHECK(hipMalloc((void **)&s->paged_key_cache, paged_kv_cache_size));
        HIP_CHECK(hipMalloc((void **)&s->paged_value_cache, paged_kv_cache_size));
        HIP_CHECK(hipMalloc((void **)&s->block_table, BATCH_SIZE * MAX_PAGES_PER_SEQ * sizeof(int)));
        
        printf("Paged KV cache size: %.2f MB (vs %.2f MB for linear cache)\n", 
               (2 * paged_kv_cache_size) / (1024.0 * 1024.0),
               (2 * BATCH_SIZE * p->n_layers * p->seq_len * kv_dim * sizeof(__hip_bfloat16)) / (1024.0 * 1024.0));
        
        // Initialize paged memory
        HIP_CHECK(hipMemset(s->paged_key_cache, 0, paged_kv_cache_size));
        HIP_CHECK(hipMemset(s->paged_value_cache, 0, paged_kv_cache_size));
        HIP_CHECK(hipMemset(s->block_table, -1, BATCH_SIZE * MAX_PAGES_PER_SEQ * sizeof(int)));
        
        // Set legacy KV cache pointers to NULL since we're using paged attention
        s->key_cache = NULL;
        s->value_cache = NULL;
    } else {
        // Legacy linear KV cache allocation
        size_t kv_cache_size = BATCH_SIZE * p->n_layers * p->seq_len * kv_dim * sizeof(__hip_bfloat16);
        printf("Allocating GPU memory with LINEAR KV cache: batch_size=%d, hidden_dim=%d, seq_len=%d\n",
               BATCH_SIZE, p->hidden_dim, p->seq_len);
        printf("Linear KV cache size per batch (BF16): %zu MB\n", kv_cache_size / (1024 * 1024));
        
        // Allocate linear KV cache
        HIP_CHECK(hipMalloc((void **)&s->key_cache, kv_cache_size));
        HIP_CHECK(hipMalloc((void **)&s->value_cache, kv_cache_size));
        
        // Initialize linear KV cache
        HIP_CHECK(hipMemset(s->key_cache, 0, kv_cache_size));
        HIP_CHECK(hipMemset(s->value_cache, 0, kv_cache_size));
    }

    // Allocate GPU memory with error checking
    HIP_CHECK(hipMalloc((void **)&s->x, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->t, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->tb, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->tb2, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->qkv, batch_qkv));
    HIP_CHECK(hipMalloc((void **)&s->q, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->k, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->v, BATCH_SIZE * kv_dim * sizeof(float)));

    // NEW: Buffers for GPU scatter-gather MoE
    // Max possible items for one expert is the entire batch
    HIP_CHECK(hipMalloc((void **)&s->expert_indices, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->expert_weights, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->batch_count, sizeof(int)));
    // This buffer holds the expert's final output before scattering
    HIP_CHECK(hipMalloc((void **)&s->expert_output_buffer, BATCH_SIZE * p->hidden_dim * sizeof(float) * expert_per_token));

    HIP_CHECK(hipMalloc((void **)&s->att, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->logits, BATCH_SIZE * p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->router_score, BATCH_SIZE * p->n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_v, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_i, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->mlp1_out, BATCH_SIZE * 2 * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->gate, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->up, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->gate_up, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->e_agg, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->positions, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->cos_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->sin_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->expert_input_buffer, batch_hidden * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->temp_buffer, batch_hidden));
    
    // Allocate continuous batching fields
    HIP_CHECK(hipMalloc((void **)&s->seq_lengths, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->slot_active, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMalloc((void **)&s->request_mapping, BATCH_SIZE * sizeof(int)));

    // Initialize all allocated memory to zero
    HIP_CHECK(hipMemset(s->x, 0, batch_hidden));
    HIP_CHECK(hipMemset(s->t, 0, batch_hidden));
    HIP_CHECK(hipMemset(s->tb, 0, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMemset(s->tb2, 0, batch_hidden));
    HIP_CHECK(hipMemset(s->qkv, 0, batch_qkv));
    HIP_CHECK(hipMemset(s->q, 0, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float)));
    HIP_CHECK(hipMemset(s->k, 0, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(s->v, 0, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(s->att, 0, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMemset(s->logits, 0, BATCH_SIZE * p->vocab_size * sizeof(float)));
    
    // Initialize continuous batching fields
    HIP_CHECK(hipMemset(s->seq_lengths, 0, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMemset(s->slot_active, 0, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMemset(s->request_mapping, -1, BATCH_SIZE * sizeof(int)));

    if (p->sliding_window > 0)
    {
        size_t mask_size = p->seq_len * p->seq_len * sizeof(float);
        HIP_CHECK(hipMalloc((void **)&s->mask, mask_size));

        // Initialize mask on GPU if needed
        float *h_mask = (float *)malloc(mask_size);
        if (!h_mask)
        {
            fprintf(stderr, "Failed to allocate host memory for mask\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < p->seq_len; i++)
        {
            for (int j = 0; j < p->seq_len; j++)
            {
                h_mask[i * p->seq_len + j] = (i - j >= p->sliding_window) ? -INFINITY : 0.0f;
            }
        }
        HIP_CHECK(hipMemcpy(s->mask, h_mask, mask_size, hipMemcpyHostToDevice));
        free(h_mask);
    }
}

void malloc_gpu_weights(GPUTransformerWeights *w, Config *p)
{
    // Allocate GPU memory for all weights in bfloat16 format
    HIP_CHECK(hipMalloc((void **)&w->token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->rms_attn_w, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->rms_ffn_w, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->rms_out_w, p->hidden_dim * sizeof(__hip_bfloat16)));

    int qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_qkv, qkv_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_qkv, p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim * sizeof(__hip_bfloat16)));

    int attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_o, attn_out_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_o, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->w_router, p->n_layers * p->hidden_dim * p->n_experts * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_router, p->n_layers * p->n_experts * sizeof(__hip_bfloat16)));

     // MoE weights allocation with MXFP4 quantization
    size_t mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    size_t mlp1_packed_size = (mlp1_size + 1) / 2;  // 2 FP4 values per byte
    size_t mlp1_num_blocks = (mlp1_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_mxfp4, mlp1_packed_size));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_scales, mlp1_num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, p->n_layers * p->n_experts * (2 * p->intermediate_dim) * sizeof(__hip_bfloat16)));

    size_t mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    size_t mlp2_packed_size = (mlp2_size + 1) / 2;  // 2 FP4 values per byte
    size_t mlp2_num_blocks = (mlp2_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_mxfp4, mlp2_packed_size));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_scales, mlp2_num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2, p->n_layers * p->n_experts * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->out, p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->attn_sinks, p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));
}

void copy_weights_to_gpu(Transformer *transformer, GPUTransformerWeights *gpu_weights)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    // Convert and copy embedding weights
    size_t embedding_size = p->vocab_size * p->hidden_dim;
    __hip_bfloat16 *h_embedding_bf16 = (__hip_bfloat16 *)malloc(embedding_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->token_embedding_table, h_embedding_bf16, embedding_size);
    HIP_CHECK(hipMemcpy(gpu_weights->token_embedding_table, h_embedding_bf16, embedding_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_embedding_bf16);

    // Convert and copy normalization weights
    size_t rms_attn_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_rms_attn_bf16 = (__hip_bfloat16 *)malloc(rms_attn_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_attn_w, h_rms_attn_bf16, rms_attn_size);
    HIP_CHECK(hipMemcpy(gpu_weights->rms_attn_w, h_rms_attn_bf16, rms_attn_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_attn_bf16);

    size_t rms_ffn_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_rms_ffn_bf16 = (__hip_bfloat16 *)malloc(rms_ffn_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_ffn_w, h_rms_ffn_bf16, rms_ffn_size);
    HIP_CHECK(hipMemcpy(gpu_weights->rms_ffn_w, h_rms_ffn_bf16, rms_ffn_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_ffn_bf16);

    size_t rms_out_size = p->hidden_dim;
    __hip_bfloat16 *h_rms_out_bf16 = (__hip_bfloat16 *)malloc(rms_out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_out_w, h_rms_out_bf16, rms_out_size);
    HIP_CHECK(hipMemcpy(gpu_weights->rms_out_w, h_rms_out_bf16, rms_out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_out_bf16);

    // Convert and copy attention weights
    size_t qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    __hip_bfloat16 *h_w_qkv_bf16 = (__hip_bfloat16 *)malloc(qkv_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_qkv, h_w_qkv_bf16, qkv_size);
    HIP_CHECK(hipMemcpy(gpu_weights->w_qkv, h_w_qkv_bf16, qkv_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_qkv_bf16);

    size_t b_qkv_size = p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    __hip_bfloat16 *h_b_qkv_bf16 = (__hip_bfloat16 *)malloc(b_qkv_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_qkv, h_b_qkv_bf16, b_qkv_size);
    HIP_CHECK(hipMemcpy(gpu_weights->b_qkv, h_b_qkv_bf16, b_qkv_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_qkv_bf16);

    size_t attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    __hip_bfloat16 *h_w_o_bf16 = (__hip_bfloat16 *)malloc(attn_out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_o, h_w_o_bf16, attn_out_size);
    HIP_CHECK(hipMemcpy(gpu_weights->w_o, h_w_o_bf16, attn_out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_o_bf16);

    size_t b_o_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_b_o_bf16 = (__hip_bfloat16 *)malloc(b_o_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_o, h_b_o_bf16, b_o_size);
    HIP_CHECK(hipMemcpy(gpu_weights->b_o, h_b_o_bf16, b_o_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_o_bf16);

    // Convert and copy attention sinks
    size_t attn_sinks_size = p->n_layers * p->n_attn_heads;
    __hip_bfloat16 *h_attn_sinks_bf16 = (__hip_bfloat16 *)malloc(attn_sinks_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->attn_sinks, h_attn_sinks_bf16, attn_sinks_size);
    HIP_CHECK(hipMemcpy(gpu_weights->attn_sinks, h_attn_sinks_bf16, attn_sinks_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_attn_sinks_bf16);

    // Convert and copy MoE weights
    size_t w_router_size = p->n_layers * p->hidden_dim * p->n_experts;
    __hip_bfloat16 *h_w_router_bf16 = (__hip_bfloat16 *)malloc(w_router_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_router, h_w_router_bf16, w_router_size);
    HIP_CHECK(hipMemcpy(gpu_weights->w_router, h_w_router_bf16, w_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_router_bf16);

    size_t b_router_size = p->n_layers * p->n_experts;
    __hip_bfloat16 *h_b_router_bf16 = (__hip_bfloat16 *)malloc(b_router_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_router, h_b_router_bf16, b_router_size);
    HIP_CHECK(hipMemcpy(gpu_weights->b_router, h_b_router_bf16, b_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_router_bf16);

    size_t mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    MXFP4Weights mlp1_mxfp4;
    quantize_to_mxfp4(w->w_mlp1, mlp1_mxfp4, mlp1_size);
    size_t mlp1_packed_size = (mlp1_size + 1) / 2;
    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp1_mxfp4, mlp1_mxfp4.packed_values, mlp1_packed_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp1_scales, mlp1_mxfp4.scales, mlp1_mxfp4.num_blocks * sizeof(float), hipMemcpyHostToDevice));
    free_mxfp4_weights(mlp1_mxfp4);

    size_t b_mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim);
    __hip_bfloat16 *h_b_mlp1_bf16 = (__hip_bfloat16 *)malloc(b_mlp1_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_mlp1, h_b_mlp1_bf16, b_mlp1_size);
    HIP_CHECK(hipMemcpy(gpu_weights->b_mlp1, h_b_mlp1_bf16, b_mlp1_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_mlp1_bf16);

    size_t mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    MXFP4Weights mlp2_mxfp4;
    quantize_to_mxfp4(w->w_mlp2, mlp2_mxfp4, mlp2_size);
    size_t mlp2_packed_size = (mlp2_size + 1) / 2;
    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp2_mxfp4, mlp2_mxfp4.packed_values, mlp2_packed_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp2_scales, mlp2_mxfp4.scales, mlp2_mxfp4.num_blocks * sizeof(float), hipMemcpyHostToDevice));
    free_mxfp4_weights(mlp2_mxfp4);

    size_t b_mlp2_size = p->n_layers * p->n_experts * p->hidden_dim;
    __hip_bfloat16 *h_b_mlp2_bf16 = (__hip_bfloat16 *)malloc(b_mlp2_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_mlp2, h_b_mlp2_bf16, b_mlp2_size);
    HIP_CHECK(hipMemcpy(gpu_weights->b_mlp2, h_b_mlp2_bf16, b_mlp2_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_mlp2_bf16);

    // Convert and copy output weights
    size_t out_size = p->hidden_dim * p->vocab_size;
    __hip_bfloat16 *h_out_bf16 = (__hip_bfloat16 *)malloc(out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->out, h_out_bf16, out_size);
    HIP_CHECK(hipMemcpy(gpu_weights->out, h_out_bf16, out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_out_bf16);

    printf("Weight conversion and copying completed successfully\n");
}

void malloc_cpu_buffers(CPUBuffers *cpu_buf, Config *p)
{
    // CPU allocation for RoPE values (used in warmup only)
    cpu_buf->cos_vals = (float *)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));
    cpu_buf->sin_vals = (float *)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));

    // CPU allocations for batch management
    cpu_buf->prompt_tokens = (int **)malloc(BATCH_SIZE * sizeof(int *));
    cpu_buf->current_tokens = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        cpu_buf->prompt_tokens[b] = (int *)malloc((p->seq_len + 3) * sizeof(int));
    }
    cpu_buf->finished = (bool *)malloc(BATCH_SIZE * sizeof(bool));
    cpu_buf->positions = (int *)malloc(BATCH_SIZE * sizeof(int));
    cpu_buf->prompt_lens = (int *)malloc(BATCH_SIZE * sizeof(int));
    
    // Allocate continuous batching fields
    cpu_buf->slot_active_cpu = (bool *)calloc(BATCH_SIZE, sizeof(bool));
    cpu_buf->seq_lengths_cpu = (int *)calloc(BATCH_SIZE, sizeof(int));
    cpu_buf->request_mapping_cpu = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int i = 0; i < BATCH_SIZE; i++) {
        cpu_buf->request_mapping_cpu[i] = -1; // Initialize to -1 (no request)
    }
    cpu_buf->next_request_idx = 0;
}

void build_gpu_transformer(GPUTransformer *gpu_t, Transformer *cpu_t, bool use_paged_attention)
{
    // Copy config
    gpu_t->config = cpu_t->config;
    
    // Initialize paged attention manager if requested
    if (use_paged_attention) {
        gpu_t->paged_attn_mgr = new PagedAttentionManager();
        gpu_t->paged_attn_mgr->initialize(
            gpu_t->config.head_dim * gpu_t->config.n_kv_heads,
            gpu_t->config.n_layers
        );
    } else {
        gpu_t->paged_attn_mgr = nullptr;
    }
    
    // Allocate GPU memory
    malloc_gpu_weights(&gpu_t->weights, &gpu_t->config);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config, use_paged_attention);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);
    
    // Copy weights to GPU
    copy_weights_to_gpu(cpu_t, &gpu_t->weights);
    
    // If using paged attention, connect the manager's memory to the state
    if (use_paged_attention && gpu_t->paged_attn_mgr) {
        gpu_t->state.paged_key_cache = gpu_t->paged_attn_mgr->get_key_cache();
        gpu_t->state.paged_value_cache = gpu_t->paged_attn_mgr->get_value_cache();
        gpu_t->state.block_table = gpu_t->paged_attn_mgr->get_block_table();
    }
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    // Create GPU transformer
    gpu_transformer = (GPUTransformer *)malloc(sizeof(GPUTransformer));
    assert(gpu_transformer != NULL);
    gpu_transformer->config = transformer->config;

    // Build GPU transformer with paged attention enabled by default
    // Set this to false to use legacy linear attention
    bool use_paged_attention = true;  
    build_gpu_transformer(gpu_transformer, transformer, use_paged_attention);
    
    printf("GPU Transformer initialized with %s attention\n", 
           use_paged_attention ? "PAGED" : "LINEAR");

    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;

    for (int pos = 0; pos < p->seq_len; ++pos)
    {
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                             p->initial_context_length, ntk_beta, ntk_alpha,
                             gpu_transformer->cpu_buffers.cos_vals + (pos * p->head_dim / 2),
                             gpu_transformer->cpu_buffers.sin_vals + (pos * p->head_dim / 2));
    }

    // Copy RoPE values to GPU
    HIP_CHECK(hipMemcpy(gpu_transformer->state.cos_vals, gpu_transformer->cpu_buffers.cos_vals, 
                        (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_transformer->state.sin_vals, gpu_transformer->cpu_buffers.sin_vals, 
                        (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
}

void free_gpu_weights(GPUTransformerWeights *w)
{
    if (w->token_embedding_table) HIP_CHECK(hipFree(w->token_embedding_table));
    if (w->rms_attn_w) HIP_CHECK(hipFree(w->rms_attn_w));
    if (w->rms_ffn_w) HIP_CHECK(hipFree(w->rms_ffn_w));
    if (w->rms_out_w) HIP_CHECK(hipFree(w->rms_out_w));
    if (w->w_qkv) HIP_CHECK(hipFree(w->w_qkv));
    if (w->b_qkv) HIP_CHECK(hipFree(w->b_qkv));
    if (w->w_o) HIP_CHECK(hipFree(w->w_o));
    if (w->b_o) HIP_CHECK(hipFree(w->b_o));
    if (w->attn_sinks) HIP_CHECK(hipFree(w->attn_sinks));
    if (w->w_router) HIP_CHECK(hipFree(w->w_router));
    if (w->b_router) HIP_CHECK(hipFree(w->b_router));
    if (w->w_mlp1_mxfp4) HIP_CHECK(hipFree(w->w_mlp1_mxfp4));
    if (w->w_mlp1_scales) HIP_CHECK(hipFree(w->w_mlp1_scales));
    if (w->b_mlp1) HIP_CHECK(hipFree(w->b_mlp1));
    if (w->w_mlp2_mxfp4) HIP_CHECK(hipFree(w->w_mlp2_mxfp4));
    if (w->w_mlp2_scales) HIP_CHECK(hipFree(w->w_mlp2_scales));
    if (w->b_mlp2) HIP_CHECK(hipFree(w->b_mlp2));
    if (w->out) HIP_CHECK(hipFree(w->out));
}

void free_gpu_run_state(GPURunState *s)
{
    if (s->x) HIP_CHECK(hipFree(s->x));
    if (s->t) HIP_CHECK(hipFree(s->t));
    if (s->tb) HIP_CHECK(hipFree(s->tb));
    if (s->tb2) HIP_CHECK(hipFree(s->tb2));
    if (s->temp_buffer) HIP_CHECK(hipFree(s->temp_buffer));
    if (s->qkv) HIP_CHECK(hipFree(s->qkv));
    if (s->q) HIP_CHECK(hipFree(s->q));
    if (s->k) HIP_CHECK(hipFree(s->k));
    if (s->v) HIP_CHECK(hipFree(s->v));
    if (s->att) HIP_CHECK(hipFree(s->att));
    if (s->mask) HIP_CHECK(hipFree(s->mask));
    
    // Free legacy linear KV cache if allocated
    if (s->key_cache) HIP_CHECK(hipFree(s->key_cache));
    if (s->value_cache) HIP_CHECK(hipFree(s->value_cache));
    
    // Free paged KV cache if allocated
    if (s->paged_key_cache) HIP_CHECK(hipFree(s->paged_key_cache));
    if (s->paged_value_cache) HIP_CHECK(hipFree(s->paged_value_cache));
    if (s->block_table) HIP_CHECK(hipFree(s->block_table));
    if (s->cos_vals) HIP_CHECK(hipFree(s->cos_vals));
    if (s->sin_vals) HIP_CHECK(hipFree(s->sin_vals));
    if (s->router_score) HIP_CHECK(hipFree(s->router_score));
    if (s->topk_v) HIP_CHECK(hipFree(s->topk_v));
    if (s->topk_i) HIP_CHECK(hipFree(s->topk_i));
    if (s->mlp1_out) HIP_CHECK(hipFree(s->mlp1_out));
    if (s->gate) HIP_CHECK(hipFree(s->gate));
    if (s->up) HIP_CHECK(hipFree(s->up));
    if (s->gate_up) HIP_CHECK(hipFree(s->gate_up));
    if (s->e_agg) HIP_CHECK(hipFree(s->e_agg));
    if (s->expert_input_buffer) HIP_CHECK(hipFree(s->expert_input_buffer));
    if (s->expert_output_buffer) HIP_CHECK(hipFree(s->expert_output_buffer));
    if (s->expert_indices) HIP_CHECK(hipFree(s->expert_indices));
    if (s->expert_weights) HIP_CHECK(hipFree(s->expert_weights));
    if (s->batch_count) HIP_CHECK(hipFree(s->batch_count));
    if (s->current_tokens) HIP_CHECK(hipFree(s->current_tokens));
    if (s->positions) HIP_CHECK(hipFree(s->positions));
    if (s->logits) HIP_CHECK(hipFree(s->logits));
    
    // Free continuous batching fields
    if (s->seq_lengths) HIP_CHECK(hipFree(s->seq_lengths));
    if (s->slot_active) HIP_CHECK(hipFree(s->slot_active));
    if (s->request_mapping) HIP_CHECK(hipFree(s->request_mapping));
}

void free_cpu_buffers(CPUBuffers *cpu_buf)
{
    if (cpu_buf->cos_vals) free(cpu_buf->cos_vals);
    if (cpu_buf->sin_vals) free(cpu_buf->sin_vals);
    if (cpu_buf->current_tokens) free(cpu_buf->current_tokens);
    if (cpu_buf->finished) free(cpu_buf->finished);
    if (cpu_buf->positions) free(cpu_buf->positions);
    if (cpu_buf->prompt_lens) free(cpu_buf->prompt_lens);
    
    // Free continuous batching fields
    if (cpu_buf->slot_active_cpu) free(cpu_buf->slot_active_cpu);
    if (cpu_buf->seq_lengths_cpu) free(cpu_buf->seq_lengths_cpu);
    if (cpu_buf->request_mapping_cpu) free(cpu_buf->request_mapping_cpu);
    
    if (cpu_buf->prompt_tokens) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            if (cpu_buf->prompt_tokens[b]) free(cpu_buf->prompt_tokens[b]);
        }
        free(cpu_buf->prompt_tokens);
    }
}

void free_gpu_transformer(GPUTransformer *gpu_t)
{
    free_gpu_weights(&gpu_t->weights);
    free_gpu_run_state(&gpu_t->state);
    free_cpu_buffers(&gpu_t->cpu_buffers);
    
    // Clean up paged attention manager if it exists
    if (gpu_t->paged_attn_mgr) {
        delete gpu_t->paged_attn_mgr;
        gpu_t->paged_attn_mgr = nullptr;
    }
}

// Paged attention utility functions
bool allocate_sequence_pages(GPUTransformer *gpu_t, int seq_id, int estimated_length) {
    if (!gpu_t->paged_attn_mgr) {
        return false;  // Paged attention not enabled
    }
    
    // Calculate required pages based on estimated sequence length
    int required_pages = (estimated_length + PAGE_SIZE - 1) / PAGE_SIZE;
    required_pages = min(required_pages, MAX_PAGES_PER_SEQ);
    
    bool success = gpu_t->paged_attn_mgr->allocate_pages(seq_id, required_pages);
    if (success) {
        printf("Allocated %d pages for sequence %d (estimated length: %d)\n", 
               required_pages, seq_id, estimated_length);
    } else {
        printf("Failed to allocate pages for sequence %d\n", seq_id);
    }
    
    return success;
}

bool extend_sequence_pages(GPUTransformer *gpu_t, int seq_id, int additional_tokens) {
    if (!gpu_t->paged_attn_mgr) {
        return false;  // Paged attention not enabled
    }
    
    // Calculate additional pages needed
    int additional_pages = (additional_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
    
    bool success = gpu_t->paged_attn_mgr->extend_pages(seq_id, additional_pages);
    if (success) {
        printf("Extended sequence %d with %d additional pages\n", seq_id, additional_pages);
    }
    
    return success;
}

void free_sequence_pages(GPUTransformer *gpu_t, int seq_id) {
    if (gpu_t->paged_attn_mgr) {
        gpu_t->paged_attn_mgr->free_pages(seq_id);
        printf("Freed pages for sequence %d\n", seq_id);
    }
}

void update_paged_attention_block_table(GPUTransformer *gpu_t, int* active_seq_ids, int num_active) {
    if (gpu_t->paged_attn_mgr) {
        gpu_t->paged_attn_mgr->update_block_table_gpu(active_seq_ids, num_active);
    }
}

void finish(Transformer *transformer, Tokenizer *tokenizer)
{
    // Free GPU transformer
    free_gpu_transformer(gpu_transformer);
    free(gpu_transformer);
}




// GPU-accelerated neural network functions
void attention_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size)
{
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    
    static Timer rms_norm_timer("RMSNorm_attention", true);
    static Timer matmul_timer("MatMul_attention", true);
    static Timer add_bias_timer("AddBias_attention", true);
    static Timer apply_rope_timer("ApplyRoPE_attention", true);
    static Timer update_kv_cache_timer("UpdateKVCache_attention", true);
    static Timer attention_scores_kernel_timer("AttentionScoresKernel_attention", true);
    static Timer add_sinks_kernel_timer("AddSinksKernel_attention", true);
    static Timer softmax_kernel_timer("SoftmaxKernel_attention", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_attention", true);
    static Timer attention_weighted_sum_kernel_timer("AttentionWeightedSumKernel_attention", true);
    static Timer accumulate_kernel_timer("AccumulateKernel_attention", true);

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
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL*b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *dst = s->q + 1LL*b * q_size;
        HIP_CHECK(hipMemcpy(dst, src, q_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy K: shape [batch_size, n_kv_heads * head_dim]
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL*b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + k_offset;
        float *dst = s->k + 1LL*b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
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

    // Update KV cache - Choose between paged and linear based on runtime flag
    if (s->use_paged_attention) {
        // Use paged KV cache update
        dim3 paged_kv_grid(batch_size, (kv_dim + 31) / 32);
        dim3 paged_kv_block(1, 32);
        {
            TIME_SCOPE(update_kv_cache_timer);
            paged_update_kv_cache_kernel<<<paged_kv_grid, paged_kv_block>>>(
                s->paged_key_cache, s->paged_value_cache, s->k, s->v, s->block_table, s->positions, 
                batch_size, kv_dim, p->n_layers, layer_idx);
            HIP_CHECK(hipGetLastError());
        }
        
        // Compute paged attention scores  
        dim3 paged_att_grid(batch_size, p->n_attn_heads, (p->seq_len + 31) / 32);
        dim3 paged_att_block(1, 1, 32);
        {
            TIME_SCOPE(attention_scores_kernel_timer);
            paged_attention_scores_kernel<<<paged_att_grid, paged_att_block>>>(
                s->att, s->q, s->paged_key_cache, s->block_table, s->positions, 
                batch_size, p->n_attn_heads, head_dim, kv_dim, p->n_layers, layer_idx);
            HIP_CHECK(hipGetLastError());
        }
        
        // Softmax attention weights
        dim3 soft_grid(batch_size * p->n_attn_heads);
        dim3 soft_block(THREADS_PER_BLOCK);
        {
            TIME_SCOPE(softmax_kernel_timer);
            softmax_kernel_variable_len<<<soft_grid, soft_block>>>(
                s->att, s->positions, batch_size, p->n_attn_heads, p->seq_len);
            HIP_CHECK(hipGetLastError());
        }
        
        // Weighted sum of values using paged cache
        dim3 paged_wsum_grid(batch_size, p->n_attn_heads);
        dim3 paged_wsum_block(head_dim);
        {
            TIME_SCOPE(attention_weighted_sum_kernel_timer);
            paged_attention_weighted_sum_kernel<<<paged_wsum_grid, paged_wsum_block>>>(
                s->tb, s->att, s->paged_value_cache, s->block_table, s->positions,
                batch_size, p->n_attn_heads, head_dim, kv_dim, p->n_layers, layer_idx);
            HIP_CHECK(hipGetLastError());
        }
    } else {
        // Use legacy linear KV cache update
        dim3 kv_grid(batch_size, (kv_dim + 31) / 32);
        dim3 kv_block(1, 32);
        {
            TIME_SCOPE(update_kv_cache_timer);
            update_kv_cache_kernel<<<kv_grid, kv_block>>>(
                s->key_cache, s->value_cache, s->k, s->v, s->positions, batch_size,
                p->n_layers, layer_idx, p->seq_len, kv_dim);
            HIP_CHECK(hipGetLastError());
        }

        // Compute attention scores using linear cache
        dim3 att_grid(batch_size, p->n_attn_heads, (p->seq_len + 31) / 32);
        dim3 att_block(1, 1, 32);
        {
            TIME_SCOPE(attention_scores_kernel_timer);
            attention_scores_kernel<<<att_grid, att_block>>>(
                s->att, s->q, s->key_cache, s->mask, s->positions, batch_size, p->n_attn_heads,
                head_dim, p->seq_len, p->n_layers, layer_idx, p->sliding_window > 0);
            HIP_CHECK(hipGetLastError());
        }

        dim3 sink_grid(batch_size, p->n_attn_heads);
        dim3 sink_block(1);
        {
            TIME_SCOPE(add_sinks_kernel_timer);
            add_sinks_kernel<<<sink_grid, sink_block>>>(
                s->att, w->attn_sinks + layer_idx * p->n_attn_heads, s->positions,
                p->seq_len, p->n_attn_heads);
            HIP_CHECK(hipGetLastError());
        }

        // Softmax attention weights
        dim3 soft_grid(batch_size * p->n_attn_heads);
        dim3 soft_block(THREADS_PER_BLOCK);
        {
            TIME_SCOPE(softmax_kernel_timer);
            softmax_kernel_variable_len<<<soft_grid, soft_block>>>(
                s->att, s->positions, batch_size, p->n_attn_heads, p->seq_len);
            HIP_CHECK(hipGetLastError());
        }

        // Weighted sum of values using linear cache
        dim3 wsum_grid(batch_size, p->n_attn_heads);
        dim3 wsum_block(head_dim);
        {
            TIME_SCOPE(attention_weighted_sum_kernel_timer);
            attention_weighted_sum_kernel<<<wsum_grid, wsum_block>>>(
                s->tb, s->att, s->value_cache, s->positions, batch_size, p->n_attn_heads,
                head_dim, p->seq_len, p->n_layers, layer_idx);
            HIP_CHECK(hipGetLastError());
        }
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
    static Timer rms_norm_timer("RMSNorm_moe", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_moe", true);
    static Timer add_bias_timer("AddBias_moe", true);
    static Timer topk_kernel_timer("TopKKernel_moe", true);
    static Timer softmax_kernel_timer("SoftmaxKernel_moe", true);
    static Timer gather_expert_inputs_kernel_timer("GatherExpertInputsKernel_moe", true);
    static Timer expert_agg_kernel_timer("ExpertAggKernel_moe", true);
    static Timer scatter_expert_outputs_kernel_timer("ScatterExpertOutputsKernel_moe", true);
    static Timer accumulate_kernel_timer("AccumulateKernel_moe", true);
    static Timer split_gate_up_kernel_timer("SplitGateUpKernel_moe", true);
    static Timer swiglu_kernel_timer("SwigluKernel_moe", true);

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
    int *d_expert_counts;
    int *d_expert_offsets;
    int *d_expert_write_idx;
    HIP_CHECK(hipMalloc(&d_expert_counts, n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_write_idx, n_experts * sizeof(int)));

    // **FIX:** Declare host-side arrays and total_tokens here, before the timed scope
    int h_expert_counts[n_experts];
    int h_expert_offsets[n_experts];
    int total_tokens = 0;

    {
        TIME_SCOPE(gather_expert_inputs_kernel_timer); // Timing the entire gather operation

        // === STAGE 1: COUNT TOKENS PER EXPERT ===
        HIP_CHECK(hipMemset(d_expert_counts, 0, n_experts * sizeof(int)));
        dim3 count_grid((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<count_grid, THREADS_PER_BLOCK>>>(
            s->topk_i, d_expert_counts, batch_size, experts_per_token);
        HIP_CHECK(hipGetLastError());

        // === CALCULATE OFFSETS (PREFIX SUM ON CPU) ===
        HIP_CHECK(hipMemcpy(h_expert_counts, d_expert_counts, n_experts * sizeof(int), hipMemcpyDeviceToHost));

        for (int i = 0; i < n_experts; ++i)
        {
            h_expert_offsets[i] = total_tokens;
            total_tokens += h_expert_counts[i];
        }
        HIP_CHECK(hipMemcpy(d_expert_offsets, h_expert_offsets, n_experts * sizeof(int), hipMemcpyHostToDevice));

        // === STAGE 2: PERMUTE INPUTS WITH COALESCED COPY ===
        HIP_CHECK(hipMemset(d_expert_write_idx, 0, n_experts * sizeof(int)));
        dim3 permute_grid(batch_size); // One block per token
        dim3 permute_block(256);       // Block size for efficient copying
        permute_expert_inputs_kernel<<<permute_grid, permute_block>>>(
            s->t, s->topk_i, s->topk_v, d_expert_offsets, d_expert_write_idx,
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

    // Free the temporary buffers used in the gather operation
    HIP_CHECK(hipFree(d_expert_counts));
    HIP_CHECK(hipFree(d_expert_offsets));
    HIP_CHECK(hipFree(d_expert_write_idx));
}

float *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
{
    static Timer copy_embed_timer("copy_embeddings_forward", true);
    static Timer rms_norm_timer("RMSNorm_forward", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_forward", true);
    
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
    static float *h_logits = nullptr;
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
            if (pos < cpu_buf->prompt_lens[slot] - 1) {
                // Still processing prompt
                next_token = cpu_buf->prompt_tokens[slot][pos + 1];
            } else {
                // Generate new token
                next_token = sample(sampler, logits_slot);
                
                // Save generated token
                int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                int gen_pos = pos - (cpu_buf->prompt_lens[slot] - 1);
                if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                    output_tokens[gen_pos] = next_token;
                    total_tokens_generated++;
                }
            }
            
            // Check for completion
            bool completed = (next_token == 199999 || next_token == 200002 || 
                            pos >= max_steps - 1 || pos >= p->seq_len - 2);
            
            if (completed) {
                if (next_token == 199999 || next_token == 200002){
                    fprintf(stderr, "Request %d received EOS token %d at position %d\n", req_idx, next_token, pos);
                }
                // fprintf(stderr, "Request %d completed at position %d with token %d\n", req_idx, pos, next_token);
                // Mark end of generation
                int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                int gen_pos = pos - (cpu_buf->prompt_lens[slot] - 1) + 1;
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
                cpu_buf->positions[slot]++;
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
        int *temp_tokens = (int *)malloc((p->seq_len + 3) * sizeof(int));
        int temp_len;
        encode(tokenizer, input_seq, -1, -1, temp_tokens, &temp_len, p->initial_context_length);
        int last_prompt_token = temp_tokens[temp_len - 1];
        free(temp_tokens);
        
        // Decode and print generated tokens
        int prev_token = last_prompt_token;
        for (int i = 0;; ++i) {
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
                if (pos < cpu_buf->prompt_lens[b] - 1)
                {
                    // Still processing prompt
                    next_token = cpu_buf->prompt_tokens[b][pos + 1];
                }
                else
                {
                    // Generate new token
                    next_token = sample(sampler, logits_b);

                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (cpu_buf->prompt_lens[b] - 1);
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
                    int gen_pos = pos - (cpu_buf->prompt_lens[b] - 1) + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }

                // Update for next iteration
                cpu_buf->positions[b]++;
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
            for (int i = 0;; ++i)
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

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    // Use continuous batching for better throughput
    return continuous_batching_inference(gpu_transformer, tokenizer, sampler, requests);
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