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
#include <cfloat> // For FLT_MAX
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
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>


#ifndef GETP_RUN
#define GETP_RUN

int num_gpus = 1;

// GPU Transformer Weights struct - stores all model weights on GPU in bfloat16 format
typedef struct
{
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
    __hip_bfloat16 *w_router; // (n_layers, hidden_dim, n_experts)
    __hip_bfloat16 *b_router; // (n_layers, n_experts)

    // MoE weights now use MXFP4 quantization
    // MoE weights (pure BF16 now)
    __hip_bfloat16 *w_mlp1;  // (n_layers, n_experts, 2*D, H) row-major [O=2D, I=H]
    __hip_bfloat16 *w_mlp2;  // (n_layers, n_experts, H,   D) row-major [O=H,  I=D]
    __hip_bfloat16 *b_mlp1, *b_mlp2;

    __hip_bfloat16 *out_w;
    // Output weights
    __hip_bfloat16 *out; // (vocab_size, hidden_dim)
} GPUTransformerWeights;

// GPU Run State struct - stores all activation buffers on GPU
typedef struct
{
    // Basic activation buffers
    float *x;           // activation at current time stamp (batch_size, hidden_dim)
    float *t;           // residual branch buffer (batch_size, hidden_dim)
    float *tb;          // temp buffer (batch_size, head_dim * n_attn_heads)
    float *tb2;         // temp buffer (batch_size, hidden_dim)
    float *temp_buffer; // general purpose temp buffer (batch_size, hidden_dim)

    // Attention buffers
    float *qkv;  // QKV buffer (batch_size, head_dim * (n_attn_heads + 2 * n_kv_heads))
    float *q;    // query buffer (batch_size, n_attn_heads * head_dim)
    float *k;    // key buffer (batch_size, n_kv_heads * head_dim)
    float *v;    // value buffer (batch_size, n_kv_heads * head_dim)
    float *att;  // attention scores (batch_size, n_attn_heads, seq_len)
    float *mask; // attention mask (seq_len, seq_len)

    // KV cache - now using BF16 for 50% memory reduction
    float *key_cache;   // (batch_size, n_layers, seq_len, kv_dim)
    float *value_cache; // (batch_size, n_layers, seq_len, kv_dim)

    // RoPE buffers
    float *cos_vals; // (head_dim/2, seq_len)
    float *sin_vals; // (head_dim/2, seq_len)

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

    // Persistent expert routing buffers (avoid hipMalloc/hipFree in forward pass)
    int *d_expert_counts;    // token counts per expert (n_experts)
    int *d_expert_offsets;   // prefix sum of expert counts (n_experts)
    int *d_expert_write_idx; // write indices for expert gathering (n_experts)
    int *d_total_tokens;     // total tokens across all experts (1 element)

    // Token and position buffers
    int *current_tokens; // current tokens (batch_size)
    int *positions;      // current positions (batch_size)

    // Output buffer
    float *logits; // output logits (batch_size, vocab_size)

    // Continuous batching fields
    int *seq_lengths;     // Current sequence length for each slot [BATCH_SIZE]
    bool *slot_active;    // Whether slot is processing a request [BATCH_SIZE]
    int *request_mapping; // Maps batch slot -> request index in Requests [BATCH_SIZE]
} GPURunState;

// CPU buffers for warmup and host-side operations
typedef struct
{
    float *cos_vals;     // RoPE cosine values (CPU)
    float *sin_vals;     // RoPE sine values (CPU)
    int **prompt_tokens; // prompt tokens for each sequence
    int *current_tokens; // current tokens (CPU copy)
    bool *finished;      // finished flags for each sequence
    int *positions;      // current positions (CPU copy)
    int *prompt_lens;    // prompt lengths
    hipStream_t sGather;
    hipStream_t sScatter;
    hipStream_t sMLP[N_MLP_STREAMS];
    int *expert_counts;
    int *expert_offsets;
    float *logits;

    // Continuous batching fields
    int next_request_idx;     // Index of next unprocessed request (0 to num_reqs-1)
    bool *slot_active_cpu;    // CPU mirror of slot_active for quick access
    int *seq_lengths_cpu;     // CPU mirror of seq_lengths
    int *request_mapping_cpu; // CPU mirror of request_mapping
} CPUBuffers;

// Main GPU Transformer struct
typedef struct
{
    Config config;                 // model configuration
    GPUTransformerWeights weights; // GPU weights
    GPURunState state;             // GPU run state buffers
    CPUBuffers cpu_buffers;        // CPU buffers for host operations
} GPUTransformer;

// Global variables for direct access in batched_generate_gpu
GPUTransformer *gpu_transformers[MAX_GPUS];

// Memory allocation functions
void malloc_gpu_run_state(GPURunState *s, Config *p)
{
    int kv_dim = p->head_dim * p->n_kv_heads;
    int expert_per_token = p->experts_per_token;

    // Initialize pointers to NULL
    s->mask = NULL;
    s->d_expert_counts = NULL;
    s->d_expert_offsets = NULL;
    s->d_expert_write_idx = NULL;
    s->d_total_tokens = NULL;

    // Check memory requirements and print for debugging
    size_t total_memory = 0;
    size_t batch_hidden = BATCH_SIZE * p->hidden_dim * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);
    // BF16 KV cache - 50% memory reduction compared to FP32
    size_t kv_cache_size = BATCH_SIZE * p->n_layers * MAX_SEQ_LEN * kv_dim * sizeof(float);

    printf("Allocating GPU memory: batch_size=%d, hidden_dim=%d, seq_len=%d\n",
           BATCH_SIZE, p->hidden_dim, MAX_SEQ_LEN);
    printf("KV cache size per batch (BF16): %zu MB (50%% reduction from FP32)\n", kv_cache_size / (1024 * 1024));

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

    // Persistent expert routing buffers to avoid hipMalloc/hipFree in forward pass
    HIP_CHECK(hipMalloc((void **)&s->d_expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_expert_offsets, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_expert_write_idx, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_total_tokens, sizeof(int)));

    // KV cache allocation - this is usually the largest allocation
    HIP_CHECK(hipMalloc((void **)&s->key_cache, kv_cache_size));
    HIP_CHECK(hipMalloc((void **)&s->value_cache, kv_cache_size));

    HIP_CHECK(hipMalloc((void **)&s->att, BATCH_SIZE * p->n_attn_heads * MAX_SEQ_LEN * sizeof(float)));
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
    HIP_CHECK(hipMalloc((void **)&s->cos_vals, (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->sin_vals, (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
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
    HIP_CHECK(hipMemset(s->key_cache, 0, kv_cache_size));
    HIP_CHECK(hipMemset(s->value_cache, 0, kv_cache_size));
    HIP_CHECK(hipMemset(s->att, 0, BATCH_SIZE * p->n_attn_heads * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMemset(s->logits, 0, BATCH_SIZE * p->vocab_size * sizeof(float)));

    // Initialize continuous batching fields
    HIP_CHECK(hipMemset(s->seq_lengths, 0, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMemset(s->slot_active, 0, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMemset(s->request_mapping, -1, BATCH_SIZE * sizeof(int)));

    if (p->sliding_window > 0)
    {
        size_t mask_size = MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(float);
        HIP_CHECK(hipMalloc((void **)&s->mask, mask_size));

        // Initialize mask on GPU if needed
        float *h_mask = (float *)malloc(mask_size);
        if (!h_mask)
        {
            fprintf(stderr, "Failed to allocate host memory for mask\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < MAX_SEQ_LEN; i++)
        {
            for (int j = 0; j < MAX_SEQ_LEN; j++)
            {
                h_mask[i * MAX_SEQ_LEN + j] = (i - j >= p->sliding_window) ? -INFINITY : 0.0f;
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

    // --- remove the whole MXFP4 allocation block ---

    // Replace with BF16 allocations:
    size_t mlp1_size = (size_t)p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1, mlp1_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1,
                        (size_t)p->n_layers * p->n_experts * (2 * p->intermediate_dim) * sizeof(__hip_bfloat16)));

    size_t mlp2_size = (size_t)p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2, mlp2_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2,
                        (size_t)p->n_layers * p->n_experts * p->hidden_dim * sizeof(__hip_bfloat16)));

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
    // --- remove the whole MXFP4 quantize/copy section ---

    // BF16 MLP1
    {
        size_t mlp1_size = (size_t)p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
        __hip_bfloat16 *h_mlp1_bf16 = (__hip_bfloat16 *)malloc(mlp1_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_mlp1, h_mlp1_bf16, mlp1_size);
        HIP_CHECK(hipMemcpy(gpu_weights->w_mlp1, h_mlp1_bf16, mlp1_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_mlp1_bf16);
    }
    {
        size_t b_mlp1_size = (size_t)p->n_layers * p->n_experts * (2 * p->intermediate_dim);
        __hip_bfloat16 *h_b_mlp1_bf16 = (__hip_bfloat16 *)malloc(b_mlp1_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_mlp1, h_b_mlp1_bf16, b_mlp1_size);
        HIP_CHECK(hipMemcpy(gpu_weights->b_mlp1, h_b_mlp1_bf16, b_mlp1_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_b_mlp1_bf16);
    }

    // BF16 MLP2
    {
        size_t mlp2_size = (size_t)p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
        __hip_bfloat16 *h_mlp2_bf16 = (__hip_bfloat16 *)malloc(mlp2_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_mlp2, h_mlp2_bf16, mlp2_size);
        HIP_CHECK(hipMemcpy(gpu_weights->w_mlp2, h_mlp2_bf16, mlp2_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_mlp2_bf16);
    }
    {
        size_t b_mlp2_size = (size_t)p->n_layers * p->n_experts * p->hidden_dim;
        __hip_bfloat16 *h_b_mlp2_bf16 = (__hip_bfloat16 *)malloc(b_mlp2_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_mlp2, h_b_mlp2_bf16, b_mlp2_size);
        HIP_CHECK(hipMemcpy(gpu_weights->b_mlp2, h_b_mlp2_bf16, b_mlp2_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_b_mlp2_bf16);
    }

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
    cpu_buf->cos_vals = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));
    cpu_buf->sin_vals = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));

    // CPU allocations for batch management
    cpu_buf->prompt_tokens = (int **)malloc(BATCH_SIZE * sizeof(int *));
    cpu_buf->current_tokens = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        cpu_buf->prompt_tokens[b] = (int *)malloc((MAX_SEQ_LEN + 3) * sizeof(int));
    }
    cpu_buf->finished = (bool *)malloc(BATCH_SIZE * sizeof(bool));
    cpu_buf->positions = (int *)malloc(BATCH_SIZE * sizeof(int));
    cpu_buf->prompt_lens = (int *)malloc(BATCH_SIZE * sizeof(int));

    // Allocate continuous batching fields
    cpu_buf->slot_active_cpu = (bool *)calloc(BATCH_SIZE, sizeof(bool));
    cpu_buf->seq_lengths_cpu = (int *)calloc(BATCH_SIZE, sizeof(int));
    cpu_buf->request_mapping_cpu = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        cpu_buf->request_mapping_cpu[i] = -1; // Initialize to -1 (no request)
    }
    cpu_buf->next_request_idx = 0;

    cpu_buf->expert_counts = (int*) malloc(p->n_experts * sizeof(int));
    cpu_buf->expert_offsets = (int*) malloc(p->n_experts * sizeof(int));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sGather, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sScatter, hipStreamNonBlocking));
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sMLP[i], hipStreamNonBlocking));
    cpu_buf->logits = (float*) malloc(BATCH_SIZE * p->vocab_size * sizeof(float));
}

void build_gpu_transformer(GPUTransformer *gpu_t, Transformer *cpu_t)
{
    // Copy config
    gpu_t->config = cpu_t->config;

    // Allocate GPU memory
    malloc_gpu_weights(&gpu_t->weights, &gpu_t->config);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);

    // Copy weights to GPU
    copy_weights_to_gpu(cpu_t, &gpu_t->weights);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    // Create GPU transformer
    // Multi-GPU support: allocate and initialize GPUTransformer for each GPU
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    #pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev) {
        HIP_CHECK(hipSetDevice(dev));
        gpu_transformers[dev] = (GPUTransformer *)malloc(sizeof(GPUTransformer));
        assert(gpu_transformers[dev] != NULL);

        GPUTransformer *gpu_transformer = gpu_transformers[dev];

        gpu_transformer->config = transformer->config;
        build_gpu_transformer(gpu_transformer, transformer);

        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;

        for (int pos = 0; pos < MAX_SEQ_LEN; ++pos)
        {
            compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                                p->initial_context_length, ntk_beta, ntk_alpha,
                                gpu_transformer->cpu_buffers.cos_vals + (pos * p->head_dim / 2),
                                gpu_transformer->cpu_buffers.sin_vals + (pos * p->head_dim / 2));
        }

        // Copy RoPE values to GPU
        HIP_CHECK(hipMemcpy(gpu_transformer->state.cos_vals, gpu_transformer->cpu_buffers.cos_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_transformer->state.sin_vals, gpu_transformer->cpu_buffers.sin_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
    }    
}

void free_gpu_weights(GPUTransformerWeights *w)
{
    if (w->token_embedding_table)
        HIP_CHECK(hipFree(w->token_embedding_table));
    if (w->rms_attn_w)
        HIP_CHECK(hipFree(w->rms_attn_w));
    if (w->rms_ffn_w)
        HIP_CHECK(hipFree(w->rms_ffn_w));
    if (w->rms_out_w)
        HIP_CHECK(hipFree(w->rms_out_w));
    if (w->w_qkv)
        HIP_CHECK(hipFree(w->w_qkv));
    if (w->b_qkv)
        HIP_CHECK(hipFree(w->b_qkv));
    if (w->w_o)
        HIP_CHECK(hipFree(w->w_o));
    if (w->b_o)
        HIP_CHECK(hipFree(w->b_o));
    if (w->attn_sinks)
        HIP_CHECK(hipFree(w->attn_sinks));
    if (w->w_router)
        HIP_CHECK(hipFree(w->w_router));
    if (w->b_router)
        HIP_CHECK(hipFree(w->b_router));
    // remove: w_mlp1_mxfp4, w_mlp1_scales, w_mlp2_mxfp4, w_mlp2_scales

    if (w->w_mlp1) HIP_CHECK(hipFree(w->w_mlp1));
    if (w->w_mlp2) HIP_CHECK(hipFree(w->w_mlp2));
    if (w->b_mlp1) HIP_CHECK(hipFree(w->b_mlp1));
    if (w->b_mlp2) HIP_CHECK(hipFree(w->b_mlp2));

    if (w->out)
        HIP_CHECK(hipFree(w->out));
}

void free_gpu_run_state(GPURunState *s)
{
    if (s->x)
        HIP_CHECK(hipFree(s->x));
    if (s->t)
        HIP_CHECK(hipFree(s->t));
    if (s->tb)
        HIP_CHECK(hipFree(s->tb));
    if (s->tb2)
        HIP_CHECK(hipFree(s->tb2));
    if (s->temp_buffer)
        HIP_CHECK(hipFree(s->temp_buffer));
    if (s->qkv)
        HIP_CHECK(hipFree(s->qkv));
    if (s->q)
        HIP_CHECK(hipFree(s->q));
    if (s->k)
        HIP_CHECK(hipFree(s->k));
    if (s->v)
        HIP_CHECK(hipFree(s->v));
    if (s->att)
        HIP_CHECK(hipFree(s->att));
    if (s->mask)
        HIP_CHECK(hipFree(s->mask));
    if (s->key_cache)
        HIP_CHECK(hipFree(s->key_cache));
    if (s->value_cache)
        HIP_CHECK(hipFree(s->value_cache));
    if (s->cos_vals)
        HIP_CHECK(hipFree(s->cos_vals));
    if (s->sin_vals)
        HIP_CHECK(hipFree(s->sin_vals));
    if (s->router_score)
        HIP_CHECK(hipFree(s->router_score));
    if (s->topk_v)
        HIP_CHECK(hipFree(s->topk_v));
    if (s->topk_i)
        HIP_CHECK(hipFree(s->topk_i));
    if (s->mlp1_out)
        HIP_CHECK(hipFree(s->mlp1_out));
    if (s->gate)
        HIP_CHECK(hipFree(s->gate));
    if (s->up)
        HIP_CHECK(hipFree(s->up));
    if (s->gate_up)
        HIP_CHECK(hipFree(s->gate_up));
    if (s->e_agg)
        HIP_CHECK(hipFree(s->e_agg));
    if (s->expert_input_buffer)
        HIP_CHECK(hipFree(s->expert_input_buffer));
    if (s->expert_output_buffer)
        HIP_CHECK(hipFree(s->expert_output_buffer));
    if (s->expert_indices)
        HIP_CHECK(hipFree(s->expert_indices));
    if (s->expert_weights)
        HIP_CHECK(hipFree(s->expert_weights));
    if (s->batch_count)
        HIP_CHECK(hipFree(s->batch_count));
    if (s->d_expert_counts)
        HIP_CHECK(hipFree(s->d_expert_counts));
    if (s->d_expert_offsets)
        HIP_CHECK(hipFree(s->d_expert_offsets));
    if (s->d_expert_write_idx)
        HIP_CHECK(hipFree(s->d_expert_write_idx));
    if (s->d_total_tokens)
        HIP_CHECK(hipFree(s->d_total_tokens));
    if (s->current_tokens)
        HIP_CHECK(hipFree(s->current_tokens));
    if (s->positions)
        HIP_CHECK(hipFree(s->positions));
    if (s->logits)
        HIP_CHECK(hipFree(s->logits));

    // Free continuous batching fields
    if (s->seq_lengths)
        HIP_CHECK(hipFree(s->seq_lengths));
    if (s->slot_active)
        HIP_CHECK(hipFree(s->slot_active));
    if (s->request_mapping)
        HIP_CHECK(hipFree(s->request_mapping));
}

void free_cpu_buffers(CPUBuffers *cpu_buf)
{
    if (cpu_buf->cos_vals)
        free(cpu_buf->cos_vals);
    if (cpu_buf->sin_vals)
        free(cpu_buf->sin_vals);
    if (cpu_buf->current_tokens)
        free(cpu_buf->current_tokens);
    if (cpu_buf->finished)
        free(cpu_buf->finished);
    if (cpu_buf->positions)
        free(cpu_buf->positions);
    if (cpu_buf->prompt_lens)
        free(cpu_buf->prompt_lens);

    // Free continuous batching fields
    if (cpu_buf->slot_active_cpu)
        free(cpu_buf->slot_active_cpu);
    if (cpu_buf->seq_lengths_cpu)
        free(cpu_buf->seq_lengths_cpu);
    if (cpu_buf->request_mapping_cpu)
        free(cpu_buf->request_mapping_cpu);

    if (cpu_buf->prompt_tokens)
    {
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            if (cpu_buf->prompt_tokens[b])
                free(cpu_buf->prompt_tokens[b]);
        }
        free(cpu_buf->prompt_tokens);
    }
}

void free_gpu_transformer(GPUTransformer *gpu_t)
{
    free_gpu_weights(&gpu_t->weights);
    free_gpu_run_state(&gpu_t->state);
    free_cpu_buffers(&gpu_t->cpu_buffers);
}

void finish(Transformer *transformer, Tokenizer *tokenizer)
{
    // Free GPU transformer
    #pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev) {
        free_gpu_transformer(gpu_transformers[dev]);
        free(gpu_transformers[dev]);
    }
}

int tokenId = 0;

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

    // if (tokenId == 0) debug(s->t, 10);

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
    dim3 bias_grid((1LL * batch_size * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    {
        TIME_SCOPE(add_bias_timer);
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            s->qkv, w->b_qkv + qkv_bias_offset, batch_size, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }
    
    // if (tokenId == 0) debug(s->qkv, 10);
    // Copy Q, K, V from qkv buffer - SIMPLIFIED AND FIXED
    int q_size = p->n_attn_heads * head_dim;
    int k_size = p->n_kv_heads * head_dim;
    int v_size = p->n_kv_heads * head_dim;

    // Copy Q: shape [batch_size, n_attn_heads * head_dim]
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL * b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *dst = s->q + 1LL * b * q_size;
        HIP_CHECK(hipMemcpy(dst, src, q_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy K: shape [batch_size, n_kv_heads * head_dim]
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL * b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + k_offset;
        float *dst = s->k + 1LL * b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL * b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + v_offset;
        float *dst = s->v + 1LL * b * v_size;
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
    // if (tokenId == 0) debug(s->k, 10);
    rope_grid.y = p->n_kv_heads;
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        s->k, s->cos_vals, s->sin_vals, s->positions, batch_size, p->n_kv_heads, head_dim);
    HIP_CHECK(hipGetLastError());
    // if (tokenId == 0) debug(s->q, 10);
    // if (tokenId == 0) debug(s->k, 10);
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
        attention_scores_shared_mem_kernel<<<att_grid, att_block>>>(
            s->att, s->q, s->key_cache, s->mask, s->positions, batch_size, p->n_attn_heads,
            head_dim, MAX_SEQ_LEN, p->n_layers, layer_idx, p->sliding_window > 0);
        HIP_CHECK(hipGetLastError());
    }
    // Compute attention scores
    //

    // Add attention sinks - FIXED: Use GPU weight pointer
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
    
    // if (tokenId == 0)
    // debug(s->att, 20);
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
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int Ktok = p->experts_per_token;

    hipEvent_t evRouterDone, evPermuteDone;
    HIP_CHECK(hipEventCreateWithFlags(&evRouterDone, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&evPermuteDone, hipEventDisableTiming));

    // 1) FFN RMSNorm + router (sGather stream)
    {
        dim3 norm_grid(batch_size), norm_block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<norm_grid, norm_block, 0, cpu_buf->sGather>>>(
            s->t, s->x, w->rms_ffn_w + (size_t)layer_idx * H, batch_size, H);
        HIP_CHECK(hipGetLastError());

        matmul(s->router_score, s->t,
               w->w_router + (size_t)layer_idx * H * E,
               batch_size, H, E, cpu_buf->sGather);

        const int elems = batch_size * E;
        add_bias_kernel<<<(elems + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,
                          THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(
            s->router_score, w->b_router + (size_t)layer_idx * E, batch_size, E);
        HIP_CHECK(hipGetLastError());

        topk_kernel<<<batch_size, 1, 0, cpu_buf->sGather>>>(
            s->topk_v, s->topk_i, s->router_score, batch_size, E, Ktok);

        softmax_kernel<<<batch_size, THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(
            s->topk_v, batch_size, Ktok);

        HIP_CHECK(hipEventRecord(evRouterDone, cpu_buf->sGather));
    }

    // 2) Count -> host prefix -> offsets D2H/D2D -> permute (sGather)
    int total_tokens = 0;
    {
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sGather, evRouterDone, 0));

        HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, E * sizeof(int), cpu_buf->sGather));
        const dim3 count_grid((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<count_grid, THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(
            s->topk_i, s->d_expert_counts, batch_size, Ktok);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipMemcpyAsync(cpu_buf->expert_counts, s->d_expert_counts,
                                 E * sizeof(int), hipMemcpyDeviceToHost, cpu_buf->sGather));
        HIP_CHECK(hipStreamSynchronize(cpu_buf->sGather));

        total_tokens = 0;
        for (int e = 0; e < E; ++e) {
            cpu_buf->expert_offsets[e] = total_tokens;
            total_tokens += cpu_buf->expert_counts[e];
        }

        HIP_CHECK(hipMemcpyAsync(s->d_expert_offsets, cpu_buf->expert_offsets,
                                 E * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));

        HIP_CHECK(hipMemsetAsync(s->d_expert_write_idx, 0, E * sizeof(int), cpu_buf->sGather));

        const dim3 permute_grid(batch_size), permute_block(256);
        const int  shared_mem_size = Ktok * sizeof(int);
        permute_expert_inputs_kernel<<<permute_grid, permute_block, shared_mem_size, cpu_buf->sGather>>>(
            s->t, s->topk_i, s->topk_v, s->d_expert_offsets, s->d_expert_write_idx,
            batch_size, H, Ktok,
            s->expert_input_buffer, s->expert_indices, s->expert_weights);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipEventRecord(evPermuteDone, cpu_buf->sGather));
    }

    // Early out if no tokens routed
    if (total_tokens == 0) {
        HIP_CHECK(hipEventDestroy(evRouterDone));
        HIP_CHECK(hipEventDestroy(evPermuteDone));
        return;
    }

    // 3) Build m-tile prefix (host) -> D2D (tiny)
    std::vector<int> h_mtiles(E+1, 0);
    for (int e = 0; e < E; ++e) {
        int mtiles = (cpu_buf->expert_counts[e] + BLOCK_M - 1) / BLOCK_M;
        h_mtiles[e+1] = h_mtiles[e] + mtiles;
    }
    const int total_mtiles = h_mtiles[E];

    int *d_mtile_prefix = nullptr;
    HIP_CHECK(hipMalloc((void**)&d_mtile_prefix, (E+1) * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_mtile_prefix, h_mtiles.data(), (E+1)*sizeof(int), hipMemcpyHostToDevice));

    // Wait for permute; from here we run on sScatter as the single pipeline stream
    HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, evPermuteDone, 0));

    // Zero aggregation buffer (once)
    HIP_CHECK(hipMemsetAsync(s->e_agg, 0, (size_t)batch_size * H * sizeof(float), cpu_buf->sScatter));

    // ------- GROUPED MLP1 (all experts) -------
    {
        // per-expert dims for bookkeeping (unchanged)
        const size_t seg1_elems = (size_t)(2 * D) * H;

        // layer base (BF16)
        const size_t layer1_elem_off = (size_t)layer_idx * E * seg1_elems;
        const __hip_bfloat16* W1_layer = w->w_mlp1 + layer1_elem_off;

        dim3 grid((2*D + BLOCK_N - 1) / BLOCK_N, total_mtiles);
        dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);
        size_t shmem = (size_t)(2*BLOCK_M*BLOCK_K + 2*BLOCK_K*BLOCK_N) * sizeof(uint16_t);

        hipLaunchKernelGGL(grouped_mlp1_bf16_kernel,
            grid, block, shmem, cpu_buf->sScatter,
            /*C=*/s->mlp1_out,
            /*A=*/s->expert_input_buffer,
            /*W1=*/W1_layer,
            s->d_expert_offsets, s->d_expert_counts, d_mtile_prefix, E,
            /*K=*/H, /*N=*/2*D);
        HIP_CHECK(hipGetLastError());
    }


    // ------- fused bias+SiLU(gate)*up over [total_tokens, D] -------
    {
        const __hip_bfloat16* b1_layer =
            w->b_mlp1 + (size_t)layer_idx * E * (2*D);

        const size_t work = (size_t)total_tokens * D;
        const int T = 256;
        const dim3 grid((work + T - 1)/T), block(T);

        bias_swiglu_epilogue_kernel<<<grid, block, 0, cpu_buf->sScatter>>>(
            s->mlp1_out, b1_layer,
            s->d_expert_offsets, s->d_expert_counts, E,
            s->gate_up, D, total_tokens, p->swiglu_limit, 1.702f);
        HIP_CHECK(hipGetLastError());
    }

    // ------- GROUPED MLP2 (all experts) + bias add -------
    {
        const size_t seg2_elems = (size_t)H * D;
        const size_t layer2_elem_off = (size_t)layer_idx * E * seg2_elems;

        const __hip_bfloat16* W2_layer = w->w_mlp2 + layer2_elem_off;
        const __hip_bfloat16* b2_layer = w->b_mlp2 + (size_t)layer_idx * E * H;

        dim3 grid((H + BLOCK_N - 1) / BLOCK_N, total_mtiles);
        dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);
        size_t shmem = (size_t)(2*BLOCK_M*BLOCK_K + 2*BLOCK_K*BLOCK_N) * sizeof(uint16_t);

        hipLaunchKernelGGL(grouped_mlp2_bf16_bias_kernel,
            grid, block, shmem, cpu_buf->sScatter,
            /*C=*/s->expert_output_buffer,
            /*A=*/s->gate_up,
            /*W2=*/W2_layer, /*b2=*/b2_layer,
            s->d_expert_offsets, s->d_expert_counts, d_mtile_prefix, E,
            /*K=*/D, /*N=*/H);
        HIP_CHECK(hipGetLastError());
    }

    // ------- single scatter & residual -------
    {
        const int elems = total_tokens * H;
        const dim3 grid((elems + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
        scatter_expert_outputs_kernel<<<grid, THREADS_PER_BLOCK, 0, cpu_buf->sScatter>>>(
            s->e_agg, s->expert_output_buffer, s->expert_indices, s->expert_weights,
            total_tokens, H);
        HIP_CHECK(hipGetLastError());
    }
    {
        const int elems = batch_size * H;
        accumulate_kernel<<<(elems + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK, 0, cpu_buf->sScatter>>>(
            s->x, s->e_agg, 1.0f, batch_size, H);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(cpu_buf->sScatter));
    HIP_CHECK(hipFree(d_mtile_prefix));
    HIP_CHECK(hipEventDestroy(evRouterDone));
    HIP_CHECK(hipEventDestroy(evPermuteDone));
}

int *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
{
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
        rmsnorm_kernel<<<final_norm_grid, final_norm_block>>>(
            s->x, s->x, w->rms_out_w, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = (p->vocab_size + block_dim.y - 1) / block_dim.y; // Ceiling division

    {
        // Launch the simple kernel
        matmul(
            s->logits, s->x, w->out, batch_size, hidden_dim, p->vocab_size);
    }
    sample_argmax(s->logits, s->current_tokens, batch_size, p->vocab_size);
    // Copy logits back to CPU (you might want to keep this on GPU for sampling)
    HIP_CHECK(hipMemcpy(cpu_buf->current_tokens, s->current_tokens, batch_size * sizeof(int), hipMemcpyDeviceToHost));
    ++tokenId;
    return cpu_buf->current_tokens; 
}

long long continuous_batching_inference(Tokenizer *tokenizer,
                                        Sampler *sampler, Requests *requests)
{
    long long total_tokens_generated = 0;
    int total_requests = requests->num_reqs;
    #pragma omp parallel num_threads(num_gpus) reduction(+:total_tokens_generated)
    {
        int gpu_id = omp_get_thread_num();
        HIP_CHECK(hipSetDevice(gpu_id));
        
        GPUTransformer *gpu_t = gpu_transformers[gpu_id];

        int requests_per_gpu = requests->num_reqs / num_gpus;
        int extra = requests->num_reqs % num_gpus;

        int local_request = requests_per_gpu + (gpu_id < extra ? 1 : 0);
        int start_request = gpu_id * requests_per_gpu + (gpu_id < extra ? gpu_id : extra);
        int end_request = start_request + local_request;

        Config *p = &gpu_t->config;
        CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
        GPURunState *state = &gpu_t->state;

        // Initialize
        cpu_buf->next_request_idx = start_request;

        // Initialize all slots to inactive
        for (int slot = 0; slot < BATCH_SIZE; slot++)
        {
            cpu_buf->slot_active_cpu[slot] = false;
            cpu_buf->request_mapping_cpu[slot] = -1;
            cpu_buf->seq_lengths_cpu[slot] = 0;
            cpu_buf->positions[slot] = 0;
            cpu_buf->finished[slot] = true;
        }

        // Fill initial batch with first requests
        for (int slot = 0; slot < BATCH_SIZE && cpu_buf->next_request_idx < end_request; slot++)
        {
            int req_idx = cpu_buf->next_request_idx++;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            // Encode prompt
            encode(tokenizer, input_seq, -1, -1, cpu_buf->prompt_tokens[slot],
                &cpu_buf->prompt_lens[slot], p->initial_context_length);

            if (cpu_buf->prompt_lens[slot] < 1)
            {
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

        while (has_active_slots)
        {
            // Forward pass on all slots (inactive ones will be skipped internally)
            int *next_tokens = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, BATCH_SIZE);

            // Process each slot
            has_active_slots = false;
            for (int slot = 0; slot < BATCH_SIZE; slot++)
            {
                if (!cpu_buf->slot_active_cpu[slot])
                    continue;

                has_active_slots = true;
                int req_idx = cpu_buf->request_mapping_cpu[slot];
                int pos = cpu_buf->positions[slot];

                int next_token;
                // Advance position first
                pos++;

                if (pos < cpu_buf->prompt_lens[slot])
                {
                    // Still processing prompt - force next prompt token
                    next_token = cpu_buf->prompt_tokens[slot][pos];
                }
                else
                {
                    // Generate new token
                    next_token = next_tokens[slot];

                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - cpu_buf->prompt_lens[slot];
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }

                // Check for completion
                bool completed = (next_token == 199999 || next_token == 200002 ||
                                pos >= max_steps - 1 || pos >= MAX_SEQ_LEN - 2);

                if (completed)
                {
                    // fprintf(stderr, "Request %d completed at position %d with token %d\n", req_idx, pos, next_token);
                    // Mark end of generation
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - cpu_buf->prompt_lens[slot] + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = -1; // End marker
                    }

                    // Try to assign a new request to this slot
                    if (cpu_buf->next_request_idx < end_request)
                    {
                        // Get next request
                        req_idx = cpu_buf->next_request_idx++;
                        const char *input_seq = get_str_req_ptr(requests, req_idx);

                        // Encode prompt
                        encode(tokenizer, input_seq, -1, -1, cpu_buf->prompt_tokens[slot],
                            &cpu_buf->prompt_lens[slot], p->initial_context_length);

                        if (cpu_buf->prompt_lens[slot] < 1)
                        {
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
                    }
                    else
                    {
                        // No more requests, deactivate slot
                        cpu_buf->slot_active_cpu[slot] = false;
                        cpu_buf->request_mapping_cpu[slot] = -1;
                    }
                }
                else
                {
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
    }

    int max_steps = gpu_transformers[0]->config.seq_len;
    Config *p = gpu_transformers[0] ? &gpu_transformers[0]->config : nullptr;

    // Print results for all requests
    for (int req_idx = 0; req_idx < requests->num_reqs; req_idx++)
    {
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

    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    // Use continuous batching for better throughput
    return continuous_batching_inference(tokenizer, sampler, requests);
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