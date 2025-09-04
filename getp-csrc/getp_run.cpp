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
    float *token_embedding_table; // (vocab_size, hidden_dim)

    // RMSNorm weights
    float *rms_attn_w; // (n_layers, hidden_dim)
    float *rms_ffn_w;  // (n_layers, hidden_dim)
    float *rms_out_w;  // (hidden_dim,)

    // Attention weights
    float *w_qkv;      // (n_layers, head_dim * (n_attn_heads + 2 * n_kv_heads), hidden_dim)
    float *b_qkv;      // (n_layers, head_dim * (n_attn_heads + 2 * n_kv_heads))
    float *w_o;        // (n_layers, hidden_dim, head_dim * n_attn_heads)
    float *b_o;        // (n_layers, hidden_dim)
    __hip_bfloat16 *attn_sinks; // (n_layers, n_attn_heads)

    // MoE router weights
    __hip_bfloat16 *w_router; // (n_layers, hidden_dim, n_experts)
    __hip_bfloat16 *b_router; // (n_layers, n_experts)

    // MoE weights now use MXFP4 quantization
    // MoE weights (pure BF16 now)
    __hip_bfloat16 *w_mlp1;  // (n_layers, n_experts, 2*D, H) row-major [O=2D, I=H]
    __hip_bfloat16 *w_mlp2;  // (n_layers, n_experts, H,   D) row-major [O=H,  I=D]
    __hip_bfloat16 *b_mlp1, *b_mlp2;
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

    int   *local_ids;   // [BATCH_SIZE * K]
    float *local_wts;   // [BATCH_SIZE * K]
    int   *n_local;     // [BATCH_SIZE]
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

    HIP_CHECK(hipMalloc((void **)&s->local_ids, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->local_wts, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->n_local, BATCH_SIZE * sizeof(int)));


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
    HIP_CHECK(hipMalloc((void **)&w->token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->rms_attn_w, p->n_layers * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->rms_ffn_w, p->n_layers * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->rms_out_w, p->hidden_dim * sizeof(float)));

    int qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_qkv, qkv_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_qkv, p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim * sizeof(float)));

    int attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_o, attn_out_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_o, p->n_layers * p->hidden_dim * sizeof(float)));

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
    HIP_CHECK(hipMemcpy(gpu_weights->token_embedding_table, w->token_embedding_table, embedding_size * sizeof(float), hipMemcpyHostToDevice));

    // Convert and copy normalization weights
    size_t rms_attn_size = p->n_layers * p->hidden_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->rms_attn_w, w->rms_attn_w, rms_attn_size * sizeof(float), hipMemcpyHostToDevice));

    size_t rms_ffn_size = p->n_layers * p->hidden_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->rms_ffn_w, w->rms_ffn_w, rms_ffn_size * sizeof(float), hipMemcpyHostToDevice));

    size_t rms_out_size = p->hidden_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->rms_out_w, w->rms_out_w, rms_out_size * sizeof(float), hipMemcpyHostToDevice));

    // Convert and copy attention weights
    size_t qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->w_qkv, w->w_qkv, qkv_size * sizeof(float), hipMemcpyHostToDevice));

    size_t b_qkv_size = p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->b_qkv, w->b_qkv, b_qkv_size * sizeof(float), hipMemcpyHostToDevice));

    size_t attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->w_o, w->w_o, attn_out_size * sizeof(float), hipMemcpyHostToDevice));

    size_t b_o_size = p->n_layers * p->hidden_dim;
    HIP_CHECK(hipMemcpy(gpu_weights->b_o, w->b_o, b_o_size * sizeof(float), hipMemcpyHostToDevice));

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


#ifndef GEMV_TILE_N
#define GEMV_TILE_N 512
#endif
#ifndef GEMV_WARPS_PER_BLOCK
#define GEMV_WARPS_PER_BLOCK 4
#endif
// extern CollectiveGroup g_world;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


void getp_rmsnorm(float *o, float *x, float *weight, int batch_size, int dim) {
  // PROFILE_FUNCTION();
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim(batch_size);
  rmsnorm_kernel<<<gridDim, blockDim>>>(o, x, weight, batch_size, dim);
  //HIP_CHECK(hipDeviceSynchronize());
}

// Warp reduce sum (HIP wavefront = 64)
__device__ inline float warp_sum_f32(float v) {
#pragma unroll
  for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

union __bf16_bits_u {
  __hip_bfloat16 b;
  unsigned short u;
};
__device__ inline float bf16bits_to_f32(unsigned short u) {
  __bf16_bits_u t;
  t.u = u;
  return __bfloat162float(t.b);
}
template <int TILE_N, int WARPS_PER_BLOCK>
__global__ void gemv_qkv_bf16_vec(float *__restrict__ q_out,
                                  float *__restrict__ k_out,
                                  float *__restrict__ v_out,
                                  const float *__restrict__ x,
                                  const __hip_bfloat16 *__restrict__ W,
                                  const __hip_bfloat16 *__restrict__ B, int n,
                                  int q_len, int k_len, int v_len) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;  // 0..63
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int d_total = q_len + k_len + v_len;

  q_out += blockIdx.y * q_len;
  k_out += blockIdx.y * k_len;
  v_out += blockIdx.y * v_len;
  x += blockIdx.y * n;

  float partial_total = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = MIN(TILE_N, n - tile);

    // x -> shared, vector 16B
    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len)
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d_total) {
      const __hip_bfloat16 *__restrict__ wrow_b =
          W + (size_t)out_row * n + tile;

      // 128-bit load: 8 bf16 / lần
      const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow_b);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);

      float partial = 0.f;
      const int it8 = tile_len >> 3;
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 wb = w8[k8];
        unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
        unsigned short w02 = (unsigned short)(wb.x >> 16);
        unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
        unsigned short w12 = (unsigned short)(wb.y >> 16);
        unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
        unsigned short w22 = (unsigned short)(wb.z >> 16);
        unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
        unsigned short w32 = (unsigned short)(wb.w >> 16);

        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        partial = fmaf(bf16bits_to_f32(w01), xb0.x, partial);
        partial = fmaf(bf16bits_to_f32(w02), xb0.y, partial);
        partial = fmaf(bf16bits_to_f32(w11), xb0.z, partial);
        partial = fmaf(bf16bits_to_f32(w12), xb0.w, partial);
        partial = fmaf(bf16bits_to_f32(w21), xb1.x, partial);
        partial = fmaf(bf16bits_to_f32(w22), xb1.y, partial);
        partial = fmaf(bf16bits_to_f32(w31), xb1.z, partial);
        partial = fmaf(bf16bits_to_f32(w32), xb1.w, partial);
      }
      for (int k = (it8 << 3) + lane; k < tile_len; k += warpSize) {
        __bf16_bits_u u;
        u.b = wrow_b[k];
        partial += __bfloat162float(u.b) * sX[k];
      }
      partial_total += partial;
    }
    __syncthreads();
  }

  if (out_row < d_total) {
    float acc = warp_sum_f32(partial_total);
    if (threadIdx.x == 0) {
      float bias = 0.f;
      if (B) {
        __bf16_bits_u u;
        u.b = B[out_row];
        bias = __bfloat162float(u.b);
      }
      float *dst = nullptr;
      int idx = 0;
      if (out_row < q_len) {
        dst = q_out;
        idx = out_row;
      } else if (out_row < q_len + k_len) {
        dst = k_out;
        idx = out_row - q_len;
      } else {
        dst = v_out;
        idx = out_row - q_len - k_len;
      }
      dst[idx] = acc + bias;
    }
  }
}

static inline void getp_matmul_qkv_fused_bf16(
    float *q, float *k, float *v, float *x, const __hip_bfloat16 *w_qkv_bf16,
    const __hip_bfloat16 *b_qkv_bf16, int n, int head_dim, int n_attn_heads,
    int n_kv_heads, int batch_size) {
  // PROFILE_FUNCTION();
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;
  const int d_total = q_len + k_len + v_len;

  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((d_total + WARPS - 1) / WARPS, batch_size);
  size_t shmem = TILE * sizeof(float);
  gemv_qkv_bf16_vec<TILE, WARPS><<<grid, block, shmem>>>(
      q, k, v, x, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len);
  //HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK>
__global__ void gemv_bf16_vec_v2(float *__restrict__ y,
                                 const float *__restrict__ x,
                                 const __hip_bfloat16 *__restrict__ W,
                                 const __hip_bfloat16 *__restrict__ B, int n,
                                 int d) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  // blockIdx.y equal to batch index
  // shift x,y to the corresponding batch
  y += blockIdx.y * d;
  x += blockIdx.y * n;

  float partial_total = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = MIN(TILE_N, n - tile);

    // x -> shared (vector 16B)
    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len) {
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      } else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d) {
      const __hip_bfloat16 *wrow_b = W + (size_t)out_row * n + tile;

      // 128-bit load: 8×bf16 mỗi lần
      const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow_b);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);

      float partial = 0.f;
      const int it8 = tile_len >> 3;  // 8 elements per 128-bit load
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 wb = w8[k8];
        // tách 8 bf16: wb.x,y,z,w mỗi cái chứa 2 bf16 (4 bytes)
        unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
        unsigned short w02 = (unsigned short)(wb.x >> 16);
        unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
        unsigned short w12 = (unsigned short)(wb.y >> 16);
        unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
        unsigned short w22 = (unsigned short)(wb.z >> 16);
        unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
        unsigned short w32 = (unsigned short)(wb.w >> 16);

        // x tương ứng: 8 phần tử = 2 * float4
        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        partial += bf16bits_to_f32(w01) * xb0.x + bf16bits_to_f32(w02) * xb0.y +
                   bf16bits_to_f32(w11) * xb0.z + bf16bits_to_f32(w12) * xb0.w +
                   bf16bits_to_f32(w21) * xb1.x + bf16bits_to_f32(w22) * xb1.y +
                   bf16bits_to_f32(w31) * xb1.z + bf16bits_to_f32(w32) * xb1.w;
      }
      // tail còn lại (<8)
      for (int k = (it8 << 3) + lane; k < tile_len; k += warpSize) {
        __bf16_bits_u u;
        u.b = wrow_b[k];
        partial += __bfloat162float(u.b) * sX[k];
      }
      partial_total += partial;
    }
    __syncthreads();
  }

  if (out_row < d) {
    float acc = warp_sum_f32(partial_total);
    if (lane == 0) {
      float bias = 0.f;
      if (B) {
        __bf16_bits_u u;
        u.b = B[out_row];
        bias = __bfloat162float(u.b);
      }
      y[out_row] = acc + bias;
    }
  }
}
template <typename T>
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d,
                 int batch_size) {
  // PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE_N = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((d + WARPS - 1) / WARPS, batch_size);
  size_t shmem = TILE_N * sizeof(float);

  gemv_bf16_vec_v2<TILE_N, WARPS><<<grid, block, shmem>>>(
      xout, x, (const __hip_bfloat16 *)w, (const __hip_bfloat16 *)b, n, d);

  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void compute_inv_freq_kernel(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *inv_freq_out) {
  const int d_half = head_dim / 2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
  float inv_freq;
  if (scaling_factor > 1.0f) {
    float low = d_half *
                logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                logf(base);
    float high = d_half *
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }
  inv_freq_out[i] = inv_freq;
}

__global__ void compute_cos_sin_kernel(float *cos_out, float *sin_out,
                                       float *inv_freq, float concentration,
                                       int pos, int d_half) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float val = (float)pos * inv_freq[i];
  cos_out[i] = cosf(val) * concentration;
  sin_out[i] = sinf(val) * concentration;
}
void getp_compute_cos_sin(int pos, float base, int head_dim,
                          float scaling_factor, float initial_context_length,
                          float ntk_beta, float ntk_alpha,
                          float *cos_out, float *sin_out,
                          float *inv_freq) {
  // PROFILE_FUNCTION();
  int d_half = head_dim / 2;
  float concentration =
      scaling_factor > 1.0f ? 0.1f * logf(scaling_factor) + 1.0f : 1.0f;
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_inv_freq_kernel<<<gridDim, blockDim>>>(
        base, head_dim, scaling_factor, initial_context_length,
        ntk_beta, ntk_alpha, inv_freq);
  }
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_cos_sin_kernel<<<gridDim, blockDim>>>(
        cos_out, sin_out, inv_freq, concentration, pos, d_half);
  }
}

__global__ void apply_rotary_emb_kernel(float *x, float *cos, float *sin,
                                        int n_heads, int head_dim) {
  const int half = head_dim / 2;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // blockIdx.z equal to batch index
  // shift x to the corresponding batch
  x += blockIdx.z * n_heads * head_dim;
  if (h >= n_heads || i >= half) return;
  float x1 = x[h * head_dim + i];
  float x2 = x[h * head_dim + half + i];
  float c = cos[i];
  float s = sin[i];
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  x[h * head_dim + i] = o1;
  x[h * head_dim + half + i] = o2;
}
void getp_apply_rotary_emb(float *x, float *cos, float *sin, int n_heads,
                           int head_dim, int batch_size) {
  // PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x,
               (n_heads + blockDim.y - 1) / blockDim.y, batch_size);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads,
                                                 head_dim);
  //HIP_CHECK(hipDeviceSynchronize());
}

#ifndef GETP_ROUTER_TOPK_MAXK
#define GETP_ROUTER_TOPK_MAXK 4
#endif

__global__ void router_topk_softmax_batch_kernel(
    const float *__restrict__ router_score,  // [B, n_experts]
    int n_experts,
    int experts_per_token,  // K <= GETP_ROUTER_TOPK_MAXK
    float *__restrict__ topk_v_out,
    int *__restrict__ topk_i_out) {  // [B, K]
  const int b = blockIdx.x;          // sample index
  const int tid = threadIdx.x;
  const int K = experts_per_token;

  router_score += (size_t)b * n_experts;
  topk_v_out += (size_t)b * K;
  topk_i_out += (size_t)b * K;

  extern __shared__ unsigned char smem_raw[];
  float *scores = reinterpret_cast<float *>(smem_raw);
  float *valbuf = scores + n_experts;
  int *idxbuf = reinterpret_cast<int *>(valbuf + blockDim.x);
  float *topv = reinterpret_cast<float *>(idxbuf + blockDim.x);
  int *topi = reinterpret_cast<int *>(topv + GETP_ROUTER_TOPK_MAXK);

  for (int i = tid; i < n_experts; i += blockDim.x) scores[i] = router_score[i];
  __syncthreads();

  for (int sel = 0; sel < K; ++sel) {
    float best = -INFINITY;
    int besti = -1;
    for (int i = tid; i < n_experts; i += blockDim.x) {
      float v = scores[i];
      if (v > best) {
        best = v;
        besti = i;
      }
    }
    valbuf[tid] = best;
    idxbuf[tid] = besti;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        if (valbuf[tid + stride] > valbuf[tid]) {
          valbuf[tid] = valbuf[tid + stride];
          idxbuf[tid] = idxbuf[tid + stride];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      topv[sel] = valbuf[0];
      topi[sel] = idxbuf[0] < 0 ? 0 : idxbuf[0];
      scores[topi[sel]] = -INFINITY;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float m = topv[0];
    for (int i = 1; i < K; ++i)
      if (topv[i] > m) m = topv[i];
    float e[GETP_ROUTER_TOPK_MAXK];
    float s = 0.f;
    for (int i = 0; i < K; ++i) {
      e[i] = expf(topv[i] - m);
      s += e[i];
    }
    float invs = 1.f / s;
    for (int i = 0; i < K; ++i) {
      topk_v_out[i] = e[i] * invs;
      topk_i_out[i] = topi[i];
    }
  }
}

static inline void getp_router_topk_softmax_batch(
    const float *router_score, int n_experts, int experts_per_token,
    float *topk_v_out, int *topk_i_out, int batch_size) {
  // PROFILE_FUNCTION();
  const int BLK = 1024;
  const dim3 block(BLK);
  const dim3 grid(batch_size);
  const size_t shmem = (size_t)n_experts * sizeof(float) +
                       (size_t)BLK * (sizeof(float) + sizeof(int)) +
                       GETP_ROUTER_TOPK_MAXK * (sizeof(float) + sizeof(int));
  router_topk_softmax_batch_kernel<<<grid, block, shmem>>>(
      router_score, n_experts, experts_per_token, topk_v_out, topk_i_out);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void map_global_to_local_batch_kernel(
    const int *__restrict__ topk_i,    // [B,K]
    const float *__restrict__ topk_v,  // [B,K]
    int *__restrict__ local_ids,       // [B,K]
    float *__restrict__ local_wts,     // [B,K]
    int *__restrict__ n_local,         // [B]
    int K, int expert_start, int expert_end, int B) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  int base = b * K, cnt = 0;
  for (int i = 0; i < K; ++i) {
    int eg = topk_i[base + i];
    float w = topk_v[base + i];
    if (eg >= expert_start && eg < expert_end) {
      local_ids[base + cnt] = eg - expert_start;
      local_wts[base + cnt] = w;
      ++cnt;
    }
  }
  for (int i = cnt; i < K; ++i) {
    local_ids[base + i] = -1;
    local_wts[base + i] = 0.f;
  }
  n_local[b] = cnt;
}

static inline void getp_map_global_to_local_batch(
    const int *topk_i, const float *topk_v, int *local_ids, float *local_wts,
    int *n_local, int K, int expert_start, int expert_end, int B) {
  // PROFILE_FUNCTION();
  const int BLK = 128, GRD = (B + BLK - 1) / BLK;
  map_global_to_local_batch_kernel<<<GRD, BLK>>>(topk_i, topk_v, local_ids,
                                                 local_wts, n_local, K,
                                                 expert_start, expert_end, B);
  //HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK, int MAX_E = 4>
__global__ void mlp1_swiglu_bf16_kernel_batch_gridy(
    float *__restrict__ gate_up_all,             // [B,K,I]
    const float *__restrict__ x,                 // [B,H]
    const __hip_bfloat16 *__restrict__ W_layer,  // [E_dev, 2I, H]
    const __hip_bfloat16 *__restrict__ B_layer,  // [E_dev, 2I]
    int H, int I, int E_dev,
    const int *__restrict__ local_ids,  // [B,K]
    const int *__restrict__ n_local,    // [B]
    int K, float swiglu_limit) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int j = blockIdx.x * WARPS_PER_BLOCK + warp;  // 0..I-1
  const int b = blockIdx.y;
  if (j >= I) return;

  const int nact = MIN(n_local[b], K);
  const int ids0 = local_ids[b * K + 0], ids1 = local_ids[b * K + 1];
  const int ids2 = local_ids[b * K + 2], ids3 = local_ids[b * K + 3];

  float g_tot[MAX_E] = {0, 0, 0, 0};
  float u_tot[MAX_E] = {0, 0, 0, 0};

  const float *xb = x + (size_t)b * H;
  for (int tile = 0; tile < H; tile += TILE_N) {
    const int tlen = MIN(TILE_N, H - tile);
    for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tlen)
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&xb[tile + t]);
      else
        for (int k = 0; k < 4 && t + k < tlen; ++k)
          sX[t + k] = xb[tile + t + k];
    }
    __syncthreads();

    const float4 *x4 = reinterpret_cast<const float4 *>(sX);
    const int it8 = tlen >> 3;

#pragma unroll
    for (int ee = 0; ee < MAX_E; ++ee) {
      if (ee >= nact) break;
      const int lid =
          (ee == 0 ? ids0 : (ee == 1 ? ids1 : (ee == 2 ? ids2 : ids3)));
      if (lid < 0) continue;

      const size_t row_stride = (size_t)H;
      const size_t expert_base = (size_t)lid * (size_t)(2 * I) * (size_t)H;
      const __hip_bfloat16 *wg =
          W_layer + expert_base + (size_t)(2 * j + 0) * row_stride + tile;
      const __hip_bfloat16 *wu =
          W_layer + expert_base + (size_t)(2 * j + 1) * row_stride + tile;

      const uint4 *wg8 = reinterpret_cast<const uint4 *>(wg);
      const uint4 *wu8 = reinterpret_cast<const uint4 *>(wu);
      float g = 0.f, u = 0.f;
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 ag = wg8[k8], au = wu8[k8];

        unsigned short g01 = (unsigned short)(ag.x & 0xFFFF);
        unsigned short g02 = (unsigned short)(ag.x >> 16);
        unsigned short g11 = (unsigned short)(ag.y & 0xFFFF);
        unsigned short g12 = (unsigned short)(ag.y >> 16);
        unsigned short g21 = (unsigned short)(ag.z & 0xFFFF);
        unsigned short g22 = (unsigned short)(ag.z >> 16);
        unsigned short g31 = (unsigned short)(ag.w & 0xFFFF);
        unsigned short g32 = (unsigned short)(ag.w >> 16);

        unsigned short u01 = (unsigned short)(au.x & 0xFFFF);
        unsigned short u02 = (unsigned short)(au.x >> 16);
        unsigned short u11 = (unsigned short)(au.y & 0xFFFF);
        unsigned short u12 = (unsigned short)(au.y >> 16);
        unsigned short u21 = (unsigned short)(au.z & 0xFFFF);
        unsigned short u22 = (unsigned short)(au.z >> 16);
        unsigned short u31 = (unsigned short)(au.w & 0xFFFF);
        unsigned short u32 = (unsigned short)(au.w >> 16);

        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        g = fmaf(bf16bits_to_f32(g01), xb0.x, g);
        g = fmaf(bf16bits_to_f32(g02), xb0.y, g);
        g = fmaf(bf16bits_to_f32(g11), xb0.z, g);
        g = fmaf(bf16bits_to_f32(g12), xb0.w, g);
        g = fmaf(bf16bits_to_f32(g21), xb1.x, g);
        g = fmaf(bf16bits_to_f32(g22), xb1.y, g);
        g = fmaf(bf16bits_to_f32(g31), xb1.z, g);
        g = fmaf(bf16bits_to_f32(g32), xb1.w, g);

        u = fmaf(bf16bits_to_f32(u01), xb0.x, u);
        u = fmaf(bf16bits_to_f32(u02), xb0.y, u);
        u = fmaf(bf16bits_to_f32(u11), xb0.z, u);
        u = fmaf(bf16bits_to_f32(u12), xb0.w, u);
        u = fmaf(bf16bits_to_f32(u21), xb1.x, u);
        u = fmaf(bf16bits_to_f32(u22), xb1.y, u);
        u = fmaf(bf16bits_to_f32(u31), xb1.z, u);
        u = fmaf(bf16bits_to_f32(u32), xb1.w, u);
      }
      for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
        g += __bfloat162float(wg[k]) * sX[k];
        u += __bfloat162float(wu[k]) * sX[k];
      }
      g_tot[ee] += g;
      u_tot[ee] += u;
    }
    __syncthreads();
  }

#pragma unroll
  for (int ee = 0; ee < MAX_E; ++ee) {
    if (ee >= nact) break;
    float g = g_tot[ee], u = u_tot[ee];
#pragma unroll
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
      g += __shfl_down(g, off);
      u += __shfl_down(u, off);
    }
    if (lane == 0) {
      const int lid =
          (ee == 0 ? ids0 : (ee == 1 ? ids1 : (ee == 2 ? ids2 : ids3)));
      if (lid >= 0) {
        const size_t b_off = (size_t)lid * (size_t)(2 * I);
        float bg =
            B_layer ? __bfloat162float(B_layer[b_off + (2 * j + 0)]) : 0.f;
        float bu =
            B_layer ? __bfloat162float(B_layer[b_off + (2 * j + 1)]) : 0.f;
        g += bg;
        u += bu;
        if (g > swiglu_limit) g = swiglu_limit;
        if (u > swiglu_limit) u = swiglu_limit;
        if (u < -swiglu_limit) u = -swiglu_limit;
        const float a = 1.702f;
        float s = 1.f / (1.f + expf(-a * g));
        float *dst = gate_up_all + ((size_t)b * K + ee) * (size_t)I;
        dst[j] = (g * s) * (u + 1.f);
      }
    }
  }
}

static inline void getp_mlp1_swiglu_bf16_batch_gridy(
    float *gate_up_all, const float *x, const __hip_bfloat16 *w1_layer_base,
    const __hip_bfloat16 *b1_layer_base, int H, int I, int E_dev,
    const int *local_ids, const int *n_local, int K, int B,
    float swiglu_limit) {
  // PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((I + WARPS - 1) / WARPS, B);
  size_t shmem = TILE * sizeof(float);
  mlp1_swiglu_bf16_kernel_batch_gridy<TILE, WARPS>
      <<<grid, block, shmem>>>(gate_up_all, x, w1_layer_base, b1_layer_base, H,
                               I, E_dev, local_ids, n_local, K, swiglu_limit);
  //HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK, int MAX_E = 4>
__global__ void mlp2_accum_bf16_kernel_batch_gridy(
    float *__restrict__ e_agg,                   // [B,H] +=
    const float *__restrict__ gate_up_all,       // [B,K,I]
    const __hip_bfloat16 *__restrict__ W_layer,  // [E_dev,H,I]
    const __hip_bfloat16 *__restrict__ B_layer,  // [E_dev,H]
    const float *__restrict__ local_wts,         // [B,K]
    const int *__restrict__ local_ids,           // [B,K]
    const int *__restrict__ n_local,             // [B]
    int K, int I, int H) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int b = blockIdx.y;
  if (row >= H) return;

  const int nact = MIN(n_local[b], K);
  const int ids0 = local_ids[b * K + 0], ids1 = local_ids[b * K + 1];
  const int ids2 = local_ids[b * K + 2], ids3 = local_ids[b * K + 3];
  const float wt0 = local_wts[b * K + 0], wt1 = local_wts[b * K + 1];
  const float wt2 = local_wts[b * K + 2], wt3 = local_wts[b * K + 3];

  float acc[MAX_E] = {0, 0, 0, 0};
  const float *gub = gate_up_all + (size_t)b * (size_t)K * (size_t)I;

  for (int tile = 0; tile < I; tile += TILE_N) {
    const int tlen = MIN(TILE_N, I - tile);
#pragma unroll
    for (int ee = 0; ee < MAX_E; ++ee) {
      if (ee >= nact) break;
      const float *src = gub + (size_t)ee * (size_t)I + tile;
      float *dst = sX + ee * TILE_N;
      for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
           t += WARPS_PER_BLOCK * warpSize * 4) {
        if (t + 3 < tlen)
          reinterpret_cast<float4 &>(dst[t]) =
              *reinterpret_cast<const float4 *>(&src[t]);
        else
          for (int k = 0; k < 4 && t + k < tlen; ++k) dst[t + k] = src[t + k];
      }
    }
    __syncthreads();

    const int it8 = tlen >> 3;
#pragma unroll 1
    for (int ee = 0; ee < nact; ++ee) {
      const int lid =
          (ee == 0 ? ids0 : (ee == 1 ? ids1 : (ee == 2 ? ids2 : ids3)));
      if (lid < 0) continue;
      const size_t expert_base = (size_t)lid * (size_t)H * (size_t)I;
      const __hip_bfloat16 *wrow =
          W_layer + expert_base + (size_t)row * (size_t)I + tile;

      const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX + ee * TILE_N);

      float part = 0.f;
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 wb = w8[k8];
        unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
        unsigned short w02 = (unsigned short)(wb.x >> 16);
        unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
        unsigned short w12 = (unsigned short)(wb.y >> 16);
        unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
        unsigned short w22 = (unsigned short)(wb.z >> 16);
        unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
        unsigned short w32 = (unsigned short)(wb.w >> 16);

        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        part = fmaf(bf16bits_to_f32(w01), xb0.x, part);
        part = fmaf(bf16bits_to_f32(w02), xb0.y, part);
        part = fmaf(bf16bits_to_f32(w11), xb0.z, part);
        part = fmaf(bf16bits_to_f32(w12), xb0.w, part);
        part = fmaf(bf16bits_to_f32(w21), xb1.x, part);
        part = fmaf(bf16bits_to_f32(w22), xb1.y, part);
        part = fmaf(bf16bits_to_f32(w31), xb1.z, part);
        part = fmaf(bf16bits_to_f32(w32), xb1.w, part);
      }
      for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
        part += __bfloat162float(wrow[k]) * (sX[ee * TILE_N + k]);
      }
      acc[ee] += part;
    }
    __syncthreads();
  }

#pragma unroll
  for (int ee = 0; ee < MAX_E; ++ee) acc[ee] = warp_sum_f32(acc[ee]);

  if (threadIdx.x == 0) {
    float sum = 0.f;
    if (nact > 0 && ids0 >= 0) {
      float bias =
          B_layer ? __bfloat162float(B_layer[(size_t)ids0 * (size_t)H + row])
                  : 0.f;
      sum += wt0 * (acc[0] + bias);
    }
    if (nact > 1 && ids1 >= 0) {
      float bias =
          B_layer ? __bfloat162float(B_layer[(size_t)ids1 * (size_t)H + row])
                  : 0.f;
      sum += wt1 * (acc[1] + bias);
    }
    if (nact > 2 && ids2 >= 0) {
      float bias =
          B_layer ? __bfloat162float(B_layer[(size_t)ids2 * (size_t)H + row])
                  : 0.f;
      sum += wt2 * (acc[2] + bias);
    }
    if (nact > 3 && ids3 >= 0) {
      float bias =
          B_layer ? __bfloat162float(B_layer[(size_t)ids3 * (size_t)H + row])
                  : 0.f;
      sum += wt3 * (acc[3] + bias);
    }
    e_agg[(size_t)b * (size_t)H + row] += sum;
  }
}

static inline void getp_mlp2_accum_bf16_batch_gridy(
    float *e_agg, const float *gate_up_all, const __hip_bfloat16 *w2_layer_base,
    const __hip_bfloat16 *b2_layer_base, const int *local_ids,
    const float *local_wts, const int *n_local, int K, int B, int I, int H) {
  // PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((H + WARPS - 1) / WARPS, B);
  size_t shmem = (size_t)TILE * 4 * sizeof(float);
  mlp2_accum_bf16_kernel_batch_gridy<TILE, WARPS>
      <<<grid, block, shmem>>>(e_agg, gate_up_all, w2_layer_base, b2_layer_base,
                               local_wts, local_ids, n_local, K, I, H);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void vecadd_kernel(float *x, float *y, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // blockIdx.y equal to batch index
  // shift x, y to the corresponding batch
  x += blockIdx.y * size;
  y += blockIdx.y * size;
  if (i >= size) return;
  x[i] += y[i];
}
void getp_vecadd(float *x, float *y, int size, int batch_size) {
  // PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x, batch_size);
  vecadd_kernel<<<gridDim, blockDim>>>(x, y, size);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void flash_attn_decode_kernel(
    float *__restrict__ tb, const float *__restrict__ key_cache,
    const float *__restrict__ value_cache, const float *__restrict__ q,
    const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int seq_len, int sliding_window, int apply_mask,
    int batch_size, int kv_dim, int kv_mul, float inv_sqrt_d) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int i = threadIdx.x;

  __shared__ float sbuf[2];

  // Pointers
  const float *qbh =
      q + (size_t)b * n_attn_heads * head_dim + (size_t)h * head_dim;
  float *obh = tb + (size_t)b * n_attn_heads * head_dim + (size_t)h * head_dim;

  const int kv_h = h / kv_mul;

  float qi = (i < head_dim) ? qbh[i] : 0.0f;
  float out_i = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  int t_start = 0;
  if (apply_mask && sliding_window > 0) {
    t_start = MAX(0, pos - sliding_window + 1);
  }

  for (int t = t_start; t <= pos; ++t) {
    float k_i = 0.f;
    if (i < head_dim) {
      const float *kptr = key_cache + (size_t)t * batch_size * kv_dim +
                          (size_t)b * kv_dim + (size_t)kv_h * head_dim + i;
      k_i = *kptr;
    }
    float part = qi * k_i;
    float sum = warp_sum_f32(part);

    if (threadIdx.x == 0) {
      float s = sum * inv_sqrt_d;
      if (apply_mask && sliding_window > 0) {
        if ((pos - t) >= sliding_window) s = -INFINITY;
      }
      float m_new = fmaxf(m, s);
      float alpha = __expf(m - m_new);
      float e = __expf(s - m_new);
      l = l * alpha + e;
      m = m_new;
      sbuf[0] = e;
      sbuf[1] = alpha;
    }
    __syncthreads();

    float e = sbuf[0];
    float alpha = sbuf[1];

    if (i < head_dim) {
      const float *vptr = value_cache + (size_t)t * batch_size * kv_dim +
                          (size_t)b * kv_dim + (size_t)kv_h * head_dim + i;
      out_i = alpha * out_i + e * (*vptr);
    }
    __syncthreads();
  }

  // Attention sink
  if (threadIdx.x == 0) {
    float s_sink = attn_sinks[h];
    float m_new = fmaxf(m, s_sink);
    float alpha = __expf(m - m_new);
    float e = __expf(s_sink - m_new);
    l = l * alpha + e;
    m = m_new;
    sbuf[0] = alpha;
    sbuf[1] = l;
  }
  __syncthreads();

  float alpha_sink = sbuf[0];
  float l_final = sbuf[1];

  if (i < head_dim) {
    out_i = (alpha_sink * out_i) / l_final;
    obh[i] = out_i;
  }
}
static inline void getp_flash_attn_decode(
    float *tb, const float *key_cache_layer, const float *value_cache_layer,
    const float *q, const float *attn_sinks_layer,  // attn_sinks + l*n_heads
    int head_dim, int n_attn_heads, int n_kv_heads, int pos, int seq_len,
    int sliding_window, int layer_id, int batch_size) {
  // PROFILE_FUNCTION();
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
  const int apply_mask = (sliding_window > 0 && ((layer_id & 1) == 0)) ? 1 : 0;

  dim3 block(64);
  dim3 grid(n_attn_heads, batch_size);
  flash_attn_decode_kernel<<<grid, block>>>(
      tb, key_cache_layer, value_cache_layer, q, attn_sinks_layer, head_dim,
      n_attn_heads, n_kv_heads, pos, seq_len, sliding_window, apply_mask,
      batch_size, kv_dim, kv_mul, inv_sqrt_d);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void gather_embedding_kernel(
    float* __restrict__ x,
    const float* __restrict__ table,
    const int* __restrict__ tok,
    int H) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int id = tok[b];
  const float* src = table + (size_t)id * H;
  float* dst = x + (size_t)b * H;
  for (int j = tid; j < H; j += blockDim.x) dst[j] = src[j];
}
static inline void getp_gather_embedding(
    float* x, const float* table, const int* tok, int H, int B) {
  dim3 block(256), grid(B);
  gather_embedding_kernel<<<grid, block>>>(x, table, tok, H);
}

__global__ void argmax_rows_kernel(
    const float* __restrict__ logits, int V, int* __restrict__ out) {
  __shared__ float smax[1024];
  __shared__ int   sidx[1024];
  int b = blockIdx.x;
  int tid = threadIdx.x;
  const float* row = logits + (size_t)b * V;
  float mv = -INFINITY; int mi = 0;
  for (int i = tid; i < V; i += blockDim.x) {
    float v = row[i];
    if (v > mv) { mv = v; mi = i; }
  }
  smax[tid] = mv; sidx[tid] = mi; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (smax[tid + s] > smax[tid]) {
        smax[tid] = smax[tid + s];
        sidx[tid] = sidx[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[b] = sidx[0];
}
static inline void getp_argmax_rows(
    const float* logits, int V, int* out, int B) {
  dim3 grid(B), block(1024);
  argmax_rows_kernel<<<grid, block>>>(logits, V, out);
}


struct GPUWorker {
  int device_index;

  int expert_start;
  int expert_end;

  int request_start;
  int request_end;
};
void moe_gpu(GPUTransformer *gpu_t, int l, int batch_size)
{
Config *p = &gpu_t->config;
GPURunState *dev_s = &gpu_t->state;
GPUTransformerWeights *dev_w = &gpu_t->weights;
CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
// ExpertiseExt *ext = &cpu_buf->expertise_ext;
GPUWorker *worker = nullptr;
worker = (GPUWorker *)malloc(sizeof(GPUWorker));
worker->device_index = 0;
worker->expert_start = 0;
worker->expert_end = p->n_experts;
worker->request_start = 0;
worker->request_end = batch_size;
float *dev_x = dev_s->x;

int hidden_dim = p->hidden_dim;
int n_experts = p->n_experts;

    {
      getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                BATCH_SIZE, hidden_dim);

  __hip_bfloat16 *dev_w_router =
      dev_w->w_router + 1ll * l * hidden_dim * n_experts;
  __hip_bfloat16 *dev_b_router = dev_w->b_router + 1ll * l * n_experts;
  getp_matmul<__hip_bfloat16>(dev_s->router_score, dev_s->t, dev_w_router,
                              dev_b_router, hidden_dim, n_experts,
                              BATCH_SIZE);

  getp_router_topk_softmax_batch(dev_s->router_score, n_experts,
                                  p->experts_per_token, dev_s->topk_v,
                                  dev_s->topk_i, BATCH_SIZE);

  getp_map_global_to_local_batch(dev_s->topk_i, dev_s->topk_v, dev_s->local_ids,
                                  dev_s->local_wts, dev_s->n_local,
                                  p->experts_per_token, worker->expert_start,
                                  worker->expert_end, BATCH_SIZE);

  HIP_CHECK(hipMemset(dev_s->e_agg, 0,
                      (size_t)BATCH_SIZE * hidden_dim * sizeof(float)));
                  }

  int experts_per_device = worker->expert_end - worker->expert_start;

  __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 *
                                                p->intermediate_dim *
                                                hidden_dim;
  __hip_bfloat16 *b1_base =
      dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;

  {
    // PROFILE_BLOCK("mlp1");
    getp_mlp1_swiglu_bf16_batch_gridy(
      dev_s->gate_up, dev_s->t, w1_base, b1_base, hidden_dim,
      p->intermediate_dim, experts_per_device, dev_s->local_ids, dev_s->n_local,
      p->experts_per_token, BATCH_SIZE, p->swiglu_limit);
    }

  __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device *
                                                hidden_dim *
                                                p->intermediate_dim;
  __hip_bfloat16 *b2_base =
      dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;

  {
    // PROFILE_BLOCK("mlp2");
    getp_mlp2_accum_bf16_batch_gridy(
      dev_s->e_agg, dev_s->gate_up, w2_base, b2_base, dev_s->local_ids,
      dev_s->local_wts, dev_s->n_local, p->experts_per_token, BATCH_SIZE,
      p->intermediate_dim, hidden_dim);
    }

  {
    // PROFILE_BLOCK("add");
    // add to residual
  getp_vecadd(dev_x, dev_s->e_agg, hidden_dim, BATCH_SIZE);
  }
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
    debug(s->x, 100);
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