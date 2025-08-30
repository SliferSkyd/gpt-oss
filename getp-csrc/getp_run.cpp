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

    // EXPERT PARALLELISM CHANGE: MoE weights are sharded. Each GPU holds half.
    // Shape is now [n_layer, n_experts/2, ...]
    uint8_t *w_mlp1_mxfp4, *w_mlp2_mxfp4; // Packed MXFP4 indices for local experts
    float *w_mlp1_scales, *w_mlp2_scales; // MXFP4 block scales for local experts
    __hip_bfloat16 *b_mlp1, *b_mlp2;      // Biases remain in bfloat16 for local experts

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
    __hip_bfloat16 *key_cache;   // (batch_size, n_layers, seq_len, kv_dim)
    __hip_bfloat16 *value_cache; // (batch_size, n_layers, seq_len, kv_dim)

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
    int *peer_indices;           // peer expert indices
    float *peer_weights;         // peer expert weights
    int *batch_count;            // number of tokens in the batch (1 element)

    // EXPERT PARALLELISM CHANGE: Buffers for peer communication
    float *peer_input_buffer;   // Buffer to send tokens to peer
    float *peer_output_buffer;  // Buffer to receive results from peer
    int *peer_expert_indices;   // Indices for tokens sent to peer
    float *peer_expert_weights; // Weights for tokens sent to peer
    int *d_peer_expert_counts;  // Counts of tokens routed to each peer expert
    int *d_peer_token_count;    // Total number of tokens to send to peer

    // Persistent expert routing buffers (avoid hipMalloc/hipFree in forward pass)
    int *d_local_expert_counts;    // token counts per local expert (n_experts/2)
    int *d_local_expert_offsets;   // prefix sum of local expert counts (n_experts/2)
    int *d_local_expert_write_idx; // write indices for local expert gathering (n_experts/2)
    int *d_total_tokens;           // total tokens across all experts (1 element)

    // Additional expert routing buffers needed by the code

    int *d_peer_expert_offsets;    // offsets for peer experts
    int *d_local_write_idx;        // write indices for local experts
    int *d_peer_write_idx;         // write indices for peer experts
    int *d_received_expert_counts; // received expert counts from peer

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
    hipStream_t sComm; // EXPERT PARALLELISM CHANGE: Stream for peer communication
    hipStream_t sMLP[N_MLP_STREAMS];

    // EXPERT PARALLELISM CHANGE: Host-side buffers for local and peer counts/offsets
    int *local_expert_counts;
    int *local_expert_offsets;
    int *peer_expert_counts;
    int *peer_expert_offsets;
    int *received_expert_counts;  // received from peer
    int *received_expert_offsets; // offsets for received tokens
    int *expert_counts;           // all expert counts
    int *expert_offsets;          // all expert offsets

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
    Config config;                    // model configuration
    GPUTransformerWeights weights;    // GPU weights
    GPURunState state;                // GPU run state buffers
    CPUBuffers cpu_buffers;           // CPU buffers for host operations
    int peer_device_id;               // EXPERT PARALLELISM CHANGE: ID of peer GPU
    hipEvent_t evPeerDataSent;        // Event for when this GPU sends data to peer
    hipEvent_t evPeerResultsSent;     // Event for when this GPU sends results to peer
    hipEvent_t evPeerDataReceived;    // Event for peer data received (deprecated - kept for compatibility)
    hipEvent_t evPeerResultsReceived; // Event for peer results received (deprecated - kept for compatibility)
} GPUTransformer;

// Global variables for direct access in batched_generate_gpu
static GPUTransformer *gpu_transformers[MAX_GPUS];

// Memory allocation functions
void malloc_gpu_run_state(GPURunState *s, Config *p)
{
    int kv_dim = p->head_dim * p->n_kv_heads;
    int expert_per_token = p->experts_per_token;
    int n_local_experts = p->n_experts / 2; // Each GPU holds half the experts
    // Initialize pointers to NULL

    // Check memory requirements and print for debugging
    size_t total_memory = 0;
    size_t batch_hidden = BATCH_SIZE * p->hidden_dim * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);
    // BF16 KV cache - 50% memory reduction compared to FP32
    size_t kv_cache_size = BATCH_SIZE * p->n_layers * MAX_SEQ_LEN * kv_dim * sizeof(__hip_bfloat16);

    printf("Allocating GPU memory: batch_size=%lld, hidden_dim=%d, seq_len=%d\n",
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

    // EXPERT PARALLELISM CHANGE: Allocate new buffers
    size_t max_peer_tokens_size = BATCH_SIZE * expert_per_token;
    HIP_CHECK(hipMalloc((void **)&s->peer_input_buffer, max_peer_tokens_size * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->peer_output_buffer, max_peer_tokens_size * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->peer_expert_indices, max_peer_tokens_size * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->peer_expert_weights, max_peer_tokens_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->d_peer_expert_counts, n_local_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_peer_token_count, sizeof(int)));

    // NEW: Buffers for GPU scatter-gather MoE
    // Max possible items for one expert is the entire batch
    HIP_CHECK(hipMalloc((void **)&s->expert_indices, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->expert_weights, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->batch_count, sizeof(int)));
    // This buffer holds the expert's final output before scattering
    HIP_CHECK(hipMalloc((void **)&s->expert_output_buffer, BATCH_SIZE * p->hidden_dim * sizeof(float) * expert_per_token));

    // Persistent expert routing buffers to avoid hipMalloc/hipFree in forward pass

    HIP_CHECK(hipMalloc((void **)&s->d_local_expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_local_expert_offsets, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_local_expert_write_idx, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_total_tokens, sizeof(int)));

    // Additional allocations for the new fields
    HIP_CHECK(hipMalloc((void **)&s->d_peer_expert_offsets, n_local_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_local_write_idx, n_local_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_peer_write_idx, n_local_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_received_expert_counts, n_local_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->peer_indices, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->peer_weights, BATCH_SIZE * p->experts_per_token * sizeof(float)));

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

void malloc_gpu_weights(GPUTransformerWeights *w, Config *p, int gpu_id)
{

    // EXPERT PARALLELISM CHANGE: Allocate only half of the MoE weights
    int n_local_experts = p->n_experts / 2;

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

    // MoE weights allocation for n_local_experts
    size_t mlp1_size = (size_t)p->n_layers * n_local_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    size_t mlp1_packed_size = (mlp1_size + 1) / 2;
    size_t mlp1_num_blocks = (mlp1_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_mxfp4, mlp1_packed_size));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_scales, mlp1_num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, (size_t)p->n_layers * n_local_experts * (2 * p->intermediate_dim) * sizeof(__hip_bfloat16)));

    size_t mlp2_size = (size_t)p->n_layers * n_local_experts * p->hidden_dim * p->intermediate_dim;
    size_t mlp2_packed_size = (mlp2_size + 1) / 2;
    size_t mlp2_num_blocks = (mlp2_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_mxfp4, mlp2_packed_size));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_scales, mlp2_num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2, (size_t)p->n_layers * n_local_experts * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->out, p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->attn_sinks, p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));
}

void copy_weights_to_gpu(Transformer *transformer, GPUTransformerWeights *gpu_weights, int gpu_id)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    // EXPERT PARALLELISM CHANGE: Determine which slice of expert weights to copy
    int n_local_experts = p->n_experts / 2;
    int expert_start_idx = (gpu_id % 2 == 0) ? 0 : n_local_experts;

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

    // --- Sao chép các trọng số MoE Expert (phân mảnh) ---

    // Trọng số MLP1 (Gate/Up) - Quantized MXFP4
    size_t mlp1_full_layer_size = (size_t)p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    size_t mlp1_local_layer_size = (size_t)n_local_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    for (int l = 0; l < p->n_layers; ++l)
    {
        float *layer_w_mlp1 = w->w_mlp1 + l * mlp1_full_layer_size;
        float *shard_start = layer_w_mlp1 + expert_start_idx * (2 * p->intermediate_dim) * p->hidden_dim;

        MXFP4Weights mlp1_mxfp4;
        quantize_to_mxfp4(shard_start, mlp1_mxfp4, mlp1_local_layer_size);

        uint8_t *gpu_w_ptr = gpu_weights->w_mlp1_mxfp4 + l * ((mlp1_local_layer_size + 1) / 2);
        float *gpu_s_ptr = gpu_weights->w_mlp1_scales + l * ((mlp1_local_layer_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE);

        HIP_CHECK(hipMemcpy(gpu_w_ptr, mlp1_mxfp4.packed_values, (mlp1_mxfp4.num_elements + 1) / 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_s_ptr, mlp1_mxfp4.scales, mlp1_mxfp4.num_blocks * sizeof(float), hipMemcpyHostToDevice));
        free_mxfp4_weights(mlp1_mxfp4);
    }

    // Biases MLP1 - bfloat16
    size_t b_mlp1_full_layer_size = (size_t)p->n_experts * (2 * p->intermediate_dim);
    size_t b_mlp1_local_layer_size = (size_t)n_local_experts * (2 * p->intermediate_dim);
    __hip_bfloat16 *h_b_mlp1_bf16 = (__hip_bfloat16 *)malloc(b_mlp1_local_layer_size * sizeof(__hip_bfloat16));
    for (int l = 0; l < p->n_layers; ++l)
    {
        float *layer_b_mlp1 = w->b_mlp1 + l * b_mlp1_full_layer_size;
        float *shard_start = layer_b_mlp1 + expert_start_idx * (2 * p->intermediate_dim);
        convert_float_array_to_bfloat16(shard_start, h_b_mlp1_bf16, b_mlp1_local_layer_size);
        HIP_CHECK(hipMemcpy(gpu_weights->b_mlp1 + l * b_mlp1_local_layer_size, h_b_mlp1_bf16, b_mlp1_local_layer_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    }
    free(h_b_mlp1_bf16);

    // Trọng số MLP2 (Down) - Quantized MXFP4
    size_t mlp2_full_layer_size = (size_t)p->n_experts * p->hidden_dim * p->intermediate_dim;
    size_t mlp2_local_layer_size = (size_t)n_local_experts * p->hidden_dim * p->intermediate_dim;
    for (int l = 0; l < p->n_layers; ++l)
    {
        float *layer_w_mlp2 = w->w_mlp2 + l * mlp2_full_layer_size;
        float *shard_start = layer_w_mlp2 + expert_start_idx * p->hidden_dim * p->intermediate_dim;

        MXFP4Weights mlp2_mxfp4;
        quantize_to_mxfp4(shard_start, mlp2_mxfp4, mlp2_local_layer_size);

        uint8_t *gpu_w_ptr = gpu_weights->w_mlp2_mxfp4 + l * ((mlp2_local_layer_size + 1) / 2);
        float *gpu_s_ptr = gpu_weights->w_mlp2_scales + l * ((mlp2_local_layer_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE);

        HIP_CHECK(hipMemcpy(gpu_w_ptr, mlp2_mxfp4.packed_values, (mlp2_mxfp4.num_elements + 1) / 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_s_ptr, mlp2_mxfp4.scales, mlp2_mxfp4.num_blocks * sizeof(float), hipMemcpyHostToDevice));
        free_mxfp4_weights(mlp2_mxfp4);
    }

    // Biases MLP2 - bfloat16
    size_t b_mlp2_full_layer_size = (size_t)p->n_experts * p->hidden_dim;
    size_t b_mlp2_local_layer_size = (size_t)n_local_experts * p->hidden_dim;
    __hip_bfloat16 *h_b_mlp2_bf16 = (__hip_bfloat16 *)malloc(b_mlp2_local_layer_size * sizeof(__hip_bfloat16));
    for (int l = 0; l < p->n_layers; ++l)
    {
        float *layer_b_mlp2 = w->b_mlp2 + l * b_mlp2_full_layer_size;
        float *shard_start = layer_b_mlp2 + expert_start_idx * p->hidden_dim;
        convert_float_array_to_bfloat16(shard_start, h_b_mlp2_bf16, b_mlp2_local_layer_size);
        HIP_CHECK(hipMemcpy(gpu_weights->b_mlp2 + l * b_mlp2_local_layer_size, h_b_mlp2_bf16, b_mlp2_local_layer_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    }
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
    int n_local_experts = p->n_experts / 2;

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

    // EXPERT PARALLELISM CHANGE: Allocate host buffers for local and peer counts
    HIP_CHECK(hipHostMalloc(&cpu_buf->local_expert_counts, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->local_expert_offsets, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->peer_expert_counts, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->peer_expert_offsets, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->received_expert_counts, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->received_expert_offsets, n_local_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->expert_offsets, p->n_experts * sizeof(int)));

    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sGather, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sScatter, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sComm, hipStreamNonBlocking)); // Comm stream
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sMLP[i], hipStreamNonBlocking));
    HIP_CHECK(hipHostMalloc(&cpu_buf->logits, BATCH_SIZE * p->vocab_size * sizeof(float)));
}

void build_gpu_transformer(GPUTransformer *gpu_t, Transformer *cpu_t, int gpu_id)
{
    // Copy config
    gpu_t->config = cpu_t->config;

    // Allocate GPU memory
    malloc_gpu_weights(&gpu_t->weights, &gpu_t->config, gpu_id);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);

    // Copy weights to GPU
    copy_weights_to_gpu(cpu_t, &gpu_t->weights, gpu_id);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    // Create GPU transformer
    // Multi-GPU support: allocate and initialize GPUTransformer for each GPU
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus > MAX_GPUS)
        num_gpus = MAX_GPUS;
    // EXPERT PARALLELISM CHANGE: Must have an even number of GPUs
    if (num_gpus % 2 != 0 && num_gpus > 1)
    {
        fprintf(stderr, "Expert Parallelism requires an even number of GPUs. Found %d.\n", num_gpus);
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev)
    {
        HIP_CHECK(hipSetDevice(dev));
        gpu_transformers[dev] = (GPUTransformer *)malloc(sizeof(GPUTransformer));
        assert(gpu_transformers[dev] != NULL);

        GPUTransformer *gpu_transformer = gpu_transformers[dev];

        // EXPERT PARALLELISM CHANGE: Enable peer access
        if (num_gpus > 1)
        {
            gpu_transformer->peer_device_id = (dev % 2 == 0) ? dev + 1 : dev - 1;
            HIP_CHECK(hipDeviceEnablePeerAccess(gpu_transformer->peer_device_id, 0));
            printf("GPU %d enabled peer access to GPU %d\n", dev, gpu_transformer->peer_device_id);
            // Create events for this GPU to signal when it sends data
            HIP_CHECK(hipEventCreateWithFlags(&gpu_transformer->evPeerDataSent, hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&gpu_transformer->evPeerResultsSent, hipEventDisableTiming));
            // Keep old events for compatibility (will be removed later)
            HIP_CHECK(hipEventCreateWithFlags(&gpu_transformer->evPeerDataReceived, hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&gpu_transformer->evPeerResultsReceived, hipEventDisableTiming));
        }
        else
        {
            gpu_transformer->peer_device_id = -1;
        }

        gpu_transformer->config = transformer->config;
        build_gpu_transformer(gpu_transformer, transformer, dev);

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
    if (w->w_mlp1_mxfp4)
        HIP_CHECK(hipFree(w->w_mlp1_mxfp4));
    if (w->w_mlp1_scales)
        HIP_CHECK(hipFree(w->w_mlp1_scales));
    if (w->b_mlp1)
        HIP_CHECK(hipFree(w->b_mlp1));
    if (w->w_mlp2_mxfp4)
        HIP_CHECK(hipFree(w->w_mlp2_mxfp4));
    if (w->w_mlp2_scales)
        HIP_CHECK(hipFree(w->w_mlp2_scales));
    if (w->b_mlp2)
        HIP_CHECK(hipFree(w->b_mlp2));
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

    if (s->d_total_tokens)
        HIP_CHECK(hipFree(s->d_total_tokens));

    // Free additional buffers
    if (s->d_peer_expert_offsets)
        HIP_CHECK(hipFree(s->d_peer_expert_offsets));
    if (s->d_local_write_idx)
        HIP_CHECK(hipFree(s->d_local_write_idx));
    if (s->d_peer_write_idx)
        HIP_CHECK(hipFree(s->d_peer_write_idx));
    if (s->d_received_expert_counts)
        HIP_CHECK(hipFree(s->d_received_expert_counts));
    if (s->peer_indices)
        HIP_CHECK(hipFree(s->peer_indices));
    if (s->peer_weights)
        HIP_CHECK(hipFree(s->peer_weights));
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
    
    // Destroy peer communication events if they were created
    if (gpu_t->peer_device_id != -1)
    {
        hipEventDestroy(gpu_t->evPeerDataSent);
        hipEventDestroy(gpu_t->evPeerResultsSent);
        hipEventDestroy(gpu_t->evPeerDataReceived);
        hipEventDestroy(gpu_t->evPeerResultsReceived);
    }
}

void finish(Transformer *transformer, Tokenizer *tokenizer)
{
// Free GPU transformer
#pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev)
    {
        free_gpu_transformer(gpu_transformers[dev]);
        free(gpu_transformers[dev]);
    }
    // Don't free gpu_transformers itself as it's a static array
}

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
    static Timer fused_attention_kernel_timer("FusedAttentionKernel_attention", true);
    static Timer fused_output_projection_timer("FusedOutputProjection_attention", true);

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;

    // RMSNorm - FIXED: Use GPU weight pointer
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    {
        rmsnorm_kernel<<<norm_grid, norm_block>>>(
            s->t, s->x, w->rms_attn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    // HIP_CHECK(hipDeviceSynchronize());

    dim3 matmul_grid(batch_size, ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + 31) / 32);
    dim3 matmul_block(32, min(32, THREADS_PER_BLOCK / 32));
    int qkv_weight_offset = layer_idx * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    // Define block and grid dimensions
    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + block_dim.y - 1) / block_dim.y; // Ceiling division
    {
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
    // HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipGetLastError());
    // Add bias - FIXED: Use GPU bias pointer and proper kernel
    int qkv_bias_offset = layer_idx * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    dim3 bias_grid((1LL * batch_size * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    {
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            s->qkv, w->b_qkv + qkv_bias_offset, batch_size, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }
    // HIP_CHECK(hipDeviceSynchronize());

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
    // HIP_CHECK(hipDeviceSynchronize());

    // Copy K: shape [batch_size, n_kv_heads * head_dim]
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL * b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + k_offset;
        float *dst = s->k + 1LL * b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(float), hipMemcpyDeviceToDevice));
    }
    // HIP_CHECK(hipDeviceSynchronize());

    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = s->qkv + 1LL * b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + v_offset;
        float *dst = s->v + 1LL * b * v_size;
        HIP_CHECK(hipMemcpy(dst, src, v_size * sizeof(float), hipMemcpyDeviceToDevice));
    }
    // HIP_CHECK(hipDeviceSynchronize());

    // Apply rotary embeddings
    dim3 rope_grid(batch_size, p->n_attn_heads);
    dim3 rope_block(head_dim / 2);
    {
        apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
            s->q, s->cos_vals, s->sin_vals, s->positions, batch_size, p->n_attn_heads, head_dim);
        HIP_CHECK(hipGetLastError());
    }
    // HIP_CHECK(hipDeviceSynchronize());

    rope_grid.y = p->n_kv_heads;
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        s->k, s->cos_vals, s->sin_vals, s->positions, batch_size, p->n_kv_heads, head_dim);
    HIP_CHECK(hipGetLastError());
    // HIP_CHECK(hipDeviceSynchronize());

    // Update KV cache - NEW: Proper GPU kernel
    dim3 kv_grid(batch_size, (kv_dim + 31) / 32);
    dim3 kv_block(1, 32);
    {
        update_kv_cache_kernel<<<kv_grid, kv_block>>>(
            s->key_cache, s->value_cache, s->k, s->v, s->positions, batch_size,
            p->n_layers, layer_idx, MAX_SEQ_LEN, kv_dim);
        HIP_CHECK(hipGetLastError());
    }

    // ------------------- FUSED KERNEL LAUNCH (REPLACES 4 OLD KERNELS) -------------------
    {
        TIME_SCOPE(fused_attention_kernel_timer);

        dim3 grid(batch_size, p->n_attn_heads);
        dim3 block(256); // 256 threads is a good default for this type of workload

        // Calculate required dynamic shared memory: Q vector + Att scores + reduction buffer
        size_t shared_mem_size = (p->head_dim + MAX_SEQ_LEN + block.x) * sizeof(float);

        fused_attention_kernel<<<grid, block, shared_mem_size>>>(
            s->tb, // Output goes to tb, matching the original weighted_sum output
            s->q,
            s->key_cache,
            s->value_cache,
            w->attn_sinks + layer_idx * p->n_attn_heads, // Offset to the current layer's sinks
            s->mask,
            s->positions,
            batch_size,
            p->n_attn_heads,
            p->n_kv_heads,
            p->head_dim,
            MAX_SEQ_LEN,
            p->n_layers,
            layer_idx,
            p->sliding_window > 0);
        HIP_CHECK(hipGetLastError());
    }
    // --------------------------------- END OF FUSED SECTION ---------------------------------

    // --- FUSED OUTPUT PROJECTION: Replaces matmul, add_bias, and residual add ---
    int attn_out_offset = layer_idx * (head_dim * p->n_attn_heads) * hidden_dim;
    int attn_bias_offset = layer_idx * hidden_dim;

    {
        TIME_SCOPE(fused_output_projection_timer);
        int M = batch_size;
        int N = hidden_dim;
        int K = head_dim * p->n_attn_heads;

        dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 blockDim(LANE_PER_WAVE, WAVES_PER_BLOCK);

        // Shared memory: 2 buffers for A [M,K] tiles, 2 for B [K,N] tiles
        size_t shared_mem_bytes = (2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);

        fused_output_projection_kernel_optimized<<<gridDim, blockDim, shared_mem_bytes>>>(
            s->x,                      // Residual input and final output
            s->tb,                     // Input from attention weighted sum
            w->w_o + attn_out_offset,  // Projection weights
            w->b_o + attn_bias_offset, // Projection bias
            M,                         // M
            K,                         // K
            N                          // N
        );
        HIP_CHECK(hipGetLastError());
    }
    // --- END OF FUSED OUTPUT PROJECTION ---
}

// ============================================================================
// ======================== EXPERT PARALLELISM KERNELS ========================
// ============================================================================

__global__ void classify_and_count_tokens_kernel(
    const int *__restrict__ topk_i, // Input: Top-k expert indices [batch_size * experts_per_token]
    int *d_local_expert_counts,     // Output: Counts for each local expert [n_local_experts]
    int *d_peer_expert_counts,      // Output: Counts for each peer expert [n_local_experts]
    int batch_size,
    int experts_per_token,
    int n_local_experts,
    int expert_offset // 0 for GPU 0, n_local_experts for GPU 1
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_token_experts = batch_size * experts_per_token;

    if (idx < total_token_experts)
    {
        int expert_id = topk_i[idx];

        // Check if the expert is handled by this GPU
        if (expert_id >= expert_offset && expert_id < expert_offset + n_local_experts)
        {
            // It's a local expert
            int local_expert_idx = expert_id - expert_offset;
            atomicAdd(&d_local_expert_counts[local_expert_idx], 1);
        }
        else
        {
            // It's a peer expert
            int peer_expert_offset = (expert_offset == 0) ? n_local_experts : 0;
            int peer_expert_idx = expert_id - peer_expert_offset;
            atomicAdd(&d_peer_expert_counts[peer_expert_idx], 1);
        }
    }
}

__global__ void permute_local_and_peer_inputs_kernel(
    const float *__restrict__ t,                    // Input: Hidden states [batch_size, hidden_dim]
    const int *__restrict__ topk_i,                 // Input: Top-k expert indices [batch_size, experts_per_token]
    const float *__restrict__ topk_v,               // Input: Top-k expert weights [batch_size, experts_per_token]
    const int *__restrict__ d_local_expert_offsets, // Input: Offsets for local experts
    const int *__restrict__ d_peer_expert_offsets,  // Input: Offsets for peer experts
    int *d_local_write_idx,                         // Atomic Counter: Write index for local experts
    int *d_peer_write_idx,                          // Atomic Counter: Write index for peer experts
    int batch_size,
    int hidden_dim,
    int experts_per_token,
    int n_local_experts,
    int expert_offset,
    // Outputs
    float *expert_input_buffer, // Output for local processing
    int *expert_indices,        // Original token index for local scatter
    float *expert_weights,      // Weight for local scatter
    float *peer_input_buffer,   // Output buffer to send to peer
    int *peer_indices,          // Original token index for peer scatter
    float *peer_weights         // Weight for peer scatter
)
{
    int b = blockIdx.x; // Each block processes one token from the batch

    if (b >= batch_size)
        return;

    const float *input_token_h = t + (size_t)b * hidden_dim;

    for (int k = 0; k < experts_per_token; ++k)
    {
        int token_expert_idx = b * experts_per_token + k;
        int expert_id = topk_i[token_expert_idx];
        float weight = topk_v[token_expert_idx];

        if (expert_id >= expert_offset && expert_id < expert_offset + n_local_experts)
        {
            // --- Handle LOCAL expert ---
            int local_expert_idx = expert_id - expert_offset;
            int offset = d_local_expert_offsets[local_expert_idx];
            int write_idx = atomicAdd(&d_local_write_idx[local_expert_idx], 1);
            int dest_idx = offset + write_idx;

            // Copy hidden state
            float *dest_ptr = expert_input_buffer + (size_t)dest_idx * hidden_dim;
            for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
            {
                dest_ptr[i] = input_token_h[i];
            }

            // Store metadata needed for scattering back later
            expert_indices[dest_idx] = b;
            expert_weights[dest_idx] = weight;
        }
        else
        {
            // --- Handle PEER expert ---
            int peer_expert_offset = (expert_offset == 0) ? n_local_experts : 0;
            int peer_expert_idx = expert_id - peer_expert_offset;
            int offset = d_peer_expert_offsets[peer_expert_idx];
            int write_idx = atomicAdd(&d_peer_write_idx[peer_expert_idx], 1);
            int dest_idx = offset + write_idx;

            // Copy hidden state
            float *dest_ptr = peer_input_buffer + (size_t)dest_idx * hidden_dim;
            for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
            {
                dest_ptr[i] = input_token_h[i];
            }

            // Store metadata needed for scattering back later
            peer_indices[dest_idx] = b;
            peer_weights[dest_idx] = weight;
        }
    }
}

// NOTE: The original scatter_expert_outputs_kernel can be reused as-is.
// Its logic of using an index and weight to atomically add to the output buffer
// is exactly what we need for both local and peer results.

void moe_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size)
{
    // Fallback to single GPU logic if not in parallel mode
    if (gpu_t->peer_device_id == -1)
    {
        // ... (Original single-GPU moe_gpu logic should be here) ...
        return;
    }

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
    GPUTransformer *peer_gpu_t = gpu_transformers[gpu_t->peer_device_id];

    const int hidden_dim = p->hidden_dim;
    const int intermediate_dim = p->intermediate_dim;
    const int n_experts = p->n_experts;
    const int n_local_experts = n_experts / 2;
    const int experts_per_token = p->experts_per_token;
    const int gpu_id = omp_get_thread_num();
    const int expert_offset = (gpu_id % 2 == 0) ? 0 : n_local_experts;

    hipEvent_t evRouterDone, evDispatchDone;
    hipEvent_t evLocalExpertsDone, evPeerProcessingDone;
    HIP_CHECK(hipEventCreateWithFlags(&evRouterDone, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&evDispatchDone, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&evLocalExpertsDone, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&evPeerProcessingDone, hipEventDisableTiming));

    // ===== 1) FFN RMSNorm + Router on sGather =====
    {
        rmsnorm_kernel<<<batch_size, THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(s->t, s->x, w->rms_ffn_w + (size_t)layer_idx * hidden_dim, batch_size, hidden_dim);
        matmul(s->router_score, s->t, w->w_router + (size_t)layer_idx * hidden_dim * n_experts, batch_size, hidden_dim, n_experts, cpu_buf->sGather);
        add_bias_kernel<<<(batch_size * n_experts + 255) / 256, 256, 0, cpu_buf->sGather>>>(s->router_score, w->b_router + (size_t)layer_idx * n_experts, batch_size, n_experts);
        topk_kernel<<<batch_size, 1, 0, cpu_buf->sGather>>>(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, experts_per_token);
        softmax_kernel<<<batch_size, THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(s->topk_v, batch_size, experts_per_token);
        HIP_CHECK(hipEventRecord(evRouterDone, cpu_buf->sGather));
    }

    // ===== 2) Dispatch: Classify, Count, Permute (sGather) =====
    // No need to wait on evRouterDone since we're on the same stream
    HIP_CHECK(hipMemsetAsync(s->d_local_expert_counts, 0, n_local_experts * sizeof(int), cpu_buf->sGather));
    HIP_CHECK(hipMemsetAsync(s->d_peer_expert_counts, 0, n_local_experts * sizeof(int), cpu_buf->sGather));

    classify_and_count_tokens_kernel<<<(batch_size * experts_per_token + 255) / 256, 256, 0, cpu_buf->sGather>>>(s->topk_i, s->d_local_expert_counts, s->d_peer_expert_counts, batch_size, experts_per_token, n_local_experts, expert_offset);

    HIP_CHECK(hipMemcpyAsync(cpu_buf->local_expert_counts, s->d_local_expert_counts, n_local_experts * sizeof(int), hipMemcpyDeviceToHost, cpu_buf->sGather));
    HIP_CHECK(hipMemcpyAsync(cpu_buf->peer_expert_counts, s->d_peer_expert_counts, n_local_experts * sizeof(int), hipMemcpyDeviceToHost, cpu_buf->sGather));
    HIP_CHECK(hipStreamSynchronize(cpu_buf->sGather));

    int total_local_tokens = 0, total_peer_tokens = 0;
    for (int i = 0; i < n_local_experts; ++i)
    {
        cpu_buf->local_expert_offsets[i] = total_local_tokens;
        total_local_tokens += cpu_buf->local_expert_counts[i];
    }
    for (int i = 0; i < n_local_experts; ++i)
    {
        cpu_buf->peer_expert_offsets[i] = total_peer_tokens;
        total_peer_tokens += cpu_buf->peer_expert_counts[i];
    }

    HIP_CHECK(hipMemcpyAsync(s->d_local_expert_offsets, cpu_buf->local_expert_offsets, n_local_experts * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));
    HIP_CHECK(hipMemcpyAsync(s->d_peer_expert_offsets, cpu_buf->peer_expert_offsets, n_local_experts * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));

    HIP_CHECK(hipMemsetAsync(s->d_local_write_idx, 0, n_local_experts * sizeof(int), cpu_buf->sGather));
    HIP_CHECK(hipMemsetAsync(s->d_peer_write_idx, 0, n_local_experts * sizeof(int), cpu_buf->sGather));

    permute_local_and_peer_inputs_kernel<<<batch_size, 256, 0, cpu_buf->sGather>>>(s->t, s->topk_i, s->topk_v, s->d_local_expert_offsets, s->d_peer_expert_offsets, s->d_local_write_idx, s->d_peer_write_idx, batch_size, hidden_dim, experts_per_token, n_local_experts, expert_offset, s->expert_input_buffer, s->expert_indices, s->expert_weights, s->peer_input_buffer, s->peer_indices, s->peer_weights);
    HIP_CHECK(hipEventRecord(evDispatchDone, cpu_buf->sGather));

    // ===== 3) P2P Communication: Send to Peer (sComm) =====
    HIP_CHECK(hipStreamWaitEvent(cpu_buf->sComm, evDispatchDone, 0));
    if (total_peer_tokens > 0)
    {
        HIP_CHECK(hipMemcpyPeerAsync(peer_gpu_t->state.d_received_expert_counts, peer_gpu_t->peer_device_id, s->d_peer_expert_counts, gpu_id, n_local_experts * sizeof(int), cpu_buf->sComm));
        HIP_CHECK(hipMemcpyPeerAsync(peer_gpu_t->state.peer_input_buffer, peer_gpu_t->peer_device_id, s->peer_input_buffer, gpu_id, (size_t)total_peer_tokens * hidden_dim * sizeof(float), cpu_buf->sComm));
        // Record on THIS GPU's event, not the peer's
        HIP_CHECK(hipEventRecord(gpu_t->evPeerDataSent, cpu_buf->sComm));
    }

    // ===== 4) Expert MLPs on multiple streams =====

    // --- VÒNG LẶP 1: Xử lý các token của chính GPU này (LOCAL TOKENS) ---
    for (int expert_idx_local = 0; expert_idx_local < n_local_experts; ++expert_idx_local)
    {
        const int h_batch_count = cpu_buf->local_expert_counts[expert_idx_local];
        if (h_batch_count == 0)
            continue;

        hipStream_t st = cpu_buf->sMLP[expert_idx_local % N_MLP_STREAMS];
        HIP_CHECK(hipStreamWaitEvent(st, evDispatchDone, 0)); // Đợi permute local hoàn tất

        const int expert_tok_off = cpu_buf->local_expert_offsets[expert_idx_local];
        const int expert_id = expert_idx_local + expert_offset;

        // Con trỏ đến các buffer cho batch của expert này
        float *expert_input_ptr = s->expert_input_buffer + (size_t)expert_tok_off * hidden_dim;
        float *mlp1_out_ptr = s->mlp1_out + (size_t)expert_tok_off * (2 * intermediate_dim);
        float *gate_ptr = s->gate + (size_t)expert_tok_off * intermediate_dim;
        float *up_ptr = s->up + (size_t)expert_tok_off * intermediate_dim;
        float *gate_up_ptr = s->gate_up + (size_t)expert_tok_off * intermediate_dim;
        float *expert_output_ptr = s->expert_output_buffer + (size_t)expert_tok_off * hidden_dim;

        // MLP1 (Gate/Up) — MXFP4
        {
            const size_t w_off1 = ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)(2 * intermediate_dim) * hidden_dim;
            const size_t w_packed_off1 = w_off1 / 2;
            const size_t w_scale_off1 = w_off1 / MXFP4_BLOCK_SIZE;
            const size_t total_elems1 = (size_t)(2 * intermediate_dim) * hidden_dim;

            matmul_mxfp4(mlp1_out_ptr, expert_input_ptr,
                         w->w_mlp1_mxfp4 + w_packed_off1,
                         w->w_mlp1_scales + w_scale_off1,
                         h_batch_count, hidden_dim, 2 * intermediate_dim,
                         total_elems1, st);
        }
        // split + swiglu
        {
            split_gate_up_kernel<<<(h_batch_count * intermediate_dim + 255) / 256, 256, 0, st>>>(
                gate_ptr, up_ptr, mlp1_out_ptr,
                w->b_mlp1 + ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)(2 * intermediate_dim),
                h_batch_count, intermediate_dim);

            swiglu_kernel<<<(h_batch_count * intermediate_dim + 255) / 256, 256, 0, st>>>(
                gate_ptr, up_ptr, gate_up_ptr,
                h_batch_count, intermediate_dim, p->swiglu_limit);
        }
        // MLP2 (Down) — MXFP4
        {
            const size_t w_off2 = ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)hidden_dim * intermediate_dim;
            const size_t w_packed_off2 = w_off2 / 2;
            const size_t w_scale_off2 = w_off2 / MXFP4_BLOCK_SIZE;
            const size_t total_elems2 = (size_t)hidden_dim * intermediate_dim;

            matmul_mxfp4(expert_output_ptr, gate_up_ptr,
                         w->w_mlp2_mxfp4 + w_packed_off2,
                         w->w_mlp2_scales + w_scale_off2,
                         h_batch_count, intermediate_dim, hidden_dim,
                         total_elems2, st);
        }
        // bias
        {
            add_bias_kernel<<<(h_batch_count * hidden_dim + 255) / 256, 256, 0, st>>>(
                expert_output_ptr,
                w->b_mlp2 + ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)hidden_dim,
                h_batch_count, hidden_dim);
        }
    }
    // Ghi nhận sự kiện khi tất cả các stream MLP cho local tokens đã được queue
    HIP_CHECK(hipEventRecord(evLocalExpertsDone, cpu_buf->sMLP[(n_local_experts - 1) % N_MLP_STREAMS]));

    // --- Chuẩn bị xử lý token từ Peer ---
    // Only process peer tokens if there are any to receive
    if (total_peer_tokens > 0)
    {
        // Cần tính toán offsets cho dữ liệu nhận được.
        // Đầu tiên, đợi để nhận được 'counts' từ peer.
        // Wait on the peer's "data sent" event
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sGather, peer_gpu_t->evPeerDataSent, 0));
        HIP_CHECK(hipMemcpyAsync(cpu_buf->received_expert_counts, s->d_received_expert_counts, n_local_experts * sizeof(int), hipMemcpyDeviceToHost, cpu_buf->sGather));
        HIP_CHECK(hipStreamSynchronize(cpu_buf->sGather));

        int total_received_tokens = 0;
        for (int i = 0; i < n_local_experts; ++i)
        {
            cpu_buf->received_expert_offsets[i] = total_received_tokens;
            total_received_tokens += cpu_buf->received_expert_counts[i];
        }

        // --- VÒNG LẶP 2: Xử lý các token nhận từ Peer (PEER TOKENS) ---
        for (int expert_idx_local = 0; expert_idx_local < n_local_experts; ++expert_idx_local)
        {
            const int h_batch_count = cpu_buf->received_expert_counts[expert_idx_local];
            if (h_batch_count == 0)
                continue;

            hipStream_t st = cpu_buf->sMLP[expert_idx_local % N_MLP_STREAMS];
            // Đợi toàn bộ payload dữ liệu từ peer đến nơi
            // Wait on the peer's "data sent" event
            HIP_CHECK(hipStreamWaitEvent(st, peer_gpu_t->evPeerDataSent, 0));

        const int expert_tok_off = cpu_buf->received_expert_offsets[expert_idx_local];
        const int expert_id = expert_idx_local + expert_offset;

        // Con trỏ đến các buffer cho batch của expert này, LƯU Ý ĐẦU VÀO/RA
        float *expert_input_ptr = s->peer_input_buffer + (size_t)expert_tok_off * hidden_dim; // <-- ĐỌC TỪ PEER BUFFER
        float *mlp1_out_ptr = s->mlp1_out + (size_t)expert_tok_off * (2 * intermediate_dim);  // (có thể dùng lại buffer trung gian)
        float *gate_ptr = s->gate + (size_t)expert_tok_off * intermediate_dim;
        float *up_ptr = s->up + (size_t)expert_tok_off * intermediate_dim;
        float *gate_up_ptr = s->gate_up + (size_t)expert_tok_off * intermediate_dim;
        float *expert_output_ptr = s->peer_output_buffer + (size_t)expert_tok_off * hidden_dim; // <-- GHI VÀO PEER BUFFER

        // MLP1 (Gate/Up) — MXFP4 (Sử dụng trọng số của GPU này)
        {
            const size_t w_off1 = ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)(2 * intermediate_dim) * hidden_dim;
            const size_t w_packed_off1 = w_off1 / 2;
            const size_t w_scale_off1 = w_off1 / MXFP4_BLOCK_SIZE;
            const size_t total_elems1 = (size_t)(2 * intermediate_dim) * hidden_dim;
            matmul_mxfp4(mlp1_out_ptr, expert_input_ptr, w->w_mlp1_mxfp4 + w_packed_off1, w->w_mlp1_scales + w_scale_off1, h_batch_count, hidden_dim, 2 * intermediate_dim, total_elems1, st);
        }
        // split + swiglu
        {
            split_gate_up_kernel<<<(h_batch_count * intermediate_dim + 255) / 256, 256, 0, st>>>(gate_ptr, up_ptr, mlp1_out_ptr, w->b_mlp1 + ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)(2 * intermediate_dim), h_batch_count, intermediate_dim);
            swiglu_kernel<<<(h_batch_count * intermediate_dim + 255) / 256, 256, 0, st>>>(gate_ptr, up_ptr, gate_up_ptr, h_batch_count, intermediate_dim, p->swiglu_limit);
        }
        // MLP2 (Down) — MXFP4
        {
            const size_t w_off2 = ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)hidden_dim * intermediate_dim;
            const size_t w_packed_off2 = w_off2 / 2;
            const size_t w_scale_off2 = w_off2 / MXFP4_BLOCK_SIZE;
            const size_t total_elems2 = (size_t)hidden_dim * intermediate_dim;
            matmul_mxfp4(expert_output_ptr, gate_up_ptr, w->w_mlp2_mxfp4 + w_packed_off2, w->w_mlp2_scales + w_scale_off2, h_batch_count, intermediate_dim, hidden_dim, total_elems2, st);
        }
        // bias
        {
            add_bias_kernel<<<(h_batch_count * hidden_dim + 255) / 256, 256, 0, st>>>(expert_output_ptr, w->b_mlp2 + ((size_t)layer_idx * n_local_experts + expert_idx_local) * (size_t)hidden_dim, h_batch_count, hidden_dim);
        }
        }
        // Ghi nhận sự kiện khi tất cả các stream MLP cho peer tokens đã được queue
        HIP_CHECK(hipEventRecord(evPeerProcessingDone, cpu_buf->sMLP[(n_local_experts - 1) % N_MLP_STREAMS]));
    }

    // ===== 5) P2P Communication: Send Results Back (sComm) =====
    if (total_peer_tokens > 0)
    {
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sComm, evPeerProcessingDone, 0));
        
        // Calculate total_received_tokens if we have peer tokens
        int total_received_tokens = 0;
        for (int i = 0; i < n_local_experts; ++i)
        {
            total_received_tokens += cpu_buf->received_expert_counts[i];
        }
        
        if (total_received_tokens > 0)
        {
            HIP_CHECK(hipMemcpyPeerAsync(
                peer_gpu_t->state.peer_output_buffer, // Đích: buffer output của peer
                peer_gpu_t->peer_device_id,           // ID của peer device
                s->peer_output_buffer,                // Nguồn: buffer output của GPU này
                gpu_id,                               // ID của GPU nguồn
                (size_t)total_received_tokens * hidden_dim * sizeof(float),
                cpu_buf->sComm));
        }
        // Record on THIS GPU's event to signal results have been sent
        HIP_CHECK(hipEventRecord(gpu_t->evPeerResultsSent, cpu_buf->sComm));
    }
    // ===== 6) Scatter & Aggregate results (sScatter) =====
    HIP_CHECK(hipMemsetAsync(s->e_agg, 0, (size_t)batch_size * hidden_dim * sizeof(float), cpu_buf->sScatter));

    // Scatter local results
    HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, evLocalExpertsDone, 0));
    if (total_local_tokens > 0)
    {
        scatter_expert_outputs_kernel<<<(total_local_tokens * hidden_dim + 255) / 256, 256, 0, cpu_buf->sScatter>>>(s->e_agg, s->expert_output_buffer, s->expert_indices, s->expert_weights, total_local_tokens, hidden_dim);
    }

    // Scatter peer results
    // Wait on the peer's "results sent" event
    if (total_peer_tokens > 0)
    {
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, peer_gpu_t->evPeerResultsSent, 0));
        scatter_expert_outputs_kernel<<<(total_peer_tokens * hidden_dim + 255) / 256, 256, 0, cpu_buf->sScatter>>>(s->e_agg, s->peer_output_buffer, s->peer_indices, s->peer_weights, total_peer_tokens, hidden_dim);
    }

    // ===== 7) Final Residual (sScatter) =====
    accumulate_kernel<<<(batch_size * hidden_dim + 255) / 256, 256, 0, cpu_buf->sScatter>>>(s->x, s->e_agg, 1.0f, batch_size, hidden_dim);

    // Final sync and cleanup
    HIP_CHECK(hipStreamSynchronize(cpu_buf->sScatter));
    HIP_CHECK(hipEventDestroy(evRouterDone));
    HIP_CHECK(hipEventDestroy(evDispatchDone));
    HIP_CHECK(hipEventDestroy(evLocalExpertsDone));
    HIP_CHECK(hipEventDestroy(evPeerProcessingDone));
}

float *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
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
    // Copy logits back to CPU (you might want to keep this on GPU for sampling)
    HIP_CHECK(hipMemcpy(cpu_buf->logits, s->logits, batch_size * p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return cpu_buf->logits;
}

long long continuous_batching_inference(Tokenizer *tokenizer,
                                        Sampler *sampler, Requests *requests)
{
    long long total_tokens_generated = 0;
    int total_requests = requests->num_reqs;
#pragma omp parallel num_threads(num_gpus) reduction(+ : total_tokens_generated)
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
            float *logits = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, BATCH_SIZE);

            // Process each slot
            has_active_slots = false;
            for (int slot = 0; slot < BATCH_SIZE; slot++)
            {
                if (!cpu_buf->slot_active_cpu[slot])
                    continue;

                has_active_slots = true;
                int req_idx = cpu_buf->request_mapping_cpu[slot];
                int pos = cpu_buf->positions[slot];
                float *logits_slot = logits + slot * p->vocab_size;

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
                    next_token = sample(sampler, logits_slot);

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