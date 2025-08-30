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
#include <mutex>
#include <iostream>
#include <thread>

#include <mutex>
#include <iostream>
#include <thread>
#include <cstdio>

std::mutex debug_mutex;

// Variadic macro hỗ trợ format như printf
#define THREAD_DEBUG(fmt, ...) do { \
    std::lock_guard<std::mutex> lock(debug_mutex); \
    std::printf("[Thread %lu] " fmt, \
                (unsigned long)(omp_get_thread_num()), \
                ##__VA_ARGS__); \
    std::fflush(stdout); \
} while(0)

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
    uint8_t *w_mlp1_mxfp4, *w_mlp2_mxfp4; // Packed MXFP4 indices
    float *w_mlp1_scales, *w_mlp2_scales; // MXFP4 block scales
    __hip_bfloat16 *b_mlp1, *b_mlp2;      // Biases remain in bfloat16
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
    int *batch_count;            // batch count for expert processing

    // Persistent expert routing buffers (avoid hipMalloc/hipFree in forward pass)
    int *d_expert_counts;    // token counts per expert (n_experts)
    int *d_expert_offsets;   // prefix sum of expert counts (n_experts)
    int *d_expert_write_idx; // write indices for expert gathering (n_experts)
    int *d_total_tokens;     // total tokens across all experts (1 element)

    // *** NEW: Buffers for Expert Parallelism (All-to-All) ***
    int *d_global_expert_counts;    // Global token counts per expert across the peer group [n_experts]
    int *d_global_expert_offsets;   // Global offsets based on global counts [n_experts]
    int *d_send_counts_per_expert;  // Atomic counters for partitioning send buffer [1]
    int *d_recv_counts_per_expert;  // Atomic counters for merging recv buffer [1]

    // Buffers for data exchange
    float *send_buffer; // Buffer to send to peer GPU [batch_size * experts_per_token, hidden_dim]
    float *recv_buffer; // Buffer to receive from peer GPU [batch_size * experts_per_token, hidden_dim]
    int* send_buffer_indices; // Expert indices for tokens being sent [batch_size * experts_per_token]
    int* recv_buffer_indices; // Expert indices for tokens being received [batch_size * experts_per_token]

    // Final contiguous buffers for MLP computation
    float *final_expert_input_buffer;  // Contiguous input for local experts [batch_size * experts_per_token, hidden_dim]
    float *final_expert_output_buffer; // Contiguous output from local experts [batch_size * experts_per_token, hidden_dim]
    
    // Token origin tracking for output routing
    bool *token_origins; // Track whether token originated locally (0) or from peer (1) [batch_size * experts_per_token]
    float *send_weights; // Router weights for tokens being sent [batch_size * experts_per_token]
    float *recv_weights; // Router weights for received tokens [batch_size * experts_per_token]


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

    int device_id;
    int peer_device_id; // ID của GPU kết cặp trong nhóm
    GPURunState* peer_state;   // Pointer to peer's state for direct access

} GPUTransformer;

// Global variables for direct access in batched_generate_gpu
static GPUTransformer *gpu_transformers[MAX_GPUS];

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
    size_t kv_cache_size = BATCH_SIZE * p->n_layers * MAX_SEQ_LEN * kv_dim * sizeof(__hip_bfloat16);

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

    // *** NEW: Buffers for Expert Parallelism (All-to-All) ***
    HIP_CHECK(hipMalloc((void **)&s->d_global_expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_global_expert_offsets, p->n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_send_counts_per_expert, sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_recv_counts_per_expert, sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->send_buffer, BATCH_SIZE * expert_per_token * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->recv_buffer, BATCH_SIZE * expert_per_token * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->send_buffer_indices, BATCH_SIZE * expert_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->recv_buffer_indices, BATCH_SIZE * expert_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->final_expert_input_buffer, BATCH_SIZE * expert_per_token * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->final_expert_output_buffer, BATCH_SIZE * expert_per_token * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->token_origins, BATCH_SIZE * expert_per_token * sizeof(bool)));
    HIP_CHECK(hipMalloc((void **)&s->send_weights, BATCH_SIZE * expert_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->recv_weights, BATCH_SIZE * expert_per_token * sizeof(float)));

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

    // MoE weights allocation with MXFP4 quantization
    size_t mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    size_t mlp1_packed_size = (mlp1_size + 1) / 2; // 2 FP4 values per byte
    size_t mlp1_num_blocks = (mlp1_size + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_mxfp4, mlp1_packed_size));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_scales, mlp1_num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, p->n_layers * p->n_experts * (2 * p->intermediate_dim) * sizeof(__hip_bfloat16)));

    size_t mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    size_t mlp2_packed_size = (mlp2_size + 1) / 2; // 2 FP4 values per byte
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

    HIP_CHECK(hipHostMalloc(&cpu_buf->expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc(&cpu_buf->expert_offsets, p->n_experts * sizeof(int)));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sGather, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sScatter, hipStreamNonBlocking));
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipStreamCreateWithFlags(&cpu_buf->sMLP[i], hipStreamNonBlocking));
    HIP_CHECK(hipHostMalloc(&cpu_buf->logits, BATCH_SIZE * p->vocab_size * sizeof(float)));
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
    
    // ===== THIẾT LẬP GIAO TIẾP PEER-TO-PEER SAU KHI TẤT CẢ GPU ĐÃ KHỞI TẠO =====
    #pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev) {
        HIP_CHECK(hipSetDevice(dev));
        GPUTransformer* gpu_t = gpu_transformers[dev];

        // Xác định peer device trong nhóm 2 GPU
        if (dev % 2 == 0) {
            gpu_t->peer_device_id = dev + 1;
        } else {
            gpu_t->peer_device_id = dev - 1;
        }

        // Kích hoạt truy cập peer
        int can_access_peer;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, dev, gpu_t->peer_device_id));
        if (can_access_peer) {
            HIP_CHECK(hipDeviceEnablePeerAccess(gpu_t->peer_device_id, 0));
        } else {
            fprintf(stderr, "FATAL: GPU %d cannot access peer GPU %d\n", dev, gpu_t->peer_device_id);
            exit(EXIT_FAILURE);
        }
        
        // Lưu con trỏ tới state của peer để tiện truy cập
        gpu_t->peer_state = &gpu_transformers[gpu_t->peer_device_id]->state;
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
    if (s->d_expert_counts)
        HIP_CHECK(hipFree(s->d_expert_counts));
    if (s->d_expert_offsets)
        HIP_CHECK(hipFree(s->d_expert_offsets));
    if (s->d_expert_write_idx)
        HIP_CHECK(hipFree(s->d_expert_write_idx));
    if (s->d_total_tokens)
        HIP_CHECK(hipFree(s->d_total_tokens));
    
    // Free expert parallelism buffers
    if (s->d_global_expert_counts)
        HIP_CHECK(hipFree(s->d_global_expert_counts));
    if (s->d_global_expert_offsets)
        HIP_CHECK(hipFree(s->d_global_expert_offsets));
    if (s->d_send_counts_per_expert)
        HIP_CHECK(hipFree(s->d_send_counts_per_expert));
    if (s->d_recv_counts_per_expert)
        HIP_CHECK(hipFree(s->d_recv_counts_per_expert));
    if (s->send_buffer)
        HIP_CHECK(hipFree(s->send_buffer));
    if (s->recv_buffer)
        HIP_CHECK(hipFree(s->recv_buffer));
    if (s->send_buffer_indices)
        HIP_CHECK(hipFree(s->send_buffer_indices));
    if (s->recv_buffer_indices)
        HIP_CHECK(hipFree(s->recv_buffer_indices));
    if (s->final_expert_input_buffer)
        HIP_CHECK(hipFree(s->final_expert_input_buffer));
    if (s->final_expert_output_buffer)
        HIP_CHECK(hipFree(s->final_expert_output_buffer));
    if (s->token_origins)
        HIP_CHECK(hipFree(s->token_origins));
    if (s->send_weights)
        HIP_CHECK(hipFree(s->send_weights));
    if (s->recv_weights)
        HIP_CHECK(hipFree(s->recv_weights));
    
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
    free(gpu_transformers);
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
            s->tb,                                      // Output goes to tb, matching the original weighted_sum output
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
            p->sliding_window > 0
        );
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
            s->x,                       // Residual input and final output
            s->tb,                      // Input from attention weighted sum
            w->w_o + attn_out_offset,   // Projection weights
            w->b_o + attn_bias_offset,  // Projection bias
            M,                          // M
            K,                          // K
            N                           // N
        );
        HIP_CHECK(hipGetLastError());
    }
    // --- END OF FUSED OUTPUT PROJECTION ---
}



// Fictitious kernel definitions for clarity. The implementation would require these kernels.

/*
__global__ void partition_tokens_kernel(
    float* partitioned_input, float* send_buffer, float* original_input,
    int* partitioned_indices, int* send_indices, int* original_indices,
    float* partitioned_weights, float* send_weights, float* original_weights,
    int* d_global_offsets, int* d_local_write_idx, int* d_send_write_idx,
    int total_tokens, int hidden_dim, int expert_start_idx) {
    // This kernel partitions the tokens from `original_input` after the initial gather.
    // - Tokens for local experts are placed into `partitioned_input`.
    // - Tokens for the peer's experts are placed into `send_buffer`.
    // It uses atomic counters (`d_local_write_idx`, `d_send_write_idx`) to manage
    // placement into the correct slots based on the global offset plan.
}

__global__ void merge_tokens_kernel(
    float* final_buffer, float* received_buffer,
    int* final_indices, int* received_indices,
    float* final_weights, float* received_weights,
    int* d_global_offsets, int* d_local_write_idx,
    int received_token_count, int hidden_dim) {
    // This kernel merges tokens received from the peer GPU (`received_buffer`)
    // into the main `final_buffer`, placing them according to the global offset plan.
}

__global__ void partition_outputs_kernel(
    float* final_local_output, float* send_buffer, float* expert_output_buffer,
    int* received_indices, int* original_indices_for_scatter,
    int received_token_count, int total_original_tokens, int hidden_dim, int expert_start_idx) {
    // After MLP, this kernel partitions the `expert_output_buffer`.
    // - Results for tokens that originated locally are placed in `final_local_output`.
    // - Results for tokens that came from the peer are placed in `send_buffer` to be returned.
}

__global__ void merge_outputs_kernel(
    float* final_local_output, float* received_buffer,
    int* original_indices_for_scatter, int total_original_tokens, int hidden_dim) {
    // This kernel merges the results received back from the peer into the
    // final output buffer (`final_local_output`), making it ready for the scatter operation.
}
*/

void moe_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size)
{
    THREAD_DEBUG("Entering MoE layer %d with batch size %d\n", layer_idx, batch_size);
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;

    const int hidden_dim = p->hidden_dim;
    const int intermediate_dim = p->intermediate_dim;
    const int n_experts = p->n_experts;
    const int experts_per_token = p->experts_per_token;
    
    // ===== Per-call events =====
    hipEvent_t evRouterDone, evPermuteDone;
    HIP_CHECK(hipEventCreateWithFlags(&evRouterDone, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&evPermuteDone, hipEventDisableTiming));

    hipEvent_t evExpertsDone[N_MLP_STREAMS];
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipEventCreateWithFlags(&evExpertsDone[i], hipEventDisableTiming));

    int total_tokens = 0;

    // ===== 1) FFN RMSNorm + Router on sGather =====
    // (This section remains unchanged)
    {
        dim3 norm_grid(batch_size);
        dim3 norm_block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<norm_grid, norm_block, 0, cpu_buf->sGather>>>(
            s->t, s->x, w->rms_ffn_w + (size_t)layer_idx * hidden_dim,
            batch_size, hidden_dim);
    }
    HIP_CHECK(hipGetLastError());
    {
        matmul(s->router_score, s->t,
               w->w_router + (size_t)layer_idx * hidden_dim * n_experts,
               batch_size, hidden_dim, n_experts, cpu_buf->sGather);
    }
    {
        const int elems = batch_size * n_experts;
        add_bias_kernel<<<(elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                          THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(
            s->router_score, w->b_router + (size_t)layer_idx * n_experts,
            batch_size, n_experts);
    }
    {
        topk_kernel<<<batch_size, 1, 0, cpu_buf->sGather>>>(
            s->topk_v, s->topk_i, s->router_score,
            batch_size, n_experts, experts_per_token);
    }
    {
        dim3 norm_block(THREADS_PER_BLOCK);
        softmax_kernel<<<batch_size, norm_block, 0, cpu_buf->sGather>>>(
            s->topk_v, batch_size, experts_per_token);
    }
    HIP_CHECK(hipEventRecord(evRouterDone, cpu_buf->sGather));

    THREAD_DEBUG("Router done, waiting to gather/permute\n");
    // ===== 2) GATHER on sGather: count -> prefix-sum(local) -> permute =====
    // (This section remains unchanged)
    {
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sGather, evRouterDone, 0));
        HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, n_experts * sizeof(int), cpu_buf->sGather));
        const dim3 count_grid((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<count_grid, THREADS_PER_BLOCK, 0, cpu_buf->sGather>>>(
            s->topk_i, s->d_expert_counts, batch_size, experts_per_token);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipMemcpyAsync(cpu_buf->expert_counts, s->d_expert_counts,
                                 n_experts * sizeof(int), hipMemcpyDeviceToHost, cpu_buf->sGather));
        HIP_CHECK(hipStreamSynchronize(cpu_buf->sGather));

        total_tokens = 0;
        for (int i = 0; i < n_experts; ++i) {
            cpu_buf->expert_offsets[i] = total_tokens;
            total_tokens += cpu_buf->expert_counts[i];
        }

        HIP_CHECK(hipMemcpyAsync(s->d_expert_offsets, cpu_buf->expert_offsets,
                                 n_experts * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));

        HIP_CHECK(hipMemsetAsync(s->d_expert_write_idx, 0, n_experts * sizeof(int), cpu_buf->sGather));
        const dim3 permute_grid(batch_size);
        const dim3 permute_block(256);
        int shared_mem_size = experts_per_token * sizeof(int);
        permute_expert_inputs_kernel<<<permute_grid, permute_block, shared_mem_size, cpu_buf->sGather>>>(
            s->t, s->topk_i, s->topk_v, s->d_expert_offsets, s->d_expert_write_idx,
            batch_size, hidden_dim, experts_per_token,
            s->expert_input_buffer, s->expert_indices, s->expert_weights);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(evPermuteDone, cpu_buf->sGather));
    }
    THREAD_DEBUG("Permute done, waiting to launch experts\n");
    HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, evPermuteDone, 0));
    HIP_CHECK(hipMemsetAsync(s->e_agg, 0, (size_t)batch_size * hidden_dim * sizeof(float), cpu_buf->sScatter));

    // =================================================================================
    // ===== NEW: EXPERT PARALLELISM ALL-TO-ALL COMMUNICATION =====
    // =================================================================================
    
    // --- Step 0: Global Coordination ---
    // Create a global plan for data placement across the GPU pair.
    int h_global_expert_counts[n_experts];
    int h_global_expert_offsets[n_experts];
    int global_total_tokens = 0;

    THREAD_DEBUG("Building global expert plan with peer GPU %d\n", gpu_t->peer_device_id);
    // Synchronize before accessing peer's CPU buffer
    #pragma omp barrier
    
    // Both threads access their peer's CPU buffer to build the same global plan.
    GPUTransformer* peer_gpu_t = gpu_transformers[gpu_t->peer_device_id];
    for (int i = 0; i < n_experts; ++i) {
        h_global_expert_counts[i] = cpu_buf->expert_counts[i] + peer_gpu_t->cpu_buffers.expert_counts[i];
        h_global_expert_offsets[i] = global_total_tokens;
        global_total_tokens += h_global_expert_counts[i];
    }
    
    // Copy the global plan to each GPU.
    HIP_CHECK(hipMemcpyAsync(s->d_global_expert_counts, h_global_expert_counts, n_experts * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));
    HIP_CHECK(hipMemcpyAsync(s->d_global_expert_offsets, h_global_expert_offsets, n_experts * sizeof(int), hipMemcpyHostToDevice, cpu_buf->sGather));

    // Define which experts are local to this GPU
    const int n_local_experts = n_experts / 2;
    const int expert_start_idx = (gpu_t->device_id % 2 == 0) ? 0 : n_local_experts;

    // --- Step 1: Partition Data for Exchange ---
    // A kernel partitions the gathered `expert_input_buffer` into two sets:
    // 1. Tokens for local experts -> `final_expert_input_buffer`
    // 2. Tokens for peer experts -> `send_buffer`
    int h_send_count = 0; // Number of tokens to send to the peer
    int h_local_count = 0; // Number of tokens that stay on this GPU
    for (int i = 0; i < n_experts; i++) {
        bool is_local = (i >= expert_start_idx) && (i < expert_start_idx + n_local_experts);
        if (is_local) {
            h_local_count += cpu_buf->expert_counts[i];
        } else {
            h_send_count += cpu_buf->expert_counts[i];
        }
    }
    
    // We can reuse d_expert_write_idx as atomic counters for partitioning.
    HIP_CHECK(hipMemsetAsync(s->d_send_counts_per_expert, 0, sizeof(int), cpu_buf->sGather)); // Local counter
    HIP_CHECK(hipMemsetAsync(s->d_recv_counts_per_expert, 0, sizeof(int), cpu_buf->sGather)); // Send counter
    // Initialize token origins buffer (use allocated size, not dynamic size)
    HIP_CHECK(hipMemsetAsync(s->token_origins, 0, batch_size * experts_per_token * sizeof(bool), cpu_buf->sGather));
    
    // Partition tokens for exchange
    if (total_tokens > 0) {
        dim3 partition_grid(total_tokens);
        dim3 partition_block(256);
        partition_tokens_kernel<<<partition_grid, partition_block, 0, cpu_buf->sGather>>>(
            s->final_expert_input_buffer, s->send_buffer,
            s->expert_indices, s->send_buffer_indices,
            s->expert_weights, s->send_weights,
            s->expert_input_buffer, s->expert_indices,
            s->expert_weights, s->topk_i,
            s->d_send_counts_per_expert, s->d_recv_counts_per_expert,
            total_tokens, hidden_dim, n_local_experts, expert_start_idx);
        HIP_CHECK(hipGetLastError());
    }

    // --- Step 2: Exchange Data Between GPU Pairs ---
    // Synchronize before peer-to-peer copy to ensure both GPUs have partitioned their data
    #pragma omp barrier
    
    hipEvent_t evPeerCopyDone;
    HIP_CHECK(hipEventCreateWithFlags(&evPeerCopyDone, hipEventDisableTiming));
    
    int peer_send_count = total_tokens - h_local_count; // What the peer is sending to us
    
    if (h_send_count > 0) {
        HIP_CHECK(hipMemcpyPeerAsync(
            gpu_t->peer_state->recv_buffer, // Peer's recv buffer
            gpu_t->peer_device_id,
            s->send_buffer,                 // My send buffer
            gpu_t->device_id,
            (size_t)h_send_count * hidden_dim * sizeof(float),
            cpu_buf->sGather
        ));
    }
    HIP_CHECK(hipEventRecord(evPeerCopyDone, cpu_buf->sGather));

    THREAD_DEBUG("Data exchange initiated: sending %d tokens, expecting %d tokens from peer GPU %d\n",
                 h_send_count, peer_send_count, gpu_t->peer_device_id);
    // --- Step 3: Merge Received Data & Process with Local Experts ---
    // The MLP streams must wait for both the local data to be ready (evPermuteDone)
    // and for the peer data to arrive (evPeerCopyDone).
    for (int i = 0; i < N_MLP_STREAMS; ++i) {
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sMLP[i], evPermuteDone, 0));
        if (peer_send_count > 0) {
             HIP_CHECK(hipStreamWaitEvent(cpu_buf->sMLP[i], evPeerCopyDone, 0));
        }
    }
    
    // Merge received tokens with local tokens
    if (peer_send_count > 0) {
        // Mark received tokens as originating from peer
        HIP_CHECK(hipMemsetAsync((bool*)s->token_origins + h_local_count, 1, 
                                 peer_send_count * sizeof(bool), cpu_buf->sGather));
        
        dim3 merge_grid((h_local_count + peer_send_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 merge_block(THREADS_PER_BLOCK);
        merge_tokens_kernel<<<merge_grid, merge_block, 0, cpu_buf->sGather>>>(
            s->final_expert_input_buffer, s->expert_indices, s->expert_weights,
            s->final_expert_input_buffer, s->expert_indices, s->expert_weights,
            s->recv_buffer, s->recv_buffer_indices, s->recv_weights,
            s->d_global_expert_offsets, s->d_expert_write_idx,
            h_local_count, peer_send_count, hidden_dim, n_experts);
        HIP_CHECK(hipGetLastError());
    }
    
    // --- Step 4: Process with Local Experts ---
    // Each GPU now processes only its assigned half of the experts.
    for (int i = 0; i < n_local_experts; ++i) {
        int expert_id = expert_start_idx + i;
        const int h_batch_count = h_global_expert_counts[expert_id];
        if (h_batch_count == 0) continue;

        hipStream_t st = cpu_buf->sMLP[i % N_MLP_STREAMS]; // Use local index for stream
        const int expert_tok_off = h_global_expert_offsets[expert_id];

        // Pointers now reference the final, contiguous buffers
        float *expert_input_ptr = s->final_expert_input_buffer + (size_t)expert_tok_off * hidden_dim;
        float *mlp1_out_ptr = s->mlp1_out + (size_t)expert_tok_off * (2 * intermediate_dim);
        float *gate_ptr = s->gate + (size_t)expert_tok_off * intermediate_dim;
        float *up_ptr = s->up + (size_t)expert_tok_off * intermediate_dim;
        float *gate_up_ptr = s->gate_up + (size_t)expert_tok_off * intermediate_dim;
        float *expert_output_ptr = s->final_expert_output_buffer + (size_t)expert_tok_off * hidden_dim;

        // MLP1 (Gate/Up)
        {
            // Weight offset uses local expert index `i`
            const size_t w_off1 = ((size_t)layer_idx * n_local_experts + i) * (size_t)(2 * intermediate_dim) * hidden_dim;
            const size_t w_packed_off1 = w_off1 / 2;
            const size_t w_scale_off1 = w_off1 / MXFP4_BLOCK_SIZE;
            const size_t total_elems1 = (size_t)(2 * intermediate_dim) * hidden_dim;
            matmul_mxfp4(mlp1_out_ptr, expert_input_ptr, w->w_mlp1_mxfp4 + w_packed_off1, w->w_mlp1_scales + w_scale_off1, h_batch_count, hidden_dim, 2 * intermediate_dim, total_elems1, st);
        }
        // split + swiglu
        {
            const int elems = h_batch_count * intermediate_dim;
            const dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            split_gate_up_kernel<<<grid, THREADS_PER_BLOCK, 0, st>>>(gate_ptr, up_ptr, mlp1_out_ptr, w->b_mlp1 + ((size_t)layer_idx * n_local_experts + i) * (size_t)(2 * intermediate_dim), h_batch_count, intermediate_dim);
        }
        {
            const int elems = h_batch_count * intermediate_dim;
            const dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            swiglu_kernel<<<grid, THREADS_PER_BLOCK, 0, st>>>(gate_ptr, up_ptr, gate_up_ptr, h_batch_count, intermediate_dim, p->swiglu_limit);
        }
        // MLP2 (Down)
        {
            const size_t w_off2 = ((size_t)layer_idx * n_local_experts + i) * (size_t)hidden_dim * intermediate_dim;
            const size_t w_packed_off2 = w_off2 / 2;
            const size_t w_scale_off2 = w_off2 / MXFP4_BLOCK_SIZE;
            const size_t total_elems2 = (size_t)hidden_dim * intermediate_dim;
            matmul_mxfp4(expert_output_ptr, gate_up_ptr, w->w_mlp2_mxfp4 + w_packed_off2, w->w_mlp2_scales + w_scale_off2, h_batch_count, intermediate_dim, hidden_dim, total_elems2, st);
        }
        // bias
        {
            const int elems = h_batch_count * hidden_dim;
            const dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            add_bias_kernel<<<grid, THREADS_PER_BLOCK, 0, st>>>(expert_output_ptr, w->b_mlp2 + ((size_t)layer_idx * n_local_experts + i) * (size_t)hidden_dim, h_batch_count, hidden_dim);
            HIP_CHECK(hipGetLastError());
        }
    }

    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipEventRecord(evExpertsDone[i], cpu_buf->sMLP[i]));

    THREAD_DEBUG("Experts done on local GPU %d, waiting for all experts to finish\n", gpu_t->device_id);
    // --- Step 5: Exchange Results Back to Original GPUs ---
    // Wait for all experts to finish processing
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, evExpertsDone[i], 0));
    
    // Reset atomic counters for output partitioning
    HIP_CHECK(hipMemsetAsync(s->d_send_counts_per_expert, 0, sizeof(int), cpu_buf->sScatter));
    HIP_CHECK(hipMemsetAsync(s->d_recv_counts_per_expert, 0, sizeof(int), cpu_buf->sScatter));
    
    // Partition outputs back to their original GPUs
    int total_processed = h_local_count + peer_send_count;
    if (total_processed > 0) {
        dim3 partition_out_grid(total_processed);
        dim3 partition_out_block(256);
        partition_outputs_kernel<<<partition_out_grid, partition_out_block, 0, cpu_buf->sScatter>>>(
            s->expert_output_buffer, s->send_buffer,
            s->final_expert_output_buffer, s->token_origins,
            s->expert_indices, s->expert_indices, s->send_buffer_indices,
            s->d_send_counts_per_expert, s->d_recv_counts_per_expert,
            total_processed, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    THREAD_DEBUG("Output partition done, initiating output exchange\n");
    // Exchange outputs back to originating GPUs
    // Synchronize before output exchange to ensure both GPUs have finished processing
    #pragma omp barrier
    
    hipEvent_t evOutputExchangeDone;
    HIP_CHECK(hipEventCreateWithFlags(&evOutputExchangeDone, hipEventDisableTiming));
    
    if (peer_send_count > 0) {
        HIP_CHECK(hipMemcpyPeerAsync(
            gpu_t->peer_state->recv_buffer,
            gpu_t->peer_device_id,
            s->send_buffer,
            gpu_t->device_id,
            (size_t)peer_send_count * hidden_dim * sizeof(float),
            cpu_buf->sScatter
        ));
    }
    HIP_CHECK(hipEventRecord(evOutputExchangeDone, cpu_buf->sScatter));
    THREAD_DEBUG("Output exchange initiated: sending back %d tokens to peer GPU %d\n",
                 peer_send_count, gpu_t->peer_device_id);
    // Wait for output exchange to complete
    HIP_CHECK(hipStreamWaitEvent(cpu_buf->sScatter, evOutputExchangeDone, 0));
    
    // Merge received outputs with local outputs
    if (h_send_count > 0) {
        dim3 merge_out_grid((h_local_count + h_send_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 merge_out_block(THREADS_PER_BLOCK);
        merge_outputs_kernel<<<merge_out_grid, merge_out_block, 0, cpu_buf->sScatter>>>(
            s->expert_output_buffer, s->expert_indices,
            s->expert_output_buffer, s->recv_buffer,
            s->expert_indices, s->recv_buffer_indices,
            h_local_count, h_send_count, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    
    HIP_CHECK(hipEventDestroy(evOutputExchangeDone));
    
    // --- Step 6: Final Scatter and Residual Connection ---

    if (total_tokens > 0) {
        const int elems = total_tokens * hidden_dim;
        const dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        scatter_expert_outputs_kernel<<<grid, THREADS_PER_BLOCK, 0, cpu_buf->sScatter>>>(
            s->e_agg, s->expert_output_buffer, s->expert_indices, s->expert_weights,
            total_tokens, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    {
        const int elems = batch_size * hidden_dim;
        accumulate_kernel<<<(elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK, 0, cpu_buf->sScatter>>>(
            s->x, s->e_agg, 1.0f, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(cpu_buf->sScatter));

    // Destroy events
    HIP_CHECK(hipEventDestroy(evRouterDone));
    HIP_CHECK(hipEventDestroy(evPermuteDone));
    HIP_CHECK(hipEventDestroy(evPeerCopyDone));
    for (int i = 0; i < N_MLP_STREAMS; ++i)
        HIP_CHECK(hipEventDestroy(evExpertsDone[i]));
    
    // Final synchronization to ensure both GPUs in the pair complete MoE before continuing
    #pragma omp barrier
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