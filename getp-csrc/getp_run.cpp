// TODO: Modify this file to optimize end-to-end throughput with HIP GPU acceleration

#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include <cstdio>
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
#include "nccl.hpp"   // <— NCCL/RCCL-free TP collectives/helpers

#ifndef GETP_RUN
#define GETP_RUN

int num_gpus = 1;

// GPU Transformer Weights struct - stores all model weights on GPU in bfloat16 format
typedef struct
{
    // Embedding weights
    float *token_embedding_table; // (vocab_size, hidden_dim)

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

    // MoE BF16 weights
    __hip_bfloat16 *w_mlp1;        // (n_layers, n_experts, 2*D, H) row-major [O=2D, I=H]
    __hip_bfloat16 *w_mlp2;        // (n_layers, n_experts, H,   D) row-major [O=H,  I=D]
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

    // KV cache
    float *key_cache;   // (batch_size, n_layers, seq_len, kv_dim)
    float *value_cache; // (batch_size, n_layers, seq_len, kv_dim)

    // RoPE buffers
    float *cos_vals; // (head_dim/2, seq_len)
    float *sin_vals; // (head_dim/2, seq_len)

    // === legacy MoE (kept for shared kernels/utilities) ===
    float *router_score;         // (batch_size, n_experts)
    float *topk_v;               // (batch_size, K)
    int   *topk_i;               // (batch_size, K)
    float *mlp1_out;             // (batch_size*K, 2*D)  — local version not used in TP path
    float *gate;                 // (batch_size*K, D)
    float *up;                   // (batch_size*K, D)
    float *gate_up;              // (batch_size*K, D)
    float *e_agg;                // (batch_size, H)
    float *expert_input_buffer;  // (batch_size*K, H)
    float *expert_output_buffer; // (batch_size*K, H)
    int   *expert_indices;       // (batch_size*K)
    float *expert_weights;       // (batch_size*K)
    int   *batch_count;

    // Persistent routing buffers
    int *d_expert_counts;    // (n_experts)
    int *d_expert_offsets;   // (n_experts)
    int *d_expert_write_idx; // (n_experts)
    int *d_total_tokens;     // (1)
    int *d_tile2expert;
    int *d_tile2local;

    // === NEW: TP-union MoE buffers (per rank) ===
    // union batch (within a TP group) = Bgrp_max = TENSOR_PARALLEL_SIZE * BATCH_SIZE
    float *gather_x_g;            // [Bgrp_max, H]       — allgathered normalized inputs
    float *router_score_g;        // [Bgrp_max, E]
    float *topk_v_g;              // [Bgrp_max, K]
    int   *topk_i_g;              // [Bgrp_max, K]
    int   *local_ids_g;           // [Bgrp_max, K]
    float *local_wts_g;           // [Bgrp_max, K]
    float *e_agg_g;               // [Bgrp_max, H]

    // expert input/output (union)
    float *expert_input_buffer_g;   // [pairs_max, H]         pairs_max = Bgrp_max * K
    float *mlp1_out_g;              // [pairs_max, o_len_max] o_len_max = ceil(2D/TP)
    float *gate_up_g;               // [pairs_max, Dloc_max]  Dloc_max  = ceil(D/TP)
    float *expert_output_partial_g; // [pairs_max, H]         this-rank partial
    float *expert_output_gather_g;  // [TP * pairs_max, H]    workspace for emulate all-reduce

    // tile maps for union
    int cap_tiles;                // capacity for tile maps
    int *d_tile2expert_g;         // [cap_tiles]
    int *d_tile2local_g;          // [cap_tiles]

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
    int *expert_counts;
    int *expert_offsets;
    float *logits;

    // host bounce for TP copies
    void  *tp_host_stage;
    size_t tp_host_stage_bytes;

    // Continuous batching fields
    int next_request_idx;     // Index of next unprocessed request (0 to num_reqs-1)
    bool *slot_active_cpu;    // CPU mirror of slot_active for quick access
    int *seq_lengths_cpu;     // CPU mirror of seq_lengths
    int *request_mapping_cpu; // CPU mirror of request_mapping
    int *h_tile2expert;
    int *h_tile2local;
} CPUBuffers;

// Main GPU Transformer struct
typedef struct
{
    Config config;                 // model configuration
    GPUTransformerWeights weights; // GPU weights
    GPURunState state;             // GPU run state buffers
    CPUBuffers cpu_buffers;        // CPU buffers for host operations
} GPUTransformer;

// Global variables
GPUTransformer *gpu_transformers[MAX_GPUS];
TPGroup tp_groups[MAX_GPUS];   // one per device

// -------------------- helpers --------------------
#ifndef USE_ASYNC_HOST_COPIES
#define USE_ASYNC_HOST_COPIES 0  // set to 1 only after switching CPU buffers to hipHostMalloc
#endif

static inline void h2d_copy(void* dst, const void* src, size_t bytes, hipStream_t s) {
#if USE_ASYNC_HOST_COPIES
    if (bytes) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, s));
#else
    (void)s;
    if (bytes) HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#endif
}
static inline void d2h_copy(void* dst, const void* src, size_t bytes, hipStream_t s) {
#if USE_ASYNC_HOST_COPIES
    if (bytes) HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, s));
#else
    (void)s;
    if (bytes) HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#endif
}
static inline size_t max_dynamic_smem_bytes() {
    int v = 0;
    HIP_CHECK(hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    return (size_t)v;
}
static inline void assert_smem_or_die(size_t bytes, const char* kernel_name) {
    const size_t limit = max_dynamic_smem_bytes();
    if (bytes > limit) {
        fprintf(stderr, "[HIP] %s needs %zuB dynamic shared mem, but limit is %zuB. "
                        "Reduce BLOCK_* or SW_WINDOW.\n", kernel_name, bytes, limit);
        fflush(stderr);
        abort();
    }
}
// -------------------------------------------------

// Memory allocation functions
void malloc_gpu_run_state(GPURunState *s, Config *p)
{
    int kv_dim = p->head_dim * p->n_kv_heads;
    int expert_per_token = p->experts_per_token;
    const int TP = TENSOR_PARALLEL_SIZE;
    const int H  = p->hidden_dim;
    const int D  = p->intermediate_dim;
    const int E  = p->n_experts;
    const int K  = p->experts_per_token;

    // union capacities
    const int Bgrp_max   = TP * BATCH_SIZE;           // max union rows per microstep
    const int pairs_max  = Bgrp_max * K;              // worst-case tokens routed

    const int o_len_max  = (2 * D + TP - 1) / TP;     // max local columns for MLP1
    const int Dloc_max   = (D     + TP - 1) / TP;     // max local cols for MLP2

    // Initialize pointers to NULL (only new ones; others allocated below)
    s->mask = NULL;
    s->d_expert_counts = NULL;
    s->d_expert_offsets = NULL;
    s->d_expert_write_idx = NULL;
    s->d_total_tokens = NULL;

    // Check memory requirements
    size_t batch_hidden = BATCH_SIZE * H * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);

    // per-layer capacities: even layers use SW_WINDOW (if sliding enabled), odd are full
    const int n_even = (p->n_layers + 1) / 2;
    const int n_odd  = p->n_layers - n_even;
    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    const size_t layers_capacity =
        (size_t)n_even * (size_t)even_cap + (size_t)n_odd * (size_t)MAX_SEQ_LEN;

    size_t kv_cache_size = (size_t)BATCH_SIZE * layers_capacity * (size_t)kv_dim * sizeof(float);

    printf("KV cache (fp32) total capacity: layers_capacity=%zu positions/layer-stack\n",
        layers_capacity);
    printf("KV cache size per batch: %zu MB\n", kv_cache_size / (1024 * 1024));

    // Base activations
    HIP_CHECK(hipMalloc((void **)&s->x, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->t, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->tb, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->tb2, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->qkv, batch_qkv));
    HIP_CHECK(hipMalloc((void **)&s->q, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->k, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->v, BATCH_SIZE * kv_dim * sizeof(float)));

    // Routing & expert buffers (legacy/local)
    HIP_CHECK(hipMalloc((void **)&s->expert_indices, BATCH_SIZE * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->expert_weights, BATCH_SIZE * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->batch_count, sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->expert_output_buffer, BATCH_SIZE * H * sizeof(float) * expert_per_token));

    // Persistent routing buffers
    HIP_CHECK(hipMalloc((void **)&s->d_expert_counts, E * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_expert_offsets, E * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_expert_write_idx, E * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_total_tokens, sizeof(int)));

    int total_mtiles = BATCH_SIZE * K;
    HIP_CHECK(hipMalloc((void**)&s->d_tile2expert, total_mtiles * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->d_tile2local,  total_mtiles * sizeof(int)));

    // KV cache
    HIP_CHECK(hipMalloc((void **)&s->key_cache, kv_cache_size));
    HIP_CHECK(hipMalloc((void **)&s->value_cache, kv_cache_size));

    HIP_CHECK(hipMalloc((void **)&s->att, BATCH_SIZE * p->n_attn_heads * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->logits, BATCH_SIZE * p->vocab_size * sizeof(float)));

    HIP_CHECK(hipMalloc((void **)&s->local_ids, BATCH_SIZE * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->local_wts, BATCH_SIZE * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->n_local, BATCH_SIZE * sizeof(int)));

    HIP_CHECK(hipMalloc((void **)&s->router_score, BATCH_SIZE * E * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_v,       BATCH_SIZE * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_i,       BATCH_SIZE * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->mlp1_out,     (size_t)BATCH_SIZE * 2 * D * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->gate,         (size_t)BATCH_SIZE * D * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->up,           (size_t)BATCH_SIZE * D * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->gate_up,      (size_t)BATCH_SIZE * D * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->e_agg,        batch_hidden));
    HIP_CHECK(hipMalloc((void **)&s->current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->positions,      BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->cos_vals,      (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->sin_vals,      (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->expert_input_buffer, batch_hidden * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&s->temp_buffer, batch_hidden));

    // Continuous batching fields
    HIP_CHECK(hipMalloc((void **)&s->seq_lengths, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->slot_active, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMalloc((void **)&s->request_mapping, BATCH_SIZE * sizeof(int)));

    // === NEW: TP-union allocations ===
    HIP_CHECK(hipMalloc((void**)&s->gather_x_g,            (size_t)Bgrp_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->router_score_g,        (size_t)Bgrp_max * E * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->topk_v_g,              (size_t)Bgrp_max * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->topk_i_g,              (size_t)Bgrp_max * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->local_ids_g,           (size_t)Bgrp_max * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->local_wts_g,           (size_t)Bgrp_max * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->e_agg_g,               (size_t)Bgrp_max * H * sizeof(float)));

    HIP_CHECK(hipMalloc((void**)&s->expert_input_buffer_g, (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->mlp1_out_g,            (size_t)pairs_max * o_len_max * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->gate_up_g,             (size_t)pairs_max * Dloc_max * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->expert_output_partial_g, (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->expert_output_gather_g, (size_t)TP * pairs_max * H * sizeof(float)));

    s->cap_tiles = (pairs_max + BLOCK_M_MLP - 1) / BLOCK_M_MLP + E;
    HIP_CHECK(hipMalloc((void**)&s->d_tile2expert_g, s->cap_tiles * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->d_tile2local_g,  s->cap_tiles * sizeof(int)));

    // Initialize to zero
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
    HIP_CHECK(hipMemset(s->logits, 0, (size_t)BATCH_SIZE * p->vocab_size * sizeof(float)));

    // Continuous batching fields
    HIP_CHECK(hipMemset(s->seq_lengths, 0, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMemset(s->slot_active, 0, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMemset(s->request_mapping, -1, BATCH_SIZE * sizeof(int)));

    // Sliding window mask (optional)
    if (p->sliding_window > 0)
    {
        size_t mask_size = MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(float);
        HIP_CHECK(hipMalloc((void **)&s->mask, mask_size));

        float *h_mask = (float *)malloc(mask_size);
        if (!h_mask) {
            fprintf(stderr, "Failed to allocate host memory for mask\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            for (int j = 0; j < MAX_SEQ_LEN; j++) {
                h_mask[i * MAX_SEQ_LEN + j] = (i - j >= p->sliding_window) ? -INFINITY : 0.0f;
            }
        }
        HIP_CHECK(hipMemcpy(s->mask, h_mask, mask_size, hipMemcpyHostToDevice));
        free(h_mask);
    }
}

void malloc_gpu_weights(GPUTransformerWeights *w, Config *p)
{
    // Figure out my TP rank from current HIP device
    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    const int TP = TENSOR_PARALLEL_SIZE;
    const int r  = dev % TP;

    const int H  = p->hidden_dim;
    const int D  = p->intermediate_dim;
    const int E  = p->n_experts;

    // Local shard geometry (no forward-declarations needed — use small lambdas)
    auto mlp1_shard = [](int twoD, int tp, int rr, int &o_start, int &o_len) {
        const int base = twoD / tp, rem = twoD % tp;
        o_len   = base + (rr < rem ? 1 : 0);
        o_start = rr * base + (rr < rem ? rr : rem);
    };
    auto mlp2_shard_input = [](int Din, int tp, int rr, int &i_start, int &i_len) {
        const int base = Din / tp, rem = Din % tp;
        i_len   = base + (rr < rem ? 1 : 0);
        i_start = rr * base + (rr < rem ? rr : rem);
    };

    int Ostart, Oloc; mlp1_shard(2 * D, TP, r, Ostart, Oloc);  // local columns for MLP1
    int Istart, Kloc; mlp2_shard_input(D, TP, r, Istart, Kloc); // local input cols for MLP2

    // Unchanged allocs
    HIP_CHECK(hipMalloc((void **)&w->token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(float)));
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

    // === SHARDED MLP ALLOCS (compact local layouts) ===
    // MLP1 weight shard: [layers, experts, Oloc, H]
    size_t mlp1_local = (size_t)p->n_layers * E * (size_t)Oloc * H;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1, mlp1_local * sizeof(__hip_bfloat16)));
    // MLP1 bias shard: [layers, experts, Oloc]
    size_t b1_local   = (size_t)p->n_layers * E * (size_t)Oloc;
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, b1_local * sizeof(__hip_bfloat16)));

    // MLP2 weight shard: [layers, experts, H, Kloc]
    size_t mlp2_local = (size_t)p->n_layers * E * (size_t)H * Kloc;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2, mlp2_local * sizeof(__hip_bfloat16)));

    // MLP2 bias kept FULL (H), but will be scaled by 1/TP on copy: [layers, experts, H]
    size_t b2_full    = (size_t)p->n_layers * E * (size_t)H;
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2, b2_full * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->out, p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->attn_sinks, p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));
}

static inline void convert_float_array_to_bfloat16_scaled(const float *src, __hip_bfloat16 *dst, size_t n, float scale) {
    std::vector<float> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = src[i] * scale;
    convert_float_array_to_bfloat16(tmp.data(), dst, n);
}

void copy_weights_to_gpu(Transformer *transformer, GPUTransformerWeights *gpu_weights)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    const float invTP = 1.0f / (float)TENSOR_PARALLEL_SIZE;

    // Embeddings
    {
        size_t embedding_size = (size_t)p->vocab_size * p->hidden_dim;
        HIP_CHECK(hipMemcpy(gpu_weights->token_embedding_table, w->token_embedding_table,
                            embedding_size * sizeof(float), hipMemcpyHostToDevice));
    }

    // Norms
    {
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
    }

    // Attention weights
    {
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
    }

    // Attention sinks
    {
        size_t attn_sinks_size = p->n_layers * p->n_attn_heads;
        __hip_bfloat16 *h_attn_sinks_bf16 = (__hip_bfloat16 *)malloc(attn_sinks_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->attn_sinks, h_attn_sinks_bf16, attn_sinks_size);
        HIP_CHECK(hipMemcpy(gpu_weights->attn_sinks, h_attn_sinks_bf16, attn_sinks_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_attn_sinks_bf16);
    }

    // Router
    {
        size_t w_router_size = (size_t)p->n_layers * p->hidden_dim * p->n_experts;
        __hip_bfloat16 *h_w_router_bf16 = (__hip_bfloat16 *)malloc(w_router_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_router, h_w_router_bf16, w_router_size);
        HIP_CHECK(hipMemcpy(gpu_weights->w_router, h_w_router_bf16,
                            w_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_w_router_bf16);

        size_t b_router_size = (size_t)p->n_layers * p->n_experts;
        __hip_bfloat16 *h_b_router_bf16 = (__hip_bfloat16 *)malloc(b_router_size * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_router, h_b_router_bf16, b_router_size);
        HIP_CHECK(hipMemcpy(gpu_weights->b_router, h_b_router_bf16,
                            b_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(h_b_router_bf16);
    }

    // ============ MLP1/2 sharded copies with BF16 staging ============
    // Assumes: convert_float_array_to_bfloat16(...) and convert_float_array_to_bfloat16_scaled(...) exist.
    {
        const int TP = TENSOR_PARALLEL_SIZE;
        int dev = 0; HIP_CHECK(hipGetDevice(&dev));
        const int group_base    = (dev / TP) * TP;
        const int rank_in_group = dev - group_base;

        const int H = p->hidden_dim;
        const int D = p->intermediate_dim;
        const int E = p->n_experts;
        const int twoD = 2 * D;
        const float invTP = 1.0f / (float)TP;

        // MLP1 column-parallel shard (O=2D)
        int Oloc, Ostart;
        {
            const int base = twoD / TP;
            const int rem  = twoD % TP;
            Oloc   = base + (rank_in_group < rem ? 1 : 0);
            Ostart = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }
        // MLP2 row-parallel shard (input D)
        int Kloc, Istart;
        {
            const int base = D / TP;
            const int rem  = D % TP;
            Kloc   = base + (rank_in_group < rem ? 1 : 0);
            Istart = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }

        // Host pinned BF16 staging buffers (small; allocate -> use -> free)
        const size_t max_w_stage = std::max((size_t)Oloc * (size_t)H, (size_t)H * (size_t)Kloc);
        const size_t max_b_stage = std::max((size_t)Oloc, (size_t)H);

        __hip_bfloat16 *staging_w = nullptr;
        __hip_bfloat16 *staging_b = nullptr;
        HIP_CHECK(hipHostMalloc((void**)&staging_w, max_w_stage * sizeof(__hip_bfloat16)));
        HIP_CHECK(hipHostMalloc((void**)&staging_b, max_b_stage * sizeof(__hip_bfloat16)));

        // ----- convenience strides for device shards -----
        const size_t seg1_loc = (size_t)Oloc * (size_t)H;     // per-expert stride for w_mlp1 shard
        const size_t seg2_loc = (size_t)H * (size_t)Kloc;     // per-expert stride for w_mlp2 shard
        const size_t segb1    = (size_t)Oloc;                 // per-expert stride for b_mlp1 shard
        const size_t segb2    = (size_t)H;                    // per-expert stride for full b_mlp2

        // ------------------ pack & copy MLP1 weight shard ------------------
        // Source layout (CPU full): [layers, experts, 2D, H] row-major by (O, then H)
        // Dest layout (GPU shard):  [layers, experts, Oloc, H]
        {
            const size_t seg1_full = (size_t)(2 * D) * (size_t)H;  // CPU per-expert stride
            for (int L = 0; L < p->n_layers; ++L) {
                for (int e = 0; e < E; ++e) {
                    const float *base = w->w_mlp1 + (size_t)L * E * seg1_full + (size_t)e * seg1_full;

                    // Build BF16 shard row-by-row in staging_w
                    for (int o = 0; o < Oloc; ++o) {
                        const float *row_f = base + (size_t)(Ostart + o) * (size_t)H;
                        __hip_bfloat16 *row_b = staging_w + (size_t)o * (size_t)H;
                        convert_float_array_to_bfloat16(row_f, row_b, (size_t)H);
                    }
                    const size_t dev_off = ((size_t)L * (size_t)E + (size_t)e) * seg1_loc;
                    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp1 + dev_off,
                                        staging_w, seg1_loc * sizeof(__hip_bfloat16),
                                        hipMemcpyHostToDevice));
                }
            }
        }

        // ------------------ pack & copy MLP1 bias shard ------------------
        // Source (CPU full): [layers, experts, 2D]; Dest (GPU shard): [layers, experts, Oloc]
        {
            const size_t segb1_full = (size_t)(2 * D);
            for (int L = 0; L < p->n_layers; ++L) {
                for (int e = 0; e < E; ++e) {
                    const float *base = w->b_mlp1 + (size_t)L * E * segb1_full + (size_t)e * segb1_full;
                    convert_float_array_to_bfloat16(base + Ostart, staging_b, (size_t)Oloc);

                    const size_t dev_off = ((size_t)L * (size_t)E + (size_t)e) * segb1;
                    HIP_CHECK(hipMemcpy(gpu_weights->b_mlp1 + dev_off,
                                        staging_b, segb1 * sizeof(__hip_bfloat16),
                                        hipMemcpyHostToDevice));
                }
            }
        }

        // ------------------ pack & copy MLP2 weight shard ------------------
        // Source (CPU full): [layers, experts, H, D] row-major; we take columns [Istart, Istart+Kloc)
        // Dest (GPU shard):  [layers, experts, H, Kloc]
        {
            const size_t seg2_full = (size_t)H * (size_t)D;
            for (int L = 0; L < p->n_layers; ++L) {
                for (int e = 0; e < E; ++e) {
                    const float *base = w->w_mlp2 + (size_t)L * E * seg2_full + (size_t)e * seg2_full;

                    // Build BF16 shard row-by-row in staging_w
                    for (int h = 0; h < H; ++h) {
                        const float *row_f = base + (size_t)h * (size_t)D + (size_t)Istart;
                        __hip_bfloat16 *row_b = staging_w + (size_t)h * (size_t)Kloc;
                        convert_float_array_to_bfloat16(row_f, row_b, (size_t)Kloc);
                    }
                    const size_t dev_off = ((size_t)L * (size_t)E + (size_t)e) * seg2_loc;
                    HIP_CHECK(hipMemcpy(gpu_weights->w_mlp2 + dev_off,
                                        staging_w, seg2_loc * sizeof(__hip_bfloat16),
                                        hipMemcpyHostToDevice));
                }
            }
        }

        // ------------------ copy MLP2 bias (FULL H) scaled by 1/TP ------------------
        // Source (CPU full): [layers, experts, H] ; Dest (GPU full): [layers, experts, H]
        {
            for (int L = 0; L < p->n_layers; ++L) {
                for (int e = 0; e < E; ++e) {
                    const float *base = w->b_mlp2 + ((size_t)L * (size_t)E + (size_t)e) * (size_t)H;
                    convert_float_array_to_bfloat16_scaled(base, staging_b, (size_t)H, invTP);

                    const size_t dev_off = ((size_t)L * (size_t)E + (size_t)e) * (size_t)H;
                    HIP_CHECK(hipMemcpy(gpu_weights->b_mlp2 + dev_off,
                                        staging_b, (size_t)H * sizeof(__hip_bfloat16),
                                        hipMemcpyHostToDevice));
                }
            }
        }

        // Free small pinned staging buffers
        HIP_CHECK(hipHostFree(staging_w));
        HIP_CHECK(hipHostFree(staging_b));
    }

    // Output head
    size_t out_size = p->hidden_dim * p->vocab_size;
    __hip_bfloat16 *h_out_bf16 = (__hip_bfloat16 *)malloc(out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->out, h_out_bf16, out_size);
    HIP_CHECK(hipMemcpy(gpu_weights->out, h_out_bf16, out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_out_bf16);

    printf("Weights copied. w_qkv and out are transposed in-place on GPU ([K,N] layout).\n");
}

void malloc_cpu_buffers(CPUBuffers *cpu_buf, Config *p)
{
    const int TP = TENSOR_PARALLEL_SIZE;
    const int H  = p->hidden_dim;
    const int Bgrp_max = TP * BATCH_SIZE;

    // Host-only
    cpu_buf->cos_vals    = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));
    cpu_buf->sin_vals    = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));
    cpu_buf->prompt_lens = (int   *)malloc(BATCH_SIZE * sizeof(int));
    cpu_buf->finished    = (bool  *)malloc(BATCH_SIZE * sizeof(bool));

    // Prompt token storage per slot
    cpu_buf->prompt_tokens = (int **)malloc(BATCH_SIZE * sizeof(int *));
    for (int b = 0; b < BATCH_SIZE; ++b) {
        cpu_buf->prompt_tokens[b] = (int *)malloc((MAX_SEQ_LEN + 3) * sizeof(int));
    }

    // Pinned buffers
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->current_tokens,      BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->positions,           BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->slot_active_cpu,     BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->seq_lengths_cpu,     BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->request_mapping_cpu, BATCH_SIZE * sizeof(int)));

    // MoE host buffers
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->expert_counts,   p->n_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->expert_offsets,  p->n_experts * sizeof(int)));

    // Tile maps (host)
    const int total_mtiles = (int)((size_t)BATCH_SIZE * p->n_experts);
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->h_tile2expert, total_mtiles * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->h_tile2local,  total_mtiles * sizeof(int)));

    // Bounce buffer for TP peer copies (largest single block in allgather = Bgrp_max*H)
    cpu_buf->tp_host_stage_bytes = (size_t)Bgrp_max * p->experts_per_token * H * sizeof(float);
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->tp_host_stage, cpu_buf->tp_host_stage_bytes));

    // Not pinned
    cpu_buf->logits = (float *)malloc((size_t)BATCH_SIZE * p->vocab_size * sizeof(float));

    // Initialize
    std::fill_n(cpu_buf->slot_active_cpu, BATCH_SIZE, false);
    std::fill_n(cpu_buf->seq_lengths_cpu, BATCH_SIZE, 0);
    for (int i = 0; i < BATCH_SIZE; ++i) cpu_buf->request_mapping_cpu[i] = -1;
    cpu_buf->next_request_idx = 0;
}

void free_cpu_buffers(CPUBuffers *cpu_buf)
{
    if (cpu_buf->cos_vals)     free(cpu_buf->cos_vals);
    if (cpu_buf->sin_vals)     free(cpu_buf->sin_vals);
    if (cpu_buf->prompt_lens)  free(cpu_buf->prompt_lens);
    if (cpu_buf->finished)     free(cpu_buf->finished);

    if (cpu_buf->prompt_tokens) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
            if (cpu_buf->prompt_tokens[b]) free(cpu_buf->prompt_tokens[b]);
        }
        free(cpu_buf->prompt_tokens);
    }

    if (cpu_buf->logits)       free(cpu_buf->logits);

    if (cpu_buf->current_tokens)      HIP_CHECK(hipHostFree(cpu_buf->current_tokens));
    if (cpu_buf->positions)           HIP_CHECK(hipHostFree(cpu_buf->positions));
    if (cpu_buf->slot_active_cpu)     HIP_CHECK(hipHostFree(cpu_buf->slot_active_cpu));
    if (cpu_buf->seq_lengths_cpu)     HIP_CHECK(hipHostFree(cpu_buf->seq_lengths_cpu));
    if (cpu_buf->request_mapping_cpu) HIP_CHECK(hipHostFree(cpu_buf->request_mapping_cpu));

    if (cpu_buf->expert_counts)       HIP_CHECK(hipHostFree(cpu_buf->expert_counts));
    if (cpu_buf->expert_offsets)      HIP_CHECK(hipHostFree(cpu_buf->expert_offsets));
    if (cpu_buf->h_tile2expert)       HIP_CHECK(hipHostFree(cpu_buf->h_tile2expert));
    if (cpu_buf->h_tile2local)        HIP_CHECK(hipHostFree(cpu_buf->h_tile2local));

    if (cpu_buf->tp_host_stage)       HIP_CHECK(hipHostFree(cpu_buf->tp_host_stage));
}

void build_gpu_transformer(GPUTransformer *gpu_t, Transformer *cpu_t)
{
    gpu_t->config = cpu_t->config;

    malloc_gpu_weights(&gpu_t->weights, &gpu_t->config);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);

    copy_weights_to_gpu(cpu_t, &gpu_t->weights);
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus > MAX_GPUS) num_gpus = MAX_GPUS;

    #pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev) {
        HIP_CHECK(hipSetDevice(dev));
        gpu_transformers[dev] = (GPUTransformer *)malloc(sizeof(GPUTransformer));
        assert(gpu_transformers[dev] != NULL);

        tp_init(tp_groups[dev], /*world_size=*/num_gpus, /*world_rank=*/dev, /*tp_size=*/TENSOR_PARALLEL_SIZE);

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

        // Copy RoPE to GPU
        HIP_CHECK(hipMemcpy(gpu_transformer->state.cos_vals, gpu_transformer->cpu_buffers.cos_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_transformer->state.sin_vals, gpu_transformer->cpu_buffers.sin_vals,
                            (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float), hipMemcpyHostToDevice));
    }
}

void free_gpu_weights(GPUTransformerWeights *w)
{
    if (w->token_embedding_table) HIP_CHECK(hipFree(w->token_embedding_table));
    if (w->rms_attn_w) HIP_CHECK(hipFree(w->rms_attn_w));
    if (w->rms_ffn_w)  HIP_CHECK(hipFree(w->rms_ffn_w));
    if (w->rms_out_w)  HIP_CHECK(hipFree(w->rms_out_w));
    if (w->w_qkv)      HIP_CHECK(hipFree(w->w_qkv));
    if (w->b_qkv)      HIP_CHECK(hipFree(w->b_qkv));
    if (w->w_o)        HIP_CHECK(hipFree(w->w_o));
    if (w->b_o)        HIP_CHECK(hipFree(w->b_o));
    if (w->attn_sinks) HIP_CHECK(hipFree(w->attn_sinks));
    if (w->w_router)   HIP_CHECK(hipFree(w->w_router));
    if (w->b_router)   HIP_CHECK(hipFree(w->b_router));
    if (w->w_mlp1)     HIP_CHECK(hipFree(w->w_mlp1));
    if (w->w_mlp2)     HIP_CHECK(hipFree(w->w_mlp2));
    if (w->b_mlp1)     HIP_CHECK(hipFree(w->b_mlp1));
    if (w->b_mlp2)     HIP_CHECK(hipFree(w->b_mlp2));
    if (w->out)        HIP_CHECK(hipFree(w->out));
}

void free_gpu_run_state(GPURunState *s)
{
    auto df = [](void* p){ if (p) HIP_CHECK(hipFree(p)); };

    df(s->x); df(s->t); df(s->tb); df(s->tb2); df(s->temp_buffer);
    df(s->qkv); df(s->q); df(s->k); df(s->v); df(s->att); df(s->mask);
    df(s->key_cache); df(s->value_cache); df(s->cos_vals); df(s->sin_vals);

    df(s->router_score); df(s->topk_v); df(s->topk_i);
    df(s->mlp1_out); df(s->gate); df(s->up); df(s->gate_up);
    df(s->e_agg); df(s->expert_input_buffer); df(s->expert_output_buffer);
    df(s->expert_indices); df(s->expert_weights); df(s->batch_count);
    df(s->d_expert_counts); df(s->d_expert_offsets); df(s->d_expert_write_idx);
    df(s->d_total_tokens); df(s->current_tokens); df(s->positions); df(s->logits);
    df(s->seq_lengths); df(s->slot_active); df(s->request_mapping);
    df(s->d_tile2expert); df(s->d_tile2local);

    // TP-union
    df(s->gather_x_g); df(s->router_score_g); df(s->topk_v_g); df(s->topk_i_g);
    df(s->local_ids_g); df(s->local_wts_g); df(s->e_agg_g);
    df(s->expert_input_buffer_g); df(s->mlp1_out_g); df(s->gate_up_g);
    df(s->expert_output_partial_g); df(s->expert_output_gather_g);
    df(s->d_tile2expert_g); df(s->d_tile2local_g);
}

void free_gpu_transformer(GPUTransformer *gpu_t)
{
    free_gpu_weights(&gpu_t->weights);
    free_gpu_run_state(&gpu_t->state);
    free_cpu_buffers(&gpu_t->cpu_buffers);
}

void finish(Transformer *transformer, Tokenizer *tokenizer)
{
    #pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev) {
        tp_free(tp_groups[dev]);
        free_gpu_transformer(gpu_transformers[dev]);
        free(gpu_transformers[dev]);
    }
}

// ----------------------------- attention path (unchanged) --------------------------
void attention_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size,
                   int row_offset, hipStream_t sAttn)
{
    if (batch_size <= 0) return;

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;

    const int H  = p->hidden_dim;
    const int Hd = p->head_dim;
    const int NA = p->n_attn_heads;
    const int NK = p->n_kv_heads;
    const int KV = Hd * NK;
    const int QKV = Hd * (NA + 2 * NK);

    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    size_t layers_capacity = 0;
    for (int L = 0; L < p->n_layers; ++L) {
        const bool evenL = ((L & 1) == 0);
        layers_capacity += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t kv_slice = layers_capacity * (size_t)KV;

    size_t layer_pos_offset = 0;
    for (int L = 0; L < layer_idx; ++L) {
        const bool evenL = ((L & 1) == 0);
        layer_pos_offset += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t layer_elem_offset = layer_pos_offset * (size_t)KV;

    float *x_mb   = s->x   + (size_t)row_offset * H;
    float *t_mb   = s->t   + (size_t)row_offset * H;
    float *tb_mb  = s->tb  + (size_t)row_offset * (Hd * NA);
    float *qkv_mb = s->qkv + (size_t)row_offset * QKV;
    float *q_mb   = s->q   + (size_t)row_offset * (Hd * NA);
    float *k_mb   = s->k   + (size_t)row_offset * KV;
    float *v_mb   = s->v   + (size_t)row_offset * KV;
    int   *pos_mb = s->positions + row_offset;

    float *key_cache_mb   = s->key_cache   + (size_t)row_offset * kv_slice;
    float *value_cache_mb = s->value_cache + (size_t)row_offset * kv_slice;

    // 1) RMSNorm
    {
        dim3 grid(batch_size), block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<grid, block, 0, sAttn>>>(t_mb, x_mb,
            w->rms_attn_w + (size_t)layer_idx * H, batch_size, H);
        HIP_CHECK(hipGetLastError());
    }
    // 2) QKV
    {
        const int woff = layer_idx * H * QKV;
        matmul_mc(qkv_mb, t_mb, w->w_qkv + woff, batch_size, H, QKV, sAttn);
        HIP_CHECK(hipGetLastError());
    }
    // 3) bias
    {
        const int boff = (size_t)layer_idx * QKV;
        const int elems = batch_size * QKV;
        if (elems > 0) {
            dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            add_bias_kernel<<<grid, THREADS_PER_BLOCK, 0, sAttn>>>(qkv_mb, w->b_qkv + boff, batch_size, QKV);
            HIP_CHECK(hipGetLastError());
        }
    }
    // 4) split + RoPE
    {
        launch_split_qkv_apply_rotary(
            qkv_mb, q_mb, k_mb, v_mb,
            s->cos_vals, s->sin_vals, pos_mb,
            batch_size, NA, NK, Hd, sAttn);
        HIP_CHECK(hipGetLastError());
    }
    // 5) KV cache update
    {
        dim3 grid(batch_size, (KV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block(1, THREADS_PER_BLOCK);
        update_kv_cache_kernel<<<grid, block, 0, sAttn>>>(
            key_cache_mb, value_cache_mb, k_mb, v_mb, pos_mb,
            batch_size, p->n_layers, layer_idx, MAX_SEQ_LEN, KV,
            /* batch_kv_stride = */ layers_capacity * (size_t)KV,
            /* layer_kv_offset = */ layer_elem_offset);
    }
    // 6) fused attention
    {
        const bool apply_window = (p->sliding_window > 0) && ((layer_idx & 1) == 0);
        constexpr int HOST_WARPSIZE = 64;
        dim3 grid(batch_size, NA);
        dim3 block(256);
        const int warps = (block.x + HOST_WARPSIZE - 1) / HOST_WARPSIZE;
        const size_t att_cap = apply_window ? (SW_WINDOW + 1) : MAX_SEQ_LEN;
        const size_t shmem = (att_cap + (size_t)warps * Hd + warps) * sizeof(float);
        assert_smem_or_die(shmem, "fused_attention_kernel");

        hipLaunchKernelGGL(fused_attention_kernel, grid, block, shmem, sAttn,
            tb_mb, q_mb, key_cache_mb, value_cache_mb,
            w->attn_sinks + (size_t)layer_idx * NA,
            s->mask, pos_mb, batch_size,
            NA, NK, Hd, MAX_SEQ_LEN, p->n_layers, layer_idx,
            p->sliding_window > 0,
            /* batch_kv_stride = */ layers_capacity * (size_t)KV,
            /* layer_kv_offset = */ layer_elem_offset);

        HIP_CHECK(hipGetLastError());
    }
    // 7) fused output projection
    {
        const int Kproj = Hd * NA;
        const int N     = H;
        const int woff  = (size_t)layer_idx * Kproj * H;
        const int boff  = (size_t)layer_idx * H;

        dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (batch_size + BLOCK_M - 1) / BLOCK_M);
        dim3 blockDim(LANE_PER_WAVE, WAVES_PER_BLOCK);
        const int ldA = BLOCK_K + PAD_K_MC;
        const int ldB = BLOCK_K + PAD_K_MC;
        const size_t shmem =
            sizeof(uint16_t) * (size_t)(2 * BLOCK_M * ldA + 2 * ldB * BLOCK_N);
        assert_smem_or_die(shmem, "fused_output_projection_kernel_optimized");

        fused_output_projection_kernel_optimized<<<gridDim, blockDim, shmem, sAttn>>>(
            s->x + (size_t)row_offset * H, s->tb + (size_t)row_offset * Kproj,
            w->w_o + woff, w->b_o + boff,
            /*M=*/batch_size, /*K=*/Kproj, /*N=*/N);
        HIP_CHECK(hipGetLastError());
    }
}

// Split helpers
static inline void mlp1_shard(int twoD, int tp_size, int r, int &o_start, int &o_len) {
  const int base = twoD / tp_size;
  const int rem  = twoD % tp_size;
  o_len   = base + (r < rem ? 1 : 0);
  o_start = r * base + (r < rem ? r : rem);
}
static inline void mlp2_shard_input(int D, int tp_size, int r, int &i_start, int &i_len) {
  const int base = D / tp_size;
  const int rem  = D % tp_size;
  i_len   = base + (r < rem ? 1 : 0);
  i_start = r * base + (r < rem ? r : rem);
}

// Sum across rank axis [TP, T, H] -> [T, H]
__global__ void sum_rank_axis_kernel(float *out, const float *parts, int TP, int T, int H) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)T * H;
    if (idx >= total) return;
    int t = (int)(idx / H), h = (int)(idx % H);
    float acc = 0.f;
    for (int r = 0; r < TP; ++r) acc += parts[((size_t)r * T + t) * H + h];
    out[(size_t)t * H + h] = acc;
}

void moe_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size,
             int row_offset, hipStream_t sMoe)
{
    if (batch_size <= 0) return;

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu = &gpu_t->cpu_buffers;

    const int TP = TENSOR_PARALLEL_SIZE;
    int my_dev = 0; HIP_CHECK(hipGetDevice(&my_dev));
    const int group_base    = (my_dev / TP) * TP;
    const int rank_in_group = my_dev - group_base;
    TPGroup &tp = tp_groups[my_dev];

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int K = p->experts_per_token;

    // local microbatch size and TP-union size for this MoE step
    const int bs_local = batch_size;
    const int Bgrp     = TP * bs_local;

    // 0) RMSNorm on local MB: t = RMS(x, w_ffn)
    {
        float *x_mb = s->x + (size_t)row_offset * H;
        float *t_mb = s->t + (size_t)row_offset * H;
        dim3 grid(bs_local), block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<grid, block, 0, sMoe>>>(t_mb, x_mb,
            w->rms_ffn_w + (size_t)layer_idx * H, bs_local, H);
        HIP_CHECK(hipGetLastError());
    }

    // Ensure local t is ready, then synchronize TP group (all ranks)
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);

    // 1) ALL-GATHER normalized inputs across TP into gather_x_g[Bgrp,H]
    {
        // self copy
        float *dst_self = s->gather_x_g + (size_t)rank_in_group * bs_local * H;
        float *src_self = s->t          + (size_t)row_offset    * H;
        HIP_CHECK(hipMemcpyAsync(dst_self, src_self,
                                 (size_t)bs_local * H * sizeof(float),
                                 hipMemcpyDeviceToDevice, sMoe));
        // peer copies
        for (int r = 0; r < TP; ++r) {
            const int peer_dev = group_base + r;
            if (peer_dev == my_dev) continue;
            float *dst      = s->gather_x_g + (size_t)r * bs_local * H;
            float *peer_src = gpu_transformers[peer_dev]->state.t + (size_t)row_offset * H;

            if (tp.p2p[rank_in_group][r]) {
                HIP_CHECK(hipMemcpyPeerAsync(dst, my_dev, peer_src, peer_dev,
                                             (size_t)bs_local * H * sizeof(float), sMoe));
            } else {
                HIP_CHECK(hipMemcpyAsync(cpu->tp_host_stage, peer_src,
                                         (size_t)bs_local * H * sizeof(float),
                                         hipMemcpyDeviceToHost, sMoe));
                HIP_CHECK(hipMemcpyAsync(dst, cpu->tp_host_stage,
                                         (size_t)bs_local * H * sizeof(float),
                                         hipMemcpyHostToDevice, sMoe));
            }
        }
        HIP_CHECK(hipStreamSynchronize(sMoe));
    }
    tp_group_barrier(tp);

    // 2) Router on union -> topk indices/weights (softmax on top-k scores)
    matmul_mc(s->router_score_g, s->gather_x_g,
              w->w_router + (size_t)layer_idx * H * E, Bgrp, H, E, sMoe);
    HIP_CHECK(hipGetLastError());
    {
        const int elems = Bgrp * E;
        if (elems > 0) {
            add_bias_kernel<<<(elems + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,
                              THREADS_PER_BLOCK, 0, sMoe>>>(
                s->router_score_g, w->b_router + (size_t)layer_idx * E, Bgrp, E);
            HIP_CHECK(hipGetLastError());
        }
        topk_kernel<<<Bgrp, 1, 0, sMoe>>>(s->topk_v_g, s->topk_i_g, s->router_score_g, Bgrp, E, K);
        HIP_CHECK(hipGetLastError());
        softmax_kernel<<<Bgrp, THREADS_PER_BLOCK, 0, sMoe>>>(s->topk_v_g, Bgrp, K);
        HIP_CHECK(hipGetLastError());
    }

    // 3) Count tokens per expert (order-independent atomics OK), build offsets host-side
    HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, E * sizeof(int), sMoe));
    {
        dim3 grid((Bgrp + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<grid, THREADS_PER_BLOCK, 0, sMoe>>>(
            s->topk_i_g, s->d_expert_counts, Bgrp, K);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(sMoe));
    HIP_CHECK(hipMemcpy(cpu->expert_counts, s->d_expert_counts,
                        E * sizeof(int), hipMemcpyDeviceToHost));

    int total_tokens = 0;
    for (int e = 0; e < E; ++e) {
        cpu->expert_offsets[e] = total_tokens;
        total_tokens += cpu->expert_counts[e];
    }
    if (E > 0) HIP_CHECK(hipMemcpyAsync(s->d_expert_offsets, cpu->expert_offsets,
                                        E * sizeof(int), hipMemcpyHostToDevice, sMoe));

    // Build tile maps for grouped matmuls (host) and copy to device
    int cur_tiles = 0;
    for (int e = 0; e < E; ++e) {
        const int cnt   = cpu->expert_counts[e];
        const int tiles = (cnt + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
        for (int m = 0; m < tiles; ++m) {
            cpu->h_tile2expert[cur_tiles + m] = e;
            cpu->h_tile2local [cur_tiles + m] = m;
        }
        cur_tiles += tiles;
    }
    if (cur_tiles > s->cap_tiles) {
        fprintf(stderr, "[MoE] tiles(%d) > cap_tiles(%d). Increase cap or adjust BLOCK_M_MLP.\n",
                cur_tiles, s->cap_tiles);
        abort();
    }
    if (cur_tiles > 0) {
        HIP_CHECK(hipMemcpyAsync(s->d_tile2expert_g, cpu->h_tile2expert,
                                 cur_tiles * sizeof(int), hipMemcpyHostToDevice, sMoe));
        HIP_CHECK(hipMemcpyAsync(s->d_tile2local_g,  cpu->h_tile2local,
                                 cur_tiles * sizeof(int), hipMemcpyHostToDevice, sMoe));
    }

    // 4) FUSED deterministic routing + packing (no atomics)
    if (total_tokens > 0) {
        int threads = 1;
        while (threads < Bgrp) threads <<= 1;
        threads = min(threads, 1024);  // hardware cap

        // 5 arrays of int[T]
        size_t shmem = (size_t)threads * 5 * sizeof(int);

        route_and_pack_fused_kernel<<<E, threads, shmem, sMoe>>>(
            s->gather_x_g, Bgrp, H,
            s->topk_i_g, s->topk_v_g, K,
            s->d_expert_offsets, E,
            s->local_ids_g, s->local_wts_g,
            s->expert_input_buffer_g);

    } else {
        // nothing routed; just residual-add zeros later
        return;
    }

    // 5) MLP1 (column-parallel on O=2D): local shard
    int o_len = 0;
    {
        const int twoD = 2 * D;
        const int base = twoD / TP;
        const int rem  = twoD % TP;
        o_len = base + (rank_in_group < rem ? 1 : 0);

        const size_t seg1_loc = (size_t)o_len * H; // per-expert stride in local shard
        const __hip_bfloat16 *W1 = w->w_mlp1
            + (size_t)layer_idx * (size_t)E * seg1_loc;

        dim3 grid((o_len + BLOCK_N_MLP - 1) / BLOCK_N_MLP, cur_tiles);
        dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK_MLP);
        const size_t shmem = (size_t)(2*BLOCK_M_MLP*BLOCK_K + 2*BLOCK_K*BLOCK_N_MLP) * sizeof(uint16_t);
        assert_smem_or_die(shmem, "grouped_mlp1_bf16_kernel(TP)");
        hipLaunchKernelGGL(grouped_mlp1_bf16_kernel, grid, block, shmem, sMoe,
            s->mlp1_out_g, s->expert_input_buffer_g, W1,
            s->d_expert_offsets, s->d_expert_counts,
            s->d_tile2expert_g, s->d_tile2local_g,
            E, /*K=*/H, /*N=*/o_len);
        HIP_CHECK(hipGetLastError());
    }

    // 6) SwiGLU + bias for local columns -> gate_up_g with Dloc = o_len/2
    const int Dloc = o_len / 2;
    {
        const __hip_bfloat16 *b1 = w->b_mlp1
            + (size_t)layer_idx * (size_t)E * (size_t)o_len;

        const size_t work = (size_t)total_tokens * Dloc;
        if (work > 0) {
            const int T = 256;
            dim3 grid2((int)((work + T - 1) / T)), block2(T);
            bias_swiglu_epilogue_kernel<<<grid2, block2, 0, sMoe>>>(
                s->mlp1_out_g, b1, s->d_expert_offsets, s->d_expert_counts, E,
                s->gate_up_g, /*D=*/Dloc, total_tokens, p->swiglu_limit, 1.702f);
            HIP_CHECK(hipGetLastError());
        }
    }

    // 7) MLP2 (row-parallel on input D) (+bias H full, pre-scaled by 1/TP)
    {
        int i_len, i_start;
        {
            const int base = D / TP, rem = D % TP;
            i_len   = base + (rank_in_group < rem ? 1 : 0);
            i_start = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }
        if (i_len != Dloc) {
            fprintf(stderr, "MLP2 shard mismatch: i_len=%d vs Dloc=%d\n", i_len, Dloc);
            abort();
        }

        const size_t seg2_loc = (size_t)H * (size_t)i_len;
        const __hip_bfloat16 *W2  = w->w_mlp2
            + (size_t)layer_idx * (size_t)E * seg2_loc;
        const __hip_bfloat16 *b2s = w->b_mlp2 + (size_t)layer_idx * (size_t)E * H;

        dim3 grid((H + BLOCK_N_MLP - 1) / BLOCK_N_MLP, cur_tiles);
        dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK_MLP);
        const size_t shmem = (size_t)(2*BLOCK_M_MLP*BLOCK_K + 2*BLOCK_K*BLOCK_N_MLP) * sizeof(uint16_t);
        assert_smem_or_die(shmem, "grouped_mlp2_bf16_bias_kernel(TP)");
        hipLaunchKernelGGL(grouped_mlp2_bf16_bias_kernel, grid, block, shmem, sMoe,
            s->expert_output_partial_g, s->gate_up_g, W2, b2s,
            s->d_expert_offsets, s->d_expert_counts,
            s->d_tile2expert_g, s->d_tile2local_g,
            E, /*K=*/i_len, /*N=*/H);
        HIP_CHECK(hipGetLastError());
    }

    // Ensure partial results are ready before cross-rank reduction
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);

    // 8) Emulate ALL-REDUCE across TP on [total_tokens, H]
    {
        const size_t bytes = (size_t)total_tokens * H * sizeof(float);
        const size_t span  = (size_t)total_tokens * H;

        // self
        HIP_CHECK(hipMemcpyAsync(
            s->expert_output_gather_g + (size_t)rank_in_group * span,
            s->expert_output_partial_g, bytes,
            hipMemcpyDeviceToDevice, sMoe));

        // peers
        for (int r = 0; r < TP; ++r) {
            const int peer_dev = group_base + r;
            if (peer_dev == my_dev) continue;
            float *peer_src = gpu_transformers[peer_dev]->state.expert_output_partial_g;
            float *dst      = s->expert_output_gather_g + (size_t)r * span;

            if (tp.p2p[rank_in_group][r]) {
                HIP_CHECK(hipMemcpyPeerAsync(dst, my_dev, peer_src, peer_dev, bytes, sMoe));
            } else {
                HIP_CHECK(hipMemcpyAsync(cpu->tp_host_stage, peer_src, bytes, hipMemcpyDeviceToHost, sMoe));
                HIP_CHECK(hipMemcpyAsync(dst, cpu->tp_host_stage, bytes, hipMemcpyHostToDevice, sMoe));
            }
        }
        HIP_CHECK(hipStreamSynchronize(sMoe));
        tp_group_barrier(tp);

        // sum across ranks -> expert_output_partial_g
        const size_t total = (size_t)total_tokens * H;
        const int BLK = 256, GRD = (int)((total + BLK - 1) / BLK);
        sum_rank_axis_kernel<<<GRD, BLK, 0, sMoe>>>(
            s->expert_output_partial_g, s->expert_output_gather_g, TP, total_tokens, H);
        HIP_CHECK(hipGetLastError());
    }

    // 9) Reduce over experts back to tokens (union) -> e_agg_g[Bgrp,H]
    {
        const int threads = 256;
        dim3 grid(Bgrp, (H + threads - 1) / threads);
        reduce_tokenwise_expert_outputs<<<grid, threads, 0, sMoe>>>(
            s->e_agg_g, s->expert_output_partial_g,
            s->local_ids_g, s->local_wts_g, Bgrp, H, K);
        HIP_CHECK(hipGetLastError());
    }

    // 10) Scatter owner rows back to this rank and residual-add into x
    {
        float *x_mb      = s->x + (size_t)row_offset * H;
        float *src_owner = s->e_agg_g + (size_t)rank_in_group * bs_local * H;
        const int elems  = bs_local * H;
        accumulate_kernel<<<(elems + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK, 0, sMoe>>>(
            x_mb, src_owner, 1.0f, bs_local, H);
        HIP_CHECK(hipGetLastError());
    }
}

// ------------------------------ Pipelined forward (layer overlap) -------------------------
#ifndef MICRO_BATCH_SIZE
#define MICRO_BATCH_SIZE 128
#endif

int *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
{
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
    const int H = p->hidden_dim;
    const int B = batch_size;
    if (B <= 0) return cpu_buf->current_tokens;

    // streams
    hipStream_t attn_stream = nullptr, moe_stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&attn_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&moe_stream,  hipStreamNonBlocking));

    // microbatching — keep your MB exactly (no TP-capping inside attention)
    const int MB = (MICRO_BATCH_SIZE <= B) ? MICRO_BATCH_SIZE : B;
    const int NUM_MB = (B + MB - 1) / MB;

    // per-microbatch events
    auto make_evt = []() {
        hipEvent_t e = nullptr;
        HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
        return e;
    };
    std::vector<hipEvent_t> evt_attn_done(NUM_MB);
    std::vector<hipEvent_t> evt_moe_done_prev(NUM_MB), evt_moe_done_cur(NUM_MB);
    for (int i = 0; i < NUM_MB; ++i) {
        evt_attn_done[i]    = make_evt();
        evt_moe_done_prev[i]= make_evt();
        evt_moe_done_cur[i] = make_evt();
    }

    // H2D tokens + positions
    h2d_copy(s->current_tokens, tokens,             (size_t)B * sizeof(int), attn_stream);
    h2d_copy(s->positions,      cpu_buf->positions, (size_t)B * sizeof(int), attn_stream);

    // embeddings on attn_stream
    {
        const int elems = B * H;
        if (elems > 0) {
            dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            copy_embeddings_kernel<<<grid, THREADS_PER_BLOCK, 0, attn_stream>>>(
                s->x, w->token_embedding_table, s->current_tokens, B, H);
            HIP_CHECK(hipGetLastError());
        }
    }

    // pipelined schedule across layers and microbatches
    for (int l = 0; l < p->n_layers; ++l) {
        int mb_idx = 0;
        for (int row = 0; row < B; row += MB, ++mb_idx) {
            const int bs = std::min(MB, B - row);
            const int i  = mb_idx;

            // cross-layer dep: Attention(l,i) waits MoE(l-1,i)
            if (l > 0) HIP_CHECK(hipStreamWaitEvent(attn_stream, evt_moe_done_prev[i], 0));

            // Attn(l,i)
            attention_gpu(gpu_t, l, bs, /*row_offset=*/row, attn_stream);
            HIP_CHECK(hipEventRecord(evt_attn_done[i], attn_stream));

            // MoE(l,i-1) on the separate stream
            if (i > 0) {
                const int prev_row = row - MB;
                const int prev_bs  = std::min(MB, B - prev_row);
                HIP_CHECK(hipStreamWaitEvent(moe_stream, evt_attn_done[i - 1], 0));
                moe_gpu(gpu_t, l, prev_bs, /*row_offset=*/prev_row, moe_stream);
                HIP_CHECK(hipEventRecord(evt_moe_done_cur[i - 1], moe_stream));
            }
        }

        // drain last microbatch for this layer
        {
            const int i_last   = NUM_MB - 1;
            const int row_last = i_last * MB;
            const int bs_last  = std::min(MB, B - row_last);
            if (bs_last > 0) {
                HIP_CHECK(hipStreamWaitEvent(moe_stream, evt_attn_done[i_last], 0));
                moe_gpu(gpu_t, l, bs_last, /*row_offset=*/row_last, moe_stream);
                HIP_CHECK(hipEventRecord(evt_moe_done_cur[i_last], moe_stream));
            }
        }
        // make next layer depend on current layer's MoE-done (per microbatch)
        std::swap(evt_moe_done_prev, evt_moe_done_cur);
        for (int i = 0; i < NUM_MB; ++i) {
            HIP_CHECK(hipEventDestroy(evt_moe_done_cur[i]));
            evt_moe_done_cur[i] = make_evt();
        }
    }

    // sync compute streams before head
    HIP_CHECK(hipStreamSynchronize(attn_stream));
    HIP_CHECK(hipStreamSynchronize(moe_stream));

    // final norm + head on default stream
    {
        dim3 grid(B), block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<grid, block>>>(s->x, s->x, w->rms_out_w, B, H);
        HIP_CHECK(hipGetLastError());
    }
    {
        matmul_mc(s->logits, s->x, w->out, B, H, p->vocab_size);
        HIP_CHECK(hipGetLastError());
    }
    {
        sample_argmax(s->logits, s->current_tokens, B, p->vocab_size);
        HIP_CHECK(hipGetLastError());
    }

    // D2H tokens
    d2h_copy(cpu_buf->current_tokens, s->current_tokens, (size_t)B * sizeof(int), 0);

    // cleanup
    for (int i = 0; i < NUM_MB; ++i) {
        HIP_CHECK(hipEventDestroy(evt_attn_done[i]));
        HIP_CHECK(hipEventDestroy(evt_moe_done_prev[i]));
        HIP_CHECK(hipEventDestroy(evt_moe_done_cur[i]));
    }
    HIP_CHECK(hipStreamDestroy(attn_stream));
    HIP_CHECK(hipStreamDestroy(moe_stream));

    return cpu_buf->current_tokens;
}

// Replace previous clear_kv_cache_for_slot kernel with host memset version
static inline void clear_kv_cache_for_slot(GPURunState* s, const Config* p, int slot)
{
    if (slot < 0 || slot >= BATCH_SIZE) return;

    const int kv_dim = p->head_dim * p->n_kv_heads;
    const size_t elem_bytes = sizeof(float);

    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    size_t layers_capacity = 0;
    for (int L = 0; L < p->n_layers; ++L) {
        const bool evenL = ((L & 1) == 0);
        layers_capacity += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t slot_base_elems = (size_t)slot * layers_capacity * (size_t)kv_dim;

    size_t layer_pos_offset = 0;
    for (int L = 0; L < p->n_layers; ++L) {
        const bool evenL = ((L & 1) == 0);
        const size_t capL = (size_t)(evenL ? even_cap : MAX_SEQ_LEN);

        const size_t layer_base_elems = slot_base_elems + layer_pos_offset * (size_t)kv_dim;
        const size_t bytes_this_layer = capL * (size_t)kv_dim * elem_bytes;

        void* k_ptr = (void*)((char*)s->key_cache   + layer_base_elems * elem_bytes);
        void* v_ptr = (void*)((char*)s->value_cache + layer_base_elems * elem_bytes);

        HIP_CHECK(hipMemset(k_ptr, 0, bytes_this_layer));
        HIP_CHECK(hipMemset(v_ptr, 0, bytes_this_layer));

        layer_pos_offset += capL;
    }
}

long long continuous_batching_inference(Tokenizer *tokenizer,
                                        Sampler *sampler, Requests *requests)
{
    long long total_tokens_generated = 0;

    const int BOS_TOKEN_ID = 1;

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
        int end_request   = start_request + local_request;

        Config     *p       = &gpu_t->config;
        CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
        GPURunState *state  = &gpu_t->state;

        cpu_buf->next_request_idx = start_request;

        for (int slot = 0; slot < BATCH_SIZE; slot++) {
            cpu_buf->slot_active_cpu[slot]   = false;
            cpu_buf->request_mapping_cpu[slot]= -1;
            cpu_buf->seq_lengths_cpu[slot]   = 0;
            cpu_buf->positions[slot]         = 0;
            cpu_buf->finished[slot]          = true;
            cpu_buf->current_tokens[slot]    = BOS_TOKEN_ID;
        }

        // Fill initial batch
        for (int slot = 0; slot < BATCH_SIZE && cpu_buf->next_request_idx < end_request; slot++) {
            int req_idx = cpu_buf->next_request_idx++;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            encode(tokenizer, input_seq, -1, -1,
                   cpu_buf->prompt_tokens[slot],
                   &cpu_buf->prompt_lens[slot],
                   p->initial_context_length);

            if (cpu_buf->prompt_lens[slot] < 1) {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                cpu_buf->prompt_lens[slot]   = 1;
                cpu_buf->prompt_tokens[slot][0] = BOS_TOKEN_ID;
            }

            cpu_buf->request_mapping_cpu[slot] = req_idx;
            cpu_buf->slot_active_cpu[slot]     = true;
            cpu_buf->seq_lengths_cpu[slot]     = 0;
            cpu_buf->positions[slot]           = 0;
            cpu_buf->finished[slot]            = false;
            cpu_buf->current_tokens[slot]      = cpu_buf->prompt_tokens[slot][0];
        }

        // Push initial slot metadata to device
        HIP_CHECK(hipMemcpy(state->slot_active,     cpu_buf->slot_active_cpu,  BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->request_mapping, cpu_buf->request_mapping_cpu, BATCH_SIZE * sizeof(int),  hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->seq_lengths,     cpu_buf->seq_lengths_cpu,  BATCH_SIZE * sizeof(int),  hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->positions,       cpu_buf->positions,        BATCH_SIZE * sizeof(int),  hipMemcpyHostToDevice));

        const int max_steps = requests->max_seq_len;

        while (true)
        {
            bool any_active = false;
            for (int slot = 0; slot < BATCH_SIZE; ++slot) {
                if (cpu_buf->slot_active_cpu[slot]) { any_active = true; break; }
            }
            if (!any_active) break;

            for (int slot = 0; slot < BATCH_SIZE; ++slot) {
                if (!cpu_buf->slot_active_cpu[slot]) {
                    cpu_buf->positions[slot]      = 0;
                    cpu_buf->current_tokens[slot] = BOS_TOKEN_ID;
                }
            }

            int *next_tokens = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, BATCH_SIZE);

            for (int slot = 0; slot < BATCH_SIZE; slot++) {
                if (!cpu_buf->slot_active_cpu[slot]) continue;

                int req_idx = cpu_buf->request_mapping_cpu[slot];
                int pos     = cpu_buf->positions[slot];

                pos++;

                int next_token;
                if (pos < cpu_buf->prompt_lens[slot]) {
                    next_token = cpu_buf->prompt_tokens[slot][pos];
                } else {
                    next_token = next_tokens[slot];
                    int *out = get_tok_gen_ptr(requests, req_idx);
                    const int gen_pos = pos - cpu_buf->prompt_lens[slot];
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        out[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }

                const bool completed =
                    (next_token == 199999 || next_token == 200002 ||
                     pos >= max_steps - 1 || pos >= MAX_SEQ_LEN - 2);

                if (completed) {
                    int *out = get_tok_gen_ptr(requests, req_idx);
                    const int gen_pos = pos - cpu_buf->prompt_lens[slot] + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        out[gen_pos] = -1;
                    }

                    if (cpu_buf->next_request_idx < end_request) {
                        clear_kv_cache_for_slot(&gpu_t->state, &gpu_t->config, slot);

                        req_idx = cpu_buf->next_request_idx++;
                        const char *input_seq = get_str_req_ptr(requests, req_idx);

                        encode(tokenizer, input_seq, -1, -1,
                               cpu_buf->prompt_tokens[slot],
                               &cpu_buf->prompt_lens[slot],
                               p->initial_context_length);

                        if (cpu_buf->prompt_lens[slot] < 1) {
                            fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                            cpu_buf->prompt_lens[slot]   = 1;
                            cpu_buf->prompt_tokens[slot][0] = 1;
                        }

                        cpu_buf->request_mapping_cpu[slot] = req_idx;
                        cpu_buf->positions[slot]           = 0;
                        cpu_buf->seq_lengths_cpu[slot]     = 0;
                        cpu_buf->finished[slot]            = false;
                        cpu_buf->current_tokens[slot]      = cpu_buf->prompt_tokens[slot][0];
                    } else {
                        cpu_buf->slot_active_cpu[slot]   = false;
                        cpu_buf->request_mapping_cpu[slot]= -1;
                        cpu_buf->positions[slot]         = 0;
                        cpu_buf->seq_lengths_cpu[slot]   = 0;
                        cpu_buf->current_tokens[slot]    = 1;
                    }
                } else {
                    cpu_buf->positions[slot]       = pos;
                    cpu_buf->seq_lengths_cpu[slot]++;
                    cpu_buf->current_tokens[slot]  = next_token;
                }
            }

            HIP_CHECK(hipMemcpy(state->slot_active, cpu_buf->slot_active_cpu,   BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(state->seq_lengths, cpu_buf->seq_lengths_cpu,   BATCH_SIZE * sizeof(int),  hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(state->positions,   cpu_buf->positions,         BATCH_SIZE * sizeof(int),  hipMemcpyHostToDevice));
        }
    }

    int max_steps = gpu_transformers[0]->config.seq_len;
    Config *p0 = gpu_transformers[0] ? &gpu_transformers[0]->config : nullptr;

    // Print all results
    for (int req_idx = 0; req_idx < requests->num_reqs; req_idx++) {
        const char *input_seq = get_str_req_ptr(requests, req_idx);
        int *output_tokens    = get_tok_gen_ptr(requests, req_idx);

        safe_printf(input_seq);
        printf("!");

        int *temp_tokens = (int *)malloc((MAX_SEQ_LEN + 3) * sizeof(int));
        int temp_len;
        encode(tokenizer, input_seq, -1, -1, temp_tokens, &temp_len, p0->initial_context_length);
        int prev_token = temp_tokens[temp_len - 1];
        free(temp_tokens);

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

    write_profile_info();
    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    return continuous_batching_inference(tokenizer, sampler, requests);
}

#endif // GETP_RUN
