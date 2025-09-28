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
#include "kernels/attention_mfma.hpp"

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
#include "nccl.hpp" // <— NCCL/RCCL-free TP collectives/helpers
#include "tp_ring.hpp"

#ifndef GETP_RUN
#define GETP_RUN

#define BLOCK_M_MLP 16 * 4

int num_gpus = 1;
bool IS_20B_MODEL = 1;

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
    __hip_bfloat16 *w_mlp1; // (n_layers, n_experts, 2*D, H) row-major [O=2D, I=H]
    __hip_bfloat16 *w_mlp2; // (n_layers, n_experts, H,   D) row-major [O=H,  I=D]
    __hip_bfloat16 *b_mlp1, *b_mlp2;

     // MoE weights now use MXFP4 quantization
    // MoE weights (pure BF16 now)
    uint8_t *w_mlp1_mxfp4, *w_mlp2_mxfp4;   // packed n/2 bytes
    uint8_t *w_mlp1_scales, *w_mlp2_scales; // n/32 bytes (e8m0 per block)
    // GPUTransformerWeights (add two pointers)
    float *w_mlp1_scales_f32, *w_mlp2_scales_f32; // one float per 32 elems


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
    __hip_bfloat16 *key_cache;   // (batch_size, n_layers, seq_len, kv_dim)
    __hip_bfloat16 *value_cache; // (batch_size, n_layers, seq_len, kv_dim)

    // RoPE buffers
    float *cos_vals; // (head_dim/2, seq_len)
    float *sin_vals; // (head_dim/2, seq_len)

    // === legacy MoE (kept for shared kernels/utilities) ===
    float *router_score; // (batch_size, n_experts)

    // Persistent routing buffers
    int *d_expert_counts;  // (n_experts)
    int *d_expert_offsets; // (n_experts)

    // === NEW: TP-union MoE buffers (per rank) ===
    // union batch (within a TP group) = Bgrp_max = TENSOR_PARALLEL_SIZE * BATCH_SIZE
    float *gather_x_g;     // [Bgrp_max, H]       — allgathered normalized inputs
    float *router_score_g; // [Bgrp_max, E]
    float *topk_v_g;       // [Bgrp_max, K]
    int *topk_i_g;         // [Bgrp_max, K]
    int *local_ids_g;      // [Bgrp_max, K]
    float *local_wts_g;    // [Bgrp_max, K]
    float *e_agg_g;        // [Bgrp_max, H]

    // expert input/output (union)
    float *expert_input_buffer_g;   // [pairs_max, H]         pairs_max = Bgrp_max * K
    float *mlp1_out_g;              // [pairs_max, o_len_max] o_len_max = ceil(2D/TP)
    float *gate_up_g;               // [pairs_max, Dloc_max]  Dloc_max  = ceil(D/TP)
    float *expert_output_partial_g; // [pairs_max, H]         this-rank partial
    float *expert_output_gather_g;  // [TP * pairs_max, H]    workspace for emulate all-reduce

    // tile maps for union
    int cap_tiles;        // capacity for tile maps
    int *d_tile2expert_g; // [cap_tiles]
    int *d_tile2local_g;  // [cap_tiles]

    // Token and position buffers
    int *current_tokens; // current tokens (batch_size)
    int *positions;      // current positions (batch_size)

    // Output buffer
    float *logits; // output logits (batch_size, vocab_size)

    // Continuous batching fields
    int *seq_lengths;     // Current sequence length for each slot [BATCH_SIZE]
    bool *slot_active;    // Whether slot is processing a request [BATCH_SIZE]
    int *request_mapping; // Maps batch slot -> request index in Requests [BATCH_SIZE]

    int *local_ids;   // [BATCH_SIZE * K]
    float *local_wts; // [BATCH_SIZE * K]
    int *n_local;     // [BATCH_SIZE]
    float *d_recv;

    // ---- P2P MoE routing (EP, order-preserving) ----
    // Sender outbox (per-destination concatenated order)
    float *obox_X;     // [pairs_max, H]
    int   *obox_tok;   // [pairs_max]
    int   *obox_exp;   // [pairs_max] global expert id
    float *obox_wt;    // [pairs_max]

    // Owner inbox (concat of senders)
    float *ibox_X;     // [pairs_max, H]
    int   *ibox_tok;   // [pairs_max] original sender's token index (0..bs_local-1)
    int   *ibox_exp;   // [pairs_max]
    float *ibox_wt;    // [pairs_max]

    // Owner grouped by local expert (for grouped matmuls)
    float *grp_X;      // [pairs_max, H]
    int   *grp_tok;    // [pairs_max]
    float *grp_wt;     // [pairs_max]
    int   *grp_perm;   // [pairs_max] maps packed index -> inbox index

    float *grp_out;    // [pairs_max, H] output aligned with grp_* (after MLP)
    float *ret_out;    // [pairs_max, H] ret_out[ grp_perm[p] ] = grp_out[p]

    // Sender return buffer (results from owners in *same order* as obox_*)
    float *ret_pairs_X; // [pairs_max, H]

    // Small device scratch for sender pack atomics (per-destination counters/off)
    int *d_peer_send_off; // [TP] prefix offsets within obox_*
    int *d_peer_counters; // [TP] (zeroed before pack) per-dest atomic counters
// put near other TP/MoE scratch
int *ctrl_mat;      // [TP * TP], device-side control matrix (row-major)
// optional local row staging on device (so we don’t depend on host during exchange)
int *ctrl_row_dev;  // [TP]

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
    void *tp_host_stage;
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
TPGroup tp_groups[MAX_GPUS]; // one per device

// -------------------- helpers --------------------
#ifndef USE_ASYNC_HOST_COPIES
#define USE_ASYNC_HOST_COPIES 0 // set to 1 only after switching CPU buffers to hipHostMalloc
#endif

static inline void h2d_copy(void *dst, const void *src, size_t bytes, hipStream_t s)
{
#if USE_ASYNC_HOST_COPIES
    if (bytes)
        HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, s));
#else
    (void)s;
    if (bytes)
        HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#endif
}
static inline void d2h_copy(void *dst, const void *src, size_t bytes, hipStream_t s)
{
#if USE_ASYNC_HOST_COPIES
    if (bytes)
        HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, s));
#else
    (void)s;
    if (bytes)
        HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#endif
}
static inline size_t max_dynamic_smem_bytes()
{
    int v = 0;
    HIP_CHECK(hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    return (size_t)v;
}
static inline void assert_smem_or_die(size_t bytes, const char *kernel_name)
{
    const size_t limit = max_dynamic_smem_bytes();
    if (bytes > limit)
    {
        fprintf(stderr, "[HIP] %s needs %zuB dynamic shared mem, but limit is %zuB. "
                        "Reduce BLOCK_* or SW_WINDOW.\n",
                kernel_name, bytes, limit);
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
    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int K = p->experts_per_token;

    // union capacities
    const int Bgrp_max = TP * BATCH_SIZE; // max union rows per microstep
    const int pairs_max = Bgrp_max * K;   // worst-case tokens routed

    int o_len_max, Dloc_max;
    if (IS_20B_MODEL) {            // TP-union (20B)
        o_len_max = (2 * D + TP - 1) / TP;
        Dloc_max  = (D + TP - 1) / TP;
    } else {                       // Expert-Parallel (120B)
        o_len_max = 2 * D;         // full 2D output of MLP1
        Dloc_max  = D;             // full D into MLP2
    }
    
    // Initialize pointers to NULL (only new ones; others allocated below)
    s->mask = NULL;
    s->d_expert_counts = NULL;
    s->d_expert_offsets = NULL;

    // Check memory requirements
    size_t batch_hidden = BATCH_SIZE * H * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);

    // per-layer capacities: even layers use SW_WINDOW (if sliding enabled), odd are full
    const int n_even = (p->n_layers + 1) / 2;
    const int n_odd = p->n_layers - n_even;
    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    const size_t layers_capacity =
        (size_t)n_even * (size_t)even_cap + (size_t)n_odd * (size_t)MAX_SEQ_LEN;

    size_t kv_cache_size = (size_t)BATCH_SIZE * layers_capacity * (size_t)kv_dim * sizeof(__hip_bfloat16);

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

    // Persistent routing buffers
    HIP_CHECK(hipMalloc((void **)&s->d_expert_counts, E * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_expert_offsets, E * sizeof(int)));

    int total_mtiles = BATCH_SIZE * K;

    // KV cache
    HIP_CHECK(hipMalloc((void **)&s->key_cache, kv_cache_size));
    HIP_CHECK(hipMalloc((void **)&s->value_cache, kv_cache_size));

    HIP_CHECK(hipMalloc((void **)&s->att, BATCH_SIZE * p->n_attn_heads * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->logits, BATCH_SIZE * p->vocab_size * sizeof(float)));

    HIP_CHECK(hipMalloc((void **)&s->local_ids, BATCH_SIZE * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->local_wts, BATCH_SIZE * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->n_local, BATCH_SIZE * sizeof(int)));

    HIP_CHECK(hipMalloc((void **)&s->router_score, BATCH_SIZE * E * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->positions, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->cos_vals, (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->sin_vals, (p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->temp_buffer, batch_hidden));

    // Continuous batching fields
    HIP_CHECK(hipMalloc((void **)&s->seq_lengths, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->slot_active, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipMalloc((void **)&s->request_mapping, BATCH_SIZE * sizeof(int)));

    // === NEW: TP-union allocations ===
    HIP_CHECK(hipMalloc((void **)&s->gather_x_g, (size_t)Bgrp_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->router_score_g, (size_t)Bgrp_max * E * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_v_g, (size_t)Bgrp_max * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->topk_i_g, (size_t)Bgrp_max * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->local_ids_g, (size_t)Bgrp_max * K * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->local_wts_g, (size_t)Bgrp_max * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->e_agg_g, (size_t)Bgrp_max * H * sizeof(float)));

    HIP_CHECK(hipMalloc((void **)&s->expert_input_buffer_g, (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->mlp1_out_g, (size_t)pairs_max * o_len_max * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->gate_up_g,  (size_t)pairs_max * Dloc_max * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->expert_output_partial_g, (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&s->expert_output_gather_g, (size_t)TP * pairs_max * H * sizeof(float)));
    
    // ---- P2P MoE routing buffers ----
    HIP_CHECK(hipMalloc((void**)&s->obox_X,       (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->obox_tok,     (size_t)pairs_max * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->obox_exp,     (size_t)pairs_max * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->obox_wt,      (size_t)pairs_max * sizeof(float)));

    HIP_CHECK(hipMalloc((void**)&s->ibox_X,       (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->ibox_tok,     (size_t)pairs_max * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->ibox_exp,     (size_t)pairs_max * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->ibox_wt,      (size_t)pairs_max * sizeof(float)));

    HIP_CHECK(hipMalloc((void**)&s->grp_X,        (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->grp_tok,      (size_t)pairs_max * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->grp_wt,       (size_t)pairs_max * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->grp_perm,     (size_t)pairs_max * sizeof(int)));

    HIP_CHECK(hipMalloc((void**)&s->grp_out,      (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->ret_out,      (size_t)pairs_max * H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&s->ret_pairs_X,  (size_t)pairs_max * H * sizeof(float)));

    HIP_CHECK(hipMalloc((void**)&s->d_peer_send_off, TENSOR_PARALLEL_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&s->d_peer_counters, TENSOR_PARALLEL_SIZE * sizeof(int)));

// after you know TP (= TENSOR_PARALLEL_SIZE)
HIP_CHECK(hipMalloc((void**)&s->ctrl_mat,     TP * TP * sizeof(int)));
HIP_CHECK(hipMalloc((void**)&s->ctrl_row_dev, TP * sizeof(int)));
HIP_CHECK(hipMemset(s->ctrl_mat, 0, TP * TP * sizeof(int)));


    // (Optional) zero-once; you’ll overwrite every step anyway
    HIP_CHECK(hipMemset(s->d_peer_counters, 0, TENSOR_PARALLEL_SIZE * sizeof(int)));


    // Calculate maximum possible tiles needed
    // In worst case, all tokens could be distributed across experts
    // Each expert processes its tokens in tiles of size BLOCK_M_MLP
    // Since each token selects K experts, max tokens per expert is min(pairs_max, Bgrp_max * K)
    // But to be safe, we assume worst-case distribution
    int max_tiles_per_expert = (pairs_max + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
    if(IS_20B_MODEL) s->cap_tiles = max_tiles_per_expert * 2;  // 2x safety factor for worst-case distribution
    else s->cap_tiles = max_tiles_per_expert * 4; // 4x safety factor for worst-case distribution

    HIP_CHECK(hipMalloc((void **)&s->d_tile2expert_g, s->cap_tiles * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&s->d_tile2local_g, s->cap_tiles * sizeof(int)));

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

    if (!IS_20B_MODEL) {
        const size_t TILE_ELEMS = RING_TILE_BYTES / sizeof(float);
        HIP_CHECK(hipMalloc((void**)&s->d_recv, TILE_ELEMS * sizeof(float)));
    }
}

void malloc_gpu_weights_20b(GPUTransformerWeights *w, Config *p)
{
    // Figure out my TP rank from current HIP device
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    const int TP = TENSOR_PARALLEL_SIZE;
    const int r = dev % TP;

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;

    // Local shard geometry (no forward-declarations needed — use small lambdas)
    auto mlp1_shard = [](int twoD, int tp, int rr, int &o_start, int &o_len)
    {
        const int base = twoD / tp, rem = twoD % tp;
        o_len = base + (rr < rem ? 1 : 0);
        o_start = rr * base + (rr < rem ? rr : rem);
    };
    auto mlp2_shard_input = [](int Din, int tp, int rr, int &i_start, int &i_len)
    {
        const int base = Din / tp, rem = Din % tp;
        i_len = base + (rr < rem ? 1 : 0);
        i_start = rr * base + (rr < rem ? rr : rem);
    };

    int Ostart, Oloc;
    mlp1_shard(2 * D, TP, r, Ostart, Oloc); // local columns for MLP1
    int Istart, Kloc;
    mlp2_shard_input(D, TP, r, Istart, Kloc); // local input cols for MLP2

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
    size_t b1_local = (size_t)p->n_layers * E * (size_t)Oloc;
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, b1_local * sizeof(__hip_bfloat16)));

    // MLP2 weight shard: [layers, experts, H, Kloc]
    size_t mlp2_local = (size_t)p->n_layers * E * (size_t)H * Kloc;
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2, mlp2_local * sizeof(__hip_bfloat16)));

    // MLP2 bias kept FULL (H), but will be scaled by 1/TP on copy: [layers, experts, H]
    size_t b2_full = (size_t)p->n_layers * E * (size_t)H;
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2, b2_full * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->out, p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->attn_sinks, p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));
}


void malloc_gpu_weights_120b(GPUTransformerWeights *w, Config *p)
{
    // TP group-local rank from current device
    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    const int TP = TENSOR_PARALLEL_SIZE;
    const int r  = dev % TP;

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int L = p->n_layers;

    // Unchanged (attention/embs/norm/router/out)
    HIP_CHECK(hipMalloc((void **)&w->token_embedding_table, (size_t)p->vocab_size * p->hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&w->rms_attn_w, (size_t)p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->rms_ffn_w, (size_t)p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->rms_out_w,  (size_t)p->hidden_dim * sizeof(__hip_bfloat16)));

    const size_t qkv_size = (size_t)p->n_layers * p->hidden_dim * (p->n_attn_heads + 2*p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_qkv, qkv_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_qkv, (size_t)p->n_layers * (p->n_attn_heads + 2*p->n_kv_heads) * p->head_dim * sizeof(__hip_bfloat16)));

    const size_t attn_out_size = (size_t)p->n_layers * (size_t)(p->n_attn_heads * p->head_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&w->w_o, attn_out_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_o, (size_t)p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->w_router, (size_t)p->n_layers * p->hidden_dim * p->n_experts * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_router, (size_t)p->n_layers * p->n_experts * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&w->out, (size_t)p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->attn_sinks, (size_t)p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));

    // ====== Expert-Parallel MoE weights: local experts only, full dense dims ======
    // Partition experts contiguously across TP ranks
    const int base = E / TP, rem = E % TP;
    const int E_loc   = base + (r < rem ? 1 : 0);
    const int E_start = r * base + (r < rem ? r : rem);

    // MLP1: [L, E_loc, 2D, H]  (row-major by 2D then H)
    const size_t mlp1_loc_elems = (size_t)L * (size_t)E_loc * (size_t)(2*D) * (size_t)H;
    const size_t mlp1_loc_packed = (mlp1_loc_elems + 1) / 2;   // nibbles packed
    const size_t mlp1_loc_scales = (mlp1_loc_elems + 31) / 32; // e8m0 per 32

    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_mxfp4, mlp1_loc_packed * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_scales, mlp1_loc_scales * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp1_scales_f32, ((mlp1_loc_elems + 31) / 32) * sizeof(float)));

    // MLP2: [L, E_loc, H, D]
    const size_t mlp2_loc_elems = (size_t)L * (size_t)E_loc * (size_t)H * (size_t)D;
    const size_t mlp2_loc_packed = (mlp2_loc_elems + 1) / 2;
    const size_t mlp2_loc_scales = (mlp2_loc_elems + 31) / 32;

    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_mxfp4, mlp2_loc_packed * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_scales, mlp2_loc_scales * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&w->w_mlp2_scales_f32, ((mlp2_loc_elems + 31) / 32) * sizeof(float)));

    // Biases: BF16, local experts only
    HIP_CHECK(hipMalloc((void **)&w->b_mlp1, (size_t)L * (size_t)E_loc * (size_t)(2*D) * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&w->b_mlp2, (size_t)L * (size_t)E_loc * (size_t)H     * sizeof(__hip_bfloat16)));
}

static inline void convert_float_array_to_bfloat16_scaled(const float *src, __hip_bfloat16 *dst, size_t n, float scale)
{
    std::vector<float> tmp(n);
    for (size_t i = 0; i < n; ++i)
        tmp[i] = src[i] * scale;
    convert_float_array_to_bfloat16(tmp.data(), dst, n);
}

void copy_weights_to_gpu_20b(Transformer *transformer, GPUTransformerWeights *gpu_weights)
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
        int dev = 0;
        HIP_CHECK(hipGetDevice(&dev));
        const int group_base = (dev / TP) * TP;
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
            const int rem = twoD % TP;
            Oloc = base + (rank_in_group < rem ? 1 : 0);
            Ostart = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }
        // MLP2 row-parallel shard (input D)
        int Kloc, Istart;
        {
            const int base = D / TP;
            const int rem = D % TP;
            Kloc = base + (rank_in_group < rem ? 1 : 0);
            Istart = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }

        // Host pinned BF16 staging buffers (small; allocate -> use -> free)
        const size_t max_w_stage = std::max((size_t)Oloc * (size_t)H, (size_t)H * (size_t)Kloc);
        const size_t max_b_stage = std::max((size_t)Oloc, (size_t)H);

        __hip_bfloat16 *staging_w = nullptr;
        __hip_bfloat16 *staging_b = nullptr;
        HIP_CHECK(hipHostMalloc((void **)&staging_w, max_w_stage * sizeof(__hip_bfloat16)));
        HIP_CHECK(hipHostMalloc((void **)&staging_b, max_b_stage * sizeof(__hip_bfloat16)));

        // ----- convenience strides for device shards -----
        const size_t seg1_loc = (size_t)Oloc * (size_t)H; // per-expert stride for w_mlp1 shard
        const size_t seg2_loc = (size_t)H * (size_t)Kloc; // per-expert stride for w_mlp2 shard
        const size_t segb1 = (size_t)Oloc;                // per-expert stride for b_mlp1 shard
        const size_t segb2 = (size_t)H;                   // per-expert stride for full b_mlp2

        // ------------------ pack & copy MLP1 weight shard ------------------
        // Source layout (CPU full): [layers, experts, 2D, H] row-major by (O, then H)
        // Dest layout (GPU shard):  [layers, experts, Oloc, H]
        {
            const size_t seg1_full = (size_t)(2 * D) * (size_t)H; // CPU per-expert stride
            for (int L = 0; L < p->n_layers; ++L)
            {
                for (int e = 0; e < E; ++e)
                {
                    const float *base = w->w_mlp1 + (size_t)L * E * seg1_full + (size_t)e * seg1_full;

                    // Build BF16 shard row-by-row in staging_w
                    for (int o = 0; o < Oloc; ++o)
                    {
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
            for (int L = 0; L < p->n_layers; ++L)
            {
                for (int e = 0; e < E; ++e)
                {
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
            for (int L = 0; L < p->n_layers; ++L)
            {
                for (int e = 0; e < E; ++e)
                {
                    const float *base = w->w_mlp2 + (size_t)L * E * seg2_full + (size_t)e * seg2_full;

                    // Build BF16 shard row-by-row in staging_w
                    for (int h = 0; h < H; ++h)
                    {
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
            for (int L = 0; L < p->n_layers; ++L)
            {
                for (int e = 0; e < E; ++e)
                {
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


__global__ void e8m0_to_f32_kernel(const uint8_t *__restrict__ e8,
                                   float *__restrict__ f, size_t nblocks)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nblocks)
    {
        // X = 2^(e8-127) via bit-cast exponent (fast!)
        f[i] = __uint_as_float(uint32_t(e8[i]) << 23);
    }
}
static inline void launch_e8m0_to_f32(const uint8_t *e8, float *f, size_t n,
                                      hipStream_t s = 0)
{
    if (!n)
        return;
    const int T = 256;
    dim3 grid((int)((n + T - 1) / T));
    e8m0_to_f32_kernel<<<grid, T, 0, s>>>(e8, f, n);
    HIP_CHECK(hipGetLastError());
}

static void streaming_quantize_copy_mxfp4_cpu_to_gpu(
    const float *__restrict__ src,
    size_t src_elems,
    uint8_t *__restrict__ dst_packed_dev,
    uint8_t *__restrict__ dst_scales_dev,
    hipStream_t /*unused_stream_ok*/)
{
    if (src_elems == 0)
        return;

    // Work in 32-sized blocks
    const size_t TOTAL_BLOCKS = (src_elems + 31) / 32;

    // Query free VRAM just to be polite, but keep small caps to avoid OOM/paging.
    size_t free_b = 0, total_b = 0;
    (void)hipMemGetInfo(&free_b, &total_b);

    // ---- Tunables (MI250X-friendly) ----
    // 128–256 MiB chunks usually win on big models + 2 GPUs.
    const size_t CAP_HOST_STAGE_BYTES = (size_t)(256ULL << 20); // 256 MiB
    const size_t CAP_DEV_STAGE_BYTES = (size_t)(256ULL << 20);  // 256 MiB
    // Stay well below free to leave headroom for other allocations
    const size_t DEV_LIMIT = (free_b > (size_t)(2ULL << 30)) ? (size_t)(free_b / 6) : (size_t)(128ULL << 20);
    const size_t STAGE_BYTES = std::min({CAP_HOST_STAGE_BYTES, CAP_DEV_STAGE_BYTES, DEV_LIMIT});

    // Compute block count per chunk
    const size_t BYTES_PER_BLOCK = 32 * sizeof(float);
    size_t chunk_blocks = STAGE_BYTES / BYTES_PER_BLOCK;
    if (chunk_blocks == 0)
        chunk_blocks = 16384; // 2 MB minimum
    if (chunk_blocks > TOTAL_BLOCKS)
        chunk_blocks = TOTAL_BLOCKS;
    const size_t chunk_elems = chunk_blocks * 32;
    const size_t chunk_bytes = chunk_elems * sizeof(float);

    // ---- Streams & events ----
    hipStream_t copy_stream, compute_stream;
    HIP_CHECK(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));

    hipEvent_t copy_done[2], compute_done[2];
    HIP_CHECK(hipEventCreateWithFlags(&copy_done[0], hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&copy_done[1], hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&compute_done[0], hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&compute_done[1], hipEventDisableTiming));

    // ---- Host pinned staging (double buffer) ----
    void *h_stage_raw[2] = {nullptr, nullptr};
    float *h_stage[2] = {nullptr, nullptr};
    HIP_CHECK(hipHostMalloc(&h_stage_raw[0], chunk_bytes, hipHostMallocPortable));
    HIP_CHECK(hipHostMalloc(&h_stage_raw[1], chunk_bytes, hipHostMallocPortable));
    h_stage[0] = reinterpret_cast<float *>(h_stage_raw[0]);
    h_stage[1] = reinterpret_cast<float *>(h_stage_raw[1]);

    // ---- Device staging (double buffer) ----
    float *d_stage[2] = {nullptr, nullptr};
    HIP_CHECK(hipMalloc((void **)&d_stage[0], chunk_bytes));
    HIP_CHECK(hipMalloc((void **)&d_stage[1], chunk_bytes));

    // Kernel launch config
    constexpr int BLOCK_THREADS = 256;
    constexpr int LANES = 32;
    const int WPB = BLOCK_THREADS / LANES;
    const size_t shmem_bytes = (size_t)WPB * (32 * sizeof(float) + 32 * sizeof(uint8_t)) + WPB * sizeof(int);

    size_t blocks_done = 0;
    int ping = 0;

    // Preload first chunk to pinned host buffer (sync memcpy on host memory bus)
    {
        const size_t elem_off = 0;
        const size_t this_elems = std::min(chunk_elems, src_elems - elem_off);
        memcpy(h_stage[ping], src + elem_off, this_elems * sizeof(float));
    }

    while (blocks_done < TOTAL_BLOCKS)
    {
        // double start_time = get_time_msec();
        const size_t elem_off = blocks_done * 32;
        const size_t this_elems = std::min(chunk_elems, src_elems - elem_off);
        const size_t this_blocks = (this_elems + 31) / 32;
        const size_t this_bytes = this_elems * sizeof(float);

        // Start H2D for current ping buffer
        HIP_CHECK(hipMemcpyAsync(d_stage[ping], h_stage[ping], this_bytes,
                                 hipMemcpyHostToDevice, copy_stream));
        HIP_CHECK(hipEventRecord(copy_done[ping], copy_stream));

        // While copy ping is in flight, prepare the NEXT chunk into the other host buffer (pong)
        const int pong = ping ^ 1;
        const size_t next_elem_off = elem_off + this_elems;
        if (next_elem_off < src_elems)
        {
            const size_t next_elems = std::min(chunk_elems, src_elems - next_elem_off);
            // CPU memcpy to pinned host buffer (overlaps with SDMA H2D + prior compute)
            memcpy(h_stage[pong], src + next_elem_off, next_elems * sizeof(float));
        }

        // Compute on ping after its H2D is done
        HIP_CHECK(hipStreamWaitEvent(compute_stream, copy_done[ping], 0));

        dim3 grid((unsigned)((this_blocks + WPB - 1) / WPB));
        dim3 block(BLOCK_THREADS);

        const size_t global_block0 = blocks_done;

        hipLaunchKernelGGL((quantize_pack_mxfp4_block32_kernel<BLOCK_THREADS>),
                           grid, block, shmem_bytes, compute_stream,
                           d_stage[ping],
                           this_blocks,
                           global_block0,
                           dst_packed_dev,
                           dst_scales_dev,
                           src_elems);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(compute_done[ping], compute_stream));

        // Flip buffers for next loop
        blocks_done += this_blocks;
        ping ^= 1;

        // Ensure we don't overwrite buffers still in use (rare with two buffers, but safe)
        HIP_CHECK(hipEventSynchronize(compute_done[ping]));

      
        fflush(stdout);
    }

    HIP_CHECK(hipStreamSynchronize(copy_stream));
    HIP_CHECK(hipStreamSynchronize(compute_stream));

    // Cleanup
    HIP_CHECK(hipFree(d_stage[0]));
    HIP_CHECK(hipFree(d_stage[1]));
    HIP_CHECK(hipHostFree(h_stage_raw[0]));
    HIP_CHECK(hipHostFree(h_stage_raw[1]));
    HIP_CHECK(hipEventDestroy(copy_done[0]));
    HIP_CHECK(hipEventDestroy(copy_done[1]));
    HIP_CHECK(hipEventDestroy(compute_done[0]));
    HIP_CHECK(hipEventDestroy(compute_done[1]));
    HIP_CHECK(hipStreamDestroy(copy_stream));
    HIP_CHECK(hipStreamDestroy(compute_stream));
}

static inline long long tick_msec()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

static size_t getenv_stage_mb_or(const char *name, size_t def_mb)
{
    const char *s = std::getenv(name);
    if (!s)
        return def_mb;
    long v = std::strtol(s, nullptr, 10);
    if (v <= 0)
        return def_mb;
    return (size_t)v;
}

// Parallel pack of rows from src (pageable) into a pinned tile.
// Each row copies 'row_len' floats starting every 'row_stride' floats.
// Work is split into N threads on whole-row boundaries.
static void pack_rows_mt(float *__restrict__ dst,
                         const float *__restrict__ src_base,
                         size_t start_row,
                         size_t n_rows,
                         size_t row_stride,
                         size_t row_len,
                         int n_threads)
{
    if (n_rows == 0)
        return;
    if (n_threads < 2)
    {
        // single-thread fallback
        const float *src = src_base + start_row * row_stride;
        for (size_t r = 0; r < n_rows; ++r)
        {
            memcpy(dst + r * row_len, src + r * row_stride, row_len * sizeof(float));
        }
        return;
    }

    auto worker = [&](size_t r0, size_t r1)
    {
        const float *src = src_base + (start_row + r0) * row_stride;
        float *out = dst + r0 * row_len;
        for (size_t r = r0; r < r1; ++r)
        {
            memcpy(out, src, row_len * sizeof(float));
            src += row_stride;
            out += row_len;
        }
    };

    const size_t chunk = (n_rows + (size_t)n_threads - 1) / (size_t)n_threads;
    std::vector<std::thread> th;
    th.reserve(n_threads);
    size_t r = 0;
    for (int t = 0; t < n_threads && r < n_rows; ++t)
    {
        size_t r1 = std::min(n_rows, r + chunk);
        th.emplace_back(worker, r, r1);
        r = r1;
    }
    for (auto &tt : th)
        tt.join();
}

static void streaming_quantize_copy_mxfp4_cpu_to_gpu_strided(
    const float *__restrict__ src_base, // host pageable
    size_t rows,                        // #segments to gather
    size_t row_stride,                  // in floats
    size_t row_len,                     // in floats
    uint8_t *__restrict__ dst_packed_dev,
    uint8_t *__restrict__ dst_scales_dev,
    hipStream_t /*unused_stream_ok*/ = nullptr)
{
    if (rows == 0 || row_len == 0)
        return;

    // === Tile size heuristics ===
    size_t free_b = 0, total_b = 0;
    (void)hipMemGetInfo(&free_b, &total_b);

    // env knob (MB)
    const size_t ENV_MB = getenv_stage_mb_or("MXFP4_STAGE_MB", /*default*/ 1024);
    const size_t CAP_HOST_STAGE_BYTES = ENV_MB * (size_t)(1ULL << 20); // MB -> bytes
    // Leave headroom in VRAM; if low, cap to free/4
    const size_t DEV_LIMIT = (free_b > (size_t)(4ULL << 30)) ? (size_t)(free_b / 4) : (size_t)(256ULL << 20);
    const size_t STAGE_BYTES = std::min(CAP_HOST_STAGE_BYTES, DEV_LIMIT);

    const size_t BYTES_PER_ROW = row_len * sizeof(float);
    size_t tile_rows = STAGE_BYTES / std::max<size_t>(BYTES_PER_ROW, 1);
    if (tile_rows == 0)
        tile_rows = 1;
    if (tile_rows > rows)
        tile_rows = rows;

    const size_t tile_elems = tile_rows * row_len;
    const size_t tile_bytes = tile_elems * sizeof(float);

    // === Triple-buffered streams/events ===
    hipStream_t s_copy, s_comp;
    HIP_CHECK(hipStreamCreateWithFlags(&s_copy, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&s_comp, hipStreamNonBlocking));

    hipEvent_t e_copy[3], e_comp[3];
    for (int i = 0; i < 3; ++i)
    {
        HIP_CHECK(hipEventCreateWithFlags(&e_copy[i], hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&e_comp[i], hipEventDisableTiming));
    }

    // === Triple host pinned + device staging ===
    void *h_stage_raw[3] = {nullptr, nullptr, nullptr};
    float *h_stage[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i)
    {
        HIP_CHECK(hipHostMalloc(&h_stage_raw[i], tile_bytes, hipHostMallocPortable));
        h_stage[i] = reinterpret_cast<float *>(h_stage_raw[i]);
    }
    float *d_stage[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i)
    {
        HIP_CHECK(hipMalloc((void **)&d_stage[i], tile_bytes));
    }

    // === Kernel config ===
    constexpr int BLOCK_THREADS = 1024; // higher occupancy; shared mem still tiny
    constexpr int LANES = 32;
    const int WPB = BLOCK_THREADS / LANES;
    const size_t shmem_bytes =
        (size_t)WPB * (32 * sizeof(float) + 32 * sizeof(uint8_t)) + WPB * sizeof(int);

    // === Parallel pack threads ===
    const int n_threads = std::max(2, (int)std::thread::hardware_concurrency()); // use all cores

    size_t rows_done = 0;
    size_t blocks_done = 0; // measured in 32-elem blocks within this shard
    int buf = 0;

    // Pre-pack first two tiles so that copy+compute can overlap ASAP
    const size_t warm_tiles = std::min<size_t>(2, (rows + tile_rows - 1) / tile_rows);
    for (size_t w = 0; w < warm_tiles; ++w)
    {
        const size_t r0 = rows_done + w * tile_rows;
        const size_t rN = std::min(tile_rows, rows - r0);
        pack_rows_mt(h_stage[(buf + w) % 3], src_base, r0, rN, row_stride, row_len, n_threads);
    }

    while (rows_done < rows)
    {
        const long long t0 = tick_msec();

        const size_t r_this = std::min(tile_rows, rows - rows_done);
        const size_t this_elems = r_this * row_len;
        const size_t this_bytes = this_elems * sizeof(float);
        const size_t this_blocks = (this_elems + 31) / 32;

        // ---- H2D on current buffer ----
        HIP_CHECK(hipMemcpyAsync(d_stage[buf], h_stage[buf], this_bytes,
                                 hipMemcpyHostToDevice, s_copy));
        HIP_CHECK(hipEventRecord(e_copy[buf], s_copy));

        // ---- Pack the next tile (in parallel) while copy/compute overlap ----
        const size_t r_next0 = rows_done + r_this;
        if (r_next0 < rows)
        {
            const size_t r_nextN = std::min(tile_rows, rows - r_next0);
            const int next_buf = (buf + 1) % 3;
            const long long tp0 = tick_msec();
            pack_rows_mt(h_stage[next_buf], src_base, r_next0, r_nextN, row_stride, row_len, n_threads);
            const long long tp1 = tick_msec();
            // (Optional) keep this lightweight; detailed per-tile print below
            (void)tp0;
            (void)tp1;
        }

        // ---- Compute when H2D is done ----
        HIP_CHECK(hipStreamWaitEvent(s_comp, e_copy[buf], 0));

        dim3 grid((unsigned)((this_blocks + WPB - 1) / WPB));
        dim3 block(BLOCK_THREADS);

        const size_t global_block0 = blocks_done;

        const long long tc0 = tick_msec();
        hipLaunchKernelGGL((quantize_pack_mxfp4_block32_kernel<BLOCK_THREADS>),
                           grid, block, shmem_bytes, s_comp,
                           d_stage[buf],
                           this_blocks,
                           global_block0,
                           dst_packed_dev,
                           dst_scales_dev,
                           rows * row_len); // shard_elems
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(e_comp[buf], s_comp));

        // ---- Advance ----
        blocks_done += this_blocks;
        rows_done += r_this;

        // ---- Optional: detailed per-tile timing (pack/H2D/ker) ----
        HIP_CHECK(hipEventSynchronize(e_comp[buf]));
        const long long t3 = tick_msec();

        // We can approximate H2D+kernel time as (t3 - t0) minus pack time of the next tile,
        // but to keep overhead low, we report wall time and effective "GB/s on bytes moved".
        const double gb = (double)this_bytes / (1024.0 * 1024.0 * 1024.0);
        const double ms = (double)(t3 - t0);
        printf("[streaming quantize strided] tile %5zu rows, %7.2f ms, %7.2f GB/s\n",
               r_this, ms, (ms > 0.0 ? gb * 1000.0 / ms : 0.0));

        // ---- Move to next buffer, ensure it's not in use ----
        buf = (buf + 1) % 3;
        HIP_CHECK(hipEventSynchronize(e_comp[buf])); // make sure the next buffer is free
    }

    HIP_CHECK(hipStreamSynchronize(s_copy));
    HIP_CHECK(hipStreamSynchronize(s_comp));

    for (int i = 0; i < 3; ++i)
    {
        HIP_CHECK(hipFree(d_stage[i]));
        HIP_CHECK(hipHostFree(h_stage_raw[i]));
        HIP_CHECK(hipEventDestroy(e_copy[i]));
        HIP_CHECK(hipEventDestroy(e_comp[i]));
    }
    HIP_CHECK(hipStreamDestroy(s_copy));
    HIP_CHECK(hipStreamDestroy(s_comp));
}

// Convert device-side E8M0 scale bytes into float multipliers.
// Each block of 32 MXFP4 weights has one scale byte (E8M0).
// The f32 expansion is stored in gpu_weights->w_mlp{1,2}_scales_f32.
static void convert_all_scales_to_f32(GPUTransformerWeights *w, Config *p, hipStream_t stream = 0)
{
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    const int TP = TENSOR_PARALLEL_SIZE;
    const int rank = dev % TP;

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int L = p->n_layers;

    // ---- MLP1: column-parallel on output (2D) ----
    int Ostart, Oloc;
    {
        const int twoD = 2 * D;
        const int base = twoD / TP, rem = twoD % TP;
        Oloc = base + (rank < rem ? 1 : 0);
        Ostart = rank * base + (rank < rem ? rank : rem);
    }
    const size_t seg1_blocks_loc = ((size_t)Oloc * (size_t)H + 31) / 32;
    const size_t total_blocks_mlp1 = (size_t)L * (size_t)E * seg1_blocks_loc;
    if (total_blocks_mlp1)
    {
        launch_e8m0_to_f32(w->w_mlp1_scales, w->w_mlp1_scales_f32,
                           total_blocks_mlp1, stream);
    }

    // ---- MLP2: row-parallel on input (D) ----
    int Istart, Kloc;
    {
        const int base = D / TP, rem = D % TP;
        Kloc = base + (rank < rem ? 1 : 0);
        Istart = rank * base + (rank < rem ? rank : rem);
    }
    const size_t seg2_blocks_loc = ((size_t)H * (size_t)Kloc + 31) / 32;
    const size_t total_blocks_mlp2 = (size_t)L * (size_t)E * seg2_blocks_loc;
    if (total_blocks_mlp2)
    {
        launch_e8m0_to_f32(w->w_mlp2_scales, w->w_mlp2_scales_f32,
                           total_blocks_mlp2, stream);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    // printf("[MXFP4] Converted %zu MLP1 + %zu MLP2 scale blocks to f32\n",
    //        total_blocks_mlp1, total_blocks_mlp2);
}
void copy_weights_to_gpu_120b(Transformer *transformer, GPUTransformerWeights *gpu_weights)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    init_mxfp4_lut_on_device();

    // ===== unchanged copies =====
    {
        const size_t embed = (size_t)p->vocab_size * p->hidden_dim;
        HIP_CHECK(hipMemcpy(gpu_weights->token_embedding_table, w->token_embedding_table, embed * sizeof(float), hipMemcpyHostToDevice));

        const size_t normL = (size_t)p->n_layers * p->hidden_dim;
        __hip_bfloat16 *tmp = (__hip_bfloat16*)malloc(normL * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->rms_attn_w, tmp, normL);
        HIP_CHECK(hipMemcpy(gpu_weights->rms_attn_w, tmp, normL * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        convert_float_array_to_bfloat16(w->rms_ffn_w, tmp, normL);
        HIP_CHECK(hipMemcpy(gpu_weights->rms_ffn_w, tmp, normL * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tmp);

        __hip_bfloat16 *t1 = (__hip_bfloat16*)malloc(p->hidden_dim * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->rms_out_w, t1, p->hidden_dim);
        HIP_CHECK(hipMemcpy(gpu_weights->rms_out_w, t1, p->hidden_dim * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(t1);

        const size_t qkv = (size_t)p->n_layers * p->hidden_dim * (p->n_attn_heads + 2*p->n_kv_heads) * p->head_dim;
        __hip_bfloat16 *tqkv = (__hip_bfloat16*)malloc(qkv * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_qkv, tqkv, qkv);
        HIP_CHECK(hipMemcpy(gpu_weights->w_qkv, tqkv, qkv * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tqkv);

        const size_t bqkv = (size_t)p->n_layers * (p->n_attn_heads + 2*p->n_kv_heads) * p->head_dim;
        __hip_bfloat16 *tbq = (__hip_bfloat16*)malloc(bqkv * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_qkv, tbq, bqkv);
        HIP_CHECK(hipMemcpy(gpu_weights->b_qkv, tbq, bqkv * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tbq);

        const size_t wo = (size_t)p->n_layers * (size_t)(p->n_attn_heads * p->head_dim) * p->hidden_dim;
        __hip_bfloat16 *two = (__hip_bfloat16*)malloc(wo * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_o, two, wo);
        HIP_CHECK(hipMemcpy(gpu_weights->w_o, two, wo * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(two);

        const size_t bo = (size_t)p->n_layers * p->hidden_dim;
        __hip_bfloat16 *tbo = (__hip_bfloat16*)malloc(bo * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_o, tbo, bo);
        HIP_CHECK(hipMemcpy(gpu_weights->b_o, tbo, bo * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tbo);

        const size_t wr = (size_t)p->n_layers * p->hidden_dim * p->n_experts;
        __hip_bfloat16 *twr = (__hip_bfloat16*)malloc(wr * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->w_router, twr, wr);
        HIP_CHECK(hipMemcpy(gpu_weights->w_router, twr, wr * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(twr);

        const size_t br = (size_t)p->n_layers * p->n_experts;
        __hip_bfloat16 *tbr = (__hip_bfloat16*)malloc(br * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->b_router, tbr, br);
        HIP_CHECK(hipMemcpy(gpu_weights->b_router, tbr, br * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tbr);

        const size_t sinks = (size_t)p->n_layers * p->n_attn_heads;
        __hip_bfloat16 *ts = (__hip_bfloat16*)malloc(sinks * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->attn_sinks, ts, sinks);
        HIP_CHECK(hipMemcpy(gpu_weights->attn_sinks, ts, sinks * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(ts);

        const size_t outsz = (size_t)p->hidden_dim * p->vocab_size;
        __hip_bfloat16 *tout = (__hip_bfloat16*)malloc(outsz * sizeof(__hip_bfloat16));
        convert_float_array_to_bfloat16(w->out, tout, outsz);
        HIP_CHECK(hipMemcpy(gpu_weights->out, tout, outsz * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        free(tout);
    }

    // ===== Expert-parallel MXFP4 pack (local experts only) =====
    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    const int TP = TENSOR_PARALLEL_SIZE;
    const int r  = dev % TP;

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int L = p->n_layers;

    const int base = E / TP, rem = E % TP;
    const int E_loc   = base + (r < rem ? 1 : 0);
    const int E_start = r * base + (r < rem ? r : rem);

    // MLP1: source full layout [L,E,2D,H]
    const size_t per_exp_mlp1 = (size_t)(2*D) * (size_t)H;
    size_t pack_off = 0, scale_off = 0;
    for (int l = 0; l < L; ++l) {
        const float* layer_base = transformer->weights.w_mlp1 + (size_t)l * (size_t)E * per_exp_mlp1;
        // Gather E_loc experts contiguous starting from E_start
        const float* src_block = layer_base + (size_t)E_start * per_exp_mlp1;
        const size_t elems_layer_loc = (size_t)E_loc * per_exp_mlp1;
        const size_t packed = (elems_layer_loc + 1) / 2;
        const size_t scales = (elems_layer_loc + 31) / 32;

        streaming_quantize_copy_mxfp4_cpu_to_gpu(
            src_block, elems_layer_loc,
            gpu_weights->w_mlp1_mxfp4 + pack_off,
            gpu_weights->w_mlp1_scales + scale_off,
            nullptr);
        pack_off  += packed;
        scale_off += scales;
    }

    // MLP1 bias [L,E,2D] → local slice
    {
        __hip_bfloat16* staging = (__hip_bfloat16*)malloc((size_t)(2*D) * sizeof(__hip_bfloat16));
        size_t dst_off = 0;
        for (int l = 0; l < L; ++l) {
            const float* layer_b = transformer->weights.b_mlp1 + (size_t)l * (size_t)E * (size_t)(2*D)
                                 + (size_t)E_start * (size_t)(2*D);
            // copy E_loc rows of length 2D
            for (int e = 0; e < E_loc; ++e) {
                convert_float_array_to_bfloat16(layer_b + (size_t)e * (size_t)(2*D),
                                                staging, (size_t)(2*D));
                HIP_CHECK(hipMemcpy(gpu_weights->b_mlp1 + dst_off,
                                    staging, (size_t)(2*D) * sizeof(__hip_bfloat16),
                                    hipMemcpyHostToDevice));
                dst_off += (size_t)(2*D);
            }
        }
        free(staging);
    }

    // MLP2: source full layout [L,E,H,D]
    const size_t per_exp_mlp2 = (size_t)H * (size_t)D;
    pack_off = 0; scale_off = 0;
    for (int l = 0; l < L; ++l) {
        const float* layer_base = transformer->weights.w_mlp2 + (size_t)l * (size_t)E * per_exp_mlp2;
        const float* src_block  = layer_base + (size_t)E_start * per_exp_mlp2;
        const size_t elems_layer_loc = (size_t)E_loc * per_exp_mlp2;
        const size_t packed = (elems_layer_loc + 1) / 2;
        const size_t scales = (elems_layer_loc + 31) / 32;

        streaming_quantize_copy_mxfp4_cpu_to_gpu(
            src_block, elems_layer_loc,
            gpu_weights->w_mlp2_mxfp4 + pack_off,
            gpu_weights->w_mlp2_scales + scale_off,
            nullptr);
        pack_off  += packed;
        scale_off += scales;
    }

    // MLP2 bias [L,E,H] → local slice (no 1/TP scaling in EP)
    {
        __hip_bfloat16* staging = (__hip_bfloat16*)malloc((size_t)H * sizeof(__hip_bfloat16));
        size_t dst_off = 0;
        for (int l = 0; l < L; ++l) {
            const float* layer_b = transformer->weights.b_mlp2 + (size_t)l * (size_t)E * (size_t)H
                                 + (size_t)E_start * (size_t)H;
            for (int e = 0; e < E_loc; ++e) {
                convert_float_array_to_bfloat16(layer_b + (size_t)e * (size_t)H,
                                                staging, (size_t)H);
                HIP_CHECK(hipMemcpy(gpu_weights->b_mlp2 + dst_off,
                                    staging, (size_t)H * sizeof(__hip_bfloat16),
                                    hipMemcpyHostToDevice));
                dst_off += (size_t)H;
            }
        }
        free(staging);
    }

    // Convert e8m0 scales -> f32 for both MLPs
    // replace this buggy upper-bound section in copy_weights_to_gpu_120b(...)
    {
        // WRONG (can overrun): uses (E/TP + 1)
        // const size_t n1_blocks = ( (size_t)p->n_layers * (size_t)(E / TP + 1) * (size_t)(2*D) * (size_t)H + 31 ) / 32;
        // const size_t n2_blocks = ( (size_t)p->n_layers * (size_t)(E / TP + 1) * (size_t)H * (size_t)D + 31 ) / 32;
        // launch_e8m0_to_f32(...);

        // CORRECT: compute the local expert slice exactly
        int dev = 0; HIP_CHECK(hipGetDevice(&dev));
        const int TP = TENSOR_PARALLEL_SIZE;
        const int r  = dev % TP;

        const int base = E / TP, rem = E % TP;
        const int E_loc = base + (r < rem ? 1 : 0);

        const size_t n1_blocks =
            ((size_t)p->n_layers * (size_t)E_loc * (size_t)(2*D) * (size_t)H + 31) / 32;
        const size_t n2_blocks =
            ((size_t)p->n_layers * (size_t)E_loc * (size_t)H * (size_t)D + 31) / 32;

        launch_e8m0_to_f32(gpu_weights->w_mlp1_scales,
                        gpu_weights->w_mlp1_scales_f32,
                        n1_blocks, 0);
        launch_e8m0_to_f32(gpu_weights->w_mlp2_scales,
                        gpu_weights->w_mlp2_scales_f32,
                        n2_blocks, 0);
        HIP_CHECK(hipDeviceSynchronize());
    }


    printf("120B MoE weights copied (Expert-Parallel, local E slice, MXFP4).\n");
}

void malloc_cpu_buffers(CPUBuffers *cpu_buf, Config *p)
{
    const int TP = TENSOR_PARALLEL_SIZE;
    const int H = p->hidden_dim;
    const int Bgrp_max = TP * BATCH_SIZE;

    // Host-only
    cpu_buf->cos_vals = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));
    cpu_buf->sin_vals = (float *)malloc((p->head_dim / 2) * MAX_SEQ_LEN * sizeof(float));
    cpu_buf->prompt_lens = (int *)malloc(BATCH_SIZE * sizeof(int));
    cpu_buf->finished = (bool *)malloc(BATCH_SIZE * sizeof(bool));

    // Prompt token storage per slot
    cpu_buf->prompt_tokens = (int **)malloc(BATCH_SIZE * sizeof(int *));
    for (int b = 0; b < BATCH_SIZE; ++b)
    {
        cpu_buf->prompt_tokens[b] = (int *)malloc((MAX_SEQ_LEN + 3) * sizeof(int));
    }

    // Pinned buffers
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->positions, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->slot_active_cpu, BATCH_SIZE * sizeof(bool)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->seq_lengths_cpu, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->request_mapping_cpu, BATCH_SIZE * sizeof(int)));

    // MoE host buffers
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->expert_counts, p->n_experts * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->expert_offsets, p->n_experts * sizeof(int)));

    // Tile maps (host)
    const int total_mtiles = (int)((size_t)BATCH_SIZE * p->n_experts);
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->h_tile2expert, total_mtiles * sizeof(int)));
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->h_tile2local, total_mtiles * sizeof(int)));

    // Bounce buffer for TP peer copies (largest single block in allgather = Bgrp_max*H)
    cpu_buf->tp_host_stage_bytes = (size_t)Bgrp_max * p->experts_per_token * H * sizeof(float);
    HIP_CHECK(hipHostMalloc((void **)&cpu_buf->tp_host_stage, cpu_buf->tp_host_stage_bytes));

    // Not pinned
    cpu_buf->logits = (float *)malloc((size_t)BATCH_SIZE * p->vocab_size * sizeof(float));

    // Initialize
    std::fill_n(cpu_buf->slot_active_cpu, BATCH_SIZE, false);
    std::fill_n(cpu_buf->seq_lengths_cpu, BATCH_SIZE, 0);
    for (int i = 0; i < BATCH_SIZE; ++i)
        cpu_buf->request_mapping_cpu[i] = -1;
    cpu_buf->next_request_idx = 0;
}

void free_cpu_buffers(CPUBuffers *cpu_buf)
{
    if (cpu_buf->cos_vals)
        free(cpu_buf->cos_vals);
    if (cpu_buf->sin_vals)
        free(cpu_buf->sin_vals);
    if (cpu_buf->prompt_lens)
        free(cpu_buf->prompt_lens);
    if (cpu_buf->finished)
        free(cpu_buf->finished);

    if (cpu_buf->prompt_tokens)
    {
        for (int b = 0; b < BATCH_SIZE; ++b)
        {
            if (cpu_buf->prompt_tokens[b])
                free(cpu_buf->prompt_tokens[b]);
        }
        free(cpu_buf->prompt_tokens);
    }

    if (cpu_buf->logits)
        free(cpu_buf->logits);

    if (cpu_buf->current_tokens)
        HIP_CHECK(hipHostFree(cpu_buf->current_tokens));
    if (cpu_buf->positions)
        HIP_CHECK(hipHostFree(cpu_buf->positions));
    if (cpu_buf->slot_active_cpu)
        HIP_CHECK(hipHostFree(cpu_buf->slot_active_cpu));
    if (cpu_buf->seq_lengths_cpu)
        HIP_CHECK(hipHostFree(cpu_buf->seq_lengths_cpu));
    if (cpu_buf->request_mapping_cpu)
        HIP_CHECK(hipHostFree(cpu_buf->request_mapping_cpu));

    if (cpu_buf->expert_counts)
        HIP_CHECK(hipHostFree(cpu_buf->expert_counts));
    if (cpu_buf->expert_offsets)
        HIP_CHECK(hipHostFree(cpu_buf->expert_offsets));
    if (cpu_buf->h_tile2expert)
        HIP_CHECK(hipHostFree(cpu_buf->h_tile2expert));
    if (cpu_buf->h_tile2local)
        HIP_CHECK(hipHostFree(cpu_buf->h_tile2local));

    if (cpu_buf->tp_host_stage)
        HIP_CHECK(hipHostFree(cpu_buf->tp_host_stage));
}

void build_gpu_transformer(GPUTransformer *gpu_t, Transformer *cpu_t)
{
    if(IS_20B_MODEL){
    gpu_t->config = cpu_t->config;

    malloc_gpu_weights_20b(&gpu_t->weights, &gpu_t->config);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);

    copy_weights_to_gpu_20b(cpu_t, &gpu_t->weights);
    }else{
        gpu_t->config = cpu_t->config;

    malloc_gpu_weights_120b(&gpu_t->weights, &gpu_t->config);
    malloc_gpu_run_state(&gpu_t->state, &gpu_t->config);
    malloc_cpu_buffers(&gpu_t->cpu_buffers, &gpu_t->config);

    copy_weights_to_gpu_120b(cpu_t, &gpu_t->weights);

    }

}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;
    if(p->n_experts == 32) IS_20B_MODEL = 1;
    else IS_20B_MODEL = 0;

    if(IS_20B_MODEL) BATCH_SIZE = 1024, TENSOR_PARALLEL_SIZE = 2;
    else BATCH_SIZE = 896, TENSOR_PARALLEL_SIZE = 4;


    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus > MAX_GPUS)
        num_gpus = MAX_GPUS;

#pragma omp parallel for
    for (int dev = 0; dev < num_gpus; ++dev)
    {
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
    if (w->w_mlp1)
        HIP_CHECK(hipFree(w->w_mlp1));
    if (w->w_mlp2)
        HIP_CHECK(hipFree(w->w_mlp2));

      if (w->w_mlp1_mxfp4)
        HIP_CHECK(hipFree(w->w_mlp1_mxfp4));
    if (w->w_mlp2_mxfp4)
        HIP_CHECK(hipFree(w->w_mlp2_mxfp4));

    if (w->b_mlp1)
        HIP_CHECK(hipFree(w->b_mlp1));
    if (w->b_mlp2)
        HIP_CHECK(hipFree(w->b_mlp2));
    if (w->out)
        HIP_CHECK(hipFree(w->out));
}

void free_gpu_run_state(GPURunState *s)
{
    auto df = [](void *p)
    { if (p) HIP_CHECK(hipFree(p)); };

    df(s->x);
    df(s->t);
    df(s->tb);
    df(s->tb2);
    df(s->temp_buffer);
    df(s->qkv);
    df(s->q);
    df(s->k);
    df(s->v);
    df(s->att);
    df(s->mask);
    df(s->key_cache);
    df(s->value_cache);
    df(s->cos_vals);
    df(s->sin_vals);

    df(s->router_score);

    df(s->d_expert_counts);
    df(s->d_expert_offsets);

    df(s->current_tokens);
    df(s->positions);
    df(s->logits);
    df(s->seq_lengths);
    df(s->slot_active);
    df(s->request_mapping);

    // TP-union
    df(s->gather_x_g);
    df(s->router_score_g);
    df(s->topk_v_g);
    df(s->topk_i_g);
    df(s->local_ids_g);
    df(s->local_wts_g);
    df(s->e_agg_g);
    df(s->expert_input_buffer_g);
    df(s->mlp1_out_g);
    df(s->gate_up_g);
    df(s->expert_output_partial_g);
    df(s->expert_output_gather_g);
    df(s->d_tile2expert_g);
    df(s->d_tile2local_g);
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
    for (int dev = 0; dev < num_gpus; ++dev)
    {
        tp_free(tp_groups[dev]);
        free_gpu_transformer(gpu_transformers[dev]);
        free(gpu_transformers[dev]);
    }
}

// ----------------------------- attention path (unchanged) --------------------------
void attention_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size,
                   int row_offset, hipStream_t sAttn)
{
    if (batch_size <= 0)
        return;

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;

    const int H = p->hidden_dim;
    const int Hd = p->head_dim;
    const int NA = p->n_attn_heads;
    const int NK = p->n_kv_heads;
    const int KV = Hd * NK;
    const int QKV = Hd * (NA + 2 * NK);

    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    size_t layers_capacity = 0;
    for (int L = 0; L < p->n_layers; ++L)
    {
        const bool evenL = ((L & 1) == 0);
        layers_capacity += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t kv_slice = layers_capacity * (size_t)KV;

    size_t layer_pos_offset = 0;
    for (int L = 0; L < layer_idx; ++L)
    {
        const bool evenL = ((L & 1) == 0);
        layer_pos_offset += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t layer_elem_offset = layer_pos_offset * (size_t)KV;

    float *x_mb = s->x + (size_t)row_offset * H;
    float *t_mb = s->t + (size_t)row_offset * H;
    float *tb_mb = s->tb + (size_t)row_offset * (Hd * NA);
    float *qkv_mb = s->qkv + (size_t)row_offset * QKV;
    float *q_mb = s->q + (size_t)row_offset * (Hd * NA);
    float *k_mb = s->k + (size_t)row_offset * KV;
    float *v_mb = s->v + (size_t)row_offset * KV;
    int *pos_mb = s->positions + row_offset;

    __hip_bfloat16 *key_cache_mb = s->key_cache + (size_t)row_offset * kv_slice;
    __hip_bfloat16 *value_cache_mb = s->value_cache + (size_t)row_offset * kv_slice;

    // 1) RMSNorm
    {
        // TIMER_BLOCK("rmsnorm_kernel_attention");
        dim3 grid(batch_size), block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<grid, block, 0, sAttn>>>(t_mb, x_mb,
                                                  w->rms_attn_w + (size_t)layer_idx * H, batch_size, H);
        HIP_CHECK(hipGetLastError());
    }
    // 2) QKV
    {
        // TIMER_BLOCK("matmul_mc_attention");
        const int woff = layer_idx * H * QKV;
        matmul_vec128_singlebuf<
            /*WM,WN,WK*/ 16, 16, 16,
            /*WAVES_M,N,K*/ 4, 4, 4,
            /*TW_M,TW_N*/ 1, 2,
            /*PAD_K*/ 8,
            /*FUSED*/ false>(qkv_mb, t_mb, w->w_qkv + woff, batch_size, H, QKV, nullptr, sAttn);
        HIP_CHECK(hipGetLastError());
    }
    // 3) bias
    {
        // TIMER_BLOCK("add_bias_kernel_attention");
        const int boff = (size_t)layer_idx * QKV;
        const int elems = batch_size * QKV;
        if (elems > 0)
        {
            dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            add_bias_kernel<<<grid, THREADS_PER_BLOCK, 0, sAttn>>>(qkv_mb, w->b_qkv + boff, batch_size, QKV);
            HIP_CHECK(hipGetLastError());
        }
    }
    // 4) split + RoPE
    {
        // TIMER_BLOCK("launch_split_qkv_apply_rotary");
        launch_split_qkv_apply_rotary(
            qkv_mb, q_mb, k_mb, v_mb,
            s->cos_vals, s->sin_vals, pos_mb,
            batch_size, NA, NK, Hd, sAttn);
        HIP_CHECK(hipGetLastError());
    }
    // 5) KV cache update
    {
        // TIMER_BLOCK("update_kv_cache_kernel");
        dim3 grid(batch_size, (KV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block(1, THREADS_PER_BLOCK);
        update_kv_cache_kernel<<<grid, block, 0, sAttn>>>(
            key_cache_mb, value_cache_mb, k_mb, v_mb, pos_mb,
            batch_size, p->n_layers, layer_idx, MAX_SEQ_LEN, KV,
            /* batch_kv_stride = */ layers_capacity * (size_t)KV,
            /* layer_kv_offset = */ layer_elem_offset);
    }
    // 6) fused attention
    // --- optimized launch (tile_t = 48, same policy as launch_optimized) ---
    {
        // TIMER_BLOCK("flashdecoding_fused");
        // --- Fused Flash-Decoding (fast-merge, no K/V staging) launch --------------------
        // Uses 4 warps/block (256 threads). Tune WARPS if needed (e.g., 2 or 6).
        const int B = batch_size;
        const int H = NA;   // n_attn_heads
        const int NKv = NK; // n_kv_heads
        const int D = Hd;

        // GQA ratio (should be 8 for this kernel path)
        const int kv_mul = H / NKv; // == 8
        if (kv_mul != 8 || D != 64)
        {
            // Fall back to your baseline if you want, or assert.
            // For now, just return to avoid a bad launch.
            HIP_CHECK(hipSuccess);
        }
        else
        {
            // Block/grid mapping: multi-warp per (b, kv_h)
            const int WARPS = 4;          // <<< tune knob
            dim3 block(64 * WARPS, 1, 1); // 64 = AMD warp size
            dim3 grid(NKv, B, 1);

            // Shared memory only for fast-merge reduction:
            // (m,l): 2 * WARPS * 8 floats  +  numerators: WARPS * 64 * 8 floats
            const size_t shmem =
                (size_t)((2 * WARPS * 8) + (WARPS * 64 * 8)) * sizeof(float);

            // Request dynamic LDS (ignore return; optional on some stacks)
            (void)hipFuncSetAttribute(
                (const void *)flashdecoding_fused_fastmerge_nostage_1warp8q,
                hipFuncAttributeMaxDynamicSharedMemorySize,
                (int)shmem);

            hipLaunchKernelGGL(
                flashdecoding_fused_fastmerge_nostage_1warp8q,
                grid, block, shmem, sAttn,
                /* output      */ tb_mb,
                /* q           */ q_mb,
                /* key_cache   */ key_cache_mb,
                /* value_cache */ value_cache_mb,
                /* sinks       */ w->attn_sinks + (size_t)layer_idx * NA,
                /* mask        */ s->mask,
                /* seq_lengths */ pos_mb,
                /* B,H,KVH,D   */ batch_size, NA, NKv, Hd,
                /* seq_len,L   */ MAX_SEQ_LEN, p->n_layers,
                /* layer_idx   */ layer_idx,
                /* use_sw      */ p->sliding_window > 0,
                /* strides     */ /* batch_kv_stride = */ layers_capacity * (size_t)KV,
                /* offsets     */ /* layer_kv_offset  = */ layer_elem_offset,
                /* tile_t_unused */ 0);

            HIP_CHECK(hipGetLastError());
        }
    }

    // 7) fused output projection
    {
        // TIMER_BLOCK("fused_output_projection_kernel_optimized");
        const int Kproj = Hd * NA;
        const int N = H;
        const int woff = (size_t)layer_idx * Kproj * H;
        const int boff = (size_t)layer_idx * H;

        matmul_vec128_singlebuf<
            16, 16, 16,
            4, 4, 4,
            1, 2,
            8,
            /*FUSED*/ true>(x_mb, tb_mb, w->w_o + woff, /*M=*/batch_size, /*K=*/Hd * NA, /*N=*/H,
                            /*bias=*/w->b_o + boff, /*stream=*/sAttn);
        HIP_CHECK(hipGetLastError());
    }
}

// Split helpers
static inline void mlp1_shard(int twoD, int tp_size, int r, int &o_start, int &o_len)
{
    const int base = twoD / tp_size;
    const int rem = twoD % tp_size;
    o_len = base + (r < rem ? 1 : 0);
    o_start = r * base + (r < rem ? r : rem);
}
static inline void mlp2_shard_input(int D, int tp_size, int r, int &i_start, int &i_len)
{
    const int base = D / tp_size;
    const int rem = D % tp_size;
    i_len = base + (r < rem ? 1 : 0);
    i_start = r * base + (r < rem ? r : rem);
}

// Sum across rank axis [TP, T, H] -> [T, H]
__global__ void sum_rank_axis_kernel(float *out, const float *parts, int TP, int T, int H)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)T * H;
    if (idx >= total)
        return;
    int t = (int)(idx / H), h = (int)(idx % H);
    float acc = 0.f;
    for (int r = 0; r < TP; ++r)
        acc += parts[((size_t)r * T + t) * H + h];
    out[(size_t)t * H + h] = acc;
}

void moe_gpu(GPUTransformer *gpu_t, int layer_idx, int batch_size,
             int row_offset, hipStream_t sMoe)
{
    if (batch_size <= 0)
        return;

    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu = &gpu_t->cpu_buffers;

    const int TP = TENSOR_PARALLEL_SIZE;
    int my_dev = 0;
    HIP_CHECK(hipGetDevice(&my_dev));
    const int group_base = (my_dev / TP) * TP;
    const int rank_in_group = my_dev - group_base;
    TPGroup &tp = tp_groups[my_dev];

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int K = p->experts_per_token;

    // local microbatch size and TP-union size for this MoE step
    const int bs_local = batch_size;
    const int Bgrp = TP * bs_local;

    // 0) RMSNorm on local MB: t = RMS(x, w_ffn)
    {
        // TIMER_BLOCK("rmsnorm_kernel_moe");
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
        // TIMER_BLOCK("moe_allgather");
        // self copy
        float *dst_self = s->gather_x_g + (size_t)rank_in_group * bs_local * H;
        float *src_self = s->t + (size_t)row_offset * H;
        HIP_CHECK(hipMemcpyAsync(dst_self, src_self,
                                 (size_t)bs_local * H * sizeof(float),
                                 hipMemcpyDeviceToDevice, sMoe));
        // peer copies
        for (int r = 0; r < TP; ++r)
        {
            const int peer_dev = group_base + r;
            if (peer_dev == my_dev)
                continue;
            float *dst = s->gather_x_g + (size_t)r * bs_local * H;
            float *peer_src = gpu_transformers[peer_dev]->state.t + (size_t)row_offset * H;

            if (tp.p2p[rank_in_group][r])
            {
                HIP_CHECK(hipMemcpyPeerAsync(dst, my_dev, peer_src, peer_dev,
                                             (size_t)bs_local * H * sizeof(float), sMoe));
            }
            else
            {
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
    {    
        // TIMER_BLOCK("matmul_mc_router");
        matmul_vec128_singlebuf<
            16,16,16,
            4,4,4,
            1,2,
            8,
            /*FUSED*/ false
        >(s->router_score_g, s->gather_x_g,
            w->w_router + (size_t)layer_idx * H * E,
            /*M=*/Bgrp, /*K=*/H, /*N=*/E,
            /*bias=*/nullptr, /*stream=*/sMoe);
    }
    
    {
        // TIMER_BLOCK("route_select_softmax_fused");
        if (Bgrp > 0 && E > 0)
        {
            run_route_select_softmax_fused(
                /*router_score=*/s->router_score_g,
                /*bias_bf16=*/w->b_router + (size_t)layer_idx * (size_t)E,
                /*B=*/Bgrp, /*E=*/E, /*K=*/K,
                /*topk_v_out=*/s->topk_v_g,
                /*topk_i_out=*/s->topk_i_g,
                /*stream=*/sMoe);
        }
        HIP_CHECK(hipGetLastError());
    }

    // 3) Count tokens per expert (order-independent atomics OK), build offsets host-side
    HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, E * sizeof(int), sMoe));
    {
        // TIMER_BLOCK("count_tokens_per_expert_kernel");
        dim3 grid((Bgrp + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<grid, THREADS_PER_BLOCK, 0, sMoe>>>(
            s->topk_i_g, s->d_expert_counts, Bgrp, K);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(sMoe));
    HIP_CHECK(hipMemcpy(cpu->expert_counts, s->d_expert_counts,
                        E * sizeof(int), hipMemcpyDeviceToHost));

    int total_tokens = 0;
    for (int e = 0; e < E; ++e)
    {
        cpu->expert_offsets[e] = total_tokens;
        total_tokens += cpu->expert_counts[e];
    }
    if (E > 0)
        HIP_CHECK(hipMemcpyAsync(s->d_expert_offsets, cpu->expert_offsets,
                                 E * sizeof(int), hipMemcpyHostToDevice, sMoe));

    // Build tile maps for grouped matmuls (host) and copy to device
    int cur_tiles = 0;
    for (int e = 0; e < E; ++e)
    {
        const int cnt = cpu->expert_counts[e];
        const int tiles = (cnt + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
        for (int m = 0; m < tiles; ++m)
        {
            cpu->h_tile2expert[cur_tiles + m] = e;
            cpu->h_tile2local[cur_tiles + m] = m;
        }
        cur_tiles += tiles;
    }
    if (cur_tiles > s->cap_tiles)
    {
        fprintf(stderr, "[MoE] tiles(%d) > cap_tiles(%d). Increase cap or adjust BLOCK_M_MLP.\n",
                cur_tiles, s->cap_tiles);
        abort();
    }
    if (cur_tiles > 0)
    {
        HIP_CHECK(hipMemcpyAsync(s->d_tile2expert_g, cpu->h_tile2expert,
                                 cur_tiles * sizeof(int), hipMemcpyHostToDevice, sMoe));
        HIP_CHECK(hipMemcpyAsync(s->d_tile2local_g, cpu->h_tile2local,
                                 cur_tiles * sizeof(int), hipMemcpyHostToDevice, sMoe));
    }

    // 4) FUSED deterministic routing + packing (no atomics)
    if (total_tokens > 0)
    {
        // TIMER_BLOCK("route_and_pack_fused_kernel");
        int threads = 1;
        while (threads < Bgrp)
            threads <<= 1;
        threads = min(threads, 1024); // hardware cap

        // 5 arrays of int[T] + 1 int for carry
        size_t shmem = (size_t)threads * 5 * sizeof(int) + sizeof(int);

        route_and_pack_fused_kernel<<<E, threads, shmem, sMoe>>>(
            s->gather_x_g, Bgrp, H,
            s->topk_i_g, s->topk_v_g, K,
            s->d_expert_offsets, E,
            s->local_ids_g, s->local_wts_g,
            s->expert_input_buffer_g);
    }
    else
    {
        // nothing routed; just residual-add zeros later
        return;
    }

    // 5) MLP1 (column-parallel on O=2D): local shard
    int o_len = 0;
    {
        // TIMER_BLOCK("MLP1");
        const int twoD = 2 * D;
        const int base = twoD / TP;
        const int rem = twoD % TP;
        o_len = base + (rank_in_group < rem ? 1 : 0);

        const size_t seg1_loc = (size_t)o_len * H; // per-expert stride in local shard
        const __hip_bfloat16 *W1 = w->w_mlp1 + (size_t)layer_idx * (size_t)E * seg1_loc;

        mlp1_optimized<16,16,16,  /*WAVES_M,N,K*/ 4,4,4,
            /*TW_M,TW_N*/ 2,4,
            /*PAD_K*/ 8>(
            s->mlp1_out_g, s->expert_input_buffer_g, W1,
            s->d_expert_offsets, s->d_expert_counts,
            s->d_tile2expert_g, s->d_tile2local_g,
            /*E=*/E, /*H=*/H, /*twoD=*/o_len, /*cur_tiles=*/cur_tiles, sMoe);
    }

    // 6) SwiGLU + bias for local columns -> gate_up_g with Dloc = o_len/2
    const int Dloc = o_len / 2;
    {
        // TIMER_BLOCK("SwiGLU + bias");
        const __hip_bfloat16 *b1 = w->b_mlp1 + (size_t)layer_idx * (size_t)E * (size_t)o_len;

        const size_t work = (size_t)total_tokens * Dloc;
        if (work > 0)
        {
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
        // TIMER_BLOCK("MLP2");
        int i_len, i_start;
        {
            const int base = D / TP, rem = D % TP;
            i_len = base + (rank_in_group < rem ? 1 : 0);
            i_start = rank_in_group * base + (rank_in_group < rem ? rank_in_group : rem);
        }
        if (i_len != Dloc)
        {
            fprintf(stderr, "MLP2 shard mismatch: i_len=%d vs Dloc=%d\n", i_len, Dloc);
            abort();
        }

        const size_t seg2_loc = (size_t)H * (size_t)i_len;
        const __hip_bfloat16 *W2 = w->w_mlp2 + (size_t)layer_idx * (size_t)E * seg2_loc;
        const __hip_bfloat16 *b2s = w->b_mlp2 + (size_t)layer_idx * (size_t)E * H;
        
        mlp2_optimized<16,16,16,  /*WAVES_M,N,K*/ 4,4,4,
            /*TW_M,TW_N*/ 1,2,
            /*PAD_K*/ 8>(
                s->expert_output_partial_g, s->gate_up_g, W2, b2s,
                           s->d_expert_offsets, s->d_expert_counts,
                           s->d_tile2expert_g, s->d_tile2local_g,
                /*E=*/E, /*D=*/i_len, /*H=*/H, /*cur_tiles=*/cur_tiles, sMoe);
        HIP_CHECK(hipGetLastError());
    }

    // 9) Reduce over experts back to tokens (union) -> e_agg_g[Bgrp,H]
    {
        // TIMER_BLOCK("reduce_tokenwise_expert_outputs");
        const int threads = 256;
        dim3 grid(Bgrp, (H + threads - 1) / threads);
        reduce_tokenwise_expert_outputs<<<grid, threads, 0, sMoe>>>(
            s->e_agg_g, s->expert_output_partial_g,
            s->local_ids_g, s->local_wts_g, Bgrp, H, K);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);

    // 9b) Emulate ALL-REDUCE across TP on token-wise results: e_agg_g[Bgrp, H]
    {
        // TIMER_BLOCK("moe_allreduce_tokenwise");
        const size_t span_elems = (size_t)Bgrp * (size_t)H;
        const size_t bytes      = span_elems * sizeof(float);

        // We reuse expert_output_gather_g as a TP×[Bgrp,H] staging buffer.
        // Layout: rank r writes to slice r * span_elems
        float* gather_base = s->expert_output_gather_g;

        // self
        HIP_CHECK(hipMemcpyAsync(
            gather_base + (size_t)rank_in_group * span_elems,
            s->e_agg_g, bytes,
            hipMemcpyDeviceToDevice, sMoe));

        // peers
        for (int r = 0; r < TP; ++r) {
            const int peer_dev = group_base + r;
            if (peer_dev == my_dev) continue;
            float* peer_src = gpu_transformers[peer_dev]->state.e_agg_g;
            float* dst      = gather_base + (size_t)r * span_elems;

            if (tp.p2p[rank_in_group][r]) {
                HIP_CHECK(hipMemcpyPeerAsync(dst, my_dev, peer_src, peer_dev, bytes, sMoe));
            } else {
                HIP_CHECK(hipMemcpyAsync(cpu->tp_host_stage, peer_src, bytes, hipMemcpyDeviceToHost, sMoe));
                HIP_CHECK(hipMemcpyAsync(dst, cpu->tp_host_stage, bytes, hipMemcpyHostToDevice, sMoe));
            }
        }
        HIP_CHECK(hipStreamSynchronize(sMoe));
        tp_group_barrier(tp);

        // Sum across ranks into e_agg_g in-place: [TP, Bgrp, H] -> [Bgrp, H]
        const size_t total = span_elems;
        const int BLK = 256, GRD = (int)((total + BLK - 1) / BLK);
        sum_rank_axis_kernel<<<GRD, BLK, 0, sMoe>>>(
            /*out=*/s->e_agg_g,
            /*parts=*/gather_base,
            /*TP=*/TP, /*T=*/Bgrp, /*H=*/H);
        HIP_CHECK(hipGetLastError());
    }

    // 10) Scatter owner rows back to this rank and residual-add into x
    {
        // TIMER_BLOCK("scatter_add_moe_output");
        float *x_mb = s->x + (size_t)row_offset * H;
        float *src_owner = s->e_agg_g + (size_t)rank_in_group * bs_local * H;
        const int elems = bs_local * H;
        accumulate_kernel<<<(elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK, 0, sMoe>>>(
            x_mb, src_owner, 1.0f, bs_local, H);
        HIP_CHECK(hipGetLastError());
    }
}

#include "ep_utils.hpp"   // add this include at top with others


__global__ void pack_owner_outputs_by_src_kernel(
    float* __restrict__ outBuf,        // [pairs, H]
    const float* __restrict__ partBuf, // [pairs, H]
    const int*   __restrict__ ids,     // [pairs] (unused here; kept for signature parity)
    const float* __restrict__ wts,     // [pairs] (unused here)
    int pairs, int H, int bs_local)    // bs_local kept for parity; not needed in this simple copy
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const float* src = partBuf + (size_t)p * H;
    float* dst = outBuf + (size_t)p * H;
    for (int j = threadIdx.y; j < H; j += blockDim.y) {
        dst[j] = src[j];  // leave weighting to your reducer (to avoid double-mul)
    }
}

// Map global expert to owner rank (contiguous partition)
__device__ __forceinline__ int owner_rank_from_expert(int e, int TP, int E) {
    const int base = E / TP, rem = E % TP;
    int acc = 0;
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        if (r >= TP) break;
        const int len = base + (r < rem ? 1 : 0);
        if (e < acc + len) return r;
        acc += len;
    }
    return TP-1; // should not happen
}

// Sender: pack X/topk into outbox by destination (stable per-peer using atomic counters)
__global__ void pack_outbox_by_dest_kernel(
    float* __restrict__ obox_X,
    int*   __restrict__ obox_tok,
    int*   __restrict__ obox_exp,
    float* __restrict__ obox_wt,
    const float* __restrict__ X_src,    // [bs_local, H]
    const int*   __restrict__ topk_i,   // [bs_local, K]
    const float* __restrict__ topk_v,   // [bs_local, K]
    const int*   __restrict__ d_send_off, // [TP] prefix offsets (within obox_* arrays)
    int*         __restrict__ d_counters, // [TP] per-dest counters (zeroed before launch)
    int bs_local, int H, int K, int TP, int E)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0..bs_local*K-1
    const int total = bs_local * K;
    if (idx >= total) return;

    const int t = idx / K;
    const int k = idx % K;

    const int e = topk_i[t*K + k];
    const int r = owner_rank_from_expert(e, TP, E);

    const int pos = atomicAdd(&d_counters[r], 1);
    const int dst = d_send_off[r] + pos;

    // copy one row
    const float* src = X_src + (size_t)t * H;
    float*       dstp = obox_X + (size_t)dst * H;
    for (int j = threadIdx.y; j < H; j += blockDim.y) {
        dstp[j] = src[j];
    }
    if (threadIdx.y == 0) {
        obox_tok[dst] = t;
        obox_exp[dst] = e;
        obox_wt [dst] = topk_v[t*K + k];
    }
}

// Owner: count per local expert from ibox_exp
__global__ void count_local_expert_pairs_kernel(
    const int* __restrict__ ibox_exp, int n_pairs,
    int E_start, int E_loc,
    int* __restrict__ d_e_counts) // [E_loc], zeroed
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const int e = ibox_exp[i];
    const int le = e - E_start;
    if (le >= 0 && le < E_loc) atomicAdd(&d_e_counts[le], 1);
}

// Owner: pack by local expert, recording permutation back to inbox
__global__ void pack_inbox_by_local_expert_kernel(
    float* __restrict__ grp_X,   int* __restrict__ grp_tok,
    float* __restrict__ grp_wt,  int* __restrict__ grp_perm,
    const float* __restrict__ ibox_X,
    const int*   __restrict__ ibox_tok,
    const float* __restrict__ ibox_wt,
    const int*   __restrict__ ibox_exp,
    int n_pairs, int H,
    int E_start, int E_loc,
    int* __restrict__ d_e_off,    // [E_loc] exclusive offsets
    int* __restrict__ d_e_counters) // [E_loc] zeroed before launch
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const int e  = ibox_exp[i];
    const int le = e - E_start;
    if (le < 0 || le >= E_loc) return; // safety
    const int j = atomicAdd(&d_e_counters[le], 1);
    const int dst = d_e_off[le] + j;

    // copy one row
    const float* src = ibox_X + (size_t)i * H;
    float*       out = grp_X   + (size_t)dst * H;
    for (int c = threadIdx.y; c < H; c += blockDim.y) out[c] = src[c];

    if (threadIdx.y == 0) {
        grp_tok [dst] = ibox_tok[i];
        grp_wt  [dst] = ibox_wt[i];
        grp_perm[dst] = i; // remember inbox slot for stable return
    }
}

// Owner: reorder grouped outputs back to inbox order using grp_perm
__global__ void scatter_by_perm_kernel(
    float* __restrict__ ret_out,        // [n_pairs, H]
    const float* __restrict__ grp_out,  // [n_pairs, H] (valid rows: total_recv_pairs)
    const int*   __restrict__ grp_perm, // [n_pairs]
    int n_pairs, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const int dst = grp_perm[i];
    const float* src = grp_out + (size_t)i   * H;
    float*       out = ret_out + (size_t)dst * H;
    for (int c = threadIdx.y; c < H; c += blockDim.y) out[c] = src[c];
}

static inline int ep_owner_rank_from_expert(int e, int TP, int E) {
    const int base = E / TP, rem = E % TP;
    int acc = 0;
    for (int r = 0; r < TP; ++r) {
        const int len = base + (r < rem ? 1 : 0);
        if (e < acc + len) return r;
        acc += len;
    }
    return TP - 1; // should not happen
}


// Small, robust TP gather for TP×TP integers via device buffers.
// - out_mat_host: host buffer [TP][TP] to fill (row-major)
// - my_row_host : this rank’s row of length TP
// Contract: tp_groups[my_dev] is valid; peers are gpu_transformers[group_base + r]
// drop-in replacement
static void tp_gather_small_ints(TPGroup &tp,
                                 int (*out_mat_host)[MAX_TP],
                                 const int *my_row_host,
                                 int TP)
{
    int my_dev = 0; HIP_CHECK(hipGetDevice(&my_dev));
    const int group_base = (my_dev / TP) * TP;
    const int my_rank    = my_dev % TP;

    GPURunState &s = gpu_transformers[my_dev]->state;

    // 1) Upload my row to my device scratch and to my row in my ctrl_mat
    HIP_CHECK(hipMemcpy(s.ctrl_row_dev, my_row_host, TP * sizeof(int), hipMemcpyHostToDevice));
    int *my_row_dst = s.ctrl_mat + my_rank * TP;
    HIP_CHECK(hipMemcpy(my_row_dst, s.ctrl_row_dev, TP * sizeof(int), hipMemcpyDeviceToDevice));

    // 2) Pull other ranks’ rows into my ctrl_mat (never write into peers)
    for (int r = 0; r < TP; ++r) {
        if (r == my_rank) continue;
        const int peer_dev = group_base + r;

        const int *peer_src = gpu_transformers[peer_dev]->state.ctrl_row_dev; // lives on peer
        int *dst_row = s.ctrl_mat + r * TP;                                   // my device

        if (tp.p2p[my_rank][r]) {
            HIP_CHECK(hipMemcpyPeerAsync(dst_row, my_dev, peer_src, peer_dev,
                                         TP*sizeof(int), /*stream*/0));
        } else {
            void *hbuf = gpu_transformers[my_dev]->cpu_buffers.tp_host_stage; // my pinned host
            HIP_CHECK(hipMemcpyAsync(hbuf, peer_src, TP*sizeof(int), hipMemcpyDeviceToHost, 0));
            HIP_CHECK(hipStreamSynchronize(0)); // tiny copy; OK to sync
            HIP_CHECK(hipMemcpyAsync(dst_row, hbuf, TP*sizeof(int), hipMemcpyHostToDevice, 0));
        }
    }

    HIP_CHECK(hipDeviceSynchronize());
    tp_group_barrier(tp);

    // 3) Copy my assembled matrix to host
    std::vector<int> tmp(TP*TP);
    HIP_CHECK(hipMemcpy(tmp.data(), s.ctrl_mat, TP*TP*sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < TP; ++i)
        for (int j = 0; j < TP; ++j)
            out_mat_host[i][j] = tmp[i*TP + j];
}

float sum_debug(float* d_x, int n) {
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> h_x(n);
    HIP_CHECK(hipMemcpy(h_x.data(), d_x, n * sizeof(float), hipMemcpyDeviceToHost));
    double acc = 0.f;
    for (int i = 0; i < n; ++i) acc += h_x[i];
    return acc;
}

__global__ void reduce_pairs_atomic(
    float* __restrict__ e_agg,          // [B, H]
    const float* __restrict__ retX,     // [P, H]  (P = total_send_pairs = B*K)
    const int*   __restrict__ obox_tok, // [P] token id for each pair
    const float* __restrict__ obox_wt,  // [P] weight for each pair
    int P, int B, int H)
{
    int p = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (p >= P || d >= H) return;

    const int t = obox_tok[p];
    const float w = obox_wt[p];
    const float v = retX[(size_t)p * H + d];

    // multiple pairs for the same token must accumulate -> atomic
    atomicAdd(&e_agg[(size_t)t * H + d], w * v);
}


void moe_gpu_120b(GPUTransformer *gpu_t, int layer_idx, int batch_size,
                  int row_offset, hipStream_t sMoe)
{
    if (batch_size <= 0) return;

    Config *p   = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu = &gpu_t->cpu_buffers;

    const int TP = TENSOR_PARALLEL_SIZE;
    int my_dev = 0; HIP_CHECK(hipGetDevice(&my_dev));
    const int group_base = (my_dev / TP) * TP;
    const int rank_in_group = my_dev % TP;
    TPGroup &tp = tp_groups[my_dev];

    const int H = p->hidden_dim;
    const int D = p->intermediate_dim;
    const int E = p->n_experts;
    const int K = p->experts_per_token;

    const int bs_local = batch_size;

    // ---- small safe helpers (no persistent allocs) -------------------------
    auto owner_rank_from_expert_host = [&](int e)->int {
        const int base = E / TP, rem = E % TP;
        int acc = 0;
        for (int r = 0; r < TP; ++r) {
            const int len = base + (r < rem ? 1 : 0);
            if (e < acc + len) return r;
            acc += len;
        }
        assert(0);
        return TP-1; // should not happen
    };
    auto copy_p2p_or_bounce_void = [&](void *dst, int dst_dev,
                                       const void *src, int src_dev,
                                       size_t bytes, hipMemcpyKind kd,
                                       hipStream_t stream) {
        if (bytes == 0) return;
        if (dst_dev == src_dev) {
            HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kd, stream));
            return;
        }
        // try P2P for device->device path
        if (kd == hipMemcpyDeviceToDevice) {
            const int r_src = src_dev % TP;
            const int r_dst = dst_dev % TP;
            if (tp_groups[src_dev].p2p[r_src][r_dst]) {
                HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
                return;
            }
        }
        // host paths (rare here)
        HIP_CHECK(hipMemcpyAsync(dst, src, bytes, kd, stream));
    };
    auto copy_p2p_or_bounce_f = [&](float *dst, int dst_dev,
                                    const float *src, int src_dev,
                                    size_t bytes, hipStream_t stream) {
        copy_p2p_or_bounce_void((void*)dst, dst_dev, (const void*)src, src_dev,
                                bytes, hipMemcpyDeviceToDevice, stream);
    };
    auto copy_p2p_or_bounce_i = [&](int *dst, int dst_dev,
                                    const int *src, int src_dev,
                                    size_t bytes, hipStream_t stream) {
        copy_p2p_or_bounce_void((void*)dst, dst_dev, (const void*)src, src_dev,
                                bytes, hipMemcpyDeviceToDevice, stream);
    };
    // -----------------------------------------------------------------------

    // 0) RMSNorm on local microbatch
    {
        float *x_mb = s->x + (size_t)row_offset * H;
        float *t_mb = s->t + (size_t)row_offset * H;
        dim3 grid(bs_local), block(THREADS_PER_BLOCK);
        rmsnorm_kernel<<<grid, block, 0, sMoe>>>(t_mb, x_mb,
            w->rms_ffn_w + (size_t)layer_idx * H, bs_local, H);
        HIP_CHECK(hipGetLastError());
    }

    // debug_print("MOE L%d: before routing, x sum %.6f\n",
    //            layer_idx, sum_debug(s->t + (size_t)row_offset * H, (size_t)bs_local * H));

    // 1) Router on local tokens → topK ids/weights (softmax over K)
    {
        matmul_vec128_singlebuf<
            16,16,16, 4,4,4, 1,2, 8, false
        >(s->router_score, s->t + (size_t)row_offset * H,
          w->w_router + (size_t)layer_idx * H * E,
          /*M=*/bs_local, /*K=*/H, /*N=*/E,
          /*bias=*/nullptr, /*stream=*/sMoe);
        HIP_CHECK(hipGetLastError());

        const int elems = bs_local * E;
        if (elems > 0) {
            add_bias_kernel<<<(elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                              THREADS_PER_BLOCK, 0, sMoe>>>(
                s->router_score, w->b_router + (size_t)layer_idx * E, bs_local, E);
            HIP_CHECK(hipGetLastError());
        }
        topk_kernel<<<bs_local, 1, 0, sMoe>>>(s->local_wts, s->local_ids, s->router_score, bs_local, E, K);
        HIP_CHECK(hipGetLastError());
        softmax_kernel<<<bs_local, THREADS_PER_BLOCK, 0, sMoe>>>(s->local_wts, bs_local, K);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);

    // 2) PLAN per-destination (owner rank) send counts (host)
    int send_counts[MAX_TP] = {0};
    for (int t = 0; t < bs_local; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = s->local_ids[t*K + k];
            const int r = owner_rank_from_expert_host(e);
            ++send_counts[r];
        }
    }
    int send_off[MAX_TP] = {0};
    for (int r = 1; r < TP; ++r) send_off[r] = send_off[r-1] + send_counts[r-1];
    const int total_send_pairs = (TP>0 ? send_off[TP-1] + send_counts[TP-1] : 0);

    // device: send prefix + zero per-destination counters
    HIP_CHECK(hipMemcpyAsync(s->d_peer_send_off, send_off, TP*sizeof(int), hipMemcpyHostToDevice, sMoe));
    HIP_CHECK(hipMemsetAsync(s->d_peer_counters, 0, TP*sizeof(int), sMoe));

    // 3) PACK sender outbox (stable per peer)
    {
        const float* X_src = s->t + (size_t)row_offset * H;
        dim3 blk(256, 1), grd((bs_local*K + blk.x - 1) / blk.x);
        pack_outbox_by_dest_kernel<<<grd, blk, 0, sMoe>>>(
            s->obox_X, s->obox_tok, s->obox_exp, s->obox_wt,
            X_src,
            s->local_ids, s->local_wts,
            s->d_peer_send_off, s->d_peer_counters,
            bs_local, H, K, TP, E);
        HIP_CHECK(hipGetLastError());
    }

    // debug_print("MOE L%d: after routing, x sum %.6f\n",
    //            layer_idx, sum_debug(s->obox_X, (size_t)bs_local * H * K));

    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);

    // 4) Gather small control (send_counts from everyone) so each owner knows recv plan
    int all_send_counts[MAX_TP][MAX_TP];
    tp_gather_small_ints(tp, all_send_counts, send_counts /* my row */, TP);

    int recv_counts[MAX_TP] = {0};
    for (int srank = 0; srank < TP; ++srank)
        recv_counts[srank] = all_send_counts[srank][rank_in_group];

    int recv_off[MAX_TP] = {0};
    for (int srank = 1; srank < TP; ++srank) recv_off[srank] = recv_off[srank-1] + recv_counts[srank-1];
    const int total_recv_pairs = (TP>0 ? recv_off[TP-1] + recv_counts[TP-1] : 0);

    // For return path: precompute, for each SENDER, the prefix offset where THIS owner’s results should land
    int sender_recv_off_for_me[MAX_TP] = {0};
    for (int srank = 0; srank < TP; ++srank) {
        int acc = 0;
        for (int d = 0; d < rank_in_group; ++d) acc += all_send_counts[srank][d];
        sender_recv_off_for_me[srank] = acc;
    }

    // For forward shipping: in each destination owner's inbox, where should *my* slice start?
    int dst_recv_off_for_me[MAX_TP] = {0};
    for (int dst = 0; dst < TP; ++dst) {
        int acc = 0;
        for (int s = 0; s < rank_in_group; ++s) acc += all_send_counts[s][dst];
        dst_recv_off_for_me[dst] = acc;
    }

    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);
    
    // 5) SHIP to owners (sender → owner). Copy by slices per destination
    for (int dst = 0; dst < TP; ++dst) {
        const int peer_dev = group_base + dst;
        const int cnt = send_counts[dst];
        if (!cnt) continue;

        const size_t bytes_X = (size_t)cnt * (size_t)H * sizeof(float);
        const size_t bytes_M = (size_t)cnt * sizeof(int);
        const size_t bytes_W = (size_t)cnt * sizeof(float);

        float* srcX = s->obox_X   + (size_t)send_off[dst] * H;
        int*   srcT = s->obox_tok + (size_t)send_off[dst];
        int*   srcE = s->obox_exp + (size_t)send_off[dst];
        float* srcW = s->obox_wt  + (size_t)send_off[dst];

        // AFTER (correct):
        float* dstX = gpu_transformers[peer_dev]->state.ibox_X   + (size_t)dst_recv_off_for_me[dst] * H;
        int*   dstT = gpu_transformers[peer_dev]->state.ibox_tok + (size_t)dst_recv_off_for_me[dst];
        int*   dstE = gpu_transformers[peer_dev]->state.ibox_exp + (size_t)dst_recv_off_for_me[dst];
        float* dstW = gpu_transformers[peer_dev]->state.ibox_wt  + (size_t)dst_recv_off_for_me[dst];


        copy_p2p_or_bounce_f(dstX, peer_dev, srcX, my_dev, bytes_X, sMoe);
        copy_p2p_or_bounce_i(dstT, peer_dev, srcT, my_dev, bytes_M, sMoe);
        copy_p2p_or_bounce_i(dstE, peer_dev, srcE, my_dev, bytes_M, sMoe);
        copy_p2p_or_bounce_f(dstW, peer_dev, srcW, my_dev, bytes_W, sMoe);
    }
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);
    
    // 6) OWNER: group inbox by local expert
    int E_loc=0, E_start=0; ep_owner_map(tp, E, E_loc, E_start);

    HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, E_loc * sizeof(int), sMoe));
    {
        dim3 blk(256,1), grd((total_recv_pairs + blk.x - 1) / blk.x);
        count_local_expert_pairs_kernel<<<grd, blk, 0, sMoe>>>(
            s->ibox_exp, total_recv_pairs, E_start, E_loc, s->d_expert_counts);
        HIP_CHECK(hipGetLastError());
    }

    
    // pull counts → exclusive scan → push offsets; reuse d_expert_counts as temp counters again
    HIP_CHECK(hipMemcpyAsync(cpu->expert_counts, s->d_expert_counts,
                             E_loc * sizeof(int), hipMemcpyDeviceToHost, sMoe));
    HIP_CHECK(hipStreamSynchronize(sMoe));
    int total_tokens_local = 0;
    {
        int acc = 0;
        for (int i = 0; i < E_loc; ++i) { cpu->expert_offsets[i] = acc; acc += cpu->expert_counts[i]; }
        total_tokens_local = acc;
    }
    HIP_CHECK(hipMemcpyAsync(s->d_expert_offsets, cpu->expert_offsets,
                             E_loc * sizeof(int), hipMemcpyHostToDevice, sMoe));
    HIP_CHECK(hipMemsetAsync(s->d_expert_counts, 0, E_loc * sizeof(int), sMoe)); // reuse as counters
        
    // // debug_print("[MoE-EP] rank %d: recv_pairs=%d, local_experts=%d (%d..%d), local_pairs=%d\n",
    //            rank_in_group, total_recv_pairs, E_loc, E_start, E_start+E_loc-1, total_tokens_local);
    // pack by local expert (+ record permutation to restore order later)
    {
        const int n = total_recv_pairs;
        if (n > 0) {
            dim3 blk(256, 1);
            dim3 grd((n + blk.x - 1) / blk.x);
            pack_inbox_by_local_expert_kernel<<<grd, blk, 0, sMoe>>>(
                s->grp_X, s->grp_tok, s->grp_wt, s->grp_perm,
                s->ibox_X, s->ibox_tok, s->ibox_wt, s->ibox_exp,
                n, H, E_start, E_loc,
                s->d_expert_offsets, s->d_expert_counts);
            HIP_CHECK(hipGetLastError());
        }
    }

    HIP_CHECK(hipStreamSynchronize(sMoe));
    // Build tile maps for grouped matmuls (local slice)
    int cur_tiles = 0;
    for (int le = 0; le < E_loc; ++le) {
        const int cnt = cpu->expert_counts[le];
        const int tiles = (cnt + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
        for (int m = 0; m < tiles; ++m) {
            cpu->h_tile2expert[cur_tiles + m] = le;
            cpu->h_tile2local [cur_tiles + m] = m;
        }
        cur_tiles += tiles;
    }
    if (cur_tiles > s->cap_tiles) { fprintf(stderr, "[MoE-EP] tiles(%d) > cap(%d)\n", cur_tiles, s->cap_tiles); abort(); }
    if (cur_tiles > 0) {
        HIP_CHECK(hipMemcpyAsync(s->d_tile2expert_g, cpu->h_tile2expert, cur_tiles*sizeof(int), hipMemcpyHostToDevice, sMoe));
        HIP_CHECK(hipMemcpyAsync(s->d_tile2local_g,  cpu->h_tile2local,  cur_tiles*sizeof(int), hipMemcpyHostToDevice, sMoe));
    }
    // debug_print("[MoE-EP] rank %d: recv_pairs=%d, local_experts=%d (%d..%d), local_pairs=%d, local tiles=%d\n, checksum=%f\n",
    //           rank_in_group, total_recv_pairs, E_loc, E_start, E_start+E_loc-1, total_tokens_local, cur_tiles, sum_debug(s->ibox_X, total_tokens_local * H));
    // // debug_print("rank %d: local tiles=%d (cap %d)\n", rank_in_group, cur_tiles, s->cap_tiles);
    // 7) Run local experts (MXFP4 packed weights + bf16 biases)
    {
        // MLP1 → s->mlp1_out_g
        {
            dim3 grid((2*D + BLOCK_N_MLP - 1) / BLOCK_N_MLP, cur_tiles);
            dim3 block(LANE_PER_WAVE, (WAVES_M_MLP) * (WAVES_N_MLP/2));
            const size_t shmem = sizeof(uint16_t) *
               (size_t)(BLOCK_M_MLP * (BLOCK_K_MLP + PAD_K_MLP) + (BLOCK_K_MLP + PAD_K_MLP) * BLOCK_N_MLP);
            assert_smem_or_die(shmem, "grouped_mlp1_mxfp4_kernel_unified(EP)");

            const size_t seg1 = (size_t)(2*D) * (size_t)H;
            const size_t pack_off = (size_t)layer_idx * (size_t)E_loc * ((seg1 + 1)/2);
            const size_t sc_off   = (size_t)layer_idx * (size_t)E_loc * ((seg1 + 31)/32);

            hipLaunchKernelGGL(
                (grouped_mlp1_mxfp4_kernel_unified<
                    16,16,16, WAVES_M_MLP, WAVES_N_MLP, WAVES_K_MLP, 1,2, PAD_K_MLP>),
                grid, block, shmem, sMoe,
                /*C*/ s->mlp1_out_g,
                /*A*/ s->grp_X,
                /*W*/ w->w_mlp1_mxfp4 + pack_off,
                /*S*/ w->w_mlp1_scales_f32 + sc_off,
                /*routing*/ s->d_expert_offsets, s->d_expert_counts,
                /*tiles*/   s->d_tile2expert_g, s->d_tile2local_g,
                /*E*/ E_loc, /*K*/ H, /*N*/ 2*D);
            HIP_CHECK(hipGetLastError());
        }
        // debug_print("MOE L%d: after MLP1, x sum %.6f\n",
        //            layer_idx, sum_debug(s->mlp1_out_g, (size_t)total_tokens_local * (size_t)(2*D)));
        // // debug_print("HERE0\n");
        // SwiGLU + bias → s->gate_up_g  (D)
        {
            const size_t work = (size_t)total_tokens_local * (size_t)D;
            if (work > 0) {
                const __hip_bfloat16* b1 = w->b_mlp1 + (size_t)layer_idx * (size_t)E_loc * (size_t)(2*D);
                const int T = 256;
                dim3 grid2((int)((work + T - 1) / T)), block2(T);
                bias_swiglu_epilogue_kernel<<<grid2, block2, 0, sMoe>>>(
                    s->mlp1_out_g, b1,
                    s->d_expert_offsets, s->d_expert_counts, E_loc,
                    s->gate_up_g, /*D=*/D, total_tokens_local, p->swiglu_limit, 1.702f);
                HIP_CHECK(hipGetLastError());
            }
        }
        // debug_print("MOE L%d: after SwiGLU, x sum %.6f\n",
        //            layer_idx, sum_debug(s->gate_up_g, (size_t)total_tokens_local * (size_t)D));
        // MLP2 + bias → s->grp_out (H)
        {
            dim3 grid((H + BLOCK_N_MLP - 1) / BLOCK_N_MLP, cur_tiles);
            dim3 block(LANE_PER_WAVE, (WAVES_M_MLP) * (WAVES_N_MLP/2));
            const size_t shmem = sizeof(uint16_t) *
               (size_t)(BLOCK_M_MLP * (BLOCK_K_MLP + PAD_K_MLP) + (BLOCK_K_MLP + PAD_K_MLP) * BLOCK_N_MLP);
            assert_smem_or_die(shmem, "grouped_mlp2_mxfp4_bias_kernel_unified(EP)");

            const size_t seg2 = (size_t)H * (size_t)D;
            const size_t pack_off = (size_t)layer_idx * (size_t)E_loc * ((seg2 + 1)/2);
            const size_t sc_off   = (size_t)layer_idx * (size_t)E_loc * ((seg2 + 31)/32);
            const __hip_bfloat16* b2 = w->b_mlp2 + (size_t)layer_idx * (size_t)E_loc * (size_t)H;

            hipLaunchKernelGGL(
                (grouped_mlp2_mxfp4_bias_kernel_unified<
                    16,16,16, WAVES_M_MLP, WAVES_N_MLP, WAVES_K_MLP, 1,2, PAD_K_MLP>),
                grid, block, shmem, sMoe,
                /*C*/ s->grp_out,
                /*A*/ s->gate_up_g,
                /*W*/ w->w_mlp2_mxfp4 + pack_off,
                /*S*/ w->w_mlp2_scales_f32 + sc_off,
                /*b*/ b2,
                /*routing*/ s->d_expert_offsets, s->d_expert_counts,
                /*tiles*/   s->d_tile2expert_g, s->d_tile2local_g,
                /*E*/ E_loc, /*K*/ D, /*N*/ H);
            HIP_CHECK(hipGetLastError());
        }
        // debug_print("MOE L%d: after MLP2, x sum %.6f\n",
        //            layer_idx, sum_debug(s->grp_out, (size_t)total_tokens_local * (size_t)H));
    }
    // // debug_print("HERE\n");
    
    // 8) Restore owner outputs back to inbox order using permutation
    {
        dim3 blk(256,1), grd((total_tokens_local + blk.x - 1) / blk.x);
        scatter_by_perm_kernel<<<grd, blk, 0, sMoe>>>(
            s->ret_out, s->grp_out, s->grp_perm, total_tokens_local, H);
        HIP_CHECK(hipGetLastError());
    }
    // // debug_print("HERE1\n");
    // debug_print("MOE L%d: after scatter, x sum %.6f\n",
    //            layer_idx, sum_debug(s->ret_out, (size_t)total_tokens_local * (size_t)H));
    HIP_CHECK(hipStreamSynchronize(sMoe));
    tp_group_barrier(tp);
    
    // 9) RETURN to senders: copy each sender’s slice back into their ret_pairs_X
    for (int srank = 0; srank < TP; ++srank) {
        const int cnt = recv_counts[srank];
        if (!cnt) continue;
        const int peer_dev = group_base + srank;

        const float* src = s->ret_out + (size_t)recv_off[srank] * H;
        float* dst = gpu_transformers[peer_dev]->state.ret_pairs_X
                   + (size_t)sender_recv_off_for_me[srank] * H;
        const size_t bytes = (size_t)cnt * (size_t)H * sizeof(float);

        copy_p2p_or_bounce_f(dst, peer_dev, src, my_dev, bytes, sMoe);
    }
    // debug_print("MOE L%d: after return, x sum %.6f\n",
    //            layer_idx, sum_debug(s->ret_pairs_X, (size_t)bs_local * (size_t)H * K));
    HIP_CHECK(hipStreamSynchronize(sMoe));

    // 9) RETURN ... (your code)

    {
        // ZERO the accumulation slice for this rank before atomic adds
        float* eagg = s->e_agg_g + (size_t)rank_in_group * bs_local * H;
        HIP_CHECK(hipMemsetAsync(eagg, 0, (size_t)bs_local * H * sizeof(float), sMoe));

        const int P = bs_local * K;
        const int T = 256;
        dim3 grid(/*x=*/P, /*y=*/(H + T - 1) / T);
        reduce_pairs_atomic<<<grid, T, 0, sMoe>>>(
            eagg, s->ret_pairs_X, s->obox_tok, s->obox_wt,
            /*P=*/P, /*B=*/bs_local, /*H=*/H);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipStreamSynchronize(sMoe));
    // debug_print("MOE L%d: after reduce, x sum %.6f\n",
    //            layer_idx, sum_debug(s->e_agg_g + (size_t)rank_in_group * (size_t)bs_local * (size_t)H, (size_t)bs_local * (size_t)H));
    // debug_print("MOE L%d: first elem %.6f\n",
    //            layer_idx, sum_debug(s->e_agg_g + (size_t)rank_in_group * (size_t)bs_local * (size_t)H, 1));
    // 11) Residual add into x
    {
        float *x_mb = s->x + (size_t)row_offset * H;
        const float *src_owner = s->e_agg_g + (size_t)rank_in_group * bs_local * H;
        const int elems = bs_local * H;
        accumulate_kernel<<<(elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                            THREADS_PER_BLOCK, 0, sMoe>>>(
            x_mb, src_owner, 1.0f, bs_local, H);
        HIP_CHECK(hipGetLastError());
    }
}

// ------------------------------ Pipelined forward (layer overlap) -------------------------

int *forward_batch_gpu(GPUTransformer *gpu_t, int *tokens, int batch_size)
{
    Config *p = &gpu_t->config;
    GPURunState *s = &gpu_t->state;
    GPUTransformerWeights *w = &gpu_t->weights;
    CPUBuffers *cpu_buf = &gpu_t->cpu_buffers;
    const int H = p->hidden_dim;
    const int B = batch_size;
    if (B <= 0)
        return cpu_buf->current_tokens;

    // streams
    hipStream_t attn_stream = nullptr, moe_stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&attn_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&moe_stream, hipStreamNonBlocking));

    // microbatching — keep your MB exactly (no TP-capping inside attention)
    const int MB = (BATCH_SIZE <= B) ? BATCH_SIZE : B;
    const int NUM_MB = (B + MB - 1) / MB;

    // per-microbatch events
    auto make_evt = []()
    {
        hipEvent_t e = nullptr;
        HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
        return e;
    };
    std::vector<hipEvent_t> evt_attn_done(NUM_MB);
    std::vector<hipEvent_t> evt_moe_done_prev(NUM_MB), evt_moe_done_cur(NUM_MB);
    for (int i = 0; i < NUM_MB; ++i)
    {
        evt_attn_done[i] = make_evt();
        evt_moe_done_prev[i] = make_evt();
        evt_moe_done_cur[i] = make_evt();
    }

    // H2D tokens + positions
    h2d_copy(s->current_tokens, tokens, (size_t)B * sizeof(int), attn_stream);
    h2d_copy(s->positions, cpu_buf->positions, (size_t)B * sizeof(int), attn_stream);

    // embeddings on attn_stream
    {
        const int elems = B * H;
        if (elems > 0)
        {
            dim3 grid((elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            copy_embeddings_kernel<<<grid, THREADS_PER_BLOCK, 0, attn_stream>>>(
                s->x, w->token_embedding_table, s->current_tokens, B, H);
            HIP_CHECK(hipGetLastError());
        }
    }

    // pipelined schedule across layers and microbatches
    for (int l = 0; l < p->n_layers; ++l)
    {
        int mb_idx = 0;
        for (int row = 0; row < B; row += MB, ++mb_idx)
        {
            const int bs = std::min(MB, B - row);
            const int i = mb_idx;

            // cross-layer dep: Attention(l,i) waits MoE(l-1,i)
            if (l > 0)
                HIP_CHECK(hipStreamWaitEvent(attn_stream, evt_moe_done_prev[i], 0));

            // Attn(l,i)
            attention_gpu(gpu_t, l, bs, /*row_offset=*/row, attn_stream);
            HIP_CHECK(hipEventRecord(evt_attn_done[i], attn_stream));

            // MoE(l,i-1) on the separate stream
            if (i > 0)
            {
                const int prev_row = row - MB;
                const int prev_bs = std::min(MB, B - prev_row);
                HIP_CHECK(hipStreamWaitEvent(moe_stream, evt_attn_done[i - 1], 0));
                if(IS_20B_MODEL) moe_gpu(gpu_t, l, prev_bs, /*row_offset=*/prev_row, moe_stream);
                else moe_gpu_120b(gpu_t, l, prev_bs, /*row_offset=*/prev_row, moe_stream);
                HIP_CHECK(hipEventRecord(evt_moe_done_cur[i - 1], moe_stream));
            }
        }

        // drain last microbatch for this layer
        {
            const int i_last = NUM_MB - 1;
            const int row_last = i_last * MB;
            const int bs_last = std::min(MB, B - row_last);
            if (bs_last > 0)
            {
                HIP_CHECK(hipStreamWaitEvent(moe_stream, evt_attn_done[i_last], 0));
                if(IS_20B_MODEL) moe_gpu(gpu_t, l, bs_last, /*row_offset=*/row_last, moe_stream);
                else moe_gpu_120b(gpu_t, l, bs_last, /*row_offset=*/row_last, moe_stream);

                HIP_CHECK(hipEventRecord(evt_moe_done_cur[i_last], moe_stream));
            }
        }
        // make next layer depend on current layer's MoE-done (per microbatch)
        std::swap(evt_moe_done_prev, evt_moe_done_cur);
        for (int i = 0; i < NUM_MB; ++i)
        {
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
        // TIMER_BLOCK("final_matmul");
        matmul_vec128_singlebuf<
            16,16,16,
            4,4,4,
            1,2,
            8,
            false>(s->logits, s->x, w->out, B, H, p->vocab_size);
        HIP_CHECK(hipGetLastError());
    }
    {
        // TIMER_BLOCK("sample_argmax");
        sample_argmax(s->logits, s->current_tokens, B, p->vocab_size);
        HIP_CHECK(hipGetLastError());
    }

    // D2H tokens
    d2h_copy(cpu_buf->current_tokens, s->current_tokens, (size_t)B * sizeof(int), 0);

    // cleanup
    for (int i = 0; i < NUM_MB; ++i)
    {
        HIP_CHECK(hipEventDestroy(evt_attn_done[i]));
        HIP_CHECK(hipEventDestroy(evt_moe_done_prev[i]));
        HIP_CHECK(hipEventDestroy(evt_moe_done_cur[i]));
    }
    HIP_CHECK(hipStreamDestroy(attn_stream));
    HIP_CHECK(hipStreamDestroy(moe_stream));

    return cpu_buf->current_tokens;
}

// Replace previous clear_kv_cache_for_slot kernel with host memset version
static inline void clear_kv_cache_for_slot(GPURunState *s, const Config *p, int slot)
{
    return;
    if (slot < 0 || slot >= BATCH_SIZE)
        return;

    const int kv_dim = p->head_dim * p->n_kv_heads;
    const size_t elem_bytes = sizeof(float);

    const int even_cap = (p->sliding_window > 0 ? SW_WINDOW : MAX_SEQ_LEN);

    size_t layers_capacity = 0;
    for (int L = 0; L < p->n_layers; ++L)
    {
        const bool evenL = ((L & 1) == 0);
        layers_capacity += (size_t)(evenL ? even_cap : MAX_SEQ_LEN);
    }
    const size_t slot_base_elems = (size_t)slot * layers_capacity * (size_t)kv_dim;

    size_t layer_pos_offset = 0;
    for (int L = 0; L < p->n_layers; ++L)
    {
        const bool evenL = ((L & 1) == 0);
        const size_t capL = (size_t)(evenL ? even_cap : MAX_SEQ_LEN);

        const size_t layer_base_elems = slot_base_elems + layer_pos_offset * (size_t)kv_dim;
        const size_t bytes_this_layer = capL * (size_t)kv_dim * elem_bytes;

        void *k_ptr = (void *)((char *)s->key_cache + layer_base_elems * elem_bytes);
        void *v_ptr = (void *)((char *)s->value_cache + layer_base_elems * elem_bytes);

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

        cpu_buf->next_request_idx = start_request;

        for (int slot = 0; slot < BATCH_SIZE; slot++)
        {
            cpu_buf->slot_active_cpu[slot] = false;
            cpu_buf->request_mapping_cpu[slot] = -1;
            cpu_buf->seq_lengths_cpu[slot] = 0;
            cpu_buf->positions[slot] = 0;
            cpu_buf->finished[slot] = true;
            cpu_buf->current_tokens[slot] = BOS_TOKEN_ID;
        }

        // Fill initial batch
        for (int slot = 0; slot < BATCH_SIZE && cpu_buf->next_request_idx < end_request; slot++)
        {
            int req_idx = cpu_buf->next_request_idx++;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            encode(tokenizer, input_seq, -1, -1,
                   cpu_buf->prompt_tokens[slot],
                   &cpu_buf->prompt_lens[slot],
                   p->initial_context_length);

            if (cpu_buf->prompt_lens[slot] < 1)
            {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                cpu_buf->prompt_lens[slot] = 1;
                cpu_buf->prompt_tokens[slot][0] = BOS_TOKEN_ID;
            }

            cpu_buf->request_mapping_cpu[slot] = req_idx;
            cpu_buf->slot_active_cpu[slot] = true;
            cpu_buf->seq_lengths_cpu[slot] = 0;
            cpu_buf->positions[slot] = 0;
            cpu_buf->finished[slot] = false;
            cpu_buf->current_tokens[slot] = cpu_buf->prompt_tokens[slot][0];
        }

        // Push initial slot metadata to device
        HIP_CHECK(hipMemcpy(state->slot_active, cpu_buf->slot_active_cpu, BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->request_mapping, cpu_buf->request_mapping_cpu, BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->seq_lengths, cpu_buf->seq_lengths_cpu, BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(state->positions, cpu_buf->positions, BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));

        const int max_steps = requests->max_seq_len;

        while (true)
        {
            bool any_active = false;
            for (int slot = 0; slot < BATCH_SIZE; ++slot)
            {
                if (cpu_buf->slot_active_cpu[slot])
                {
                    any_active = true;
                    break;
                }
            }
            if (!any_active)
                break;

            for (int slot = 0; slot < BATCH_SIZE; ++slot)
            {
                if (!cpu_buf->slot_active_cpu[slot])
                {
                    cpu_buf->positions[slot] = 0;
                    cpu_buf->current_tokens[slot] = BOS_TOKEN_ID;
                }
            }

            int *next_tokens = forward_batch_gpu(gpu_t, cpu_buf->current_tokens, BATCH_SIZE);

            for (int slot = 0; slot < BATCH_SIZE; slot++)
            {
                if (!cpu_buf->slot_active_cpu[slot])
                    continue;

                int req_idx = cpu_buf->request_mapping_cpu[slot];
                int pos = cpu_buf->positions[slot];

                pos++;

                int next_token;
                if (pos < cpu_buf->prompt_lens[slot])
                {
                    next_token = cpu_buf->prompt_tokens[slot][pos];
                }
                else
                {
                    next_token = next_tokens[slot];
                    int *out = get_tok_gen_ptr(requests, req_idx);
                    const int gen_pos = pos - cpu_buf->prompt_lens[slot];
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        out[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }

                const bool completed =
                    (next_token == 199999 || next_token == 200002 ||
                     pos >= max_steps - 1 || pos >= MAX_SEQ_LEN - 2);

                if (completed)
                {
                    int *out = get_tok_gen_ptr(requests, req_idx);
                    const int gen_pos = pos - cpu_buf->prompt_lens[slot] + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        out[gen_pos] = -1;
                    }

                    if (cpu_buf->next_request_idx < end_request)
                    {
                        clear_kv_cache_for_slot(&gpu_t->state, &gpu_t->config, slot);

                        req_idx = cpu_buf->next_request_idx++;
                        const char *input_seq = get_str_req_ptr(requests, req_idx);

                        encode(tokenizer, input_seq, -1, -1,
                               cpu_buf->prompt_tokens[slot],
                               &cpu_buf->prompt_lens[slot],
                               p->initial_context_length);

                        if (cpu_buf->prompt_lens[slot] < 1)
                        {
                            fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                            cpu_buf->prompt_lens[slot] = 1;
                            cpu_buf->prompt_tokens[slot][0] = 1;
                        }

                        cpu_buf->request_mapping_cpu[slot] = req_idx;
                        cpu_buf->positions[slot] = 0;
                        cpu_buf->seq_lengths_cpu[slot] = 0;
                        cpu_buf->finished[slot] = false;
                        cpu_buf->current_tokens[slot] = cpu_buf->prompt_tokens[slot][0];
                    }
                    else
                    {
                        cpu_buf->slot_active_cpu[slot] = false;
                        cpu_buf->request_mapping_cpu[slot] = -1;
                        cpu_buf->positions[slot] = 0;
                        cpu_buf->seq_lengths_cpu[slot] = 0;
                        cpu_buf->current_tokens[slot] = 1;
                    }
                }
                else
                {
                    cpu_buf->positions[slot] = pos;
                    cpu_buf->seq_lengths_cpu[slot]++;
                    cpu_buf->current_tokens[slot] = next_token;
                }
            }

            HIP_CHECK(hipMemcpy(state->slot_active, cpu_buf->slot_active_cpu, BATCH_SIZE * sizeof(bool), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(state->seq_lengths, cpu_buf->seq_lengths_cpu, BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(state->positions, cpu_buf->positions, BATCH_SIZE * sizeof(int), hipMemcpyHostToDevice));
        }
    }

    write_profile_info();
    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    return continuous_batching_inference(tokenizer, sampler, requests);
    
}

#endif // GETP_RUN
