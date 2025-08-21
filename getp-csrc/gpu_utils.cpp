#include "../include/gpu_utils.hpp"
#include "../include/configs.hpp"
#include <cstring>

extern int **prompt_tokens;
extern int *current_tokens;
extern bool *finished;
extern int *positions;
extern int *prompt_lens;

void allocate_gpu_weights(GPUWeights *gw, Config *p) {
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;
    size_t n_layers = p->n_layers;
    size_t head_dim = p->head_dim;
    size_t n_attn_heads = p->n_attn_heads;
    size_t n_kv_heads = p->n_kv_heads;
    size_t n_experts = p->n_experts;
    size_t intermediate_dim = p->intermediate_dim;
    
    CHECK_HIP(hipMalloc(&gw->token_embedding_table, vocab_size * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->rms_attn_w, n_layers * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->rms_ffn_w, n_layers * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->w_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->w_o, n_layers * hidden_dim * head_dim * n_attn_heads * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->b_qkv, n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->b_o, n_layers * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->attn_sinks, n_layers * n_attn_heads * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->w_router, n_layers * hidden_dim * n_experts * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->b_router, n_layers * n_experts * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->w_mlp1, n_layers * n_experts * 2 * intermediate_dim * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->w_mlp2, n_layers * n_experts * hidden_dim * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->b_mlp1, n_layers * n_experts * 2 * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->b_mlp2, n_layers * n_experts * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->rms_out_w, hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gw->out, vocab_size * hidden_dim * sizeof(bf16)));
}

void allocate_gpu_runstate(GPURunState *gs, Config *p) {
    size_t hidden_dim = p->hidden_dim;
    size_t head_dim = p->head_dim;
    size_t n_attn_heads = p->n_attn_heads;
    size_t n_kv_heads = p->n_kv_heads;
    size_t n_experts = p->n_experts;
    size_t experts_per_token = p->experts_per_token;
    size_t intermediate_dim = p->intermediate_dim;
    size_t vocab_size = p->vocab_size;
    size_t seq_len = p->seq_len;
    size_t n_layers = p->n_layers;
    size_t kv_dim = head_dim * n_kv_heads;
    
    CHECK_HIP(hipMalloc(&gs->x, BATCH_SIZE * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->t, BATCH_SIZE * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->tb, BATCH_SIZE * head_dim * n_attn_heads * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->tb2, BATCH_SIZE * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->router_score, BATCH_SIZE * n_experts * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->topk_v, BATCH_SIZE * experts_per_token * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->topk_i, BATCH_SIZE * experts_per_token * sizeof(int)));
    CHECK_HIP(hipMalloc(&gs->mlp1_out, BATCH_SIZE * 2 * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->gate, BATCH_SIZE * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->up, BATCH_SIZE * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->gate_up, BATCH_SIZE * intermediate_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->e_agg, BATCH_SIZE * hidden_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->qkv, BATCH_SIZE * head_dim * (n_attn_heads + 2 * n_kv_heads) * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->q, BATCH_SIZE * n_attn_heads * head_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->k, BATCH_SIZE * kv_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->v, BATCH_SIZE * kv_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->key_cache, BATCH_SIZE * n_layers * seq_len * kv_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->value_cache, BATCH_SIZE * n_layers * seq_len * kv_dim * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->att, BATCH_SIZE * n_attn_heads * seq_len * sizeof(bf16)));
    CHECK_HIP(hipMalloc(&gs->logits, BATCH_SIZE * vocab_size * sizeof(bf16)));
    
    if (p->sliding_window > 0) {
        CHECK_HIP(hipMalloc(&gs->mask, seq_len * seq_len * sizeof(bf16)));
    } else {
        gs->mask = nullptr;
    }
}

__global__ void convert_fp32_to_bf16_kernel(bf16 *dst, float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = float_to_bf16(src[idx]);
    }
}

void copy_weights_to_gpu(GPUWeights *gw, TransformerWeights *w, Config *p) {
    size_t vocab_size = p->vocab_size;
    size_t hidden_dim = p->hidden_dim;
    size_t n_layers = p->n_layers;
    size_t head_dim = p->head_dim;
    size_t n_attn_heads = p->n_attn_heads;
    size_t n_kv_heads = p->n_kv_heads;
    size_t n_experts = p->n_experts;
    size_t intermediate_dim = p->intermediate_dim;
    
    size_t block_size = 256;
    size_t n;
    
    // Convert and copy token_embedding_table
    n = vocab_size * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->token_embedding_table, w->token_embedding_table, n);
    
    // Convert and copy rms_attn_w
    n = n_layers * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->rms_attn_w, w->rms_attn_w, n);
    
    // Convert and copy rms_ffn_w
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->rms_ffn_w, w->rms_ffn_w, n);
    
    // Convert and copy w_qkv
    n = n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->w_qkv, w->w_qkv, n);
    
    // Convert and copy w_o
    n = n_layers * hidden_dim * head_dim * n_attn_heads;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->w_o, w->w_o, n);
    
    // Convert and copy b_qkv
    n = n_layers * (head_dim * n_attn_heads + 2 * head_dim * n_kv_heads);
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->b_qkv, w->b_qkv, n);
    
    // Convert and copy b_o
    n = n_layers * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->b_o, w->b_o, n);
    
    // Convert and copy attn_sinks
    n = n_layers * n_attn_heads;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->attn_sinks, w->attn_sinks, n);
    
    // Convert and copy w_router
    n = n_layers * hidden_dim * n_experts;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->w_router, w->w_router, n);
    
    // Convert and copy b_router
    n = n_layers * n_experts;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->b_router, w->b_router, n);
    
    // Convert and copy w_mlp1
    n = n_layers * n_experts * 2 * intermediate_dim * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->w_mlp1, w->w_mlp1, n);
    
    // Convert and copy w_mlp2
    n = n_layers * n_experts * hidden_dim * intermediate_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->w_mlp2, w->w_mlp2, n);
    
    // Convert and copy b_mlp1
    n = n_layers * n_experts * 2 * intermediate_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->b_mlp1, w->b_mlp1, n);
    
    // Convert and copy b_mlp2
    n = n_layers * n_experts * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->b_mlp2, w->b_mlp2, n);
    
    // Convert and copy rms_out_w
    n = hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->rms_out_w, w->rms_out_w, n);
    
    // Convert and copy out
    n = vocab_size * hidden_dim;
    convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        gw->out, w->out, n);
    
    CHECK_HIP(hipDeviceSynchronize());
}

void copy_runstate_to_gpu(GPURunState *gs, RunState *s, Config *p) {
    // Initialize mask if needed
    if (p->sliding_window > 0 && s->mask) {
        size_t n = p->seq_len * p->seq_len;
        size_t block_size = 256;
        convert_fp32_to_bf16_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
            gs->mask, s->mask, n);
        CHECK_HIP(hipDeviceSynchronize());
    }
}

__global__ void convert_bf16_to_fp32_kernel(float *dst, bf16 *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = bf16_to_float(src[idx]);
    }
}

void copy_logits_from_gpu(float *cpu_logits, bf16 *gpu_logits, int batch_size, int vocab_size) {
    size_t n = batch_size * vocab_size;
    size_t block_size = 256;
    convert_bf16_to_fp32_kernel<<<(n + block_size - 1) / block_size, block_size>>>(
        cpu_logits, gpu_logits, n);
    CHECK_HIP(hipDeviceSynchronize());
}

void free_gpu_weights(GPUWeights *gw) {
    hipFree(gw->token_embedding_table);
    hipFree(gw->rms_attn_w);
    hipFree(gw->rms_ffn_w);
    hipFree(gw->w_qkv);
    hipFree(gw->w_o);
    hipFree(gw->b_qkv);
    hipFree(gw->b_o);
    hipFree(gw->attn_sinks);
    hipFree(gw->w_router);
    hipFree(gw->b_router);
    hipFree(gw->w_mlp1);
    hipFree(gw->w_mlp2);
    hipFree(gw->b_mlp1);
    hipFree(gw->b_mlp2);
    hipFree(gw->rms_out_w);
    hipFree(gw->out);
}

void free_gpu_runstate(GPURunState *gs) {
    hipFree(gs->x);
    hipFree(gs->t);
    hipFree(gs->tb);
    hipFree(gs->tb2);
    hipFree(gs->router_score);
    hipFree(gs->topk_v);
    hipFree(gs->topk_i);
    hipFree(gs->mlp1_out);
    hipFree(gs->gate);
    hipFree(gs->up);
    hipFree(gs->gate_up);
    hipFree(gs->e_agg);
    hipFree(gs->qkv);
    hipFree(gs->q);
    hipFree(gs->k);
    hipFree(gs->v);
    hipFree(gs->att);
    hipFree(gs->logits);
    hipFree(gs->key_cache);
    hipFree(gs->value_cache);
    if (gs->mask) hipFree(gs->mask);
}