#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_HIP(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d - %s\n", \
                __FILE__, __LINE__, hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

typedef __hip_bfloat16 bf16;

inline __device__ __host__ bf16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

inline __device__ __host__ float bf16_to_float(bf16 b) {
    return __bfloat162float(b);
}

struct GPUWeights {
    bf16 *token_embedding_table;
    bf16 *rms_attn_w;
    bf16 *rms_ffn_w;
    bf16 *w_qkv;
    bf16 *w_o;
    bf16 *b_qkv;
    bf16 *b_o;
    bf16 *attn_sinks;
    bf16 *w_router;
    bf16 *b_router;
    bf16 *w_mlp1;
    bf16 *w_mlp2;
    bf16 *b_mlp1;
    bf16 *b_mlp2;
    bf16 *rms_out_w;
    bf16 *out;
};

struct GPURunState {
    bf16 *x;
    bf16 *t;
    bf16 *tb;
    bf16 *tb2;
    bf16 *router_score;
    bf16 *topk_v;
    int *topk_i;
    bf16 *mlp1_out;
    bf16 *gate;
    bf16 *up;
    bf16 *gate_up;
    bf16 *e_agg;
    bf16 *qkv;
    bf16 *q;
    bf16 *k;
    bf16 *v;
    bf16 *att;
    bf16 *logits;
    bf16 *key_cache;
    bf16 *value_cache;
    bf16 *mask;
};

void allocate_gpu_weights(GPUWeights *gw, Config *p);
void allocate_gpu_runstate(GPURunState *gs, Config *p);
void copy_weights_to_gpu(GPUWeights *gw, TransformerWeights *w, Config *p);
void copy_runstate_to_gpu(GPURunState *gs, RunState *s, Config *p);
void copy_logits_from_gpu(float *cpu_logits, bf16 *gpu_logits, int batch_size, int vocab_size);
void free_gpu_weights(GPUWeights *gw);
void free_gpu_runstate(GPURunState *gs);