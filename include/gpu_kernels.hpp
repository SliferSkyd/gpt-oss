#pragma once

#include "gpu_utils.hpp"

// BLAS GPU kernels
void matmul_batch_gpu(bf16 *xout, bf16 *x, bf16 *w, int batch_size, int n, int d);

// DNN GPU kernels
void rmsnorm_batch_gpu(bf16 *o, bf16 *x, bf16 *weight, int batch_size, int hidden_dim);
void rope_batch_gpu(bf16 *q, bf16 *k, float *cos_vals, float *sin_vals, 
                    int *positions, int batch_size, int head_dim, 
                    int n_attn_heads, int n_kv_heads);
void softmax_batch_gpu(bf16 *x, int batch_size, int n_heads, int seq_len, int *positions);
void swiglu_batch_gpu(bf16 *gate, float swiglu_limit, int batch_size, int intermediate_dim);
void attention_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p, 
                        unsigned long long l, int batch_size, int *positions);
void moe_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p,
                   unsigned long long l, int batch_size);

// Main forward pass on GPU
bf16* forward_batch_gpu(GPUWeights *gw, GPURunState *gs, Config *p,
                       int *tokens, int batch_size, int *positions);