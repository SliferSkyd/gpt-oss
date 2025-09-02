#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/fp8_e4m3.hpp"

// FP8-aware fused attention kernel
__global__ void fused_attention_kernel_fp8(
    float * __restrict__ output,               // [batch, n_heads, head_dim]
    const float * __restrict__ q,              // [batch, n_heads * head_dim]
    const uint8_t * __restrict__ key_cache,    // FP8 [batch, n_layers, seq_len, kv_dim]
    const uint8_t * __restrict__ value_cache,  // FP8 [batch, n_layers, seq_len, kv_dim]
    const __hip_bfloat16 * __restrict__ key_cache_bf16,   // BF16 for first tokens
    const __hip_bfloat16 * __restrict__ value_cache_bf16, // BF16 for first tokens
    const __hip_bfloat16 * __restrict__ sinks,       // [n_heads]
    const float * __restrict__ mask,                 // [seq_len, seq_len]
    const int * __restrict__ positions,              // [batch]
    const float * __restrict__ kv_scale,             // [n_layers * 2] quantization scales
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx,
    bool use_sliding_window)
{
    extern __shared__ float s_sh[];  // layout later
    
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (b >= batch_size || h >= n_heads) return;
    
    // Wave bookkeeping
    const int WARP  = 64;  // AMD warpSize
    const int WARPS = (blockDim.x + WARP - 1) / WARP;
    const int lane  = tid & (WARP - 1);
    const int wid   = tid / WARP;
    
    // Pointers / shapes
    const int pos        = positions[b];
    if (pos < 0 || pos >= seq_len) return;  // Safety check
    
    const int gqa_ratio  = n_heads / n_kv_heads;
    const int kv_h       = h / gqa_ratio;
    const int kv_dim     = head_dim * n_kv_heads;
    
    // Get quantization scales for this layer (with safety)
    const float k_inv_scale = (kv_scale != nullptr) ? (1.0f / kv_scale[layer_idx * 2]) : 1.0f;
    const float v_inv_scale = (kv_scale != nullptr) ? (1.0f / kv_scale[layer_idx * 2 + 1]) : 1.0f;
    
    const float *q_head = q + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;
    
    // Base pointers for FP8 caches
    const uint8_t *k_base_fp8 = 
        key_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    const uint8_t *v_base_fp8 = 
        value_cache + 1LL * b * n_layers * seq_len * kv_dim + 1LL * layer_idx * seq_len * kv_dim;
    
    // Base pointers for BF16 caches (first few tokens)
    const __hip_bfloat16 *k_base_bf16 = nullptr;
    const __hip_bfloat16 *v_base_bf16 = nullptr;
    if (FP8_MIXED_PRECISION_TOKENS > 0) {
        k_base_bf16 = key_cache_bf16 + 1LL * b * n_layers * FP8_MIXED_PRECISION_TOKENS * kv_dim + 
                     1LL * layer_idx * FP8_MIXED_PRECISION_TOKENS * kv_dim;
        v_base_bf16 = value_cache_bf16 + 1LL * b * n_layers * FP8_MIXED_PRECISION_TOKENS * kv_dim + 
                     1LL * layer_idx * FP8_MIXED_PRECISION_TOKENS * kv_dim;
    }
    
    float *out_head = output + 1LL * b * n_heads * head_dim + 1LL * h * head_dim;
    
    // Windowing + capacities
    const bool apply_window = use_sliding_window && ((layer_idx & 1) == 0);
    const int  win_start    = apply_window ? max(0, pos - 127) : 0;  // 128 token window
    const int  win_core_len = pos - win_start + 1;
    const int  att_cap      = apply_window ? 129 : seq_len;
    
    float *s_att      = s_sh;
    float *s_partials = s_att + att_cap;
    float *s_reduce   = s_partials + WARPS * head_dim;
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    
    // Fast path: map one lane per head-dim element
    const float q_lane = (lane < head_dim) ? q_head[lane] : 0.0f;
    
    // Pass 1: compute attention scores
    for (int w = wid; w < win_core_len; w += WARPS) {
        const int t = win_start + w;
        
        float prod = 0.0f;
        if (lane < head_dim) {
            float k_val = 0.0f;  // Initialize
            
            // Use BF16 cache for first few tokens, FP8 for the rest
            if (t < FP8_MIXED_PRECISION_TOKENS && k_base_bf16 != nullptr) {
                const __hip_bfloat16 *k_vec = k_base_bf16 + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
                k_val = __bfloat162float(k_vec[lane]);
            } else {
                const uint8_t *k_vec = k_base_fp8 + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
                k_val = fp8_e4m3::fp8_e4m3_to_float(k_vec[lane]) * k_inv_scale;
            }
            
            prod = q_lane * k_val;
        }
        
        // Warp reduction
        for (int off = WARP >> 1; off > 0; off >>= 1) {
            prod += __shfl_down(prod, off);
        }
        
        if (lane == 0) {
            float acc = prod * inv_sqrt_d;
            if (apply_window) {
                acc += mask[1LL * pos * seq_len + t];
            }
            s_att[w] = acc;
        }
    }
    __syncthreads();
    
    // Append sink if needed
    int softmax_len = win_core_len;
    if (pos + 1 < seq_len) {
        if (tid == 0) s_att[softmax_len] = __bfloat162float(sinks[h]);
        softmax_len += 1;
    }
    __syncthreads();
    
    // Softmax
    float tmax = -INFINITY;
    for (int i = tid; i < softmax_len; i += blockDim.x) {
        tmax = fmaxf(tmax, s_att[i]);
    }
    
    // Block reduction for max
    s_reduce[tid] = tmax;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + offset]);
        }
        __syncthreads();
    }
    const float max_val = s_reduce[0];
    __syncthreads();
    
    // Compute exp and sum
    float tsum = 0.0f;
    for (int i = tid; i < softmax_len; i += blockDim.x) {
        float v = expf(s_att[i] - max_val);
        s_att[i] = v;
        tsum += v;
    }
    
    // Block reduction for sum
    s_reduce[tid] = tsum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_reduce[tid] += s_reduce[tid + offset];
        }
        __syncthreads();
    }
    const float sum_val = s_reduce[0];
    const float inv_sum = 1.0f / (sum_val + 1e-9f);
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < softmax_len; i += blockDim.x) {
        s_att[i] *= inv_sum;
    }
    __syncthreads();
    
    // Pass 2: V-weighted sum
    if (lane < head_dim) {
        float partial = 0.0f;
        for (int w = wid; w < win_core_len; w += WARPS) {
            const int t = win_start + w;
            float v_val = 0.0f;  // Initialize
            
            // Use BF16 cache for first few tokens, FP8 for the rest
            if (t < FP8_MIXED_PRECISION_TOKENS && v_base_bf16 != nullptr) {
                const __hip_bfloat16 *v_vec = v_base_bf16 + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
                v_val = __bfloat162float(v_vec[lane]);
            } else {
                const uint8_t *v_vec = v_base_fp8 + 1LL * t * kv_dim + 1LL * kv_h * head_dim;
                v_val = fp8_e4m3::fp8_e4m3_to_float(v_vec[lane]) * v_inv_scale;
            }
            
            partial += s_att[w] * v_val;
        }
        s_partials[wid * head_dim + lane] = partial;
    }
    __syncthreads();
    
    // Cross-warp reduce
    if (wid == 0 && lane < head_dim) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            acc += s_partials[w * head_dim + lane];
        }
        out_head[lane] = acc;
    }
}