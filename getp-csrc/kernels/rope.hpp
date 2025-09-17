#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void apply_rotary_emb_kernel(float *x, const float *cos_vals, const float *sin_vals,
                                        const int *seq_lengths, int batch_size,
                                        int n_heads, int head_dim)
{
    size_t batch_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int half = head_dim / 2;
    if (dim_idx >= half)
        return;

    int pos = seq_lengths[batch_idx];
    float *x_batch = x + batch_idx * n_heads * head_dim;
    const float *cos_pos = cos_vals + pos * half;
    const float *sin_pos = sin_vals + pos * half;

    float x1 = x_batch[head_idx * head_dim + dim_idx];
    float x2 = x_batch[head_idx * head_dim + half + dim_idx];

    float c = cos_pos[dim_idx];
    float s = sin_pos[dim_idx];

    x_batch[head_idx * head_dim + dim_idx] = x1 * c - x2 * s;
    x_batch[head_idx * head_dim + half + dim_idx] = x2 * c + x1 * s;
}


// Fused split + rotary (Q, K rotated; V copied)
// Layout in qkv per batch: [ Q (Hq*D) | K (Hkv*D) | V (Hkv*D) ]
template<int V>
__global__ void split_qkv_apply_rotary_kernel_vec(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ cos_vals,   // [*, D/2]
    const float* __restrict__ sin_vals,   // [*, D/2]
    const int*   __restrict__ seq_lengths,  // [B]
    int B, int Hq, int Hkv, int D)
{
    // Requirements:
    // - D must be even (for rotary pair split)
    // - V must divide D and also divide D/2 for the Q/K loops
    const int half = D >> 1;
    const size_t Dv_half = (size_t)half / V;   // vector tiles across the first half
    const size_t Dv_full = (size_t)D    / V;   // for V loop

    const size_t stride   = (size_t)(Hq + 2*Hkv) * D;
    const size_t q_off    = 0;
    const size_t k_off    = (size_t)Hq * D;
    const size_t v_off    = (size_t)(Hq + Hkv) * D;

    const size_t idx0 = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride_threads = (size_t)gridDim.x * blockDim.x;

    // ----- Q (apply rotary) -----
    if (Dv_half) {
        const size_t total_v = (size_t)B * Dv_half; // per-head vector tiles
        for (size_t i = idx0; i < (size_t)Hq * total_v; i += stride_threads) {
            size_t h   = i / total_v;
            size_t r   = i - h * total_v;
            size_t b   = r / Dv_half;
            size_t dv  = r - b * Dv_half;          // vector index along half
            int    pos = seq_lengths[b];
            size_t d   = dv * V;                    // scalar index into half

            const float* __restrict__ src0 = qkv + b*stride + q_off + h*(size_t)D + d;         // first half
            const float* __restrict__ src1 = src0 + half;                                       // second half
            float*       __restrict__ dst0 = q   + b*(size_t)Hq*D + h*(size_t)D + d;
            float*       __restrict__ dst1 = dst0 + half;

            // Load vectors from the two halves
            float a[V], b2[V]; // a = first half, b2 = second half
            #pragma unroll
            for (int t=0; t<V; ++t) {
                a[t]  = src0[t];
                b2[t] = src1[t];
            }

            // Apply rotary per element with cos/sin at this token position
            const float* __restrict__ cos_pos = cos_vals + (size_t)pos * half + d;
            const float* __restrict__ sin_pos = sin_vals + (size_t)pos * half + d;

            float y0[V], y1[V];
            #pragma unroll
            for (int t=0; t<V; ++t) {
                float c = cos_pos[t];
                float s = sin_pos[t];
                y0[t] = a[t] * c - b2[t] * s;  // rotated first half
                y1[t] = b2[t] * c + a[t] * s;  // rotated second half
            }

            // Store back
            #pragma unroll
            for (int t=0; t<V; ++t) {
                dst0[t] = y0[t];
                dst1[t] = y1[t];
            }
        }
    }

    // ----- K (apply rotary) -----
    if (Dv_half) {
        const size_t total_v = (size_t)B * Dv_half;
        for (size_t i = idx0; i < (size_t)Hkv * total_v; i += stride_threads) {
            size_t h   = i / total_v;
            size_t r   = i - h * total_v;
            size_t b   = r / Dv_half;
            size_t dv  = r - b * Dv_half;
            int    pos = seq_lengths[b];
            size_t d   = dv * V;

            const float* __restrict__ src0 = qkv + b*stride + k_off + h*(size_t)D + d;
            const float* __restrict__ src1 = src0 + half;
            float*       __restrict__ dst0 = k   + b*(size_t)Hkv*D + h*(size_t)D + d;
            float*       __restrict__ dst1 = dst0 + half;

            float a[V], b2[V];
            #pragma unroll
            for (int t=0; t<V; ++t) {
                a[t]  = src0[t];
                b2[t] = src1[t];
            }

            const float* __restrict__ cos_pos = cos_vals + (size_t)pos * half + d;
            const float* __restrict__ sin_pos = sin_vals + (size_t)pos * half + d;

            float y0[V], y1[V];
            #pragma unroll
            for (int t=0; t<V; ++t) {
                float c = cos_pos[t];
                float s = sin_pos[t];
                y0[t] = a[t] * c - b2[t] * s;
                y1[t] = b2[t] * c + a[t] * s;
            }

            #pragma unroll
            for (int t=0; t<V; ++t) {
                dst0[t] = y0[t];
                dst1[t] = y1[t];
            }
        }
    }

    // ----- V (just copy full D) -----
    if (Dv_full) {
        const size_t total_v = (size_t)B * Dv_full;
        for (size_t i = idx0; i < (size_t)Hkv * total_v; i += stride_threads) {
            size_t h   = i / total_v;
            size_t r   = i - h * total_v;
            size_t b   = r / Dv_full;
            size_t dv  = r - b * Dv_full;
            size_t d   = dv * V;

            const float* __restrict__ src = qkv + b*stride + v_off + h*(size_t)D + d;
            float*       __restrict__ dst = v   + b*(size_t)Hkv*D + h*(size_t)D + d;

            #pragma unroll
            for (int t=0; t<V; ++t) dst[t] = src[t];
        }
    }
}


// Dispatcher: prefer float4, then float2, else scalar
inline void launch_split_qkv_apply_rotary(
    const float* qkv, float* q, float* k, float* v,
    const float* cos_vals, const float* sin_vals, const int* seq_lengths,
    int B, int Hq, int Hkv, int D,
    hipStream_t stream = 0)
{
    const int block = 256;

    // Estimate total vector work items; pick a grid that keeps CUs busy.
    auto work_q = (size_t)Hq  * (size_t)B * (size_t)(D/2);
    auto work_k = (size_t)Hkv * (size_t)B * (size_t)(D/2);
    auto work_v = (size_t)Hkv * (size_t)B * (size_t) D;
    size_t work = work_q + work_k + work_v;

    if ((D % 4) == 0) {
        size_t grid = ((work/4) + block - 1) / block; if (!grid) grid = 1;
        hipLaunchKernelGGL(split_qkv_apply_rotary_kernel_vec<4>,
            dim3((unsigned)grid), dim3(block), 0, stream,
            qkv, q, k, v, cos_vals, sin_vals, seq_lengths, B, Hq, Hkv, D);
    } else if ((D % 2) == 0) {
        size_t grid = ((work/2) + block - 1) / block; if (!grid) grid = 1;
        hipLaunchKernelGGL(split_qkv_apply_rotary_kernel_vec<2>,
            dim3((unsigned)grid), dim3(block), 0, stream,
            qkv, q, k, v, cos_vals, sin_vals, seq_lengths, B, Hq, Hkv, D);
    } else {
        size_t grid = ( work     + block - 1) / block; if (!grid) grid = 1;
        hipLaunchKernelGGL(split_qkv_apply_rotary_kernel_vec<1>,
            dim3((unsigned)grid), dim3(block), 0, stream,
            qkv, q, k, v, cos_vals, sin_vals, seq_lengths, B, Hq, Hkv, D);
    }
}
