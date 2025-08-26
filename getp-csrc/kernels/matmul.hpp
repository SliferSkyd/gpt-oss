// fmaf_tiled_gemm.hpp
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_cooperative_groups.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"

// ===== Tunables =====
#ifndef WM
#define WM 16
#endif
#ifndef WN
#define WN 16
#endif
#ifndef WK
#define WK 16
#endif

#ifndef WAVES_M
#define WAVES_M 4
#endif
#ifndef WAVES_N
#define WAVES_N 4
#endif

// Block tile (keep these matching your previous config)
constexpr int BLOCK_M = WM * WAVES_M;  // default 64
constexpr int BLOCK_N = WN * WAVES_N;  // default 64
constexpr int BLOCK_K = WK;            // default 16

// Per-thread micro-tile (each thread computes TM x TN C-elements)
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif

static_assert(BLOCK_M % TM == 0, "BLOCK_M must be divisible by TM");
static_assert(BLOCK_N % TN == 0, "BLOCK_N must be divisible by TN");

// Thread block shape derived from micro-tiling
constexpr int TB_X = BLOCK_N / TN;   // e.g., 64/4 = 16
constexpr int TB_Y = BLOCK_M / TM;   // e.g., 64/4 = 16

// ===== Helpers =====

// Safe, branchless bit-cast bf16 -> f32
__device__ inline float bf16_to_f32(__hip_bfloat16 x) {
    uint16_t lo = *reinterpret_cast<const uint16_t*>(&x);
    uint32_t u = static_cast<uint32_t>(lo) << 16;
    union { uint32_t u; float f; } v{u};
    return v.f;
}

// (Optional) f32 -> bf16 with round-to-nearest-even, if you ever need it
__device__ inline __hip_bfloat16 f32_to_bf16(float f) {
    union { float f; uint32_t u; } v{f};
    // RNE: add 0x7FFF + LSB of upper to implement round-to-nearest-even
    uint32_t r = v.u + ((v.u >> 16) & 1U ? 0x7FFFU : 0x7FFFU);
    uint16_t hi = static_cast<uint16_t>(r >> 16);
    __hip_bfloat16 out;
    *reinterpret_cast<uint16_t*>(&out) = hi;
    return out;
}

// ===== fmaf GEMM: bf16 weights =====
__global__ void gemm_fmaf_bf16_kernel(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] fp32
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major (O x I)
    int M, int K, int N)
{
    // Shared tiles in fp32 for compute
    extern __shared__ float smem[];
    float* sA = smem;                                      // [BLOCK_M, BLOCK_K]
    float* sB = sA + (size_t)BLOCK_M * BLOCK_K;            // [BLOCK_K, BLOCK_N]

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lx = threadIdx.x;   // 0..TB_X-1
    const int ly = threadIdx.y;   // 0..TB_Y-1

    // Each thread computes a TM x TN micro-tile of C
    const int local_m0 = ly * TM;
    const int local_n0 = lx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; ++r)
      #pragma unroll
      for (int c = 0; c < TN; ++c)
        acc[r][c] = 0.0f;

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linT = ly * blockDim.x + lx;

    // K loop in BLOCK_K slices
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // ===== Load A tile: [BLOCK_M x BLOCK_K] from global (fp32) -> sA (fp32)
        for (int idx = linT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = k0 + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA[r * BLOCK_K + c] = a;
        }

        // ===== Load B tile: [BLOCK_K x BLOCK_N] from bf16 -> sB (fp32)
        // Keep sB row-major with K as the leading dim for nice access in compute.
        for (int idx = linT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int r = idx / BLOCK_N;  // 0..BLOCK_K-1 = kk within slice
            const int c = idx % BLOCK_N;  // 0..BLOCK_N-1 = n within block
            const int gk = k0 + r;
            const int gn = n0 + c;
            float b = 0.0f;
            if (gk < K && gn < N) {
                __hip_bfloat16 wb = Wbf16[(size_t)gn * K + gk]; // [N,K]
                b = bf16_to_f32(wb);
            }
            sB[r * BLOCK_N + c] = b;
        }

        __syncthreads();

        // ===== Compute TMxTN micro-tile with fmaf
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float aReg[TM];
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int lm = local_m0 + r;
                aReg[r] = sA[lm * BLOCK_K + kk];
            }

            float bReg[TN];
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                const int ln = local_n0 + c;
                bReg[c] = sB[kk * BLOCK_N + ln];
            }

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    acc[r][c] = fmaf(aReg[r], bReg[c], acc[r][c]);
                }
            }
        }

        __syncthreads();
    }

    // ===== Store
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int gm = m0 + local_m0 + r;
        if (gm >= M) break;
        #pragma unroll
        for (int c = 0; c < TN; ++c) {
            const int gn = n0 + local_n0 + c;
            if (gn < N) {
                C[(size_t)gm * N + gn] = acc[r][c];
            }
        }
    }
}

// ===== fmaf GEMM: MXFP4 packed weights (uint8) + scales =====
__global__ void gemm_fmaf_uint8_kernel(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K] fp32
    const uint8_t* __restrict__ Wp,               // [N, K] packed (MXFP4)
    const float* __restrict__ weight_scales,      // per your layout
    int M, int K, int N, size_t total_weight_elements)
{
    extern __shared__ float smem[];
    float* sA = smem;                                      // [BLOCK_M, BLOCK_K]
    float* sB = sA + (size_t)BLOCK_M * BLOCK_K;            // [BLOCK_K, BLOCK_N]

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int local_m0 = ly * TM;
    const int local_n0 = lx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; ++r)
      #pragma unroll
      for (int c = 0; c < TN; ++c)
        acc[r][c] = 0.0f;

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linT = ly * blockDim.x + lx;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // A -> sA
        for (int idx = linT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = k0 + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA[r * BLOCK_K + c] = a;
        }

        // Wp -> sB (dequantize to fp32)
        for (int idx = linT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int r = idx / BLOCK_N; // kk
            const int c = idx % BLOCK_N; // n
            const int gk = k0 + r;
            const int gn = n0 + c;
            float b = 0.0f;
            if (gk < K && gn < N) {
                const size_t widx = (size_t)gn * K + gk; // [N,K], row-major
                b = dequantize_mxfp4(Wp, weight_scales, widx, total_weight_elements);
            }
            sB[r * BLOCK_N + c] = b;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float aReg[TM];
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                const int lm = local_m0 + r;
                aReg[r] = sA[lm * BLOCK_K + kk];
            }

            float bReg[TN];
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                const int ln = local_n0 + c;
                bReg[c] = sB[kk * BLOCK_N + ln];
            }

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    acc[r][c] = fmaf(aReg[r], bReg[c], acc[r][c]);
                }
            }
        }

        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int gm = m0 + local_m0 + r;
        if (gm >= M) break;
        #pragma unroll
        for (int c = 0; c < TN; ++c) {
            const int gn = n0 + local_n0 + c;
            if (gn < N) {
                C[(size_t)gm * N + gn] = acc[r][c];
            }
        }
    }
}

// ===== Host wrappers (API unchanged) =====
inline void matmul(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const __hip_bfloat16* __restrict__ weight,  // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(TB_X, TB_Y); // (BLOCK_N/TN, BLOCK_M/TM)

    size_t shmem_bytes =
        (size_t)(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_fmaf_bf16_kernel,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
}

inline void matmul_mxfp4(
    float* __restrict__ output,
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight_packed,
    const float* __restrict__ weight_scales,
    int batch_size, int input_dim, int output_dim,
    size_t total_weight_elements,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(TB_X, TB_Y);

    size_t shmem_bytes =
        (size_t)(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_fmaf_uint8_kernel,
        grid, block, shmem_bytes, stream,
        output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
}
