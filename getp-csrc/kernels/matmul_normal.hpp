// mfma_gemm_gfx90a.hpp (FMA-only version)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"
#include <hip/hip_fp16.h>

// ===== Tile Tunables (FMA path) =====
// Workgroup computes BM x BN tile of C
#ifndef BM
#define BM 32
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif
// Per-thread micro-tile (TM x TN) -> BM/BN must be multiples of TM/TN * block dims
#ifndef TM
#define TM 2
#endif
#ifndef TN
#define TN 4
#endif

// Threadblock shape (TX x TY) so that TX*TM = BN and TY*TM = BM when mapping below
#ifndef TB_X
#define TB_X 16  // along N
#endif
#ifndef TB_Y
#define TB_Y 16  // along M
#endif

static_assert(BM % (TB_Y * TM) == 0 || (TB_Y * TM) == BM, "BM must be compatible with TB_Y and TM");
static_assert(BN % (TB_X * TN) == 0 || (TB_X * TN) == BN, "BN must be compatible with TB_X and TN");

__device__ __forceinline__ float bfloat16_to_float(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

// ====== FMA GEMM (A: FP32 row-major [M,K], B: BF16 row-major [N,K]) ======
__global__ void gemm_fma_bf16_kernel_opt(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K]
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major
    int M, int K, int N)
{
    // Tile origins
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    // Per-thread coordinates within the block
    const int tx = threadIdx.x; // 0..TB_X-1 maps across N
    const int ty = threadIdx.y; // 0..TB_Y-1 maps across M

    // Each thread computes a TM x TN microtile
    const int rowBase = m0 + ty * TM;
    const int colBase = n0 + tx * TN;

    // Double-buffered shared memory: sA: [BM x BK], sB: [BK x BN], both float32
    extern __shared__ float smem[];
    float* sA0 = smem;
    float* sA1 = sA0 + (BM * BK);
    float* sB0 = sA1 + (BM * BK);
    float* sB1 = sB0 + (BK * BN);

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    const int threadsPerBlock = TB_X * TB_Y;
    const int linearT = ty * TB_X + tx;

    // Helper lambdas to cooperatively load tiles from global -> shared
    auto loadA = [&](float* dst, int kBase) {
        // row-major [BM x BK]
        for (int idx = linearT; idx < BM * BK; idx += threadsPerBlock) {
            int r = idx / BK;
            int c = idx % BK;
            int gm = m0 + r;
            int gk = kBase + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            dst[r * BK + c] = a;
        }
    };
    auto loadB = [&](float* dst, int kBase) {
        // We store sB as row-major [BK x BN] for unit-stride kk access
        for (int idx = linearT; idx < BK * BN; idx += threadsPerBlock) {
            int r = idx / BN; // 0..BK-1  (k within this slice)
            int c = idx % BN; // 0..BN-1  (n within this block)
            int gk = kBase + r;
            int gn = n0 + c;
            float b = (gk < K && gn < N) ? bfloat16_to_float(Wbf16[(size_t)gn * K + gk]) : 0.0f;
            dst[r * BN + c] = b;
        }
    };

    // Preload k-slice 0
    loadA(sA0, /*kBase=*/0);
    loadB(sB0, /*kBase=*/0);
    __syncthreads();

    float* currA = sA0; float* nextA = sA1;
    float* currB = sB0; float* nextB = sB1;

    // Compute across K in BK chunks
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Preload next if any
        if (k0 + BK < K) {
            loadA(nextA, k0 + BK);
            loadB(nextB, k0 + BK);
        }

        // Compute: for kk in [0..BK)
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aFrag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                int rr = r; // within [0..BM)
                aFrag[i] = currA[rr * BK + kk];
            }

            float bFrag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j; // within [0..BN)
                bFrag[j] = currB[kk * BN + c];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += (double)aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
        // Swap buffers
        float* tA = currA; currA = nextA; nextA = tA;
        float* tB = currB; currB = nextB; nextB = tB;
    }

    // Store back to C with bounds checks
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = rowBase + i;
        if (gm >= M) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = colBase + j;
            if (gn < N) {
                C[(size_t)gm * N + gn] = acc[i][j];
            }
        }
    }
}

// ====== FMA GEMM (A: FP32 row-major [M,K], B: MXFP4 packed + scales -> dequant to FP32) ======
__global__ void gemm_fma_uint8_kernel_opt(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K]
    const uint8_t* __restrict__ Wp,               // [N, K] packed (MXFP4)
    const float* __restrict__ weight_scales,      // per-block scales (your layout)
    int M, int K, int N, size_t total_weight_elements)
{
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int rowBase = m0 + ty * TM;
    const int colBase = n0 + tx * TN;

    extern __shared__ float smem[];
    float* sA0 = smem;
    float* sA1 = sA0 + (BM * BK);
    float* sB0 = sA1 + (BM * BK);
    float* sB1 = sB0 + (BK * BN);

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int threadsPerBlock = TB_X * TB_Y;
    const int linearT = ty * TB_X + tx;

    auto loadA = [&](float* dst, int kBase) {
        for (int idx = linearT; idx < BM * BK; idx += threadsPerBlock) {
            int r = idx / BK;
            int c = idx % BK;
            int gm = m0 + r;
            int gk = kBase + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            dst[r * BK + c] = a;
        }
    };
    auto loadB = [&](float* dst, int kBase) {
        for (int idx = linearT; idx < BK * BN; idx += threadsPerBlock) {
            int r = idx / BN; // kk within this slice
            int c = idx % BN; // n within this block
            int gk = kBase + r;
            int gn = n0 + c;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[r * BN + c] = wb;
        }
    };

    loadA(sA0, 0);
    loadB(sB0, 0);
    __syncthreads();

    float* currA = sA0; float* nextA = sA1;
    float* currB = sB0; float* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            loadA(nextA, k0 + BK);
            loadB(nextB, k0 + BK);
        }

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aFrag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                aFrag[i] = currA[r * BK + kk];
            }
            float bFrag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j;
                bFrag[j] = currB[kk * BN + c];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += (double)aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
        float* tA = currA; currA = nextA; nextA = tA;
        float* tB = currB; currB = nextB; nextB = tB;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = rowBase + i;
        if (gm >= M) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = colBase + j;
            if (gn < N) {
                C[(size_t)gm * N + gn] = acc[i][j];
            }
        }
    }
}

// ===== Host wrappers (API unchanged) =====
void matmul_normal(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const __hip_bfloat16* __restrict__ weight,  // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(TB_X, TB_Y);

    // Double-buffered sA,sB
    size_t shmem_bytes = (size_t)(2 * (BM * BK + BK * BN)) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_fma_bf16_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
}

void matmul_mxfp4(
    float* __restrict__ output,
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight_packed,
    const float* __restrict__ weight_scales,
    int batch_size, int input_dim, int output_dim,
    size_t total_weight_elements,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(TB_X, TB_Y);

    size_t shmem_bytes = (size_t)(2 * (BM * BK + BK * BN)) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_fma_uint8_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
}



// ====== FMA GEMM (A: FP32 row-major [M,K], B: BF16 row-major [N,K]) ======
__global__ void gemm_fma_bf16_kernel_opt(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K]
    const float* __restrict__ Wbf16,  // [N, K] row-major
    int M, int K, int N)
{
    // Tile origins
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    // Per-thread coordinates within the block
    const int tx = threadIdx.x; // 0..TB_X-1 maps across N
    const int ty = threadIdx.y; // 0..TB_Y-1 maps across M

    // Each thread computes a TM x TN microtile
    const int rowBase = m0 + ty * TM;
    const int colBase = n0 + tx * TN;

    // Double-buffered shared memory: sA: [BM x BK], sB: [BK x BN], both float32
    extern __shared__ float smem[];
    float* sA0 = smem;
    float* sA1 = sA0 + (BM * BK);
    float* sB0 = sA1 + (BM * BK);
    float* sB1 = sB0 + (BK * BN);

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    const int threadsPerBlock = TB_X * TB_Y;
    const int linearT = ty * TB_X + tx;

    // Helper lambdas to cooperatively load tiles from global -> shared
    auto loadA = [&](float* dst, int kBase) {
        // row-major [BM x BK]
        for (int idx = linearT; idx < BM * BK; idx += threadsPerBlock) {
            int r = idx / BK;
            int c = idx % BK;
            int gm = m0 + r;
            int gk = kBase + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            dst[r * BK + c] = a;
        }
    };
    auto loadB = [&](float* dst, int kBase) {
        // We store sB as row-major [BK x BN] for unit-stride kk access
        for (int idx = linearT; idx < BK * BN; idx += threadsPerBlock) {
            int r = idx / BN; // 0..BK-1  (k within this slice)
            int c = idx % BN; // 0..BN-1  (n within this block)
            int gk = kBase + r;
            int gn = n0 + c;
            float b = (gk < K && gn < N) ? (Wbf16[(size_t)gn * K + gk]) : 0.0f;
            dst[r * BN + c] = b;
        }
    };

    // Preload k-slice 0
    loadA(sA0, /*kBase=*/0);
    loadB(sB0, /*kBase=*/0);
    __syncthreads();

    float* currA = sA0; float* nextA = sA1;
    float* currB = sB0; float* nextB = sB1;

    // Compute across K in BK chunks
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Preload next if any
        if (k0 + BK < K) {
            loadA(nextA, k0 + BK);
            loadB(nextB, k0 + BK);
        }

        // Compute: for kk in [0..BK)
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float aFrag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                int rr = r; // within [0..BM)
                aFrag[i] = currA[rr * BK + kk];
            }

            float bFrag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j; // within [0..BN)
                bFrag[j] = currB[kk * BN + c];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += (double)aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
        // Swap buffers
        float* tA = currA; currA = nextA; nextA = tA;
        float* tB = currB; currB = nextB; nextB = tB;
    }

    // Store back to C with bounds checks
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = rowBase + i;
        if (gm >= M) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = colBase + j;
            if (gn < N) {
                C[(size_t)gm * N + gn] = acc[i][j];
            }
        }
    }
}

// ===== Host wrappers (API unchanged) =====
void matmul_normal(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const float* __restrict__ weight,  // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(TB_X, TB_Y);

    // Double-buffered sA,sB
    size_t shmem_bytes = (size_t)(2 * (BM * BK + BK * BN)) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_fma_bf16_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
}
