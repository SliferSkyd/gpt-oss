#pragma once

// mfma_gemm_gfx90a.hpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"
#include <hip/hip_fp16.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// ===== Tunables (same defaults you had) =====
#ifndef WM
#define WM 16
#endif
#ifndef WN
#define WN 16
#endif
#ifndef WK
#define WK 16   // MFMA K-chunk is 16 for bf16 path
#endif

#ifndef WAVES_M
#define WAVES_M 4
#endif
#ifndef WAVES_N
#define WAVES_N 4
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M = WM * WAVES_M;        // e.g. 64 if WAVES_M=4
constexpr int BLOCK_N = WN * WAVES_N;        // e.g. 64 if WAVES_N=4
constexpr int BLOCK_K = WK;                  // 16
constexpr int LANE_PER_WAVE = 64;
constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

// ===== Small vector helpers (types that match MFMA signatures) =====
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4))); // 4×i16 carrying BF16 bit patterns

// ===== Conversions =====
__device__ inline uint16_t f32_to_bf16_bits(float x) {
    __hip_bfloat16 t = __float2bfloat16(x);
    return *reinterpret_cast<uint16_t*>(&t);
}

__device__ inline uint16_t hipbf16_to_bits(__hip_bfloat16 x) {
    return *reinterpret_cast<uint16_t*>(&x);
}

// ===== MFMA wrapper (gfx90a supports BF16->F32 accumulate) =====
// d = a*b + c, one 16x16x16 bf16 MMA per wave (returns 4 accumulators / lane)
__device__ inline f32x4 mfma_16x16x16_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x4 c_vec) {
    // cbsz, abid, blgp must be immediates
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
    // Syntax & wavefront-wide semantics: AMD Lab Notes on Matrix Cores.  :contentReference[oaicite:4]{index=4}
}

// ===== Lane → (row/col-group) mapping helpers =====
__device__ inline int lane_row(int lane)   { return lane & 15; }   // 0..15
__device__ inline int lane_group(int lane) { return lane >> 4; }   // 0..3

// ===== Build the per-lane bf16x4 operands from LDS tiles =====
// A is in LDS row-major [BLOCK_M x BLOCK_K] with ldA = BLOCK_K
// Take row = (wave_m*WM) + lane_row, columns (grp*4 + i), i=0..3
__device__ inline bf16x4 make_a_vec(const uint16_t* __restrict__ sA,
                                    int ldA, int aRowBase, int lane)
{
    const int r   = aRowBase + lane_row(lane);
    const int grp = lane_group(lane); // 0..3
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = sA[r * ldA + (grp * 4 + i)];
    }
    return v;
}

// B is in LDS column-major [BLOCK_K x BLOCK_N] with ldB = BLOCK_K
// Take column = (wave_n*WN) + lane_row (0..15), rows (grp*4 + i)
__device__ inline bf16x4 make_b_vec(const uint16_t* __restrict__ sB,
                                    int ldB, int bColBase, int lane)
{
    const int col = bColBase + lane_row(lane); // 0..15 within the 16x16 tile
    const int grp = lane_group(lane);          // 0..3
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // column-major addressing: index = col*ldB + row
        v[i] = sB[col * ldB + (grp * 4 + i)];
    }
    return v;
}

// ===== Store 16x16 accum tile from per-lane f32x4 =====
template<bool InteriorStore>
__device__ inline void store_c_tile_mfma(
float* __restrict__ C, const f32x4& acc,
    int M, int N, int m0, int n0, int wave_m, int wave_n, int lane)
{
    const int rowBase = m0 + wave_m * WM + lane_group(lane) * 4; // 4 rows per group
    const int col     = n0 + wave_n * WN + lane_row(lane);       // one column per lane_row

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if constexpr (InteriorStore) {
            C[(size_t)row * N + col] = acc[i];
        } else {
            if (row < M && col < N) C[(size_t)row * N + col] = acc[i];
        }
    }
}
// ===== BF16 kernel (A: FP32, B: BF16) =====
__global__ void gemm_mfma_bf16_kernel_opt(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] FP32
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major (weights)
    int M, int K, int N)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;               // 0..63
    const int wave = threadIdx.y;               // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // LDS layout (ping–pong):
    // sA0, sA1: [BLOCK_M x BLOCK_K] row-major (uint16_t BF16 bits)
    // sB0, sB1: [BLOCK_K x BLOCK_N] col-major (uint16_t BF16 bits)
    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // Preload k-slice 0
    {
        // A -> sA0 (row-major), convert FP32->BF16
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + c] = f32_to_bf16_bits(a);
        }
        // B -> sB0 (col-major), Wbf16 is [N,K] row-major
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K;  // tile column within N-block
            const int r = idx % BLOCK_K;  // k inside this slice
            const int gn = n0 + c;
            const int gk = r;
            __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
            sB0[c * BLOCK_K + r] = hipbf16_to_bits(wb);
        }
    }
    __syncthreads();

    auto* currA = sA0;
    auto* currB = sB0;
    auto* nextA = sA1;
    auto* nextB = sB1;

    // MFMA microkernel: one MFMA per K-chunk (16)
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Preload next stage
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;

            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + c] = f32_to_bf16_bits(a);
            }
            for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K;
                const int r = idx % BLOCK_K;
                const int gn = n0 + c;
                const int gk = kBase + r;
                __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
                nextB[c * BLOCK_K + r] = hipbf16_to_bits(wb);
            }
        }

        // Build lane operands and issue MFMA
        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;  // selects our 16 rows within the block
        const int bColBase = wave_n * WN;  // selects our 16 cols within the block

        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        // Swap buffers
        auto* tmpA = currA; currA = nextA; nextA = tmpA;
        auto* tmpB = currB; currB = nextB; nextB = tmpB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_mfma<true>(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
    }
}

// ===== MXFP4 (packed 4-bit) path: dequantize -> BF16 -> MFMA =====
__global__ void gemm_mfma_uint8_kernel_opt(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K] FP32
    const uint8_t* __restrict__ Wp,               // [N, K] packed (MXFP4)
    const float* __restrict__ weight_scales,      // [N / 32] (or your layout)
    int M, int K, int N, size_t total_weight_elements)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    auto load_B_tile = [&](uint16_t* dst, int kBase) {
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K;       // col within block
            const int r = idx % BLOCK_K;       // k within slice
            const int gn = n0 + c;
            const int gk = kBase + r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
        }
    };

    // Preload stage 0
    {
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + c] = f32_to_bf16_bits(a);
        }
        load_B_tile(sB0, /*kBase=*/0);
    }
    __syncthreads();

    auto* currA = sA0;
    auto* currB = sB0;
    auto* nextA = sA1;
    auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;

            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + c] = f32_to_bf16_bits(a);
            }
            load_B_tile(nextB, kBase);
        }

        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;

        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        auto* tmpA = currA; currA = nextA; nextA = tmpA;
        auto* tmpB = currB; currB = nextB; nextB = tmpB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_mfma<true>(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
    }
}

#pragma once

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
void matmul(
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
void matmul(
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
