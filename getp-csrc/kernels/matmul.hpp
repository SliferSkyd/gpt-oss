#pragma once

// mfma_gemm_gfx90a.hpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include "../utils.hpp"

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
#define WAVES_M 2
#endif
#ifndef WAVES_N
#define WAVES_N 8
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M = WM * WAVES_M;        // e.g. 64 if WAVES_M=4
constexpr int BLOCK_N = WN * WAVES_N;        // e.g. 64 if WAVES_N=4
constexpr int BLOCK_K = WK;                  // 16
constexpr int LANE_PER_WAVE = 64;
constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

// ===== Optional fast-path toggles =====
// If B is already BF16 in HBM, define this to avoid FP32->BF16 converts:
// #define B_IN_HBM_IS_BF16 1
// If B pointer is 16B-aligned and N % 8 == 0, enable vectorized BF16 loads:
// #define VECTORIZED_B_LOAD 1

// ===== Small vector helpers (types that match MFMA signatures) =====
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4))); // 4×i16 carrying BF16 bit patterns

// ===== Conversions =====
__device__ inline uint16_t f32_to_bf16_bits(float x) {
    __hip_bfloat16 t = __float2bfloat16(x);
    return *reinterpret_cast<uint16_t*>(&t);
}

// ===== MFMA wrapper (gfx90a supports BF16->F32 accumulate) =====
// d = a*b + c, one 16x16x16 bf16 MMA per wave (returns 4 accumulators / lane)
__device__ inline f32x4 mfma_16x16x16_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x4 c_vec) {
    // cbsz, abid, blgp must be immediates
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
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

// B is stored in LDS column-major [BLOCK_K x BLOCK_N], BUT we swizzle rows at store-time
// to avoid LDS bank conflicts; we undo it here during the read.
__device__ inline bf16x4 make_b_vec_swizzled(const uint16_t* __restrict__ sB,
                                             int ldB, int bColBase, int lane)
{
    const int col = bColBase + lane_row(lane); // 0..15 within the 16x16 sub-tile handled by this wave
    const int grp = lane_group(lane);          // 0..3
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r     = grp * 4 + i;        // 0..15
        const int r_sw  = r ^ (col & 0xF);    // undo the column-dependent XOR used on store
        v[i] = sB[col * ldB + r_sw];
    }
    return v;
}

__device__ inline bf16x4 make_b_vec(const uint16_t* __restrict__ sB, int ldB, int bColBase, int lane) { 
    const int col = bColBase + lane_row(lane); // 0..15 within the 16x16 tile
    const int grp = lane_group(lane); // 0..3
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
    const float* __restrict__ Wbf16,  // [N, K] row-major (weights)
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
            float wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : 0.0f;
            sB0[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
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
                float wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : 0.0f;
                nextB[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
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


// ===== Host wrappers (unchanged API) =====
void matmul_mc(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const float* __restrict__ weight,  // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    // Double-buffered A/B only (no sC spill needed; we mask-ed store)
    size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);

    hipLaunchKernelGGL(
        gemm_mfma_bf16_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
}

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
// Keep your BM/BN/BK/TB_X/TB_Y/TM/TN as-is

#ifndef LDS_PAD_A
#define LDS_PAD_A 1   // pad A's leading dimension by +1 column
#endif
#ifndef PAD_SB
#define PAD_SB 1      // add one padded row to B tile
#endif

// Derived LDS strides with padding
constexpr int LD_SA = BK + LDS_PAD_A;  // sA: [BM x (BK+pad)] row-major
constexpr int LD_SB = BN;              // sB: [(BK+pad_row) x BN] row-major


__device__ __forceinline__ float bf16_bits_to_f32_fix(uint16_t bits) {
    const __hip_bfloat16 bx = *reinterpret_cast<const __hip_bfloat16*>(&bits);
    return __bfloat162float(bx);
}
__launch_bounds__(TB_X * TB_Y, 2)
__global__ void gemm_fma_bf16_kernel_opt(
    float* __restrict__ C,                          // [M, N] row-major
    const float* __restrict__ A,                    // [M, K] row-major (fp32)
    const __hip_bfloat16* __restrict__ Wbf16,       // [N, K] row-major (bf16)
    int M, int K, int N)
{
    // --- Tile origin in C ---
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    // --- Thread coords ---
    const int tx = threadIdx.x;               // 0..TB_X-1
    const int ty = threadIdx.y;               // 0..TB_Y-1
    const int linearT = ty * TB_X + tx;
    const int threadsPerBlock = TB_X * TB_Y;

    // --- Microtile per thread ---
    const int rowBase = m0 + ty * TM;
    const int colBase = n0 + tx * TN;

    // --- Padded shared strides to avoid bank conflicts ---
    const int LD_SA = BK + 1;                 // sA: [BM x (BK+1)] row-major (float)
    const int LD_SB = BN + 1;                 // sB: [BK x (BN+1)] row-major (float)

    // --- Double-buffered shared memory ---
    extern __shared__ float smem[];
    float* sA0 = smem;                                   // size BM*LD_SA
    float* sA1 = sA0 + (BM * LD_SA);
    float* sB0 = sA1 + (BM * LD_SA);                     // size BK*LD_SB
    float* sB1 = sB0 + (BK * LD_SB);

    // --- Accumulators ---
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // --- Cooperative loaders (vectorized & padded) ---

    // A: load as float4 along K (contiguous in memory), store to sA row-major with LD_SA
    auto loadA_vec = [&](float* __restrict__ dst, int kBase) {
        const int vecs_per_row = BK / 4; // 4 floats (16B) per vector
        for (int idx = linearT; idx < BM * vecs_per_row; idx += threadsPerBlock) {
            const int r  = idx / vecs_per_row;          // 0..BM-1
            const int c4 = idx - r * vecs_per_row;      // 0..(BK/4)-1
            const int gm = m0 + r;
            const int gk = kBase + c4 * 4;

            float4 v4 = {0.f, 0.f, 0.f, 0.f};
            if (gm < M) {
                const float* gptr = A + (size_t)gm * K + gk;
                v4 = *reinterpret_cast<const float4*>(gptr);
            }
            float* srow = dst + r * LD_SA + c4 * 4;
            srow[0] = v4.x; srow[1] = v4.y; srow[2] = v4.z; srow[3] = v4.w;
        }
        // zero pad column
        if (linearT < BM) dst[linearT * LD_SA + BK] = 0.0f;
    };

    // B: load 8×bf16 (16B) along K for each column n, convert to f32, store row-major [BK x LD_SB]
    auto loadB_vec = [&](float* __restrict__ dst, int kBase) {
        const int vecs_per_col = BK / 8;          // each vector covers 8 contiguous k's
        const int total_vecs   = vecs_per_col * BN;

        for (int idx = linearT; idx < total_vecs; idx += threadsPerBlock) {
            const int c    = idx / vecs_per_col;  // 0..BN-1 (column in the tile)
            const int r8v  = idx - c * vecs_per_col; // 0..(BK/8)-1
            const int gn   = n0 + c;
            const int gk   = kBase + r8v * 8;     // global K index of this 16B vector
            const int rloc = r8v * 8;             // *** local row offset inside BK slice ***

            uint16_t b16[8];
            if (gn < N) {
                const uint4* gptr = reinterpret_cast<const uint4*>(
                    reinterpret_cast<const char*>(Wbf16) + ((size_t)gn * K + gk) * sizeof(__hip_bfloat16));
                const uint4 v = *gptr;           // 16B load (8×bf16)
                memcpy(b16, &v, 16);
            } else {
#pragma unroll
                for (int t = 0; t < 8; ++t) b16[t] = uint16_t(0);
            }

            // *** FIXED: use LOCAL row index (rloc), not absolute gk ***
            float* base = dst + (size_t)rloc * LD_SB + c;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                base[t * LD_SB] = bf16_bits_to_f32_fix(b16[t]);
            }
        }
        // zero pad column
        if (linearT < BK) dst[linearT * LD_SB + BN] = 0.0f;
    };

    // --- Preload first BK-slice ---
    loadA_vec(sA0, 0);
    loadB_vec(sB0, 0);
    __syncthreads();

    float* currA = sA0;  float* nextA = sA1;
    float* currB = sB0;  float* nextB = sB1;

#pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            loadA_vec(nextA, k0 + BK);
            loadB_vec(nextB, k0 + BK);
        }

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // A fragment: TM rows from column kk
            float aFrag[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int rr = ty * TM + i;
                aFrag[i] = currA[rr * LD_SA + kk];
            }
            // B fragment: TN cols from row kk
            float bFrag[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int cc = tx * TN + j;
                bFrag[j] = currB[kk * LD_SB + cc];
            }
            // FMA outer product
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(aFrag[i], bFrag[j], acc[i][j]);
        }

        __syncthreads();
        float* tA = currA; currA = nextA; nextA = tA;
        float* tB = currB; currB = nextB; nextB = tB;
    }

    // --- Store C ---
    float* __restrict__ Cout = C + (size_t)rowBase * N + colBase;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = rowBase + i;
        if (gm >= M) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = colBase + j;
            if (gn < N) Cout[j] = acc[i][j];
        }
        Cout += N;
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
    std::stringstream timer_name;
    timer_name << "matmul_kernel_" << batch_size << "x" << input_dim << "x" << output_dim;
    TIMER_BLOCK(timer_name.str());
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(TB_X, TB_Y);

    size_t shmem_bytes = sizeof(float) * (2*(BM*LD_SA) + 2*((BK + PAD_SB)*LD_SB));

    hipLaunchKernelGGL(
        gemm_fma_bf16_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
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
    std::stringstream timer_name;
    timer_name << "matmul_fma_" << batch_size << "x" << input_dim << "x" << output_dim;
    TIMER_BLOCK(timer_name.str());
    
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

__device__ inline uint16_t hipbf16_to_bits(__hip_bfloat16 x) { return *reinterpret_cast<uint16_t*>(&x); }
