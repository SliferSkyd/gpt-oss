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

// ===== Host wrappers (unchanged API) =====
void matmul(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const __hip_bfloat16* __restrict__ weight,  // [O, I] bf16 row-major
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
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);

    hipLaunchKernelGGL(
        gemm_mfma_uint8_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
}

// ====== FUSED (2-HEAD) MXFP4 GEMM (gate+up in one launch; reuses A) ======
__global__ void gemm_mfma_uint8_kernel_opt_fused2(
    float* __restrict__ C,                         // [M, 2*Nhalf] row-major
    const float* __restrict__ A,                   // [M, K]       row-major (fp32)
    const uint8_t* __restrict__ Wp0,               // [Nhalf, K]   MXFP4 packed (first half)
    const float*   __restrict__ scales0,           // scales for first half
    const uint8_t* __restrict__ Wp1,               // [Nhalf, K]   MXFP4 packed (second half)
    const float*   __restrict__ scales1,           // scales for second half
    int M, int K, int Nhalf,
    size_t total_w_elems_half0,                    // = Nhalf*K (elements, not bytes)
    size_t total_w_elems_half1)                    // = Nhalf*K (elements, not bytes)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;           // block’s column start within each half

    const int lane   = threadIdx.x;                // 0..63
    const int wave   = threadIdx.y;                // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // Dynamic shared memory (raw bytes -> cast to uint16_t for bf16 bits)
    extern __shared__ unsigned char smem[];
    uint16_t* sA0  = reinterpret_cast<uint16_t*>(smem);
    uint16_t* sA1  = sA0  + (BLOCK_M * BLOCK_K);

    uint16_t* sB0h0 = sA1  + (BLOCK_M * BLOCK_K);           // stage 0, half 0
    uint16_t* sB0h1 = sB0h0 + (BLOCK_K * BLOCK_N);          // stage 0, half 1
    uint16_t* sB1h0 = sB0h1 + (BLOCK_K * BLOCK_N);          // stage 1, half 0
    uint16_t* sB1h1 = sB1h0 + (BLOCK_K * BLOCK_N);          // stage 1, half 1

    f32x4 acc0 = {0.f, 0.f, 0.f, 0.f};  // accum: first half (e.g., gate)
    f32x4 acc1 = {0.f, 0.f, 0.f, 0.f};  // accum: second half (e.g., up)

    const int threadsPerBlock = blockDim.x * blockDim.y;     // 64 * WAVES_PER_BLOCK
    const int linearT         = wave * blockDim.x + lane;

    auto load_A_tile = [&](uint16_t* dst, int kBase) {
        // A in LDS as row-major [BLOCK_M x BLOCK_K] of bf16 bits
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r  = idx / BLOCK_K;    // 0..BLOCK_M-1
            const int c  = idx % BLOCK_K;    // 0..BLOCK_K-1
            const int gm = m0 + r;
            const int gk = kBase + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            dst[r * BLOCK_K + c] = f32_to_bf16_bits(a);
        }
    };

    auto load_B_tile_half = [&](uint16_t* dst, int kBase, int half) {
        const uint8_t* Wp     = (half == 0) ? Wp0     : Wp1;
        const float*   scales = (half == 0) ? scales0 : scales1;
        const size_t   totalW = (half == 0) ? total_w_elems_half0 : total_w_elems_half1;

        // B in LDS as column-major [BLOCK_K x BLOCK_N] (ldB = BLOCK_K)
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c  = idx / BLOCK_K;              // local column within BLOCK_N
            const int r  = idx % BLOCK_K;              // k within this slice
            const int gn = n0 + c;                     // column index within the half
            const int gk = kBase + r;                  // K index
            float wb = (gk < K && gn < Nhalf)
                ? dequantize_mxfp4(Wp, scales, (size_t)gn * K + gk, totalW)
                : 0.0f;
            dst[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
        }
    };

    // ---- Preload stage 0 ----
    load_A_tile(sA0, 0);
    load_B_tile_half(sB0h0, 0, /*half=*/0);
    load_B_tile_half(sB0h1, 0, /*half=*/1);
    __syncthreads();

    uint16_t* currA  = sA0;
    uint16_t* nextA  = sA1;
    uint16_t* currB0 = sB0h0;   // half 0
    uint16_t* nextB0 = sB1h0;
    uint16_t* currB1 = sB0h1;   // half 1
    uint16_t* nextB1 = sB1h1;

    // ---- K loop (16-wide MFMA slices) ----
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Preload next stage
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            load_A_tile(nextA, kBase);
            load_B_tile_half(nextB0, kBase, /*half=*/0);
            load_B_tile_half(nextB1, kBase, /*half=*/1);
        }

        // Build lane operands once for A, twice for B (two halves)
        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;

        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 b0   = make_b_vec(currB0, ldB, bColBase, lane);
        bf16x4 b1   = make_b_vec(currB1, ldB, bColBase, lane);

        acc0 = mfma_16x16x16_bf16(avec, b0, acc0);
        acc1 = mfma_16x16x16_bf16(avec, b1, acc1);

        __syncthreads(); // make sure preloads completed & no LDS hazards

        // swap ping–pong buffers
        uint16_t* tA = currA;  currA  = nextA;  nextA  = tA;
        uint16_t* t0 = currB0; currB0 = nextB0; nextB0 = t0;
        uint16_t* t1 = currB1; currB1 = nextB1; nextB1 = t1;
    }

    // ---- Stores ----
    const int Ntot = 2 * Nhalf;                  // full row stride
    const bool full_tile_h0 = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= Nhalf;

    if (full_tile_h0) {
        store_c_tile_mfma<true >(C, acc0, M, Ntot, m0, /*n0=*/n0,       wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C, acc0, M, Ntot, m0, /*n0=*/n0,       wave_m, wave_n, lane);
    }

    const int n0_h1 = n0 + Nhalf;
    const bool full_tile_h1 = (m0 + BLOCK_M) <= M && (n0_h1 + BLOCK_N) <= Ntot;

    if (full_tile_h1) {
        store_c_tile_mfma<true >(C, acc1, M, Ntot, m0, /*n0=*/n0_h1,    wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C, acc1, M, Ntot, m0, /*n0=*/n0_h1,    wave_m, wave_n, lane);
    }
}

// Host wrapper: computes two contiguous halves (each Nhalf) into [M, 2*Nhalf]
inline void matmul_mxfp4_fused2(
    float* __restrict__ output,                 // [M, 2*Nhalf]
    const float* __restrict__ input,            // [M, K]
    const uint8_t* __restrict__ weight0_packed, // base of first half
    const float*   __restrict__ scales0,
    const uint8_t* __restrict__ weight1_packed, // base of second half
    const float*   __restrict__ scales1,
    int batch_size, int input_dim, int Nhalf,
    size_t total_weight_elements_half,          // = Nhalf * input_dim (elements)
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim;

    dim3 grid((Nhalf + BLOCK_N - 1) / BLOCK_N,
              (M     + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    // Shared memory: 2*A tiles + 4*B tiles, all as bf16 bits (uint16_t)
    const size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 4 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);

    hipLaunchKernelGGL(
        gemm_mfma_uint8_kernel_opt_fused2,
        grid, block, shmem_bytes, stream,
        /*C*/  output,
        /*A*/  input,
        /*B0*/ weight0_packed, /*S0*/ scales0,
        /*B1*/ weight1_packed, /*S1*/ scales1,
        /*dims*/ M, K, Nhalf,
        /*totals*/ total_weight_elements_half, total_weight_elements_half);
}

