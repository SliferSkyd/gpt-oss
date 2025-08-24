#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"

// ===== Tunables =====
// Default to smaller, higher-occupancy 64x64 tile with 8 waves (512 threads).
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
#define WAVES_M 4   // 2x4 waves -> 32x64 sub-tiles per block
#endif
#ifndef WAVES_N
#define WAVES_N 4
#endif

constexpr int BLOCK_M = WM * WAVES_M;        // 32 or 64 depending on WAVES_M
constexpr int BLOCK_N = WN * WAVES_N;        // 64 or 32 depending on WAVES_N
constexpr int BLOCK_K = WK;
constexpr int LANE_PER_WAVE = 64;
constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

// ===== Helpers =====
__device__ inline rocwmma::bfloat16_t f32_to_bf16(float x) {
    __hip_bfloat16 t = __float2bfloat16(x);           // HIP intrinsic, RN
    rocwmma::bfloat16_t y;
    *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<const uint16_t*>(&t);
    return y;
}

template<bool InteriorStore>
__device__ inline void store_c_tile(
    float* __restrict__ C,
    const rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>& cFrag,
    int M, int N, int m0, int n0,
    int wave_m, int wave_n,
    int lane,
    float* __restrict__ sC) // may be null when InteriorStore==true
{
    if constexpr (InteriorStore) {
        // Store directly to global row-major with stride N
        const int gm = m0 + wave_m * WM;
        const int gn = n0 + wave_n * WN;
        rocwmma::store_matrix_sync(C + (size_t)gm * N + gn, cFrag, N, rocwmma::mem_row_major);
    } else {
        // Ragged edges: spill to LDS then masked global write
        float* sCbase = sC + (wave_m * WAVES_N + wave_n) * (WM * WN);
        rocwmma::store_matrix_sync(sCbase, cFrag, WN, rocwmma::mem_row_major);
        __syncthreads();

        const int c_m0 = m0 + wave_m * WM;
        const int c_n0 = n0 + wave_n * WN;
        for (int t = lane; t < WM * WN; t += LANE_PER_WAVE) {
            const int r = t / WN;
            const int c = t % WN;
            const int gm = c_m0 + r;
            const int gn = c_n0 + c;
            if (gm < M && gn < N) {
                C[(size_t)gm * N + gn] = sCbase[r * WN + c];
            }
        }
    }
}

// ===== BF16 kernel =====
__global__ void gemm_wmma_bf16_kernel_opt(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] (FP32)
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major
    int M, int K, int N)
{
    using namespace rocwmma;

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;               // 0..63
    const int wave = threadIdx.y;               // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // LDS layout with ping-pong buffers:
    // sA0, sA1: [BLOCK_M, BLOCK_K] row-major
    // sB0, sB1: [BLOCK_K, BLOCK_N] col-major
    // sC:       [WAVES_PER_BLOCK, WM*WN] floats (only when ragged)
    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<rocwmma::bfloat16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC   = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N)); // size: WAVES_PER_BLOCK * WM * WN

    fragment<accumulator, WM, WN, WK, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // Preload stage 0
    {
        // A -> sA0
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + c] = f32_to_bf16(a);
        }
        // B (already bf16) -> sB0 (col-major in LDS)
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K;
            const int r = idx % BLOCK_K;
            const int gn = n0 + c;
            const int gk = r;
            __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
            rocwmma::bfloat16_t y;
            *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<uint16_t*>(&wb);
            sB0[c * BLOCK_K + r] = y;
        }
    }
    __syncthreads();

    // Main loop with ping-pong buffers
    rocwmma::fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> aFrag;
    rocwmma::fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> bFrag;

    auto* currA = sA0;
    auto* currB = sB0;
    auto* nextA = sA1;
    auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Preload next stage if any
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            // A -> nextA
            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + c] = f32_to_bf16(a);
            }
            // B -> nextB (col-major)
            for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K;
                const int r = idx % BLOCK_K;
                const int gn = n0 + c;
                const int gk = kBase + r;
                __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
                rocwmma::bfloat16_t y;
                *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<uint16_t*>(&wb);
                nextB[c * BLOCK_K + r] = y;
            }
        }

        // MMA on current stage
        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRow = wave_m * WM;
        const int bCol = wave_n * WN;

        load_matrix_sync(aFrag, currA + aRow * ldA, ldA);
        load_matrix_sync(bFrag, currB + bCol * ldB, ldB);
        mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
        // Swap buffers
        auto* tmpA = currA; currA = nextA; nextA = tmpA;
        auto* tmpB = currB; currB = nextB; nextB = tmpB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile<true>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, nullptr);
    } else {
        store_c_tile<false>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, sC);
    }
}

// ===== MXFP4 (uint8-packed) kernel =====
__global__ void gemm_wmma_uint8_kernel_opt(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K] (FP32)
    const uint8_t* __restrict__ Wp,               // [N, K] packed (MXFP4)
    const float* __restrict__ weight_scales,      // [N / 32] or your layout
    int M, int K, int N, size_t total_weight_elements)
{
    using namespace rocwmma;

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<rocwmma::bfloat16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC  = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N));

    fragment<accumulator, WM, WN, WK, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // Helper lambda to load a B tile into LDS (col-major), calling your dequantize
    auto load_B_tile = [&](rocwmma::bfloat16_t* dst, int kBase) {
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K;       // column within the block
            const int r = idx % BLOCK_K;       // k within the slice
            const int gn = n0 + c;
            const int gk = kBase + r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[c * BLOCK_K + r] = f32_to_bf16(wb);
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
            sA0[r * BLOCK_K + c] = f32_to_bf16(a);
        }
        load_B_tile(sB0, /*kBase=*/0);
    }
    __syncthreads();

    rocwmma::fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> aFrag;
    rocwmma::fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> bFrag;

    auto* currA = sA0;
    auto* currB = sB0;
    auto* nextA = sA1;
    auto* nextB = sB1;

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
                nextA[r * BLOCK_K + c] = f32_to_bf16(a);
            }
            load_B_tile(nextB, kBase);
        }

        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRow = wave_m * WM;
        const int bCol = wave_n * WN;

        load_matrix_sync(aFrag, currA + aRow * ldA, ldA);
        load_matrix_sync(bFrag, currB + bCol * ldB, ldB);
        mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
        auto* tmpA = currA; currA = nextA; nextA = tmpA;
        auto* tmpB = currB; currB = nextB; nextB = tmpB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile<true>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, nullptr);
    } else {
        store_c_tile<false>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, sC);
    }
}

// ===== Host wrappers =====
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

    // Double-buffered A/B + optional sC for ragged edges
    size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(rocwmma::bfloat16_t) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_wmma_bf16_kernel_opt,
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
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(rocwmma::bfloat16_t) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_wmma_uint8_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
}
