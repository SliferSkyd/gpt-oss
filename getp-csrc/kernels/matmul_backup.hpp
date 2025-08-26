#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"
// Keep WM/WN/WK fixed (hardware-fragment sizes)
static constexpr int WM = 16;
static constexpr int WN = 16;
static constexpr int WK = 16;

__device__ inline rocwmma::bfloat16_t f32_to_bf16(float x) {
    __hip_bfloat16 t = __float2bfloat16(x);
    rocwmma::bfloat16_t y;
    *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<const uint16_t*>(&t);
    return y;
}

template<bool InteriorStore, int WAVES_M, int WAVES_N>
__device__ inline void store_c_tile(
    float* __restrict__ C,
    const rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>& cFrag,
    int M, int N, int m0, int n0,
    int wave_m, int wave_n,
    int lane,
    float* __restrict__ sC) // may be null when InteriorStore==true
{
    constexpr int LANE_PER_WAVE = 64;
    if constexpr (InteriorStore) {
        const int gm = m0 + wave_m * WM;
        const int gn = n0 + wave_n * WN;
        rocwmma::store_matrix_sync(C + (size_t)gm * N + gn, cFrag, N, rocwmma::mem_row_major);
    } else {
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

template<int WAVES_M, int WAVES_N>
__global__ void gemm_wmma_bf16_kernel_opt_t(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] (FP32)
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major
    int M, int K, int N)
{
    using namespace rocwmma;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK;
    constexpr int LANE_PER_WAVE = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;               // 0..63
    const int wave = threadIdx.y;               // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // LDS: sA0,sA1 [BLOCK_M x BLOCK_K] row-major
    //      sB0,sB1 [BLOCK_K x BLOCK_N] col-major
    //      sC      [WAVES_PER_BLOCK x (WM*WN)] floats (ragged edges only)
    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<bfloat16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC  = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N));

    fragment<accumulator, WM, WN, WK, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    const int threadsPerBlock = blockDim.x * blockDim.y; // 64 * WAVES_PER_BLOCK
    const int linearT = wave * blockDim.x + lane;

    // --- Preload stage 0 ---
    {
        // A -> sA0
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            const float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + c] = f32_to_bf16(a);
        }
        // B (already bf16) -> sB0 (col-major in LDS)
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K; // col in block
            const int r = idx % BLOCK_K; // k
            const int gn = n0 + c;
            const int gk = r;
            __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
            bfloat16_t y;
            *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<uint16_t*>(&wb);
            sB0[c * BLOCK_K + r] = y;
        }
    }
    __syncthreads();

    fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> aFrag;
    fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> bFrag;

    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Preload next stage
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            // A -> nextA
            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                const float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + c] = f32_to_bf16(a);
            }
            // B -> nextB
            for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K;
                const int r = idx % BLOCK_K;
                const int gn = n0 + c;
                const int gk = kBase + r;
                __hip_bfloat16 wb = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
                bfloat16_t y;
                *reinterpret_cast<uint16_t*>(&y) = *reinterpret_cast<uint16_t*>(&wb);
                nextB[c * BLOCK_K + r] = y;
            }
        }

        const int ldA = BLOCK_K;
        const int ldB = BLOCK_K;
        const int aRow = wave_m * WM;
        const int bCol = wave_n * WN;

        load_matrix_sync(aFrag, currA + aRow * ldA, ldA);
        load_matrix_sync(bFrag, currB + bCol * ldB, ldB);
        mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile<true, WAVES_M, WAVES_N>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, nullptr);
    } else {
        store_c_tile<false, WAVES_M, WAVES_N>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, sC);
    }
}
template<int WAVES_M, int WAVES_N>
__global__ void gemm_wmma_uint8_kernel_opt_t(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K] (FP32)
    const uint8_t* __restrict__ Wp,               // [N, K] packed (MXFP4)
    const float* __restrict__ weight_scales,      // [N / 32] or your layout
    int M, int K, int N, size_t total_weight_elements)
{
    using namespace rocwmma;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK;
    constexpr int LANE_PER_WAVE = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<bfloat16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC  = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N));

    fragment<accumulator, WM, WN, WK, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    auto load_B_tile = [&](bfloat16_t* dst, int kBase) {
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K;
            const int r = idx % BLOCK_K;
            const int gn = n0 + c;
            const int gk = kBase + r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[c * BLOCK_K + r] = f32_to_bf16(wb);
        }
    };

    // Preload 0
    {
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            const float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + c] = f32_to_bf16(a);
        }
        load_B_tile(sB0, 0);
    }
    __syncthreads();

    fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> aFrag;
    fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> bFrag;

    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K;
                const int c = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + c;
                const float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
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
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile<true, WAVES_M, WAVES_N>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, nullptr);
    } else {
        store_c_tile<false, WAVES_M, WAVES_N>(C, cFrag, M, N, m0, n0, wave_m, wave_n, lane, sC);
    }
}

template<int WAVES_M, int WAVES_N>
static inline void launch_configs(int M, int N, dim3& grid, dim3& block, size_t& shmem_bytes) {
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;
    constexpr int LANE_PER_WAVE = 64;

    grid  = dim3( (N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M );
    block = dim3( LANE_PER_WAVE, WAVES_PER_BLOCK );

    shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(rocwmma::bfloat16_t) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);
}
void matmul(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const __hip_bfloat16* __restrict__ weight,  // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;

    if (M < 64) {
        // Small-M: fewer waves along M to avoid idle threads
        dim3 grid, block; size_t shmem;
        launch_configs<2,4>(M, N, grid, block, shmem);  // BLOCK_M=32, BLOCK_N=64
        hipLaunchKernelGGL(
            (gemm_wmma_bf16_kernel_opt_t<2,4>),
            grid, block, shmem, stream,
            output, input, weight, M, K, N);
    } else {
        // Default: 64x64 block, 16 waves (1024 threads)
        dim3 grid, block; size_t shmem;
        launch_configs<4,4>(M, N, grid, block, shmem);  // BLOCK_M=64, BLOCK_N=64
        hipLaunchKernelGGL(
            (gemm_wmma_bf16_kernel_opt_t<4,4>),
            grid, block, shmem, stream,
            output, input, weight, M, K, N);
    }
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

    if (M < 64) {
        dim3 grid, block; size_t shmem;
        launch_configs<2,4>(M, N, grid, block, shmem);
        hipLaunchKernelGGL(
            (gemm_wmma_uint8_kernel_opt_t<2,4>),
            grid, block, shmem, stream,
            output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
    } else {
        dim3 grid, block; size_t shmem;
        launch_configs<4,4>(M, N, grid, block, shmem);
        hipLaunchKernelGGL(
            (gemm_wmma_uint8_kernel_opt_t<4,4>),
            grid, block, shmem, stream,
            output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
    }
}
