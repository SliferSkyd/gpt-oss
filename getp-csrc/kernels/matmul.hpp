#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"

// ===== Tunables (unchanged) =====
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

constexpr int BLOCK_M = WM * WAVES_M;      // e.g., 64
constexpr int BLOCK_N = WN * WAVES_N;      // e.g., 64
constexpr int BLOCK_K = WK;                // must be 16 for bf16 MFMA below
constexpr int LANE_PER_WAVE = 64;
constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

// === Short vector helpers for MFMA operands / accumulators ===
using f32x4  = float __attribute__((ext_vector_type(4)));
using u16x4  = unsigned short __attribute__((ext_vector_type(4)));

__device__ inline __hip_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);  // round-to-nearest
}

// v_mfma_f32_16x16x16bf16: (<4xi16> A, <4xi16> B, <4xf32> C) -> <4xf32>
__device__ inline f32x4 mfma_16x16x16_bf16(u16x4 a4, u16x4 b4, f32x4 c4) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16)
    return __builtin_amdgcn_mfma_f32_16x16x16bf16(a4, b4, c4, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
#else
#   error "MFMA BF16 intrinsic not available for this target/compiler."
#endif
}

// ======== BF16 path (A: fp32, W: bf16 row-major [N,K]) ========
__global__ void gemm_mfma_bf16_kernel(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] (fp32)
    const __hip_bfloat16* __restrict__ Wbf16,  // [N, K] row-major
    int M, int K, int N)
{
    // Block origins
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    // Wave/lane IDs
    const int lane    = threadIdx.x;             // 0..63
    const int wave    = threadIdx.y;             // 0..(WAVES_PER_BLOCK-1)
    const int wave_m  = wave / WAVES_N;          // 0..WAVES_M-1
    const int wave_n  = wave % WAVES_N;          // 0..WAVES_N-1

    // Per-lane row / column-group inside the 16x16 tile
    const int lane_row      = lane & 0xF;        // 0..15
    const int lane_col_grp  = lane >> 4;         // 0..3 (each lane holds 4 cols)

    // LDS layout (ping-pong): A row-major [BLOCK_M, BLOCK_K], B col-major [BLOCK_K, BLOCK_N]
    extern __shared__ unsigned char smem[];
    auto* sA0 = reinterpret_cast<__hip_bfloat16*>(smem);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC  = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N)); // scratch for ragged stores

    // Accumulator for this lane (4 outputs)
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // Thread-collaborative prefetch
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // ===== Preload k-slice 0 into LDS =====
    {
        // A -> sA0 (row-major)
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r  = idx / BLOCK_K;
            const int kc = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = kc;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + kc] = f32_to_bf16(a);
        }
        // B -> sB0 (col-major in LDS)
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c  = idx / BLOCK_K;    // N-of-block
            const int rk = idx % BLOCK_K;    // K-of-slice
            const int gn = n0 + c;
            const int gk = rk;
            __hip_bfloat16 w = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
            sB0[c * BLOCK_K + rk] = w;       // col-major: [c][rk]
        }
    }
    __syncthreads();

    // Ping-pong pointers and constants
    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;
    const int ldA = BLOCK_K;                   // row-major K stride
    const int ldB = BLOCK_K;                   // col-major K stride

    // Wave-tile anchors inside block
    const int aRowBase = wave_m * WM;          // 0,16,32,48 within the block
    const int bColBase = wave_n * WN;          // 0,16,32,48 within the block

    // ===== Main K loop (step 16 for BF16 MFMA) =====
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {

        // Load per-lane A[ row , k0 .. k0+15 ] as <4xi16> (packed 4 BF16 along K)
        {
            const int aRow = aRowBase + lane_row;                   // 0..63 within block
            const __hip_bfloat16* aPtr = currA + aRow * ldA;        // points at k0 slice
            u16x4 a4 = *reinterpret_cast<const u16x4*>(aPtr);       // {k0+0, k0+1, k0+2, k0+3} bf16

            // Load per-lane B[ k0..k0+15 , col ] as <4xi16>, col = bColBase + lane_col_grp*4 .. +3
            const int col = bColBase + lane_col_grp * 4;            // starting column
            const __hip_bfloat16* bPtr = currB + col * ldB;         // column-major: K stride
            u16x4 b4 = *reinterpret_cast<const u16x4*>(bPtr);       // 4 bf16 along K

            // One MFMA covers the whole 16 K-depth for BF16 variant
            acc = mfma_16x16x16_bf16(a4, b4, acc);
        }

        __syncthreads();

        // Preload next k-slice while others compute
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;

            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r  = idx / BLOCK_K;
                const int kc = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + kc;
                float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + kc] = f32_to_bf16(a);
            }
            for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
                const int c  = idx / BLOCK_K;
                const int rk = idx % BLOCK_K;
                const int gn = n0 + c;
                const int gk = kBase + rk;
                __hip_bfloat16 w = (gk < K && gn < N) ? Wbf16[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
                nextB[c * BLOCK_K + rk] = w;
            }
        }

        __syncthreads();
        // swap ping-pong
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    // ===== Store (map float4 per lane to (row, colGroup*4 + i)) =====
    const int gm = m0 + aRowBase + lane_row;
    const int gn0 = n0 + bColBase + lane_col_grp * 4;
    const bool interior = (gm < M) && (gn0 + 3 < N) &&
                          ((m0 + BLOCK_M) <= M) && ((n0 + BLOCK_N) <= N);

    if (interior) {
        // fast path: all 4 cols in-bounds
        float* out = C + (size_t)gm * N + gn0;
        out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2]; out[3] = acc[3];
    } else {
        // ragged store with guards
        float* out = C + (size_t)gm * N + gn0;
        if (gm < M) {
            if (gn0 + 0 < N) out[0] = acc[0];
            if (gn0 + 1 < N) out[1] = acc[1];
            if (gn0 + 2 < N) out[2] = acc[2];
            if (gn0 + 3 < N) out[3] = acc[3];
        }
    }
}

// ======== MXFP4 path (A: fp32, W: uint8 packed → dequant → bf16 in LDS) ========
__global__ void gemm_mfma_mxfp4_kernel(
    float* __restrict__ C,                        // [M, N]
    const float* __restrict__ A,                  // [M, K]
    const uint8_t* __restrict__ Wp,               // [N, K] packed
    const float* __restrict__ weight_scales,      // dequant params
    int M, int K, int N, size_t total_weight_elements)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane    = threadIdx.x;
    const int wave    = threadIdx.y;
    const int wave_m  = wave / WAVES_N;
    const int wave_n  = wave % WAVES_N;

    const int lane_row      = lane & 0xF;
    const int lane_col_grp  = lane >> 4;

    extern __shared__ unsigned char smem[];
    auto* sA0 = reinterpret_cast<__hip_bfloat16*>(smem);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);
    auto* sC  = reinterpret_cast<float*>(sB1 + (BLOCK_K * BLOCK_N));

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    auto load_B_tile = [&](__hip_bfloat16* dst, int kBase) {
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            const int c  = idx / BLOCK_K;            // 0..BLOCK_N-1
            const int rk = idx % BLOCK_K;            // 0..BLOCK_K-1
            const int gn = n0 + c;
            const int gk = kBase + rk;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[c * BLOCK_K + rk] = __float2bfloat16(wb);
        }
    };

    // Preload k-slice 0
    {
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            const int r  = idx / BLOCK_K;
            const int kc = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = kc;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA0[r * BLOCK_K + kc] = __float2bfloat16(a);
        }
        load_B_tile(sB0, 0);
    }
    __syncthreads();

    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;
    const int ldA = BLOCK_K, ldB = BLOCK_K;

    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Issue MFMA on current slice
        {
            const int aRow = aRowBase + lane_row;
            const __hip_bfloat16* aPtr = currA + aRow * ldA;
            u16x4 a4 = *reinterpret_cast<const u16x4*>(aPtr);

            const int col = bColBase + lane_col_grp * 4;
            const __hip_bfloat16* bPtr = currB + col * ldB;
            u16x4 b4 = *reinterpret_cast<const u16x4*>(bPtr);

            acc = mfma_16x16x16_bf16(a4, b4, acc);
        }

        __syncthreads();

        // Preload next slice
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
                const int r  = idx / BLOCK_K;
                const int kc = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + kc;
                float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                nextA[r * BLOCK_K + kc] = __float2bfloat16(a);
            }
            load_B_tile(nextB, kBase);
        }

        __syncthreads();
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    // Store
    const int gm = m0 + aRowBase + lane_row;
    const int gn0 = n0 + bColBase + lane_col_grp * 4;
    if (gm < M) {
        float* out = C + (size_t)gm * N + gn0;
        if (gn0 + 0 < N) out[0] = acc[0];
        if (gn0 + 1 < N) out[1] = acc[1];
        if (gn0 + 2 < N) out[2] = acc[2];
        if (gn0 + 3 < N) out[3] = acc[3];
    }
}

// ===== Host wrappers (unchanged API) =====
void matmul(
    float* __restrict__ output,                  // [B, O]
    const float* __restrict__ input,             // [B, I]
    const __hip_bfloat16* __restrict__ weight,   // [O, I] bf16 row-major
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    // shared: 2*A + 2*B + optional sC
    size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(__hip_bfloat16) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_mfma_bf16_kernel,
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
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(__hip_bfloat16) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_mfma_mxfp4_kernel,
        grid, block, shmem_bytes, stream,
        output, input, weight_packed, weight_scales, M, K, N, total_weight_elements);
}
