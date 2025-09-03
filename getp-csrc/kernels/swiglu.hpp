// ====== FUSED MXFP4 GEMM + BIAS + SWiGLU ======
// Keeps your tiling/tunables. Writes [M,N/2] directly (no [M,2N] spill).

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"   // dequantize_mxfp4(...)
#include <hip/hip_fp16.h>

// Store the per-lane accumulators (one column, 4 rows) into LDS sC[BLOCK_M x BLOCK_N]
__device__ inline void spill_acc_to_sC(float* __restrict__ sC,
                                       const f32x4& acc,
                                       int m0, int n0, int M, int N,
                                       int wave_m, int wave_n, int lane) {
    const int rowBase = wave_m * WM + lane_group(lane) * 4; // local row in block
    const int colBlk  = wave_n * WN + lane_row(lane);       // local col in block
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gr = m0 + rowBase + i;
        const int gc = n0 + colBlk;
        // mask global bounds; write 0 for OOB so fuser can safely read
        sC[(rowBase + i) * BLOCK_N + colBlk] =
            (gr < M && gc < N) ? acc[i] : 0.0f;
    }
}

#if MXFP4_FAST_UNPACK
// Fast symmetric 4-bit decode: q in [-8..7] * scale
__device__ inline float mxfp4_decode_q(uint8_t nibble, float scale) {
    int v = int(nibble);
    if (v >= 8) v -= 16;
    return float(v) * scale;
}
#endif

// Keep register pressure bounded to allow >=2 blocks per CU on gfx90a
__launch_bounds__(LANE_PER_WAVE * WAVES_PER_BLOCK, 2)
__global__ void gemm_mfma_uint8_swiglu_fused_kernel_v4(
    float* __restrict__ Y,                          // [M, N/2]
    const float* __restrict__ A,                    // [M, K] FP32
    const uint8_t* __restrict__ Wp,                 // [N, K] packed row-major (two 4-bit per byte)
    const float* __restrict__ weight_scales,        // per-MXFP4_BLOCK_SIZE scales
    const __hip_bfloat16* __restrict__ bias_mlp1,   // [N] bf16 (N even)
    int M, int K, int N,                            // N = 2*I
    size_t total_weight_elements,
    float clamp_limit)
{
    const int lane   = threadIdx.x;               // 0..63
    const int wave   = threadIdx.y;               // 0..WAVES_PER_BLOCK-1
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    // LDS: only A/B ping-pong (bf16 bits). No sC.
    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // A tile loader (vector-friendly index math)
    auto load_A_tile = [&](uint16_t* dst, int kBase) __device__ {
        // idx in [0, BLOCK_M*BLOCK_K). r = idx>>4, c = idx&15
        for (int idx = linearT; idx < (BLOCK_M * BLOCK_K); idx += threadsPerBlock) {
            const int r = idx >> 4;          // /16
            const int c = idx & 15;          // %16
            const int gm = m0 + r;
            const int gk = kBase + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            dst[(r << 4) + c] = f32_to_bf16_bits(a); // row-major ld=16
        }
    };

    // B tile loader
    auto load_B_tile = [&](uint16_t* dst, int kBase) __device__ {
#if MXFP4_FAST_UNPACK
        // Each thread handles a stripe in (c,r) with r in [0..15], c in [0..BLOCK_N-1]
        for (int idx = linearT; idx < (BLOCK_K * BLOCK_N); idx += threadsPerBlock) {
            const int c  = idx >> 4;           // /16
            const int r  = idx & 15;           // %16
            const int gn = n0 + c;
            const int gk = kBase + r;
            if (gn < N && gk < K) {
                // linear element index in row-major [N,K] is gn*K + gk
                const size_t lin = (size_t)gn * K + gk;
                const size_t byte_idx = lin >> 1;          // 2 vals per byte
                const bool   hi = (lin & 1);
                const uint8_t byte = Wp[byte_idx];
                const uint8_t nibble = hi ? (byte >> 4) : (byte & 0xF);
                // scale index by your block size (e.g., 32)
                const size_t sidx = lin / MXFP4_BLOCK_SIZE;
                const float scale = weight_scales[sidx];
                float wb = mxfp4_decode_q(nibble, scale);
                dst[c * BLOCK_K + r] = f32_to_bf16_bits(wb); // col-major ld=16
            } else {
                dst[c * BLOCK_K + r] = 0;
            }
        }
#else
        // Fallback to your existing exact dequant function (still faster from bit ops in idx)
        for (int idx = linearT; idx < (BLOCK_K * BLOCK_N); idx += threadsPerBlock) {
            const int c  = idx >> 4;   // /16
            const int r  = idx & 15;   // %16
            const int gn = n0 + c;
            const int gk = kBase + r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp, weight_scales, (size_t)gn * K + gk, total_weight_elements)
                : 0.0f;
            dst[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
        }
#endif
    };

    // Preload k-slice 0
    load_A_tile(sA0, /*kBase=*/0);
    load_B_tile(sB0, /*kBase=*/0);
    __syncthreads();

    auto* currA = sA0; auto* currB = sB0;
    auto* nextA = sA1; auto* nextB = sB1;

#pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Preload next (software pipeline)
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            load_A_tile(nextA, kBase);
            load_B_tile(nextB, kBase);
        }

        const int ldA = BLOCK_K, ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;

        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        // swap buffers
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    // ---- Register epilogue (ALL lanes shuffle, only even lanes store) ----
    float acc_nb0 = __shfl_xor(acc[0], 1, 64);
    float acc_nb1 = __shfl_xor(acc[1], 1, 64);
    float acc_nb2 = __shfl_xor(acc[2], 1, 64);
    float acc_nb3 = __shfl_xor(acc[3], 1, 64);

    const int colLocal = wave_n * WN + lane_row(lane);
    if ((colLocal & 1) == 0) {
        const int gcol_even = n0 + colLocal;
        const int gcol_odd  = gcol_even + 1;
        if (gcol_even < N && gcol_odd < N) {
            // vectorized bias pair load (optional)
            // NOTE: layout-safe simple path (two loads) tends to compile just as fast.
            const float bias_g = __bfloat162float(bias_mlp1[gcol_even]);
            const float bias_u = __bfloat162float(bias_mlp1[gcol_odd]);

            const int rowBaseLocal = wave_m * WM + (lane_group(lane) << 2); // *4
            const int N_half = N >> 1;
            const int outCol = (n0 >> 1) + (colLocal >> 1);
            const float alpha = 1.702f;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int grow = m0 + rowBaseLocal + i;
                if (grow >= M) break;

                float gate = (i==0?acc[0]  : i==1?acc[1]  : i==2?acc[2]  : acc[3])   + bias_g;
                float up   = (i==0?acc_nb0: i==1?acc_nb1: i==2?acc_nb2: acc_nb3) + bias_u;

                // clamp + SiLU(alpha*x) * (up+1)
                gate = fminf(fmaxf(gate, -clamp_limit), clamp_limit);
                up   = fminf(fmaxf(up,   -clamp_limit), clamp_limit);
                const float sig = 1.0f / (1.0f + __expf(-alpha * gate));
                float y = (gate * sig) * (up + 1.0f);

                if (outCol < N_half) {
                    Y[(size_t)grow * N_half + outCol] = y;
                }
            }
        }
    }
}

// ---- Host wrapper (reduced shmem; same signature) ----
inline void matmul_mxfp4_swiglu_fused(
    float* __restrict__ out_gate_up,                 // [B, I]
    const float* __restrict__ input,                 // [B, H]
    const uint8_t* __restrict__ weight_packed,       // [2I, H] packed row-major
    const float* __restrict__ weight_scales,
    const __hip_bfloat16* __restrict__ bias_mlp1,    // [2I]
    int batch_size, int hidden_dim, int intermediate_dim,
    size_t total_weight_elements,
    float clamp_limit,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = hidden_dim, N = 2 * intermediate_dim;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    // A/B ping-pong (bf16) only; no sC
    size_t shmem_bytes =
        (size_t)(2 * BLOCK_M * BLOCK_K + 2 * BLOCK_K * BLOCK_N) * sizeof(uint16_t);

    hipLaunchKernelGGL(
        gemm_mfma_uint8_swiglu_fused_kernel_v4,
        grid, block, shmem_bytes, stream,
        out_gate_up, input, weight_packed, weight_scales, bias_mlp1,
        M, K, N, total_weight_elements, clamp_limit);
}
