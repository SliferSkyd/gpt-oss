#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "../utils.hpp"
// ===================================
// Vector types and small helpers
// ===================================
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4))); // 4×i16 bit-cast

__device__ inline uint16_t f32_to_bf16_bits(float x) {
    __hip_bfloat16 t = __float2bfloat16(x);
    return *reinterpret_cast<uint16_t*>(&t);
}
__device__ inline uint16_t hipbf16_to_bits(__hip_bfloat16 x) {
    return *reinterpret_cast<uint16_t*>(&x);
}
__device__ inline int lane_row(int lane)   { return lane & 15; }   // 0..15
__device__ inline int lane_group(int lane) { return lane >> 4; }   // 0..3

// MFMA wrapper (gfx90a bf16->f32 accumulate)
__device__ inline f32x4 mfma_16x16x16_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x4 c_vec) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
}

// ===================================
// Tile builders and stores (templated on WM/WN)
// ===================================
template<int WM>
__device__ inline bf16x4 make_a_vec_k(const uint16_t* __restrict__ sA,
                                      int ldA, int aRowBase, int kOff, int lane) {
    const int r   = aRowBase + lane_row(lane);
    const int grp = lane_group(lane); // 0..3
    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = sA[r * ldA + (kOff + grp * 4 + i)];
    return v;
}

template<int WN>
__device__ inline bf16x4 make_b_vec_k(const uint16_t* __restrict__ sB,
                                      int ldB, int bColBase, int kOff, int lane) {
    const int col = bColBase + lane_row(lane); // 0..15 within 16×16
    const int grp = lane_group(lane);          // 0..3
    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = sB[col * ldB + (kOff + grp * 4 + i)];
    return v;
}

template<int WM, int WN, bool Interior>
__device__ inline void store_c_tile(const f32x4& acc,
                                    float* __restrict__ C,
                                    int M, int N, int m0, int n0,
                                    int wave_m_tile, int wave_n_tile, int lane) {
    const int rowBase = m0 + wave_m_tile * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n_tile * WN + lane_row(lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if constexpr (Interior) {
            C[(size_t)row * N + col] = acc[i];
        } else {
            if (row < M && col < N) C[(size_t)row * N + col] = acc[i];
        }
    }
}

template<int WM, int WN, bool Interior>
__device__ inline void store_and_fuse_tile(const f32x4& acc,
                                           float* __restrict__ X,                  // in/out [M,N]
                                           const __hip_bfloat16* __restrict__ bias,// [N] bf16
                                           int M, int N, int m0, int n0,
                                           int wave_m_tile, int wave_n_tile, int lane) {
    const int rowBase = m0 + wave_m_tile * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n_tile * WN + lane_row(lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if constexpr (!Interior) {
            if (row >= M || col >= N) continue;
        }
        float v = acc[i];
        v += __bfloat162float(bias[col]);      // bias (bf16 -> f32)
        v += X[(size_t)row * N + col];         // residual
        X[(size_t)row * N + col] = v;
    }
}

// ===================================
// Vectorized global->LDS copies (templated on BLOCK_* and LD_*)
// ===================================
__device__ inline uint32_t pack2_bf16_bits_f32(float a0, float a1) {
    const __hip_bfloat16 b0 = __float2bfloat16(a0);
    const __hip_bfloat16 b1 = __float2bfloat16(a1);
    return (uint32_t(hipbf16_to_bits(b1)) << 16) | uint32_t(hipbf16_to_bits(b0));
}

template<int BLOCK_M, int BLOCK_K, int LD_A, bool Aligned>
__device__ inline void copy_A_tile_vec(uint32_t* __restrict__ dst_u32,
                                       const float* __restrict__ A,
                                       int m_start, int M, int K, int kBase,
                                       int linearT, int threadsPerBlock) {
    const int pairsPerRow = BLOCK_K >> 1;
    const int totalPairs  = BLOCK_M * pairsPerRow;
    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int r  = t / pairsPerRow;
        const int p  = t % pairsPerRow;
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);
        uint32_t val = 0u;
        if (gm < M && gk < K) {
            const size_t base = (size_t)gm * K + gk;
            if constexpr (Aligned) {
                const float2 v = *reinterpret_cast<const float2*>(&A[base]);
                val = pack2_bf16_bits_f32(v.x, v.y);
            } else {
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits_f32(a0, a1);
            }
        }
        reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)r * LD_A >> 1))[p] = val;
    }
}

template<int BLOCK_N, int BLOCK_K, int LD_B, bool Use128b>
__device__ inline void copy_B_tile_vec(uint32_t* __restrict__ dst_u32,
                                       const __hip_bfloat16* __restrict__ W, // [N,K] row-major
                                       int n0, int N, int K, int kBase,
                                       int linearT, int threadsPerBlock) {
    if constexpr (Use128b) {
        const int quadPerCol = BLOCK_K / 8;   // 8 bf16 per 128b
        const int totalQuads = BLOCK_N * quadPerCol;
        for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
            const int c   = t / quadPerCol;
            const int q   = t % quadPerCol;
            const int gn  = n0 + c;
            const int gk8 = kBase + (q << 3);
            uint4 v{0,0,0,0};
            if (gn < N && (gk8 + 7) < K) {
                const uint4* src = reinterpret_cast<const uint4*>(&W[(size_t)gn * K + gk8]);
                v = *src;
            } else {
                __hip_bfloat16 tmp[8] = {};
                #pragma unroll
                for (int i=0;i<8 && (gk8+i)<K && gn<N;i++) tmp[i] = W[(size_t)gn*K + gk8 + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0],p[1],p[2],p[3]);
            }
            uint32_t* col = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1));
            const int off = q << 2;
            col[off+0]=v.x; col[off+1]=v.y; col[off+2]=v.z; col[off+3]=v.w;
        }
    } else {
        const int pairsPerCol = BLOCK_K >> 1;
        const int totalPairs  = BLOCK_N * pairsPerCol;
        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int c  = t / pairsPerCol;
            const int p  = t % pairsPerCol;
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);
            uint32_t val = 0u;
            if (gn < N && gk < K) {
                const size_t base = (size_t)gn * K + gk;
                if (gk + 1 < K) val = *reinterpret_cast<const uint32_t*>(&W[base]);
                else            val = uint32_t(hipbf16_to_bits(W[base]));
            }
            reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = val;
        }
    }
}

// ===================================
// Unified MFMA kernel (templated)
// FUSED=false: C = A*W
// FUSED=true : C += A*W + bias
// ===================================
template<
    // MFMA micro-tile
    int WM, int WN, int WK,
    // Block waves (logical)
    int WAVES_M, int WAVES_N, int WAVES_K,
    // Tiles per wave (multi-tile per wave)
    int TW_M, int TW_N,
    // Padding along K in LDS
    int PAD_K_MC,
    // Fused epilogue?
    bool FUSED
>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2)
void mfma_bf16_kernel(
    float* __restrict__ C,                        // [M,N] (out or in/out if FUSED)
    const float* __restrict__ A,                  // [M,K] fp32
    const __hip_bfloat16* __restrict__ Wbf16,     // [N,K] bf16 row-major
    const __hip_bfloat16* __restrict__ bias,      // [N] bf16 (used only if FUSED)
    int M, int K, int N)
{
    static_assert(WM==16 && WN==16 && WK==16, "This kernel assumes 16x16x16 bf16 MFMA.");
    static_assert((WAVES_M % TW_M)==0 && (WAVES_N % TW_N)==0, "TW_* must divide WAVES_*");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    constexpr int WAVES_PER_BLOCK_E = WAVES_M_E * WAVES_N_E;
    static_assert((BLOCK_K % 2)==0, "BLOCK_K must be even");
    static_assert((PAD_K_MC % 2)==0, "PAD_K_MC must be even");

    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane = threadIdx.x;                    // 0..63
    const int wave = threadIdx.y;                    // 0..WAVES_PER_BLOCK_E-1
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    // Base tile indices (in 16s) owned by this wave
    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    // Bases in elements inside the block tile
    const int aBase0 = tile_m0 * WM;
    const int bBase0 = tile_n0 * WN;

    // LDS ping–pong (bf16 bits)
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    // Multi-tile accumulators
    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn)
        acc[tm][tn] = {0.f,0.f,0.f,0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    // Decide vectorization paths once per block
    const bool alignedA  = (((uintptr_t)A     & 0x7)==0) && ((K & 1)==0); // 8B & even
    const bool use128bW  = (((uintptr_t)Wbf16 & 0xF)==0) && ((K & 7)==0); // 16B & K%8==0

    // Preload kBase=0
    if (alignedA) copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, true >(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, false>(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);

    if (use128bW) copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, true >(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);
    else          copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, false>(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    // Ping-pong swap vars
    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    const int Kmain     = (K / BLOCK_K) * BLOCK_K;
    const bool has_tail = (Kmain < K);

    #pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BLOCK_K) {
        const int kNext = k0 + BLOCK_K;

        // Prefetch next
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, true >(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, false>(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);

            if (use128bW) copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, true >(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
            else          copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, false>(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        // Consume current tiles, MFMA in steps of WK=16
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec[TW_M];
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase, kk, lane);
            }
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
        if (kNext < Kmain) {
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    // Tail slice
    if (has_tail) {
        if (alignedA) copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, true >(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, false>(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, true >(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);
        else          copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, false>(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();
        currA = nextA; currB = nextB;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec[TW_M];
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase, kk, lane);
            }
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // Stores
    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm) {
    #pragma unroll
        for (int tn = 0; tn < TW_N; ++tn) {
            const int wmt = tile_m0 + tm;
            const int wnt = tile_n0 + tn;
            if constexpr (FUSED) {
                if (interior) store_and_fuse_tile<WM,WN,true >(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt, lane);
                else          store_and_fuse_tile<WM,WN,false>(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt, lane);
            } else {
                if (interior) store_c_tile<WM,WN,true >(acc[tm][tn], C, M, N, m0, n0, wmt, wnt, lane);
                else          store_c_tile<WM,WN,false>(acc[tm][tn], C, M, N, m0, n0, wmt, wnt, lane);
            }
        }
    }
}

// ===================================
// Host launcher (templated)
// - FUSED=false: plain GEMM, C = A*W
// - FUSED=true : fused epilogue, C += A*W + bias
// ===================================
template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=4, int WAVES_K=2,
    int TW_M=1, int TW_N=2,
    int PAD_K_MC=0,
    bool FUSED=false
>
inline void matmul(
    float* __restrict__ C,                        // [M,N]   (in/out if FUSED, out if plain)
    const float* __restrict__ A,                  // [M,K]
    const __hip_bfloat16* __restrict__ Wbf16,     // [N,K]
    int M, int K, int N,
    const __hip_bfloat16* __restrict__ bias=nullptr, // required if FUSED=true
    hipStream_t stream=nullptr)
{
    static_assert(WM==16 && WN==16 && WK==16, "This launcher assumes MFMA 16x16x16 bf16.");
    static_assert((WAVES_M % TW_M)==0 && (WAVES_N % TW_N)==0, "TW_* must divide WAVES_*");

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    constexpr int WAVES_PER_BLOCK_E = WAVES_M_E * WAVES_N_E;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(64, WAVES_PER_BLOCK_E);

    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(2 * BLOCK_M * ldA + 2 * ldB * BLOCK_N);

    // Optional: sanity check for FUSED
#ifndef NDEBUG
    if constexpr (FUSED) {
        if (!bias) {
            printf("launch_matmul_tuned<FUSED=true>: bias must be non-null\n");
        }
    }
#endif

    hipLaunchKernelGGL(
        (mfma_bf16_kernel<
            WM, WN, WK,
            WAVES_M, WAVES_N, WAVES_K,
            TW_M, TW_N,
            PAD_K_MC,
            FUSED>),
        grid, block, shmem_bytes, stream,
        C, A, Wbf16, bias, M, K, N);
}



// ---- 16B alignment helper ----
__device__ __forceinline__ bool is_aligned_16B(const void* p) {
    return (((uintptr_t)p) & 0xF) == 0;
}

// ---- A (global FP32) -> LDS (bf16 as u16; we write u32 pairs) ----
// One 16B unit = float4 => 4 bf16 => 2 u32 pairs.
template<int LD_A, int BM, int BK>
__device__ inline void copy_A_tile_vec128_fp32_tiled_2(
    uint32_t* __restrict__ dst_u32,
    const float* __restrict__ A,
    int m_start, int M_bound, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BK % 4) == 0, "BK must be multiple of 4 floats (16B).");
    constexpr int quadsPerRow = BK / 4;      // 4 floats per 16B
    const int totalQuads      = BM * quadsPerRow;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int r   = t / quadsPerRow;         // 0..BM-1
        const int q   = t % quadsPerRow;         // 16B unit along K
        const int gm  = m_start + r;             // global row
        const int gk4 = kBase + (q << 2);        // float index (×4)

        uint32_t p0 = 0u, p1 = 0u;
        if (gm < M_bound) {
            const size_t base = (size_t)gm * K + gk4;
            if ((gk4 + 3) < K && is_aligned_16B(&A[base])) {
                const float4 v = *reinterpret_cast<const float4*>(&A[base]); // 16B coalesced
                p0 = pack2_bf16_bits_f32(v.x, v.y);
                p1 = pack2_bf16_bits_f32(v.z, v.w);
            } else {
                float tmp[4] = {0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int i=0;i<4 && (gk4+i)<K;++i) tmp[i] = A[base + i];
                p0 = pack2_bf16_bits_f32(tmp[0], tmp[1]);
                p1 = pack2_bf16_bits_f32(tmp[2], tmp[3]);
            }
        }
        // write 2×u32 to the pair-addressed row
        uint32_t* row = dst_u32 + ((size_t)r * LD_A >> 1);
        const int off = (q << 1);
        row[off + 0] = p0;
        row[off + 1] = p1;
    }
}

// ---- B (global bf16) -> LDS (bf16; write u32 pairs) ----
// One 16B unit = 8 bf16 => 4 u32 pairs.
template<int LD_B, int BN, int BK>
__device__ inline void copy_B_tile_vec128_bf16_tiled_2(
    uint32_t* __restrict__ dst_u32,
    const __hip_bfloat16* __restrict__ W, // [N,K] row-major
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BK % 8) == 0, "BK must be multiple of 8 bf16 (16B).");
    constexpr int quadsPerCol = BK / 8;      // 8 bf16 per 16B
    const int totalQuads      = BN * quadsPerCol;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int c   = t / quadsPerCol;         // 0..BN-1
        const int q   = t % quadsPerCol;         // 16B unit along K
        const int gn  = n0 + c;                  // global col (N)
        const int gk8 = kBase + (q << 3);        // bf16 index (×8)

        uint4 v = {0,0,0,0};
        if (gn < N) {
            const size_t base = (size_t)gn * K + gk8;
            if ((gk8 + 7) < K && is_aligned_16B(&W[base])) {
                v = *reinterpret_cast<const uint4*>(&W[base]);  // 16B coalesced
            } else {
                __hip_bfloat16 tmp[8] = {};
                #pragma unroll
                for (int i=0;i<8 && (gk8+i)<K;++i) tmp[i] = W[base + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0], p[1], p[2], p[3]);
            }
        }
        uint32_t* col = dst_u32 + ((size_t)c * LD_B >> 1);
        const int off = (q << 2);
        col[off + 0] = v.x;
        col[off + 1] = v.y;
        col[off + 2] = v.z;
        col[off + 3] = v.w;
    }
}


template<
    // MFMA micro-tile
    int WM, int WN, int WK,
    // Block waves
    int WAVES_M, int WAVES_N, int WAVES_K,
    // Multi-tiles per wave
    int TW_M, int TW_N,
    // LDS padding along K (bf16 elems)
    int PAD_K_MC,
    // Fused epilogue?
    bool FUSED
>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2)
void mfma_bf16_kernel_vec128_singlebuf(
    float* __restrict__ C,                        // [M,N] (in/out if FUSED)
    const float* __restrict__ A,                  // [M,K] fp32
    const __hip_bfloat16* __restrict__ Wbf16,     // [N,K] bf16 row-major
    const __hip_bfloat16* __restrict__ bias,      // [N] (used iff FUSED)
    int M, int K, int N)
{
    static_assert(WM==16 && WN==16 && WK==16, "16x16x16 bf16 MFMA required.");
    static_assert((WAVES_M % TW_M)==0 && (WAVES_N % TW_N)==0, "TW_* must divide WAVES_*");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;  // must be multiple of 16
    static_assert((BLOCK_K % 2)==0, "BLOCK_K must be even");
    static_assert((PAD_K_MC % 2)==0, "PAD_K_MC must be even");

    // ---- block origin ----
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    // ---- wave & lane geometry ----
    const int lane = threadIdx.x;                      // 0..63
    const int wave = threadIdx.y;                      // 0..(WAVES_M/TW_M*WAVES_N/TW_N-1)
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    const int tile_m0 = wave_m_eff * TW_M;            // multi-tile indices
    const int tile_n0 = wave_n_eff * TW_N;

    const int aBase0  = tile_m0 * WM;                 // element offsets inside block tile
    const int bBase0  = tile_n0 * WN;

    // ---- LDS (single buffer): [A | B_aligned] ----
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;           // bf16 pitch
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BLOCK_M * ldA);
    const size_t sB_off   = (sA_bytes + 15) & ~size_t(15); // 16B align B tile
    uint16_t* sA_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sB_u16 = reinterpret_cast<uint16_t*>(smemRaw + sB_off);

    uint32_t* sA_u32 = reinterpret_cast<uint32_t*>(sA_u16);
    uint32_t* sB_u32 = reinterpret_cast<uint32_t*>(sB_u16);

    // ---- accumulators ----
    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm)
    #pragma unroll
    for (int tn=0; tn<TW_N; ++tn)
        acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    // ---- threading helpers ----
    const int threadsPerBlock = blockDim.x * blockDim.y;    // 64 * (#waves)
    const int linearT         = wave * blockDim.x + lane;

    // ---- main K loop: copy -> sync -> compute -> sync ----
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        copy_A_tile_vec128_fp32_tiled_2<ldA, BLOCK_M, BLOCK_K>(
            sA_u32, A, m0, M, K, k0, linearT, threadsPerBlock);

        copy_B_tile_vec128_bf16_tiled_2<ldB, BLOCK_N, BLOCK_K>(
            sB_u32, Wbf16, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

        // consume current tiles in steps of WK (=16)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK)
        {
            bf16x4 avec[TW_M];
            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase, kk, lane);
            }

            bf16x4 bvec[TW_N];
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase, kk, lane);
            }

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // ---- stores (masked on borders) ----
    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;

    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm)
    {
        const int wmt = tile_m0 + tm;
        #pragma unroll
        for (int tn=0; tn<TW_N; ++tn)
        {
            const int wnt = tile_n0 + tn;
            if constexpr (FUSED) {
                if (interior) store_and_fuse_tile<WM,WN,true >(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt, lane);
                else          store_and_fuse_tile<WM,WN,false>(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt, lane);
            } else {
                if (interior) store_c_tile<WM,WN,true >(acc[tm][tn], C, M, N, m0, n0, wmt, wnt, lane);
                else          store_c_tile<WM,WN,false>(acc[tm][tn], C, M, N, m0, n0, wmt, wnt, lane);
            }
        }
    }
}

// ===================================
// Host launcher (vec128 + single buffer)
// Pick PAD_K_MC=16 if you see any LDS conflicts.
// Example fast defaults (gfx90a):
//  - WAVES_M=1, WAVES_N=8, WAVES_K=2, TW_M=1, TW_N=2, PAD_K_MC=16
// ===================================
template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=8, int WAVES_K=2,
    int TW_M=1, int TW_N=2,
    int PAD_K_MC=16,
    bool FUSED=false
>
inline void matmul_vec128_singlebuf(
    float* __restrict__ C,                        // [M,N] (in/out if FUSED)
    const float* __restrict__ A,                  // [M,K]
    const __hip_bfloat16* __restrict__ Wbf16,     // [N,K]
    int M, int K, int N,
    const __hip_bfloat16* __restrict__ bias=nullptr,
    hipStream_t stream=nullptr)
{

    // char timer_name[256];
    // snprintf(timer_name, sizeof(timer_name), "matmul_vec128_singlebuf_M%d_K%d_N%d", M, K, N);
    // TIMER_BLOCK(timer_name);
    static_assert(WM==16 && WN==16 && WK==16, "16x16x16 bf16 MFMA required.");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);
    constexpr int WAVES_PER_BLOCK_E = (WAVES_M / TW_M) * (WAVES_N / TW_N);
    dim3 block(64, WAVES_PER_BLOCK_E);

    // single-buffer LDS size: A + aligned B
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BLOCK_M * ldA);
    const size_t sB_bytes = sizeof(uint16_t) * (size_t)(BLOCK_N * ldB);
    const size_t shmem_bytes = ((sA_bytes + 15) & ~size_t(15)) + sB_bytes;

#ifndef NDEBUG
    if constexpr (FUSED) {
        if (!bias) fprintf(stderr, "matmul_vec128_singlebuf<FUSED>: bias is null\n");
    }
#endif

    hipLaunchKernelGGL(
        (mfma_bf16_kernel_vec128_singlebuf<
            WM,WN,WK,
            WAVES_M,WAVES_N,WAVES_K,
            TW_M,TW_N,
            PAD_K_MC,
            FUSED>),
        grid, block, shmem_bytes, stream,
        C, A, Wbf16, bias, M, K, N);
    HIP_CHECK(hipGetLastError());
}


// ==== 32x32x8 path (gfx90a bf16 -> f32 MFMA) =================================

// Types (define once in the TU; remove if already present)
using f32x16 = float __attribute__((ext_vector_type(16)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4)));

// MFMA wrapper: v_mfma_f32_32x32x8bf16_1k
__device__ inline f32x16 mfma_32x32x8_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x16 c_vec) {
    return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
}

__device__ inline int lane_row32(int lane)  { return lane & 31; } // 0..31
__device__ inline int lane_half32(int lane) { return lane >> 5; } // 0(left)/1(right)

// A: read 4 bf16 from LDS for this lane for the current kk-group (k_group ∈ {0,1})
template<int WM /*=32*/>
__device__ inline bf16x4 make_a_vec_k32(const uint16_t* __restrict__ sA,
                                        int ldA, int aRowBase, int kOff, int lane)
{
    static_assert(WM == 32, "make_a_vec_k32 assumes WM=32");
    const int r       = aRowBase + lane_row32(lane);   // 0..31 row inside 32×32 wave-tile
    const int k_group = (lane >> 4) & 1;               // bit4 selects 0..3 vs 4..7 within WK=8

    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = sA[r * ldA + (kOff + k_group * 4 + i)];
    return v;
}

// B: read 4 bf16 from LDS for this lane for the current kk-group
template<int WN /*=32*/>
__device__ inline bf16x4 make_b_vec_k32(const uint16_t* __restrict__ sB,
                                        int ldB, int bColBase, int kOff, int lane)
{
    static_assert(WN == 32, "make_b_vec_k32 assumes WN=32");
    const int col     = bColBase + lane_row32(lane);   // 0..31 col inside 32×32 wave-tile
    const int k_group = (lane >> 4) & 1;               // matches make_a_vec_k32

    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = sB[col * ldB + (kOff + k_group * 4 + i)];
    return v;
}

// ---- 16B vector copies (single-buffer LDS). Keep only one copy in TU. ----

// A (global FP32) -> LDS (bf16 as u16; we write u32 pairs)
template<int LD_A, int BM, int BK>
__device__ inline void copy_A_tile_vec128_fp32_tiled_32(
    uint32_t* __restrict__ dst_u32,
    const float* __restrict__ A,
    int m_start, int M_bound, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BK % 4) == 0, "BK must be multiple of 4 floats (16B).");
    constexpr int quadsPerRow = BK / 4;  // 4 floats per 16B
    const int totalQuads      = BM * quadsPerRow;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int r   = t / quadsPerRow;       // 0..BM-1
        const int q   = t % quadsPerRow;       // 16B unit along K
        const int gm  = m_start + r;
        const int gk4 = kBase + (q << 2);

        uint32_t p0 = 0u, p1 = 0u;
        if (gm < M_bound) {
            const size_t base = (size_t)gm * K + gk4;
            if ((gk4 + 3) < K && is_aligned_16B(&A[base])) {
                const float4 v = *reinterpret_cast<const float4*>(&A[base]); // 16B
                p0 = pack2_bf16_bits_f32(v.x, v.y);
                p1 = pack2_bf16_bits_f32(v.z, v.w);
            } else {
                float tmp[4] = {0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int i=0;i<4 && (gk4+i)<K;++i) tmp[i] = A[base + i];
                p0 = pack2_bf16_bits_f32(tmp[0], tmp[1]);
                p1 = pack2_bf16_bits_f32(tmp[2], tmp[3]);
            }
        }
        uint32_t* row = dst_u32 + ((size_t)r * LD_A >> 1);
        const int off = (q << 1);
        row[off + 0] = p0;
        row[off + 1] = p1;
    }
}

// B (global bf16) -> LDS (bf16; write u32 pairs)
template<int LD_B, int BN, int BK>
__device__ inline void copy_B_tile_vec128_bf16_tiled_32(
    uint32_t* __restrict__ dst_u32,
    const __hip_bfloat16* __restrict__ W, // [N,K] row-major
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BK % 8) == 0, "BK must be multiple of 8 bf16 (16B).");
    constexpr int quadsPerCol = BK / 8;  // 8 bf16 per 16B
    const int totalQuads      = BN * quadsPerCol;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int c   = t / quadsPerCol;    // 0..BN-1
        const int q   = t % quadsPerCol;    // 16B unit along K
        const int gn  = n0 + c;
        const int gk8 = kBase + (q << 3);

        uint4 v = {0,0,0,0};
        if (gn < N) {
            const size_t base = (size_t)gn * K + gk8;
            if ((gk8 + 7) < K && is_aligned_16B(&W[base])) {
                v = *reinterpret_cast<const uint4*>(&W[base]); // 16B
            } else {
                __hip_bfloat16 tmp[8] = {};
                #pragma unroll
                for (int i=0;i<8 && (gk8+i)<K;++i) tmp[i] = W[base + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0], p[1], p[2], p[3]);
            }
        }
        uint32_t* col = dst_u32 + ((size_t)c * LD_B >> 1);
        const int off = (q << 2);
        col[off + 0] = v.x;
        col[off + 1] = v.y;
        col[off + 2] = v.z;
        col[off + 3] = v.w;
    }
}

// ---- lane helpers for 32x32x8 ----
// We launch each wave as a 2D shape: x=32 lanes, y=2 rows (two half-waves).
// If we pack multiple waves per block, y spans 2 * WAVES_PER_BLOCK_E.
__device__ inline int lane_col32() { return threadIdx.x; }          // 0..31 = column within 32x32 tile
__device__ inline int lane_half2() { return (threadIdx.y & 1); }    // 0 or 1 : selects K-subgroup and row offsets
__device__ inline int wave_eid()   { return threadIdx.y >> 1; }     // effective wave id inside the block (excludes the 2 half rows)

// ---- A/B LDS readers: k_group is from threadIdx.y (NOT from lane bits) ----
template<int WM /*=32*/>
__device__ inline bf16x4 make_a_vec_k32(const uint16_t* __restrict__ sA,
                                        int ldA, int aRowBase, int kOff)
{
    static_assert(WM == 32, "WM must be 32");
    const int row     = aRowBase + lane_col32(); // lanes index rows
    const int k_group = lane_half2();            // 0: kOff+0..3, 1: kOff+4..7
    bf16x4 v;
    #pragma unroll
    for (int i=0;i<4;++i) v[i] = sA[row * ldA + (kOff + k_group * 4 + i)];
    return v;
}

template<int WN /*=32*/>
__device__ inline bf16x4 make_b_vec_k32(const uint16_t* __restrict__ sB,
                                        int ldB, int bColBase, int kOff)
{
    static_assert(WN == 32, "WN must be 32");
    const int col     = bColBase + lane_col32(); // lanes index columns
    const int k_group = lane_half2();
    bf16x4 v;
    #pragma unroll
    for (int i=0;i<4;++i) v[i] = sB[col * ldB + (kOff + k_group * 4 + i)];
    return v;
}

// ---- Stores (lane = column; 16 acc elems = 4× blocks-of-8 rows) ----
// AMD mapping: For each block j=0..3 (8-row block), and i=0..3 within that block,
// row = j*8 + (lane_half2()*4) + i ; column = lane_col32()
template<int WM /*=32*/, int WN /*=32*/, bool Interior>
__device__ inline void store_c_tile32(const f32x16& acc,
                                      float* __restrict__ C,
                                      int M, int N, int m0, int n0,
                                      int wmt, int wnt)
{
    static_assert(WM==32 && WN==32, "store_c_tile32 assumes 32x32 wave-tile");
    const int col = n0 + wnt * WN + lane_col32(); // fixed column for this lane

    #pragma unroll
    for (int j=0; j<4; ++j) {          // 4 blocks of 8 rows
        #pragma unroll
        for (int i=0; i<4; ++i) {      // 4 rows per half (we have 2 halves)
            const int elem = i + 4*j;  // selects which of the 16 acc elements
            const int row = m0 + wmt * WM + (j*8 + lane_half2()*4 + i);
            if constexpr (Interior) {
                C[(size_t)row * N + col] = acc[elem];
            } else {
                if (row < M && col < N) C[(size_t)row * N + col] = acc[elem];
            }
        }
    }
}

template<int WM /*=32*/, int WN /*=32*/, bool Interior>
__device__ inline void store_and_fuse_tile32(const f32x16& acc,
                                             float* __restrict__ X,  // in/out
                                             const __hip_bfloat16* __restrict__ bias,
                                             int M, int N, int m0, int n0,
                                             int wmt, int wnt)
{
    static_assert(WM==32 && WN==32, "store_and_fuse_tile32 assumes 32x32 wave-tile");
    const int col = n0 + wnt * WN + lane_col32();

    #pragma unroll
    for (int j=0; j<4; ++j) {
        #pragma unroll
        for (int i=0; i<4; ++i) {
            const int elem = i + 4*j;
            const int row  = m0 + wmt * WM + (j*8 + lane_half2()*4 + i);
            if constexpr (!Interior) {
                if (row >= M || col >= N) continue;
            }
            float v = acc[elem];
            v += __bfloat162float(bias[col]);
            v += X[(size_t)row * N + col];
            X[(size_t)row * N + col] = v;
        }
    }
}

// ----------------- main kernel (single-buffer LDS) -----------------
template<
    int WM, int WN, int WK,             // expect 32, 32, 8
    int WAVES_M, int WAVES_N, int WAVES_K,
    int TW_M, int TW_N,
    int PAD_K_MC,
    bool FUSED
>
__global__ __launch_bounds__(32 * (2 * ((WAVES_M / TW_M) * (WAVES_N / TW_N))), 2)
void mfma_bf16_kernel32_vec128_singlebuf(
    float* __restrict__ C,
    const float* __restrict__ A,
    const __hip_bfloat16* __restrict__ Wbf16,
    const __hip_bfloat16* __restrict__ bias,
    int M, int K, int N)
{
    static_assert(WM==32 && WN==32 && WK==8, "32x32x8 path");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    constexpr int WAVES_PER_BLOCK_E = WAVES_M_E * WAVES_N_E;

    // block origin
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    // effective wave id (strip the 2-row shape)
    const int wave_eff = wave_eid();
    const int wave_m_eff = wave_eff / WAVES_N_E;
    const int wave_n_eff = wave_eff % WAVES_N_E;

    // multi-tile indices inside the block
    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    // element bases inside the block
    const int aBase0  = tile_m0 * WM;
    const int bBase0  = tile_n0 * WN;

    // LDS single buffer: [A | B]
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC; // bf16 pitch
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BLOCK_M * ldA);
    const size_t sB_off   = (sA_bytes + 15) & ~size_t(15);
    uint16_t* sA_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sB_u16 = reinterpret_cast<uint16_t*>(smemRaw + sB_off);
    uint32_t* sA_u32 = reinterpret_cast<uint32_t*>(sA_u16);
    uint32_t* sB_u32 = reinterpret_cast<uint32_t*>(sB_u16);

    // accumulators (per-lane 16 floats map to 32 rows via the (j,i) scheme)
    f32x16 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm)
    #pragma unroll
    for (int tn=0; tn<TW_N; ++tn) {
        f32x16 z; 
        #pragma unroll 
        for (int i=0;i<16;++i) z[i]=0.0f; 
        acc[tm][tn]=z;
    }

    // thread linear id for copies (now with x=32, y=2*(waves))
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = threadIdx.y * blockDim.x + threadIdx.x;

    // main K loop
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        copy_A_tile_vec128_fp32_tiled_32<ldA, BLOCK_M, BLOCK_K>(
            sA_u32, A, m0, M, K, k0, linearT, threadsPerBlock);
        copy_B_tile_vec128_bf16_tiled_32<ldB, BLOCK_N, BLOCK_K>(
            sB_u32, Wbf16, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

        // consume WK=8
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k32<WM>(sA_u16, ldA, aRowBase, kk);
            }
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k32<WN>(sB_u16, ldB, bColBase, kk);
            }

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_32x32x8_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // stores
    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm) {
        const int wmt = tile_m0 + tm;
        #pragma unroll
        for (int tn=0; tn<TW_N; ++tn) {
            const int wnt = tile_n0 + tn;
            if constexpr (FUSED) {
                if (interior) store_and_fuse_tile32<WM,WN,true >(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt);
                else          store_and_fuse_tile32<WM,WN,false>(acc[tm][tn], C, bias, M, N, m0, n0, wmt, wnt);
            } else {
                if (interior) store_c_tile32<WM,WN,true >(acc[tm][tn], C, M, N, m0, n0, wmt, wnt);
                else          store_c_tile32<WM,WN,false>(acc[tm][tn], C, M, N, m0, n0, wmt, wnt);
            }
        }
    }
}

// ----------------- launcher (NOTE THE BLOCK SHAPE) -----------------
template<
    int WM=32, int WN=32, int WK=8,
    int WAVES_M=1, int WAVES_N=4, int WAVES_K=4,   // BLOCK_K=32
    int TW_M=1, int TW_N=2,
    int PAD_K_MC=16,
    bool FUSED=false
>
inline void matmul32x32x8_vec128_singlebuf(
    float* __restrict__ C,
    const float* __restrict__ A,
    const __hip_bfloat16* __restrict__ Wbf16,
    int M, int K, int N,
    const __hip_bfloat16* __restrict__ bias=nullptr,
    hipStream_t stream=nullptr)
{
    static_assert(WM==32 && WN==32 && WK==8, "32x32x8 path");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_PER_BLOCK_E = (WAVES_M / TW_M) * (WAVES_N / TW_N);

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    // IMPORTANT: 32×2 per wave (not 64×1)
    dim3 block(32, 2 * WAVES_PER_BLOCK_E);

    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BLOCK_M * ldA);
    const size_t sB_bytes = sizeof(uint16_t) * (size_t)(BLOCK_N * ldB);
    const size_t shmem_bytes = ((sA_bytes + 15) & ~size_t(15)) + sB_bytes;

#ifndef NDEBUG
    if constexpr (FUSED) {
        if (!bias) fprintf(stderr, "matmul32x32x8_vec128_singlebuf<FUSED>: bias is null\n");
    }
#endif

    hipLaunchKernelGGL(
        (mfma_bf16_kernel32_vec128_singlebuf<
            WM,WN,WK,
            WAVES_M,WAVES_N,WAVES_K,
            TW_M,TW_N,
            PAD_K_MC,
            FUSED>),
        grid, block, shmem_bytes, stream,
        C, A, Wbf16, bias, M, K, N);
    HIP_CHECK(hipGetLastError());
}
