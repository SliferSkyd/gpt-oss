#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

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
template<int WM>
__device__ inline bf16x4 make_a_vec_k(const uint16_t* __restrict__ sA,
                                      int ldA, int aRowBase, int kOff, int lane) {
    const int r   = aRowBase + lane_row(lane);
    const int grp = lane_group(lane); // 0..3
    // 4 consecutive bf16 along K for this subgroup
    const bf16x4* p = reinterpret_cast<const bf16x4*>(&sA[(size_t)r * ldA + (kOff + grp * 4)]);
    return *p; // single LDS read
}

template<int WN>
__device__ inline bf16x4 make_b_vec_k(const uint16_t* __restrict__ sB,
                                      int ldB, int bColBase, int kOff, int lane) {
    const int col = bColBase + lane_row(lane); // 0..15 within 16×16
    const int grp = lane_group(lane);          // 0..3
    const bf16x4* p = reinterpret_cast<const bf16x4*>(&sB[(size_t)col * ldB + (kOff + grp * 4)]);
    return *p; // single LDS read
}


template<int WM, int WN, bool Interior>
__device__ inline void store_c_tile(const f32x4& acc,
                                    float* __restrict__ C,
                                    int M, int N, int m0, int n0,
                                    int wave_m_tile, int wave_n_tile, int lane) {
    const int rowBase = m0 + wave_m_tile * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n_tile * WN + lane_row(lane);

    if constexpr (Interior) {
        // one vector store if fully interior
        float* dst = &C[(size_t)rowBase * N + col];
        *reinterpret_cast<f32x4*>(dst) = acc;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rowBase + i;
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

    if constexpr (Interior) {
        f32x4 v = acc;
        const float b = __bfloat162float(bias[col]);
        // add bias + residual vectorized
        float* dst = &X[(size_t)rowBase * N + col];
        f32x4 res = *reinterpret_cast<const f32x4*>(dst);
        v[0] = v[0] + b + res[0];
        v[1] = v[1] + b + res[1];
        v[2] = v[2] + b + res[2];
        v[3] = v[3] + b + res[3];
        *reinterpret_cast<f32x4*>(dst) = v;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = rowBase + i;
            if (row >= M || col >= N) continue;
            float v = acc[i] + __bfloat162float(bias[col]) + X[(size_t)row * N + col];
            X[(size_t)row * N + col] = v;
        }
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

template<int BLOCK_M, int BLOCK_K, int LD_A, bool Aligned16B>
__device__ inline void copy_A_tile_vec(uint32_t* __restrict__ dst_u32,
                                       const float* __restrict__ A,
                                       int m_start, int M, int K, int kBase,
                                       int linearT, int threadsPerBlock) {
    // Each “pair” is 2 bf16 = 1 u32; 4 floats => 8 bf16 => 4 u32
    if constexpr (Aligned16B) {
        const int quadsPerRow = BLOCK_K >> 2;        // 4 fp32 per quad
        const int totalQuads  = BLOCK_M * quadsPerRow;
        for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
            const int r  = t / quadsPerRow;
            const int q  = t % quadsPerRow;
            const int gm = m_start + r;
            const int gk = kBase + (q << 2);
            uint4 out = {0,0,0,0};
            if (gm < M && (gk + 3) < K) {
                const float4 v = *reinterpret_cast<const float4*>(&A[(size_t)gm * K + gk]);
                // pack 4 fp32 -> 8 bf16 -> 4 u32
                __hip_bfloat16 tmp[8];
                tmp[0] = __float2bfloat16(v.x); tmp[1] = __float2bfloat16(v.y);
                tmp[2] = __float2bfloat16(v.z); tmp[3] = __float2bfloat16(v.w);
                tmp[4] = __float2bfloat16(0.f); tmp[5] = __float2bfloat16(0.f);
                tmp[6] = __float2bfloat16(0.f); tmp[7] = __float2bfloat16(0.f);
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                out = make_uint4(p[0], p[1], p[2], p[3]);
            } else if (gm < M && gk < K) {
                float a[4] = {0,0,0,0};
                #pragma unroll
                for (int i=0;i<4 && (gk+i)<K;i++) a[i] = A[(size_t)gm * K + gk + i];
                __hip_bfloat16 tmp[8] = {
                    __float2bfloat16(a[0]), __float2bfloat16(a[1]),
                    __float2bfloat16(a[2]), __float2bfloat16(a[3]),
                    __float2bfloat16(0.f),  __float2bfloat16(0.f),
                    __float2bfloat16(0.f),  __float2bfloat16(0.f)
                };
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                out = make_uint4(p[0], p[1], p[2], p[3]);
            }
            // write 4*32b to LDS row r at column offset q*4 pairs
            uint32_t* row = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)r * LD_A >> 1));
            const int off = q << 2;
            row[off+0] = out.x; row[off+1] = out.y; row[off+2] = out.z; row[off+3] = out.w;
        }
    } else {
        // fallback (your existing float2 path)
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
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits_f32(a0, a1);
            }
            reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)r * LD_A >> 1))[p] = val;
        }
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

    // ===== Single-buffer LDS (no ping-pong) =====
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t* sA_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sB_u16 = sA_u16 + (BLOCK_M * ldA);

    uint32_t* sA_u32 = reinterpret_cast<uint32_t*>(sA_u16);
    uint32_t* sB_u32 = reinterpret_cast<uint32_t*>(sB_u16);

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
    const bool alignedA  = (((uintptr_t)A     & 0x7)==0) && ((K & 3)==0); // 8B & even
    const bool use128bW  = (((uintptr_t)Wbf16 & 0xF)==0) && ((K & 7)==0); // 16B & K%8==0

    // Iterate K in chunks; copy -> sync -> compute; no prefetch
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (alignedA) copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, true >(sA_u32, A, m0, M, K, k0, linearT, threadsPerBlock);
        else          copy_A_tile_vec<BLOCK_M, BLOCK_K, ldA, false>(sA_u32, A, m0, M, K, k0, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, true >(sB_u32, Wbf16, n0, N, K, k0, linearT, threadsPerBlock);
        else          copy_B_tile_vec<BLOCK_N, BLOCK_K, ldB, false>(sB_u32, Wbf16, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

        // Consume current tiles, MFMA in steps of WK=16.
        // Note: copy helpers already zero-pad OOB for the tail slice.
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec[TW_M];
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase, kk, lane);
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

    // Single-buffer shared memory footprint
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(BLOCK_M * ldA + ldB * BLOCK_N);

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
