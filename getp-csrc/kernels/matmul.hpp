#pragma once

// mfma_gemm_gfx90a.hpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"
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

#ifndef WAVES_K
#define WAVES_K 2
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M = WM * WAVES_M;        // e.g. 64 if WAVES_M=4
constexpr int BLOCK_N = WN * WAVES_N;        // e.g. 64 if WAVES_N=4
constexpr int BLOCK_K = WK * WAVES_K;                  // 16
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

__device__ inline bf16x4 make_a_vec_k(const uint16_t* __restrict__ sA,
                                       int ldA, int aRowBase, int kOff, int lane) {
    const int r   = aRowBase + lane_row(lane);
    const int grp = lane_group(lane);
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = sA[r * ldA + (kOff + grp * 4 + i)];
    }
    return v;
}

__device__ inline bf16x4 make_b_vec_k(const uint16_t* __restrict__ sB,
                                       int ldB, int bColBase, int kOff, int lane) {
    const int col = bColBase + lane_row(lane);
    const int grp = lane_group(lane);
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = sB[col * ldB + (kOff + grp * 4 + i)];
    }
    return v;
}


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


// ===== Helpers (same style as your MLP path) =================================

// Packs two fp32 into one u32 holding 2×bf16 (lo=bf16(a0), hi=bf16(a1))
__device__ inline uint32_t pack2_bf16_bits_f32(float a0, float a1) {
    const __hip_bfloat16 b0 = __float2bfloat16(a0);
    const __hip_bfloat16 b1 = __float2bfloat16(a1);
    return (uint32_t(hipbf16_to_bits(b1)) << 16) | uint32_t(hipbf16_to_bits(b0));
}

// Vectorized A load: read float2 (64b), convert to 2×bf16 in a single u32
// before: ... int m_start, int M_bound, int K, ...
template<bool Aligned, int LD_A>
__device__ inline void copy_A_tile_vec(uint32_t* __restrict__ dst_u32,
                                       const float* __restrict__ A,
                                       int m_start, int M, int K, int kBase,
                                       int linearT, int threadsPerBlock)
{
    constexpr int BM_ = BLOCK_M;
    constexpr int BK_ = BLOCK_K;
    const int pairsPerRow = BK_ >> 1;
    const int totalPairs  = BM_ * pairsPerRow;

    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int r  = t / pairsPerRow;
        const int p  = t % pairsPerRow;
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);

        uint32_t val = 0u;
        if (gm < M && gk < K) {                // <<< fix: compare to M
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

// Vectorized B load: read uint4 (128b = 8×bf16) when aligned & K%8==0, else u32 pairs
template<bool Use128b, int LD_B>
__device__ inline void copy_B_tile_vec(uint32_t* __restrict__ dst_u32,
                                       const __hip_bfloat16* __restrict__ W_e, // [N,K] row-major
                                       int n0, int N, int K, int kBase,
                                       int linearT, int threadsPerBlock)
{
    constexpr int BN_ = BLOCK_N;
    constexpr int BK_ = BLOCK_K;

    if constexpr (Use128b) {
        const int quadPerCol = BK_ / 8;              // 8 bf16 per 128b
        const int totalQuads = BN_ * quadPerCol;
        for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
            const int c   = t / quadPerCol;          // column in this BN tile
            const int q   = t % quadPerCol;          // which 8-elem chunk along K
            const int gn  = n0 + c;
            const int gk8 = kBase + (q << 3);

            uint4 v = {0,0,0,0};
            if (gn < N && (gk8 + 7) < K) {
                const uint4* src = reinterpret_cast<const uint4*>(&W_e[(size_t)gn * K + gk8]);
                v = *src; // 128b global load
            } else {
                __hip_bfloat16 tmp[8] = {};
                for (int i=0;i<8 && (gk8+i)<K && gn<N;i++) tmp[i] = W_e[(size_t)gn*K + gk8 + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0],p[1],p[2],p[3]);
            }
            uint32_t* col = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1));
            const int off = q << 2;                   // 4×u32 per 128b
            col[off + 0] = v.x;
            col[off + 1] = v.y;
            col[off + 2] = v.z;
            col[off + 3] = v.w;
        }
    } else {
        const int pairsPerCol = BK_ >> 1;
        const int totalPairs  = BN_ * pairsPerCol;
        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int c  = t / pairsPerCol;
            const int p  = t % pairsPerCol;
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);
            uint32_t val = 0u;
            if (gn < N && gk < K) {
                const size_t base = (size_t)gn * K + gk;
                if (gk + 1 < K) val = *reinterpret_cast<const uint32_t*>(&W_e[base]);
                else            val = uint32_t(hipbf16_to_bits(W_e[base]));
            }
            reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = val;
        }
    }
}

// ===== Optional tiny LDS padding to cut bank conflicts =====
#ifndef PAD_K_MC
#define PAD_K_MC 8   // try 2 or 8 if you observe LDS conflicts; must be even
#endif
static_assert((BLOCK_K % 2) == 0, "BLOCK_K must be even (packs 2×bf16).");
static_assert((PAD_K_MC % 2) == 0, "PAD_K_MC must be even (u32 pair addressing).");

// ===== FP32 × BF16 GEMM using MFMA (gfx90a) ==================================
// A: float [M,K] (row-major)  → converted to bf16 in LDS
// B: bf16  [N,K] (row-major)  → copied to LDS as column-major [K, BN]
// C: float [M,N]
__global__ __launch_bounds__(LANE_PER_WAVE * WAVES_PER_BLOCK, 2)
void gemm_mfma_f32xbf16_kernel_opt(
    float* __restrict__ C,
    const float* __restrict__ A,
    const __hip_bfloat16* __restrict__ Wbf16,
    int M, int K, int N)
{
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x;               // 0..63
    const int wave   = threadIdx.y;               // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // LDS ping–pong:
    // sA0,sA1: [BLOCK_M x (BLOCK_K+pad)] row-major (bf16 bits)
    // sB0,sB1: [(BLOCK_K+pad) x BLOCK_N] col-major (bf16 bits)
    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K + PAD_K_MC;          // leading dim in bf16 elems
    const int ldB = BLOCK_K + PAD_K_MC;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m0 + BLOCK_M;

    // Alignment guards (decide paths once per block)
    const bool alignedA  = (((uintptr_t)A     & 0x7)==0) && ((K & 1)==0); // 8B & even
    const bool use128bW  = (((uintptr_t)Wbf16 & 0xF)==0) && ((K & 7)==0); // 16B & K%8==0

    // Preload kBase = 0
    if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(sA0_u32, A, m0, M, K, 0, linearT, threadsPerBlock);

    if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);
    else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(sB0_u32, Wbf16, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    // Ping–pong pointers
    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    // Per-wave MFMA bases
    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    // Split K into main slabs of BLOCK_K and one possible tail
    const int Kmain = (K / BLOCK_K) * BLOCK_K;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BLOCK_K) {
        const int kNext = k0 + BLOCK_K;

        // Prefetch next main slab
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(nextA32, A, m0, M, K, kNext, linearT, threadsPerBlock);

            if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
            else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        // Consume current tiles
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }

        __syncthreads();
        if (kNext < Kmain) {
            // swap
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    // Tail slab (0 < K - Kmain < BLOCK_K)
    // Tail slab (0 < K - Kmain < BLOCK_K)
    if (has_tail) {
        // load the tail into next*
        if (alignedA) copy_A_tile_vec</*Aligned*/true,  /*LD_A=*/ldA>(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec</*Aligned*/false, /*LD_A=*/ldA>(nextA32, A, m0, M, K, Kmain, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec</*Use128b*/true,  /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);
        else          copy_B_tile_vec</*Use128b*/false, /*LD_B=*/ldB>(nextB32, Wbf16, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

        // <<< fix: consume the tail you just loaded
        currA = nextA;
        currB = nextB;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }


    // Stores
    const bool interior = (m0 + BLOCK_M) <= M && (n0 + BLOCK_N) <= N;
    if (interior) store_c_tile_mfma<true >(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
    else          store_c_tile_mfma<false>(C, acc, M, N, m0, n0, wave_m, wave_n, lane);
}

// ===== Host wrapper: FP32 × BF16 ==================================================
inline void matmul_mc(
    float* __restrict__ output,                    // [B, O]
    const float* __restrict__ input,               // [B, I] (fp32)
    const __hip_bfloat16* __restrict__ weight,     // [O, I] (bf16 row-major)
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    const int ldA = BLOCK_K + PAD_K_MC;
    const int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(2 * BLOCK_M * ldA + 2 * ldB * BLOCK_N);

    hipLaunchKernelGGL(
        gemm_mfma_f32xbf16_kernel_opt,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
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