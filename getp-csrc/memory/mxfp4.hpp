#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

static const float MXFP4_LUT_CPU[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __constant__ float MXFP4_LUT_DEV[16];

static inline void init_mxfp4_lut_on_device() {
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(MXFP4_LUT_DEV),
                                MXFP4_LUT_CPU, sizeof(MXFP4_LUT_CPU)));
}

// E8M0 scale byte: encode exponent + 127 (no sign)
__device__ __forceinline__ float ldexp_pow2_e8m0(uint8_t e8m0) {
    const int exp_unbiased = (int)e8m0 - 127;
    // ldexpf is supported in device code on HIP and CUDA
    return ldexpf(1.0f, exp_unbiased);
}

// Dequant one element from MXFP4 (block-of-32 with e8m0 scales)
__device__ __forceinline__ float dequantize_mxfp4_block32(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    size_t idx)
{
    const size_t blk_id   = idx >> 5;       // /32
    const uint8_t e8m0    = scales[blk_id]; // (exp + 127)
    const float X         = ldexp_pow2_e8m0(e8m0);

    const size_t byte_id  = idx >> 1;       // /2
    const uint8_t byte    = packed[byte_id];
    const uint8_t nib     = (idx & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);

    return MXFP4_LUT_DEV[nib] * X;
}


// --- mxfp4_gpu.cuh (or append to memory/mxfp4.hpp) --------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

// Reuse MXFP4_LUT_DEV[16] and ldexp_pow2_e8m0() from your header.

__device__ __forceinline__ uint8_t encode_e8m0_from_exp_dev(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0)   e = 0;
    if (e > 255) e = 255;
    return static_cast<uint8_t>(e);
}

__device__ __forceinline__ uint8_t enc_fp4_e2m1_nearest_dev(float x) {
    // 16-entry brute force; small and fast enough.
    uint8_t best = 0;
    float best_diff = fabsf(x - MXFP4_LUT_DEV[0]);
    #pragma unroll
    for (uint8_t i = 1; i < 16; ++i) {
        float d = fabsf(x - MXFP4_LUT_DEV[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return best;
}

// Each "logical warp" (32 lanes) handles one 32-elem block:
// - computes maxabs -> exp_scale = ilog2(maxabs) - 2
// - writes e8m0 scale byte
// - quantizes and packs to nibbles (2 per byte) directly to final dst
//
// Grid: any; BlockDim must be a multiple of 32
// Shared mem layout: [WPB*32 floats] + [WPB*32 uint8] + [WPB int]
template<int BLOCK_THREADS>
__global__ void quantize_pack_mxfp4_block32_kernel(
    const float* __restrict__ src_chunk,   // FP32 chunk on device
    size_t      chunk_blocks,              // number of 32-elem blocks in this chunk
    size_t      global_block0,             // global block-id offset (elem_off/32)
    uint8_t*    __restrict__ dst_packed,   // final destination on device
    uint8_t*    __restrict__ dst_scales,   // final destination on device
    size_t      total_elems                // total original element count
){
    constexpr int LANES = 32;
    static_assert(BLOCK_THREADS % LANES == 0, "BLOCK_THREADS must be multiple of 32");
    const int WPB = BLOCK_THREADS / LANES; // logical warps per block

        extern __shared__ __align__(16) unsigned char mxfp4_smem[];
    float*   svals = reinterpret_cast<float*>(mxfp4_smem);
    uint8_t* snibs = reinterpret_cast<uint8_t*>(svals + WPB * LANES);
    int*     sexps = reinterpret_cast<int*>(snibs + WPB * LANES);

    const int tid = threadIdx.x;
    const int warp_local = tid / LANES;
    const int lane       = tid % LANES;

    // Which 32-elem block this logical warp handles (within the chunk)
    size_t block_in_chunk = (size_t)blockIdx.x * WPB + warp_local;
    if (block_in_chunk >= chunk_blocks) return;

    const size_t elem_base = block_in_chunk * LANES; // elem offset within chunk
    const size_t g_block   = global_block0 + block_in_chunk; // global 32-elem block id
    const size_t g_elem0   = g_block * (size_t)LANES;

    // Load value (guard tail)
    float v = 0.0f;
    const size_t g_elem = g_elem0 + lane;
    if (g_elem < total_elems) {
        v = src_chunk[elem_base + lane];
    }

    // Stash to shared for a simple reduction (avoid warp intrinsics; AMD wavefront is 64)
    svals[warp_local * LANES + lane] = fabsf(v);
    __syncthreads();

    // lane 0 of the logical warp computes maxabs & exp
    if (lane == 0) {
        float maxabs = 0.f;
        #pragma unroll
        for (int i = 0; i < LANES; ++i) {
            float a = svals[warp_local * LANES + i];
            if (a > maxabs) maxabs = a;
        }
        int exp_scale = 0;
        if (maxabs > 0.f) {
            // frexpf: x = m * 2^e, with 0.5 <= m < 1  => ilogb(x) = e-1
            int e;
            frexpf(maxabs, &e);
            exp_scale = (e - 1) - 2; // largest pow2 representable in E2M1 is 2^2
        }
        sexps[warp_local] = exp_scale;
        dst_scales[g_block] = encode_e8m0_from_exp_dev(exp_scale);
    }
    __syncthreads();

    // Broadcast exp -> compute X
    const int exp_scale = sexps[warp_local];
    const float X = ldexpf(1.0f, exp_scale);

    // Quantize to nearest LUT code
    uint8_t nib = enc_fp4_e2m1_nearest_dev(v / X);
    snibs[warp_local * LANES + lane] = nib;
    __syncthreads();

    // Pack two nibbles per byte (even lanes write)
    if ((lane & 1) == 0) {
        const uint8_t lo = snibs[warp_local * LANES + lane];
        const uint8_t hi = (lane + 1 < LANES) ? snibs[warp_local * LANES + lane + 1] : 0;
        const size_t byte_in_block = (size_t)(lane >> 1); // 0..15
        const size_t byte_global   = (size_t)g_block * 16 + byte_in_block;
        dst_packed[byte_global] = (uint8_t)((hi << 4) | (lo & 0xF));
    }
}
