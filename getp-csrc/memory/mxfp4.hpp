#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

static const float MXFP4_LUT_CPU[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline uint8_t encode_e8m0_from_exp_host(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0)   e = 0;
    if (e > 255) e = 255;
    return static_cast<uint8_t>(e);
}

// nearest LUT index (E2M1)
static inline uint8_t enc_fp4_e2m1_nearest(float x) {
    uint8_t best = 0;
    float best_diff = std::fabs(x - MXFP4_LUT_CPU[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        float d = std::fabs(x - MXFP4_LUT_CPU[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return best;
}

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