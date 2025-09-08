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
