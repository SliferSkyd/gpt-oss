#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// FP8 E4M3 format:
// Sign: 1 bit
// Exponent: 4 bits (bias = 7)
// Mantissa: 3 bits
// Range: approximately -448 to 448
// Special values: NaN when all bits are 1

namespace fp8_e4m3 {

// Constants for E4M3 format
constexpr int FP8_E4M3_EXPONENT_BITS = 4;
constexpr int FP8_E4M3_MANTISSA_BITS = 3;
constexpr int FP8_E4M3_EXPONENT_BIAS = 7;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float FP8_E4M3_MIN_VALUE = -448.0f;
constexpr uint8_t FP8_E4M3_NAN = 0xFF;

// Helper functions for bit conversion
__host__ __device__ inline uint32_t float_as_uint(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    return u.i;
}

__host__ __device__ inline float uint_as_float(uint32_t i) {
    union { float f; uint32_t i; } u;
    u.i = i;
    return u.f;
}

// Host-side conversion functions
__host__ __device__ inline uint8_t float_to_fp8_e4m3(float value, bool saturate = true) {
    // Handle special cases
    if (isnan(value)) {
        return FP8_E4M3_NAN;
    }
    
    if (saturate) {
        value = fminf(fmaxf(value, FP8_E4M3_MIN_VALUE), FP8_E4M3_MAX_VALUE);
    }
    
    // Extract components from float
    uint32_t bits = float_as_uint(value);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Remove IEEE bias
    uint32_t mantissa = bits & 0x7FFFFF;
    
    // Handle zero
    if (exponent == -127 && mantissa == 0) {
        return sign << 7;  // Signed zero
    }
    
    // Add FP8 bias
    exponent += FP8_E4M3_EXPONENT_BIAS;
    
    // Handle underflow
    if (exponent <= 0) {
        // Denormalized or zero
        return sign << 7;
    }
    
    // Handle overflow
    if (exponent >= 15) {
        if (saturate) {
            // Return max value with appropriate sign
            return (sign << 7) | 0x7E;  // Max normal value
        } else {
            // Return infinity (represented as max exponent with zero mantissa)
            return (sign << 7) | 0x78;
        }
    }
    
    // Normal case: truncate mantissa to 3 bits
    uint8_t fp8_mantissa = (mantissa >> 20) & 0x7;
    uint8_t fp8_exponent = exponent & 0xF;
    
    return (sign << 7) | (fp8_exponent << 3) | fp8_mantissa;
}

__host__ __device__ inline float fp8_e4m3_to_float(uint8_t value) {
    // Handle NaN
    if (value == FP8_E4M3_NAN) {
        return nanf("");
    }
    
    uint32_t sign = (value >> 7) & 0x1;
    uint32_t exponent = (value >> 3) & 0xF;
    uint32_t mantissa = value & 0x7;
    
    // Handle zero
    if (exponent == 0 && mantissa == 0) {
        return sign ? -0.0f : 0.0f;
    }
    
    // Convert to IEEE float
    int32_t ieee_exponent = exponent - FP8_E4M3_EXPONENT_BIAS + 127;
    
    // Handle denormalized numbers
    if (exponent == 0) {
        // Find leading one in mantissa
        int shift = 0;
        uint32_t temp = mantissa;
        while (temp && !(temp & 0x4)) {
            temp <<= 1;
            shift++;
        }
        ieee_exponent -= shift;
        mantissa = (mantissa << (shift + 1)) & 0x7;
    }
    
    // Ensure exponent is in valid range
    if (ieee_exponent <= 0) {
        return sign ? -0.0f : 0.0f;
    }
    if (ieee_exponent >= 255) {
        return sign ? -INFINITY : INFINITY;
    }
    
    // Construct IEEE float
    uint32_t ieee_mantissa = mantissa << 20;  // Shift to IEEE position
    uint32_t ieee_bits = (sign << 31) | (ieee_exponent << 23) | ieee_mantissa;
    
    return uint_as_float(ieee_bits);
}

// BFloat16 conversion functions
__host__ __device__ inline uint8_t bfloat16_to_fp8_e4m3(__hip_bfloat16 value, bool saturate = true) {
    return float_to_fp8_e4m3(__bfloat162float(value), saturate);
}

__host__ __device__ inline __hip_bfloat16 fp8_e4m3_to_bfloat16(uint8_t value) {
    return __float2bfloat16(fp8_e4m3_to_float(value));
}

// Vectorized conversion kernels for KV cache

// Convert FP32 to FP8 during KV cache update
__global__ void convert_fp32_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ fp8_output,
    const float* __restrict__ fp32_input,
    const float scale,
    const size_t num_elements,
    const bool saturate = true
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float value = fp32_input[idx] * scale;
        fp8_output[idx] = float_to_fp8_e4m3(value, saturate);
    }
}

// Convert BF16 to FP8 during KV cache update
__global__ void convert_bf16_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ fp8_output,
    const __hip_bfloat16* __restrict__ bf16_input,
    const float scale,
    const size_t num_elements,
    const bool saturate = true
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float value = __bfloat162float(bf16_input[idx]) * scale;
        fp8_output[idx] = float_to_fp8_e4m3(value, saturate);
    }
}

// Convert FP8 to FP32 during attention computation
__global__ void convert_fp8_e4m3_to_fp32_kernel(
    float* __restrict__ fp32_output,
    const uint8_t* __restrict__ fp8_input,
    const float inv_scale,
    const size_t num_elements
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        fp32_output[idx] = fp8_e4m3_to_float(fp8_input[idx]) * inv_scale;
    }
}

// Convert FP8 to BF16 during attention computation
__global__ void convert_fp8_e4m3_to_bf16_kernel(
    __hip_bfloat16* __restrict__ bf16_output,
    const uint8_t* __restrict__ fp8_input,
    const float inv_scale,
    const size_t num_elements
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float value = fp8_e4m3_to_float(fp8_input[idx]) * inv_scale;
        bf16_output[idx] = __float2bfloat16(value);
    }
}

// Vectorized load/store for better performance (load 4 FP8 values as uint32_t)
__device__ inline void load_fp8_e4m3_vec4(
    float4& output,
    const uint8_t* input,
    const float inv_scale
) {
    uint32_t packed = *reinterpret_cast<const uint32_t*>(input);
    
    output.x = fp8_e4m3_to_float((packed >> 0) & 0xFF) * inv_scale;
    output.y = fp8_e4m3_to_float((packed >> 8) & 0xFF) * inv_scale;
    output.z = fp8_e4m3_to_float((packed >> 16) & 0xFF) * inv_scale;
    output.w = fp8_e4m3_to_float((packed >> 24) & 0xFF) * inv_scale;
}

__device__ inline void store_fp8_e4m3_vec4(
    uint8_t* output,
    const float4& input,
    const float scale,
    const bool saturate = true
) {
    uint32_t packed = 0;
    packed |= float_to_fp8_e4m3(input.x * scale, saturate) << 0;
    packed |= float_to_fp8_e4m3(input.y * scale, saturate) << 8;
    packed |= float_to_fp8_e4m3(input.z * scale, saturate) << 16;
    packed |= float_to_fp8_e4m3(input.w * scale, saturate) << 24;
    
    *reinterpret_cast<uint32_t*>(output) = packed;
}

// Compute optimal scale for a tensor (for per-tensor quantization)
__global__ void compute_fp8_scale_kernel(
    float* scale_output,
    const float* input,
    const size_t num_elements
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    // Find local max absolute value
    float local_max = 0.0f;
    if (idx < num_elements) {
        local_max = fabsf(input[idx]);
    }
    
    // Reduce within block
    shared_mem[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        // Compute scale to fit in FP8 range
        float max_val = shared_mem[0];
        float scale = (max_val > 0) ? (FP8_E4M3_MAX_VALUE * 0.95f / max_val) : 1.0f;
        atomicMax(scale_output, 1.0f / scale);  // Store inverse for efficiency
    }
}

} // namespace fp8_e4m3