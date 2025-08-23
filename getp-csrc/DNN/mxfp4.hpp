#ifndef MXFP4_HPP
#define MXFP4_HPP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// MXFP4 format constants
constexpr int MXFP4_BLOCK_SIZE = 32;  // 32 FP4 values share one scale
constexpr int MXFP4_VALUES_PER_BYTE = 2;  // 2 4-bit values per byte

// MXFP4 lookup table - 16 possible FP4 values
// Based on OpenAI GPT-OSS implementation
__device__ __constant__ float MXFP4_LUT[16] = {
    0.0f,    // 0b0000
    0.5f,    // 0b0001
    1.0f,    // 0b0010
    1.5f,    // 0b0011
    2.0f,    // 0b0100
    3.0f,    // 0b0101
    4.0f,    // 0b0110
    6.0f,    // 0b0111
    -0.0f,   // 0b1000 (negative zero, treated as zero)
    -0.5f,   // 0b1001
    -1.0f,   // 0b1010
    -1.5f,   // 0b1011
    -2.0f,   // 0b1100
    -3.0f,   // 0b1101
    -4.0f,   // 0b1110
    -6.0f    // 0b1111
};

// CPU-side lookup table for quantization
const float MXFP4_LUT_CPU[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Structure to hold MXFP4 quantized weights
struct MXFP4Weights {
    uint8_t* packed_values;  // Packed 4-bit indices (2 per byte)
    float* scales;           // Scale factor for each block
    size_t num_elements;     // Total number of elements
    size_t num_blocks;       // Number of MXFP4 blocks
};

// Find closest FP4 value index
inline uint8_t find_closest_fp4_index(float value) {
    float min_diff = std::abs(value - MXFP4_LUT_CPU[0]);
    uint8_t best_idx = 0;
    
    for (uint8_t i = 1; i < 16; i++) {
        float diff = std::abs(value - MXFP4_LUT_CPU[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

// CPU function to quantize FP32 to MXFP4
inline void quantize_to_mxfp4(const float* input, MXFP4Weights& output, size_t count) {
    output.num_elements = count;
    output.num_blocks = (count + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    
    // Allocate memory for packed values and scales
    size_t packed_size = (count + 1) / 2;  // 2 values per byte
    output.packed_values = new uint8_t[packed_size];
    output.scales = new float[output.num_blocks];
    
    // Process each block
    for (size_t block_idx = 0; block_idx < output.num_blocks; block_idx++) {
        size_t block_start = block_idx * MXFP4_BLOCK_SIZE;
        size_t block_end = std::min(block_start + MXFP4_BLOCK_SIZE, count);
        
        // Find max absolute value in block for scaling
        float max_abs = 0.0f;
        for (size_t i = block_start; i < block_end; i++) {
            max_abs = std::max(max_abs, std::abs(input[i]));
        }
        
        // Calculate scale (avoid division by zero)
        float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
        output.scales[block_idx] = scale;
        
        // Quantize values in block
        for (size_t i = block_start; i < block_end; i++) {
            float scaled_value = (scale > 0.0f) ? input[i] / scale : 0.0f;
            uint8_t idx = find_closest_fp4_index(scaled_value);
            
            // Pack two 4-bit values into one byte
            size_t byte_idx = i / 2;
            if (i % 2 == 0) {
                // Store in lower 4 bits
                output.packed_values[byte_idx] = (output.packed_values[byte_idx] & 0xF0) | (idx & 0x0F);
            } else {
                // Store in upper 4 bits
                output.packed_values[byte_idx] = (output.packed_values[byte_idx] & 0x0F) | ((idx << 4) & 0xF0);
            }
        }
    }
}

// GPU function to dequantize MXFP4 to FP32
__device__ __forceinline__ float dequantize_mxfp4(
    const uint8_t* packed_values,
    const float* scales,
    size_t element_idx,
    size_t total_elements
) {
    if (element_idx >= total_elements) return 0.0f;
    
    // Calculate block index and scale
    size_t block_idx = element_idx / MXFP4_BLOCK_SIZE;
    float scale = scales[block_idx];
    
    // Extract 4-bit index from packed byte
    size_t byte_idx = element_idx / 2;
    uint8_t packed_byte = packed_values[byte_idx];
    uint8_t fp4_idx;
    
    if (element_idx % 2 == 0) {
        // Extract from lower 4 bits
        fp4_idx = packed_byte & 0x0F;
    } else {
        // Extract from upper 4 bits
        fp4_idx = (packed_byte >> 4) & 0x0F;
    }
    
    // Lookup FP4 value and apply scale
    return MXFP4_LUT[fp4_idx] * scale;
}

// Helper function to calculate MXFP4 storage size
inline size_t calculate_mxfp4_size(size_t num_elements) {
    size_t packed_size = (num_elements + 1) / 2;  // 2 values per byte
    size_t num_blocks = (num_elements + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;
    size_t scales_size = num_blocks * sizeof(float);
    return packed_size + scales_size;
}

// Cleanup function
inline void free_mxfp4_weights(MXFP4Weights& weights) {
    if (weights.packed_values) {
        delete[] weights.packed_values;
        weights.packed_values = nullptr;
    }
    if (weights.scales) {
        delete[] weights.scales;
        weights.scales = nullptr;
    }
    weights.num_elements = 0;
    weights.num_blocks = 0;
}

#endif // MXFP4_HPP