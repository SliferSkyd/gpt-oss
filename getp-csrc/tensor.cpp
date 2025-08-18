/**
 * Tensor Library for HIP (Heterogeneous-Compute Interface for Portability)
 * 
 * This library provides a comprehensive tensor system with support for:
 * - CPU and GPU computation using HIP
 * - Multiple data types (float32, float16, bfloat16, int32)
 * - Memory management and device transfers
 * - Type conversions and tensor operations
 */

#ifndef TENSOR_HIP
#define TENSOR_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

/**
 * Enumeration of supported tensor data types
 */
typedef enum {
    TENSOR_DTYPE_FLOAT32 = 0,    // 32-bit floating point
    TENSOR_DTYPE_FLOAT16 = 1,    // 16-bit floating point (half precision)
    TENSOR_DTYPE_BFLOAT16 = 2,   // Brain floating point 16-bit
    TENSOR_DTYPE_INT32 = 3       // 32-bit integer
} TensorDType;

/**
 * Enumeration of supported tensor devices
 */
typedef enum {
    TENSOR_DEVICE_CPU = 0,       // CPU device
    TENSOR_DEVICE_GPU = 1        // GPU device (HIP)
} TensorDevice;

/**
 * Main tensor structure
 * 
 * Contains all necessary information to manage a multi-dimensional array
 * that can reside on CPU or GPU memory.
 */
typedef struct {
    void* data;              // Pointer to the actual tensor data
    size_t* shape;           // Array containing the size of each dimension
    size_t* strides;         // Array containing stride for each dimension (for memory layout)
    size_t ndim;             // Number of dimensions
    size_t size;             // Total number of elements in the tensor
    size_t element_size;     // Size of each element in bytes
    TensorDType dtype;       // Data type of tensor elements
    TensorDevice device;     // Device where tensor data is stored
    int device_id;           // ID of the device (for multi-GPU systems)
    bool owns_data;          // Whether this tensor owns the data pointer (for memory management)
} Tensor;

/**
 * Macro for checking HIP errors
 * 
 * This macro wraps HIP API calls and automatically checks for errors.
 * If an error occurs, it prints detailed error information and exits.
 */
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error %d at %s:%d - %s\n", \
                    error, __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Brain floating point 16-bit representation
 * 
 * bfloat16 is a 16-bit floating point format commonly used in machine learning.
 * It has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
 */
typedef union {
    uint16_t bits;           // Raw 16-bit representation
    struct {
        uint16_t mantissa : 7;   // 7-bit mantissa (fractional part)
        uint16_t exponent : 8;   // 8-bit exponent
        uint16_t sign : 1;       // 1-bit sign
    };
} bfloat16_t;

/**
 * Get the size in bytes for a given data type
 * 
 * @param dtype The tensor data type
 * @return Size in bytes of the data type
 */
size_t dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT32: return sizeof(float);      // 4 bytes
        case TENSOR_DTYPE_FLOAT16: return sizeof(uint16_t);   // 2 bytes
        case TENSOR_DTYPE_BFLOAT16: return sizeof(uint16_t);  // 2 bytes
        case TENSOR_DTYPE_INT32: return sizeof(int);          // 4 bytes
        default: return sizeof(float);                        // Default to float32
    }
}

/**
 * Convert bfloat16 to float32
 * 
 * @param bf16 The bfloat16 value to convert
 * @return The equivalent float32 value
 */
float bfloat16_to_float(bfloat16_t bf16) {
    union {
        float f;
        uint32_t i;
    } u;
    // bfloat16 to float32: shift left by 16 bits to expand mantissa
    u.i = ((uint32_t)bf16.bits) << 16;
    return u.f;
}

/**
 * Convert float32 to bfloat16
 * 
 * @param f The float32 value to convert
 * @return The equivalent bfloat16 value
 */
bfloat16_t float_to_bfloat16(float f) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = f;
    bfloat16_t result;
    
    // Round to nearest even for better numerical stability
    uint32_t rounding_bias = 0x7FFF + ((u.i >> 16) & 1);
    result.bits = (u.i + rounding_bias) >> 16;
    return result;
}

/**
 * Calculate the total number of elements in a tensor
 * 
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 * @return Total number of elements
 */
size_t compute_size(const size_t* shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }
    return total;
}

/**
 * Compute strides for a tensor with given shape
 * 
 * Strides determine how to navigate through the tensor in memory.
 * For a tensor with shape [A, B, C], strides will be [B*C, C, 1]
 * 
 * @param strides Output array to store computed strides
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 */
void compute_strides(size_t* strides, const size_t* shape, size_t ndim) {
    if (ndim == 0) return;
    
    // Start from the last dimension with stride 1
    strides[ndim - 1] = 1;
    // Work backwards, each stride is the product of all following dimensions
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/**
 * Create a new tensor with specified shape, data type, and device
 * 
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Target device (CPU or GPU)
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_create(const size_t* shape, size_t ndim, TensorDType dtype, TensorDevice device) {
    // Allocate memory for tensor structure
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    // Initialize tensor properties
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->device_id = 0;
    tensor->element_size = dtype_size(dtype);
    tensor->owns_data = true;  // This tensor owns its data
    
    // Allocate memory for shape and strides arrays
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and compute strides and total size
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    compute_strides(tensor->strides, tensor->shape, ndim);
    tensor->size = compute_size(shape, ndim);
    
    size_t bytes = tensor->size * tensor->element_size;
    
    // Allocate data on appropriate device
    if (device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMalloc(&tensor->data, bytes));
        HIP_CHECK(hipMemset(tensor->data, 0, bytes));  // Initialize to zero
    } else {
        tensor->data = calloc(tensor->size, tensor->element_size);  // CPU allocation with zero init
        if (!tensor->data) {
            free(tensor->shape);
            free(tensor->strides);
            free(tensor);
            return NULL;
        }
    }
    
    return tensor;
}

/**
 * Create a tensor filled with zeros
 * 
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 * @param dtype Data type of tensor elements
 * @param device Target device (CPU or GPU)
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_zeros(const size_t* shape, size_t ndim, TensorDType dtype, TensorDevice device) {
    return tensor_create(shape, ndim, dtype, device);  // tensor_create already initializes to zero
}

/**
 * Create a tensor from existing float data (for loading weights)
 * 
 * @param data Pointer to existing float data
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 * @param dtype Target data type for the tensor
 * @param device Target device (CPU or GPU)
 * @param copy_data Whether to copy the data (true) or just reference it (false)
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_from_float_data(const float* data, const size_t* shape, size_t ndim, 
                               TensorDType dtype, TensorDevice device, bool copy_data) {
    if (!data || !shape) return NULL;
    
    // Allocate memory for tensor structure
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    // Initialize tensor properties
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->device_id = 0;
    tensor->element_size = dtype_size(dtype);
    tensor->owns_data = copy_data;  // Only own data if we're copying it
    
    // Allocate memory for shape and strides arrays
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and compute strides and total size
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    compute_strides(tensor->strides, tensor->shape, ndim);
    tensor->size = compute_size(shape, ndim);
    
    size_t bytes = tensor->size * tensor->element_size;
    
    if (copy_data) {
        // Allocate new memory and copy data
        if (device == TENSOR_DEVICE_GPU) {
            HIP_CHECK(hipMalloc(&tensor->data, bytes));
            
            if (dtype == TENSOR_DTYPE_FLOAT32) {
                // Direct copy for float32
                HIP_CHECK(hipMemcpy(tensor->data, data, tensor->size * sizeof(float), hipMemcpyHostToDevice));
            } else if (dtype == TENSOR_DTYPE_BFLOAT16) {
                // Convert float32 to bfloat16 on GPU
                float* temp_gpu_f32;
                HIP_CHECK(hipMalloc(&temp_gpu_f32, tensor->size * sizeof(float)));
                HIP_CHECK(hipMemcpy(temp_gpu_f32, data, tensor->size * sizeof(float), hipMemcpyHostToDevice));
                
                size_t threads_per_block = 256;
                size_t blocks = (tensor->size + threads_per_block - 1) / threads_per_block;
                hipLaunchKernelGGL(convert_f32_to_bf16_kernel,
                                  dim3(blocks), dim3(threads_per_block), 0, 0,
                                  (uint16_t*)tensor->data, temp_gpu_f32, tensor->size);
                HIP_CHECK(hipDeviceSynchronize());
                
                hipFree(temp_gpu_f32);
            } else {
                // Unsupported dtype for GPU
                hipFree(tensor->data);
                free(tensor->shape);
                free(tensor->strides);
                free(tensor);
                return NULL;
            }
        } else {
            // CPU allocation
            tensor->data = malloc(bytes);
            if (!tensor->data) {
                free(tensor->shape);
                free(tensor->strides);
                free(tensor);
                return NULL;
            }
            
            if (dtype == TENSOR_DTYPE_FLOAT32) {
                // Direct copy for float32
                memcpy(tensor->data, data, tensor->size * sizeof(float));
            } else if (dtype == TENSOR_DTYPE_BFLOAT16) {
                // Convert float32 to bfloat16 on CPU
                uint16_t* dst = (uint16_t*)tensor->data;
                for (size_t i = 0; i < tensor->size; i++) {
                    bfloat16_t bf16 = float_to_bfloat16(data[i]);
                    dst[i] = bf16.bits;
                }
            } else {
                // Unsupported dtype for CPU
                free(tensor->data);
                free(tensor->shape);
                free(tensor->strides);
                free(tensor);
                return NULL;
            }
        }
    } else {
        // Just reference the data (only works for CPU float32)
        if (device == TENSOR_DEVICE_GPU || dtype != TENSOR_DTYPE_FLOAT32) {
            // Cannot reference data on GPU or with different dtype
            free(tensor->shape);
            free(tensor->strides);
            free(tensor);
            return NULL;
        }
        tensor->data = (void*)data;  // Just point to existing data
    }
    
    return tensor;
}

/**
 * Create a tensor from existing float data with default settings
 * 
 * This is a convenience function that creates a float32 tensor on CPU
 * by copying the provided data.
 * 
 * @param data Pointer to existing float data
 * @param shape Array containing the size of each dimension
 * @param ndim Number of dimensions
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_from_floats(const float* data, const size_t* shape, size_t ndim) {
    return tensor_from_float_data(data, shape, ndim, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU, true);
}

/**
 * Create a 1D tensor from existing float array
 * 
 * @param data Pointer to existing float data
 * @param size Number of elements in the array
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_from_float_array(const float* data, size_t size) {
    size_t shape[1] = {size};
    return tensor_from_float_data(data, shape, 1, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU, true);
}

/**
 * Create a 2D tensor from existing float data (for matrices/weights)
 * 
 * @param data Pointer to existing float data (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the created tensor, or NULL on failure
 */
Tensor* tensor_from_float_matrix(const float* data, size_t rows, size_t cols) {
    size_t shape[2] = {rows, cols};
    return tensor_from_float_data(data, shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU, true);
}

/**
 * Free memory allocated for a tensor
 * 
 * @param tensor Pointer to the tensor to free
 */
void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    
    // Free data only if this tensor owns it
    if (tensor->owns_data && tensor->data) {
        if (tensor->device == TENSOR_DEVICE_GPU) {
            hipFree(tensor->data);  // Free GPU memory
        } else {
            free(tensor->data);     // Free CPU memory
        }
    }
    
    // Free shape and strides arrays
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

/**
 * Transfer tensor data to a different device
 * 
 * @param tensor The tensor to transfer
 * @param target_device The target device (CPU or GPU)
 * @return 0 on success, -1 on failure
 */
int tensor_to_device(Tensor* tensor, TensorDevice target_device) {
    // No operation needed if already on target device
    if (tensor->device == target_device) return 0;
    
    size_t bytes = tensor->size * tensor->element_size;
    void* new_data;
    
    if (target_device == TENSOR_DEVICE_GPU) {
        // Transfer to GPU
        HIP_CHECK(hipMalloc(&new_data, bytes));
        if (tensor->device == TENSOR_DEVICE_CPU) {
            // CPU to GPU transfer
            HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyHostToDevice));
        } else {
            // GPU to GPU transfer (different devices)
            HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyDeviceToDevice));
        }
        
        // Free old data if owned
        if (tensor->owns_data) {
            if (tensor->device == TENSOR_DEVICE_CPU) {
                free(tensor->data);
            } else {
                hipFree(tensor->data);
            }
        }
    } else {
        // Transfer to CPU
        new_data = malloc(bytes);
        if (!new_data) return -1;
        
        // GPU to CPU transfer
        HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyDeviceToHost));
        
        // Free old GPU data if owned
        if (tensor->owns_data) {
            hipFree(tensor->data);
        }
    }
    
    // Update tensor properties
    tensor->data = new_data;
    tensor->device = target_device;
    tensor->owns_data = true;
    
    return 0;
}

/**
 * Copy data from source tensor to destination tensor
 * 
 * Both tensors must have the same size and element size.
 * Handles all combinations of CPU/GPU transfers.
 * 
 * @param dst Destination tensor
 * @param src Source tensor
 */
void tensor_copy_data(Tensor* dst, const Tensor* src) {
    assert(dst->size == src->size);              // Same number of elements
    assert(dst->element_size == src->element_size);  // Same element size
    
    size_t bytes = src->size * src->element_size;
    
    // Handle all combinations of device transfers
    if (src->device == TENSOR_DEVICE_CPU && dst->device == TENSOR_DEVICE_CPU) {
        // CPU to CPU: use standard memcpy
        memcpy(dst->data, src->data, bytes);
    } else if (src->device == TENSOR_DEVICE_GPU && dst->device == TENSOR_DEVICE_GPU) {
        // GPU to GPU: device-to-device copy
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyDeviceToDevice));
    } else if (src->device == TENSOR_DEVICE_CPU && dst->device == TENSOR_DEVICE_GPU) {
        // CPU to GPU: host-to-device copy
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyHostToDevice));
    } else {
        // GPU to CPU: device-to-host copy
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyDeviceToHost));
    }
}

// ==================== GPU KERNELS ====================

/**
 * GPU kernel to fill float32 tensor with a specific value
 * 
 * @param data Pointer to tensor data on GPU
 * @param value Value to fill the tensor with
 * @param size Total number of elements
 */
__global__ void fill_kernel_f32(float* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

/**
 * GPU kernel to fill bfloat16 tensor with a specific value
 * 
 * @param data Pointer to tensor data on GPU
 * @param value Value to fill the tensor with (as uint16_t bits)
 * @param size Total number of elements
 */
__global__ void fill_kernel_bf16(uint16_t* data, uint16_t value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

/**
 * GPU kernel to convert float32 to bfloat16
 * 
 * @param dst Destination buffer (bfloat16 as uint16_t)
 * @param src Source buffer (float32)
 * @param size Number of elements to convert
 */
__global__ void convert_f32_to_bf16_kernel(uint16_t* dst, float* src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        union {
            float f;
            uint32_t i;
        } u;
        u.f = src[idx];
        // Round to nearest even for better numerical stability
        uint32_t rounding_bias = 0x7FFF + ((u.i >> 16) & 1);
        dst[idx] = (u.i + rounding_bias) >> 16;
    }
}

/**
 * GPU kernel to convert bfloat16 to float32
 * 
 * @param dst Destination buffer (float32)
 * @param src Source buffer (bfloat16 as uint16_t)
 * @param size Number of elements to convert
 */
__global__ void convert_bf16_to_f32_kernel(float* dst, uint16_t* src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        union {
            float f;
            uint32_t i;
        } u;
        // Convert bfloat16 to float32 by shifting left 16 bits
        u.i = ((uint32_t)src[idx]) << 16;
        dst[idx] = u.f;
    }
}

/**
 * Fill tensor with a specific value
 * 
 * This function handles both CPU and GPU tensors, using appropriate
 * methods for each device and data type.
 * 
 * @param tensor The tensor to fill
 * @param value The value to fill the tensor with
 */
void tensor_fill(Tensor* tensor, float value) {
    size_t threads_per_block = 256;
    size_t blocks = (tensor->size + threads_per_block - 1) / threads_per_block;
    
    if (tensor->device == TENSOR_DEVICE_GPU) {
        // GPU implementation using kernels
        switch (tensor->dtype) {
            case TENSOR_DTYPE_FLOAT32:
                hipLaunchKernelGGL(fill_kernel_f32, 
                                  dim3(blocks), dim3(threads_per_block), 0, 0,
                                  (float*)tensor->data, value, tensor->size);
                break;
            case TENSOR_DTYPE_BFLOAT16: {
                bfloat16_t bf16_value = float_to_bfloat16(value);
                hipLaunchKernelGGL(fill_kernel_bf16, 
                                  dim3(blocks), dim3(threads_per_block), 0, 0,
                                  (uint16_t*)tensor->data, bf16_value.bits, tensor->size);
                break;
            }
            default:
                break;
        }
        HIP_CHECK(hipDeviceSynchronize());  // Wait for kernel completion
    } else if (tensor->device == TENSOR_DEVICE_CPU) {
        // CPU implementation using loops
        switch (tensor->dtype) {
            case TENSOR_DTYPE_FLOAT32: {
                float* data = (float*)tensor->data;
                for (size_t i = 0; i < tensor->size; i++) {
                    data[i] = value;
                }
                break;
            }
            case TENSOR_DTYPE_BFLOAT16: {
                uint16_t* data = (uint16_t*)tensor->data;
                bfloat16_t bf16_value = float_to_bfloat16(value);
                for (size_t i = 0; i < tensor->size; i++) {
                    data[i] = bf16_value.bits;
                }
                break;
            }
            default:
                break;
        }
    }
}

/**
 * Create a view of an existing tensor with different shape
 * 
 * A view shares the same data as the original tensor but has a different shape.
 * The total number of elements must remain the same.
 * 
 * @param tensor The original tensor
 * @param new_shape Array containing the new shape
 * @param new_ndim Number of dimensions in the new shape
 * @return Pointer to the new tensor view, or NULL on failure
 */
Tensor* tensor_view(Tensor* tensor, const size_t* new_shape, size_t new_ndim) {
    size_t new_size = compute_size(new_shape, new_ndim);
    if (new_size != tensor->size) return NULL;  // Size must match
    
    // Allocate memory for the view tensor
    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) return NULL;
    
    // Copy all properties from original tensor
    *view = *tensor;
    view->owns_data = false;  // View doesn't own the data
    
    // Allocate new shape and strides arrays
    view->shape = (size_t*)malloc(new_ndim * sizeof(size_t));
    view->strides = (size_t*)malloc(new_ndim * sizeof(size_t));
    if (!view->shape || !view->strides) {
        free(view->shape);
        free(view->strides);
        free(view);
        return NULL;
    }
    
    // Set up new shape and compute corresponding strides
    memcpy(view->shape, new_shape, new_ndim * sizeof(size_t));
    compute_strides(view->strides, view->shape, new_ndim);
    view->ndim = new_ndim;
    
    return view;
}

// ==================== TENSOR POOL ====================

/**
 * Tensor pool structure for managing multiple tensors
 * 
 * Provides efficient memory management by grouping related tensors
 * and tracking total memory usage.
 */
typedef struct {
    Tensor** tensors;        // Array of tensor pointers
    size_t capacity;         // Maximum number of tensors the pool can hold
    size_t count;            // Current number of tensors in the pool
    size_t total_memory;     // Total memory used by all tensors in bytes
} TensorPool;

/**
 * Create a new tensor pool
 * 
 * @param initial_capacity Initial capacity of the pool
 * @return Pointer to the created tensor pool, or NULL on failure
 */
TensorPool* tensor_pool_create(size_t initial_capacity) {
    TensorPool* pool = (TensorPool*)malloc(sizeof(TensorPool));
    if (!pool) return NULL;
    
    pool->tensors = (Tensor**)malloc(initial_capacity * sizeof(Tensor*));
    if (!pool->tensors) {
        free(pool);
        return NULL;
    }
    
    pool->capacity = initial_capacity;
    pool->count = 0;
    pool->total_memory = 0;
    
    return pool;
}

/**
 * Add a tensor to the pool
 * 
 * The pool will automatically resize if it reaches capacity.
 * 
 * @param pool The tensor pool
 * @param tensor The tensor to add
 */
void tensor_pool_add(TensorPool* pool, Tensor* tensor) {
    // Resize pool if necessary
    if (pool->count >= pool->capacity) {
        pool->capacity *= 2;
        pool->tensors = (Tensor**)realloc(pool->tensors, pool->capacity * sizeof(Tensor*));
    }
    
    // Add tensor and update statistics
    pool->tensors[pool->count++] = tensor;
    pool->total_memory += tensor->size * tensor->element_size;
}

/**
 * Free all tensors in the pool and the pool itself
 * 
 * @param pool The tensor pool to free
 */
void tensor_pool_free(TensorPool* pool) {
    if (!pool) return;
    
    // Free all tensors in the pool
    for (size_t i = 0; i < pool->count; i++) {
        tensor_free(pool->tensors[i]);
    }
    
    // Free the pool structure
    free(pool->tensors);
    free(pool);
}

/**
 * Clear GPU cache and synchronize device
 * 
 * Useful for ensuring all GPU operations are complete before
 * measuring memory usage or performance.
 */
void tensor_pool_clear_gpu_cache() {
    HIP_CHECK(hipDeviceSynchronize());
}

// ==================== UTILITY FUNCTIONS ====================

/**
 * Calculate memory usage of a tensor
 * 
 * @param tensor The tensor to calculate memory usage for
 * @return Memory usage in bytes
 */
size_t tensor_memory_usage(const Tensor* tensor) {
    return tensor->size * tensor->element_size;
}

/**
 * Print detailed information about a tensor
 * 
 * Displays shape, size, memory usage, device, and data type.
 * 
 * @param tensor The tensor to print information about
 */
void tensor_print_info(const Tensor* tensor) {
    printf("Tensor Info:\n");
    printf("  Shape: [");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("  Size: %zu elements\n", tensor->size);
    printf("  Memory: %zu bytes\n", tensor->size * tensor->element_size);
    printf("  Device: %s\n", tensor->device == TENSOR_DEVICE_GPU ? "GPU" : "CPU");
    printf("  Data Type: %s\n", 
           tensor->dtype == TENSOR_DTYPE_FLOAT32 ? "float32" :
           tensor->dtype == TENSOR_DTYPE_FLOAT16 ? "float16" :
           tensor->dtype == TENSOR_DTYPE_BFLOAT16 ? "bfloat16" : "int32");
}

/**
 * Convert tensor to a different data type
 * 
 * Currently supports conversions between float32 and bfloat16.
 * The conversion is performed in-place, replacing the tensor's data.
 * 
 * @param tensor The tensor to convert
 * @param target_dtype The target data type
 * @return 0 on success, -1 on failure
 */
int tensor_convert_dtype(Tensor* tensor, TensorDType target_dtype) {
    // No conversion needed if already the target type
    if (tensor->dtype == target_dtype) return 0;
    
    size_t target_element_size = dtype_size(target_dtype);
    size_t target_bytes = tensor->size * target_element_size;
    void* new_data;
    
    if (tensor->device == TENSOR_DEVICE_GPU) {
        // GPU conversion using kernels
        HIP_CHECK(hipMalloc(&new_data, target_bytes));
        
        size_t threads_per_block = 256;
        size_t blocks = (tensor->size + threads_per_block - 1) / threads_per_block;
        
        if (tensor->dtype == TENSOR_DTYPE_FLOAT32 && target_dtype == TENSOR_DTYPE_BFLOAT16) {
            // Float32 to bfloat16 conversion
            hipLaunchKernelGGL(convert_f32_to_bf16_kernel,
                              dim3(blocks), dim3(threads_per_block), 0, 0,
                              (uint16_t*)new_data, (float*)tensor->data, tensor->size);
        } else if (tensor->dtype == TENSOR_DTYPE_BFLOAT16 && target_dtype == TENSOR_DTYPE_FLOAT32) {
            // Bfloat16 to float32 conversion
            hipLaunchKernelGGL(convert_bf16_to_f32_kernel,
                              dim3(blocks), dim3(threads_per_block), 0, 0,
                              (float*)new_data, (uint16_t*)tensor->data, tensor->size);
        } else {
            // Unsupported conversion
            hipFree(new_data);
            return -1;
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Free old data if owned
        if (tensor->owns_data) {
            hipFree(tensor->data);
        }
    } else {
        // CPU conversion using loops
        new_data = malloc(target_bytes);
        if (!new_data) return -1;
        
        if (tensor->dtype == TENSOR_DTYPE_FLOAT32 && target_dtype == TENSOR_DTYPE_BFLOAT16) {
            // Float32 to bfloat16 conversion
            float* src = (float*)tensor->data;
            uint16_t* dst = (uint16_t*)new_data;
            for (size_t i = 0; i < tensor->size; i++) {
                bfloat16_t bf16 = float_to_bfloat16(src[i]);
                dst[i] = bf16.bits;
            }
        } else if (tensor->dtype == TENSOR_DTYPE_BFLOAT16 && target_dtype == TENSOR_DTYPE_FLOAT32) {
            // Bfloat16 to float32 conversion
            uint16_t* src = (uint16_t*)tensor->data;
            float* dst = (float*)new_data;
            for (size_t i = 0; i < tensor->size; i++) {
                bfloat16_t bf16;
                bf16.bits = src[i];
                dst[i] = bfloat16_to_float(bf16);
            }
        } else {
            // Unsupported conversion
            free(new_data);
            return -1;
        }
        
        // Free old data if owned
        if (tensor->owns_data) {
            free(tensor->data);
        }
    }
    
    // Update tensor properties
    tensor->data = new_data;
    tensor->dtype = target_dtype;
    tensor->element_size = target_element_size;
    tensor->owns_data = true;
    
    return 0;
}

#endif // TENSOR_HIP