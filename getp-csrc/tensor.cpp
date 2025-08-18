#ifndef TENSOR_HIP
#define TENSOR_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

typedef enum {
    TENSOR_DTYPE_FLOAT32 = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_BFLOAT16 = 2,
    TENSOR_DTYPE_INT32 = 3
} TensorDType;

typedef enum {
    TENSOR_DEVICE_CPU = 0,
    TENSOR_DEVICE_GPU = 1
} TensorDevice;

typedef struct {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    size_t element_size;
    TensorDType dtype;
    TensorDevice device;
    int device_id;
    bool owns_data;
} Tensor;

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error %d at %s:%d - %s\n", \
                    error, __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef union {
    uint16_t bits;
    struct {
        uint16_t mantissa : 7;
        uint16_t exponent : 8;
        uint16_t sign : 1;
    };
} bfloat16_t;

size_t dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_DTYPE_FLOAT32: return sizeof(float);
        case TENSOR_DTYPE_FLOAT16: return sizeof(uint16_t);
        case TENSOR_DTYPE_BFLOAT16: return sizeof(uint16_t);
        case TENSOR_DTYPE_INT32: return sizeof(int);
        default: return sizeof(float);
    }
}

float bfloat16_to_float(bfloat16_t bf16) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = ((uint32_t)bf16.bits) << 16;
    return u.f;
}

bfloat16_t float_to_bfloat16(float f) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = f;
    bfloat16_t result;
    
    uint32_t rounding_bias = 0x7FFF + ((u.i >> 16) & 1);
    result.bits = (u.i + rounding_bias) >> 16;
    return result;
}

size_t compute_size(const size_t* shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }
    return total;
}

void compute_strides(size_t* strides, const size_t* shape, size_t ndim) {
    if (ndim == 0) return;
    
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor* tensor_create(const size_t* shape, size_t ndim, TensorDType dtype, TensorDevice device) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->device_id = 0;
    tensor->element_size = dtype_size(dtype);
    tensor->owns_data = true;
    
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    compute_strides(tensor->strides, tensor->shape, ndim);
    tensor->size = compute_size(shape, ndim);
    
    size_t bytes = tensor->size * tensor->element_size;
    
    if (device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMalloc(&tensor->data, bytes));
        HIP_CHECK(hipMemset(tensor->data, 0, bytes));
    } else {
        tensor->data = calloc(tensor->size, tensor->element_size);
        if (!tensor->data) {
            free(tensor->shape);
            free(tensor->strides);
            free(tensor);
            return NULL;
        }
    }
    
    return tensor;
}

Tensor* tensor_zeros(const size_t* shape, size_t ndim, TensorDType dtype, TensorDevice device) {
    return tensor_create(shape, ndim, dtype, device);
}

void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    
    if (tensor->owns_data && tensor->data) {
        if (tensor->device == TENSOR_DEVICE_GPU) {
            hipFree(tensor->data);
        } else {
            free(tensor->data);
        }
    }
    
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

int tensor_to_device(Tensor* tensor, TensorDevice target_device) {
    if (tensor->device == target_device) return 0;
    
    size_t bytes = tensor->size * tensor->element_size;
    void* new_data;
    
    if (target_device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMalloc(&new_data, bytes));
        if (tensor->device == TENSOR_DEVICE_CPU) {
            HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyHostToDevice));
        } else {
            HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyDeviceToDevice));
        }
        
        if (tensor->owns_data) {
            if (tensor->device == TENSOR_DEVICE_CPU) {
                free(tensor->data);
            } else {
                hipFree(tensor->data);
            }
        }
    } else {
        new_data = malloc(bytes);
        if (!new_data) return -1;
        
        HIP_CHECK(hipMemcpy(new_data, tensor->data, bytes, hipMemcpyDeviceToHost));
        
        if (tensor->owns_data) {
            hipFree(tensor->data);
        }
    }
    
    tensor->data = new_data;
    tensor->device = target_device;
    tensor->owns_data = true;
    
    return 0;
}

void tensor_copy_data(Tensor* dst, const Tensor* src) {
    assert(dst->size == src->size);
    assert(dst->element_size == src->element_size);
    
    size_t bytes = src->size * src->element_size;
    
    if (src->device == TENSOR_DEVICE_CPU && dst->device == TENSOR_DEVICE_CPU) {
        memcpy(dst->data, src->data, bytes);
    } else if (src->device == TENSOR_DEVICE_GPU && dst->device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyDeviceToDevice));
    } else if (src->device == TENSOR_DEVICE_CPU && dst->device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyHostToDevice));
    } else {
        HIP_CHECK(hipMemcpy(dst->data, src->data, bytes, hipMemcpyDeviceToHost));
    }
}

__global__ void fill_kernel_f32(float* data, float value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void fill_kernel_bf16(uint16_t* data, uint16_t value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void convert_f32_to_bf16_kernel(uint16_t* dst, float* src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        union {
            float f;
            uint32_t i;
        } u;
        u.f = src[idx];
        uint32_t rounding_bias = 0x7FFF + ((u.i >> 16) & 1);
        dst[idx] = (u.i + rounding_bias) >> 16;
    }
}

__global__ void convert_bf16_to_f32_kernel(float* dst, uint16_t* src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        union {
            float f;
            uint32_t i;
        } u;
        u.i = ((uint32_t)src[idx]) << 16;
        dst[idx] = u.f;
    }
}

void tensor_fill(Tensor* tensor, float value) {
    size_t threads_per_block = 256;
    size_t blocks = (tensor->size + threads_per_block - 1) / threads_per_block;
    
    if (tensor->device == TENSOR_DEVICE_GPU) {
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
        HIP_CHECK(hipDeviceSynchronize());
    } else if (tensor->device == TENSOR_DEVICE_CPU) {
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

Tensor* tensor_view(Tensor* tensor, const size_t* new_shape, size_t new_ndim) {
    size_t new_size = compute_size(new_shape, new_ndim);
    if (new_size != tensor->size) return NULL;
    
    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) return NULL;
    
    *view = *tensor;
    view->owns_data = false;
    
    view->shape = (size_t*)malloc(new_ndim * sizeof(size_t));
    view->strides = (size_t*)malloc(new_ndim * sizeof(size_t));
    if (!view->shape || !view->strides) {
        free(view->shape);
        free(view->strides);
        free(view);
        return NULL;
    }
    
    memcpy(view->shape, new_shape, new_ndim * sizeof(size_t));
    compute_strides(view->strides, view->shape, new_ndim);
    view->ndim = new_ndim;
    
    return view;
}

typedef struct {
    Tensor** tensors;
    size_t capacity;
    size_t count;
    size_t total_memory;
} TensorPool;

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

void tensor_pool_add(TensorPool* pool, Tensor* tensor) {
    if (pool->count >= pool->capacity) {
        pool->capacity *= 2;
        pool->tensors = (Tensor**)realloc(pool->tensors, pool->capacity * sizeof(Tensor*));
    }
    
    pool->tensors[pool->count++] = tensor;
    pool->total_memory += tensor->size * tensor->element_size;
}

void tensor_pool_free(TensorPool* pool) {
    if (!pool) return;
    
    for (size_t i = 0; i < pool->count; i++) {
        tensor_free(pool->tensors[i]);
    }
    
    free(pool->tensors);
    free(pool);
}

void tensor_pool_clear_gpu_cache() {
    HIP_CHECK(hipDeviceSynchronize());
}

size_t tensor_memory_usage(const Tensor* tensor) {
    return tensor->size * tensor->element_size;
}

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

int tensor_convert_dtype(Tensor* tensor, TensorDType target_dtype) {
    if (tensor->dtype == target_dtype) return 0;
    
    size_t target_element_size = dtype_size(target_dtype);
    size_t target_bytes = tensor->size * target_element_size;
    void* new_data;
    
    if (tensor->device == TENSOR_DEVICE_GPU) {
        HIP_CHECK(hipMalloc(&new_data, target_bytes));
        
        size_t threads_per_block = 256;
        size_t blocks = (tensor->size + threads_per_block - 1) / threads_per_block;
        
        if (tensor->dtype == TENSOR_DTYPE_FLOAT32 && target_dtype == TENSOR_DTYPE_BFLOAT16) {
            hipLaunchKernelGGL(convert_f32_to_bf16_kernel,
                              dim3(blocks), dim3(threads_per_block), 0, 0,
                              (uint16_t*)new_data, (float*)tensor->data, tensor->size);
        } else if (tensor->dtype == TENSOR_DTYPE_BFLOAT16 && target_dtype == TENSOR_DTYPE_FLOAT32) {
            hipLaunchKernelGGL(convert_bf16_to_f32_kernel,
                              dim3(blocks), dim3(threads_per_block), 0, 0,
                              (float*)new_data, (uint16_t*)tensor->data, tensor->size);
        } else {
            hipFree(new_data);
            return -1;
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        if (tensor->owns_data) {
            hipFree(tensor->data);
        }
    } else {
        new_data = malloc(target_bytes);
        if (!new_data) return -1;
        
        if (tensor->dtype == TENSOR_DTYPE_FLOAT32 && target_dtype == TENSOR_DTYPE_BFLOAT16) {
            float* src = (float*)tensor->data;
            uint16_t* dst = (uint16_t*)new_data;
            for (size_t i = 0; i < tensor->size; i++) {
                bfloat16_t bf16 = float_to_bfloat16(src[i]);
                dst[i] = bf16.bits;
            }
        } else if (tensor->dtype == TENSOR_DTYPE_BFLOAT16 && target_dtype == TENSOR_DTYPE_FLOAT32) {
            uint16_t* src = (uint16_t*)tensor->data;
            float* dst = (float*)new_data;
            for (size_t i = 0; i < tensor->size; i++) {
                bfloat16_t bf16;
                bf16.bits = src[i];
                dst[i] = bfloat16_to_float(bf16);
            }
        } else {
            free(new_data);
            return -1;
        }
        
        if (tensor->owns_data) {
            free(tensor->data);
        }
    }
    
    tensor->data = new_data;
    tensor->dtype = target_dtype;
    tensor->element_size = target_element_size;
    tensor->owns_data = true;
    
    return 0;
}

#endif // TENSOR_HIP