#include "tensor.cpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

class TensorTester {
public:
    static void run_all_tests() {
        std::cout << "Running Tensor Tests...\n";
        
        test_tensor_creation();
        test_bf16_conversion();
        test_tensor_fill();
        test_tensor_device_transfer();
        test_tensor_view();
        test_tensor_pool();
        test_memory_management();
        test_dtype_conversion();
        
        std::cout << "All tests passed!\n";
    }

private:
    static void test_tensor_creation() {
        std::cout << "Testing tensor creation...\n";
        
        // Test CPU tensor creation
        size_t shape[] = {2, 3, 4};
        Tensor* cpu_tensor = tensor_create(shape, 3, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        
        assert(cpu_tensor != nullptr);
        assert(cpu_tensor->ndim == 3);
        assert(cpu_tensor->shape[0] == 2);
        assert(cpu_tensor->shape[1] == 3);
        assert(cpu_tensor->shape[2] == 4);
        assert(cpu_tensor->size == 24);
        assert(cpu_tensor->dtype == TENSOR_DTYPE_FLOAT32);
        assert(cpu_tensor->device == TENSOR_DEVICE_CPU);
        assert(cpu_tensor->owns_data == true);
        
        tensor_free(cpu_tensor);
        
        // Test GPU tensor creation
        Tensor* gpu_tensor = tensor_create(shape, 3, TENSOR_DTYPE_BFLOAT16, TENSOR_DEVICE_GPU);
        assert(gpu_tensor != nullptr);
        assert(gpu_tensor->dtype == TENSOR_DTYPE_BFLOAT16);
        assert(gpu_tensor->device == TENSOR_DEVICE_GPU);
        assert(gpu_tensor->element_size == 2);
        
        tensor_free(gpu_tensor);
        
        std::cout << "✓ Tensor creation tests passed\n";
    }
    
    static void test_bf16_conversion() {
        std::cout << "Testing BF16 conversion...\n";
        
        // Test individual conversions
        float test_values[] = {0.0f, 1.0f, -1.0f, 3.14159f, -2.71828f, 1e-5f, 1e5f};
        
        for (float val : test_values) {
            bfloat16_t bf16 = float_to_bfloat16(val);
            float converted_back = bfloat16_to_float(bf16);
            
            // BF16 has lower precision, so we need a reasonable tolerance
            float relative_error = std::abs(val - converted_back) / (std::abs(val) + 1e-8f);
            assert(relative_error < 0.01f || std::abs(val - converted_back) < 1e-4f);
        }
        
        // Test edge cases
        bfloat16_t zero = float_to_bfloat16(0.0f);
        assert(bfloat16_to_float(zero) == 0.0f);
        
        bfloat16_t one = float_to_bfloat16(1.0f);
        assert(bfloat16_to_float(one) == 1.0f);
        
        std::cout << "✓ BF16 conversion tests passed\n";
    }
    
    static void test_tensor_fill() {
        std::cout << "Testing tensor fill operations...\n";
        
        size_t shape[] = {10, 10};
        
        // Test CPU FLOAT32 fill
        Tensor* cpu_f32 = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        tensor_fill(cpu_f32, 3.14f);
        
        float* cpu_data = (float*)cpu_f32->data;
        for (size_t i = 0; i < cpu_f32->size; i++) {
            assert(std::abs(cpu_data[i] - 3.14f) < 1e-6f);
        }
        tensor_free(cpu_f32);
        
        // Test CPU BF16 fill
        Tensor* cpu_bf16 = tensor_create(shape, 2, TENSOR_DTYPE_BFLOAT16, TENSOR_DEVICE_CPU);
        tensor_fill(cpu_bf16, 2.718f);
        
        uint16_t* cpu_bf16_data = (uint16_t*)cpu_bf16->data;
        bfloat16_t expected_bf16 = float_to_bfloat16(2.718f);
        for (size_t i = 0; i < cpu_bf16->size; i++) {
            assert(cpu_bf16_data[i] == expected_bf16.bits);
        }
        tensor_free(cpu_bf16);
        
        // Test GPU operations
        Tensor* gpu_f32 = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_GPU);
        tensor_fill(gpu_f32, 1.618f);
        
        // Copy back to CPU to verify
        float* host_data = (float*)malloc(gpu_f32->size * sizeof(float));
        HIP_CHECK(hipMemcpy(host_data, gpu_f32->data, gpu_f32->size * sizeof(float), hipMemcpyDeviceToHost));
        
        for (size_t i = 0; i < gpu_f32->size; i++) {
            assert(std::abs(host_data[i] - 1.618f) < 1e-6f);
        }
        
        free(host_data);
        tensor_free(gpu_f32);
        
        std::cout << "✓ Tensor fill tests passed\n";
    }
    
    static void test_tensor_device_transfer() {
        std::cout << "Testing device transfer...\n";
        
        size_t shape[] = {5, 4};
        
        // Create CPU tensor and fill with data
        Tensor* cpu_tensor = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        float* cpu_data = (float*)cpu_tensor->data;
        for (size_t i = 0; i < cpu_tensor->size; i++) {
            cpu_data[i] = (float)i * 0.1f;
        }
        
        // Transfer to GPU
        int result = tensor_to_device(cpu_tensor, TENSOR_DEVICE_GPU);
        assert(result == 0);
        assert(cpu_tensor->device == TENSOR_DEVICE_GPU);
        
        // Transfer back to CPU
        result = tensor_to_device(cpu_tensor, TENSOR_DEVICE_CPU);
        assert(result == 0);
        assert(cpu_tensor->device == TENSOR_DEVICE_CPU);
        
        // Verify data integrity
        float* final_data = (float*)cpu_tensor->data;
        for (size_t i = 0; i < cpu_tensor->size; i++) {
            assert(std::abs(final_data[i] - (float)i * 0.1f) < 1e-6f);
        }
        
        tensor_free(cpu_tensor);
        
        std::cout << "✓ Device transfer tests passed\n";
    }
    
    static void test_tensor_view() {
        std::cout << "Testing tensor views...\n";
        
        size_t original_shape[] = {6, 4};
        Tensor* original = tensor_create(original_shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        
        // Fill with test data
        float* data = (float*)original->data;
        for (size_t i = 0; i < original->size; i++) {
            data[i] = (float)i;
        }
        
        // Create view with different shape but same size
        size_t view_shape[] = {2, 3, 4};
        Tensor* view = tensor_view(original, view_shape, 3);
        
        assert(view != nullptr);
        assert(view->ndim == 3);
        assert(view->size == original->size);
        assert(view->data == original->data);
        assert(view->owns_data == false);
        
        // Test invalid view (different size)
        size_t invalid_shape[] = {5, 5};
        Tensor* invalid_view = tensor_view(original, invalid_shape, 2);
        assert(invalid_view == nullptr);
        
        tensor_free(view);
        tensor_free(original);
        
        std::cout << "✓ Tensor view tests passed\n";
    }
    
    static void test_tensor_pool() {
        std::cout << "Testing tensor pool...\n";
        
        TensorPool* pool = tensor_pool_create(10);
        assert(pool != nullptr);
        assert(pool->count == 0);
        assert(pool->capacity == 10);
        
        // Create some tensors and add to pool
        size_t shape[] = {3, 3};
        for (int i = 0; i < 5; i++) {
            Tensor* tensor = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
            tensor_pool_add(pool, tensor);
        }
        
        assert(pool->count == 5);
        assert(pool->total_memory == 5 * 9 * sizeof(float));
        
        tensor_pool_free(pool);
        
        std::cout << "✓ Tensor pool tests passed\n";
    }
    
    static void test_memory_management() {
        std::cout << "Testing memory management...\n";
        
        size_t shape[] = {100, 100};
        
        // Test large tensor creation and cleanup
        Tensor* large_tensor = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        
        size_t expected_memory = 100 * 100 * sizeof(float);
        size_t actual_memory = tensor_memory_usage(large_tensor);
        assert(actual_memory == expected_memory);
        
        tensor_free(large_tensor);
        
        // Test GPU memory management
        Tensor* gpu_tensor = tensor_create(shape, 2, TENSOR_DTYPE_BFLOAT16, TENSOR_DEVICE_GPU);
        assert(gpu_tensor != nullptr);
        
        size_t gpu_memory = tensor_memory_usage(gpu_tensor);
        assert(gpu_memory == 100 * 100 * 2); // BF16 is 2 bytes
        
        tensor_free(gpu_tensor);
        
        std::cout << "✓ Memory management tests passed\n";
    }
    
    static void test_dtype_conversion() {
        std::cout << "Testing dtype conversion...\n";
        
        size_t shape[] = {10, 10};
        
        // Create FLOAT32 tensor
        Tensor* f32_tensor = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_CPU);
        float* f32_data = (float*)f32_tensor->data;
        
        // Fill with test data
        for (size_t i = 0; i < f32_tensor->size; i++) {
            f32_data[i] = (float)i * 0.01f;
        }
        
        // Convert to BF16
        int result = tensor_convert_dtype(f32_tensor, TENSOR_DTYPE_BFLOAT16);
        assert(result == 0);
        assert(f32_tensor->dtype == TENSOR_DTYPE_BFLOAT16);
        assert(f32_tensor->element_size == 2);
        
        // Convert back to FLOAT32
        result = tensor_convert_dtype(f32_tensor, TENSOR_DTYPE_FLOAT32);
        assert(result == 0);
        assert(f32_tensor->dtype == TENSOR_DTYPE_FLOAT32);
        assert(f32_tensor->element_size == 4);
        
        // Verify data with reasonable tolerance
        float* final_data = (float*)f32_tensor->data;
        for (size_t i = 0; i < f32_tensor->size; i++) {
            float original = (float)i * 0.01f;
            float relative_error = std::abs(original - final_data[i]) / (std::abs(original) + 1e-8f);
            assert(relative_error < 0.01f || std::abs(original - final_data[i]) < 1e-4f);
        }
        
        tensor_free(f32_tensor);
        
        // Test GPU conversion
        Tensor* gpu_f32 = tensor_create(shape, 2, TENSOR_DTYPE_FLOAT32, TENSOR_DEVICE_GPU);
        tensor_fill(gpu_f32, 3.14159f);
        
        result = tensor_convert_dtype(gpu_f32, TENSOR_DTYPE_BFLOAT16);
        assert(result == 0);
        assert(gpu_f32->dtype == TENSOR_DTYPE_BFLOAT16);
        
        result = tensor_convert_dtype(gpu_f32, TENSOR_DTYPE_FLOAT32);
        assert(result == 0);
        assert(gpu_f32->dtype == TENSOR_DTYPE_FLOAT32);
        
        tensor_free(gpu_f32);
        
        std::cout << "✓ Dtype conversion tests passed\n";
    }
};

int main() {
    // Initialize HIP
    hipError_t hip_status = hipSetDevice(0);
    if (hip_status != hipSuccess) {
        std::cout << "Warning: HIP device initialization failed, GPU tests will be skipped\n";
        std::cout << "HIP Error: " << hipGetErrorString(hip_status) << std::endl;
        return 1;
    }
    
    try {
        TensorTester::run_all_tests();
        std::cout << "\n🎉 All tensor tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception\n";
        return 1;
    }
}