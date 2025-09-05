#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include "../kernels/matmul.hpp"
#include "../config.hpp"

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t error = call;                                              \
        if (error != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    hipGetErrorString(error));                                \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

void reference_matmul_cpu(
    float* C,           // [M, N]
    const float* A,     // [M, K]
    const float* B,     // [K, N] (logical), stored as [N, K] (transposed)
    int M, int K, int N,
    bool B_transposed = false)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // If B is transposed, access as B[n][k] instead of B[k][n]
                float b_val = B_transposed ? B[n * K + k] : B[k * N + n];
                sum += A[m * K + k] * b_val;
            }
            C[m * N + n] = sum;
        }
    }
}

float compute_relative_error(float* gpu_result, float* cpu_result, int size) {
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < size; ++i) {
        float abs_diff = std::fabs(gpu_result[i] - cpu_result[i]);
        float rel_error = 0.0f;
        
        if (std::fabs(cpu_result[i]) > 1e-6) {
            rel_error = abs_diff / std::fabs(cpu_result[i]);
        } else if (abs_diff > 1e-6) {
            rel_error = abs_diff;
        }
        
        avg_error += rel_error;
        if (rel_error > max_error) {
            max_error = rel_error;
        }
        
        // Count significant errors (> 10% relative error for BF16)
        if (rel_error > 0.10f) {
            error_count++;
            if (error_count <= 5) { // Print first 5 errors
                printf("  Error at index %d: GPU=%.6f, CPU=%.6f, rel_error=%.4f\n",
                       i, gpu_result[i], cpu_result[i], rel_error);
            }
        }
    }
    
    avg_error /= size;
    printf("  Max relative error: %.6f\n", max_error);
    printf("  Avg relative error: %.6f\n", avg_error);
    printf("  Elements with >10%% error: %d/%d\n", error_count, size);
    
    return max_error;
}

void initialize_matrix(float* mat, int size, float min_val = -1.0f, float max_val = 1.0f, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        mat[i] = dist(gen);
    }
}

void test_basic_correctness() {
    printf("\n=== Test 1: Basic Correctness ===\n");
    
    // Small test case for exact verification
    const int M = 64, K = 128, N = 96;
    
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C_gpu = (float*)malloc(M * N * sizeof(float));
    h_C_cpu = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    initialize_matrix(h_A, M * K, -1.0f, 1.0f, 123);
    // Note: B is stored transposed as [N, K] for matmul_mc
    initialize_matrix(h_B, N * K, -1.0f, 1.0f, 456);  // Changed from K*N to N*K
    memset(h_C_gpu, 0, M * N * sizeof(float));
    memset(h_C_cpu, 0, M * N * sizeof(float));
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N * K * sizeof(float)));  // Changed from K*N to N*K (transposed)
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, N * K * sizeof(float), hipMemcpyHostToDevice));  // Changed from K*N to N*K
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
    
    // Run GPU kernel
    printf("Running matmul_mc with M=%d, K=%d, N=%d\n", M, K, N);
    matmul_mc(d_C, d_A, d_B, M, K, N);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy back result
    HIP_CHECK(hipMemcpy(h_C_gpu, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Compute reference on CPU (with B_transposed=true since weights are stored transposed)
    printf("Computing reference on CPU...\n");
    reference_matmul_cpu(h_C_cpu, h_A, h_B, M, K, N, true);
    
    // Compare results
    float max_error = compute_relative_error(h_C_gpu, h_C_cpu, M * N);
    
    // BF16 tolerance: ~5% error is acceptable due to BF16 precision
    if (max_error < 0.05f) {
        printf("✓ Basic correctness test PASSED\n");
    } else {
        printf("✗ Basic correctness test FAILED (error too high)\n");
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
}

void test_qkv_projection() {
    printf("\n=== Test 2: QKV Projection (Attention Layer) ===\n");
    
    // Dimensions from actual usage in attention_gpu
    const int batch_size = 32;
    const int hidden_dim = 2880;
    const int n_attn_heads = 64;
    const int n_kv_heads = 8;
    const int head_dim = 64;
    const int output_dim = (n_attn_heads + 2 * n_kv_heads) * head_dim;
    
    printf("Testing QKV projection: batch=%d, hidden=%d, output=%d\n", 
           batch_size, hidden_dim, output_dim);
    printf("  n_attn_heads=%d, n_kv_heads=%d, head_dim=%d\n",
           n_attn_heads, n_kv_heads, head_dim);
    
    float *h_input, *h_weight, *h_output_gpu, *h_output_cpu;
    float *d_input, *d_weight, *d_output;
    
    // Allocate host memory
    h_input = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    h_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));  // Transposed: [N, K]
    h_output_gpu = (float*)malloc(batch_size * output_dim * sizeof(float));
    h_output_cpu = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize with realistic values
    initialize_matrix(h_input, batch_size * hidden_dim, -2.0f, 2.0f, 789);
    initialize_matrix(h_weight, output_dim * hidden_dim, -0.1f, 0.1f, 321);  // Transposed dimensions
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_input, batch_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, output_dim * hidden_dim * sizeof(float)));  // Transposed
    HIP_CHECK(hipMalloc(&d_output, batch_size * output_dim * sizeof(float)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_input, h_input, batch_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight, output_dim * hidden_dim * sizeof(float), hipMemcpyHostToDevice));  // Transposed
    HIP_CHECK(hipMemset(d_output, 0, batch_size * output_dim * sizeof(float)));
    
    // Warmup
    for (int i = 0; i < 3; ++i) {
        matmul_mc(d_output, d_input, d_weight, batch_size, hidden_dim, output_dim);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    const int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        matmul_mc(d_output, d_input, d_weight, batch_size, hidden_dim, output_dim);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = elapsed_ms / num_iterations;
    
    // Calculate FLOPS
    double flops = 2.0 * batch_size * hidden_dim * output_dim;
    double gflops = (flops * num_iterations) / (elapsed_ms * 1e6);
    
    printf("Performance: %.2f ms/iteration, %.2f GFLOPS\n", avg_time, gflops);
    
    // Copy back and validate
    HIP_CHECK(hipMemcpy(h_output_gpu, d_output, batch_size * output_dim * sizeof(float), hipMemcpyDeviceToHost));
    
    // CPU reference (only validate subset for speed)
    printf("Validating subset of results...\n");
    const int validate_rows = std::min(4, batch_size);
    reference_matmul_cpu(h_output_cpu, h_input, h_weight, validate_rows, hidden_dim, output_dim, true);  // B_transposed=true
    
    float max_error = compute_relative_error(h_output_gpu, h_output_cpu, validate_rows * output_dim);
    
    if (max_error < 0.10f) { // BF16 precision tolerance
        printf("✓ QKV projection test PASSED\n");
    } else {
        printf("✗ QKV projection test FAILED (max error %.4f exceeds 10%% threshold)\n", max_error);
    }
    
    // Cleanup
    free(h_input); free(h_weight); free(h_output_gpu); free(h_output_cpu);
    hipFree(d_input); hipFree(d_weight); hipFree(d_output);
}

void test_edge_cases() {
    printf("\n=== Test 3: Edge Cases ===\n");
    
    struct TestCase {
        int M, K, N;
        const char* description;
    };
    
    TestCase cases[] = {
        {1, 128, 128, "Single batch"},
        {32, 1, 128, "K=1"},
        {32, 128, 1, "N=1"},
        {17, 97, 63, "Non-aligned dimensions"},
        {128, 2048, 2048, "Large square matrices"},
        {1, 2880, 3600, "Single token QKV (60*60 output)"}
    };
    
    for (const auto& tc : cases) {
        printf("\nTesting: %s (M=%d, K=%d, N=%d)\n", tc.description, tc.M, tc.K, tc.N);
        
        float *d_A, *d_B, *d_C;
        float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
        
        // Allocate
        h_A = (float*)malloc(tc.M * tc.K * sizeof(float));
        h_B = (float*)malloc(tc.N * tc.K * sizeof(float));  // Transposed: [N, K]
        h_C_gpu = (float*)malloc(tc.M * tc.N * sizeof(float));
        h_C_cpu = (float*)malloc(tc.M * tc.N * sizeof(float));
        
        HIP_CHECK(hipMalloc(&d_A, tc.M * tc.K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, tc.N * tc.K * sizeof(float)));  // Transposed
        HIP_CHECK(hipMalloc(&d_C, tc.M * tc.N * sizeof(float)));
        
        // Initialize
        initialize_matrix(h_A, tc.M * tc.K);
        initialize_matrix(h_B, tc.N * tc.K);  // Transposed dimensions
        
        // Copy to device
        HIP_CHECK(hipMemcpy(d_A, h_A, tc.M * tc.K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, tc.N * tc.K * sizeof(float), hipMemcpyHostToDevice));  // Transposed
        
        // Run kernel
        matmul_mc(d_C, d_A, d_B, tc.M, tc.K, tc.N);
        HIP_CHECK(hipDeviceSynchronize());
        
        // Validate small cases only
        if (tc.M * tc.N <= 10000) {
            HIP_CHECK(hipMemcpy(h_C_gpu, d_C, tc.M * tc.N * sizeof(float), hipMemcpyDeviceToHost));
            reference_matmul_cpu(h_C_cpu, h_A, h_B, tc.M, tc.K, tc.N, true);  // B_transposed=true
            float error = compute_relative_error(h_C_gpu, h_C_cpu, tc.M * tc.N);
            
            if (error < 0.05f) {
                printf("  ✓ Test passed\n");
            } else {
                printf("  ✗ Test failed (error=%.4f)\n", error);
            }
        } else {
            printf("  ✓ Kernel executed successfully (validation skipped for large size)\n");
        }
        
        // Cleanup
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
    }
}

void test_numerical_stability() {
    printf("\n=== Test 4: Numerical Stability ===\n");
    
    const int M = 32, K = 512, N = 256;
    
    struct StabilityTest {
        float min_val, max_val;
        const char* description;
    };
    
    StabilityTest tests[] = {
        {-1e-3f, 1e-3f, "Small values"},
        {-1e3f, 1e3f, "Large values"},
        {0.0f, 1.0f, "Positive only"},
        {-1.0f, 0.0f, "Negative only"},
        {-10.0f, 10.0f, "Standard range"}
    };
    
    for (const auto& test : tests) {
        printf("\nTesting %s (range [%.2e, %.2e])\n", test.description, test.min_val, test.max_val);
        
        float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
        float *d_A, *d_B, *d_C;
        
        // Allocate
        h_A = (float*)malloc(M * K * sizeof(float));
        h_B = (float*)malloc(N * K * sizeof(float));  // Transposed: [N, K]
        h_C_gpu = (float*)malloc(M * N * sizeof(float));
        h_C_cpu = (float*)malloc(M * N * sizeof(float));
        
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, N * K * sizeof(float)));  // Transposed
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
        
        // Initialize with specific range
        initialize_matrix(h_A, M * K, test.min_val, test.max_val, 111);
        initialize_matrix(h_B, N * K, test.min_val, test.max_val, 222);  // Transposed
        
        // Copy to device
        HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, N * K * sizeof(float), hipMemcpyHostToDevice));  // Transposed
        
        // Run kernel
        matmul_mc(d_C, d_A, d_B, M, K, N);
        HIP_CHECK(hipDeviceSynchronize());
        
        // Validate
        HIP_CHECK(hipMemcpy(h_C_gpu, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
        reference_matmul_cpu(h_C_cpu, h_A, h_B, M, K, N, true);  // B_transposed=true
        
        // Check for NaN/Inf
        bool has_nan = false, has_inf = false;
        for (int i = 0; i < M * N; ++i) {
            if (std::isnan(h_C_gpu[i])) has_nan = true;
            if (std::isinf(h_C_gpu[i])) has_inf = true;
        }
        
        if (has_nan || has_inf) {
            printf("  ✗ Numerical instability detected (NaN=%d, Inf=%d)\n", has_nan, has_inf);
        } else {
            float error = compute_relative_error(h_C_gpu, h_C_cpu, M * N);
            if (error < 0.05f) { // Higher tolerance for extreme values
                printf("  ✓ Numerically stable (max error=%.4f)\n", error);
            } else {
                printf("  ✗ Error too high (%.4f)\n", error);
            }
        }
        
        // Cleanup
        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
    }
}

int main(int argc, char** argv) {
    printf("=================================\n");
    printf("matmul_mc Validation Test Suite\n");
    printf("=================================\n");
    
    // Get device info
    int device;
    hipGetDevice(&device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Run all tests
    // test_basic_correctness();
    test_qkv_projection();
    // test_edge_cases();
    // test_numerical_stability();
    
    printf("\n=================================\n");
    printf("All tests completed!\n");
    printf("=================================\n");
    
    return 0;
}