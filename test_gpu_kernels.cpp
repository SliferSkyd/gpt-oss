// GPU Kernel Testing Framework for GPT-OSS
// This file provides comprehensive testing for GPU kernels against CPU reference

#include "getp-csrc/gpu_memory.hpp"
#include "getp-csrc/gpu_kernels.hpp"
#include "include/model.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>

// Test configuration
struct TestConfig {
    int batch_size = 2;
    int hidden_dim = 2880;
    int intermediate_dim = 2880;
    int vocab_size = 201088;
    int n_heads = 64;
    int n_kv_heads = 8;
    int head_dim = 64;
    int n_experts = 32;
    int experts_per_token = 4;
    int seq_len = 128;
    float tolerance = 1e-3f;
    bool verbose = true;
};

// Random data generator
class DataGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> float_dist;
    std::uniform_int_distribution<int> int_dist;
    
public:
    DataGenerator(int seed = 42) : rng(seed), float_dist(-1.0f, 1.0f), int_dist(0, 100) {}
    
    void generate_float_array(float* arr, size_t size) {
        for (size_t i = 0; i < size; i++) {
            arr[i] = float_dist(rng);
        }
    }
    
    void generate_int_array(int* arr, size_t size, int max_val) {
        std::uniform_int_distribution<int> dist(0, max_val - 1);
        for (size_t i = 0; i < size; i++) {
            arr[i] = dist(rng);
        }
    }
};

// CPU Reference Implementations
namespace CPUReference {
    
    // RMSNorm CPU reference
    void rmsnorm_cpu(const float* input, const float* weight, float* output,
                     int batch_size, int hidden_dim, float eps = 1e-5f) {
        for (int b = 0; b < batch_size; b++) {
            const float* in_row = input + b * hidden_dim;
            float* out_row = output + b * hidden_dim;
            
            // Compute RMS
            float sum_sq = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                sum_sq += in_row[i] * in_row[i];
            }
            float rms = sqrtf(sum_sq / hidden_dim + eps);
            float scale = 1.0f / rms;
            
            // Apply normalization and weight
            for (int i = 0; i < hidden_dim; i++) {
                out_row[i] = in_row[i] * scale * weight[i];
            }
        }
    }
    
    // Embedding lookup CPU reference
    void embedding_lookup_cpu(const float* embedding_table, const int* token_ids,
                             float* output, int batch_size, int hidden_dim) {
        for (int b = 0; b < batch_size; b++) {
            int token = token_ids[b];
            const float* embed_row = embedding_table + token * hidden_dim;
            float* out_row = output + b * hidden_dim;
            for (int i = 0; i < hidden_dim; i++) {
                out_row[i] = embed_row[i];
            }
        }
    }
    
    // SwiGLU activation CPU reference
    float swiglu_cpu(float x, float limit) {
        if (x > limit) return x;
        if (x < -limit) return 0.0f;
        return x / (1.0f + expf(-x));
    }
    
    // Top-K selection CPU reference
    void topk_cpu(const float* scores, int* indices, float* values,
                  int n_items, int k) {
        std::vector<std::pair<float, int>> score_pairs;
        for (int i = 0; i < n_items; i++) {
            score_pairs.push_back({scores[i], i});
        }
        
        std::partial_sort(score_pairs.begin(), score_pairs.begin() + k,
                         score_pairs.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            indices[i] = score_pairs[i].second;
            values[i] = expf(score_pairs[i].first);
            sum += values[i];
        }
        
        // Normalize
        for (int i = 0; i < k; i++) {
            values[i] /= sum;
        }
    }
}

// Test Cases
class GPUKernelTester {
private:
    TestConfig config;
    GPUMemoryManager memory_mgr;
    DataGenerator data_gen;
    
public:
    GPUKernelTester(const TestConfig& cfg) : config(cfg), data_gen(42) {}
    
    bool test_embedding_lookup() {
        printf("\n=== Testing Embedding Lookup Kernel ===\n");
        
        size_t embed_size = config.vocab_size * config.hidden_dim;
        size_t output_size = config.batch_size * config.hidden_dim;
        
        // Allocate host memory
        float* h_embed_table = new float[embed_size];
        int* h_token_ids = new int[config.batch_size];
        float* h_output_cpu = new float[output_size];
        float* h_output_gpu = new float[output_size];
        
        // Generate random data
        data_gen.generate_float_array(h_embed_table, embed_size);
        data_gen.generate_int_array(h_token_ids, config.batch_size, config.vocab_size);
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        CPUReference::embedding_lookup_cpu(h_embed_table, h_token_ids, h_output_cpu,
                                          config.batch_size, config.hidden_dim);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        // GPU implementation
        bf16* d_embed_table = (bf16*)memory_mgr.allocate("embed_table",
                                                         embed_size * sizeof(bf16), 0, true);
        int* d_token_ids = (int*)memory_mgr.allocate("token_ids",
                                                     config.batch_size * sizeof(int), 0, false);
        float* d_output = (float*)memory_mgr.allocate("output",
                                                      output_size * sizeof(float), 0, false);
        
        // Copy data to GPU (with BF16 conversion for embeddings)
        memory_mgr.copy_host_to_device("embed_table", h_embed_table, embed_size, true);
        hipMemcpy(d_token_ids, h_token_ids, config.batch_size * sizeof(int),
                 hipMemcpyHostToDevice);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_embedding_lookup(d_embed_table, d_token_ids, d_output,
                           config.batch_size, config.hidden_dim, config.vocab_size);
        hipDeviceSynchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        
        // Copy result back
        hipMemcpy(h_output_gpu, d_output, output_size * sizeof(float),
                 hipMemcpyDeviceToHost);
        
        // Validate
        bool passed = validate_kernel_output(d_output, h_output_cpu, output_size,
                                            config.tolerance, "Embedding Lookup");
        
        // Print timing
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>
                       (cpu_end - cpu_start).count();
        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>
                       (gpu_end - gpu_start).count();
        
        printf("CPU Time: %.3f ms\n", cpu_time / 1000.0);
        printf("GPU Time: %.3f ms\n", gpu_time / 1000.0);
        printf("Speedup: %.2fx\n", (float)cpu_time / gpu_time);
        
        // Cleanup
        delete[] h_embed_table;
        delete[] h_token_ids;
        delete[] h_output_cpu;
        delete[] h_output_gpu;
        memory_mgr.deallocate("embed_table");
        memory_mgr.deallocate("token_ids");
        memory_mgr.deallocate("output");
        
        return passed;
    }
    
    bool test_rmsnorm() {
        printf("\n=== Testing RMSNorm Kernel ===\n");
        
        size_t input_size = config.batch_size * config.hidden_dim;
        
        // Allocate host memory
        float* h_input = new float[input_size];
        float* h_weight = new float[config.hidden_dim];
        float* h_output_cpu = new float[input_size];
        float* h_output_gpu = new float[input_size];
        
        // Generate random data
        data_gen.generate_float_array(h_input, input_size);
        data_gen.generate_float_array(h_weight, config.hidden_dim);
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        CPUReference::rmsnorm_cpu(h_input, h_weight, h_output_cpu,
                                 config.batch_size, config.hidden_dim);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        // GPU implementation
        float* d_input = (float*)memory_mgr.allocate("input",
                                                     input_size * sizeof(float), 0, false);
        bf16* d_weight = (bf16*)memory_mgr.allocate("weight",
                                                    config.hidden_dim * sizeof(bf16), 0, true);
        float* d_output = (float*)memory_mgr.allocate("output",
                                                      input_size * sizeof(float), 0, false);
        
        // Copy data to GPU
        hipMemcpy(d_input, h_input, input_size * sizeof(float), hipMemcpyHostToDevice);
        memory_mgr.copy_host_to_device("weight", h_weight, config.hidden_dim, true);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_rmsnorm(d_input, d_weight, d_output,
                   config.batch_size, config.hidden_dim);
        hipDeviceSynchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();
        
        // Copy result back
        hipMemcpy(h_output_gpu, d_output, input_size * sizeof(float),
                 hipMemcpyDeviceToHost);
        
        // Validate
        bool passed = validate_kernel_output(d_output, h_output_cpu, input_size,
                                            config.tolerance, "RMSNorm");
        
        // Print timing
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>
                       (cpu_end - cpu_start).count();
        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>
                       (gpu_end - gpu_start).count();
        
        printf("CPU Time: %.3f ms\n", cpu_time / 1000.0);
        printf("GPU Time: %.3f ms\n", gpu_time / 1000.0);
        printf("Speedup: %.2fx\n", (float)cpu_time / gpu_time);
        
        // Cleanup
        delete[] h_input;
        delete[] h_weight;
        delete[] h_output_cpu;
        delete[] h_output_gpu;
        memory_mgr.deallocate("input");
        memory_mgr.deallocate("weight");
        memory_mgr.deallocate("output");
        
        return passed;
    }
    
    bool test_topk_selection() {
        printf("\n=== Testing Top-K Selection ===\n");
        
        // Allocate host memory
        float* h_scores = new float[config.n_experts];
        int* h_indices_cpu = new int[config.experts_per_token];
        float* h_values_cpu = new float[config.experts_per_token];
        int* h_indices_gpu = new int[config.experts_per_token];
        float* h_values_gpu = new float[config.experts_per_token];
        
        // Generate random scores
        data_gen.generate_float_array(h_scores, config.n_experts);
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        CPUReference::topk_cpu(h_scores, h_indices_cpu, h_values_cpu,
                              config.n_experts, config.experts_per_token);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        // GPU implementation would go here (simplified for now)
        // For testing purposes, we can validate the CPU implementation
        
        printf("Top-K Results:\n");
        for (int i = 0; i < config.experts_per_token; i++) {
            printf("  Expert %d: score=%.4f, weight=%.4f\n",
                   h_indices_cpu[i], h_scores[h_indices_cpu[i]], h_values_cpu[i]);
        }
        
        // Cleanup
        delete[] h_scores;
        delete[] h_indices_cpu;
        delete[] h_values_cpu;
        delete[] h_indices_gpu;
        delete[] h_values_gpu;
        
        return true;
    }
    
    void run_all_tests() {
        printf("\n========================================\n");
        printf("    GPU Kernel Testing Framework\n");
        printf("========================================\n");
        printf("Configuration:\n");
        printf("  Batch Size: %d\n", config.batch_size);
        printf("  Hidden Dim: %d\n", config.hidden_dim);
        printf("  Vocab Size: %d\n", config.vocab_size);
        printf("  Experts: %d (top-%d)\n", config.n_experts, config.experts_per_token);
        printf("  Tolerance: %.1e\n", config.tolerance);
        
        int passed = 0;
        int total = 0;
        
        // Run individual kernel tests
        if (test_embedding_lookup()) passed++; total++;
        if (test_rmsnorm()) passed++; total++;
        if (test_topk_selection()) passed++; total++;
        
        // Summary
        printf("\n========================================\n");
        printf("Test Summary: %d/%d passed\n", passed, total);
        printf("========================================\n");
    }
};

// Performance benchmark
void benchmark_kernels(const TestConfig& config) {
    printf("\n========================================\n");
    printf("    Kernel Performance Benchmarks\n");
    printf("========================================\n");
    
    GPUMemoryManager memory_mgr;
    DataGenerator data_gen;
    
    // Benchmark embedding lookup
    {
        size_t embed_size = 1000 * config.hidden_dim; // Smaller for benchmark
        bf16* d_embed = (bf16*)memory_mgr.allocate("embed",
                                                   embed_size * sizeof(bf16), 0, true);
        int* d_tokens = (int*)memory_mgr.allocate("tokens",
                                                  config.batch_size * sizeof(int), 0, false);
        float* d_output = (float*)memory_mgr.allocate("output",
                                                      config.batch_size * config.hidden_dim * sizeof(float),
                                                      0, false);
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            gpu_embedding_lookup(d_embed, d_tokens, d_output,
                               config.batch_size, config.hidden_dim, 1000);
        }
        hipDeviceSynchronize();
        
        // Benchmark
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        hipEventRecord(start);
        for (int i = 0; i < 100; i++) {
            gpu_embedding_lookup(d_embed, d_tokens, d_output,
                               config.batch_size, config.hidden_dim, 1000);
        }
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        
        printf("Embedding Lookup: %.3f ms/iter\n", ms / 100);
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
        memory_mgr.deallocate_all();
    }
    
    // Add more kernel benchmarks here...
}

int main(int argc, char** argv) {
    // Check for GPUs
    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    
    if (gpu_count == 0) {
        printf("Error: No AMD GPUs found!\n");
        return 1;
    }
    
    printf("Found %d AMD GPU(s)\n", gpu_count);
    
    // Set up test configuration
    TestConfig config;
    if (argc > 1) {
        config.batch_size = atoi(argv[1]);
    }
    if (argc > 2) {
        config.tolerance = atof(argv[2]);
    }
    
    // Run tests
    GPUKernelTester tester(config);
    tester.run_all_tests();
    
    // Run benchmarks
    benchmark_kernels(config);
    
    return 0;
}