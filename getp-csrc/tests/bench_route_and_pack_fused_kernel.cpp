// bench_route_and_pack_fused_kernel_optimized.cpp
// Benchmark harness for route_and_pack_fused_kernel_optimized from MoE GPU implementation
// Simulates real-world usage in moe_gpu function with multiple experts and token routing

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Configuration from the actual codebase
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef TENSOR_PARALLEL_SIZE
#define TENSOR_PARALLEL_SIZE 2
#endif

// Include necessary utilities inline
using f32x4  = float __attribute__((ext_vector_type(4)));
using bf16x4 = unsigned short __attribute__((ext_vector_type(4)));

// For ROCm, use hip_bfloat16 instead of __hip_bfloat16
using __hip_bfloat16 = hip_bfloat16;

__device__ inline uint16_t f32_to_bf16_bits(float x) {
    hip_bfloat16 t = __float2bfloat16(x);
    return *reinterpret_cast<uint16_t*>(&t);
}
__device__ inline uint16_t hipbf16_to_bits(hip_bfloat16 x) {
    return *reinterpret_cast<uint16_t*>(&x);
}
__device__ inline int lane_row(int lane)   { return lane & 15; }
__device__ inline int lane_group(int lane) { return lane >> 4; }

__device__ inline uint32_t pack2_bf16_bits_f32(float a0, float a1) {
    hip_bfloat16 b0 = __float2bfloat16(a0);
    hip_bfloat16 b1 = __float2bfloat16(a1);
    return (uint32_t(hipbf16_to_bits(b1)) << 16) | uint32_t(hipbf16_to_bits(b0));
}

// MFMA wrapper
__device__ inline f32x4 mfma_16x16x16_bf16(bf16x4 a_vec, bf16x4 b_vec, f32x4 c_vec) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
}

// Error handling - must be before including moe.hpp
#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    std::cerr << "HIP error " << hipGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// Include the MoE kernel
#include "../kernels/moe.hpp"

// CLI argument parsing
template<typename T>
static T get_arg(int argc, char** argv, const char* key, T def) {
  for (int i=1;i<argc-1;++i) {
    if (std::string(argv[i])==key) {
      if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::strtoll(argv[i+1], nullptr, 10));
      } else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(std::atof(argv[i+1]));
      }
    }
  }
  return def;
}

// Random data generation
static void fill_uniform_float(float* ptr, size_t n, uint64_t seed, float lo=-1.f, float hi=1.f) {
  std::mt19937 rng((uint32_t)seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i=0;i<n;++i) ptr[i] = dist(rng);
}

static void generate_topk_indices(int* topk_i, int B, int K, int E, uint64_t seed) {
  std::mt19937 rng((uint32_t)seed);
  std::uniform_int_distribution<int> dist(0, E-1);

  for (int b = 0; b < B; ++b) {
    // Generate K distinct experts for each token
    std::vector<int> experts;
    while (experts.size() < K) {
      int e = dist(rng);
      if (std::find(experts.begin(), experts.end(), e) == experts.end()) {
        experts.push_back(e);
      }
    }
    for (int k = 0; k < K; ++k) {
      topk_i[b * K + k] = experts[k];
    }
  }
}

// Count tokens per expert (host version for verification)
static void count_tokens_host(const int* topk_i, int* counts, int B, int K, int E) {
  std::fill(counts, counts + E, 0);
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < K; ++k) {
      int expert = topk_i[b * K + k];
      counts[expert]++;
    }
  }
}

// Build expert offsets (exclusive prefix sum)
static int build_offsets(const int* counts, int* offsets, int E) {
  int total = 0;
  for (int e = 0; e < E; ++e) {
    offsets[e] = total;
    total += counts[e];
  }
  return total;
}

// Verification function
static bool verify_routing(
    const float* expert_input, const float* input_x,
    const int* topk_i, const float* topk_v,
    const int* local_ids, const float* local_wts,
    const int* expert_offsets, const int* expert_counts,
    int B, int H, int K, int E, float tol = 1e-4f) {

  bool correct = true;

  // Verify each token's routing
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < K; ++k) {
      int idx = b * K + k;
      int expert = topk_i[idx];
      float weight = topk_v[idx];
      int compact_idx = local_ids[idx];

      // Check weight is correctly stored
      if (std::abs(local_wts[idx] - weight) > tol) {
        std::cerr << "Weight mismatch at token " << b << ", k=" << k
                  << ": expected " << weight << ", got " << local_wts[idx] << std::endl;
        correct = false;
      }

      // Check compact_idx is within expert's range
      int expert_start = expert_offsets[expert];
      int expert_end = expert_start + expert_counts[expert];
      if (compact_idx < expert_start || compact_idx >= expert_end) {
        std::cerr << "Compact index out of range for token " << b << ", k=" << k
                  << ": idx=" << compact_idx << ", range=[" << expert_start
                  << "," << expert_end << ")" << std::endl;
        correct = false;
      }

      // Check that input was copied correctly (sample a few dimensions)
      const float* src = input_x + (size_t)b * H;
      const float* dst = expert_input + (size_t)compact_idx * H;
      for (int h = 0; h < std::min(10, H); h += 1) {
        if (std::abs(src[h] - dst[h]) > tol) {
          std::cerr << "Data copy mismatch at token " << b << ", k=" << k
                    << ", dim " << h << ": expected " << src[h]
                    << ", got " << dst[h] << std::endl;
          correct = false;
          break;
        }
      }
    }
  }

  return correct;
}

int main(int argc, char** argv) {
  // Default parameters matching real MoE usage
  int batch_size = get_arg<int>(argc, argv, "--batch", 1024);
  int hidden_dim = get_arg<int>(argc, argv, "--hidden", 2880);  // Model hidden dimension
  int n_experts = get_arg<int>(argc, argv, "--experts", 128);  // 120B model has 128 experts
  int experts_per_token = get_arg<int>(argc, argv, "--topk", 4);
  int iters = get_arg<int>(argc, argv, "--iters", 20);
  uint64_t seed = (uint64_t)get_arg<long long>(argc, argv, "--seed", 42);
  bool verify = get_arg<int>(argc, argv, "--verify", 1);

  // For TP simulation
  const int TP = TENSOR_PARALLEL_SIZE;
  const int Bgrp = TP * batch_size;  // Union batch size across TP group

  std::cout << "=== route_and_pack_fused_kernel_optimized Benchmark ===\n";
  std::cout << "Batch (union): " << Bgrp << " (local: " << batch_size << " × TP: " << TP << ")\n";
  std::cout << "Hidden dim: " << hidden_dim << "\n";
  std::cout << "Experts: " << n_experts << "\n";
  std::cout << "Experts per token: " << experts_per_token << "\n";
  std::cout << "Iterations: " << iters << "\n\n";

  // Allocate host memory
  std::vector<float> h_x(Bgrp * hidden_dim);
  std::vector<int> h_topk_i(Bgrp * experts_per_token);
  std::vector<float> h_topk_v(Bgrp * experts_per_token);
  std::vector<int> h_expert_counts(n_experts);
  std::vector<int> h_expert_offsets(n_experts);
  std::vector<int> h_local_ids(Bgrp * experts_per_token);
  std::vector<float> h_local_wts(Bgrp * experts_per_token);

  // Initialize input data
  fill_uniform_float(h_x.data(), Bgrp * hidden_dim, seed, -1.0f, 1.0f);
  generate_topk_indices(h_topk_i.data(), Bgrp, experts_per_token, n_experts, seed + 1);
  fill_uniform_float(h_topk_v.data(), Bgrp * experts_per_token, seed + 2, 0.0f, 1.0f);

  // Normalize topk_v (simulate softmax output)
  for (int b = 0; b < Bgrp; ++b) {
    float sum = 0.0f;
    for (int k = 0; k < experts_per_token; ++k) {
      sum += h_topk_v[b * experts_per_token + k];
    }
    if (sum > 0) {
      for (int k = 0; k < experts_per_token; ++k) {
        h_topk_v[b * experts_per_token + k] /= sum;
      }
    }
  }

  // Count tokens and build offsets
  count_tokens_host(h_topk_i.data(), h_expert_counts.data(), Bgrp, experts_per_token, n_experts);
  int total_tokens = build_offsets(h_expert_counts.data(), h_expert_offsets.data(), n_experts);

  std::cout << "Total routed tokens: " << total_tokens << " (avg per expert: "
            << (float)total_tokens / n_experts << ")\n\n";

  // Allocate device memory
  float *d_x = nullptr;
  int *d_topk_i = nullptr;
  float *d_topk_v = nullptr;
  int *d_expert_offsets = nullptr;
  int *d_local_ids = nullptr;
  float *d_local_wts = nullptr;
  float *d_expert_input = nullptr;

  HIP_CHECK(hipMalloc(&d_x, Bgrp * hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_i, Bgrp * experts_per_token * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_topk_v, Bgrp * experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_expert_offsets, n_experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_local_ids, Bgrp * experts_per_token * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_local_wts, Bgrp * experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_expert_input, total_tokens * hidden_dim * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), Bgrp * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_topk_i, h_topk_i.data(), Bgrp * experts_per_token * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_topk_v, h_topk_v.data(), Bgrp * experts_per_token * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_expert_offsets, h_expert_offsets.data(), n_experts * sizeof(int), hipMemcpyHostToDevice));

  // Clear output buffers
  HIP_CHECK(hipMemset(d_local_ids, 0, Bgrp * experts_per_token * sizeof(int)));
  HIP_CHECK(hipMemset(d_local_wts, 0, Bgrp * experts_per_token * sizeof(float)));
  HIP_CHECK(hipMemset(d_expert_input, 0, total_tokens * hidden_dim * sizeof(float)));

  // Create stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Kernel configuration (matching moe_gpu function)
  int threads = THREADS_PER_BLOCK;
  while (threads < Bgrp) threads <<= 1;
  threads = std::min(threads, 1024);

  size_t shmem = (size_t)threads * 5 * sizeof(int) + sizeof(int);

  std::cout << "Kernel configuration:\n";
  std::cout << "  Grid: " << n_experts << " blocks\n";
  std::cout << "  Block: " << threads << " threads\n";
  std::cout << "  Shared memory: " << shmem << " bytes\n\n";

  // Warmup
  for (int i = 0; i < 3; ++i) {
    route_and_pack_fused_kernel_optimized<<<n_experts, threads, shmem, stream>>>(
        d_x, Bgrp, hidden_dim,
        d_topk_i, d_topk_v, experts_per_token,
        d_expert_offsets, n_experts,
        d_local_ids, d_local_wts,
        d_expert_input);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGetLastError());

  // Benchmark
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    route_and_pack_fused_kernel_optimized<<<n_experts, threads, shmem, stream>>>(
        d_x, Bgrp, hidden_dim,
        d_topk_i, d_topk_v, experts_per_token,
        d_expert_offsets, n_experts,
        d_local_ids, d_local_wts,
        d_expert_input);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / iters;

  // Calculate throughput
  size_t bytes_read = Bgrp * hidden_dim * sizeof(float) +  // input x
                      Bgrp * experts_per_token * (sizeof(int) + sizeof(float)) +  // topk
                      n_experts * sizeof(int);  // offsets
  size_t bytes_written = Bgrp * experts_per_token * (sizeof(int) + sizeof(float)) +  // local_ids, local_wts
                         total_tokens * hidden_dim * sizeof(float);  // expert_input
  size_t total_bytes = bytes_read + bytes_written;
  float bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

  std::cout << "--- Performance Results ---\n";
  std::cout << "Average kernel time: " << avg_ms << " ms\n";
  std::cout << "Throughput: " << (Bgrp * 1000.0 / avg_ms) << " tokens/sec\n";
  std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s\n\n";

  // Verification (optional)
  if (verify && total_tokens > 0) {
    std::cout << "Verifying correctness...\n";

    // Copy results back
    std::vector<float> h_expert_input(total_tokens * hidden_dim);
    HIP_CHECK(hipMemcpy(h_local_ids.data(), d_local_ids, Bgrp * experts_per_token * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_local_wts.data(), d_local_wts, Bgrp * experts_per_token * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_input.data(), d_expert_input, total_tokens * hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

    bool correct = verify_routing(
        h_expert_input.data(), h_x.data(),
        h_topk_i.data(), h_topk_v.data(),
        h_local_ids.data(), h_local_wts.data(),
        h_expert_offsets.data(), h_expert_counts.data(),
        Bgrp, hidden_dim, experts_per_token, n_experts);

    if (correct) {
      std::cout << "✓ Verification PASSED\n";
    } else {
      std::cout << "✗ Verification FAILED\n";
    }
  }

  // Cleanup
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_topk_i));
  HIP_CHECK(hipFree(d_topk_v));
  HIP_CHECK(hipFree(d_expert_offsets));
  HIP_CHECK(hipFree(d_local_ids));
  HIP_CHECK(hipFree(d_local_wts));
  HIP_CHECK(hipFree(d_expert_input));

  std::cout << "\nDone.\n";
  return 0;
}