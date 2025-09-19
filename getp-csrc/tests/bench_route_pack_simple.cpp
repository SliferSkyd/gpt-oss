// bench_route_pack_simple.cpp
// Simple benchmark for route_and_pack_fused_kernel
// Extracts just the kernel and benchmarks it in isolation

#include <hip/hip_runtime.h>
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

// Error handling
#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    std::cerr << "HIP error " << hipGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// Extracted route_and_pack_fused_kernel from moe.hpp
__global__ void route_and_pack_fused_kernel(
    const float *__restrict__ x, // [B, H]
    int B, int H,
    const int *__restrict__ topk_i,   // [B, K]
    const float *__restrict__ topk_v, // [B, K]
    int K,
    const int *__restrict__ expert_offsets, // [E] (exclusive prefix)
    int E,
    int *__restrict__ local_ids,   // [B, K] (out)
    float *__restrict__ local_wts, // [B, K] (out)
    float *__restrict__ expert_in) // [sum_tokens, H] (out)
{
    const int e = blockIdx.x;
    if (e >= E)
        return;

    const int T = blockDim.x; // threads per block
    const int tid = threadIdx.x;

    extern __shared__ int smem[];
    int *flags = smem;               // [T]
    int *excl = flags + T;           // [T]  (inclusive scan buffer)
    int *kidx = excl + T;            // [T]  (which k matched e, or -1)
    int *toklist = kidx + T;         // [T]  (selected token indices in this chunk)
    int *cmplist = toklist + T;      // [T]  (their compact indices)
    int *carry_shared = cmplist + T; // [1]  (shared carry value)

    if (tid == 0)
        carry_shared[0] = 0; // Initialize shared carry
    __syncthreads();

    // Process tokens in tiles of T to support B > T
    for (int base = 0; base < B; base += T)
    {

        // ---- 1) Flag tokens in this chunk and remember which k matched ----
        int t_global = base + tid;
        int f = 0, kk = -1;
        if (t_global < B)
        {
            const int off = t_global * K;
#pragma unroll
            for (int i = 0; i < K; ++i)
            {
                if (topk_i[off + i] == e)
                {
                    f = 1;
                    kk = i;
                    break;
                }
            }
        }
        flags[tid] = f;
        kidx[tid] = kk;
        __syncthreads();

        // ---- 2) Inclusive scan on flags (Hillis–Steele in-place) ----
        excl[tid] = flags[tid];
        __syncthreads();
        for (int ofs = 1; ofs < T; ofs <<= 1)
        {
            int v = (tid >= ofs) ? excl[tid - ofs] : 0;
            __syncthreads();
            excl[tid] += v;
            __syncthreads();
        }

        // ---- 3) Build compacted lists of selected tokens and compute their compact positions ----
        const int chunkCnt = excl[T - 1];
        const int prevCarry = carry_shared[0];

        if (flags[tid] && (excl[tid] - 1) < T)
        {
            toklist[excl[tid] - 1] = t_global;
            cmplist[excl[tid] - 1] = expert_offsets[e] + prevCarry + excl[tid] - 1;
        }
        __syncthreads();

        // ---- 4) Copy data for selected tokens ----
        for (int s = 0; s < chunkCnt; ++s)
        {
            int token_idx = toklist[s];
            int compact_idx = cmplist[s];
            int k_for_token = kidx[token_idx - base];

            // Store routing metadata
            if (tid == 0)
            {
                local_ids[token_idx * K + k_for_token] = compact_idx;
                local_wts[token_idx * K + k_for_token] = topk_v[token_idx * K + k_for_token];
            }

            // Coalesced copy of hidden vector
            const float *src = x + (size_t)token_idx * H;
            float *dst = expert_in + (size_t)compact_idx * H;
            for (int h = tid; h < H; h += T)
            {
                dst[h] = src[h];
            }
        }
        __syncthreads();

        // Update carry for next tile
        if (tid == 0)
            carry_shared[0] = prevCarry + chunkCnt;
        __syncthreads();
    }
}

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

// Check if a flag exists (for boolean flags without values)
static bool has_flag(int argc, char** argv, const char* key) {
  for (int i=1;i<argc;++i) {
    if (std::string(argv[i])==key) {
      return true;
    }
  }
  return false;
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

// Count tokens per expert (host version)
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

int main(int argc, char** argv) {
  // Default parameters matching real MoE usage
  int batch_size = get_arg<int>(argc, argv, "--batch", 2048);  // Union batch size
  int hidden_dim = get_arg<int>(argc, argv, "--hidden", 2880);
  int n_experts = get_arg<int>(argc, argv, "--experts", 128);
  int experts_per_token = get_arg<int>(argc, argv, "--topk", 4);
  int iters = get_arg<int>(argc, argv, "--iters", 20);
  uint64_t seed = (uint64_t)get_arg<long long>(argc, argv, "--seed", 42);
  // Check for --verify flag (with or without value)
  bool verify = has_flag(argc, argv, "--verify") || get_arg<int>(argc, argv, "--verify", 0);

  std::cout << "=== route_and_pack_fused_kernel Benchmark ===\n";
  std::cout << "Batch size: " << batch_size << "\n";
  std::cout << "Hidden dim: " << hidden_dim << "\n";
  std::cout << "Experts: " << n_experts << "\n";
  std::cout << "Experts per token: " << experts_per_token << "\n";
  std::cout << "Iterations: " << iters << "\n\n";

  // Allocate host memory
  std::vector<float> h_x(batch_size * hidden_dim);
  std::vector<int> h_topk_i(batch_size * experts_per_token);
  std::vector<float> h_topk_v(batch_size * experts_per_token);
  std::vector<int> h_expert_counts(n_experts);
  std::vector<int> h_expert_offsets(n_experts);

  // Initialize input data
  fill_uniform_float(h_x.data(), batch_size * hidden_dim, seed, -1.0f, 1.0f);
  generate_topk_indices(h_topk_i.data(), batch_size, experts_per_token, n_experts, seed + 1);
  fill_uniform_float(h_topk_v.data(), batch_size * experts_per_token, seed + 2, 0.0f, 1.0f);

  // Normalize topk_v (simulate softmax output)
  for (int b = 0; b < batch_size; ++b) {
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
  count_tokens_host(h_topk_i.data(), h_expert_counts.data(), batch_size, experts_per_token, n_experts);
  int total_tokens = build_offsets(h_expert_counts.data(), h_expert_offsets.data(), n_experts);

  std::cout << "Total routed tokens: " << total_tokens << "\n";
  std::cout << "Average tokens per expert: " << (float)total_tokens / n_experts << "\n\n";

  // Allocate device memory
  float *d_x = nullptr;
  int *d_topk_i = nullptr;
  float *d_topk_v = nullptr;
  int *d_expert_offsets = nullptr;
  int *d_local_ids = nullptr;
  float *d_local_wts = nullptr;
  float *d_expert_input = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch_size * hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_topk_i, batch_size * experts_per_token * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_topk_v, batch_size * experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_expert_offsets, n_experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_local_ids, batch_size * experts_per_token * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_local_wts, batch_size * experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_expert_input, total_tokens * hidden_dim * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_topk_i, h_topk_i.data(), batch_size * experts_per_token * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_topk_v, h_topk_v.data(), batch_size * experts_per_token * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_expert_offsets, h_expert_offsets.data(), n_experts * sizeof(int), hipMemcpyHostToDevice));

  // Create stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Kernel configuration
  int threads = 256;
  while (threads < batch_size && threads < 1024) threads <<= 1;
  threads = std::min(threads, 1024);

  size_t shmem = (size_t)threads * 5 * sizeof(int) + sizeof(int);

  std::cout << "Kernel configuration:\n";
  std::cout << "  Grid: " << n_experts << " blocks\n";
  std::cout << "  Block: " << threads << " threads\n";
  std::cout << "  Shared memory: " << shmem << " bytes\n\n";

  // Warmup
  for (int i = 0; i < 3; ++i) {
    route_and_pack_fused_kernel<<<n_experts, threads, shmem, stream>>>(
        d_x, batch_size, hidden_dim,
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
    route_and_pack_fused_kernel<<<n_experts, threads, shmem, stream>>>(
        d_x, batch_size, hidden_dim,
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
  size_t bytes_read = batch_size * hidden_dim * sizeof(float) +  // input x
                      batch_size * experts_per_token * (sizeof(int) + sizeof(float)) +  // topk
                      n_experts * sizeof(int);  // offsets
  size_t bytes_written = batch_size * experts_per_token * (sizeof(int) + sizeof(float)) +  // local_ids, local_wts
                         total_tokens * hidden_dim * sizeof(float);  // expert_input
  size_t total_bytes = bytes_read + bytes_written;
  float bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

  std::cout << "--- Performance Results ---\n";
  std::cout << "Average kernel time: " << avg_ms << " ms\n";
  std::cout << "Throughput: " << (batch_size * 1000.0 / avg_ms) << " tokens/sec\n";
  std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s\n";
  std::cout << "Tokens routed per second: " << (total_tokens * 1000.0 / avg_ms) << "\n\n";

  // Verification (optional)
  if (verify && total_tokens > 0) {
    std::cout << "Verifying correctness...\n";

    // Allocate host memory for results
    std::vector<int> h_local_ids(batch_size * experts_per_token);
    std::vector<float> h_local_wts(batch_size * experts_per_token);
    std::vector<float> h_expert_input(total_tokens * hidden_dim);

    // Copy results back
    HIP_CHECK(hipMemcpy(h_local_ids.data(), d_local_ids,
                        batch_size * experts_per_token * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_local_wts.data(), d_local_wts,
                        batch_size * experts_per_token * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_expert_input.data(), d_expert_input,
                        total_tokens * hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

    bool correct = true;
    int errors = 0;
    const int max_errors = 10;
    const float tol = 1e-4f;

    // Verify each token's routing
    for (int b = 0; b < batch_size && errors < max_errors; ++b) {
      for (int k = 0; k < experts_per_token && errors < max_errors; ++k) {
        int idx = b * experts_per_token + k;
        int expert = h_topk_i[idx];
        float weight = h_topk_v[idx];
        int compact_idx = h_local_ids[idx];

        // Check weight is correctly stored
        if (std::abs(h_local_wts[idx] - weight) > tol) {
          std::cerr << "Weight mismatch at token " << b << ", k=" << k
                    << ": expected " << weight << ", got " << h_local_wts[idx] << std::endl;
          correct = false;
          errors++;
        }

        // Check compact_idx is within expert's range
        int expert_start = h_expert_offsets[expert];
        int expert_end = expert_start + h_expert_counts[expert];
        if (compact_idx < expert_start || compact_idx >= expert_end) {
          std::cerr << "Compact index out of range for token " << b << ", k=" << k
                    << ": idx=" << compact_idx << ", range=[" << expert_start
                    << "," << expert_end << ")" << std::endl;
          correct = false;
          errors++;
        }

        // Check that input was copied correctly (sample a few dimensions)
        const float* src = h_x.data() + (size_t)b * hidden_dim;
        const float* dst = h_expert_input.data() + (size_t)compact_idx * hidden_dim;
        for (int h = 0; h < std::min(10, hidden_dim); h += 1) {
          if (std::abs(src[h] - dst[h]) > tol) {
            std::cerr << "Data copy mismatch at token " << b << ", k=" << k
                      << ", dim " << h << ": expected " << src[h]
                      << ", got " << dst[h] << std::endl;
            correct = false;
            errors++;
            break;
          }
        }
      }
    }

    if (correct) {
      std::cout << "✓ Verification PASSED\n\n";
    } else {
      std::cout << "✗ Verification FAILED (found " << errors << " errors)\n\n";
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

  std::cout << "Done.\n";
  return 0;
}