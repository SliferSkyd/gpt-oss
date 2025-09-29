// bench_mlp1_smallM.cu
// Benchmark MLP1 kernel with grouped experts at model scale.
// Build this next to your existing sources (it #includes "moe.hpp").
//
// Usage (examples at bottom):
//   ./bench_mlp1_smallM [iters]
// Defaults: iters=50

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>

#ifndef VERSION_LABEL
#define VERSION_LABEL "unspecified"
#endif

// Your optimized launcher lives in moe.hpp
// It should declare:
// inline void mlp1_optimized_outbf16_optimized(__hip_bfloat16* C,
//                                    const __hip_bfloat16* A,
//                                    const __hip_bfloat16* W1,
//                                    const int* expert_offsets,
//                                    const int* expert_counts,
//                                    const int* tile2expert, const int* tile2local,
//                                    int E, int K, int N, int cur_tiles, hipStream_t s);
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include "../utils.hpp"
#include "../kernels/matmul.hpp"
#include "../kernels/moe.hpp"

// -------------------- Config (from user) --------------------
static constexpr int E_experts = 128;   // n_expert
static constexpr int N_out     = 1440;  // N
static constexpr int K_in      = 2880;  // K
static constexpr int BATCH     = 928;   // batch size

// Kernel tiling (must match the launcher template used in your code)
// static constexpr int WM = 16, WN = 16, WK = 16;
static constexpr int WAVES_M = 4, WAVES_N = 8, WAVES_K = 4;
static constexpr int TW_M = 1, TW_N = 2;
static constexpr int PAD_K_MC = 4;
// static constexpr int BLOCK_M_MLP = WM * WAVES_M; // 16*4 = 64

// -------------------- HIP helpers ---------------------------
#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__, hipGetErrorString(e), (int)e); \
    std::exit(1); \
  } \
} while(0)

// Simple device PRNG + fillers to avoid >1GB host staging
__device__ inline uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
__device__ inline float prng_uniform(uint64_t& s) {
    uint32_t r = (uint32_t)xorshift64(s);
    // Map to (-0.5,0.5)
    return (float)( (r / 4294967296.0f) - 0.5f );
}

__global__ void fill_rand_bf16(__hip_bfloat16* dst, size_t n, uint64_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    uint64_t state = (seed ^ (0x9E3779B97f4A7C15ull + i));
    for (size_t idx = i; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        float v = prng_uniform(state);
        dst[idx] = __float2bfloat16(v);
    }
}

__global__ void zero_bf16(__hip_bfloat16* dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (size_t idx = i; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
        reinterpret_cast<uint16_t*>(dst)[idx] = 0;
    }
}

// -------------------- Host utilities ------------------------
static void build_counts_offsets_and_tiles(
    int total_tokens_target,
    std::vector<int>& counts,
    std::vector<int>& offsets,
    std::vector<int>& tile2expert,
    std::vector<int>& tile2local,
    int& cur_tiles_out,
    int& total_tokens_out,
    int& small_expert_tokens_out)
{
    counts.assign(E_experts, 0);

    // Uniformly throw tokens into experts — mean ~ total/E (≈ 14.5) so many <16.
    for (int t = 0; t < total_tokens_target; ++t) {
        counts[rand() % E_experts] += 1;
    }

    // Compute offsets
    offsets.resize(E_experts);
    int running = 0;
    int small_tok = 0;
    for (int e = 0; e < E_experts; ++e) {
        offsets[e] = running;
        running += counts[e];
        if (counts[e] < 16) small_tok += counts[e];
    }
    total_tokens_out = running;
    small_expert_tokens_out = small_tok;

    // Tiles
    tile2expert.clear();
    tile2local.clear();
    int cur_tiles = 0;
    for (int e = 0; e < E_experts; ++e) {
        const int cnt = counts[e];
        const int tiles = (cnt + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
        for (int m = 0; m < tiles; ++m) {
            tile2expert.push_back(e);
            tile2local.push_back(m);
        }
        cur_tiles += tiles;
    }
    cur_tiles_out = cur_tiles;
}

static double elapsed_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return (double)ms;
}

// Compute total MACs for benchmarking: sum_e (2 * M_e * N * K)
static double total_flops_theoretical(const std::vector<int>& counts) {
    long long Msum = 0;
    for (int c : counts) Msum += c;
    return 2.0 * (double)Msum * (double)N_out * (double)K_in;
}

int main(int argc, char** argv) {
    int iters = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 50;

    // Make results reproducible-ish.
    srand(12345);

    // Simulate top-k=2 routing to create plenty of small-M experts
    const int total_tokens_target = BATCH * 2;

    std::vector<int> h_counts, h_offsets, h_tile2expert, h_tile2local;
    int cur_tiles = 0, total_tokens = 0, small_tok = 0;
    build_counts_offsets_and_tiles(total_tokens_target,
                                   h_counts, h_offsets,
                                   h_tile2expert, h_tile2local,
                                   cur_tiles, total_tokens, small_tok);

    const size_t A_elems   = (size_t)total_tokens * (size_t)K_in;
    const size_t C_elems   = (size_t)total_tokens * (size_t)N_out;
    const size_t W_elems   = (size_t)E_experts * (size_t)N_out * (size_t)K_in;

    std::cout << "=== MLP1 Benchmark (" << VERSION_LABEL << ") ===\n";
    std::cout << "E=" << E_experts << ", N=" << N_out << ", K=" << K_in
              << ", batch=" << BATCH << ", target_tokens=" << total_tokens_target << "\n";
    std::cout << "sum_tokens=" << total_tokens
              << " (small-expert tokens=" << small_tok << ", "
              << std::fixed << std::setprecision(1)
              << (100.0 * small_tok / std::max(1, total_tokens)) << "%)\n";
    std::cout << "cur_tiles=" << cur_tiles
              << " (BLOCK_M_MLP=" << BLOCK_M_MLP << ")\n";
    std::cout << "Alloc sizes: A=" << (A_elems*2/1e6) << " MB, "
              << "W=" << (W_elems*2/1e6) << " MB, "
              << "C=" << (C_elems*2/1e6) << " MB\n";

    // Device allocations
    __hip_bfloat16 *dA = nullptr, *dW = nullptr, *dC = nullptr;
    int *d_offsets = nullptr, *d_counts = nullptr, *d_t2e = nullptr, *d_t2l = nullptr;

    HIP_CHECK(hipMalloc(&dA, A_elems * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&dW, W_elems * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&dC, C_elems * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_offsets, E_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_counts,  E_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_t2e,     h_tile2expert.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_t2l,     h_tile2local.size() * sizeof(int)));

    // Upload meta
    HIP_CHECK(hipMemcpy(d_offsets, h_offsets.data(), E_experts * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_counts,  h_counts.data(),  E_experts * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_t2e,     h_tile2expert.data(), h_tile2expert.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_t2l,     h_tile2local.data(),  h_tile2local.size() * sizeof(int), hipMemcpyHostToDevice));

    // Fill A, W, zero C (device-side)
    const int T = 256;
    const int gridA = (int)std::min<size_t>((A_elems + T - 1) / T, 16*1024);
    const int gridW = (int)std::min<size_t>((W_elems + T - 1) / T, 16*1024);
    const int gridC = (int)std::min<size_t>((C_elems + T - 1) / T, 16*1024);

    fill_rand_bf16<<<gridA, T>>>(dA, A_elems, 0xA53F1234ull);
    fill_rand_bf16<<<gridW, T>>>(dW, W_elems, 0xBEEFCAFEull);
    zero_bf16     <<<gridC, T>>>(dC, C_elems);
    HIP_CHECK(hipDeviceSynchronize());

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        mlp1_optimized_outbf16_optimized<WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
            dC, dA, dW, d_offsets, d_counts, d_t2e, d_t2l,
            E_experts, K_in, N_out, cur_tiles, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // Time iters
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        mlp1_optimized_outbf16_optimized<WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
            dC, dA, dW, d_offsets, d_counts, d_t2e, d_t2l,
            E_experts, K_in, N_out, cur_tiles, stream);
    }
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    double ms = elapsed_ms(ev_start, ev_stop) / std::max(1, iters);
    double flops = total_flops_theoretical(h_counts);          // per call
    double tflops = (flops / 1e12) / (ms / 1e3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg time: " << ms << " ms     "
              << "Throughput: " << std::setprecision(2) << tflops << " TFLOP/s\n";

    // Cleanup
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(d_offsets));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_t2e));
    HIP_CHECK(hipFree(d_t2l));

    return 0;
}

/*
==========================
Build & compare (examples)
==========================

# 1) Baseline (OLD): build WITHOUT the small-M fast path.
#    Make sure moe.hpp / your kernel is compiled without the new path,
#    or compiled with it disabled (e.g., -DENABLE_SMALLM_FASTPATH=0).
hipcc -O3 -DNDEBUG -DVERSION_LABEL=\"old\" bench_mlp1_smallM.cu -o bench_old

# 2) Optimized (NEW): build WITH the small-M fast path enabled.
hipcc -O3 -DNDEBUG -DENABLE_SMALLM_FASTPATH=1 -DVERSION_LABEL=\"new\" bench_mlp1_smallM.cu -o bench_new

# Run (default 50 iters):
./bench_old
./bench_new

# Or specify iterations:
./bench_old 100
./bench_new 100

The program prints:
- sum_tokens and % that belong to small-count experts (<16)
- tiles used (cur_tiles)
- average kernel time and effective TFLOP/s

*/
