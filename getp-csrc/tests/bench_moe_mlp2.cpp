// bench_moe_mlp2.cpp
// Benchmark the legacy fp32 MLP2 path vs the bf16 grouped kernel
// Uses the MoE microkernel path from moe.hpp targeting grouped_mlp2_bf16_bias_kernel_tiled_optimized

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

#include "../utils.hpp"
#include "../kernels/matmul.hpp"
#include "../kernels/moe.hpp"  // provides mlp2<...>, mlp2_optimized<...>, BLOCK_M_MLP, etc.

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)
#endif

struct SimConfig {
  int num_hidden_layers = 24;
  int num_experts       = 32;   // E
  int experts_per_token = 4;    // K_topk (routing)
  int vocab_size        = 201088;
  int hidden_size       = 2880; // H
  int intermediate_size = 2880; // D
  float swiglu_limit    = 7.0f;
  int head_dim          = 64;
  int num_attention_heads = 64;
  int num_key_value_heads = 8;
  int sliding_window      = 128;
  int initial_context_length = 4096;
  int rope_theta = 150000;
  float rope_scaling_factor = 32.0f;
  int rope_ntk_alpha = 1;
  int rope_ntk_beta  = 32;
  int BATCH_SIZE      = 1024;
  int _MAX_SEQ_LEN    = 1024;
  int TP              = 2;
};

static inline float frand(std::mt19937 &rng) {
  static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  return dist(rng);
}

static inline __hip_bfloat16 f2bf16(float x) {
#if defined(__HIP_PLATFORM_AMD__)
  return __float2bfloat16(x);
#else
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  uint32_t rounding_bias = ((u >> 16) & 1u) + 0x7FFFu;
  u += rounding_bias;
  uint16_t b = static_cast<uint16_t>(u >> 16);
  __hip_bfloat16 out;
  std::memcpy(&out, &b, sizeof(b));
  return out;
#endif
}

struct TuneResult {
  const char *name;
  float avg_ms;
  double gflops;
  double speedup;
  double max_abs;
  double max_rel;
};

static float time_baseline_kernel(
    float *dC,
    const float *dA,
    const __hip_bfloat16 *dW2,
    const __hip_bfloat16 *db2,
    const int *d_offsets,
    const int *d_counts,
    const int *d_t2e,
    const int *d_t2l,
    int E, int K, int N, int cur_tiles,
    hipStream_t stream,
    hipEvent_t ev_start,
    hipEvent_t ev_stop,
    int warmup, int iters)
{
  for (int i = 0; i < warmup; ++i) {
    mlp2<16,16,16, 4,4,4, 1,2, 4>(
        dC, dA, dW2, db2,
        d_offsets, d_counts,
        d_t2e, d_t2l,
        E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventRecord(ev_start, stream));
  for (int i = 0; i < iters; ++i) {
    mlp2<16,16,16, 4,4,4, 1,2, 4>(
        dC, dA, dW2, db2,
        d_offsets, d_counts,
        d_t2e, d_t2l,
        E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipEventRecord(ev_stop, stream));
  HIP_CHECK(hipEventSynchronize(ev_stop));

  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
  HIP_CHECK(hipGetLastError());
  return ms / static_cast<float>(iters);
}

template <
    int WM, int WN, int WK,
    int WAVES_M, int WAVES_N, int WAVES_K,
    int TW_M, int TW_N,
    int PAD_K>
static float time_optimized_variant(
    float *dC,
    const __hip_bfloat16 *dA,
    const __hip_bfloat16 *dW2,
    const __hip_bfloat16 *db2,
    const int *d_offsets,
    const int *d_counts,
    const int *d_t2e,
    const int *d_t2l,
    int E, int K, int N, int cur_tiles,
    hipStream_t stream,
    hipEvent_t ev_start,
    hipEvent_t ev_stop,
    int warmup, int iters)
{
  for (int i = 0; i < warmup; ++i) {
    mlp2_optimized<WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>(
        dC, dA, dW2, db2,
        d_offsets, d_counts,
        d_t2e, d_t2l,
        E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventRecord(ev_start, stream));
  for (int i = 0; i < iters; ++i) {
    mlp2_optimized<WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>(
        dC, dA, dW2, db2,
        d_offsets, d_counts,
        d_t2e, d_t2l,
        E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipEventRecord(ev_stop, stream));
  HIP_CHECK(hipEventSynchronize(ev_stop));

  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
  HIP_CHECK(hipGetLastError());
  return ms / static_cast<float>(iters);
}

template <
    int WM, int WN, int WK,
    int WAVES_M, int WAVES_N, int WAVES_K,
    int TW_M, int TW_N,
    int PAD_K>
static TuneResult benchmark_variant(
    const char *name,
    float *d_C_opt,
    const __hip_bfloat16 *d_A_bf16,
    const __hip_bfloat16 *d_W2,
    const __hip_bfloat16 *d_b2,
    const int *d_offsets,
    const int *d_counts,
    const int *d_t2e,
    const int *d_t2l,
    int E, int K, int N, int cur_tiles,
    hipStream_t stream,
    hipEvent_t ev_start,
    hipEvent_t ev_stop,
    int warmup, int iters,
    std::vector<float> &h_C_opt,
    const std::vector<float> &h_C_ref,
    double flops,
    float baseline_ms,
    size_t bytes_out)
{
  TuneResult result{name, 0.f, 0.0, 0.0, 0.0, 0.0};

  HIP_CHECK(hipMemset(d_C_opt, 0, bytes_out));
  mlp2_optimized<WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>(
      d_C_opt, d_A_bf16, d_W2, d_b2,
      d_offsets, d_counts,
      d_t2e, d_t2l,
      E, K, N, cur_tiles, stream);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(h_C_opt.data(), d_C_opt, bytes_out, hipMemcpyDeviceToHost));

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < h_C_opt.size(); ++i) {
    const double ref = static_cast<double>(h_C_ref[i]);
    const double got = static_cast<double>(h_C_opt[i]);
    const double abs_diff = std::abs(ref - got);
    const double rel_diff = abs_diff / (std::abs(ref) + 1e-7);
    max_abs = std::max(max_abs, abs_diff);
    max_rel = std::max(max_rel, rel_diff);
  }

  const float ms = time_optimized_variant<
      WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>(
          d_C_opt, d_A_bf16, d_W2, d_b2,
          d_offsets, d_counts,
          d_t2e, d_t2l,
          E, K, N, cur_tiles,
          stream, ev_start, ev_stop,
          warmup, iters);

  const double gflops = (flops * 1e-9) / (ms * 1e-3);

  result.avg_ms = ms;
  result.gflops = gflops;
  result.speedup = baseline_ms / ms;
  result.max_abs = max_abs;
  result.max_rel = max_rel;
  return result;
}

int main(int argc, char** argv)
{
  int iters = 50;
  int warmup = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
  }

  SimConfig cfg;
  const int E        = cfg.num_experts;
  const int out_dim  = cfg.hidden_size;       // N dimension of GEMM
  const int in_dim   = cfg.intermediate_size / cfg.TP; // K dimension of GEMM
  const int Bgrp     = cfg.TP * cfg.BATCH_SIZE;
  const int K_topk   = cfg.experts_per_token;
  const int total_pairs = Bgrp * K_topk;
  const size_t out_elems = static_cast<size_t>(total_pairs) * static_cast<size_t>(out_dim);
  const size_t out_bytes = sizeof(float) * out_elems;

  printf("=== MoE MLP2 Benchmark ===\n");
  printf("E=%d, in_dim=%d, out_dim=%d, Bgrp=%d, K_topk=%d, total_pairs=%d\n",
         E, in_dim, out_dim, Bgrp, K_topk, total_pairs);
  printf("BLOCK_M_MLP=%d (from moe.hpp), expected tiles/expert ~ %d\n",
         BLOCK_M_MLP, (total_pairs / E + BLOCK_M_MLP - 1) / BLOCK_M_MLP);

  std::vector<int> h_expert_counts(E, total_pairs / E);
  int rem = total_pairs - (total_pairs / E) * E;
  for (int i = 0; i < rem; ++i) h_expert_counts[i]++;

  std::vector<int> h_expert_offsets(E, 0);
  int running = 0;
  for (int e = 0; e < E; ++e) { h_expert_offsets[e] = running; running += h_expert_counts[e]; }

  int cur_tiles = 0;
  for (int e = 0; e < E; ++e) {
    int tiles = (h_expert_counts[e] + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
    cur_tiles += tiles;
  }
  std::vector<int> h_tile2expert(cur_tiles), h_tile2local(cur_tiles);
  {
    int cursor = 0;
    for (int e = 0; e < E; ++e) {
      int tiles = (h_expert_counts[e] + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
      for (int t = 0; t < tiles; ++t) {
        h_tile2expert[cursor + t] = e;
        h_tile2local[cursor + t]  = t;
      }
      cursor += tiles;
    }
  }

  std::mt19937 rng(123);
  std::vector<float> h_A(static_cast<size_t>(total_pairs) * in_dim);
  std::vector<__hip_bfloat16> h_A_bf16(static_cast<size_t>(total_pairs) * in_dim);
  std::vector<__hip_bfloat16> h_W2(static_cast<size_t>(E) * out_dim * in_dim);
  std::vector<__hip_bfloat16> h_b2(static_cast<size_t>(E) * out_dim);
  std::vector<float> h_C_ref(out_elems, 0.f);
  std::vector<float> h_C_opt(out_elems, 0.f);

  for (size_t i = 0; i < h_A.size(); ++i) {
    float v = frand(rng);
    h_A[i] = v;
    h_A_bf16[i] = f2bf16(v);
  }
  for (size_t e = 0; e < static_cast<size_t>(E); ++e) {
    for (size_t i = 0; i < static_cast<size_t>(out_dim) * in_dim; ++i) {
      float w = frand(rng);
      h_W2[e * static_cast<size_t>(out_dim) * in_dim + i] = f2bf16(w);
    }
    for (int i = 0; i < out_dim; ++i) {
      float b = frand(rng);
      h_b2[e * out_dim + i] = f2bf16(b);
    }
  }

  float *d_A_fp32 = nullptr, *d_C_ref = nullptr, *d_C_opt = nullptr;
  __hip_bfloat16 *d_A_bf16 = nullptr, *d_W2 = nullptr, *d_b2 = nullptr;
  int *d_counts = nullptr, *d_offsets = nullptr, *d_t2e = nullptr, *d_t2l = nullptr;

  HIP_CHECK(hipMalloc(&d_A_fp32, sizeof(float) * static_cast<size_t>(total_pairs) * in_dim));
  HIP_CHECK(hipMalloc(&d_A_bf16, sizeof(__hip_bfloat16) * static_cast<size_t>(total_pairs) * in_dim));
  HIP_CHECK(hipMalloc(&d_C_ref, out_bytes));
  HIP_CHECK(hipMalloc(&d_C_opt, out_bytes));
  HIP_CHECK(hipMalloc(&d_W2, sizeof(__hip_bfloat16) * static_cast<size_t>(E) * out_dim * in_dim));
  HIP_CHECK(hipMalloc(&d_b2, sizeof(__hip_bfloat16) * static_cast<size_t>(E) * out_dim));
  HIP_CHECK(hipMalloc(&d_counts, sizeof(int) * E));
  HIP_CHECK(hipMalloc(&d_offsets, sizeof(int) * E));
  HIP_CHECK(hipMalloc(&d_t2e, sizeof(int) * cur_tiles));
  HIP_CHECK(hipMalloc(&d_t2l, sizeof(int) * cur_tiles));

  HIP_CHECK(hipMemcpy(d_A_fp32, h_A.data(), sizeof(float) * static_cast<size_t>(total_pairs) * in_dim, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_A_bf16, h_A_bf16.data(), sizeof(__hip_bfloat16) * static_cast<size_t>(total_pairs) * in_dim, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W2, h_W2.data(), sizeof(__hip_bfloat16) * static_cast<size_t>(E) * out_dim * in_dim, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b2, h_b2.data(), sizeof(__hip_bfloat16) * static_cast<size_t>(E) * out_dim, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_counts, h_expert_counts.data(), sizeof(int) * E, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_offsets, h_expert_offsets.data(), sizeof(int) * E, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_t2e, h_tile2expert.data(), sizeof(int) * cur_tiles, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_t2l, h_tile2local.data(), sizeof(int) * cur_tiles, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemset(d_C_ref, 0, out_bytes));
  HIP_CHECK(hipMemset(d_C_opt, 0, out_bytes));

  mlp2<16,16,16, 4,4,4, 1,2, 4>(
      d_C_ref, d_A_fp32, d_W2, d_b2,
      d_offsets, d_counts,
      d_t2e, d_t2l,
      E, in_dim, out_dim, cur_tiles, stream);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(h_C_ref.data(), d_C_ref, out_bytes, hipMemcpyDeviceToHost));

  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  const float t_base_ms = time_baseline_kernel(
      d_C_ref, d_A_fp32, d_W2, d_b2,
      d_offsets, d_counts, d_t2e, d_t2l,
      E, in_dim, out_dim, cur_tiles,
      stream, ev_start, ev_stop,
      warmup, iters);

  double flops = 0.0;
  for (int e = 0; e < E; ++e) {
    flops += 2.0 * static_cast<double>(h_expert_counts[e]) * static_cast<double>(in_dim) * static_cast<double>(out_dim);
  }
  auto gflops = [&](float ms_per_iter) {
    return (flops * 1e-9) / (ms_per_iter * 1e-3);
  };

  const double base_gflops = gflops(t_base_ms);

  std::vector<TuneResult> tune_results;
  tune_results.reserve(8);

#define RUN_VARIANT(NAME, WM, WN, WK, WMV, WNV, WKV, TWM, TWN, PADK) \
  do { \
    tune_results.push_back(benchmark_variant<WM, WN, WK, WMV, WNV, WKV, TWM, TWN, PADK>( \
        NAME, \
        d_C_opt, d_A_bf16, d_W2, d_b2, \
        d_offsets, d_counts, \
        d_t2e, d_t2l, \
        E, in_dim, out_dim, cur_tiles, \
        stream, ev_start, ev_stop, \
        warmup, iters, \
        h_C_opt, h_C_ref, \
        flops, t_base_ms, \
        out_bytes)); \
  } while (0)

  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 4);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=2", 16,16,16, 4,8,4, 1,2, 2);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=0", 16,16,16, 4,8,4, 1,2, 0);
  // RUN_VARIANT("waves=8x4x4 tw=1x2 pad=4", 16,16,16, 8,4,4, 2,1, 4);
  // RUN_VARIANT("waves=2x16x4 tw=1x2 pad=4", 16,16,16, 2,16,4, 1,, 4);
  // RUN_VARIANT("waves=4x8x2 tw=1x2 pad=4", 16,16,16, 4,8,2, 1,2, 4);
  // RUN_VARIANT("waves=4x8x4 tw=2x1 pad=4", 16,16,16, 4,8,4, 2,1, 4);
  // RUN_VARIANT("waves=4x8x4 tw=1x1 pad=4", 16,16,16, 4,8,4, 1,1, 4);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=4", 16,16,16, 4,4,4, 1,2, 4);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=4", 16,16,16, 4,4,4, 1,2, 6);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=4", 16,16,16, 4,4,4, 1,2, 4);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=4", 16,16,16, 4,4,4, 1,2, 8);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 4);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 6);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 4);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 8);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 8);
  // RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 4);





#undef RUN_VARIANT

  const TuneResult *best = tune_results.empty() ? nullptr : &tune_results.front();
  if (best) {
    for (const auto &res : tune_results) {
      if (res.avg_ms < best->avg_ms) best = &res;
    }
  }

  printf("\n--- Results ---\n");
  printf("Baseline (mlp2 fp32):    %.3f ms  |  %.2f GFLOP/s\n", t_base_ms, base_gflops);

  if (!tune_results.empty()) {
    printf("\n--- Optimized Kernel Variants ---\n");
    for (const auto &res : tune_results) {
      const bool ok = (res.max_abs < 1e-4) && (res.max_rel < 1e-6);
      printf("%-32s  %.3f ms  |  %.2f GFLOP/s  |  speedup %.3fx  |  max_abs=%.3g  max_rel=%.3g  %s\n",
             res.name,
             res.avg_ms,
             res.gflops,
             res.speedup,
             res.max_abs,
             res.max_rel,
             ok ? "OK" : "WARN");
    }
    if (best) {
      printf("\nBest config: %s (%.3f ms, speedup %.3fx)\n",
             best->name,
             best->avg_ms,
             best->speedup);
    }
  }

  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_A_fp32));
  HIP_CHECK(hipFree(d_A_bf16));
  HIP_CHECK(hipFree(d_C_ref));
  HIP_CHECK(hipFree(d_C_opt));
  HIP_CHECK(hipFree(d_W2));
  HIP_CHECK(hipFree(d_b2));
  HIP_CHECK(hipFree(d_counts));
  HIP_CHECK(hipFree(d_offsets));
  HIP_CHECK(hipFree(d_t2e));
  HIP_CHECK(hipFree(d_t2l));

  return 0;
}
