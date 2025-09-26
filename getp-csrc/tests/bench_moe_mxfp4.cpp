// bench_moe_mxfp4.cpp
// Benchmark grouped_mlp1_mxfp4_kernel_unified_A_bf16 against the fp32 MLP1 path

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
#include <cstdint>

#include "../utils.hpp"
#include "../kernels/matmul.hpp"
#include "../kernels/moe.hpp"

struct SimConfig {
  int num_hidden_layers = 24;
  int num_experts       = 32;
  int experts_per_token = 4;
  int vocab_size        = 201088;
  int hidden_size       = 2880;
  int intermediate_size = 2880;
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

static inline uint8_t enc_fp4_e2m1_nearest_host(float x) {
  uint8_t best = 0;
  float best_diff = std::fabs(x - MXFP4_LUT_CPU[0]);
  for (uint8_t i = 1; i < 16; ++i) {
    float d = std::fabs(x - MXFP4_LUT_CPU[i]);
    if (d < best_diff) {
      best_diff = d;
      best = i;
    }
  }
  return best;
}

static void quantize_to_mxfp4_host(const std::vector<float> &src,
                                   std::vector<uint8_t> &packed,
                                   std::vector<float> &scales)
{
  const size_t total = src.size();
  const size_t blocks = (total + 31) / 32;
  packed.assign((total + 1) / 2, 0);
  scales.assign(blocks, 0.0f);

  for (size_t blk = 0; blk < blocks; ++blk) {
    const size_t base = blk * 32;
    const size_t remain = std::min<size_t>(32, total - base);

    float maxabs = 0.0f;
    for (size_t i = 0; i < remain; ++i) {
      maxabs = std::max(maxabs, std::fabs(src[base + i]));
    }

    int exp_scale = 0;
    if (maxabs > 0.0f) {
      int e = 0;
      std::frexp(maxabs, &e);
      exp_scale = (e - 1) - 2;
    }
    const float scale = std::ldexp(1.0f, exp_scale);
    scales[blk] = scale;

    for (size_t i = 0; i < remain; ++i) {
      const size_t idx = base + i;
      const float v = src[idx] / scale;
      const uint8_t nib = enc_fp4_e2m1_nearest_host(v);
      const size_t byte_idx = idx >> 1;
      uint8_t &ref = packed[byte_idx];
      if ((idx & 1u) == 0u) {
        ref = (ref & 0xF0u) | (nib & 0x0Fu);
      } else {
        ref = (ref & 0x0Fu) | ((nib & 0x0Fu) << 4);
      }
    }
  }
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
    const __hip_bfloat16 *dW1,
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
    mlp1<16,16,16, 2,4,2, 1,1, 8>(
        dC, dA, dW1,
        d_offsets, d_counts,
        d_t2e, d_t2l,
        E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventRecord(ev_start, stream));
  for (int i = 0; i < iters; ++i) {
    mlp1<16,16,16, 4,4,2, 1,1, 8>(
        dC, dA, dW1,
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
    int PAD_K_MC>
static float time_mxfp4_variant(
    float *dC,
    const __hip_bfloat16 *dA_bf16,
    const uint8_t *dW_packed,
    const float *dW_scales,
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
    mlp1_mxfp4_optimized_unified_A_bf16<
        WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
            dC, dA_bf16, dW_packed, dW_scales,
            d_offsets, d_counts,
            d_t2e, d_t2l,
            E, K, N, cur_tiles, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventRecord(ev_start, stream));
  for (int i = 0; i < iters; ++i) {
    mlp1_mxfp4_optimized_unified_A_bf16<
        WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
            dC, dA_bf16, dW_packed, dW_scales,
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
    int PAD_K_MC>
static TuneResult benchmark_variant(
    const char *name,
    float *d_C_opt,
    const __hip_bfloat16 *d_A_bf16,
    const uint8_t *d_W_packed,
    const float *d_W_scales,
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
  mlp1_mxfp4_optimized_unified_A_bf16<
      WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
          d_C_opt, d_A_bf16, d_W_packed, d_W_scales,
          d_offsets, d_counts,
          d_t2e, d_t2l,
          E, K, N, cur_tiles, stream);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(h_C_opt.data(), d_C_opt, bytes_out, hipMemcpyDeviceToHost));

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < h_C_opt.size(); ++i) {
    const double a = static_cast<double>(h_C_ref[i]);
    const double b = static_cast<double>(h_C_opt[i]);
    const double ad = std::abs(a - b);
    const double rd = ad / (std::abs(a) + 1e-7);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, rd);
  }

  const float ms = time_mxfp4_variant<
      WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K_MC>(
          d_C_opt, d_A_bf16, d_W_packed, d_W_scales,
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

int main(int argc, char **argv)
{
  int iters = 50;
  int warmup = 5;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
  }

  init_mxfp4_lut_on_device();

  SimConfig cfg;
  const int E = cfg.num_experts;
  const int H = cfg.hidden_size;
  const int D = cfg.intermediate_size / cfg.TP;
  const int o_len = 2 * D / cfg.TP;
  const int Bgrp = cfg.TP * cfg.BATCH_SIZE;
  const int K_topk = cfg.experts_per_token;
  const int total_pairs = Bgrp * K_topk;
  const size_t out_elems = static_cast<size_t>(total_pairs) * static_cast<size_t>(o_len);
  const size_t out_bytes = sizeof(float) * out_elems;

  printf("=== MoE MLP1 MXFP4 Benchmark (bf16 inputs) ===\n");
  printf("E=%d, H=%d, D=%d, o_len=%d, Bgrp=%d, K_topk=%d, total_pairs=%d\n",
         E, H, D, o_len, Bgrp, K_topk, total_pairs);

  std::vector<int> h_counts(E, total_pairs / E);
  int rem = total_pairs - (total_pairs / E) * E;
  for (int i = 0; i < rem; ++i) h_counts[i]++;

  std::vector<int> h_offsets(E, 0);
  int running = 0;
  for (int e = 0; e < E; ++e) {
    h_offsets[e] = running;
    running += h_counts[e];
  }

  int cur_tiles = 0;
  for (int e = 0; e < E; ++e) {
    cur_tiles += (h_counts[e] + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
  }
  std::vector<int> h_tile2expert(cur_tiles), h_tile2local(cur_tiles);
  {
    int cursor = 0;
    for (int e = 0; e < E; ++e) {
      int tiles = (h_counts[e] + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
      for (int t = 0; t < tiles; ++t) {
        h_tile2expert[cursor + t] = e;
        h_tile2local[cursor + t] = t;
      }
      cursor += tiles;
    }
  }

  std::mt19937 rng(42);
  std::vector<float> h_A_fp32(static_cast<size_t>(total_pairs) * H);
  std::vector<__hip_bfloat16> h_A_bf16(static_cast<size_t>(total_pairs) * H);
  std::vector<float> h_W1_fp32(static_cast<size_t>(E) * o_len * H);
  std::vector<__hip_bfloat16> h_W1_bf16(h_W1_fp32.size());
  for (size_t i = 0; i < h_A_fp32.size(); ++i) {
    float v = frand(rng);
    h_A_fp32[i] = v;
    h_A_bf16[i] = f2bf16(v);
  }
  for (size_t i = 0; i < h_W1_fp32.size(); ++i) {
    float x = frand(rng);
    h_W1_fp32[i] = x;
    h_W1_bf16[i] = f2bf16(x);
  }

  std::vector<uint8_t> h_W1_packed;
  std::vector<float> h_W1_scales_f32;
  quantize_to_mxfp4_host(h_W1_fp32, h_W1_packed, h_W1_scales_f32);

  std::vector<float> h_C_ref(out_elems, 0.f);
  std::vector<float> h_C_opt(out_elems, 0.f);

  float *d_A_fp32 = nullptr, *d_C_ref = nullptr, *d_C_opt = nullptr;
  __hip_bfloat16 *d_A_bf16 = nullptr, *d_W1_bf16 = nullptr;
  uint8_t *d_W1_packed = nullptr;
  float *d_W1_scales = nullptr;
  int *d_counts = nullptr, *d_offsets = nullptr, *d_t2e = nullptr, *d_t2l = nullptr;

  HIP_CHECK(hipMalloc(&d_A_fp32, sizeof(float) * h_A_fp32.size()));
  HIP_CHECK(hipMalloc(&d_A_bf16, sizeof(__hip_bfloat16) * h_A_bf16.size()));
  HIP_CHECK(hipMalloc(&d_W1_bf16, sizeof(__hip_bfloat16) * h_W1_bf16.size()));
  HIP_CHECK(hipMalloc(&d_W1_packed, h_W1_packed.size() * sizeof(uint8_t)));
  HIP_CHECK(hipMalloc(&d_W1_scales, h_W1_scales_f32.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C_ref, out_bytes));
  HIP_CHECK(hipMalloc(&d_C_opt, out_bytes));
  HIP_CHECK(hipMalloc(&d_counts, sizeof(int) * E));
  HIP_CHECK(hipMalloc(&d_offsets, sizeof(int) * E));
  HIP_CHECK(hipMalloc(&d_t2e, sizeof(int) * cur_tiles));
  HIP_CHECK(hipMalloc(&d_t2l, sizeof(int) * cur_tiles));

  HIP_CHECK(hipMemcpy(d_A_fp32, h_A_fp32.data(), sizeof(float) * h_A_fp32.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_A_bf16, h_A_bf16.data(), sizeof(__hip_bfloat16) * h_A_bf16.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W1_bf16, h_W1_bf16.data(), sizeof(__hip_bfloat16) * h_W1_bf16.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W1_packed, h_W1_packed.data(), h_W1_packed.size() * sizeof(uint8_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W1_scales, h_W1_scales_f32.data(), h_W1_scales_f32.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_counts, h_counts.data(), sizeof(int) * E, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_offsets, h_offsets.data(), sizeof(int) * E, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_t2e, h_tile2expert.data(), sizeof(int) * cur_tiles, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_t2l, h_tile2local.data(), sizeof(int) * cur_tiles, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemset(d_C_ref, 0, out_bytes));
  HIP_CHECK(hipMemset(d_C_opt, 0, out_bytes));

  mlp1<16,16,16, 4,4,2, 1,1, 8>(
      d_C_ref, d_A_fp32, d_W1_bf16,
      d_offsets, d_counts,
      d_t2e, d_t2l,
      E, H, o_len, cur_tiles, stream);
  HIP_CHECK(hipGetLastError());

  mlp1_mxfp4_optimized_unified_A_bf16<
      16,16,16, WAVES_M_MLP, WAVES_N_MLP, WAVES_K_MLP, 1,2, PAD_K_MLP>(
          d_C_opt, d_A_bf16, d_W1_packed, d_W1_scales,
          d_offsets, d_counts,
          d_t2e, d_t2l,
          E, H, o_len, cur_tiles, stream);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(h_C_ref.data(), d_C_ref, out_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_C_opt.data(), d_C_opt, out_bytes, hipMemcpyDeviceToHost));

  double check_abs = 0.0;
  double check_rel = 0.0;
  for (size_t i = 0; i < h_C_ref.size(); ++i) {
    const double a = static_cast<double>(h_C_ref[i]);
    const double b = static_cast<double>(h_C_opt[i]);
    const double ad = std::abs(a - b);
    const double rd = ad / (std::abs(a) + 1e-7);
    check_abs = std::max(check_abs, ad);
    check_rel = std::max(check_rel, rd);
  }
  printf("Initial diff: max_abs=%.3g max_rel=%.3g\n", check_abs, check_rel);

  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  const float t_base_ms = time_baseline_kernel(
      d_C_ref, d_A_fp32, d_W1_bf16,
      d_offsets, d_counts, d_t2e, d_t2l,
      E, H, o_len, cur_tiles,
      stream, ev_start, ev_stop,
      warmup, iters);

  double flops = 0.0;
  for (int e = 0; e < E; ++e) {
    flops += 2.0 * static_cast<double>(h_counts[e]) * static_cast<double>(H) * static_cast<double>(o_len);
  }
  auto gflops = [&](float ms_per_iter) {
    return (flops * 1e-9) / (ms_per_iter * 1e-3);
  };
  const double base_gflops = gflops(t_base_ms);

  std::vector<TuneResult> tune_results;

#define RUN_VARIANT(NAME, WM, WN, WK, WMV, WNV, WKV, TWM, TWN, PAD) \
  tune_results.push_back(benchmark_variant<WM, WN, WK, WMV, WNV, WKV, TWM, TWN, PAD>( \
      NAME, \
      d_C_opt, d_A_bf16, d_W1_packed, d_W1_scales, \
      d_offsets, d_counts, \
      d_t2e, d_t2l, \
      E, H, o_len, cur_tiles, \
      stream, ev_start, ev_stop, \
      warmup, iters, \
      h_C_opt, h_C_ref, \
      flops, t_base_ms, \
      out_bytes))

  RUN_VARIANT("default waves", 16,16,16, WAVES_M_MLP, WAVES_N_MLP, WAVES_K_MLP, 1,2, PAD_K_MLP);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=4", 16,16,16, 4,4,4, 1,2, 4);
  RUN_VARIANT("waves=4x4x4 tw=1x2 pad=6", 16,16,16, 4,4,4, 1,2, 6);
  RUN_VARIANT("waves=4x8x4 tw=1x2 pad=4", 16,16,16, 4,8,4, 1,2, 4);
  RUN_VARIANT("waves=4x8x4 tw=1x2 pad=6", 16,16,16, 4,8,4, 1,2, 6);
  RUN_VARIANT("waves=8x4x4 tw=1x2 pad=4", 16,16,16, 8,4,4, 1,2, 4);
  RUN_VARIANT("waves=8x4x4 tw=1x2 pad=6", 16,16,16, 8,4,4, 1,2, 6);
  RUN_VARIANT("waves=8x8x4 tw=1x2 pad=4", 16,16,16, 8,8,4, 1,2, 4);
  RUN_VARIANT("waves=8x8x4 tw=1x2 pad=6", 16,16,16, 8,8,4, 1,2, 6);

#undef RUN_VARIANT

  const TuneResult *best = nullptr;
  for (const auto &res : tune_results) {
    if (!best || res.avg_ms < best->avg_ms) best = &res;
  }

  printf("\n--- Results ---\n");
  printf("Baseline (mlp1 fp32):    %.3f ms  |  %.2f GFLOP/s\n", t_base_ms, base_gflops);

  if (!tune_results.empty()) {
    printf("\n--- Unified MXFP4 Kernel Autotune ---\n");
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
  HIP_CHECK(hipFree(d_W1_bf16));
  HIP_CHECK(hipFree(d_W1_packed));
  HIP_CHECK(hipFree(d_W1_scales));
  HIP_CHECK(hipFree(d_C_ref));
  HIP_CHECK(hipFree(d_C_opt));
  HIP_CHECK(hipFree(d_counts));
  HIP_CHECK(hipFree(d_offsets));
  HIP_CHECK(hipFree(d_t2e));
  HIP_CHECK(hipFree(d_t2l));

  return 0;
}

