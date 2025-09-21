// test_attention.cpp
// Benchmark + diff-check harness for fused_attention_kernel (HIP, ROCm)
// - Treats the kernel in kernels/attention.hpp as the **baseline**
// - "Optimized" path initially launches the same kernel (placeholder)
// - Measures runtime with hipEvents and compares outputs bitwise/tolerant
// - Focused on the provided model spec and defaults:
//     BATCH_SIZE=1024, WARP_SIZE=64, MAX_SEQ_LEN=1024, SW_WINDOW=128
//
// Build (example):
//   hipcc -O3 -std=c++17 -I. test_attention.cpp -o bench
//
// Run (defaults target even layer 0 with sliding window):
//   ./bench
//
// Optional flags:
//   --batch N          (default 1024)
//   --iters N          (default 20)
//   --layer L          (default 0; even=L%2==0 to exercise SW_WINDOW path)
//   --heads H          (default 64)
//   --kvheads K        (default 8)
//   --headdim D        (default 64)
//   --maxseq S         (default 1024)
//   --swwindow W       (default 128)
//   --tile T           (override tile length; default picks 64 for D=64)
//   --seed S           (default 123)

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 1024
#endif

#ifndef SW_WINDOW
#define SW_WINDOW 128
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

// Bring in the baseline kernel + helpers
#include "../kernels/attention.hpp"
#include "../kernels/attention_mfma.hpp"


// ---- Error handling ----
#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    std::cerr << "HIP error " << hipGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// ---- Utility: CLI parsing ----
static bool get_flag(int argc, char** argv, const char* key) {
  for (int i=1;i<argc;++i) if (std::string(argv[i])==key) return true;
  return false;
}

template<typename T>
static T get_arg(int argc, char** argv, const char* key, T def) {
  for (int i=1;i<argc-1;++i) {
    if (std::string(argv[i])==key) {
      if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::strtoll(argv[i+1], nullptr, 10));
      } else if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(std::atof(argv[i+1]));
      } else if constexpr (std::is_same_v<T, std::string>) {
        return std::string(argv[i+1]);
      } else if constexpr (std::is_same_v<T, const char*>) {
        return argv[i+1];
      } else {
        // Unsupported type path; return default.
        return def;
      }
    }
  }
  return def;
}

// ---- Host-side random fill ----
static void fill_uniform_float(float* ptr, size_t n, uint64_t seed, float lo=-1.f, float hi=1.f) {
  std::mt19937 rng((uint32_t)seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i=0;i<n;++i) ptr[i] = dist(rng);
}
static void fill_uniform_bf16(__hip_bfloat16* ptr, size_t n, uint64_t seed, float lo=-1.f, float hi=1.f) {
  std::mt19937 rng((uint32_t)seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i=0;i<n;++i) {
    float f = dist(rng);
    ptr[i] = __float2bfloat16(f);
  }
}
static void fill_seq_lengths(int* ptr, int B, int max_seq, uint64_t seed) {
  std::mt19937 rng((uint32_t)seed);
  std::uniform_int_distribution<int> dist(1, std::max(1, max_seq-1));
  for (int i=0;i<B;++i) ptr[i] = dist(rng);
}

// ---- Diff check ----
struct DiffStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t count_bad = 0;
};

static DiffStats diff_check(const float* a, const float* b, size_t n, double atol=1e-3, double rtol=1e-2, bool verbose=false) {
  DiffStats stats;
  size_t shown = 0;
  for (size_t i=0;i<n;++i) {
    double va = (double)a[i];
    double vb = (double)b[i];
    double absd = std::abs(va - vb);
    double denom = std::max({1e-12, std::abs(va), std::abs(vb)});
    double reld = absd / denom;
    stats.max_abs = std::max(stats.max_abs, absd);
    stats.max_rel = std::max(stats.max_rel, reld);
    bool bad = absd > (atol + rtol * denom);
    if (bad) {
      stats.count_bad++;
      if (verbose && shown < 5) {
        std::cout << "  mismatch[" << i << "]: a=" << va << " b=" << vb
                  << " abs=" << absd << " rel=" << reld << "\n";
        shown++;
      }
    }
  }
  return stats;
}

// ---- Launch helpers ----
static size_t compute_shmem_bytes(int tile_t, int head_dim) {
  return (size_t)2 * (size_t)tile_t * (size_t)head_dim * sizeof(float);
}

static void pick_defaults_for_launch(
    int n_heads, int n_kv_heads, int head_dim,
    int& block_x, int& block_y, int& tile_t, size_t& shmem_bytes) {

  const int kv_mul = n_heads / n_kv_heads;
  block_x = 64;
  block_y = std::min(kv_mul, 16);

  int T = 64; // For D=64, 2*T*D*4 = 32768 bytes
  if (compute_shmem_bytes(T, head_dim) > 64*1024) {
    int cand[] = {48, 32, 16, 8, 4, 1};
    for (int c : cand) {
      if (compute_shmem_bytes(c, head_dim) <= 64*1024) { T = c; break; }
    }
  }
  tile_t = T;
  shmem_bytes = compute_shmem_bytes(tile_t, head_dim);
}

static void launch_baseline_old(
    dim3 grid, dim3 block, size_t shmem, hipStream_t stream,
    float* out, const float* q,
    const __hip_bfloat16* key_cache, const __hip_bfloat16* value_cache,
    const __hip_bfloat16* sinks, const float* mask, const int* seq_lengths,
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int tile_t) {

  hipLaunchKernelGGL(
    fused_attention_kernel_optimized, // baseline symbol from attention.hpp
    grid, block, shmem, stream,
    out, q, key_cache, value_cache,
    sinks, mask, seq_lengths,
    batch_size, n_heads, n_kv_heads, head_dim,
    seq_len, n_layers, layer_idx, use_sw,
    batch_kv_stride, layer_kv_offset, tile_t
  );
}
static size_t get_device_max_dyn_shmem() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  int basic = 0;
  hipError_t st_basic = hipDeviceGetAttribute(
      &basic, hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
  if (st_basic != hipSuccess || basic <= 0) {
    // Fallback if the query ever fails: assume 64 KiB (typical default)
    basic = 64 * 1024;
  }

  // On NVIDIA (HIP on CUDA), an "optin" cap may exist. Guard it so ROCm compiles.
  // Only prefer it if it’s larger than the basic per-block limit.
#if defined(__HIP_PLATFORM_NVIDIA__) && defined(hipDeviceAttributeMaxSharedMemoryPerBlockOptin)
  int optin = 0;
  hipError_t st_optin = hipDeviceGetAttribute(
      &optin, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, dev);
  if (st_optin == hipSuccess && optin > basic) {
    return static_cast<size_t>(optin);
  }
#endif

  return static_cast<size_t>(basic);
}

// static void launch_baseline(
//     dim3 /*grid_ignored*/, dim3 /*block_ignored*/, size_t /*shmem_ignored*/, hipStream_t stream,
//     float* out, const float* q,
//     const __hip_bfloat16* key_cache, const __hip_bfloat16* value_cache,
//     const __hip_bfloat16* sinks, const float* mask, const int* seq_lengths,
//     int batch_size, int n_heads, int n_kv_heads, int head_dim,
//     int seq_len, int n_layers, int layer_idx, bool use_sw,
//     size_t batch_kv_stride, size_t layer_kv_offset, int tile_t_in)
// {
//   // This helper matches the host launcher you had before (specialized for D=64).
//   if (batch_size <= 0 || n_kv_heads <= 0 || n_heads <= 0 || head_dim != 64) return;

//   const int gqa_ratio = n_heads / n_kv_heads;

//   // Respect sliding-window cap (even layers) and default max tile 112 from the kernel file.
//   const int sw_cap = (use_sw && ((layer_idx & 1) == 0)) ? SW_WINDOW : 112;
//   int tile_t = tile_t_in > 0 ? tile_t_in : std::min(112, sw_cap);

//   // Block/Grid mapping (64 lanes × gqa_ratio warps = 512 threads)
//   dim3 block(64, gqa_ratio, 1);
//   dim3 grid(n_kv_heads, batch_size, 1);

//   // Dynamic LDS: K and V only, with PADDED=72 (64 + 8) to avoid bank conflicts
//   constexpr int PADDED = 72;
//   size_t shmem_bytes = (size_t)2 * (size_t)tile_t * (size_t)PADDED * sizeof(__hip_bfloat16);

//   // Cap to device max dynamic shared memory if necessary
//   size_t max_dyn = get_device_max_dyn_shmem();
//   if (max_dyn > 0 && shmem_bytes > max_dyn) {
//     int max_tile = (int)(max_dyn / (2 * PADDED * sizeof(__hip_bfloat16)));
//     max_tile = std::max(1, std::min(max_tile, sw_cap));
//     tile_t = std::min(tile_t, max_tile);
//     shmem_bytes = (size_t)2 * (size_t)tile_t * (size_t)PADDED * sizeof(__hip_bfloat16);
//   }

//   // Try to request the needed dynamic shared memory; ignore failure on stacks that don't use opt-in.
//   (void)hipFuncSetAttribute((const void*)fused_attention_kernel_optimized,
//                             hipFuncAttributeMaxDynamicSharedMemorySize,
//                             (int)shmem_bytes);

//   hipLaunchKernelGGL(
//     fused_attention_kernel_optimized,
//     grid, block, shmem_bytes, stream,
//     out, q, key_cache, value_cache,
//     sinks, mask, seq_lengths,
//     batch_size, n_heads, n_kv_heads, head_dim,
//     seq_len, n_layers, layer_idx, use_sw,
//     batch_kv_stride, layer_kv_offset, tile_t
//   );
// }


static void launch_baseline(
    dim3, dim3, size_t, hipStream_t stream,
    float* out, const float* q,
    const __hip_bfloat16* key_cache, const __hip_bfloat16* value_cache,
    const __hip_bfloat16* sinks, const float* mask, const int* seq_lengths,
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int tile_t_in)
{
  if (batch_size<=0 || n_kv_heads<=0 || n_heads<=0 || head_dim!=64) return;
  const int gqa_ratio = n_heads / n_kv_heads;
  if (gqa_ratio != 8) return;

  const int sw_cap = (use_sw && ((layer_idx & 1) == 0)) ? SW_WINDOW : 112;
  int tile_t = tile_t_in > 0 ? tile_t_in : std::min(112, sw_cap);

  dim3 block(64, 1, 1);                 // <-- single warp
  dim3 grid(n_kv_heads, batch_size, 1);

  size_t sK_bytes = (size_t)tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16);
  size_t sV_bytes = (size_t)tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16);
  size_t shmem_bytes = sK_bytes + sV_bytes;

  size_t max_dyn = get_device_max_dyn_shmem();
  if (max_dyn > 0 && shmem_bytes > max_dyn) {
    int max_tile = (int)(max_dyn / (2 * PADDED_HEAD_DIM * sizeof(__hip_bfloat16)));
    max_tile = std::max(1, std::min(max_tile, sw_cap));
    tile_t   = std::min(tile_t, max_tile);
    shmem_bytes = (size_t)2 * tile_t * PADDED_HEAD_DIM * sizeof(__hip_bfloat16);
  }

  (void)hipFuncSetAttribute((const void*)fused_attention_kernel_1warp8q_fast,
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)shmem_bytes);

  hipLaunchKernelGGL(
      fused_attention_kernel_1warp8q_fast,
      grid, block, shmem_bytes, stream,
      out, q, key_cache, value_cache,
      sinks, mask, seq_lengths,
      batch_size, n_heads, n_kv_heads, head_dim,
      seq_len, n_layers, layer_idx, use_sw,
      batch_kv_stride, layer_kv_offset, tile_t
  );
}




// Fused Flash-Decoding launcher with the original interface.
// Replaces the previous single-warp path with a multi-warp split-parallel kernel.
// Assumes the kernel `flashdecoding_fused_multiwarp_1warp8q` is defined elsewhere.

#ifndef FD_MAX_WARPS_PER_CTA
#define FD_MAX_WARPS_PER_CTA 4   // Good default; try 4 or 6 on MI2xx/MI3xx
#endif

// extern size_t get_device_max_dyn_shmem();

// ---------------------------------
// Launcher
// ---------------------------------
static void launch_optimized(
    dim3, dim3, size_t, hipStream_t stream,
    float* out, const float* q,
    const __hip_bfloat16* key_cache, const __hip_bfloat16* value_cache,
    const __hip_bfloat16* sinks, const float* mask, const int* seq_lengths,
    int batch_size, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int n_layers, int layer_idx, bool use_sw,
    size_t batch_kv_stride, size_t layer_kv_offset, int tile_t_ignored)
{
  if (batch_size<=0 || n_kv_heads<=0 || n_heads<=0 || head_dim!=64) return;
  const int gqa_ratio = n_heads / n_kv_heads;
  if (gqa_ratio != 8) return;

  // Use 4 warps per block (256 threads) by default — good balance of latency hiding
  // without inflating the reduction buffer.
  const int warps_per_block = 4;

  dim3 block(64 * warps_per_block, 1, 1);
  dim3 grid(n_kv_heads, batch_size, 1);

  // Shared memory for the fast-merge reduction only:
  // (m,l): 2 * W * 8 floats  +  numerators: W * 64 * 8 floats
  const size_t red_bytes = (size_t)((2 * warps_per_block * 8) + (warps_per_block * 64 * 8)) * sizeof(float);
  const size_t shmem_bytes = red_bytes;

  (void)hipFuncSetAttribute((const void*)flashdecoding_fused_fastmerge_nostage_1warp8q,
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            (int)shmem_bytes);

  hipLaunchKernelGGL(
      flashdecoding_fused_fastmerge_nostage_1warp8q,
      grid, block, shmem_bytes, stream,
      out, q, key_cache, value_cache,
      sinks, mask, seq_lengths,
      batch_size, n_heads, n_kv_heads, head_dim,
      seq_len, n_layers, layer_idx, use_sw,
      batch_kv_stride, layer_kv_offset, /*tile_t_unused=*/0
  );
}




// In your benchmark switching logic, you would now call launch_mfma


// ---- Main ----
int main(int argc, char** argv) {
  // Model / test defaults matching the provided spec
  int n_layers   = get_arg<int>(argc, argv, "--layers",   24);
  int n_heads    = get_arg<int>(argc, argv, "--heads",    64);
  int n_kv_heads = get_arg<int>(argc, argv, "--kvheads",  8);
  int head_dim   = get_arg<int>(argc, argv, "--headdim",  64);
  int seq_len    = get_arg<int>(argc, argv, "--maxseq",   MAX_SEQ_LEN);
  int sw_window  = get_arg<int>(argc, argv, "--swwindow", SW_WINDOW);
  int layer_idx  = get_arg<int>(argc, argv, "--layer",    0);   // even by default
  int batch_size = get_arg<int>(argc, argv, "--batch",    1024);
  int iters      = get_arg<int>(argc, argv, "--iters",    20);
  int tile_override = get_arg<int>(argc, argv, "--tile",  -1);
  uint64_t seed  = (uint64_t)get_arg<long long>(argc, argv, "--seed", 123);

  bool use_sliding_window = (sw_window > 0);
  bool even_layer = ((layer_idx & 1) == 0);
  const int kv_dim = head_dim * n_kv_heads;

  // For this focused test, allocate cache sized to ONE layer's capacity
  const int layer_cap = even_layer ? sw_window : seq_len;

  // Sanity prints
  std::cout << "=== fused_attention_kernel test ===\n";
  std::cout << "B=" << batch_size << " H=" << n_heads << " KVH=" << n_kv_heads
            << " D=" << head_dim << " seq_len=" << seq_len
            << " sw_window=" << sw_window << " layer=" << layer_idx
            << " (even=" << (even_layer?"yes":"no") << ")\n";
  std::cout << "Cache per batch layer_cap=" << layer_cap
            << " kv_dim=" << kv_dim << " (elements)" << std::endl;

  // Derived sizes
  const size_t Q_elems   = (size_t)batch_size * n_heads * head_dim;
  const size_t OUT_elems = Q_elems;
  const size_t SINK_elems= (size_t)n_heads;
  const size_t SEQL_elems= (size_t)batch_size;
  const size_t CACHE_elems = (size_t)batch_size * layer_cap * kv_dim; // per cache (keys or values)

  // Host buffers
  std::vector<float> h_q(Q_elems);
  std::vector<float> h_out_baseline(OUT_elems, 0.f);
  std::vector<float> h_out_opt(OUT_elems, 0.f);
  std::vector<int>   h_seq(SEQL_elems);
  std::vector<__hip_bfloat16> h_sinks(SINK_elems);
  std::vector<__hip_bfloat16> h_key(CACHE_elems);
  std::vector<__hip_bfloat16> h_val(CACHE_elems);

  // Initialize host data deterministically
  fill_uniform_float(h_q.data(), Q_elems, seed+1, -1.f, 1.f);
  fill_uniform_bf16(h_key.data(), CACHE_elems, seed+2, -0.5f, 0.5f);
  fill_uniform_bf16(h_val.data(), CACHE_elems, seed+3, -0.5f, 0.5f);
  fill_uniform_bf16(h_sinks.data(), SINK_elems, seed+4, -2.f, 2.f);
  fill_seq_lengths(h_seq.data(), batch_size, seq_len, seed+5);

  // Device buffers
  float *d_q = nullptr, *d_out_baseline = nullptr, *d_out_opt = nullptr;
  __hip_bfloat16 *d_key = nullptr, *d_val = nullptr, *d_sinks = nullptr;
  int *d_seq = nullptr;

  HIP_CHECK(hipMalloc(&d_q, Q_elems*sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_baseline, OUT_elems*sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_opt, OUT_elems*sizeof(float)));
  HIP_CHECK(hipMalloc(&d_key, CACHE_elems*sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_val, CACHE_elems*sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_sinks, SINK_elems*sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_seq, SEQL_elems*sizeof(int)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), Q_elems*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_key, h_key.data(), CACHE_elems*sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_val, h_val.data(), CACHE_elems*sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sinks, h_sinks.data(), SINK_elems*sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_seq, h_seq.data(), SEQL_elems*sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out_baseline, 0, OUT_elems*sizeof(float)));
  HIP_CHECK(hipMemset(d_out_opt, 0, OUT_elems*sizeof(float)));

  // Launch config
  int block_x, block_y, tile_t;
  size_t shmem_bytes;
  pick_defaults_for_launch(n_heads, n_kv_heads, head_dim, block_x, block_y, tile_t, shmem_bytes);
  if (tile_override > 0) {
    tile_t = tile_override;
    shmem_bytes = compute_shmem_bytes(tile_t, head_dim);
  }

  dim3 block(block_x, block_y);
  dim3 grid(n_kv_heads, batch_size);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Choose batch_kv_stride + layer_kv_offset for our single-layer allocation
  const size_t batch_kv_stride = (size_t)layer_cap * (size_t)kv_dim;
  const size_t layer_kv_offset = 0;

  // Warmup
  for (int i=0;i<3;++i) {
    launch_baseline(grid, block, shmem_bytes, stream,
      d_out_baseline, d_q, d_key, d_val, d_sinks, /*mask*/nullptr, d_seq,
      batch_size, n_heads, n_kv_heads, head_dim, seq_len,
      n_layers, layer_idx, use_sliding_window,
      batch_kv_stride, layer_kv_offset, tile_t);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipGetLastError());

  // Benchmark baseline
  hipEvent_t ev0, ev1;
  HIP_CHECK(hipEventCreate(&ev0));
  HIP_CHECK(hipEventCreate(&ev1));
  HIP_CHECK(hipEventRecord(ev0, stream));
  for (int i=0;i<iters;++i) {
    launch_baseline(grid, block, shmem_bytes, stream,
      d_out_baseline, d_q, d_key, d_val, d_sinks, /*mask*/nullptr, d_seq,
      batch_size, n_heads, n_kv_heads, head_dim, seq_len,
      n_layers, layer_idx, use_sliding_window,
      batch_kv_stride, layer_kv_offset, tile_t);
  }
  HIP_CHECK(hipEventRecord(ev1, stream));
  HIP_CHECK(hipEventSynchronize(ev1));
  float ms_baseline = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms_baseline, ev0, ev1));
  ms_baseline /= (float)iters;

  // Benchmark "optimized" (currently identical)
  HIP_CHECK(hipEventRecord(ev0, stream));
  for (int i=0;i<iters;++i) {
    launch_optimized(grid, block, shmem_bytes, stream,
      d_out_opt, d_q, d_key, d_val, d_sinks, /*mask*/nullptr, d_seq,
      batch_size, n_heads, n_kv_heads, head_dim, seq_len,
      n_layers, layer_idx, use_sliding_window,
      batch_kv_stride, layer_kv_offset, tile_t);
  }
  HIP_CHECK(hipEventRecord(ev1, stream));
  HIP_CHECK(hipEventSynchronize(ev1));
  float ms_opt = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms_opt, ev0, ev1));
  ms_opt /= (float)iters;

  // Copy back and diff
  HIP_CHECK(hipMemcpy(h_out_baseline.data(), d_out_baseline, OUT_elems*sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_out_opt.data(), d_out_opt, OUT_elems*sizeof(float), hipMemcpyDeviceToHost));

  DiffStats stats = diff_check(h_out_baseline.data(), h_out_opt.data(), OUT_elems, 1e-3, 1e-2, /*verbose*/false);

  std::cout << "\n--- Results ---\n";
  std::cout << "Baseline avg time:  " << ms_baseline << " ms\n";
  std::cout << "Optimized avg time: " << ms_opt      << " ms\n";
  if (ms_opt > 0) std::cout << "Speedup:           " << (ms_baseline / ms_opt) << "x\n";
  std::cout << "Diff max_abs=" << stats.max_abs
            << " max_rel=" << stats.max_rel
            << " bad=" << stats.count_bad << "/" << OUT_elems << "\n";

  // Cleanup
  HIP_CHECK(hipEventDestroy(ev0));
  HIP_CHECK(hipEventDestroy(ev1));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out_baseline));
  HIP_CHECK(hipFree(d_out_opt));
  HIP_CHECK(hipFree(d_key));
  HIP_CHECK(hipFree(d_val));
  HIP_CHECK(hipFree(d_sinks));
  HIP_CHECK(hipFree(d_seq));

  std::cout << "Done.\n";
  return 0;
}

