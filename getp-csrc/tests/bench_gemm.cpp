// bench_gemm.cpp
// Baseline vs optimized (small-M) mfma_bf16 matmul benchmark.
// - Baseline: matmul_vec128_singlebuf with config 16 16 16  4 8 4  1 2 4
// - Optimized: new small-M kernel+launcher (matmul_smallM_opt) for M in {1,2,3,4}, N=1440, K=2880

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../utils.hpp"
#include "../kernels/matmul.hpp"

// -------------------------------------------------------------------------------------
// CLI and stats
// -------------------------------------------------------------------------------------
struct CliOptions {
    int M = 3712;         // keep same defaults as the previous version
    int K = 2880;
    int N = 1440;
    int warmup = 5;
    int iters = 50;
    unsigned seed = 123u;
    bool run_validate = false;
    bool run_real_case = false;
    bool compare_outputs = true;
};

struct TimingStats {
    double avg_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
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

static TimingStats summarize_samples(const std::vector<float> &samples) {
    TimingStats stats;
    if (samples.empty()) return stats;

    double sum = 0.0;
    double min_v = std::numeric_limits<double>::infinity();
    double max_v = 0.0;
    for (float v : samples) {
        sum += static_cast<double>(v);
        min_v = std::min(min_v, static_cast<double>(v));
        max_v = std::max(max_v, static_cast<double>(v));
    }
    const double mean = sum / static_cast<double>(samples.size());
    double var = 0.0;
    for (float v : samples) {
        const double diff = static_cast<double>(v) - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(samples.size());

    stats.avg_ms = mean;
    stats.stddev_ms = std::sqrt(var);
    stats.min_ms = min_v;
    stats.max_ms = max_v;
    return stats;
}

static void compute_errors(const float *ref, const float *test, size_t elems,
                           double &max_abs, double &max_rel) {
    max_abs = 0.0;
    max_rel = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        const double a = static_cast<double>(ref[i]);
        const double b = static_cast<double>(test[i]);
        const double diff = std::fabs(a - b);
        const double denom = std::max(std::fabs(a), 1e-8);
        if (diff > max_abs) max_abs = diff;
        const double rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
    }
}

// -------------------------------------------------------------------------------------
// Baseline runner: matmul_vec128_singlebuf<16,16,16, 4,8,4, 1,2,4, false>
// -------------------------------------------------------------------------------------
static TimingStats run_baseline(float *dC,
                                const float *dA,
                                const __hip_bfloat16 *dW,
                                int M, int K, int N,
                                hipStream_t stream,
                                hipEvent_t ev_start,
                                hipEvent_t ev_stop,
                                int warmup,
                                int iters) {
    const auto launch = [&]() {
        matmul_vec128_singlebuf<
            16, 16, 16,   // WM, WN, WK
            4,  8,  4,    // WAVES_M, WAVES_N, WAVES_K
            1,  2,        // TW_M, TW_N
            4,            // PAD_K_MC
            false         // FUSED
        >(dC, dA, dW, M, K, N, /*bias=*/nullptr, stream);
    };

    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(ev_start, stream));
        launch();
        HIP_CHECK(hipEventRecord(ev_stop, stream));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        samples.push_back(ms);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());
    return summarize_samples(samples);
}

// -------------------------------------------------------------------------------------
// Optimized (new) kernel launcher path for small M
// Assumes a new function exists in matmul.hpp:
//     void matmul_smallM_opt(float* C, const float* A, const __hip_bfloat16* Wbf16,
//                            int M, int K, int N, hipStream_t stream);
// -------------------------------------------------------------------------------------
static TimingStats run_smallM_optimized(float *dC,
                                        const float *dA,
                                        const __hip_bfloat16 *dW,
                                        int M, int K, int N,
                                        hipStream_t stream,
                                        hipEvent_t ev_start,
                                        hipEvent_t ev_stop,
                                        int warmup,
                                        int iters) {
    const auto launch = [&]() {
        matmul_smallM_opt(dC, dA, dW, M, K, N, stream);
    };

    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(ev_start, stream));
        launch();
        HIP_CHECK(hipEventRecord(ev_stop, stream));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        samples.push_back(ms);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());
    return summarize_samples(samples);
}

// -------------------------------------------------------------------------------------
// CPU reference (optional, used by validation)
// -------------------------------------------------------------------------------------
static void host_reference(const std::vector<float> &hA,
                           const std::vector<float> &hW,
                           std::vector<float> &hC,
                           int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(hA[(size_t)m * K + k]) *
                       static_cast<double>(hW[(size_t)n * K + k]);
            }
            hC[(size_t)m * N + n] = static_cast<float>(acc);
        }
    }
}

static void run_validation(const CliOptions &opts) {
    const int M = 64;
    const int K = 128;
    const int N = 96;
    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_W = static_cast<size_t>(N) * K;
    const size_t elems_C = static_cast<size_t>(M) * N;
    const size_t bytes_A = elems_A * sizeof(float);
    const size_t bytes_W = elems_W * sizeof(__hip_bfloat16);
    const size_t bytes_C = elems_C * sizeof(float);

    std::mt19937 rng(opts.seed);
    std::vector<float> hA(elems_A);
    std::vector<float> hW_float(elems_W);
    std::vector<__hip_bfloat16> hW_bf16(elems_W);
    std::vector<float> hC_cpu(elems_C);
    std::vector<float> hC_gpu(elems_C);

    for (size_t i = 0; i < elems_A; ++i) hA[i] = frand(rng);
    for (size_t i = 0; i < elems_W; ++i) {
        const float v = frand(rng);
        hW_float[i] = v;
        hW_bf16[i] = f2bf16(v);
    }

    host_reference(hA, hW_float, hC_cpu, M, K, N);

    float *dA = nullptr;
    __hip_bfloat16 *dW = nullptr;
    float *dC = nullptr;

    HIP_CHECK(hipMalloc(&dA, bytes_A));
    HIP_CHECK(hipMalloc(&dW, bytes_W));
    HIP_CHECK(hipMalloc(&dC, bytes_C));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, hW_bf16.data(), bytes_W, hipMemcpyHostToDevice));

    hipStream_t stream;
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipMemset(dC, 0, bytes_C));
    const TimingStats timing = run_baseline(dC, dA, dW, M, K, N,
                                            stream, ev_start, ev_stop,
                                            opts.warmup, opts.iters);

    HIP_CHECK(hipMemcpy(hC_gpu.data(), dC, bytes_C, hipMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0;
    compute_errors(hC_cpu.data(), hC_gpu.data(), elems_C, max_abs, max_rel);

    printf("Validation shape M=%d K=%d N=%d\n", M, K, N);
    printf("  GPU avg %.3f ms  min %.3f  max %.3f  stddev %.4f\n",
           timing.avg_ms, timing.min_ms, timing.max_ms, timing.stddev_ms);
    printf("  Max abs diff = %.6g   max rel diff = %.6g\n",
           max_abs, max_rel);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dC));
}

// -------------------------------------------------------------------------------------
// Run a single benchmark: Baseline (always). If shape is small-M (M<=4, N=1440, K=2880),
// also time the optimized kernel and compare outputs (baseline vs optimized).
// -------------------------------------------------------------------------------------
static void run_single_benchmark(const CliOptions &opts) {
    const int M = opts.M;
    const int K = opts.K;
    const int N = opts.N;

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_W = static_cast<size_t>(N) * K;
    const size_t elems_C = static_cast<size_t>(M) * N;
    const size_t bytes_A = elems_A * sizeof(float);
    const size_t bytes_W = elems_W * sizeof(__hip_bfloat16);
    const size_t bytes_C = elems_C * sizeof(float);

    printf("=== Matmul Benchmark ===\n");
    printf("Shape: M=%d K=%d N=%d\n", M, K, N);
    printf("Warmup=%d  Iters=%d  Seed=%u\n",
           opts.warmup, opts.iters, opts.seed);
    printf("Comparing outputs: %s\n", opts.compare_outputs ? "yes" : "no");

    std::mt19937 rng(opts.seed);
    std::vector<float> hA(elems_A);
    std::vector<float> hW_float(elems_W);
    std::vector<__hip_bfloat16> hW_bf16(elems_W);

    for (size_t i = 0; i < elems_A; ++i) hA[i] = frand(rng);
    for (size_t i = 0; i < elems_W; ++i) {
        const float v = frand(rng);
        hW_float[i] = v;
        hW_bf16[i]  = f2bf16(v);
    }

    std::vector<float> hC_base, hC_opt;
    if (opts.compare_outputs) {
        hC_base.resize(elems_C);
        hC_opt.resize(elems_C);
    }

    float *dA = nullptr;
    __hip_bfloat16 *dW = nullptr;
    float *dC_base = nullptr;
    float *dC_opt  = nullptr;

    HIP_CHECK(hipMalloc(&dA, bytes_A));
    HIP_CHECK(hipMalloc(&dW, bytes_W));
    HIP_CHECK(hipMalloc(&dC_base, bytes_C));
    HIP_CHECK(hipMalloc(&dC_opt,  bytes_C));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, hW_bf16.data(), bytes_W, hipMemcpyHostToDevice));

    hipStream_t stream;
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    const double flop_count = 2.0 * static_cast<double>(M) * N * K;

    // ---- Baseline ----
    HIP_CHECK(hipMemset(dC_base, 0, bytes_C));
    const TimingStats base_timing = run_baseline(dC_base, dA, dW, M, K, N,
                                                 stream, ev_start, ev_stop,
                                                 opts.warmup, opts.iters);
    const double base_gflops = (flop_count * 1e-6) / base_timing.avg_ms;
    if (opts.compare_outputs) {
        HIP_CHECK(hipMemcpy(hC_base.data(), dC_base, bytes_C, hipMemcpyDeviceToHost));
    }

    printf("\nBaseline (vec128 singlebuf @ waves=4x8x4, tw=1x2, pad=4):\n");
    printf("  avg %.3f ms  min %.3f  max %.3f  stddev %.4f  |  %.2f GFLOP/s\n",
           base_timing.avg_ms,
           base_timing.min_ms,
           base_timing.max_ms,
           base_timing.stddev_ms,
           base_gflops);

    // ---- Optimized (only when small-M shape matches the intended target) ----
    const bool smallM_shape = (M >= 1 && M <= 128 && N == 1440 && K == 2880);
    if (smallM_shape) {
        HIP_CHECK(hipMemset(dC_opt, 0, bytes_C));
        const TimingStats opt_timing = run_smallM_optimized(dC_opt, dA, dW, M, K, N,
                                                            stream, ev_start, ev_stop,
                                                            opts.warmup, opts.iters);
        const double opt_gflops = (flop_count * 1e-6) / opt_timing.avg_ms;

        printf("\nOptimized small-M kernel:\n");
        printf("  avg %.3f ms  min %.3f  max %.3f  stddev %.4f  |  %.2f GFLOP/s\n",
               opt_timing.avg_ms,
               opt_timing.min_ms,
               opt_timing.max_ms,
               opt_timing.stddev_ms,
               opt_gflops);

        printf("\nSpeedup vs baseline: %.3fx\n", base_timing.avg_ms / opt_timing.avg_ms);

        if (opts.compare_outputs) {
            HIP_CHECK(hipMemcpy(hC_opt.data(), dC_opt, bytes_C, hipMemcpyDeviceToHost));
            double max_abs = 0.0, max_rel = 0.0;
            compute_errors(hC_base.data(), hC_opt.data(), elems_C, max_abs, max_rel);
            printf("Output diff (baseline vs optimized): max_abs=%.3g  max_rel=%.3g  %s\n",
                   max_abs, max_rel, (max_abs < 1e-4 && max_rel < 1e-4) ? "OK" : "WARN");
        }
    } else {
        printf("\nOptimized small-M kernel not run (shape not targeted: requires M in {1..4}, N=1440, K=2880).\n");
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dC_base));
    HIP_CHECK(hipFree(dC_opt));
}

// -------------------------------------------------------------------------------------
// Real-case scenario (kept, but uses baseline only; small-M optimized path is not relevant)
// -------------------------------------------------------------------------------------
struct RealCase {
    const char *label;
    int repeats;
    int M;
    int K;
    int N;
};

static void run_real_scenario(const CliOptions &opts) {
    static const RealCase cases[] = {
        {"matmul_vec128_singlebuf_M1024_K2880_N201088", 2044, 1024, 2880, 201088},
        {"matmul_vec128_singlebuf_M1024_K2880_N5120",   49056, 1024, 2880, 5120},
        {"matmul_vec128_singlebuf_M1024_K4096_N2880",   49056, 1024, 4096, 2880},
        {"matmul_vec128_singlebuf_M2048_K2880_N32",     49056, 2048, 2880, 32},
    };

    double total_ms = 0.0;

    printf("Running real-case scenario (warmup=%d, iters=%d)\n",
           opts.warmup, opts.iters);

    for (const auto &rc : cases) {
        CliOptions local = opts;
        local.M = rc.M;
        local.K = rc.K;
        local.N = rc.N;

        const size_t elems_A = static_cast<size_t>(local.M) * local.K;
        const size_t elems_W = static_cast<size_t>(local.N) * local.K;
        const size_t elems_C = static_cast<size_t>(local.M) * local.N;
        const size_t bytes_A = elems_A * sizeof(float);
        const size_t bytes_W = elems_W * sizeof(__hip_bfloat16);
        const size_t bytes_C = elems_C * sizeof(float);

        std::mt19937 rng(local.seed);
        std::vector<float> hA(elems_A);
        std::vector<__hip_bfloat16> hW(elems_W);
        for (size_t i = 0; i < elems_A; ++i) hA[i] = frand(rng);
        for (size_t i = 0; i < elems_W; ++i) hW[i] = f2bf16(frand(rng));

        float *dA = nullptr;
        __hip_bfloat16 *dW = nullptr;
        float *dC = nullptr;
        HIP_CHECK(hipMalloc(&dA, bytes_A));
        HIP_CHECK(hipMalloc(&dW, bytes_W));
        HIP_CHECK(hipMalloc(&dC, bytes_C));

        HIP_CHECK(hipMemcpy(dA, hA.data(), bytes_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dW, hW.data(), bytes_W, hipMemcpyHostToDevice));

        hipStream_t stream;
        hipEvent_t ev_start, ev_stop;
        HIP_CHECK(hipStreamCreate(&stream));
        HIP_CHECK(hipEventCreate(&ev_start));
        HIP_CHECK(hipEventCreate(&ev_stop));

        HIP_CHECK(hipMemset(dC, 0, bytes_C));
        const TimingStats timing = run_baseline(dC, dA, dW, local.M, local.K, local.N,
                                                stream, ev_start, ev_stop,
                                                local.warmup, local.iters);

        const double avg_ms = timing.avg_ms;
        const double std_ms = timing.stddev_ms;
        total_ms += avg_ms * rc.repeats;

        printf("%-45s %8d  avg %.4f ms  std %.4f  min %.3f  max %.3f  total %.3f ms\n",
               rc.label,
               rc.repeats,
               avg_ms,
               std_ms,
               timing.min_ms,
               timing.max_ms,
               avg_ms * rc.repeats);

        HIP_CHECK(hipEventDestroy(ev_start));
        HIP_CHECK(hipEventDestroy(ev_stop));
        HIP_CHECK(hipStreamDestroy(stream));
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dW));
        HIP_CHECK(hipFree(dC));
    }

    printf("\n==================================================\n");
    printf("\xE2\x9A\xA1 PERFORMANCE BREAKDOWN\n");
    printf("==================================================\n");
    printf("Total execution time: %.3f ms\n", total_ms);
}

// -------------------------------------------------------------------------------------
// CLI parsing / main
// -------------------------------------------------------------------------------------
static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --M <int>           Rows of A/C (default 3712)\n");
    printf("  --K <int>           Common dimension (default 2880)\n");
    printf("  --N <int>           Columns of C (default 1440)\n");
    printf("  --warmup <int>      Warmup iterations (default 5)\n");
    printf("  --iters <int>       Timed iterations (default 50)\n");
    printf("  --seed <int>        RNG seed (default 123)\n");
    printf("  --validate          Run small-shape CPU validation (baseline)\n");
    printf("  --real-case         Run predefined real workload mix (baseline only)\n");
    printf("  --no-compare        Skip baseline vs optimized comparison\n");
    printf("  --help              Show this message\n");
}

static CliOptions parse_cli(int argc, char **argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--M") && i + 1 < argc) {
            opts.M = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--K") && i + 1 < argc) {
            opts.K = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--N") && i + 1 < argc) {
            opts.N = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            opts.seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--validate")) {
            opts.run_validate = true;
        } else if (!std::strcmp(argv[i], "--real-case")) {
            opts.run_real_case = true;
        } else if (!std::strcmp(argv[i], "--no-compare")) {
            opts.compare_outputs = false;
        } else if (!std::strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return opts;
}

int main(int argc, char **argv) {
    CliOptions opts = parse_cli(argc, argv);

    if (opts.run_validate) {
        run_validation(opts);
        return 0;
    }

    if (opts.run_real_case) {
        run_real_scenario(opts);
        return 0;
    }

    run_single_benchmark(opts);
    return 0;
}
