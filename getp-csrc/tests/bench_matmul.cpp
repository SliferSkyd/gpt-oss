// bench_matmul.cpp
// Benchmark and correctness-check for the final logits matmul using the
// current MFMA kernel defined in matmul.hpp. No rocBLAS.
//
// Op: C[M,N] = A[M,K] * W^T[N,K]   (A: f32 row-major, W: bf16 row-major [N,K], C: f32 row-major)
//
// Build (example):
//   hipcc -O3 -std=c++17 -march=gfx90a bench_matmul.cpp -o bench_matmul
//
// Usage examples:
//   ./bench_matmul --mode=check --small-check
//   ./bench_matmul --mode=perf --iters=50 --warmup=10
//   ./bench_matmul --mode=real --steps=1024
//
// Defaults are chosen to match the head GEMM in run.cpp:
//   M=512 (BATCH_SIZE), K=2880 (hidden), N=201088 (vocab)

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>


#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <functional>

// Bring in the kernel/launcher under test
#include "../utils.hpp"
#include "../kernels/matmul.hpp"

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d -> %s\n", hipGetErrorName(_e), __FILE__, __LINE__, hipGetErrorString(_e)); \
        std::exit(1); \
    } \
} while(0)
#endif

// -----------------------------
// CLI parsing (simple)
// -----------------------------
struct Options {
    std::string mode = "perf";  // check | perf | real
    int M = 512;
    int K = 2880;
    int N = 201088;
    int iters = 50;
    int warmup = 10;
    uint64_t seed = 123;
    int steps = 1024;            // for --mode=real
    bool small_check = false;    // use a reduced shape for CPU reference
    bool dump_maxerr = true;
};

static bool parse_flag(int argc, char** argv, const char* flag) {
    for (int i=0;i<argc;++i) if (std::strcmp(argv[i], flag)==0) return true; return false;
}
static const char* parse_opt(int argc, char** argv, const char* key) {
    for (int i=0;i<argc-1;++i) if (std::strcmp(argv[i], key)==0) return argv[i+1]; return nullptr;
}

static Options parse_cli(int argc, char** argv) {
    Options o;
    if (const char* v = parse_opt(argc, argv, (char*)"--mode")) o.mode = v;
    if (const char* v = parse_opt(argc, argv, (char*)"--M")) o.M = std::atoi(v);
    if (const char* v = parse_opt(argc, argv, (char*)"--K")) o.K = std::atoi(v);
    if (const char* v = parse_opt(argc, argv, (char*)"--N")) o.N = std::atoi(v);
    if (const char* v = parse_opt(argc, argv, (char*)"--iters")) o.iters = std::atoi(v);
    if (const char* v = parse_opt(argc, argv, (char*)"--warmup")) o.warmup = std::atoi(v);
    if (const char* v = parse_opt(argc, argv, (char*)"--seed")) o.seed = std::strtoull(v, nullptr, 10);
    if (const char* v = parse_opt(argc, argv, (char*)"--steps")) o.steps = std::atoi(v);
    o.small_check = parse_flag(argc, argv, (char*)"--small-check");
    o.dump_maxerr = !parse_flag(argc, argv, (char*)"--no-dump-maxerr");
    return o;
}

// -----------------------------
// Host-side bf16 helpers
// -----------------------------
static inline uint16_t float_to_bf16_bits(float f) {
    // Round-to-nearest-even conversion from IEEE754 float32 to bfloat16 bits
    union { uint32_t u; float f; } x; x.f = f;
    uint32_t u = x.u;
    uint32_t lsb = (u >> 16) & 1u;                 // tie-breaker bit
    uint32_t rounding_bias = 0x00007FFFu + lsb;    // RNE
    u += rounding_bias;
    uint16_t h = static_cast<uint16_t>(u >> 16);
    return h;
}
static inline float bf16_bits_to_float(uint16_t h) {
    union { uint32_t u; float f; } x; x.u = static_cast<uint32_t>(h) << 16;
    return x.f;
}

// -----------------------------
// Random init
// -----------------------------
static void init_random_f32(float* p, size_t n, uint64_t seed, float lo=-0.5f, float hi=0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i=0;i<n;++i) p[i] = dist(rng);
}

// -----------------------------
// CPU reference: C = A * W^T  (W stored as [N,K] bf16)
// -----------------------------
static void cpu_reference_matmul(float* C, const float* A, const uint16_t* Wbf16_bits,
                                 int M, int K, int N) {
    // Row-major storage. Compute C[m,n] = sum_k A[m,k] * float(W[n,k])
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0; // use double on CPU for a stable reference
            const float* arow = A + (size_t)m * K;
            const uint16_t* wrow = Wbf16_bits + (size_t)n * K; // W is [N,K]
            for (int k=0;k<K;++k) {
                float w = bf16_bits_to_float(wrow[k]);
                acc += (double)arow[k] * (double)w;
            }
            C[(size_t)m * N + n] = static_cast<float>(acc);
        }
    }
}

// -----------------------------
// Device helpers
// -----------------------------
__global__ void add_alpha_kernel(float* A, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = i; idx < n; idx += stride) A[idx] += alpha;
}

static void device_add_alpha(float* dA, float alpha, int M, int K) {
    int total = M * K;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_alpha_kernel<<<grid, block>>>(dA, alpha, total);
    HIP_CHECK(hipGetLastError());
}

// -----------------------------
// Timing
// -----------------------------
struct TimingStats { double mean_ms=0, median_ms=0, min_ms=0, max_ms=0; };

static TimingStats time_kernel(std::function<void()> fn, int warmup, int iters) {
    hipEvent_t start, stop; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
    for (int i=0;i<warmup;++i) { fn(); }
    std::vector<float> times;
    times.reserve(iters);
    for (int i=0;i<iters;++i) {
        HIP_CHECK(hipEventRecord(start));
        fn();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms=0; HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    std::sort(times.begin(), times.end());
    TimingStats s;
    s.median_ms = times[times.size()/2];
    s.min_ms = times.front();
    s.max_ms = times.back();
    double sum=0; for (float v: times) sum += v; s.mean_ms = sum / times.size();
    return s;
}

static double tflops(int M, int N, int K, double ms) {
    // 2*M*N*K flops per GEMM
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / (ms * 1e6); // TFLOP/s
}

// -----------------------------
// Baseline runner (current kernel)
// -----------------------------
static void run_baseline(float* dC, const float* dA, const __hip_bfloat16* dW, int M, int K, int N) {
    // Use the same template args as in run.cpp final matmul
    matmul<16,16,16, 4,4,2, 1,1, 8, false>(dC, dA, dW, M, K, N, nullptr, nullptr);
}

// -----------------------------
// Error metrics
// -----------------------------
struct ErrStats { double max_abs=0, mean_abs=0, max_rel=0; };

static ErrStats compare_host(const float* x, const float* y, size_t n) {
    double max_abs=0, mean_abs=0, max_rel=0;
    for (size_t i=0;i<n;++i) {
        double a = (double)x[i];
        double b = (double)y[i];
        double da = std::abs(a-b);
        max_abs = std::max(max_abs, da);
        mean_abs += da;
        double denom = std::abs(b) + 1e-7;
        double dr = da / denom;
        max_rel = std::max(max_rel, dr);
    }
    mean_abs /= (double)n;
    return {max_abs, mean_abs, max_rel};
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv) {
    Options opt = parse_cli(argc, argv);

    // Device info
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s", prop.name);

    int M = opt.M, K = opt.K, N = opt.N;

    if (opt.mode == "check" && opt.small_check) {
        // Pick a small, fast shape that still exercises vector paths
        M = std::min(M, 64);
        K = ((std::min(K, 256) + 7) / 8) * 8;   // make divisible by 8
        N = std::min(N, 4096);
    }

    const size_t sizeA = (size_t)M * K;
    const size_t sizeW = (size_t)N * K; // [N,K]
    const size_t sizeC = (size_t)M * N;

    printf("Op: C[%d,%d] = A[%d,%d] * W^T[%d,%d]\n", M, N, M, K, N, K);
    printf("A: f32, W: bf16, C: f32\n");

    // Host allocations (pinned for speed if possible)
    float *hA=nullptr, *hC=nullptr, *hCref=nullptr;
    uint16_t *hW_bf16_bits=nullptr; // host bf16 as raw bits

    if (hipHostMalloc((void**)&hA,     sizeA*sizeof(float)) != hipSuccess) hA = (float*)std::malloc(sizeA*sizeof(float));
    if (hipHostMalloc((void**)&hC,     sizeC*sizeof(float)) != hipSuccess) hC = (float*)std::malloc(sizeC*sizeof(float));
    if (hipHostMalloc((void**)&hCref,  sizeC*sizeof(float)) != hipSuccess) hCref = (float*)std::malloc(sizeC*sizeof(float));
    if (hipHostMalloc((void**)&hW_bf16_bits, sizeW*sizeof(uint16_t)) != hipSuccess) hW_bf16_bits = (uint16_t*)std::malloc(sizeW*sizeof(uint16_t));

    if (!hA || !hC || !hCref || !hW_bf16_bits) {
        fprintf(stderr, "Host allocation failed.\n");
        return 1;
    }

    // Init A and W (random), convert W to bf16 bits
    init_random_f32(hA, sizeA, opt.seed + 1);
    {
        // generate W in float then cast to bf16
        std::vector<float> hW_f32(sizeW);
        init_random_f32(hW_f32.data(), sizeW, opt.seed + 2);
        for (size_t i=0;i<sizeW;++i) hW_bf16_bits[i] = float_to_bf16_bits(hW_f32[i]);
    }

    // Device allocations
    float *dA=nullptr, *dC=nullptr; __hip_bfloat16* dW=nullptr;
    HIP_CHECK(hipMalloc(&dA, sizeA*sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, sizeC*sizeof(float)));
    HIP_CHECK(hipMalloc(&dW, sizeW*sizeof(__hip_bfloat16)));

    // H2D
    HIP_CHECK(hipMemcpy(dA, hA, sizeA*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, hW_bf16_bits, sizeW*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, sizeC*sizeof(float)));

    if (opt.mode == "check") {
        // Run baseline on device
        run_baseline(dC, dA, dW, M, K, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipGetLastError());

        // Copy to host
        HIP_CHECK(hipMemcpy(hC, dC, sizeC*sizeof(float), hipMemcpyDeviceToHost));

        // CPU reference on host (may be slow if large; we reduced with --small-check)
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_reference_matmul(hCref, hA, hW_bf16_bits, M, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count();

        ErrStats err = compare_host(hC, hCref, sizeC);
        printf("CPU reference time: %.2f ms\n", ms_ref);
        if (opt.dump_maxerr) {
            printf("Errors: max_abs=%.6g  mean_abs=%.6g  max_rel=%.6g\n", err.max_abs, err.mean_abs, err.max_rel);
        }
        // Simple thresholds suitable for bf16 weights + f32 accum
        const double MAX_ABS_T = 2e-1, MAX_REL_T = 1e-2;
        bool pass = (err.max_abs <= MAX_ABS_T) || (err.max_rel <= MAX_REL_T);
        printf("CHECK %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 2;
    }

    if (opt.mode == "perf") {
        auto fn = [&]() { run_baseline(dC, dA, dW, M, K, N); };
        TimingStats stats = time_kernel(fn, opt.warmup, opt.iters);
        printf("\nMode: perf  iters=%d warmup=%d\n", opt.iters, opt.warmup);
        printf("-------------------------------------------------------------\n");
        printf("Kernel           |  mean ms |  median |   min   |  TFLOP/s\n");
        printf("-------------------------------------------------------------\n");
        printf("baseline(mfma)   | %8.3f | %7.3f | %7.3f | %8.2f\n",
               stats.mean_ms, stats.median_ms, stats.min_ms, tflops(M,N,K,stats.mean_ms));
        return 0;
    }

    if (opt.mode == "real") {
        // Simulate decode steps. Mutate A slightly each step to avoid perfect cache behavior.
        printf("\nMode: real  steps=%d (decode simulation)\n", opt.steps);
        hipEvent_t start, stop; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (int s=0; s<opt.steps; ++s) {
            float alpha = 1e-5f * float(s & 7);
            if (alpha != 0.0f) device_add_alpha(dA, alpha, M, K);
            run_baseline(dC, dA, dW, M, K, N);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float total_ms=0; HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
        double avg_ms = total_ms / std::max(1, opt.steps);
        printf("Total time: %.2f ms  |  Avg per step: %.3f ms  |  Avg TFLOP/s: %.2f\n",
               total_ms, avg_ms, tflops(M,N,K,avg_ms));
        return 0;
    }

    fprintf(stderr, "Unknown --mode=\"%s\". Use check | perf | real\n", opt.mode.c_str());
    return 1;
}
Collapse














Message Thanh Nguyen









