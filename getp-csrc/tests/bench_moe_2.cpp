#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <functional>

// --- your kernels/launchers (adjust paths) ---
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include "../utils.hpp"
#include "../kernels/matmul.hpp"
#include "../kernels/moe.hpp"

// Random init helpers
static void init_bf16(std::vector<__hip_bfloat16>& dst, float scale=1.0f) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& v : dst) {
        float x = uni(rng) * scale;
        v = __float2bfloat16(x);
    }
}

static double tflops_from_gemm(long long M, long long N, long long K, double ms, int iters) {
    // 2 * M * N * K * iters FLOPs, convert to TFLOP/s
    double flops = 2.0 * (double)M * (double)N * (double)K * (double)iters;
    double sec = ms / 1e3;
    return (flops / sec) / 1e12;
}

struct DevBuf {
    __hip_bfloat16* A = nullptr; // [M,K]
    __hip_bfloat16* W = nullptr; // [N,K] (treated as [E=1,N,K])
    __hip_bfloat16* C = nullptr; // [M,N]

    int* d_expert_offsets = nullptr;
    int* d_expert_counts  = nullptr;
    int* d_tile2expert    = nullptr;
    int* d_tile2local     = nullptr;
    int* d_small_ids      = nullptr;
};

static void alloc_dev(DevBuf& db, int M, int N, int K) {
    HIP_CHECK(hipMalloc(&db.A, sizeof(__hip_bfloat16) * (size_t)M * K));
    HIP_CHECK(hipMalloc(&db.W, sizeof(__hip_bfloat16) * (size_t)N * K));
    HIP_CHECK(hipMalloc(&db.C, sizeof(__hip_bfloat16) * (size_t)M * N));
    HIP_CHECK(hipMalloc(&db.d_expert_offsets, sizeof(int) * 1));
    HIP_CHECK(hipMalloc(&db.d_expert_counts,  sizeof(int) * 1));
    HIP_CHECK(hipMalloc(&db.d_tile2expert,    sizeof(int) * 1));
    HIP_CHECK(hipMalloc(&db.d_tile2local,     sizeof(int) * 1));
    HIP_CHECK(hipMalloc(&db.d_small_ids,      sizeof(int) * 1));
}

static void free_dev(DevBuf& db) {
    HIP_CHECK(hipFree(db.A));
    HIP_CHECK(hipFree(db.W));
    HIP_CHECK(hipFree(db.C));
    HIP_CHECK(hipFree(db.d_expert_offsets));
    HIP_CHECK(hipFree(db.d_expert_counts));
    HIP_CHECK(hipFree(db.d_tile2expert));
    HIP_CHECK(hipFree(db.d_tile2local));
    HIP_CHECK(hipFree(db.d_small_ids));
    db = {};
}

struct HBuf {
    std::vector<__hip_bfloat16> hA;
    std::vector<__hip_bfloat16> hW;
    std::vector<__hip_bfloat16> hC;
};

static void upload_inputs(DevBuf& db, HBuf& hb, int M, int N, int K) {
    HIP_CHECK(hipMemcpy(db.A, hb.hA.data(), sizeof(__hip_bfloat16) * (size_t)M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.W, hb.hW.data(), sizeof(__hip_bfloat16) * (size_t)N * K, hipMemcpyHostToDevice));
}

static float time_ms(std::function<void()> f, int warmup, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) f();
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) f();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms;
}

// ---- benchmark one M ----
static void bench_M(int M, int N, int K, int iters) {
    std::cout << "M=" << std::setw(2) << M << "  N=" << N << "  K=" << K << "  iters=" << iters << "\n";

    // Host buffers
    HBuf hb;
    hb.hA.resize((size_t)M * K);
    hb.hW.resize((size_t)N * K);
    hb.hC.resize((size_t)M * N);
    init_bf16(hb.hA, 1.0f);
    init_bf16(hb.hW, 1.0f);

    // Device buffers
    DevBuf db;
    alloc_dev(db, M, N, K);
    upload_inputs(db, hb, M, N, K);

    // Expert meta (E=1)
    int h_off[1]  = {0};
    int h_cnt[1]  = {M};
    int h_t2e[1]  = {0};
    int h_t2l[1]  = {0};
    int h_small[1]= {0};
    HIP_CHECK(hipMemcpy(db.d_expert_offsets, h_off, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.d_expert_counts,  h_cnt, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.d_tile2expert,    h_t2e, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.d_tile2local,     h_t2l, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db.d_small_ids,      h_small, sizeof(int), hipMemcpyHostToDevice));

    hipStream_t stream = nullptr; // default

    // === Grouped MFMA (original) ===
    auto run_grouped = [&](){
        // For M < 64 (BLOCK_M=16*WAVES_M with WAVES_M=4), one tile is enough
        const int cur_tiles = 1;
        mlp1_optimized_outbf16<16, 16, 16, 4, 8, 4, 1, 2, 4>(
            /*C=*/db.C,
            /*A=*/db.A,
            /*W1=*/db.W, // [E=1,N,K]
            db.d_expert_offsets, db.d_expert_counts,
            db.d_tile2expert, db.d_tile2local,
            /*E=*/1, /*K=*/K, /*N=*/N, /*cur_tiles=*/cur_tiles, stream);
    };

    float ms_grouped = time_ms(run_grouped, /*warmup*/5, iters);
    HIP_CHECK(hipDeviceSynchronize());
    double tflops_grouped = tflops_from_gemm(M, N, K, ms_grouped, iters);

    // === Small-M GEMV (new) ===
    auto run_small = [&](){
        mlp1_smallM_outbf16_gemv</*TILE_K=*/256, /*WARPS=*/8>(
            /*C=*/db.C,
            /*A=*/db.A,
            /*Wbf16=*/db.W, // [E=1,N,K]
            db.d_expert_offsets, db.d_expert_counts,
            db.d_small_ids, /*num_small=*/1,
            /*K=*/K, /*N=*/N, stream);
    };

    float ms_small = time_ms(run_small, /*warmup*/5, iters);
    HIP_CHECK(hipDeviceSynchronize());
    double tflops_small = tflops_from_gemm(M, N, K, ms_small, iters);

    std::cout << std::fixed << std::setprecision(2)
              << "  Grouped(MFMA) : " << std::setw(8) << tflops_grouped << " TFLOP/s"
              << "   |   SmallM(GEMV): " << std::setw(8) << tflops_small << " TFLOP/s\n";

    free_dev(db);
}

int main() {
    // Fixed problem size per request
    const int N = 1440;
    const int K = 2880;

    // Tiny-Ms to probe (all < 16)
    std::vector<int> Ms = {1, 2, 4, 8, 12, 15};

    // Iterations per measurement (tweak if too fast/slow)
    const int iters = 8;

    // Print header
    std::cout << "Benchmark: MLP1 small-M vs grouped (bf16 in/out)\n";
    std::cout << "Sizes: N=" << N << ", K=" << K << " (FLOPs = 2*M*N*K)\n\n";

    for (int M : Ms) {
        bench_M(M, N, K, iters);
    }

    return 0;
}
