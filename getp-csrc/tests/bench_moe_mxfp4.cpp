// bench_moe_mxfp4.cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ── Project headers (adjust include paths to your tree) ─────────────────────────
#include"../utils.hpp"
#include "../kernels/matmul.hpp" // fused_attention_kernel_1warp8q_fast(...)
#include "../kernels/moe.hpp"            // grouped_mlp1_mxfp4_kernel(...)
#include "memory/mxfp4.hpp"   // device E2M1 helpers (not strictly needed here)
#include "config.hpp"         // for consistency; not used directly

// ── Safety / utils ─────────────────────────────────────────────────────────────
#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do {                                      \
    hipError_t e = (cmd);                                        \
    if (e != hipSuccess) {                                       \
        fprintf(stderr, "HIP error %s:%d: %s\n",                 \
                __FILE__, __LINE__, hipGetErrorString(e));       \
        std::fflush(stderr);                                     \
        std::abort();                                            \
    }                                                            \
} while (0)
#endif

// ── Problem config (matches your "real case") ──────────────────────────────────
static constexpr int LAYERS = 36;
static constexpr int E      = 128;   // num_experts
static constexpr int K_TOP  = 4;     // experts_per_token
static constexpr int H      = 2880;  // hidden_size
static constexpr int D      = 2880;  // intermediate_size
static constexpr int TP     = 4;     // tensor parallel
static constexpr int BATCH  = 512;   // per-rank batch
static constexpr int BGRP   = TP * BATCH;  // union batch = 2048

// Microkernel layout (must match moe.hpp)

static inline uint8_t enc_fp4_e2m1_nearest_host(float x) {
    uint8_t best = 0;
    float best_diff = std::fabs(x - MXFP4_LUT_CPU[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        float d = std::fabs(x - MXFP4_LUT_CPU[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return best;
}

static inline __hip_bfloat16 float_to_bf16(float x) {
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

// ── Routing structures ─────────────────────────────────────────────────────────
struct Routing {
    std::vector<int> counts;   // per expert
    std::vector<int> offsets;  // prefix sum
    int total_pairs;           // = BGRP * K_TOP
};

static Routing make_routing_uniform(int E_, int total_pairs) {
    Routing r;
    r.total_pairs = total_pairs;
    r.counts.assign(E_, total_pairs / E_);
    for (int i = 0, rem = total_pairs % E_; i < rem; ++i) r.counts[i]++;
    r.offsets.resize(E_);
    int acc = 0;
    for (int e = 0; e < E_; ++e) { r.offsets[e] = acc; acc += r.counts[e]; }
    return r;
}

static void build_tiles(const Routing& r,
                        std::vector<int>& tile2expert,
                        std::vector<int>& tile2local) {
    for (int e = 0; e < E; ++e) {
        const int cnt = r.counts[e];
        const int tiles = (cnt + BLOCK_M_MLP - 1) / BLOCK_M_MLP;
        for (int m = 0; m < tiles; ++m) {
            tile2expert.push_back(e);
            tile2local.push_back(m);
        }
    }
}

// ── Packed MXFP4 container per expert ──────────────────────────────────────────
struct PackedExpert {
    std::vector<uint8_t> packed;  // ceil(N*K/2)
    std::vector<float>   scales;  // ceil(N*K/32) (pre-expanded X=2^exp)
    int N, K;
};

static PackedExpert quantize_expert_to_mxfp4(const std::vector<float>& W_rowmajor, int N, int K) {
    const size_t elems = (size_t)N * (size_t)K;
    PackedExpert out;
    out.N = N; out.K = K;
    out.packed.resize((elems + 1) / 2);
    out.scales.resize((elems + 31) / 32);

    size_t byte_pos = 0;
    for (size_t base = 0; base < elems; base += 32) {
        const size_t len = std::min<size_t>(32, elems - base);

        // maxabs over the block
        float maxabs = 0.f;
        for (size_t i = 0; i < len; ++i)
            maxabs = std::max(maxabs, std::fabs(W_rowmajor[base + i]));

        int exp_scale = 0;
        if (maxabs > 0.f) {
            int e;
            std::frexp(maxabs, &e);       // maxabs = m * 2^e, 0.5 <= m < 1
            exp_scale = (e - 1) - 2;      // E2M1 largest pow2 is 2^2
        }
        const float X = std::ldexp(1.0f, exp_scale);
        out.scales[base / 32] = X;

        // quantize block & pack two nibbles per byte
        uint8_t carry = 0;
        bool have_carry = false;
        for (size_t i = 0; i < len; ++i) {
            float q = W_rowmajor[base + i] / X;
            uint8_t nib = enc_fp4_e2m1_nearest_host(q) & 0xF;
            if (!have_carry) {
                carry = nib;
                have_carry = true;
            } else {
                out.packed[byte_pos++] = (uint8_t)((nib << 4) | (carry & 0xF));
                have_carry = false;
            }
        }
        if (have_carry) { out.packed[byte_pos++] = (uint8_t)(carry & 0xF); }
    }
    return out;
}

// ── CPU reference GEMM (per expert) ────────────────────────────────────────────
// Y = A[M,K] @ W^T[K,N] where W is stored row-major [N,K] (per expert)
static void cpu_ref_mlp1(const std::vector<float>& A_pairsH,               // [total_pairs, H]
                         const std::vector<std::vector<float>>& W_e_row,   // E × [N,H]
                         const Routing& r, int N,
                         std::vector<float>& out_pairsN) {                 // [total_pairs, N]
    const int total_pairs = r.total_pairs;
    out_pairsN.assign((size_t)total_pairs * N, 0.f);
    for (int e = 0; e < E; ++e) {
        const int off = r.offsets[e], cnt = r.counts[e];
        const auto& W = W_e_row[e]; // [N,H] row-major
        for (int p = 0; p < cnt; ++p) {
            const float* a = &A_pairsH[(size_t)(off + p) * H];
            float* y = &out_pairsN[(size_t)(off + p) * N];
            for (int n = 0; n < N; ++n) {
                const float* wrow = &W[(size_t)n * H];
                float s = 0.f;
                for (int k = 0; k < H; ++k) s += a[k] * wrow[k];
                y[n] = s;
            }
        }
    }
}

// ── Kernel launch wrappers ─────────────────────────────────────────────────────
static void launch_grouped_mlp1_baseline(
    float* dC, const float* dA,
    const uint8_t* dW_packed, const float* dScales,
    const int* d_expert_offsets, const int* d_expert_counts,
    const int* d_tile2expert, const int* d_tile2local,
    int N, int Kdim, int cur_tiles, hipStream_t stream)
{
    // return;
    dim3 grid((N + BLOCK_N_MLP - 1) / BLOCK_N_MLP, cur_tiles);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK_MLP);

    const int ldA = BLOCK_K_MLP + PAD_K_MLP;
    const int ldB = BLOCK_K_MLP + PAD_K_MLP;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(2 * BLOCK_M_MLP * ldA + 2 * ldB * BLOCK_N_MLP);

    hipLaunchKernelGGL(grouped_mlp1_mxfp4_kernel, grid, block, shmem_bytes, stream,
        dC, dA, dW_packed, dScales,
        d_expert_offsets, d_expert_counts,
        d_tile2expert, d_tile2local,
        /*E=*/E, /*K=*/Kdim, /*N=*/N);
    HIP_CHECK(hipGetLastError());
}

static void launch_grouped_mlp1_optimized(
    float* dC, const __hip_bfloat16* dA_bf16,
    const uint8_t* dW_packed, const float* dScales,
    const int* d_expert_offsets, const int* d_expert_counts,
    const int* d_tile2expert, const int* d_tile2local,
    int E, int N, int Kdim, int cur_tiles, hipStream_t stream)
{
    mlp1_mxfp4_optimized_unified_A_bf16<
        16, 16, 16,
        WAVES_M_MLP, WAVES_N_MLP, WAVES_K_MLP,
        1, 2,
        PAD_K_MLP>(
            dC, dA_bf16, dW_packed, dScales,
            d_expert_offsets, d_expert_counts,
            d_tile2expert, d_tile2local,
            E, Kdim, N, cur_tiles, stream);
}


// ── Simple CLI ────────────────────────────────────────────────────────────────
struct Args {
    int rank      = 0;     // TP rank to simulate
    int iters     = 100;   // timing iterations
    bool do_check = true;  // CPU+baseline checks
    unsigned seed = 12345; // RNG seed
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto eat = [&](int& var){
            if (i + 1 < argc) { var = std::atoi(argv[++i]); }
        };
        if (s == "--rank") { eat(a.rank); }
        else if (s == "--iters") { eat(a.iters); }
        else if (s == "--nocheck") { a.do_check = false; }
        else if (s == "--seed") { if (i + 1 < argc) a.seed = (unsigned)std::stoul(argv[++i]); }
        else if (s == "--help" || s == "-h") {
            printf("Usage: %s [--rank N] [--iters N] [--nocheck] [--seed S]\n", argv[0]);
            std::exit(0);
        }
    }
    if (a.rank < 0 || a.rank >= TP) {
        fprintf(stderr, "Invalid --rank %d (must be 0..%d)\n", a.rank, TP - 1);
        std::exit(2);
    }
    return a;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // rank-local shard for MLP1 (column-parallel on O=2D)
    const int twoD = 2 * D;
    const int base = twoD / TP, rem = twoD % TP;
    const int o_len = base + (args.rank < rem ? 1 : 0);

    printf("[cfg] H=%d D=%d twoD=%d E=%d K=%d TP=%d rank=%d o_len=%d Bgrp=%d layers=%d\n",
           H, D, twoD, E, K_TOP, TP, args.rank, o_len, BGRP, LAYERS);

    // ── Build a uniform routing map over the union batch ──────────────────────
    const int total_pairs = BGRP * K_TOP; // 2048 * 4 = 8192
    Routing routing = make_routing_uniform(E, total_pairs);

    std::vector<int> tile2expert, tile2local;
    build_tiles(routing, tile2expert, tile2local);
    const int cur_tiles = (int)tile2expert.size();
    printf("[routing] total_pairs=%d tiles=%d (BLOCK_M=%d)\n",
           total_pairs, cur_tiles, BLOCK_M_MLP);

    // ── Host buffers ──────────────────────────────────────────────────────────
    std::mt19937 rng(args.seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // A buffer: [total_pairs, H] in expert-concatenated order (what grouped kernel expects)
    std::vector<float> hA((size_t)total_pairs * H);
    for (auto& v : hA) v = dist(rng);
    std::vector<__hip_bfloat16> hA_bf16(hA.size());
    for (size_t i = 0; i < hA.size(); ++i) {
        hA_bf16[i] = float_to_bf16(hA[i]);
    }

    // Per-expert float weights: row-major [N=o_len, K=H]
    std::vector<std::vector<float>> hW_e_row(E);
    for (int e = 0; e < E; ++e) {
        hW_e_row[e].resize((size_t)o_len * H);
        for (auto& v : hW_e_row[e]) v = dist(rng);
    }

    // Quantize each expert to MXFP4 (packed nibbles + f32 scales)
    const size_t seg_elems        = (size_t)o_len * H;
    const size_t seg_packed_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks       = (seg_elems + 31) / 32;

    std::vector<uint8_t> hW_layer_packed((size_t)E * seg_packed_bytes);
    std::vector<float>   hS_layer_scales((size_t)E * seg_blocks);

    for (int e = 0; e < E; ++e) {
        PackedExpert pe = quantize_expert_to_mxfp4(hW_e_row[e], o_len, H);
        std::memcpy(&hW_layer_packed[(size_t)e * seg_packed_bytes],
                    pe.packed.data(), pe.packed.size());
        std::memcpy(&hS_layer_scales[(size_t)e * seg_blocks],
                    pe.scales.data(), pe.scales.size() * sizeof(float));
    }

    // ── Device buffers ────────────────────────────────────────────────────────
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    float*    dA = nullptr;
    __hip_bfloat16* dA_bf16 = nullptr;
    float*    dC_base = nullptr;
    float*    dC_opt  = nullptr;
    uint8_t*  dW_packed = nullptr;
    float*    dS_scales = nullptr;
    int *d_counts = nullptr, *d_offsets = nullptr, *d_t2e = nullptr, *d_t2l = nullptr;

    HIP_CHECK(hipMalloc(&dA,        hA.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC_base,   (size_t)total_pairs * o_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC_opt,    (size_t)total_pairs * o_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&dW_packed, hW_layer_packed.size() * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&dS_scales, hS_layer_scales.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_counts,  E * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_offsets, E * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_t2e,     tile2expert.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_t2l,     tile2local.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&dA_bf16,   hA_bf16.size() * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMemcpyAsync(dA, hA.data(), hA.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dA_bf16, hA_bf16.data(), hA_bf16.size() * sizeof(__hip_bfloat16),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dW_packed, hW_layer_packed.data(), hW_layer_packed.size(),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(dS_scales, hS_layer_scales.data(), hS_layer_scales.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_counts,  routing.counts.data(),  E * sizeof(int),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_offsets, routing.offsets.data(), E * sizeof(int),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_t2e,     tile2expert.data(), tile2expert.size() * sizeof(int),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_t2l,     tile2local.data(),  tile2local.size()  * sizeof(int),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // ── Correctness check (CPU reference and baseline) ────────────────────────


    // ── Timing: baseline ──────────────────────────────────────────────────────
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));

    // warmup
    for (int w = 0; w < 5; ++w) {
        launch_grouped_mlp1_baseline(
            dC_base, dA, dW_packed, dS_scales,
            d_offsets, d_counts, d_t2e, d_t2l,
            o_len, H, cur_tiles, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipEventRecord(e0, stream));
    for (int it = 0; it < args.iters; ++it) {
        launch_grouped_mlp1_baseline(
            dC_base, dA, dW_packed, dS_scales,
            d_offsets, d_counts, d_t2e, d_t2l,
            o_len, H, cur_tiles, stream);
    }
    HIP_CHECK(hipEventRecord(e1, stream));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_base = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_base, e0, e1));
    ms_base /= args.iters;

    const double flops = 2.0 * (double)total_pairs * (double)H * (double)o_len;
    const double tflops_base = (flops / 1.0e12) / (ms_base / 1.0e3);

    printf("[perf] baseline : %.3f ms | %.2f TFLOP/s (M=%d K=%d N=%d pairs=%d)\n",
           ms_base, tflops_base, BLOCK_M_MLP, H, o_len, total_pairs);

    // ── Timing: optimized (currently same kernel) ─────────────────────────────
    for (int w = 0; w < 5; ++w) {
        launch_grouped_mlp1_optimized(
            dC_opt, dA_bf16, dW_packed, dS_scales,
            d_offsets, d_counts, d_t2e, d_t2l, E,
            o_len, H, cur_tiles, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipEventRecord(e0, stream));
    for (int it = 0; it < args.iters; ++it) {
        launch_grouped_mlp1_optimized(
            dC_opt, dA_bf16, dW_packed, dS_scales,
            d_offsets, d_counts, d_t2e, d_t2l, E,
            o_len, H, cur_tiles, stream);
    }
    HIP_CHECK(hipEventRecord(e1, stream));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms_opt = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_opt, e0, e1));
    ms_opt /= args.iters;

    const double tflops_opt = (flops / 1.0e12) / (ms_opt / 1.0e3);
    printf("[perf] optimized: %.3f ms | %.2f TFLOP/s | speedup = %.3fx\n",
           ms_opt, tflops_opt, ms_base / ms_opt);

    // sanity: optimized vs baseline numerical diff
    if (args.do_check) {
        std::vector<float> Cb((size_t)total_pairs * o_len), Co(Cb.size());
        HIP_CHECK(hipMemcpy(Cb.data(), dC_base, Cb.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Co.data(), dC_opt,  Co.size() * sizeof(float), hipMemcpyDeviceToHost));
        double max_abs = 0.0, mean_abs = 0.0;
        for (size_t i = 0; i < Cb.size(); ++i) {
            double d = std::fabs(Cb[i] - Co[i]);
            max_abs = std::max(max_abs, d);
            mean_abs += d;
        }
        mean_abs /= Cb.size();
        printf("[check] optimized vs baseline: max_abs=%.6f  mean_abs=%.6f\n", max_abs, mean_abs);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    hipEventDestroy(e0); hipEventDestroy(e1);
    hipStreamDestroy(stream);
    hipFree(dA); hipFree(dA_bf16);
    hipFree(dC_base); hipFree(dC_opt);
    hipFree(dW_packed); hipFree(dS_scales);
    hipFree(d_counts); hipFree(d_offsets);
    hipFree(d_t2e); hipFree(d_t2l);
    return 0;
}
