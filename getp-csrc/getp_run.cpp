// TODO: Modify this file to optimize end-to-end throughput with HIP GPU acceleration

#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include "../include/utils.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#ifndef GETP_RUN
#define GETP_RUN

// BATCH_SIZE can be increased for higher throughput
#define BATCH_SIZE 72
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

// CPU variables (used only in warmup)
float *cos_vals_cpu, *sin_vals_cpu;
int **prompt_tokens;
int *current_tokens;
bool *finished;
int *positions;
int *prompt_lens;

// GPU variables
float *d_cos_vals, *d_sin_vals;
float *d_x, *d_t, *d_tb, *d_tb2, *d_qkv, *d_q, *d_k, *d_v;
float *d_key_cache, *d_value_cache, *d_att, *d_logits;
float *d_router_score, *d_topk_v, *d_mlp1_out, *d_gate, *d_up, *d_gate_up, *d_e_agg;
float *d_mask, *d_expert_input_buffer;
int *d_topk_i, *d_current_tokens, *d_positions;
float *d_temp_buffer;

// GPU weight pointers - CRITICAL FIX: Copy all weights to GPU
__hip_bfloat16 *d_attn_sinks;
__hip_bfloat16 *d_token_embedding_table;
__hip_bfloat16 *d_rms_attn_w, *d_rms_ffn_w, *d_rms_out_w;
__hip_bfloat16 *d_w_qkv, *d_b_qkv, *d_w_o, *d_b_o;
__hip_bfloat16 *d_w_router, *d_b_router;
__hip_bfloat16 *d_w_mlp1, *d_b_mlp1, *d_w_mlp2, *d_b_mlp2;
__hip_bfloat16 *d_out_w;

// Add to list of GPU variables
int *d_expert_indices;
float *d_expert_weights;
int *d_batch_count; // A single integer on the GPU
float *d_expert_output_buffer;

// HIP error checking macro
#define HIP_CHECK(call)                                                                                 \
    do                                                                                                  \
    {                                                                                                   \
        hipError_t error = call;                                                                        \
        if (error != hipSuccess)                                                                        \
        {                                                                                               \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

// Float to bfloat16 conversion functions using library function
void convert_float_array_to_bfloat16(const float *src, __hip_bfloat16 *dst, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void debug(float *d_val, int size = 1)
{
    float *h_val = (float *)malloc(size * sizeof(float));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_val, d_val, size * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < size; i++)
    {
        fprintf(stderr, "Debug value %d: %f\n", i, h_val[i]);
    }
    free(h_val);
    exit(0);
}

//------------------------------------------------------------------------------//
// Timer class for measuring execution time                                     //
//------------------------------------------------------------------------------//

struct Timer
{
    // hip events
    hipEvent_t hip_start, hip_stop;
    // CPU timing
    std::chrono::high_resolution_clock::time_point cpu_start;

    std::string name;
    std::ofstream log_file;
    bool use_hip;

    Timer(const std::string &timer_name, bool is_hip = false, const std::string &filename = "times.csv")
        : name(timer_name), use_hip(is_hip)
    {

        if (use_hip)
        {
            hipEventCreate(&hip_start);
            hipEventCreate(&hip_stop);
        }

        log_file.open(filename, std::ios::app);
        if (log_file.tellp() == 0)
        {
            log_file << "name,time_ms\n";
        }
    }

    ~Timer()
    {
        if (use_hip)
        {
            hipEventDestroy(hip_start);
            hipEventDestroy(hip_stop);
        }
        log_file.close();
    }

    void start()
    {
        if (use_hip)
        {
            hipEventRecord(hip_start);
        }
        else
        {
            cpu_start = std::chrono::high_resolution_clock::now();
        }
    }

    void stop()
    {
        float ms = 0;

        if (use_hip)
        {
            hipEventRecord(hip_stop);
            hipEventSynchronize(hip_stop);
            hipEventElapsedTime(&ms, hip_start, hip_stop);
        }
        else
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - cpu_start);
            ms = duration.count() / 1000.0f;
        }

        log_file << name << "," << ms << "\n";
        log_file.flush();
    }
};

// RAII wrapper
struct ScopedTimer
{
    Timer &timer;
    ScopedTimer(Timer &t) : timer(t) { timer.start(); }
    ~ScopedTimer() { timer.stop(); }
};

#define TIME_SCOPE(timer) ScopedTimer _t(timer)

// CPU warmup functions (same as before)
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                             float scaling_factor,
                                             float initial_context_length,
                                             float ntk_beta, float ntk_alpha,
                                             float *concentration_out,
                                             float *inv_freq_out)
{
    int d_half = head_dim / 2;
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++)
    {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f)
    {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;
        float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
        float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
        assert(0 < low && low < high && high < d_half - 1);

        for (int i = 0; i < d_half; i++)
        {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];
            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0)
                ramp = 0;
            if (ramp > 1)
                ramp = 1;
            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    }
    else
    {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++)
        {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }
    *concentration_out = concentration;
    free(freq);
}

void compute_cos_sin_getp(int pos, float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha, float *cos_out, float *sin_out)
{
    int d_half = head_dim / 2;
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                            initial_context_length, ntk_beta,
                                            ntk_alpha, &concentration, inv_freq);

    for (int j = 0; j < d_half; j++)
    {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }
    free(inv_freq);
}

// GPU kernels with bfloat16 weights support
__global__ void rmsnorm_kernel(float *output, const float *input, const __hip_bfloat16 *weight,
                               int batch_size, int size)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    const float *x = input + batch_idx * size;
    float *o = output + batch_idx * size;

    // Shared memory for reduction
    __shared__ float shared_ss[THREADS_PER_BLOCK];

    // Calculate sum of squares
    float ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        ss += x[i] * x[i];
    }
    shared_ss[tid] = ss;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_ss[tid] += shared_ss[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        ss = shared_ss[0] / size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss[0] = ss;
    }
    __syncthreads();

    ss = shared_ss[0];

    // Normalize and scale - convert bfloat16 weight to fp32 on-the-fly
    for (int i = tid; i < size; i += blockDim.x)
    {
        float weight_fp32 = __bfloat162float(weight[i]);
        o[i] = weight_fp32 * (ss * x[i]);
    }
}

__global__ void softmax_kernel(float *x, int batch_size, int size)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    float *batch_x = x + batch_idx * size;

    __shared__ float shared_max[THREADS_PER_BLOCK];
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    // Find max value
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x)
    {
        max_val = fmaxf(max_val, batch_x[i]);
    }
    shared_max[tid] = max_val;
    __syncthreads();

    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    max_val = shared_max[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_x[i] = expf(batch_x[i] - max_val);
        sum += batch_x[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    sum = shared_sum[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_x[i] /= sum;
    }
}

__global__ void accumulate_kernel(float *a, const float *b, float factor,
                                  int batch_size, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * size;

    if (idx < total_size)
    {
        a[idx] += b[idx] * factor;
    }
}

// NEW: Kernel to add bias to matrix multiplication result with bfloat16 bias
__global__ void add_bias_kernel(float *output, const __hip_bfloat16 *bias, int batch_size, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size)
    {
        int dim_idx = idx % size;
        // Convert bfloat16 bias to fp32 on-the-fly
        float bias_fp32 = __bfloat162float(bias[dim_idx]);
        output[idx] += bias_fp32;
    }
}

__global__ void apply_rotary_emb_kernel(float *x, const float *cos_vals, const float *sin_vals,
                                        const int *positions, int batch_size,
                                        int n_heads, int head_dim)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int half = head_dim / 2;
    if (dim_idx >= half)
        return;

    int pos = positions[batch_idx];
    float *x_batch = x + batch_idx * n_heads * head_dim;
    const float *cos_pos = cos_vals + pos * half;
    const float *sin_pos = sin_vals + pos * half;

    float x1 = x_batch[head_idx * head_dim + dim_idx];
    float x2 = x_batch[head_idx * head_dim + half + dim_idx];

    float c = cos_pos[dim_idx];
    float s = sin_pos[dim_idx];

    x_batch[head_idx * head_dim + dim_idx] = x1 * c - x2 * s;
    x_batch[head_idx * head_dim + half + dim_idx] = x2 * c + x1 * s;
}

// NEW: KV cache update kernel
__global__ void update_kv_cache_kernel(float *key_cache, float *value_cache,
                                       const float *k, const float *v,
                                       const int *positions, int batch_size,
                                       int n_layers, int layer_idx, int seq_len,
                                       int kv_dim)
{
    int batch_idx = blockIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || dim_idx >= kv_dim)
        return;

    int pos = positions[batch_idx];
    if (pos >= seq_len)
        return; // Safety check

    // Update key cache
    int k_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    key_cache[k_cache_idx] = k[batch_idx * kv_dim + dim_idx];

    // Update value cache
    int v_cache_idx = batch_idx * n_layers * seq_len * kv_dim +
                      layer_idx * seq_len * kv_dim + pos * kv_dim + dim_idx;
    value_cache[v_cache_idx] = v[batch_idx * kv_dim + dim_idx];
}

__global__ void attention_scores_kernel(float *att, const float *q, const float *key_cache,
                                        const float *mask, const int *positions,
                                        int batch_size, int n_heads, int head_dim,
                                        int seq_len, int n_layers, int layer_idx,
                                        bool use_sliding_window)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_idx >= batch_size || head_idx >= n_heads)
        return;

    int pos = positions[batch_idx];
    if (t > pos)
        return;

    int kv_dim = head_dim * (n_heads / 8); // Assuming GQA with 4:1 ratio
    int kv_head = head_idx / 8;

    const float *q_head = q + batch_idx * n_heads * head_dim + head_idx * head_dim;
    const float *k_head = key_cache + batch_idx * n_layers * seq_len * kv_dim +
                          layer_idx * seq_len * kv_dim + t * kv_dim + kv_head * head_dim;

    float score = 0.0f;
    for (int i = 0; i < head_dim; i++)
    {
        score += q_head[i] * k_head[i];
    }
    score /= sqrtf((float)head_dim);

    // Apply sliding window mask if enabled
    if (use_sliding_window && (layer_idx % 2 == 0))
    {
        score += mask[pos * seq_len + t];
    }

    att[batch_idx * n_heads * seq_len + head_idx * seq_len + t] = score;
}

__global__ void attention_weighted_sum_kernel(float *output, const float *att,
                                              const float *value_cache, const int *positions,
                                              int batch_size, int n_heads, int head_dim,
                                              int seq_len, int n_layers, int layer_idx)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= n_heads || dim_idx >= head_dim)
        return;

    int pos = positions[batch_idx];
    int kv_dim = head_dim * (n_heads / 8);
    int kv_head = head_idx / 8;

    const float *att_head = att + batch_idx * n_heads * seq_len + head_idx * seq_len;
    float *out_head = output + batch_idx * n_heads * head_dim + head_idx * head_dim;

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const float *v_head = value_cache + batch_idx * n_layers * seq_len * kv_dim +
                              layer_idx * seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
        sum += att_head[t] * v_head[dim_idx];
    }
    out_head[dim_idx] = sum;
}

__global__ void swiglu_kernel(float *gate, float *up, float *output,
                              int batch_size, int intermediate_dim, float limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / intermediate_dim;
    int dim_idx = idx % intermediate_dim;

    if (batch_idx >= batch_size || dim_idx >= intermediate_dim)
        return;

    const float alpha = 1.702f;
    float gate_val = gate[idx];
    float up_val = up[idx];

    // Clamping
    gate_val = fminf(fmaxf(gate_val, -limit), limit);
    up_val = fminf(fmaxf(up_val, -limit), limit);

    // SiLU activation
    gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));
    gate_val *= (up_val + 1.0f);

    output[idx] = gate_val;
}

__global__ void split_gate_up_kernel(float *gate, float *up, const float *mlp1_out,
                                     const __hip_bfloat16 *bias, int batch_size, int intermediate_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / intermediate_dim;
    int dim_idx = idx % intermediate_dim;

    if (batch_idx >= batch_size || dim_idx >= intermediate_dim)
        return;

    int mlp1_idx = batch_idx * 2 * intermediate_dim;
    // Convert bfloat16 bias to fp32 on-the-fly
    float bias_gate_fp32 = __bfloat162float(bias[2 * dim_idx]);
    float bias_up_fp32 = __bfloat162float(bias[2 * dim_idx + 1]);

    gate[idx] = mlp1_out[mlp1_idx + 2 * dim_idx] + bias_gate_fp32;
    up[idx] = mlp1_out[mlp1_idx + 2 * dim_idx + 1] + bias_up_fp32;
}

__global__ void topk_kernel(float *topk_values, int *topk_indices, const float *scores,
                            int batch_size, int n_experts, int k)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    const float *batch_scores = scores + batch_idx * n_experts;
    float *batch_topk_v = topk_values + batch_idx * k;
    int *batch_topk_i = topk_indices + batch_idx * k;

    // Simple selection sort for top-k (works well for small k)
    for (int i = 0; i < k; i++)
    {
        float max_val = -INFINITY;
        int max_idx = -1;

        for (int j = 0; j < n_experts; j++)
        {
            bool already_selected = false;
            for (int prev = 0; prev < i; prev++)
            {
                if (batch_topk_i[prev] == j)
                {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && batch_scores[j] > max_val)
            {
                max_val = batch_scores[j];
                max_idx = j;
            }
        }

        batch_topk_v[i] = max_val;
        batch_topk_i[i] = max_idx;
    }
}

__global__ void copy_embeddings_kernel(float *output, const __hip_bfloat16 *embeddings,
                                       const int *tokens, int batch_size, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_dim;
    int dim_idx = idx % hidden_dim;

    if (batch_idx >= batch_size || dim_idx >= hidden_dim)
        return;
    int token = tokens[batch_idx];
    if (token < 0)
        return; // Safety check for invalid tokens

    // Convert bfloat16 embedding to fp32 on-the-fly
    float embedding_fp32 = __bfloat162float(embeddings[token * hidden_dim + dim_idx]);
    output[idx] = embedding_fp32;
}

#include <rocwmma/rocwmma.hpp>

// Wave-level WMMA tile sizes (fixed by hardware)
constexpr int WM = 16;   // wmma M
constexpr int WN = 16;   // wmma N
constexpr int WK = 16;   // wmma K  (rocWMMA handles issuing 2xK=8 mfmas for BF16 under the hood)

// Block tile = 64x64 made of 4x4 WMMA tiles (16 waves per block)
constexpr int WAVES_M = 4;                   // number of WMMA tiles along M in a block
constexpr int WAVES_N = 4;                   // number of WMMA tiles along N in a block
constexpr int BLOCK_M = WM * WAVES_M;        // 64
constexpr int BLOCK_N = WN * WAVES_N;        // 64
constexpr int BLOCK_K = WK;                  // iterate over K in steps of 16

// Workgroup: 64 lanes per wave x (WAVES_M * WAVES_N) waves = 64 x 16 = 1024 threads
constexpr int LANE_PER_WAVE = 64;
constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

__device__ inline rocwmma::bfloat16_t f32_to_bf16_rn(float x) {
    // round-to-nearest: add 0x8000 to lower 16 bits before truncation
    uint32_t u = __float_as_uint(x);
    u += 0x8000u;
    rocwmma::bfloat16_t y;
    // rocWMMA bfloat16_t is 16-bit storage type; alias-safe cast
    reinterpret_cast<uint16_t&>(y) = static_cast<uint16_t>(u >> 16);
    return y;
}

__global__ void gemm_wmma_bf16_kernel(
    float* __restrict__ C,                     // [M, N]
    const float* __restrict__ A,               // [M, K] (FP32)
    const __hip_bfloat16* __restrict__ W,      // [N, K] row-major (== [O, I])
    int M, int K, int N)
{
    using namespace rocwmma;

    // Block origin
    const int m0 = blockIdx.y * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    // Wave & lane ids
    const int lane = threadIdx.x;              // 0..63
    const int wave = threadIdx.y;              // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N;         // 0..(WAVES_M-1)
    const int wave_n = wave % WAVES_N;         // 0..(WAVES_N-1)

    // LDS layout:
    // sA: [BLOCK_M, BLOCK_K] row-major -> ldA = BLOCK_K
    // sB: [BLOCK_K, BLOCK_N] col-major -> ldB = BLOCK_K (so columns are contiguous for WMMA col_major)
    extern __shared__ uint8_t smemRaw[];
    auto* sA = reinterpret_cast<rocwmma::bfloat16_t*>(smemRaw);
    auto* sB = reinterpret_cast<rocwmma::bfloat16_t*>(sA + (BLOCK_M * BLOCK_K));

    // Optional per-wave scratch to mask ragged stores (only used at edges)
    auto* sC = reinterpret_cast<float*>(sB + (BLOCK_K * BLOCK_N));  // size: WAVES_PER_BLOCK * WM * WN

    // Accumulator fragment for this wave's 16x16 tile
    fragment<accumulator, WM, WN, WK, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    // Iterate over K in BLOCK_K(=16) steps
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {

        // --- Cooperative load of A tile [BLOCK_M x BLOCK_K], FP32->BF16 (row-major) ---
        // Use all threads in the block
        const int threadsPerBlock = blockDim.x * blockDim.y; // 64 * WAVES_PER_BLOCK
        int linearT = wave * blockDim.x + lane;              // 0..(threadsPerBlock-1)
        for (int idx = linearT; idx < BLOCK_M * BLOCK_K; idx += threadsPerBlock) {
            int r = idx / BLOCK_K;           // 0..BLOCK_M-1
            int c = idx % BLOCK_K;           // 0..BLOCK_K-1
            int gm = m0 + r;
            int gk = k0 + c;
            float a = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
            sA[r * BLOCK_K + c] = f32_to_bf16_rn(a);
        }

        // --- Cooperative load of B tile as col-major in LDS: [BLOCK_K x BLOCK_N] ---
        for (int idx = linearT; idx < BLOCK_K * BLOCK_N; idx += threadsPerBlock) {
            int c = idx / BLOCK_K;           // 0..BLOCK_N-1 (column index)
            int r = idx % BLOCK_K;           // 0..BLOCK_K-1 (row index == k)
            int gn = n0 + c;
            int gk = k0 + r;
            __hip_bfloat16 wb = (gk < K && gn < N) ? W[(size_t)gn * K + gk] : __float2bfloat16(0.0f);
            // store col-major: element(r,c) at c*ld + r, with ld = BLOCK_K
            reinterpret_cast<rocwmma::bfloat16_t&>(wb); // alias ok
            sB[c * BLOCK_K + r] = *reinterpret_cast<rocwmma::bfloat16_t*>(&wb);
        }

        __syncthreads();

        // --- Load A/B WMMA fragments for this wave's 16x16 subtile and MMA ---
        fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major> aFrag;
        fragment<matrix_b, WM, WN, WK, bfloat16_t, col_major> bFrag;

        // Wave's local offset inside the block tiles
        const int aRow = wave_m * WM;           // start row in sA
        const int bCol = wave_n * WN;           // start col in sB

        // Leading dimensions
        const int ldA = BLOCK_K;                // row-major A in LDS
        const int ldB = BLOCK_K;                // col-major B in LDS

        // Each K-slice is entire fragment depth WK
        load_matrix_sync(aFrag, sA + aRow * ldA, ldA);
        load_matrix_sync(bFrag, sB + bCol * ldB, ldB);

        mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
    }

    // --- Masked store: write to LDS scratch per-wave, then guarded global write ---
    // Per-wave scratch base
    float* sCbase = sC + wave * (WM * WN);
    // Store 16x16 to LDS (row-major, ld = WN)
    store_matrix_sync(sCbase, cFrag, WN, mem_row_major);
    __syncthreads();

    // Each lane writes several elements with bounds checks
    const int c_m0 = m0 + wave_m * WM;
    const int c_n0 = n0 + wave_n * WN;
    for (int t = lane; t < WM * WN; t += LANE_PER_WAVE) {
        int r = t / WN;
        int c = t % WN;
        int gm = c_m0 + r;
        int gn = c_n0 + c;
        if (gm < M && gn < N) {
            C[(size_t)gm * N + gn] = sCbase[r * WN + c];
        }
    }
}

// Host wrapper using WMMA path
void matmul(
    float* __restrict__ output,                 // [B, O]
    const float* __restrict__ input,            // [B, I]
    const __hip_bfloat16* __restrict__ weight,  // [O, I] row-major  == [N, K]
    int batch_size, int input_dim, int output_dim,
    hipStream_t stream = nullptr)
{
    const int M = batch_size, K = input_dim, N = output_dim;

    // Grid tiles the output into 64x64 blocks
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    // 64 lanes x 16 waves = 1024 threads per block
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    // LDS: sA(BLOCK_M*BLOCK_K) + sB(BLOCK_K*BLOCK_N) + sC(WAVES_PER_BLOCK*WM*WN)
    size_t shmem_bytes =
        (size_t)(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(rocwmma::bfloat16_t) +
        (size_t)(WAVES_PER_BLOCK * WM * WN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_wmma_bf16_kernel,
        grid, block, shmem_bytes, stream,
        output, input, weight, M, K, N);
}


// Memory allocation functions
void malloc_gpu_run_state(RunState *s, Config *p)
{
    int kv_dim = p->head_dim * p->n_kv_heads;
    int expert_per_token = p->experts_per_token;

    // Initialize pointers to NULL
    d_mask = NULL;

    // Check memory requirements and print for debugging
    size_t total_memory = 0;
    size_t batch_hidden = BATCH_SIZE * p->hidden_dim * sizeof(float);
    size_t batch_qkv = BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float);
    size_t kv_cache_size = BATCH_SIZE * p->n_layers * p->seq_len * kv_dim * sizeof(float);

    printf("Allocating GPU memory: batch_size=%d, hidden_dim=%d, seq_len=%d\n",
           BATCH_SIZE, p->hidden_dim, p->seq_len);
    printf("KV cache size per batch: %zu MB\n", kv_cache_size / (1024 * 1024));

    // Allocate GPU memory with error checking
    HIP_CHECK(hipMalloc((void **)&d_x, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&d_t, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&d_tb, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_tb2, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&d_qkv, batch_qkv));
    HIP_CHECK(hipMalloc((void **)&d_q, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_k, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_v, BATCH_SIZE * kv_dim * sizeof(float)));

    // NEW: Buffers for GPU scatter-gather MoE
    // Max possible items for one expert is the entire batch
    HIP_CHECK(hipMalloc((void **)&d_expert_indices, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_expert_weights, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_batch_count, sizeof(int)));
    // This buffer holds the expert's final output before scattering
    HIP_CHECK(hipMalloc((void **)&d_expert_output_buffer, BATCH_SIZE * p->hidden_dim * sizeof(float) * expert_per_token));

    // KV cache allocation - this is usually the largest allocation
    HIP_CHECK(hipMalloc((void **)&d_key_cache, kv_cache_size));
    HIP_CHECK(hipMalloc((void **)&d_value_cache, kv_cache_size));

    HIP_CHECK(hipMalloc((void **)&d_att, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_logits, BATCH_SIZE * p->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_router_score, BATCH_SIZE * p->n_experts * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_topk_v, BATCH_SIZE * p->experts_per_token * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_topk_i, BATCH_SIZE * p->experts_per_token * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_mlp1_out, BATCH_SIZE * 2 * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&d_gate, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&d_up, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&d_gate_up, BATCH_SIZE * p->intermediate_dim * sizeof(float) * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&d_e_agg, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&d_current_tokens, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_positions, BATCH_SIZE * sizeof(int)));
    HIP_CHECK(hipMalloc((void **)&d_cos_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_sin_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_expert_input_buffer, batch_hidden * expert_per_token));
    HIP_CHECK(hipMalloc((void **)&d_temp_buffer, batch_hidden));
    HIP_CHECK(hipMalloc((void **)&d_token_embedding_table, p->vocab_size * p->hidden_dim * sizeof(__hip_bfloat16)));

    // CRITICAL FIX: Allocate GPU memory for all weights in bfloat16 format
    HIP_CHECK(hipMalloc((void **)&d_rms_attn_w, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_rms_ffn_w, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_rms_out_w, p->hidden_dim * sizeof(__hip_bfloat16)));

    int qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    HIP_CHECK(hipMalloc((void **)&d_w_qkv, qkv_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_b_qkv, p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim * sizeof(__hip_bfloat16)));

    int attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&d_w_o, attn_out_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_b_o, p->n_layers * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&d_w_router, p->n_layers * p->hidden_dim * p->n_experts * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_b_router, p->n_layers * p->n_experts * sizeof(__hip_bfloat16)));

    int mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    HIP_CHECK(hipMalloc((void **)&d_w_mlp1, mlp1_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_b_mlp1, p->n_layers * p->n_experts * (2 * p->intermediate_dim) * sizeof(__hip_bfloat16)));

    int mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    HIP_CHECK(hipMalloc((void **)&d_w_mlp2, mlp2_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_b_mlp2, p->n_layers * p->n_experts * p->hidden_dim * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMalloc((void **)&d_out_w, p->hidden_dim * p->vocab_size * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc((void **)&d_attn_sinks, p->n_layers * p->n_attn_heads * sizeof(__hip_bfloat16)));

    // Initialize all allocated memory to zero
    HIP_CHECK(hipMemset(d_x, 0, batch_hidden));
    HIP_CHECK(hipMemset(d_t, 0, batch_hidden));
    HIP_CHECK(hipMemset(d_tb, 0, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float)));
    HIP_CHECK(hipMemset(d_tb2, 0, batch_hidden));
    HIP_CHECK(hipMemset(d_qkv, 0, batch_qkv));
    HIP_CHECK(hipMemset(d_q, 0, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_k, 0, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_v, 0, BATCH_SIZE * kv_dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_key_cache, 0, kv_cache_size));
    HIP_CHECK(hipMemset(d_value_cache, 0, kv_cache_size));
    HIP_CHECK(hipMemset(d_att, 0, BATCH_SIZE * p->n_attn_heads * p->seq_len * sizeof(float)));
    HIP_CHECK(hipMemset(d_logits, 0, BATCH_SIZE * p->vocab_size * sizeof(float)));

    if (p->sliding_window > 0)
    {
        size_t mask_size = p->seq_len * p->seq_len * sizeof(float);
        HIP_CHECK(hipMalloc((void **)&d_mask, mask_size));

        // Initialize mask on GPU if needed
        float *h_mask = (float *)malloc(mask_size);
        if (!h_mask)
        {
            fprintf(stderr, "Failed to allocate host memory for mask\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < p->seq_len; i++)
        {
            for (int j = 0; j < p->seq_len; j++)
            {
                h_mask[i * p->seq_len + j] = (i - j >= p->sliding_window) ? -INFINITY : 0.0f;
            }
        }
        HIP_CHECK(hipMemcpy(d_mask, h_mask, mask_size, hipMemcpyHostToDevice));
        free(h_mask);
    }
}

void copy_weights_to_gpu(Transformer *transformer)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    // Convert and copy normalization weights
    size_t rms_attn_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_rms_attn_bf16 = (__hip_bfloat16 *)malloc(rms_attn_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_attn_w, h_rms_attn_bf16, rms_attn_size);
    HIP_CHECK(hipMemcpy(d_rms_attn_w, h_rms_attn_bf16, rms_attn_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_attn_bf16);

    size_t rms_ffn_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_rms_ffn_bf16 = (__hip_bfloat16 *)malloc(rms_ffn_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_ffn_w, h_rms_ffn_bf16, rms_ffn_size);
    HIP_CHECK(hipMemcpy(d_rms_ffn_w, h_rms_ffn_bf16, rms_ffn_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_ffn_bf16);

    size_t rms_out_size = p->hidden_dim;
    __hip_bfloat16 *h_rms_out_bf16 = (__hip_bfloat16 *)malloc(rms_out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->rms_out_w, h_rms_out_bf16, rms_out_size);
    HIP_CHECK(hipMemcpy(d_rms_out_w, h_rms_out_bf16, rms_out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_rms_out_bf16);

    // Convert and copy attention weights
    size_t qkv_size = p->n_layers * p->hidden_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    __hip_bfloat16 *h_w_qkv_bf16 = (__hip_bfloat16 *)malloc(qkv_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_qkv, h_w_qkv_bf16, qkv_size);
    HIP_CHECK(hipMemcpy(d_w_qkv, h_w_qkv_bf16, qkv_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_qkv_bf16);

    size_t b_qkv_size = p->n_layers * (p->n_attn_heads + 2 * p->n_kv_heads) * p->head_dim;
    __hip_bfloat16 *h_b_qkv_bf16 = (__hip_bfloat16 *)malloc(b_qkv_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_qkv, h_b_qkv_bf16, b_qkv_size);
    HIP_CHECK(hipMemcpy(d_b_qkv, h_b_qkv_bf16, b_qkv_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_qkv_bf16);

    size_t attn_out_size = p->n_layers * (p->n_attn_heads * p->head_dim) * p->hidden_dim;
    __hip_bfloat16 *h_w_o_bf16 = (__hip_bfloat16 *)malloc(attn_out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_o, h_w_o_bf16, attn_out_size);
    HIP_CHECK(hipMemcpy(d_w_o, h_w_o_bf16, attn_out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_o_bf16);

    size_t b_o_size = p->n_layers * p->hidden_dim;
    __hip_bfloat16 *h_b_o_bf16 = (__hip_bfloat16 *)malloc(b_o_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_o, h_b_o_bf16, b_o_size);
    HIP_CHECK(hipMemcpy(d_b_o, h_b_o_bf16, b_o_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_o_bf16);

    // Convert and copy attention sinks
    size_t attn_sinks_size = p->n_layers * p->n_attn_heads;
    __hip_bfloat16 *h_attn_sinks_bf16 = (__hip_bfloat16 *)malloc(attn_sinks_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->attn_sinks, h_attn_sinks_bf16, attn_sinks_size);
    HIP_CHECK(hipMemcpy(d_attn_sinks, h_attn_sinks_bf16, attn_sinks_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_attn_sinks_bf16);

    // Convert and copy MoE weights
    size_t w_router_size = p->n_layers * p->hidden_dim * p->n_experts;
    __hip_bfloat16 *h_w_router_bf16 = (__hip_bfloat16 *)malloc(w_router_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_router, h_w_router_bf16, w_router_size);
    HIP_CHECK(hipMemcpy(d_w_router, h_w_router_bf16, w_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_router_bf16);

    size_t b_router_size = p->n_layers * p->n_experts;
    __hip_bfloat16 *h_b_router_bf16 = (__hip_bfloat16 *)malloc(b_router_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_router, h_b_router_bf16, b_router_size);
    HIP_CHECK(hipMemcpy(d_b_router, h_b_router_bf16, b_router_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_router_bf16);

    size_t mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim) * p->hidden_dim;
    __hip_bfloat16 *h_w_mlp1_bf16 = (__hip_bfloat16 *)malloc(mlp1_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_mlp1, h_w_mlp1_bf16, mlp1_size);
    HIP_CHECK(hipMemcpy(d_w_mlp1, h_w_mlp1_bf16, mlp1_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_mlp1_bf16);

    size_t b_mlp1_size = p->n_layers * p->n_experts * (2 * p->intermediate_dim);
    __hip_bfloat16 *h_b_mlp1_bf16 = (__hip_bfloat16 *)malloc(b_mlp1_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_mlp1, h_b_mlp1_bf16, b_mlp1_size);
    HIP_CHECK(hipMemcpy(d_b_mlp1, h_b_mlp1_bf16, b_mlp1_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_mlp1_bf16);

    size_t mlp2_size = p->n_layers * p->n_experts * p->hidden_dim * p->intermediate_dim;
    __hip_bfloat16 *h_w_mlp2_bf16 = (__hip_bfloat16 *)malloc(mlp2_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->w_mlp2, h_w_mlp2_bf16, mlp2_size);
    HIP_CHECK(hipMemcpy(d_w_mlp2, h_w_mlp2_bf16, mlp2_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_w_mlp2_bf16);

    size_t b_mlp2_size = p->n_layers * p->n_experts * p->hidden_dim;
    __hip_bfloat16 *h_b_mlp2_bf16 = (__hip_bfloat16 *)malloc(b_mlp2_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->b_mlp2, h_b_mlp2_bf16, b_mlp2_size);
    HIP_CHECK(hipMemcpy(d_b_mlp2, h_b_mlp2_bf16, b_mlp2_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_b_mlp2_bf16);

    // Convert and copy output weights
    size_t out_size = p->hidden_dim * p->vocab_size;
    __hip_bfloat16 *h_out_bf16 = (__hip_bfloat16 *)malloc(out_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(w->out, h_out_bf16, out_size);
    HIP_CHECK(hipMemcpy(d_out_w, h_out_bf16, out_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_out_bf16);

    printf("Weight conversion and copying completed successfully\n");
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer)
{
    Config *p = &transformer->config;

    // Allocate GPU memory first
    malloc_gpu_run_state(&transformer->state, p);

    // CRITICAL FIX: Copy all weights to GPU
    copy_weights_to_gpu(transformer);

    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;

    // CPU allocation for RoPE values (used in warmup only)
    cos_vals_cpu = (float *)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));
    sin_vals_cpu = (float *)malloc((p->head_dim / 2) * p->seq_len * sizeof(float));

    for (int pos = 0; pos < p->seq_len; ++pos)
    {
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                             p->initial_context_length, ntk_beta, ntk_alpha,
                             cos_vals_cpu + (pos * p->head_dim / 2),
                             sin_vals_cpu + (pos * p->head_dim / 2));
    }

    // Copy RoPE values to GPU (now that GPU memory is allocated)
    HIP_CHECK(hipMemcpy(d_cos_vals, cos_vals_cpu, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sin_vals, sin_vals_cpu, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));

    // Convert and copy token embedding table to bfloat16
    size_t embedding_size = p->vocab_size * p->hidden_dim;
    __hip_bfloat16 *h_embedding_bf16 = (__hip_bfloat16 *)malloc(embedding_size * sizeof(__hip_bfloat16));
    convert_float_array_to_bfloat16(transformer->weights.token_embedding_table, h_embedding_bf16, embedding_size);
    HIP_CHECK(hipMemcpy(d_token_embedding_table, h_embedding_bf16, embedding_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    free(h_embedding_bf16);

    // CPU allocations for batch management
    prompt_tokens = (int **)malloc(BATCH_SIZE * sizeof(int *));
    current_tokens = (int *)malloc(BATCH_SIZE * sizeof(int));
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        prompt_tokens[b] = (int *)malloc((p->seq_len + 3) * sizeof(int));
    }
    finished = (bool *)malloc(BATCH_SIZE * sizeof(bool));
    positions = (int *)malloc(BATCH_SIZE * sizeof(int));
    prompt_lens = (int *)malloc(BATCH_SIZE * sizeof(int));
}

void finish(Transformer *transformer, Tokenizer *tokenizer)
{
    // Free GPU memory
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_t));
    HIP_CHECK(hipFree(d_tb));
    HIP_CHECK(hipFree(d_tb2));
    HIP_CHECK(hipFree(d_qkv));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_key_cache));
    HIP_CHECK(hipFree(d_value_cache));
    HIP_CHECK(hipFree(d_att));
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_router_score));
    HIP_CHECK(hipFree(d_topk_v));
    HIP_CHECK(hipFree(d_topk_i));
    HIP_CHECK(hipFree(d_mlp1_out));
    HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_up));
    HIP_CHECK(hipFree(d_gate_up));
    HIP_CHECK(hipFree(d_e_agg));
    HIP_CHECK(hipFree(d_current_tokens));
    HIP_CHECK(hipFree(d_positions));
    HIP_CHECK(hipFree(d_cos_vals));
    HIP_CHECK(hipFree(d_sin_vals));
    HIP_CHECK(hipFree(d_expert_input_buffer));
    HIP_CHECK(hipFree(d_temp_buffer));
    HIP_CHECK(hipFree(d_token_embedding_table));
    if (d_mask)
        HIP_CHECK(hipFree(d_mask));

    // Free GPU weight memory
    HIP_CHECK(hipFree(d_rms_attn_w));
    HIP_CHECK(hipFree(d_rms_ffn_w));
    HIP_CHECK(hipFree(d_rms_out_w));
    HIP_CHECK(hipFree(d_w_qkv));
    HIP_CHECK(hipFree(d_b_qkv));
    HIP_CHECK(hipFree(d_w_o));
    HIP_CHECK(hipFree(d_b_o));
    HIP_CHECK(hipFree(d_w_router));
    HIP_CHECK(hipFree(d_b_router));
    HIP_CHECK(hipFree(d_w_mlp1));
    HIP_CHECK(hipFree(d_b_mlp1));
    HIP_CHECK(hipFree(d_w_mlp2));
    HIP_CHECK(hipFree(d_b_mlp2));
    HIP_CHECK(hipFree(d_out_w));
    HIP_CHECK(hipFree(d_attn_sinks));

    // Free CPU memory
    free(cos_vals_cpu);
    free(sin_vals_cpu);
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        free(prompt_tokens[b]);
    }
    free(prompt_tokens);
    free(current_tokens);
    free(finished);
    free(positions);
    free(prompt_lens);
}

// "Gather" kernel: Finds tokens for an expert and creates a compact list.
__global__ void gather_expert_inputs_kernel(const float *d_t, const int *topk_i, const float *topk_v,
                                            int expert_id, int batch_size, int hidden_dim, int experts_per_token,
                                            float *expert_input_buffer, int *expert_indices, float *expert_weights,
                                            int *d_batch_count)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size)
        return;

    // Check if this token 'b' selected the current 'expert_id'
    for (int k = 0; k < experts_per_token; ++k)
    {
        int topk_idx = b * experts_per_token + k;
        if (topk_i[topk_idx] == expert_id)
        {
            // This token is routed to this expert.
            // Atomically get a unique index for this token in the compact buffer.
            int compact_idx = atomicAdd(d_batch_count, 1);

            // Store the original batch index and weight for the scatter step.
            expert_indices[compact_idx] = b;
            expert_weights[compact_idx] = topk_v[topk_idx];

            // Copy the hidden state from d_t into the compact input buffer.
            // This is a strided copy, which is slow. For max performance,
            // a second kernel could re-format this into a dense matrix.
            // For logic matching, this is correct.
            const float *src = d_t + b * hidden_dim;
            float *dst = expert_input_buffer + compact_idx * hidden_dim;
            for (int i = 0; i < hidden_dim; ++i)
            {
                dst[i] = src[i];
            }
            // break; // Token found its expert, move to next token
        }
    }
}

// "Scatter" kernel: Adds the expert outputs back to the final aggregation buffer.
__global__ void scatter_expert_outputs_kernel(float *d_e_agg, const float *expert_output_buffer,
                                              const int *expert_indices, const float *expert_weights,
                                              int batch_count, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count * hidden_dim)
        return;

    int compact_idx = idx / hidden_dim;
    int dim = idx % hidden_dim;

    // Get the original batch index and the expert's router weight
    int original_batch_idx = expert_indices[compact_idx];
    float weight = expert_weights[compact_idx];

    // Calculate the destination address in the main aggregation buffer
    float *dst = d_e_agg + original_batch_idx * hidden_dim + dim;
    float value = expert_output_buffer[idx];

    // Atomically add the weighted result. This is crucial because multiple
    // experts (if experts_per_token > 1) write to the same d_e_agg location.
    atomicAdd(dst, value * weight);
}

__global__ void add_sinks_kernel(float *att, const __hip_bfloat16 *sinks, const int *positions,
                                 int seq_len, int n_heads)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int pos = positions[batch_idx];
    if (pos + 1 < seq_len)
    {
        // Index for the sink is at pos + 1
        int sink_att_idx = batch_idx * n_heads * seq_len + head_idx * seq_len + (pos + 1);
        // Convert bfloat16 sink to fp32 on-the-fly
        float sink_fp32 = __bfloat162float(sinks[head_idx]);
        att[sink_att_idx] = sink_fp32;
    }
}

__global__ void softmax_kernel_variable_len(float *x, const int *positions,
                                            int batch_size, int n_heads, int max_seq_len)
{
    // Each block processes one head for one batch item
    int batch_idx = blockIdx.x / n_heads;
    int head_idx = blockIdx.x % n_heads;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    int pos = positions[batch_idx];
    int size = pos + 2; // Real size including the attention sink

    float *batch_head_x = x + (batch_idx * n_heads + head_idx) * max_seq_len;

    __shared__ float shared_max[THREADS_PER_BLOCK];
    __shared__ float shared_sum[THREADS_PER_BLOCK];

    // Find max value in the valid range
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x)
    {
        max_val = fmaxf(max_val, batch_head_x[i]);
    }
    shared_max[tid] = max_val;
    __syncthreads();

    // Reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    __syncthreads();

    // Compute exp and sum over the valid range
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_head_x[i] = expf(batch_head_x[i] - max_val);
        sum += batch_head_x[i];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    sum = shared_sum[0];
    __syncthreads();

    // Normalize over the valid range
    for (int i = tid; i < size; i += blockDim.x)
    {
        batch_head_x[i] /= sum;
    }
}

// GPU-accelerated neural network functions
void attention_gpu(Transformer *transformer, int layer_idx, int batch_size)
{
    Config *p = &transformer->config;
    static Timer rms_norm_timer("RMSNorm_attention", true);
    static Timer matmul_timer("MatMul_attention", true);
    static Timer add_bias_timer("AddBias_attention", true);
    static Timer apply_rope_timer("ApplyRoPE_attention", true);
    static Timer update_kv_cache_timer("UpdateKVCache_attention", true);
    static Timer attention_scores_kernel_timer("AttentionScoresKernel_attention", true);
    static Timer add_sinks_kernel_timer("AddSinksKernel_attention", true);
    static Timer softmax_kernel_timer("SoftmaxKernel_attention", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_attention", true);
    static Timer attention_weighted_sum_kernel_timer("AttentionWeightedSumKernel_attention", true);
    static Timer accumulate_kernel_timer("AccumulateKernel_attention", true);

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;

    // RMSNorm - FIXED: Use GPU weight pointer
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<norm_grid, norm_block>>>(
            d_t, d_x, d_rms_attn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    dim3 matmul_grid(batch_size, ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + 31) / 32);
    dim3 matmul_block(32, min(32, THREADS_PER_BLOCK / 32));
    int qkv_weight_offset = layer_idx * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    // Your existing variables: d_qkv, d_t, d_w_qkv, etc.
    // ...

    // Define block and grid dimensions
    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = ((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + block_dim.y - 1) / block_dim.y; // Ceiling division
    {
        TIME_SCOPE(matmul_timer);
        // QKV projection using safer matmul kernel - FIXED: Use GPU weight pointer
        // Launch the simple kernel
        /*
        matmul_kernel_simple<<<grid_dim, block_dim>>>(
            d_qkv,
            d_t,
            d_w_qkv + qkv_weight_offset,
            batch_size,
            hidden_dim,
            (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        */

        matmul(
            d_qkv,
            d_t,
            d_w_qkv + qkv_weight_offset,
            batch_size,
            hidden_dim,
            (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipGetLastError());
    // Add bias - FIXED: Use GPU bias pointer and proper kernel
    int qkv_bias_offset = layer_idx * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    dim3 bias_grid((batch_size * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    {
        TIME_SCOPE(add_bias_timer);
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            d_qkv, d_b_qkv + qkv_bias_offset, batch_size, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Copy Q, K, V from qkv buffer - SIMPLIFIED AND FIXED
    int q_size = p->n_attn_heads * head_dim;
    int k_size = p->n_kv_heads * head_dim;
    int v_size = p->n_kv_heads * head_dim;

    // Copy Q: shape [batch_size, n_attn_heads * head_dim]
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = d_qkv + b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *dst = d_q + b * q_size;
        HIP_CHECK(hipMemcpy(dst, src, q_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy K: shape [batch_size, n_kv_heads * head_dim]
    int k_offset = p->n_attn_heads * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = d_qkv + b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + k_offset;
        float *dst = d_k + b * k_size;
        HIP_CHECK(hipMemcpy(dst, src, k_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Copy V: shape [batch_size, n_kv_heads * head_dim]
    int v_offset = (p->n_attn_heads + p->n_kv_heads) * head_dim;
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float *src = d_qkv + b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim + v_offset;
        float *dst = d_v + b * v_size;
        HIP_CHECK(hipMemcpy(dst, src, v_size * sizeof(float), hipMemcpyDeviceToDevice));
    }

    // Apply rotary embeddings
    dim3 rope_grid(batch_size, p->n_attn_heads);
    dim3 rope_block(head_dim / 2);
    {
        TIME_SCOPE(apply_rope_timer);
        apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
            d_q, d_cos_vals, d_sin_vals, d_positions, batch_size, p->n_attn_heads, head_dim);
        HIP_CHECK(hipGetLastError());
    }

    rope_grid.y = p->n_kv_heads;
    apply_rotary_emb_kernel<<<rope_grid, rope_block>>>(
        d_k, d_cos_vals, d_sin_vals, d_positions, batch_size, p->n_kv_heads, head_dim);
    HIP_CHECK(hipGetLastError());

    // Update KV cache - NEW: Proper GPU kernel
    dim3 kv_grid(batch_size, (kv_dim + 31) / 32);
    dim3 kv_block(1, 32);
    {
        TIME_SCOPE(update_kv_cache_timer);
        update_kv_cache_kernel<<<kv_grid, kv_block>>>(
            d_key_cache, d_value_cache, d_k, d_v, d_positions, batch_size,
            p->n_layers, layer_idx, p->seq_len, kv_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Compute attention scores
    dim3 att_grid(batch_size, p->n_attn_heads, (p->seq_len + 31) / 32);
    dim3 att_block(1, 1, 32);
    {
        TIME_SCOPE(attention_scores_kernel_timer);
        attention_scores_kernel<<<att_grid, att_block>>>(
            d_att, d_q, d_key_cache, d_mask, d_positions, batch_size, p->n_attn_heads,
            head_dim, p->seq_len, p->n_layers, layer_idx, p->sliding_window > 0);
        HIP_CHECK(hipGetLastError());
    }

    dim3 sink_grid(batch_size, p->n_attn_heads);
    dim3 sink_block(1);
    {
        TIME_SCOPE(add_sinks_kernel_timer);
        add_sinks_kernel<<<sink_grid, sink_block>>>(
            d_att, d_attn_sinks + layer_idx * p->n_attn_heads, d_positions,
            p->seq_len, p->n_attn_heads);
        HIP_CHECK(hipGetLastError());
    }

    // Softmax attention weights
    dim3 soft_grid(batch_size * p->n_attn_heads);
    dim3 soft_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(softmax_kernel_timer);
        softmax_kernel_variable_len<<<soft_grid, soft_block>>>(
            d_att, d_positions, batch_size, p->n_attn_heads, p->seq_len);
        HIP_CHECK(hipGetLastError());
    }

    // Weighted sum of values
    dim3 wsum_grid(batch_size, p->n_attn_heads);
    dim3 wsum_block(head_dim);
    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        attention_weighted_sum_kernel<<<wsum_grid, wsum_block>>>(
            d_tb, d_att, d_value_cache, d_positions, batch_size, p->n_attn_heads,
            head_dim, p->seq_len, p->n_layers, layer_idx);
        HIP_CHECK(hipGetLastError());
    }
    // Output projection - FIXED: Use GPU weight pointer

    grid_dim.y = (hidden_dim + block_dim.y - 1) / block_dim.y; // Ceiling division

    int attn_out_offset = layer_idx * (head_dim * p->n_attn_heads) * hidden_dim;

    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        // Launch the simple kernel
        matmul(
            d_tb2, d_tb, d_w_o + attn_out_offset, batch_size, head_dim * p->n_attn_heads, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Add bias and residual connection - FIXED: Use GPU bias pointer
    int attn_bias_offset = layer_idx * hidden_dim;
    {
        TIME_SCOPE(accumulate_kernel_timer);
        add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
            d_tb2, d_b_o + attn_bias_offset, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    {
        TIME_SCOPE(accumulate_kernel_timer);
        accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_x, d_tb2, 1.0f, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
}

// NEW: Improved MoE implementation with better expert routing
__global__ void expert_routing_kernel(float *expert_weights, const int *topk_indices,
                                      const float *topk_values, int batch_size,
                                      int experts_per_token, int n_experts)
{
    int batch_idx = blockIdx.x;
    int expert_slot = blockIdx.y;

    if (batch_idx >= batch_size || expert_slot >= experts_per_token)
        return;

    int expert_id = topk_indices[batch_idx * experts_per_token + expert_slot];
    float weight = topk_values[batch_idx * experts_per_token + expert_slot];

    expert_weights[batch_idx * n_experts + expert_id] = weight;
}

// NEW KERNEL for correct MoE aggregation
__global__ void aggregate_expert_output_kernel(float *d_e_agg, const float *expert_output,
                                               const int *topk_indices, const float *topk_values,
                                               int current_expert_id, int batch_size, int hidden_dim,
                                               int experts_per_token)
{
    // Each thread handles one dimension of one token in the batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_dim)
        return;

    int b = idx / hidden_dim; // Get the batch index for this thread

    // Check if the current expert (current_expert_id) was selected for this token (b)
    for (int k = 0; k < experts_per_token; ++k)
    {
        int expert_slot_idx = b * experts_per_token + k;
        if (topk_indices[expert_slot_idx] == current_expert_id)
        {
            // This token uses this expert. Add the weighted output to the aggregation buffer.
            float weight = topk_values[expert_slot_idx];
            d_e_agg[idx] += weight * expert_output[idx];

            // Since top-k indices are unique for a token, we can stop after finding the match
            break;
        }
    }
}

/**
 * @brief Stage 1: Counts the number of tokens assigned to each expert in parallel.
 *
 * This kernel launches one thread per token. Each thread iterates through its
 * top-k expert choices and atomically increments the counter for each chosen expert.
 * Contention is low as it's distributed across all expert counters.
 *
 * @param topk_i Device pointer to the top-k expert indices for each token.
 * @param d_expert_counts Device pointer to an array of size n_experts (pre-filled with zeros).
 * @param batch_size Total number of tokens.
 * @param experts_per_token The 'k' in top-k.
 */
__global__ void count_tokens_per_expert_kernel(const int *topk_i, int *d_expert_counts,
                                               int batch_size, int experts_per_token)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size)
    {
        return;
    }

    // Each token contributes to the count of its assigned experts
    for (int k = 0; k < experts_per_token; ++k)
    {
        int expert_id = topk_i[token_idx * experts_per_token + k];
        // This atomic is low-contention because updates are spread across n_experts counters
        atomicAdd(&d_expert_counts[expert_id], 1);
    }
}

/**
 * @brief Stage 2: Gathers/permutes expert inputs into a compact buffer using a
 * block-per-token strategy for coalesced memory access.
 *
 * This kernel is the high-performance replacement for the original gather_expert_inputs_kernel.
 * It uses a full thread block to process each token, allowing the `hidden_dim` vector
 * to be copied in a fully parallel and coalesced manner.
 *
 * @param d_t The source hidden states (batch_size, hidden_dim).
 * @param topk_i The top-k expert indices for each token.
 * @param topk_v The top-k expert weights for each token.
 * @param d_expert_offsets The starting index for each expert in the compact buffer.
 * @param d_expert_write_idx A temporary counter for each expert to get a local index.
 * @param batch_size Total number of tokens.
 * @param hidden_dim Dimension of the hidden state.
 * @param experts_per_token The 'k' in top-k.
 * @param expert_input_buffer Destination compact buffer for hidden states.
 * @param expert_indices Destination buffer for original token indices for scattering.
 * @param expert_weights Destination buffer for router weights for scattering.
 */
__global__ void permute_expert_inputs_kernel(const float *d_t, const int *topk_i, const float *topk_v,
                                             const int *d_expert_offsets, int *d_expert_write_idx,
                                             int batch_size, int hidden_dim, int experts_per_token,
                                             float *expert_input_buffer, int *expert_indices, float *expert_weights)
{
    // Each BLOCK processes one token to enable parallel copying
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size)
    {
        return;
    }

    // Use shared memory to communicate the calculated destination index to all threads in the block.
    // Size must match experts_per_token.
    __shared__ int destination_indices[2]; // Assumes experts_per_token <= 2

    // The first few threads handle the logic for each of the token's expert choices
    if (threadIdx.x < experts_per_token)
    {
        int k = threadIdx.x;
        int topk_flat_idx = token_idx * experts_per_token + k;
        int expert_id = topk_i[topk_flat_idx];

        // Atomically get the local write position within this expert's designated data block
        int local_idx = atomicAdd(&d_expert_write_idx[expert_id], 1);

        // Calculate the final destination index in the large compact buffer
        int compact_idx = d_expert_offsets[expert_id] + local_idx;

        // Store metadata needed for the later scatter step
        expert_indices[compact_idx] = token_idx;
        expert_weights[compact_idx] = topk_v[topk_flat_idx];

        // Share the destination index with all threads in this block
        destination_indices[k] = compact_idx;
    }

    // Synchronize to ensure destination_indices is visible to all threads in the block
    __syncthreads();

    // Now, all threads in the block cooperate to copy the hidden state for each expert choice.
    // This loop ensures we handle all `experts_per_token` assignments for the current token.
    for (int k = 0; k < experts_per_token; ++k)
    {
        const float *src = d_t + token_idx * hidden_dim;
        float *dst = expert_input_buffer + destination_indices[k] * hidden_dim;

        // This is the coalesced copy: each thread copies a different element of the hidden_dim vector
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }
}

void moe_gpu(Transformer *transformer, int layer_idx, int batch_size)
{
    // Timer declarations (assuming they are defined elsewhere)
    static Timer rms_norm_timer("RMSNorm_moe", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_moe", true);
    static Timer add_bias_timer("AddBias_moe", true);
    static Timer topk_kernel_timer("TopKKernel_moe", true);
    static Timer softmax_kernel_timer("SoftmaxKernel_moe", true);
    static Timer gather_expert_inputs_kernel_timer("GatherExpertInputsKernel_moe", true);
    static Timer expert_agg_kernel_timer("ExpertAggKernel_moe", true);
    static Timer scatter_expert_outputs_kernel_timer("ScatterExpertOutputsKernel_moe", true);
    static Timer accumulate_kernel_timer("AccumulateKernel_moe", true);
    static Timer split_gate_up_kernel_timer("SplitGateUpKernel_moe", true);
    static Timer swiglu_kernel_timer("SwigluKernel_moe", true);
    static Timer matmul_kernel_timer("MatMulKernel_moe", true);

    Config *p = &transformer->config;
    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    int experts_per_token = p->experts_per_token;

    // FFN RMSNorm
    dim3 norm_grid(batch_size);
    dim3 norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<norm_grid, norm_block>>>(d_t, d_x, d_rms_ffn_w + layer_idx * hidden_dim, batch_size, hidden_dim);
    }
    HIP_CHECK(hipGetLastError());

    // Router computation & bias
    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        matmul(d_router_score, d_t, d_w_router + layer_idx * hidden_dim * n_experts, batch_size, hidden_dim, n_experts);
    }
    {
        TIME_SCOPE(add_bias_timer);
        add_bias_kernel<<<(batch_size * n_experts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_router_score, d_b_router + layer_idx * n_experts, batch_size, n_experts);
    }

    // Top-k expert selection
    {
        TIME_SCOPE(topk_kernel_timer);
        topk_kernel<<<batch_size, 1>>>(d_topk_v, d_topk_i, d_router_score, batch_size, n_experts, experts_per_token);
    }

    // Softmax on top-k values to get weights
    {
        TIME_SCOPE(softmax_kernel_timer);
        softmax_kernel<<<batch_size, norm_block>>>(d_topk_v, batch_size, experts_per_token);
    }
    HIP_CHECK(hipGetLastError());

    // Initialize final aggregation buffer
    HIP_CHECK(hipMemset(d_e_agg, 0, batch_size * hidden_dim * sizeof(float)));

    // --- 🚀 OPTIMIZED TWO-STAGE GATHER ALGORITHM 🚀 ---
    int *d_expert_counts;
    int *d_expert_offsets;
    int *d_expert_write_idx;
    HIP_CHECK(hipMalloc(&d_expert_counts, n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_offsets, n_experts * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_expert_write_idx, n_experts * sizeof(int)));

    // **FIX:** Declare host-side arrays and total_tokens here, before the timed scope
    int h_expert_counts[n_experts];
    int h_expert_offsets[n_experts];
    int total_tokens = 0;

    {
        TIME_SCOPE(gather_expert_inputs_kernel_timer); // Timing the entire gather operation

        // === STAGE 1: COUNT TOKENS PER EXPERT ===
        HIP_CHECK(hipMemset(d_expert_counts, 0, n_experts * sizeof(int)));
        dim3 count_grid((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        count_tokens_per_expert_kernel<<<count_grid, THREADS_PER_BLOCK>>>(
            d_topk_i, d_expert_counts, batch_size, experts_per_token);
        HIP_CHECK(hipGetLastError());

        // === CALCULATE OFFSETS (PREFIX SUM ON CPU) ===
        HIP_CHECK(hipMemcpy(h_expert_counts, d_expert_counts, n_experts * sizeof(int), hipMemcpyDeviceToHost));

        for (int i = 0; i < n_experts; ++i)
        {
            h_expert_offsets[i] = total_tokens;
            total_tokens += h_expert_counts[i];
        }
        HIP_CHECK(hipMemcpy(d_expert_offsets, h_expert_offsets, n_experts * sizeof(int), hipMemcpyHostToDevice));

        // === STAGE 2: PERMUTE INPUTS WITH COALESCED COPY ===
        HIP_CHECK(hipMemset(d_expert_write_idx, 0, n_experts * sizeof(int)));
        dim3 permute_grid(batch_size); // One block per token
        dim3 permute_block(256);       // Block size for efficient copying
        permute_expert_inputs_kernel<<<permute_grid, permute_block>>>(
            d_t, d_topk_i, d_topk_v, d_expert_offsets, d_expert_write_idx,
            batch_size, hidden_dim, experts_per_token,
            d_expert_input_buffer, d_expert_indices, d_expert_weights);
        HIP_CHECK(hipGetLastError());
    } // End of gather timer scope

    // --- 2. COMPUTE STEP (EXPERT MLPS) ---
    // Loop through each expert to run its MLP on its dedicated slice of the compact buffer.
    for (int expert_id = 0; expert_id < n_experts; expert_id++)
    {
        int h_batch_count = h_expert_counts[expert_id];
        if (h_batch_count == 0)
        {
            continue; // Skip experts with no tokens
        }

        // Calculate pointers to this expert's slice of the data
        int expert_offset = h_expert_offsets[expert_id];
        float *expert_input_ptr = d_expert_input_buffer + expert_offset * hidden_dim;
        float *mlp1_out_ptr = d_mlp1_out + expert_offset * 2 * intermediate_dim;
        float *gate_ptr = d_gate + expert_offset * intermediate_dim;
        float *up_ptr = d_up + expert_offset * intermediate_dim;
        float *gate_up_ptr = d_gate_up + expert_offset * intermediate_dim;
        float *expert_output_ptr = d_expert_output_buffer + expert_offset * hidden_dim;
        // MLP1 (Gate/Up projections)
        {
            TIME_SCOPE(matmul_kernel_timer);
            matmul(mlp1_out_ptr, expert_input_ptr,
                   d_w_mlp1 + (layer_idx * n_experts + expert_id) * (2 * intermediate_dim) * hidden_dim,
                   h_batch_count, hidden_dim, 2 * intermediate_dim);
        }

        // Split, add bias, and apply SwiGLU
        dim3 expert_grid((h_batch_count * intermediate_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        {
            TIME_SCOPE(split_gate_up_kernel_timer);
            split_gate_up_kernel<<<expert_grid, THREADS_PER_BLOCK>>>(gate_ptr, up_ptr, mlp1_out_ptr,
                                                                     d_b_mlp1 + (layer_idx * n_experts + expert_id) * (2 * intermediate_dim), h_batch_count, intermediate_dim);
        }
        {
            TIME_SCOPE(swiglu_kernel_timer);
            swiglu_kernel<<<expert_grid, THREADS_PER_BLOCK>>>(gate_ptr, up_ptr, gate_up_ptr,
                                                              h_batch_count, intermediate_dim, p->swiglu_limit);
        }

        // MLP2 (Down projection)
        {
            TIME_SCOPE(matmul_kernel_timer);
            matmul(expert_output_ptr, gate_up_ptr,
                   d_w_mlp2 + (layer_idx * n_experts + expert_id) * hidden_dim * intermediate_dim,
                   h_batch_count, intermediate_dim, hidden_dim);
        }

        // Add MLP2 bias
        {
            TIME_SCOPE(add_bias_timer);
            dim3 bias_grid((h_batch_count * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            add_bias_kernel<<<bias_grid, THREADS_PER_BLOCK>>>(
                expert_output_ptr, d_b_mlp2 + (layer_idx * n_experts + expert_id) * hidden_dim, h_batch_count, hidden_dim);
            HIP_CHECK(hipGetLastError());
        }
    }

    // --- 3. SCATTER STEP ---
    // The scatter kernel works as before, but we launch it over the total number of processed tokens.
    if (total_tokens > 0)
    {
        dim3 scatter_grid((total_tokens * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        {
            TIME_SCOPE(scatter_expert_outputs_kernel_timer);
            scatter_expert_outputs_kernel<<<scatter_grid, THREADS_PER_BLOCK>>>(
                d_e_agg, d_expert_output_buffer, d_expert_indices, d_expert_weights,
                total_tokens, hidden_dim);
            HIP_CHECK(hipGetLastError());
        }
    }

    // --- FINAL RESIDUAL CONNECTION ---
    {
        TIME_SCOPE(accumulate_kernel_timer);
        accumulate_kernel<<<(batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_x, d_e_agg, 1.0f, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    // Free the temporary buffers used in the gather operation
    HIP_CHECK(hipFree(d_expert_counts));
    HIP_CHECK(hipFree(d_expert_offsets));
    HIP_CHECK(hipFree(d_expert_write_idx));
}

float *forward_batch_gpu(Transformer *transformer, int *tokens, int batch_size)
{
    static Timer copy_embed_timer("copy_embeddings_forward", true);
    static Timer rms_norm_timer("RMSNorm_forward", true);
    static Timer matmul_kernel_simple_timer("MatMulKernelSimple_forward", true);
    Config *p = &transformer->config;

    int hidden_dim = p->hidden_dim;

    // Copy tokens to GPU
    HIP_CHECK(hipMemcpy(d_current_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_positions, positions, batch_size * sizeof(int), hipMemcpyHostToDevice));

    // Copy token embeddings
    dim3 embed_grid((batch_size * hidden_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    {
        TIME_SCOPE(copy_embed_timer);
        copy_embeddings_kernel<<<embed_grid, THREADS_PER_BLOCK>>>(
            d_x, d_token_embedding_table, d_current_tokens, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }
    // Forward through all layers - UNCOMMENTED: All layers now enabled
    for (int l = 0; l < p->n_layers; l++)
    {
        attention_gpu(transformer, l, batch_size);
        moe_gpu(transformer, l, batch_size);
    }
    // Final RMSNorm - UNCOMMENTED: Now enabled with GPU weight pointer
    dim3 final_norm_grid(batch_size);
    dim3 final_norm_block(THREADS_PER_BLOCK);
    {
        TIME_SCOPE(rms_norm_timer);
        rmsnorm_kernel<<<final_norm_grid, final_norm_block>>>(
            d_x, d_x, d_rms_out_w, batch_size, hidden_dim);
        HIP_CHECK(hipGetLastError());
    }

    dim3 block_dim(32, 32); // A 2D block, e.g., 32x32 = 1024 threads.
    dim3 grid_dim;
    grid_dim.x = batch_size;
    grid_dim.y = (p->vocab_size + block_dim.y - 1) / block_dim.y; // Ceiling division

    {
        TIME_SCOPE(matmul_kernel_simple_timer);
        // Launch the simple kernel
        matmul(
            d_logits, d_x, d_out_w, batch_size, hidden_dim, p->vocab_size);
    }
    // Copy logits back to CPU (you might want to keep this on GPU for sampling)
    static float *h_logits = nullptr;
    if (!h_logits)
    {
        h_logits = (float *)malloc(batch_size * p->vocab_size * sizeof(float));
    }
    HIP_CHECK(hipMemcpy(h_logits, d_logits, batch_size * p->vocab_size * sizeof(float), hipMemcpyDeviceToHost));
    return h_logits;
}

long long batched_generate_gpu(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, Requests *requests)
{

    Config *p = &transformer->config;
    long long total_tokens_generated = 0;

    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE)
    {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                     ? (requests->num_reqs - req_start)
                                     : BATCH_SIZE;

        // Initialize batch
        for (int b = 0; b < current_batch_size; b++)
        {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);

            // Encode prompt
            encode(tokenizer, input_seq, 1, 0, prompt_tokens[b],
                   &prompt_lens[b], p->initial_context_length);

            if (prompt_lens[b] < 1)
            {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                prompt_lens[b] = 1;
                prompt_tokens[b][0] = 1; // BOS token
            }

            // Initialize sequence state
            positions[b] = 0;
            finished[b] = false;
            current_tokens[b] = prompt_tokens[b][0];
        }

        // Generation loop
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;

        for (int step = 0; step < max_steps && alive > 0; step++)
        {
            // Forward pass on GPU
            float *logits = forward_batch_gpu(transformer, current_tokens, current_batch_size);

            // Sample next tokens (on CPU for now, could be moved to GPU)
            for (int b = 0; b < current_batch_size; b++)
            {
                if (finished[b])
                    continue;

                int req_idx = req_start + b;
                int pos = positions[b];
                float *logits_b = logits + b * p->vocab_size;

                int next_token;
                if (pos < prompt_lens[b] - 1)
                {
                    // Still processing prompt
                    next_token = prompt_tokens[b][pos + 1];
                }
                else
                {
                    // Generate new token
                    next_token = sample(sampler, logits_b);

                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1);
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }

                // Check for termination
                if (next_token == 199999 || next_token == 200002 || pos >= max_steps - 1)
                {
                    --alive;
                    finished[b] = true;
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1) + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len)
                    {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }

                // Update for next iteration
                positions[b]++;
                current_tokens[b] = next_token;
            }
        }

        // Print results
        for (int b = 0; b < current_batch_size; b++)
        {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            int *output_tokens = get_tok_gen_ptr(requests, req_idx);

            // Print the original prompt string
            safe_printf(input_seq);
            printf("!");

            // Decode and print generated tokens
            int last_prompt_token = prompt_tokens[b][prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0;; ++i)
            {
                int token = output_tokens[i];
                if (token == -1)
                    break;

                const char *piece = decode_piece(tokenizer, prev_token, token);
                safe_printf(piece);
                prev_token = token;
            }
            printf("\n");
        }
        fflush(stdout);
    }

    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
{
    return batched_generate_gpu(transformer, tokenizer, sampler, requests);
}

#endif // GETP_RUN