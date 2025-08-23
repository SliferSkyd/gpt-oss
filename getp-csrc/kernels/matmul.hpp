#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <rocwmma/rocwmma.hpp>
#include "../config.hpp"

__global__ void matmul_kernel(float *output, const float *input, const __hip_bfloat16 *weight,
                              int batch_size, int input_dim, int output_dim)
{
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= output_dim)
        return;

    const float *x = input + batch_idx * input_dim;
    float *out = output + batch_idx * output_dim;

    __shared__ float shared_val[THREADS_PER_BLOCK];

    float val = 0.0f;
    for (int i = tid; i < input_dim; i += blockDim.x)
    {
        // Convert bfloat16 weight to fp32 on-the-fly
        float weight_fp32 = __bfloat162float(weight[out_idx * input_dim + i]);
        val += weight_fp32 * x[i];
    }
    shared_val[tid] = val;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_val[tid] += shared_val[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[out_idx] = shared_val[0];
    }
}



// Add bounds checking to matmul kernel
__global__ void matmul_kernel_safe(float *output, const float *input, const __hip_bfloat16 *weight,
                                   int batch_size, int input_dim, int output_dim)
{
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= output_dim)
        return;

    const float *x = input + batch_idx * input_dim;
    float *out = output + batch_idx * output_dim;

    __shared__ float shared_val[THREADS_PER_BLOCK];

    float val = 0.0f;
    for (int i = tid; i < input_dim; i += blockDim.x)
    {
        // Convert bfloat16 weight to fp32 on-the-fly
        float weight_fp32 = __bfloat162float(weight[out_idx * input_dim + i]);
        val += weight_fp32 * x[i];
    }

    if (tid < THREADS_PER_BLOCK)
    {
        shared_val[tid] = val;
    }
    __syncthreads();

    // Reduction with bounds checking
    for (int stride = min(blockDim.x, THREADS_PER_BLOCK) / 2; stride > 0; stride /= 2)
    {
        if (tid < stride && tid + stride < THREADS_PER_BLOCK)
        {
            shared_val[tid] += shared_val[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[out_idx] = shared_val[0];
    }
}

__global__ void matmul_kernel_simple(float *output, const float *input, const __hip_bfloat16 *weight,
                                     int batch_size, int input_dim, int output_dim)
{
    // Each thread will compute one element in the output matrix.
    // blockIdx.x maps to the batch dimension.
    // The combination of blockIdx.y, blockDim.y, and threadIdx.y maps to the output dimension.
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check: Ensure we are not writing out of bounds.
    if (batch_idx >= batch_size || out_idx >= output_dim)
    {
        return;
    }

    // Initialize accumulator for this thread's output value.
    float sum = 0.0f;

    // Get a pointer to the start of the correct input vector for this batch.
    const float *x_b = input + batch_idx * input_dim;

    // Get a pointer to the start of the correct weight matrix row for this output.
    const __hip_bfloat16 *w_row = weight + out_idx * input_dim;

    // --- Core Logic (Same as CPU) ---
    // This single thread performs the entire dot product.
    for (int j = 0; j < input_dim; j++)
    {
        // Convert bfloat16 weight to fp32 on-the-fly
        float weight_fp32 = __bfloat162float(w_row[j]);
        sum += weight_fp32 * x_b[j];
    }
    // ---------------------------------

    // Write the final result to the correct position in the output matrix.
    output[batch_idx * output_dim + out_idx] = sum;
}

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
