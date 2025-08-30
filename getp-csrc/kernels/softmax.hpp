#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void softmax_kernel(float *x, int batch_size, int size)
{
    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    float *batch_x = x + 1LL * batch_idx * size;

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

// sample_argmax_batch_hip.hpp
#include <hip/hip_runtime.h>
#include <limits.h>   // INT_MAX
#include <math.h>     // optional: isnan/isfinite

// Pair of (value, index)
struct ArgMaxPair {
    float p;
    int   i;
};

// Merge that mimics the CPU loop's behavior:
// - Prefer larger probability
// - On equal probabilities, keep the *earlier* index (smaller i)
__device__ inline ArgMaxPair merge_argmax(ArgMaxPair a, ArgMaxPair b) {
    // Note: if probabilities are equal, choose the smaller index (earlier)
    const bool take_b = (b.p > a.p) || ((b.p == a.p) && (b.i < a.i));
    return take_b ? b : a;
}

// Warp-wide argmax reduction
__device__ inline ArgMaxPair warpReduceArgMax(ArgMaxPair v) {
    // HIP warpSize is 64 on AMD, 32 on NVIDIA; this handles both.
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
        float p_other = __shfl_down(v.p, off);
        int   i_other = __shfl_down(v.i, off);
        v = merge_argmax(v, {p_other, i_other});
    }
    return v;
}

// One block per row: finds argmax over n columns for that row.
// probs: [batch_size, ld] (row-major), usually ld == n
__global__ void sample_argmax_kernel(
    const float* __restrict__ probs,
    int* __restrict__ out_idx,
    int batch_size, int n)
{
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const float* row = probs + (size_t)b * n;

    // Each thread scans a strided chunk and keeps a local best.
    ArgMaxPair best = {-INFINITY, INT_MAX};  // INT_MAX ensures "earlier index wins" on ties

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row[i];
        // If you want to treat NaNs as -inf (like many reductions), uncomment:
        // if (!isfinite(p)) p = -INFINITY;
        best = merge_argmax(best, {p, i});
    }

    // Warp reduce
    best = warpReduceArgMax(best);

    // Cross-warp reduce using shared memory
    // Max number of warps per block: 32 (NVIDIA) or 16 (AMD @ 1024 threads, warpSize 64)
    __shared__ float s_p[32];
    __shared__ int   s_i[32];

    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x / warpSize;

    if (lane == 0) {
        s_p[wid] = best.p;
        s_i[wid] = best.i;
    }
    __syncthreads();

    // Final reduce by warp 0
    if (wid == 0) {
        ArgMaxPair v;
        const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        v.p = (lane < nwarps) ? s_p[lane] : -INFINITY;
        v.i = (lane < nwarps) ? s_i[lane] : INT_MAX;
        v = warpReduceArgMax(v);
        if (lane == 0) out_idx[b] = v.i;
    }
}

inline void sample_argmax(
    const float* d_probs, int* d_out_idx,
    int batch_size, int n,
    hipStream_t stream = nullptr,
    int threads_per_block = 256)
{
    dim3 grid(batch_size);
    dim3 block(threads_per_block);
    hipLaunchKernelGGL(
        sample_argmax_kernel,
        grid, block, /*sharedMemBytes=*/0, stream,
        d_probs, d_out_idx, batch_size, n
    );
}