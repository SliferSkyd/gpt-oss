#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

// GPU kernels with bfloat16 weights support
__global__ void rmsnorm_kernel(float *output, const float *input, const __hip_bfloat16 *weight,
                               int batch_size, int size)
{
    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    const float *x = input + 1LL * batch_idx * size;
    float *o = output + 1LL * batch_idx * size;

    // Shared memory for reduction
    __shared__ float shared_ss[THREADS_PER_BLOCK];

    // Calculate sum of squares
    double ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
    {
        ss += (double)x[i] * x[i];
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
        o[i] = (double)(__bfloat162float(weight[i])) * (ss * x[i]);
    }
}

// GPU kernel: RMSNorm with bfloat16 weights and bfloat16 output
__global__ void rmsnorm_kernel_bf16_out(__hip_bfloat16 *output,
                                        const float *input,
                                        const __hip_bfloat16 *weight,
                                        int batch_size, int size)
{
    const size_t batch_idx = blockIdx.x;
    const size_t tid = threadIdx.x;

    if (batch_idx >= (size_t)batch_size) return;

    const float *x = input + 1LL * batch_idx * size;
    __hip_bfloat16 *o = output + 1LL * batch_idx * size;

    // Shared memory for reduction (one float per thread)
    __shared__ float shared_ss[THREADS_PER_BLOCK];

    // Sum of squares
    double ss = 0.0;
    for (int i = tid; i < size; i += blockDim.x) {
        const double v = (double)x[i];
        ss += v * v;
    }
    shared_ss[tid] = (float)ss;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < (size_t)stride) {
            shared_ss[tid] += shared_ss[tid + stride];
        }
        __syncthreads();
    }

    // Compute inverse RMS (in float)
    if (tid == 0) {
        float mean_ss = shared_ss[0] / (float)size;
        mean_ss += 1e-5f;
        shared_ss[0] = rsqrtf(mean_ss);  // 1 / sqrt(mean_ss)
    }
    __syncthreads();

    const float inv_rms = shared_ss[0];

    // Normalize, scale by weight (bf16->f32), then convert result to bf16
    for (int i = tid; i < size; i += blockDim.x) {
        const float wi = __bfloat162float(weight[i]);
        const float yi = wi * (inv_rms * x[i]);
        o[i] = __float2bfloat16(yi);
    }
}
