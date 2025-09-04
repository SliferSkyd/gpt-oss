#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

// GPU kernels with bfloat16 weights support
__global__ void rmsnorm_kernel(float *output, const float *input, const float *weight,
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
        o[i] = (double)(weight[i]) * (ss * x[i]);
    }
}
