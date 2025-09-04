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

    // Calculate sum of squares with double precision
    double ss = 0.0;
    for (int i = tid; i < size; i += blockDim.x)
    {
        double val = (double)x[i];
        ss += val * val;
    }
    shared_ss[tid] = (float)ss;
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

    // Normalize and scale - use double precision for intermediate calculation
    for (int i = tid; i < size; i += blockDim.x)
    {
        double weight_fp64 = (double)__bfloat162float(weight[i]);
        double x_val = (double)x[i];
        double ss_val = (double)ss;
        o[i] = (float)(weight_fp64 * (ss_val * x_val));
    }
}
