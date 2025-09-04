#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

__global__ void swiglu_kernel(float *mlp1_out, __hip_bfloat16 *bias, float *output,
                              int batch_size, int intermediate_dim, float limit)
{
    size_t idx = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    // size_t mlp1_idx = 1LL * batch_idx * 2 * intermediate_dim;
    size_t bias_idx = idx % intermediate_dim;
    size_t total = 1LL * batch_size * intermediate_dim;

    if (idx >= total)
        return;

    const float alpha = 1.702f;
    float gate_val = mlp1_out[idx] + __bfloat162float(bias[bias_idx]);
    float up_val = mlp1_out[idx + total] + __bfloat162float(bias[bias_idx + intermediate_dim]);

    // Clamping
    gate_val = fminf(fmaxf(gate_val, -limit), limit);
    up_val = fminf(fmaxf(up_val, -limit), limit);

    // SiLU activation
    gate_val *= (1.0f / (1.0f + __expf(-alpha * gate_val)));
    gate_val *= (up_val + 1.0f);

    output[idx] = gate_val;
}
