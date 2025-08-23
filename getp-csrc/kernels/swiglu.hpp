#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

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
