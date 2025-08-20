#include <cmath>


void swiglu_batch_getp(float *gate, float *up, float *gate_up, int batch_size,
                       int intermediate_dim, float swiglu_limit) {
    const float alpha = 1.702f;

#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *gate_b = gate + b * intermediate_dim;
        float *up_b = up + b * intermediate_dim;
        float *gate_up_b = gate_up + b * intermediate_dim;

        for (int i = 0; i < intermediate_dim; i++) {
            float val = gate_b[i];
            float up_val = up_b[i];

            // Clamping
            if (val > swiglu_limit) val = swiglu_limit;
            if (up_val > swiglu_limit) up_val = swiglu_limit;
            if (up_val < -swiglu_limit) up_val = -swiglu_limit;

            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-alpha * val)));
            // elementwise multiply with up(x)
            val *= (up_val + 1.0f); // gpt-oss adds an extra bias of 1 to the up layer
            gate_up_b[i] = val;
        }
    }
}
