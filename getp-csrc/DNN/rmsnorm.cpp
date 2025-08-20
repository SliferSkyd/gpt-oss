
#include <cmath>

void rmsnorm_batch_getp(float *o, float *x, float *weight, int batch_size, int size) {
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *x_b = x + b * size;
        float *o_b = o + b * size;

        // calculate sum of squares
        double ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x_b[j] * x_b[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o_b[j] = weight[j] * (ss * x_b[j]);
        }
    }
}

