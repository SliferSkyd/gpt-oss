
void matmul_batch_getp(float *xout, float *x, float *w, int batch_size, int n, int d) {
    // Batch matrix multiplication: [batch_size, n] @ [n, d] -> [batch_size, d]
    for (int b = 0; b < batch_size; b++) {
        float *x_b = x + b * n;
        float *xout_b = xout + b * d;
        int i;
#pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            double val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[1ll * i * n + j] * x_b[j];
            }
            xout_b[i] = val;
        }
    }
}