#pragma once
void matmul_batch_getp(float *xout, float *x, float *w, int batch_size, int n, int d);

#include "../getp-csrc/BLAS.cpp"