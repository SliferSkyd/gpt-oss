#pragma once

void rmsnorm_batch_getp(float *o, float *x, float *weight, int batch_size, int size);

#include "../getp-csrc/DNN/rmsnorm.cpp"