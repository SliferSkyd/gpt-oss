#pragma once
void swiglu_batch_getp(float *gate, float *up, float *gate_up, int batch_size,
                       int intermediate_dim, float swiglu_limit);
#include "../getp-csrc/DNN/swiglu.cpp"