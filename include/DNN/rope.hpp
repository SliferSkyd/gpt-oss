#pragma once

void apply_rotary_emb_batch_getp(float *x, float *cos, float *sin, int batch_size,
                                 int *positions, int n_heads, int head_dim);


#include "../../getp-csrc/DNN/rope.cpp"