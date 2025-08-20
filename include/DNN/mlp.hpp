#pragma once
void mlp_batch_getp(Transformer *transformer, RunState *s,
                    float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2,
                    float *expert_weights, int batch_size, float *t_buffer, float* e_agg_buffer);
#include "../../getp-csrc/DNN/mlp.cpp"