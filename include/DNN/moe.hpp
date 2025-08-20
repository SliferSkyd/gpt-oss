#pragma once
void moe_batch_getp(Transformer *transformer, RunState *s,
                    unsigned long long l, int batch_size);


#include "../../getp-csrc/DNN/moe.cpp"