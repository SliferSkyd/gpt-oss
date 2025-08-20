void sdpa_batch_getp(Transformer *transformer, RunState *s, unsigned long long l, int batch_size);
void attention_batch_getp(Transformer *transformer, RunState *s,
                          unsigned long long l, int batch_size);

#include "../../getp-csrc/DNN/attention.cpp"                       