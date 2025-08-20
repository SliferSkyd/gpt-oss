void mlp_batch_getp(Transformer *transformer, RunState *s,
                    float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2,
                    float *expert_weights, int batch_size, float *t_buffer, float* e_agg_buffer) {
    Config *p = &transformer->config;

    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;

    // First linear layer
    matmul_batch_getp(s->mlp1_out, t_buffer, w_mlp1,
                      batch_size, hidden_dim, 2 * intermediate_dim);

    // Add bias and split into gate and up
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *mlp1_out_b = s->mlp1_out + b * 2 * intermediate_dim;
        float *gate_b = s->gate + b * intermediate_dim;
        float *up_b = s->up + b * intermediate_dim;

        for (int j = 0; j < intermediate_dim; j++) {
            gate_b[j] = mlp1_out_b[2 * j] + b_mlp1[2 * j];
            up_b[j] = mlp1_out_b[2 * j + 1] + b_mlp1[2 * j + 1];
        }
    }

    // Apply SwiGLU activation
    swiglu_batch_getp(s->gate, s->up,
                      s->gate_up, batch_size, intermediate_dim, p->swiglu_limit);

    // Second linear layer
    matmul_batch_getp(s->tb2, s->gate_up, w_mlp2,
                      batch_size, intermediate_dim, hidden_dim);

    // Add bias. The result is now in tb2.
    // The calling function (moe_batch_getp) will handle accumulation.
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *tb2_b = s->tb2 + b * hidden_dim;
        accumulate(tb2_b, b_mlp2, 1.0f, hidden_dim);
    }
}