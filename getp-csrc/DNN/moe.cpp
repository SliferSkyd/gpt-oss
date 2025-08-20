
void moe_batch_getp(Transformer *transformer, RunState *s,
                    unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int hidden_dim = p->hidden_dim;
    int n_experts = p->n_experts;

    // FFN RMSNorm
    rmsnorm_batch_getp(s->t, s->x,
                       w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

    // Compute router scores for all batch items
    float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    float *b_router = w->b_router + 1ll * l * n_experts;

    matmul_batch_getp(s->router_score, s->t, w_router,
                      batch_size, hidden_dim, n_experts);

    // Add bias and select top-k experts for each batch item
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *router_score_b = s->router_score + b * n_experts;
        float *topk_v_b = s->topk_v + b * p->experts_per_token;
        int *topk_i_b = s->topk_i + b * p->experts_per_token;

        // Add bias
        accumulate(router_score_b, b_router, 1.0f, n_experts);

        // Select top-k experts
        topk(topk_v_b, topk_i_b, router_score_b, n_experts, p->experts_per_token);

        // Normalize selected experts
        softmax_getp(topk_v_b, p->experts_per_token);
    }

    // Initialize expert aggregation buffer
    memset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float));

    for (int expert_id = 0; expert_id < n_experts; expert_id++) {
        float expert_weights[BATCH_SIZE];
        int batch_indices[BATCH_SIZE];
        int batch_count = 0;

        // Gather all tokens in the batch that need to be processed by this expert
        for (int b = 0; b < batch_size; b++) {
            int *topk_i_b = s->topk_i + b * p->experts_per_token;
            float *topk_v_b = s->topk_v + b * p->experts_per_token;

            for (int i = 0; i < p->experts_per_token; ++i) {
                if (topk_i_b[i] == expert_id) {
                    batch_indices[batch_count] = b;
                    expert_weights[batch_count] = topk_v_b[i];
                    // Copy the input for this token to our temporary buffer
                    memcpy(expert_input_buffer + batch_count * hidden_dim,
                           s->t + b * hidden_dim,
                           hidden_dim * sizeof(float));
                    batch_count++;
                }
            }
        }

        if (batch_count == 0) continue;

        // Get expert weights
        float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + expert_id) * (2 * p->intermediate_dim) * hidden_dim;
        float *b_mlp1 = w->b_mlp1 + 1ll * (l * n_experts + expert_id) * (2 * p->intermediate_dim);
        float *w_mlp2 = w->w_mlp2 + 1ll * (l * n_experts + expert_id) * hidden_dim * p->intermediate_dim;
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + expert_id) * hidden_dim;

        // Process this expert with the selected batch items.
        // mlp_batch_getp will write its output to s->tb2
        mlp_batch_getp(transformer, s, w_mlp1, b_mlp1, w_mlp2, b_mlp2,
                       expert_weights, batch_count, expert_input_buffer, NULL);

        // Scatter and accumulate the results from s->tb2 back to the main e_agg buffer
        for (int i = 0; i < batch_count; i++) {
            int original_batch_idx = batch_indices[i];
            float weight = expert_weights[i];
            float *result = s->tb2 + i * hidden_dim;
            float *destination = s->e_agg + original_batch_idx * hidden_dim;

            accumulate(destination, result, weight, hidden_dim);
        }
    }

    // Residual connection
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *e_agg_b = s->e_agg + b * hidden_dim;
        accumulate(x_b, e_agg_b, 1.0f, hidden_dim);
    }
}
