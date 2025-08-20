



void sdpa_batch_getp(Transformer *transformer, RunState *s, unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int head_dim = p->head_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;

    for (int b = 0; b < batch_size; b++) {
        int pos = positions[b];
        int loff = l * p->seq_len * kv_dim;

        float *q_b = s->q + b * p->n_attn_heads * head_dim;
        float *att_b = s->att + b * p->n_attn_heads * p->seq_len;
        float *tb_b = s->tb + b * p->n_attn_heads * head_dim;
        float *key_cache_b = s->key_cache + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = s->value_cache + b * p->n_layers * p->seq_len * kv_dim;
        float *mask_b = s->mask; // Mask is shared across the batch

        // multihead attention. iterate over all heads
#pragma omp parallel for
        for (int h = 0; h < p->n_attn_heads; h++) {
            // get the query vector for this head
            float *q = q_b + h * head_dim;
            // attention scores for this head
            float *att = att_b + h * p->seq_len;

            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float *k = key_cache_b + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // calculate the attention score as the dot product of q and k
                double score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_dim);

                // Apply sliding window mask if enabled
                if (p->sliding_window > 0 && (l % 2 == 0)) {
                    score += mask_b[pos * p->seq_len + t];
                }
                // save the score to the attention buffer
                att[t] = score;
            }

            // Add attention sink score
            att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];

            // softmax the scores to get attention weights
            softmax_getp(att, pos + 2);

            // weighted sum of the values
            float *tb = tb_b + h * head_dim;
            memset(tb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float *v = value_cache_b + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // accumulate the weighted value into tb
                accumulate(tb, v, att[t], head_dim);
            }
        }
    }
}


void attention_batch_getp(Transformer *transformer, RunState *s,
                          unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;

    // RMSNorm for all batch items
    rmsnorm_batch_getp(s->t, s->x,
                       w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

    // QKV projection for entire batch
    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    float *b_qkv = w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    matmul_batch_getp(s->qkv, s->t, w_qkv, batch_size,
                      hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

    // Add bias and separate Q, K, V for all batch items
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *qkv_b = s->qkv + b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *q_b = s->q + b * p->n_attn_heads * head_dim;
        float *k_b = s->k + b * p->n_kv_heads * head_dim;
        float *v_b = s->v + b * p->n_kv_heads * head_dim;

        // Add bias
        accumulate(qkv_b, b_qkv, 1.0f, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        // Separate Q, K, V
        memcpy(q_b, qkv_b, head_dim * p->n_attn_heads * sizeof(float));
        memcpy(k_b, qkv_b + head_dim * p->n_attn_heads, head_dim * p->n_kv_heads * sizeof(float));
        memcpy(v_b, qkv_b + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
               head_dim * p->n_kv_heads * sizeof(float));
    }

    // Apply rotary embeddings
    apply_rotary_emb_batch_getp(s->q, cos_vals, sin_vals, batch_size,
                                positions, p->n_attn_heads, head_dim);
    apply_rotary_emb_batch_getp(s->k, cos_vals, sin_vals, batch_size,
                                positions, p->n_kv_heads, head_dim);

    // Update KV cache with the new, rotary-embedded values
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        int pos = positions[b];
        int loff = l * p->seq_len * kv_dim;
        float *key_cache_b = s->key_cache + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = s->value_cache + b * p->n_layers * p->seq_len * kv_dim;

        float *k_b = s->k + b * p->n_kv_heads * head_dim;
        float *v_b = s->v + b * p->n_kv_heads * head_dim;

        memcpy(key_cache_b + loff + pos * kv_dim, k_b, kv_dim * sizeof(float));
        memcpy(value_cache_b + loff + pos * kv_dim, v_b, kv_dim * sizeof(float));
    }

    // Scaled dot-product attention
    sdpa_batch_getp(transformer, s, l, batch_size);

    // Output projection
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;

    matmul_batch_getp(s->tb2, s->tb, w_o, batch_size,
                      head_dim * p->n_attn_heads, hidden_dim);

    // Add bias and residual connection
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *tb2_b = s->tb2 + b * hidden_dim;
        accumulate(tb2_b, b_o, 1.0f, hidden_dim);
        accumulate(x_b, tb2_b, 1.0f, hidden_dim);
    }
}