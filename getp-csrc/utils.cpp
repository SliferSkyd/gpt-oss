float *cos_vals, *sin_vals;
int **prompt_tokens;
int *current_tokens;
bool *finished;
int *positions;
int *prompt_lens;
float *expert_input_buffer; // Buffer for MoE expert inputs


void accumulate_batch(float *a, float *b, float factor, int batch_size, int size) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        float *a_b = a + batch_idx * size;
        float *b_b = b + batch_idx * size;
        for (int i = 0; i < size; ++i) {
            a_b[i] += b_b[i] * factor;
        }
    }
}
void accumulate(float *a, float *b, float factor, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] += b[i] * factor;
    }
}
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                             float scaling_factor,
                                             float initial_context_length,
                                             float ntk_beta, float ntk_alpha,
                                             float *concentration_out,
                                             float *inv_freq_out // length head_dim/2
)
{
    int d_half = head_dim / 2;

    // freq[i] = base ** (i / head_dim)
    float *freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++)
    {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f)
    {
        // YaRN concentration
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        // NTK by parts
        float low = d_half *
                    logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                    logf(base);
        float high = d_half *
                     logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                     logf(base);

        assert(0 < low && low < high && high < d_half - 1);

        // interpolation = 1 / (scaling_factor * freq)
        // extrapolation = 1 / freq
        for (int i = 0; i < d_half; i++)
        {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];

            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0)
                ramp = 0;
            if (ramp > 1)
                ramp = 1;

            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    }
    else
    {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++)
        {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }

    *concentration_out = concentration;

    free(freq);
}

void compute_cos_sin_getp(int pos, // position index
                          float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha,
                          float *cos_out, // shape: head_dim/2
                          float *sin_out  // shape: head_dim/2
)
{
    int d_half = head_dim / 2;

    // Get concentration + inv_freq
    float concentration;
    float *inv_freq = (float *)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                            initial_context_length, ntk_beta,
                                            ntk_alpha, &concentration, inv_freq);

    // Compute cos and sin for this position
    for (int j = 0; j < d_half; j++)
    {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }

    free(inv_freq);
}

void malloc_batch_run_state(RunState *s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = p->head_dim * p->n_kv_heads;
    s->x = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->hidden_dim, sizeof(float)));
    s->t = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->hidden_dim, sizeof(float)));
    s->tb = reinterpret_cast<float *>(
        calloc(BATCH_SIZE * p->head_dim * p->n_attn_heads, sizeof(float)));
    s->tb2 = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->hidden_dim, sizeof(float)));

    s->router_score =
        reinterpret_cast<float *>(calloc(BATCH_SIZE * p->n_experts, sizeof(float)));
    s->topk_v =
        reinterpret_cast<float *>(calloc(BATCH_SIZE * p->experts_per_token, sizeof(float)));
    s->topk_i =
        reinterpret_cast<int *>(calloc(BATCH_SIZE * p->experts_per_token, sizeof(int)));

    s->mlp1_out =
        reinterpret_cast<float *>(calloc(BATCH_SIZE * 2 * p->intermediate_dim, sizeof(float)));
    s->gate =
        reinterpret_cast<float *>(calloc(BATCH_SIZE * p->intermediate_dim, sizeof(float)));
    s->up = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->intermediate_dim, sizeof(float)));
    s->gate_up =
        reinterpret_cast<float *>(calloc(BATCH_SIZE * p->intermediate_dim, sizeof(float)));
    s->e_agg = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->hidden_dim, sizeof(float)));

    s->qkv = reinterpret_cast<float *>(calloc(
        BATCH_SIZE * p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads), sizeof(float)));
    s->q = reinterpret_cast<float *>(
        calloc(BATCH_SIZE * p->n_attn_heads * p->head_dim, sizeof(float)));

    s->key_cache = reinterpret_cast<float *>(
        calloc(BATCH_SIZE * p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->value_cache = reinterpret_cast<float *>(
        calloc(BATCH_SIZE * p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->att = reinterpret_cast<float *>(
        calloc(BATCH_SIZE * p->n_attn_heads * p->seq_len, sizeof(float)));
    s->logits = reinterpret_cast<float *>(calloc(BATCH_SIZE * p->vocab_size, sizeof(float)));
    s->mask = p->sliding_window > 0 ? reinterpret_cast<float *>(calloc(
                                            p->seq_len * p->seq_len, sizeof(float)))
                                    : NULL;
    s->k = reinterpret_cast<float *>(calloc(BATCH_SIZE * kv_dim, sizeof(float)));
    s->v = reinterpret_cast<float *>(calloc(BATCH_SIZE * kv_dim, sizeof(float)));
    
    // Buffer for MoE expert inputs, allocated once.
    expert_input_buffer = reinterpret_cast<float *>(malloc(BATCH_SIZE * p->hidden_dim * sizeof(float)));

    // ensure all mallocs went fine
    if (!s->x || !s->t || !s->tb || !s->tb2 || !s->qkv || !s->q ||
        !s->key_cache || !s->value_cache || !s->att || !s->logits ||
        (p->sliding_window > 0 && !s->mask) || !s->e_agg || !expert_input_buffer) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    // initialize mask
    for (int i = 0; i < p->seq_len; i++) {
        for (int j = 0; j < p->seq_len; j++) {
            if (p->sliding_window > 0 && i - j >= p->sliding_window) {
                s->mask[i * p->seq_len + j] = -INFINITY; // Sliding window mask
            }
        }
    }
}

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head Adapted from
    // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
    // RoPE with YaRN scaling adapted from Python code

    Config *p = &transformer->config;
    RunState *s = &transformer->state;

    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    cos_vals =
        reinterpret_cast<float *>(malloc((p->head_dim / 2) * p->seq_len * sizeof(float)));
    sin_vals =
        reinterpret_cast<float *>(malloc((p->head_dim / 2) * p->seq_len * sizeof(float)));
    for (int pos = 0; pos < p->seq_len; ++pos)
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                           p->initial_context_length, ntk_beta, ntk_alpha, cos_vals + (pos * p->head_dim / 2),
                           sin_vals + (pos * p->head_dim / 2));

    // re-allocate run state (more space for batching)
    malloc_batch_run_state(s, p);

    prompt_tokens = (int**)malloc(BATCH_SIZE * sizeof(int*));
    current_tokens = (int*)malloc(BATCH_SIZE * sizeof(int));

    for (int b = 0; b < BATCH_SIZE; b++) {
        prompt_tokens[b] = (int*)malloc((p->seq_len + 3) * sizeof(int));
    }

    finished = (bool*)malloc(BATCH_SIZE * sizeof(bool));
    positions = (int*)malloc(BATCH_SIZE * sizeof(int));
    prompt_lens = (int*)malloc(BATCH_SIZE * sizeof(int));
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
    // Do not inference here
    // You should handle the finish process
    // TODO:
    // - Memory deallocation
    // - Unload model
    // - ...
}

