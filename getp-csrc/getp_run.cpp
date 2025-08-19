// TODO: Modify this file to optimize end-to-end throughput


#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>

#ifndef GETP_RUN
#define GETP_RUN

#define BATCH_SIZE 32

float *cos_vals, *sin_vals;
int **prompt_tokens;
int *current_tokens;
bool *finished;
int *positions;
int *prompt_lens;

void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *concentration_out,
                                        float *inv_freq_out // length head_dim/2
) {
  int d_half = head_dim / 2;

  // freq[i] = base ** (i / head_dim)
  float *freq = (float *)malloc(d_half * sizeof(float));
  for (int i = 0; i < d_half; i++) {
    freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
  }

  float concentration;
  if (scaling_factor > 1.0f) {
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
    for (int i = 0; i < d_half; i++) {
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
  } else {
    concentration = 1.0f;
    for (int i = 0; i < d_half; i++) {
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
) {
  int d_half = head_dim / 2;

  // Get concentration + inv_freq
  float concentration;
  float *inv_freq = (float *)malloc(d_half * sizeof(float));

  compute_concentration_and_inv_freq_getp(base, head_dim, scaling_factor,
                                     initial_context_length, ntk_beta,
                                     ntk_alpha, &concentration, inv_freq);

  // Compute cos and sin for this position
  for (int j = 0; j < d_half; j++) {
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
                                        BATCH_SIZE * p->seq_len * p->seq_len, sizeof(float)))
                                  : NULL;

  // ensure all mallocs went fine
  if (!s->x || !s->t || !s->tb || !s->tb2 || !s->qkv || !s->q ||
      !s->key_cache || !s->value_cache || !s->att || !s->logits ||
      (p->sliding_window > 0 && !s->mask) || !s->e_agg) {
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
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...

  
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
    compute_cos_sin(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
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


// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm_getp(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  double ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax_getp(float *x, int size) {
  // find max value (for numerical stability)
  double max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  double sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void matmul_getp(float *xout, float *x, float *w, int n, int d) {
  // n := in_features, d := out_features
  // W (out,in) @ x (in,) -> xout (out,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    double val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[1ll * i * n + j] * x[j];
    }
    xout[i] = val;
  }
}


void accumulate(float *a, float *b, float factor, int size) {
  for (int i = 0; i < size; ++i) {
    a[i] += b[i] * factor;
  }
}

void apply_rotary_emb_getp(float *x, float *cos, float *sin, int n_heads,
                      int head_dim) {
  int half = head_dim / 2;

  for (int h = 0; h < n_heads; h++) {
    for (int i = 0; i < half; i++) {
      // Indexing: head h, dim i
      float x1 = x[h * head_dim + i];        // first half
      float x2 = x[h * head_dim + half + i]; // second half

      float c = cos[i];
      float s = sin[i];

      float o1 = x1 * c - x2 * s;
      float o2 = x2 * c + x1 * s;

      x[h * head_dim + i] = o1;
      x[h * head_dim + half + i] = o2;
    }
  }
}

void sdpa_getp(Transformer *transformer, unsigned long long l, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul =
      p->n_attn_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

  // multihead attention_getp. iterate over all heads
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < p->n_attn_heads; h++) {
    // get the query vector for this head
    float *q = s->q + h * head_dim;
    // attention_getp scores for this head
    float *att = s->att + h * p->seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      // GQA
      float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      // calculate the attention_getp score as the dot product of q and k
      double score = 0.0f;
      for (int i = 0; i < head_dim; i++) {
        score += q[i] * k[i];
      }
      score /= sqrtf(head_dim);
      // Apply sliding window mask if enabled
      if (p->sliding_window > 0 && (l % 2 == 0)) {
        score += s->mask[pos * p->seq_len + t];
      }
      // save the score to the attention_getp buffer
      att[t] = score;
    }
    // Add attention_getp sink score
    att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
    // softmax_getp the scores to get attention_getp weights, from 0..pos inclusively
    softmax_getp(att, pos + 2);

    // weighted sum of the values
    float *tb = s->tb + h * head_dim;
    memset(tb, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      // GQA
      float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
      
      // accumulate the weighted value into xb
      accumulate(tb, v, att[t], head_dim);
    }
  }
}

void attention_getp(Transformer *transformer, float *x, unsigned long long l, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul =
      p->n_attn_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

  // s->t (hidden_dim, )
  rmsnorm_getp(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

  // key and value point to the kv cache
  int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
  s->k = s->key_cache + loff + pos * kv_dim;
  s->v = s->value_cache + loff + pos * kv_dim;

  // s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
  // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
  float *w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                (head_dim * p->n_attn_heads +
                                  2 * head_dim * p->n_kv_heads);
  float *b_qkv =
      w->b_qkv +
      1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
  matmul_getp(s->qkv, s->t, w_qkv, hidden_dim,
          (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
  // add bias
  accumulate(s->qkv, b_qkv, 1.0f, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
  // Separate q, k, v
  memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float)); // gate
  memcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
          head_dim * p->n_kv_heads * sizeof(float)); // gate
  memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
          head_dim * p->n_kv_heads * sizeof(float)); // gate

  apply_rotary_emb_getp(s->q, cos_vals + pos * (head_dim / 2), sin_vals + pos * (head_dim / 2), p->n_attn_heads, head_dim);
  apply_rotary_emb_getp(s->k, cos_vals + pos * (head_dim / 2), sin_vals + pos * (head_dim / 2), p->n_kv_heads, head_dim);

  sdpa_getp(transformer, l, pos);
  
  // final matmul_getp to get the output of the attention_getp
  float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
  float *b_o = w->b_o + 1ll * l * hidden_dim;
  matmul_getp(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
  // add bias b_o
  accumulate(s->tb2, b_o, 1.0f, hidden_dim);

  // residual connection back into x
  accumulate(x, s->tb2, 1.0f, hidden_dim);
}

void swiglu_getp(float *x, float *gate, float *up, float *gate_up, int intermediate_dim, float swiglu_getp_limit) {
  // swiglu_getp non-linearity
  const float alpha = 1.702f;
  for (int i = 0; i < intermediate_dim; i++) {
    float val = gate[i];
    float up_val = up[i];
    // Clamping
    if (val > swiglu_getp_limit)
      val = swiglu_getp_limit;
    if (up_val > swiglu_getp_limit)
      up_val = swiglu_getp_limit;
    if (up_val < -swiglu_getp_limit)
      up_val = -swiglu_getp_limit;
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-alpha * val)));
    // elementwise multiply with w_gate(x)
    val *= (up_val +
            1.0f); // gpt-oss adds an extra bias of 1 to the up layer
    gate_up[i] = val;
  }
}

void mlp_getp(Transformer *transformer, float *x, float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2, float expert_w) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  int hidden_dim = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  matmul_getp(s->mlp1_out, s->t, w_mlp1, hidden_dim,
          2 * p->intermediate_dim); // (2 * intermediate_dim, )
  accumulate(s->mlp1_out, b_mlp1, 1.0f, 2 * p->intermediate_dim);
  // Split mlp1_out into gate and up
  for (int j = 0; j < p->intermediate_dim; j++) {
    s->gate[j] = s->mlp1_out[2 * j];
    s->up[j] = s->mlp1_out[2 * j + 1];
  }

  swiglu_getp(x, s->gate, s->up, s->gate_up, p->intermediate_dim, p->swiglu_limit);

  // final matmul to get the output of the ffn
  matmul_getp(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim,
          hidden_dim); // (hidden_dim, )
  accumulate(s->tb2, b_mlp2, 1.0f, hidden_dim); // add bias b_mlp2

  // aggregate topk experts using weighted sum
  accumulate(s->e_agg, s->tb2, expert_w, hidden_dim);
}

void moe_getp(Transformer *transformer, float *x, unsigned long long l, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  int hidden_dim = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  // ffn rmsnorm
  rmsnorm_getp(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

  // MoE
  // Compute router_score
  float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
  float *b_router = w->b_router + 1ll * l * n_experts;
  matmul_getp(s->router_score, s->t, w_router, hidden_dim,
          n_experts); // s->router_score now stores router_score (n_experts, )
  // add bias b_router
  for (int i = 0; i < n_experts; i++) {
    s->router_score[i] += b_router[i];
  }
  // Select top-k experts
  topk(s->topk_v, s->topk_i, s->router_score, n_experts,
        p->experts_per_token);
  // Normalize selected experts using softmax or sigmoid
  softmax_getp(s->topk_v, p->experts_per_token); // expert

  // Route the tokens to their corresponding top-k experts
  memset(s->e_agg, 0, hidden_dim * sizeof(float));
  
  for (int idx = 0; idx < p->experts_per_token; idx++) {
    int e = s->topk_i[idx];
    float expert_w = s->topk_v[idx];
    
    float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + e) *
                                (2 * p->intermediate_dim) * hidden_dim;
    float *b_mlp1 =
        w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
        
    float *w_mlp2 =
        w->w_mlp2 +
        1ll * (l * n_experts + e) * hidden_dim *
            p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
    float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;          

    mlp_getp(transformer, x, w_mlp1, b_mlp1, w_mlp2, b_mlp2, expert_w);
  }

  // residual connection
  accumulate(x, s->e_agg, 1.0f, hidden_dim);
}

float *forward_getp(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  float *x = s->x;
  int hidden_dim = p->hidden_dim;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));

  // forward_getp all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    attention_getp(transformer, x, l, pos); // attention_getp block
    moe_getp(transformer, x, l, pos);
  }
  // final rmsnorm_getp
  rmsnorm_getp(x, x, w->rms_out_w, hidden_dim);

  // classifier into logits
  matmul_getp(s->logits, x, w->out, hidden_dim, p->vocab_size);
  return s->logits;
}

void matmul_batch_getp(float *xout, float *x, float *w, int batch_size, int n, int d) {
    // Batch matrix multiplication: [batch_size, n] @ [n, d] -> [batch_size, d]
    int b, i;
#pragma omp parallel for private(b, i)
    for (b = 0; b < batch_size; b++) {
        float *x_b = x + b * n;
        float *xout_b = xout + b * d;
        
        for (i = 0; i < d; i++) {
            double val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[1ll * i * n + j] * x_b[j];
            }
            xout_b[i] = val;
        }
    }
}

void rmsnorm_batch_getp(float *o, float *x, float *weight, int batch_size, int size) {
  for (int b = 0; b < batch_size; b++) {
    float *x_b = x + b * size;
    float *o_b = o + b * size;
    
    // calculate sum of squares
    double ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x_b[j] * x_b[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o_b[j] = weight[j] * (ss * x_b[j]);
    }
  }
}


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
        float *mask_b = s->mask + b * p->seq_len * p->seq_len;

        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_attn_heads; h++) {
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
            // For single sequence, we can use the original function
            softmax_getp(att, pos + 2);

            // weighted sum of the values
            float *tb = tb_b + h * head_dim;
            memset(tb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float *v = value_cache_b + loff + t * kv_dim + (h / kv_mul) * head_dim;
                
                // accumulate the weighted value into tb
                for (int i = 0; i < head_dim; i++) {
                    tb[i] += v[i] * att[t];
                }
            }
        }
    }
}

void accumulate_batch(float *a, float *b, float factor, int batch_size, int size) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        float *a_b = a + batch_idx * size;
        float *b_b = b + batch_idx * size;
        for (int i = 0; i < size; ++i) {
            a_b[i] += b_b[i] * factor;
        }
    }
}

void apply_rotary_emb_batch_getp(float *x, float *cos, float *sin, int batch_size, 
                                int *positions, int n_heads, int head_dim) {
    int half = head_dim / 2;

    for (int b = 0; b < batch_size; b++) {
        int pos = positions[b];
        float *x_b = x + b * n_heads * head_dim;
        float *cos_pos = cos + pos * half;
        float *sin_pos = sin + pos * half;
        
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < half; i++) {
                // Indexing: batch b, head h, dim i
                float x1 = x_b[h * head_dim + i];        // first half
                float x2 = x_b[h * head_dim + half + i]; // second half

                float c = cos_pos[i];
                float s = sin_pos[i];

                float o1 = x1 * c - x2 * s;
                float o2 = x2 * c + x1 * s;

                x_b[h * head_dim + i] = o1;
                x_b[h * head_dim + half + i] = o2;
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

    // Update KV cache pointers for current positions
    for (int b = 0; b < batch_size; b++) {
        int pos = positions[b];
        int loff = l * p->seq_len * kv_dim;
        float *key_cache_b = s->key_cache + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = s->value_cache + b * p->n_layers * p->seq_len * kv_dim;

        float *k_b = s->k + b * p->n_kv_heads * head_dim;
        float *v_b = s->v + b * p->n_kv_heads * head_dim;

        // Point to current position in cache
        float *k_cache_pos = key_cache_b + loff + pos * kv_dim;
        float *v_cache_pos = value_cache_b + loff + pos * kv_dim;
        
        // These will be updated after QKV projection
        memcpy(k_cache_pos, k_b, kv_dim * sizeof(float));
        memcpy(v_cache_pos, v_b, kv_dim * sizeof(float));
    }

    // QKV projection for entire batch
    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    float *b_qkv = w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    matmul_batch_getp(s->qkv, s->t, w_qkv, batch_size,
                     hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    
    // Add bias and separate Q, K, V for all batch items
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

    // Update KV cache with rotary-embedded values
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
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *tb2_b = s->tb2 + b * hidden_dim;
        accumulate(tb2_b, b_o, 1.0f, hidden_dim);
        accumulate(x_b, tb2_b, 1.0f, hidden_dim);
    }
}

void swiglu_batch_getp(float *gate, float *up, float *gate_up, int batch_size, 
                      int intermediate_dim, float swiglu_limit) {
    const float alpha = 1.702f;
    
    for (int b = 0; b < batch_size; b++) {
        float *gate_b = gate + b * intermediate_dim;
        float *up_b = up + b * intermediate_dim;
        float *gate_up_b = gate_up + b * intermediate_dim;
        
        for (int i = 0; i < intermediate_dim; i++) {
            float val = gate_b[i];
            float up_val = up_b[i];
            
            // Clamping
            if (val > swiglu_limit) val = swiglu_limit;
            if (up_val > swiglu_limit) up_val = swiglu_limit;
            if (up_val < -swiglu_limit) up_val = -swiglu_limit;
            
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-alpha * val)));
            // elementwise multiply with up(x)
            val *= (up_val + 1.0f); // gpt-oss adds an extra bias of 1 to the up layer
            gate_up_b[i] = val;
        }
    }
}

void mlp_batch_getp(Transformer *transformer, RunState *s, 
                   float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2, 
                   float *expert_weights, int batch_size) {
    Config *p = &transformer->config;
    
    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;

    // First linear layer
    matmul_batch_getp(s->mlp1_out, s->t, w_mlp1, 
                     batch_size, hidden_dim, 2 * intermediate_dim);
    
    // Add bias and split into gate and up
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
    
    // Add bias and accumulate with expert weights
    for (int b = 0; b < batch_size; b++) {
        float *tb2_b = s->tb2 + b * hidden_dim;
        float *e_agg_b = s->e_agg + b * hidden_dim;
        float expert_w = expert_weights[b];
        
        for (int i = 0; i < hidden_dim; i++) {
            tb2_b[i] += b_mlp2[i];
            e_agg_b[i] += tb2_b[i] * expert_w;
        }
    }
}

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
    for (int b = 0; b < batch_size; b++) {
        float *router_score_b = s->router_score + b * n_experts;
        float *topk_v_b = s->topk_v + b * p->experts_per_token;
        int *topk_i_b = s->topk_i + b * p->experts_per_token;
        
        // Add bias
        accumulate(router_score_b, b_router, 1.0f, n_experts);
        
        // Select top-k experts
        topk(topk_v_b, topk_i_b, router_score_b, n_experts, p->experts_per_token);
        
        // Normalize selected experts - use single sequence function for single batch item
        softmax_getp(topk_v_b, p->experts_per_token);
    }

    // Initialize expert aggregation buffers
    memset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float));

    for (int expert_id = 0; expert_id < n_experts; expert_id++) {
        // Find all batch items that use this expert at this expert_idx
        float expert_weights[batch_size];
        int batch_indices[batch_size];
        int batch_count = 0;
        
        for (int b = 0; b < batch_size; b++) {
            int *topk_i_b = s->topk_i + b * p->experts_per_token;
            float *topk_v_b = s->topk_v + b * p->experts_per_token;
            
            for (int i = 0; i < p->experts_per_token; ++i) {
              if (topk_i_b[i] == expert_id) {
                  batch_indices[batch_count] = b;
                  expert_weights[batch_count] = topk_v_b[i];
                  batch_count++;
              }
            }
        }
        
        if (batch_count == 0) continue;
        
        // Get expert weights
        float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + expert_id) * 
                        (2 * p->intermediate_dim) * hidden_dim;
        float *b_mlp1 = w->b_mlp1 + 1ll * (l * n_experts + expert_id) * 
                        (2 * p->intermediate_dim);
        float *w_mlp2 = w->w_mlp2 + 1ll * (l * n_experts + expert_id) * 
                        hidden_dim * p->intermediate_dim;
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + expert_id) * hidden_dim;
        
        // Create temporary batch state for this expert's computation
        // We reuse the existing buffers but only process selected batch items
        RunState temp_s = *s;
        
        // Copy input data for selected batch items to beginning of buffers
        for (int i = 0; i < batch_count; i++) {
            int src_b = batch_indices[i];
            memcpy(temp_s.t + i * hidden_dim,
                    s->t + src_b * hidden_dim,
                    hidden_dim * sizeof(float));
        }
        
        // Process this expert with the selected batch items
        mlp_batch_getp(transformer, &temp_s, w_mlp1, b_mlp1, w_mlp2, b_mlp2,
                      expert_weights, batch_count);
        
        // Copy results back to original positions in e_agg_batch
        for (int i = 0; i < batch_count; i++) {
            int dst_b = batch_indices[i];
            float *src = temp_s.e_agg + i * hidden_dim;
            float *dst = s->e_agg + dst_b * hidden_dim;
            
            for (int j = 0; j < hidden_dim; j++) {
                dst[j] += src[j];  // Accumulate from multiple experts
            }
        }
    }

    // Residual connection
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *e_agg_b = s->e_agg + b * hidden_dim;

        for (int i = 0; i < hidden_dim; i++) {
            x_b[i] += e_agg_b[i];
        }
    }
}


float *forward_batch_getp(Transformer *transformer, RunState *s,
                         int *tokens, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int hidden_dim = p->hidden_dim;

    // Copy token embeddings into x_batch
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *content_row = w->token_embedding_table + tokens[b] * hidden_dim;
        memcpy(x_b, content_row, hidden_dim * sizeof(float));
    }

    // Forward through all layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_batch_getp(transformer, s, l, batch_size);
        moe_batch_getp(transformer, s, l, batch_size);
    }
    
    rmsnorm_batch_getp(s->x, s->x, 
                  w->rms_attn_w, batch_size, hidden_dim);


    // Classifier into logits
    matmul_batch_getp(s->logits, s->x, w->out, batch_size, hidden_dim, p->vocab_size);

    return s->logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {

    // forward_getp the transformer to get logits for the next token
    float *logits = forward_getp(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens - 1] = next;
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens;
}

long long batched_generate(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  Config *p = &transformer->config;
  long long total_tokens_generated = 0;

  // Process requests in batches
  for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE) {
      int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs) ? 
                              (requests->num_reqs - req_start) : BATCH_SIZE;
      
      RunState *s = &transformer->state;

      for (int b = 0; b < current_batch_size; b++) {
          int req_idx = req_start + b;
          const char *input_seq = get_str_req_ptr(requests, req_idx);
          
          // Encode prompt
          encode(tokenizer, input_seq, 1, 0, prompt_tokens[b], 
                &prompt_lens[b], p->initial_context_length);
          
          if (prompt_lens[b] < 1) {
              fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
              prompt_lens[b] = 1;
              prompt_tokens[b][0] = 1; // BOS token
          }
          
          // Initialize sequence state
          positions[b] = 0;
          finished[b] = false;
          current_tokens[b] = prompt_tokens[b][0];
      }
      
      // Generation loop
      int max_steps = requests->max_seq_len;
      int alive = current_batch_size;
      for (int step = 0; step < max_steps && alive > 0; step++) {
          // Forward pass for active sequences
          float *logits = forward_batch_getp(transformer, s, 
                                            current_tokens, current_batch_size);
          
          // Sample next tokens
          for (int b = 0; b < current_batch_size; b++) {
              if (finished[b]) continue;
              
              int req_idx = req_start + b;
              int pos = positions[b];
              float *logits_b = logits + b * p->vocab_size;
              
              int next_token;
              if (pos < prompt_lens[b] - 1) {
                  // Still processing prompt
                  next_token = prompt_tokens[b][pos + 1];
              } else {
                  // Generate new token
                  next_token = sample(sampler, logits_b);
                  
                  // Save generated token
                  int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                  int gen_pos = pos - prompt_lens[b] + 1;
                  if (gen_pos >= 0) {
                      output_tokens[gen_pos] = next_token;
                      total_tokens_generated++;
                  }
              }
              
              // Check for termination
              if (next_token == 1) { // BOS token marks end
                  --alive;
                  finished[b] = true;
                  int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                  int gen_pos = pos - prompt_lens[b] + 2;
                  if (gen_pos >= 0) {
                      output_tokens[gen_pos] = -1; // End marker
                  }
                  continue;
              }
              
              // Update for next iteration
              positions[b]++;
              current_tokens[b] = next_token;
              
              // Print token (should be removed in production)
              if (pos >= prompt_lens[b] - 1) {
                  const char *piece = decode_piece(tokenizer, current_tokens[b], next_token);
                  printf("Batch %d: %s", b, piece);
                  fflush(stdout);
              }
          }
      }
      
      printf("\n"); // End of batch
  }
  
  return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
