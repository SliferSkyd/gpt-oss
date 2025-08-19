// TODO: Modify this file to optimize end-to-end throughput - BATCHING SUPPORT ADDED

#include "../tokenizer.hpp"
#include "getp_eval.cpp"

#ifndef GETP_RUN
#define GETP_RUN

// Forward declarations for single-sequence functions used in batch functions
void softmax_getp(float *x, int size);
void accumulate(float *a, float *b, float factor, int size);
void topk(float *topk_v, int *topk_i, float *router_score, int n_experts, int k);

float *cos_vals, *sin_vals;

// Batch state structure
typedef struct {
    float *x_batch;           // [batch_size, hidden_dim]
    float *t_batch;           // [batch_size, hidden_dim] 
    float *q_batch;           // [batch_size, n_attn_heads * head_dim]
    float *k_batch;           // [batch_size, n_kv_heads * head_dim]
    float *v_batch;           // [batch_size, n_kv_heads * head_dim]
    float *qkv_batch;         // [batch_size, (n_attn_heads + 2 * n_kv_heads) * head_dim]
    float *att_batch;         // [batch_size, n_attn_heads * seq_len]
    float *tb_batch;          // [batch_size, n_attn_heads * head_dim]
    float *tb2_batch;         // [batch_size, hidden_dim]
    float *logits_batch;      // [batch_size, vocab_size]
    
    // MoE specific
    float *router_score_batch; // [batch_size, n_experts]
    float *topk_v_batch;       // [batch_size, experts_per_token]
    int *topk_i_batch;         // [batch_size, experts_per_token]
    float *mlp1_out_batch;     // [batch_size, 2 * intermediate_dim]
    float *gate_batch;         // [batch_size, intermediate_dim]
    float *up_batch;           // [batch_size, intermediate_dim]
    float *gate_up_batch;      // [batch_size, intermediate_dim]
    float *e_agg_batch;        // [batch_size, hidden_dim]
    
    // KV cache - [batch_size, n_layers, seq_len, n_kv_heads * head_dim]
    float *key_cache_batch;
    float *value_cache_batch;
    
    // Mask for attention - [batch_size, seq_len, seq_len]
    float *mask_batch;
    
    // Sequence tracking
    int *positions;            // [batch_size] - current position for each sequence
    int *prompt_lens;          // [batch_size] - length of prompt for each sequence
    bool *finished;            // [batch_size] - whether sequence is finished
    
} BatchState;

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

BatchState* alloc_batch_state(Config *p, int batch_size) {
    BatchState *batch_state = (BatchState*)malloc(sizeof(BatchState));
    
    int hidden_dim = p->hidden_dim;
    int head_dim = p->head_dim;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;
    int kv_dim = p->head_dim * p->n_kv_heads;
    
    // Allocate all batch tensors
    batch_state->x_batch = (float*)calloc(batch_size * hidden_dim, sizeof(float));
    batch_state->t_batch = (float*)calloc(batch_size * hidden_dim, sizeof(float));
    batch_state->q_batch = (float*)calloc(batch_size * p->n_attn_heads * head_dim, sizeof(float));
    batch_state->k_batch = (float*)calloc(batch_size * p->n_kv_heads * head_dim, sizeof(float));
    batch_state->v_batch = (float*)calloc(batch_size * p->n_kv_heads * head_dim, sizeof(float));
    batch_state->qkv_batch = (float*)calloc(batch_size * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim, sizeof(float));
    batch_state->att_batch = (float*)calloc(batch_size * p->n_attn_heads * p->seq_len, sizeof(float));
    batch_state->tb_batch = (float*)calloc(batch_size * p->n_attn_heads * head_dim, sizeof(float));
    batch_state->tb2_batch = (float*)calloc(batch_size * hidden_dim, sizeof(float));
    batch_state->logits_batch = (float*)calloc(batch_size * p->vocab_size, sizeof(float));
    
    // MoE specific
    batch_state->router_score_batch = (float*)calloc(batch_size * n_experts, sizeof(float));
    batch_state->topk_v_batch = (float*)calloc(batch_size * p->experts_per_token, sizeof(float));
    batch_state->topk_i_batch = (int*)calloc(batch_size * p->experts_per_token, sizeof(int));
    batch_state->mlp1_out_batch = (float*)calloc(batch_size * 2 * intermediate_dim, sizeof(float));
    batch_state->gate_batch = (float*)calloc(batch_size * intermediate_dim, sizeof(float));
    batch_state->up_batch = (float*)calloc(batch_size * intermediate_dim, sizeof(float));
    batch_state->gate_up_batch = (float*)calloc(batch_size * intermediate_dim, sizeof(float));
    batch_state->e_agg_batch = (float*)calloc(batch_size * hidden_dim, sizeof(float));
    
    // KV cache
    batch_state->key_cache_batch = (float*)calloc(batch_size * p->n_layers * p->seq_len * kv_dim, sizeof(float));
    batch_state->value_cache_batch = (float*)calloc(batch_size * p->n_layers * p->seq_len * kv_dim, sizeof(float));
    
    // Mask
    batch_state->mask_batch = (float*)calloc(batch_size * p->seq_len * p->seq_len, sizeof(float));
    
    // Sequence tracking
    batch_state->positions = (int*)calloc(batch_size, sizeof(int));
    batch_state->prompt_lens = (int*)calloc(batch_size, sizeof(int));
    batch_state->finished = (bool*)calloc(batch_size, sizeof(bool));
    
    return batch_state;
}

void free_batch_state(BatchState *batch_state) {
    free(batch_state->x_batch);
    free(batch_state->t_batch);
    free(batch_state->q_batch);
    free(batch_state->k_batch);
    free(batch_state->v_batch);
    free(batch_state->qkv_batch);
    free(batch_state->att_batch);
    free(batch_state->tb_batch);
    free(batch_state->tb2_batch);
    free(batch_state->logits_batch);
    free(batch_state->router_score_batch);
    free(batch_state->topk_v_batch);
    free(batch_state->topk_i_batch);
    free(batch_state->mlp1_out_batch);
    free(batch_state->gate_batch);
    free(batch_state->up_batch);
    free(batch_state->gate_up_batch);
    free(batch_state->e_agg_batch);
    free(batch_state->key_cache_batch);
    free(batch_state->value_cache_batch);
    free(batch_state->mask_batch);
    free(batch_state->positions);
    free(batch_state->prompt_lens);
    free(batch_state->finished);
    free(batch_state);
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
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  
  free(cos_vals);
  free(sin_vals);
}

// Batched operations
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

void softmax_batch_getp(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        float *x_b = x + b * size;
        
        // find max value (for numerical stability)
        double max_val = x_b[0];
        for (int i = 1; i < size; i++) {
            if (x_b[i] > max_val) {
                max_val = x_b[i];
            }
        }
        // exp and sum
        double sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x_b[i] = expf(x_b[i] - max_val);
            sum += x_b[i];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x_b[i] /= sum;
        }
    }
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

void sdpa_batch_getp(Transformer *transformer, BatchState *batch_state, 
                    unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int head_dim = p->head_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;

    // Process each batch item
    for (int b = 0; b < batch_size; b++) {
        int pos = batch_state->positions[b];
        int loff = l * p->seq_len * kv_dim;
        
        float *q_b = batch_state->q_batch + b * p->n_attn_heads * head_dim;
        float *att_b = batch_state->att_batch + b * p->n_attn_heads * p->seq_len;
        float *tb_b = batch_state->tb_batch + b * p->n_attn_heads * head_dim;
        float *key_cache_b = batch_state->key_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = batch_state->value_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        float *mask_b = batch_state->mask_batch + b * p->seq_len * p->seq_len;

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

void attention_batch_getp(Transformer *transformer, BatchState *batch_state,
                         unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;

    // RMSNorm for all batch items
    rmsnorm_batch_getp(batch_state->t_batch, batch_state->x_batch, 
                      w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

    // Update KV cache pointers for current positions
    for (int b = 0; b < batch_size; b++) {
        int pos = batch_state->positions[b];
        int loff = l * p->seq_len * kv_dim;
        float *key_cache_b = batch_state->key_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = batch_state->value_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        
        float *k_b = batch_state->k_batch + b * p->n_kv_heads * head_dim;
        float *v_b = batch_state->v_batch + b * p->n_kv_heads * head_dim;
        
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
    
    matmul_batch_getp(batch_state->qkv_batch, batch_state->t_batch, w_qkv, batch_size,
                     hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    
    // Add bias and separate Q, K, V for all batch items
    for (int b = 0; b < batch_size; b++) {
        float *qkv_b = batch_state->qkv_batch + b * (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim;
        float *q_b = batch_state->q_batch + b * p->n_attn_heads * head_dim;
        float *k_b = batch_state->k_batch + b * p->n_kv_heads * head_dim;
        float *v_b = batch_state->v_batch + b * p->n_kv_heads * head_dim;
        
        // Add bias
        for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; i++) {
            qkv_b[i] += b_qkv[i];
        }
        
        // Separate Q, K, V
        memcpy(q_b, qkv_b, head_dim * p->n_attn_heads * sizeof(float));
        memcpy(k_b, qkv_b + head_dim * p->n_attn_heads, head_dim * p->n_kv_heads * sizeof(float));
        memcpy(v_b, qkv_b + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads, 
               head_dim * p->n_kv_heads * sizeof(float));
    }

    // Apply rotary embeddings
    apply_rotary_emb_batch_getp(batch_state->q_batch, cos_vals, sin_vals, batch_size, 
                               batch_state->positions, p->n_attn_heads, head_dim);
    apply_rotary_emb_batch_getp(batch_state->k_batch, cos_vals, sin_vals, batch_size,
                               batch_state->positions, p->n_kv_heads, head_dim);

    // Update KV cache with rotary-embedded values
    for (int b = 0; b < batch_size; b++) {
        int pos = batch_state->positions[b];
        int loff = l * p->seq_len * kv_dim;
        float *key_cache_b = batch_state->key_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        float *value_cache_b = batch_state->value_cache_batch + b * p->n_layers * p->seq_len * kv_dim;
        
        float *k_b = batch_state->k_batch + b * p->n_kv_heads * head_dim;
        float *v_b = batch_state->v_batch + b * p->n_kv_heads * head_dim;
        
        memcpy(key_cache_b + loff + pos * kv_dim, k_b, kv_dim * sizeof(float));
        memcpy(value_cache_b + loff + pos * kv_dim, v_b, kv_dim * sizeof(float));
    }

    // Scaled dot-product attention
    sdpa_batch_getp(transformer, batch_state, l, batch_size);
    
    // Output projection
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;
    
    matmul_batch_getp(batch_state->tb2_batch, batch_state->tb_batch, w_o, batch_size,
                     head_dim * p->n_attn_heads, hidden_dim);
    
    // Add bias and residual connection
    for (int b = 0; b < batch_size; b++) {
        float *x_b = batch_state->x_batch + b * hidden_dim;
        float *tb2_b = batch_state->tb2_batch + b * hidden_dim;
        
        for (int i = 0; i < hidden_dim; i++) {
            tb2_b[i] += b_o[i];
            x_b[i] += tb2_b[i];  // residual connection
        }
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

void mlp_batch_getp(Transformer *transformer, BatchState *batch_state, 
                   float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2, 
                   float *expert_weights, int batch_size) {
    Config *p = &transformer->config;
    
    int hidden_dim = p->hidden_dim;
    int intermediate_dim = p->intermediate_dim;

    // First linear layer
    matmul_batch_getp(batch_state->mlp1_out_batch, batch_state->t_batch, w_mlp1, 
                     batch_size, hidden_dim, 2 * intermediate_dim);
    
    // Add bias and split into gate and up
    for (int b = 0; b < batch_size; b++) {
        float *mlp1_out_b = batch_state->mlp1_out_batch + b * 2 * intermediate_dim;
        float *gate_b = batch_state->gate_batch + b * intermediate_dim;
        float *up_b = batch_state->up_batch + b * intermediate_dim;
        
        for (int j = 0; j < intermediate_dim; j++) {
            gate_b[j] = mlp1_out_b[2 * j] + b_mlp1[2 * j];
            up_b[j] = mlp1_out_b[2 * j + 1] + b_mlp1[2 * j + 1];
        }
    }

    // Apply SwiGLU activation
    swiglu_batch_getp(batch_state->gate_batch, batch_state->up_batch, 
                     batch_state->gate_up_batch, batch_size, intermediate_dim, p->swiglu_limit);

    // Second linear layer
    matmul_batch_getp(batch_state->tb2_batch, batch_state->gate_up_batch, w_mlp2, 
                     batch_size, intermediate_dim, hidden_dim);
    
    // Add bias and accumulate with expert weights
    for (int b = 0; b < batch_size; b++) {
        float *tb2_b = batch_state->tb2_batch + b * hidden_dim;
        float *e_agg_b = batch_state->e_agg_batch + b * hidden_dim;
        float expert_w = expert_weights[b];
        
        for (int i = 0; i < hidden_dim; i++) {
            tb2_b[i] += b_mlp2[i];
            e_agg_b[i] += tb2_b[i] * expert_w;
        }
    }
}

void moe_batch_getp(Transformer *transformer, BatchState *batch_state,
                   unsigned long long l, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int hidden_dim = p->hidden_dim;
    int n_experts = p->n_experts;

    // FFN RMSNorm
    rmsnorm_batch_getp(batch_state->t_batch, batch_state->x_batch, 
                      w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

    // Compute router scores for all batch items
    float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    float *b_router = w->b_router + 1ll * l * n_experts;
    
    matmul_batch_getp(batch_state->router_score_batch, batch_state->t_batch, w_router, 
                     batch_size, hidden_dim, n_experts);
    
    // Add bias and select top-k experts for each batch item
    for (int b = 0; b < batch_size; b++) {
        float *router_score_b = batch_state->router_score_batch + b * n_experts;
        float *topk_v_b = batch_state->topk_v_batch + b * p->experts_per_token;
        int *topk_i_b = batch_state->topk_i_batch + b * p->experts_per_token;
        
        // Add bias
        for (int i = 0; i < n_experts; i++) {
            router_score_b[i] += b_router[i];
        }
        
        // Select top-k experts
        topk(topk_v_b, topk_i_b, router_score_b, n_experts, p->experts_per_token);
        
        // Normalize selected experts - use single sequence function for single batch item
        softmax_getp(topk_v_b, p->experts_per_token);
    }

    // Initialize expert aggregation buffers
    memset(batch_state->e_agg_batch, 0, batch_size * hidden_dim * sizeof(float));
    
    // Route tokens to their corresponding top-k experts
    for (int expert_idx = 0; expert_idx < p->experts_per_token; expert_idx++) {
        // Group batch items by expert for better cache efficiency
        for (int expert_id = 0; expert_id < n_experts; expert_id++) {
            // Find all batch items that use this expert at this expert_idx
            float expert_weights[batch_size];
            int batch_indices[batch_size];
            int batch_count = 0;
            
            for (int b = 0; b < batch_size; b++) {
                int *topk_i_b = batch_state->topk_i_batch + b * p->experts_per_token;
                float *topk_v_b = batch_state->topk_v_batch + b * p->experts_per_token;
                
                if (topk_i_b[expert_idx] == expert_id) {
                    batch_indices[batch_count] = b;
                    expert_weights[batch_count] = topk_v_b[expert_idx];
                    batch_count++;
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
            BatchState temp_batch_state = *batch_state;
            
            // Copy input data for selected batch items to beginning of buffers
            for (int i = 0; i < batch_count; i++) {
                int src_b = batch_indices[i];
                memcpy(temp_batch_state.t_batch + i * hidden_dim,
                       batch_state->t_batch + src_b * hidden_dim,
                       hidden_dim * sizeof(float));
            }
            
            // Process this expert with the selected batch items
            mlp_batch_getp(transformer, &temp_batch_state, w_mlp1, b_mlp1, w_mlp2, b_mlp2,
                          expert_weights, batch_count);
            
            // Copy results back to original positions in e_agg_batch
            for (int i = 0; i < batch_count; i++) {
                int dst_b = batch_indices[i];
                float *src = temp_batch_state.e_agg_batch + i * hidden_dim;
                float *dst = batch_state->e_agg_batch + dst_b * hidden_dim;
                
                for (int j = 0; j < hidden_dim; j++) {
                    dst[j] += src[j];  // Accumulate from multiple experts
                }
            }
        }
    }

    // Residual connection
    for (int b = 0; b < batch_size; b++) {
        float *x_b = batch_state->x_batch + b * hidden_dim;
        float *e_agg_b = batch_state->e_agg_batch + b * hidden_dim;
        
        for (int i = 0; i < hidden_dim; i++) {
            x_b[i] += e_agg_b[i];
        }
    }
}

float *forward_batch_getp(Transformer *transformer, BatchState *batch_state,
                         int *tokens, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int hidden_dim = p->hidden_dim;

    // Copy token embeddings into x_batch
    for (int b = 0; b < batch_size; b++) {
        float *x_b = batch_state->x_batch + b * hidden_dim;
        float *content_row = w->token_embedding_table + tokens[b] * hidden_dim;
        memcpy(x_b, content_row, hidden_dim * sizeof(float));
    }

    // Forward through all layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_batch_getp(transformer, batch_state, l, batch_size);
        moe_batch_getp(transformer, batch_state, l, batch_size);
    }
    
    // Final RMSNorm
    rmsnorm_batch_getp(batch_state->x_batch, batch_state->x_batch, w->rms_out_w, 
                      batch_size, hidden_dim);

    // Classifier into logits
    matmul_batch_getp(batch_state->logits_batch, batch_state->x_batch, w->out, 
                     batch_size, hidden_dim, p->vocab_size);
    
    return batch_state->logits_batch;
}

long long batched_generate(Transformer *transformer, Tokenizer *tokenizer,
                          Sampler *sampler, Requests *requests, int batch_size) {
    Config *p = &transformer->config;
    long long total_tokens_generated = 0;
    
    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += batch_size) {
        int current_batch_size = (req_start + batch_size > requests->num_reqs) ? 
                                (requests->num_reqs - req_start) : batch_size;
        
        // Allocate batch state
        BatchState *batch_state = alloc_batch_state(p, current_batch_size);
        
        // Initialize batch with prompts
        int **prompt_tokens = (int**)malloc(current_batch_size * sizeof(int*));
        int *current_tokens = (int*)malloc(current_batch_size * sizeof(int));
        
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            
            // Encode prompt
            prompt_tokens[b] = (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
            encode(tokenizer, input_seq, 1, 0, prompt_tokens[b], 
                  &batch_state->prompt_lens[b], p->initial_context_length);
            
            if (batch_state->prompt_lens[b] < 1) {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                batch_state->prompt_lens[b] = 1;
                prompt_tokens[b][0] = 1; // BOS token
            }
            
            // Initialize sequence state
            batch_state->positions[b] = 0;
            batch_state->finished[b] = false;
            current_tokens[b] = prompt_tokens[b][0];
        }
        
        // Generation loop
        int max_steps = requests->max_seq_len;
        for (int step = 0; step < max_steps; step++) {
            // Check if all sequences are finished
            bool all_finished = true;
            for (int b = 0; b < current_batch_size; b++) {
                if (!batch_state->finished[b]) {
                    all_finished = false;
                    break;
                }
            }
            if (all_finished) break;
            
            // Forward pass for active sequences
            float *logits = forward_batch_getp(transformer, batch_state, 
                                             current_tokens, current_batch_size);
            
            // Sample next tokens
            for (int b = 0; b < current_batch_size; b++) {
                if (batch_state->finished[b]) continue;
                
                int req_idx = req_start + b;
                int pos = batch_state->positions[b];
                float *logits_b = logits + b * p->vocab_size;
                
                int next_token;
                if (pos < batch_state->prompt_lens[b] - 1) {
                    // Still processing prompt
                    next_token = prompt_tokens[b][pos + 1];
                } else {
                    // Generate new token
                    next_token = sample(sampler, logits_b);
                    
                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - batch_state->prompt_lens[b] + 1;
                    if (gen_pos >= 0) {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }
                
                // Check for termination
                if (next_token == 1) { // BOS token marks end
                    batch_state->finished[b] = true;
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - batch_state->prompt_lens[b] + 2;
                    if (gen_pos >= 0) {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }
                
                // Update for next iteration
                batch_state->positions[b]++;
                current_tokens[b] = next_token;
                
                // Print token (should be removed in production)
                if (pos >= batch_state->prompt_lens[b] - 1) {
                    const char *piece = decode_piece(tokenizer, current_tokens[b], next_token);
                    printf("Batch %d: %s", b, piece);
                    fflush(stdout);
                }
            }
        }
        
        // Cleanup
        for (int b = 0; b < current_batch_size; b++) {
            free(prompt_tokens[b]);
        }
        free(prompt_tokens);
        free(current_tokens);
        free_batch_state(batch_state);
        
        printf("\n"); // End of batch
    }
    
    return total_tokens_generated;
}

// Legacy single-sequence functions (kept for compatibility)
void rmsnorm_getp(float *o, float *x, float *weight, int size) {
    double ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax_getp(float *x, int size) {
    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    double sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul_getp(float *xout, float *x, float *w, int n, int d) {
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

void apply_rotary_emb_getp(float *x, float *cos, float *sin, int n_heads, int head_dim) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            float x1 = x[h * head_dim + i];
            float x2 = x[h * head_dim + half + i];
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
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int loff = l * p->seq_len * kv_dim;

    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_attn_heads; h++) {
        float *q = s->q + h * head_dim;
        float *att = s->att + h * p->seq_len;
        
        for (int t = 0; t <= pos; t++) {
            float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
            double score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_dim);
            if (p->sliding_window > 0 && (l % 2 == 0)) {
                score += s->mask[pos * p->seq_len + t];
            }
            att[t] = score;
        }
        
        att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
        softmax_getp(att, pos + 2);

        float *tb = s->tb + h * head_dim;
        memset(tb, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
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

    rmsnorm_getp(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

    int loff = l * p->seq_len * kv_dim;
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    float *b_qkv = w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    
    matmul_getp(s->qkv, s->t, w_qkv, hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    accumulate(s->qkv, b_qkv, 1.0f, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    
    memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float));
    memcpy(s->k, s->qkv + head_dim * p->n_attn_heads, head_dim * p->n_kv_heads * sizeof(float));
    memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads, head_dim * p->n_kv_heads * sizeof(float));

    apply_rotary_emb_getp(s->q, cos_vals + pos * (head_dim / 2), sin_vals + pos * (head_dim / 2), p->n_attn_heads, head_dim);
    apply_rotary_emb_getp(s->k, cos_vals + pos * (head_dim / 2), sin_vals + pos * (head_dim / 2), p->n_kv_heads, head_dim);

    sdpa_getp(transformer, l, pos);
    
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;
    matmul_getp(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    accumulate(s->tb2, b_o, 1.0f, hidden_dim);
    accumulate(x, s->tb2, 1.0f, hidden_dim);
}

void swiglu_getp(float *x, float *gate, float *up, float *gate_up, int intermediate_dim, float swiglu_limit) {
    const float alpha = 1.702f;
    for (int i = 0; i < intermediate_dim; i++) {
        float val = gate[i];
        float up_val = up[i];
        if (val > swiglu_limit) val = swiglu_limit;
        if (up_val > swiglu_limit) up_val = swiglu_limit;
        if (up_val < -swiglu_limit) up_val = -swiglu_limit;
        val *= (1.0f / (1.0f + expf(-alpha * val)));
        val *= (up_val + 1.0f);
        gate_up[i] = val;
    }
}

void mlp_getp(Transformer *transformer, float *x, float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2, float expert_w) {
    Config *p = &transformer->config;
    RunState *s = &transformer->state;

    matmul_getp(s->mlp1_out, s->t, w_mlp1, p->hidden_dim, 2 * p->intermediate_dim);
    accumulate(s->mlp1_out, b_mlp1, 1.0f, 2 * p->intermediate_dim);
    
    for (int j = 0; j < p->intermediate_dim; j++) {
        s->gate[j] = s->mlp1_out[2 * j];
        s->up[j] = s->mlp1_out[2 * j + 1];
    }

    swiglu_getp(x, s->gate, s->up, s->gate_up, p->intermediate_dim, p->swiglu_limit);
    matmul_getp(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim, p->hidden_dim);
    accumulate(s->tb2, b_mlp2, 1.0f, p->hidden_dim);
    accumulate(s->e_agg, s->tb2, expert_w, p->hidden_dim);
}

void moe_getp(Transformer *transformer, float *x, unsigned long long l, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;

    rmsnorm_getp(s->t, x, w->rms_ffn_w + 1ll * l * p->hidden_dim, p->hidden_dim);

    float *w_router = w->w_router + 1ll * l * p->hidden_dim * p->n_experts;
    float *b_router = w->b_router + 1ll * l * p->n_experts;
    matmul_getp(s->router_score, s->t, w_router, p->hidden_dim, p->n_experts);
    
    for (int i = 0; i < p->n_experts; i++) {
        s->router_score[i] += b_router[i];
    }
    
    topk(s->topk_v, s->topk_i, s->router_score, p->n_experts, p->experts_per_token);
    softmax_getp(s->topk_v, p->experts_per_token);

    memset(s->e_agg, 0, p->hidden_dim * sizeof(float));
    
    for (int idx = 0; idx < p->experts_per_token; idx++) {
        int e = s->topk_i[idx];
        float expert_w = s->topk_v[idx];
        
        float *w_mlp1 = w->w_mlp1 + 1ll * (l * p->n_experts + e) * (2 * p->intermediate_dim) * p->hidden_dim;
        float *b_mlp1 = w->b_mlp1 + 1ll * (l * p->n_experts + e) * (2 * p->intermediate_dim);
        float *w_mlp2 = w->w_mlp2 + 1ll * (l * p->n_experts + e) * p->hidden_dim * p->intermediate_dim;
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * p->n_experts + e) * p->hidden_dim;

        mlp_getp(transformer, x, w_mlp1, b_mlp1, w_mlp2, b_mlp2, expert_w);
    }

    accumulate(x, s->e_agg, 1.0f, p->hidden_dim);
}

float *forward_getp(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;

    float *x = s->x;
    int hidden_dim = p->hidden_dim;

    float *content_row = w->token_embedding_table + token * hidden_dim;
    memcpy(x, content_row, hidden_dim * sizeof(*x));

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_getp(transformer, x, l, pos);
        moe_getp(transformer, x, l, pos);
    }
    
    rmsnorm_getp(x, x, w->rms_out_w, hidden_dim);
    matmul_getp(s->logits, x, w->out, hidden_dim, p->vocab_size);
    return s->logits;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
    const char *empty_prompt = "";
    if (input_seq == NULL) {
        input_seq = empty_prompt;
    }

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) * sizeof(int));
    encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
           transformer->config.initial_context_length);
    
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    while (pos < steps) {
        float *logits = forward_getp(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
            output_tokens[pos - num_prompt_tokens - 1] = next;
        }
        pos++;

        if (next == 1) {
            break;
        }

        const char *piece = decode_piece(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
    }

    printf("\n");
    output_tokens[pos - num_prompt_tokens] = -1;
    free(prompt_tokens);

    return pos - num_prompt_tokens;
}

// Main inference function with batching support
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
    // Determine optimal batch size based on available memory and model size
    int optimal_batch_size = 8; // Can be made configurable
    
    // Use batched generation for better throughput
    return batched_generate(transformer, tokenizer, sampler, requests, optimal_batch_size);
}

#endif // GETP_RUN