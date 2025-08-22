# Model Spec Overview
![Overall Architecture](https://substackcdn.com/image/fetch/$s_!PKaP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe804b20e-7196-4529-9ca1-13a946123c7c_1589x734.png)

**Core Architecture:** Decoder-only Transformer, pre-norm (RMSNorm)
- **Embedding dim:** 2,880 | **Heads:** 64 (head dim = 45)
- **Attention:** GQA; alternating between full causal attention and sliding-window (128) every 2nd layer
- **FFN:** SwiGLU MoE | **Positional:** RoPE | **Vocab:** 200k | **Context:** 131k tokens

**Model Variants:**
- **20B:** 24 blocks, 32 experts/block, ~3.6B active params
- **120B:** 36 blocks, 128 experts/block, ~5.1B active params

## Forward Pass
1. Tokenize → Embedding (200k × 2,880 = 576M params)
2. Transformer blocks × N: RMSNorm → GQA → residual → RMSNorm → MoE → residual
3. Final RMSNorm → Linear output → 200k logits

## Layer Details

### 1. Token Embedding
- **I/O:** (batch, seq_len) → (batch, seq_len, 2880)
- **Params:** 576M (embedding table), +576M if output untied

**Implementation:**
```cpp
// run.cpp:543-545
// copy the token embedding into x
float *content_row = w->token_embedding_table + token * hidden_dim;
memcpy(x, content_row, hidden_dim * sizeof(*x));
```

**Weight Structure:**
```cpp
// run.cpp:48-49
// token_embedding_table - embedding.weight
float *token_embedding_table; // (vocab_size, hidden_dim) (in, out)
```

### 2. RoPE
![Rotary Positional Embedding](https://substackcdn.com/image/fetch/$s_!YCov!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62fdfc32-605c-4065-abc6-689756d53b87_1195x533.png)

- **I/O:** (batch, seq_len, 2880) unchanged
- **Params:** 0 (functional rotation on Q,K)

**Implementation:**
```cpp
// run.cpp:506-526 - Apply RoPE rotation
void apply_rotary_emb(float *x, float *cos, float *sin, int n_heads, int head_dim) {
  int half = head_dim / 2;
  for (int h = 0; h < n_heads; h++) {
    for (int i = 0; i < half; i++) {
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

// run.cpp:579-592 - Usage in forward pass
compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);
```

### 3. RMSNorm
![RMSNorm](https://substackcdn.com/image/fetch/$s_!H32R!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4ac713f9-4f15-4104-9b67-eff1c4f29f95_1367x599.png)

- **Usage:** Pre-attention, pre-MoE, final output
- **Params:** 2,880 scalars per norm
  - 20B: 141,120 total | 120B: 210,240 total

**Implementation:**
```cpp
// run.cpp:320-333 - RMSNorm function
void rmsnorm(float *o, float *x, float *weight, int size) {
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

// Usage in forward pass:
// run.cpp:550 - Pre-attention norm
rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);
// run.cpp:658 - Pre-MoE norm  
rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);
// run.cpp:751 - Final output norm
rmsnorm(x, x, w->rms_out_w, hidden_dim);
```

### 4. GQA - Full Context
![Group query attention](https://substackcdn.com/image/fetch/$s_!Kohq!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa2347cc2-3685-4547-b31b-be7bbfb21201_1237x588.png)

- **I/O:** (batch, seq_len, 2880) unchanged
- **Config:** 64 Q heads, shared K/V across groups, RoPE on Q/K
- **Features:** Bias terms, attention sinks (learned per-head logits)
- **Params:** <33.18M per block (GQA reduces K/V sharing)

**Implementation:**
```cpp
// run.cpp:557-576 - QKV projection
float *w_qkv = w->w_qkv + 1ll * l * hidden_dim * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
float *b_qkv = w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
matmul(s->qkv, s->t, w_qkv, hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
// add bias
for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
  s->qkv[i] += b_qkv[i];
}
// Separate q, k, v
memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float));
memcpy(s->k, s->qkv + head_dim * p->n_attn_heads, head_dim * p->n_kv_heads * sizeof(float));
memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads, head_dim * p->n_kv_heads * sizeof(float));

// run.cpp:597-642 - Multi-head attention computation
for (h = 0; h < p->n_attn_heads; h++) {
  float *q = s->q + h * head_dim;
  float *att = s->att + h * p->seq_len;
  // iterate over all timesteps
  for (int t = 0; t <= pos; t++) {
    // GQA: get key vector for this head
    float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    // attention score as dot product of q and k
    double score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      score += q[i] * k[i];
    }
    score /= sqrtf(head_dim);
    // Apply sliding window mask if enabled
    if (p->sliding_window > 0 && (l % 2 == 0)) {
      score += s->mask[pos * p->seq_len + t];
    }
    att[t] = score;
  }
  // Add attention sink score
  att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
  softmax(att, pos + 2);
  // weighted sum of values
  float *tb = s->tb + h * head_dim;
  memset(tb, 0, head_dim * sizeof(float));
  for (int t = 0; t <= pos; t++) {
    float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    float a = att[t];
    for (int i = 0; i < head_dim; i++) {
      tb[i] += a * v[i];
    }
  }
}
```

### 5. GQA - Sliding Window
![Grouped Query Attention](https://substackcdn.com/image/fetch/$s_!wwFe!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe79d48c3-2bdf-41bb-9e18-45f128cd5c01_1600x792.jpeg)

- **Pattern:** Every 2nd layer uses 128-token window
- **Benefit:** Reduces KV cache bandwidth with minimal quality loss

**Implementation:**
```cpp
// run.cpp:169-176 - Initialize sliding window mask
for (int i = 0; i < p->seq_len; i++) {
  for (int j = 0; j < p->seq_len; j++) {
    if (p->sliding_window > 0 && i - j >= p->sliding_window) {
      s->mask[i * p->seq_len + j] = -INFINITY; // Sliding window mask
    }
  }
}

// run.cpp:617-619 - Apply mask during attention (every 2nd layer)
if (p->sliding_window > 0 && (l % 2 == 0)) {
  score += s->mask[pos * p->seq_len + t];
}
```

### 6. MoE Feed-Forward
![MoE forward](https://substackcdn.com/image/fetch/$s_!SYqb!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa3367a4-914a-49e0-8c94-016969397ab3_1307x640.png)
![Moe FeedFowrard](https://substackcdn.com/image/fetch/$s_!T1lR!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6cc115d2-4f0a-4d54-8edd-d737ea4dc221_1044x779.png)

- **Experts:** 32 (20B) / 128 (120B), top-4 active per token
- **Router:** 2,880 × num_experts linear layer
- **SwiGLU dims:** input=2,880, intermediate=2,880 (3 linears: W,V,U + SiLU)
- **Params per expert:** 24.88M (3 × 2,880²)
- **Active MoE params:** 99.53M per block, 2.39B (20B) / 3.58B (120B) total

**Implementation:**
```cpp
// run.cpp:661-675 - Router computation and expert selection
float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
float *b_router = w->b_router + 1ll * l * n_experts;
matmul(s->router_score, s->t, w_router, hidden_dim, n_experts);
// add bias b_router
for (int i = 0; i < n_experts; i++) {
  s->router_score[i] += b_router[i];
}
// Select top-k experts
topk(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
// Normalize selected experts using softmax
softmax(s->topk_v, p->experts_per_token);

// run.cpp:676-743 - Expert processing loop
memset(s->e_agg, 0, hidden_dim * sizeof(float));
for (int e = 0; e < n_experts; e++) {
  float expert_w = 0;
  int in_topk = 0;
  // Check if expert e is in top-k experts
  for (int idx = 0; idx < p->experts_per_token; idx++) {
    if (s->topk_i[idx] == e) {
      in_topk = 1;
      expert_w = s->topk_v[idx];
      break;
    }
  }
  
  if (in_topk) {
    // Gate-Up projection (SwiGLU first part)
    float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
    float *b_mlp1 = w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
    matmul(s->mlp1_out, s->t, w_mlp1, hidden_dim, 2 * p->intermediate_dim);
    for (int i = 0; i < 2 * p->intermediate_dim; i++) {
      s->mlp1_out[i] += b_mlp1[i];
    }
    // Split mlp1_out into gate and up
    for (int j = 0; j < p->intermediate_dim; j++) {
      s->gate[j] = s->mlp1_out[2 * j];
      s->up[j] = s->mlp1_out[2 * j + 1];
    }
    
    // SwiGLU non-linearity
    const float alpha = 1.702f;
    for (int i = 0; i < p->intermediate_dim; i++) {
      float val = s->gate[i];
      float up_val = s->up[i];
      // Clamping
      if (val > p->swiglu_limit) val = p->swiglu_limit;
      if (up_val > p->swiglu_limit) up_val = p->swiglu_limit;
      if (up_val < -p->swiglu_limit) up_val = -p->swiglu_limit;
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-alpha * val)));
      // elementwise multiply with w_gate(x)
      val *= (up_val + 1.0f); // gpt-oss adds an extra bias of 1 to the up layer
      s->gate_up[i] = val;
    }
    
    // Down projection
    float *w_mlp2 = w->w_mlp2 + 1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
    float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
    matmul(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim, hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_mlp2[i];
    }
    
    // aggregate topk experts using weighted sum
    for (int i = 0; i < hidden_dim; i++) {
      s->e_agg[i] += s->tb2[i] * expert_w;
    }
  }
}
```

### 7. Output Projection
- **I/O:** (batch, seq_len, 2880) → (batch, seq_len, 200k)
- **Params:** 576M (untied) or 0 (tied with embedding)

**Implementation:**
```cpp
// run.cpp:753-754 - Final output projection  
// classifier into logits
matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);

// Weight structure (run.cpp:81-82):
// classifier weights for the logits [unembedding.weight]
float *out; // (vocab_size, hidden_dim) (out, in)
```

## Block Structure
```
x = RMSNorm(x)
x = x + Attention(x)  # GQA full (odd) / sliding-window-128 (even)
x = RMSNorm(x)  
x = x + MoE_SwiGLU(x)  # Router top-4 selection
```
Final: RMSNorm → Linear output

**Full Forward Pass Implementation:**
```cpp
// run.cpp:528-756 - Complete forward function
float *forward(Transformer *transformer, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  
  // 1. Token embedding
  float *content_row = w->token_embedding_table + token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));

  // 2. Forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // Pre-attention RMSNorm
    rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);
    
    // Attention block (QKV projection, RoPE, multi-head attention)
    // ... [attention implementation from section 4] ...
    
    // Attention output projection + residual
    matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_o[i];
    }
    // residual connection
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->tb2[i];
    }

    // Pre-MoE RMSNorm
    rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);
    
    // MoE block (router + expert selection + SwiGLU)
    // ... [MoE implementation from section 6] ...
    
    // MoE residual connection
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->e_agg[i];
    }
  }
  
  // 3. Final RMSNorm
  rmsnorm(x, x, w->rms_out_w, hidden_dim);
  
  // 4. Output projection
  matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
  return s->logits;
}
```

## Scaling Configuration
**Constant:** d_model=2,880, heads=64, RoPE, RMSNorm, GQA, sliding-window-128

**Variable:**
- **20B:** 24 blocks, 32 experts → 3.6B active/step
- **120B:** 36 blocks, 128 experts → 5.1B active/step