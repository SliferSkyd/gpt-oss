# README-getp_run.md

## Tổng quan
File `getp_run.cpp` chứa các hàm tối ưu hóa cho việc inference của mô hình GPT-OSS. Đây là phiên bản được tối ưu hóa để cải thiện throughput end-to-end so với implementation gốc trong `run.cpp`.

## Định nghĩa struct từ run.cpp

### Config
```cpp
typedef struct {
  int vocab_size;              // Kích thước từ vựng
  int hidden_dim;              // Số chiều ẩn của mô hình
  int n_experts;               // Số lượng expert trong MoE
  int experts_per_token;       // Số expert được chọn cho mỗi token (top-k)
  int intermediate_dim;        // Số chiều trung gian trong FFN
  int n_layers;                // Số layer của transformer
  int head_dim;                // Số chiều mỗi attention head
  int n_attn_heads;            // Số attention head
  int n_kv_heads;              // Số key/value head (có thể < n_attn_heads do MQA)
  int seq_len;                 // Độ dài sequence tối đa
  int initial_context_length;  // Độ dài context ban đầu
  float rope_theta;            // Tham số theta cho RoPE
  float rope_scaling_factor;   // Hệ số scale cho RoPE
  int sliding_window;          // Kích thước sliding window attention
  float swiglu_limit;          // Giới hạn cho activation SwiGLU
} Config;
```

### TransformerWeights
Chứa tất cả weights của mô hình:
- `token_embedding_table`: Embedding table cho tokens
- `rms_attn_w`, `rms_ffn_w`, `rms_out_w`: Weights cho RMSNorm
- `w_qkv`, `b_qkv`: Weights và bias cho projection QKV
- `w_o`, `b_o`: Weights và bias cho output projection
- `attn_sinks`: Attention sink scores
- `w_router`, `b_router`: Weights và bias cho MoE router
- `w_mlp1`, `b_mlp1`, `w_mlp2`, `b_mlp2`: Weights và bias cho MLP layers
- `out`: Classifier weights

### RunState
Chứa các buffer tạm thời trong quá trình forward pass:
- `x`, `t`, `tb`, `tb2`: Activation buffers
- `router_score`, `topk_v`, `topk_i`: Buffers cho MoE routing
- `mlp1_out`, `gate`, `up`, `gate_up`, `e_agg`: Buffers cho MLP computation
- `qkv`, `q`, `k`, `v`: Buffers cho attention
- `att`: Attention scores
- `key_cache`, `value_cache`: KV cache
- `logits`: Output logits
- `mask`: Sliding window mask

### Requests (từ getp_eval.cpp)
```cpp
typedef struct {
  int num_reqs;        // Số lượng request
  int max_token_len;   // Độ dài token tối đa
  int max_seq_len;     // Độ dài sequence tối đa
  char *str_reqs;      // Buffer chứa các request string
  int *tok_gens;       // Buffer chứa các token được generate
} Requests;
```

### Sampler (từ run.cpp)
```cpp
typedef struct {
  int vocab_size;                // Kích thước từ vựng
  ProbIndex *probindex;          // Buffer cho top-p sampling
  float temperature;             // Temperature cho sampling
  float topp;                    // Threshold cho top-p sampling
  unsigned long long rng_state;  // Trạng thái random number generator
} Sampler;
```

## Giải thích từng hàm

### 1. compute_concentration_and_inv_freq_getp
```cpp
void compute_concentration_and_inv_freq_getp(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *concentration_out,
                                        float *inv_freq_out)
```

**Mục đích**: Tính toán concentration factor và inverse frequency cho RoPE (Rotary Position Embedding) với YaRN scaling.

**Tham số**:
- `base`: Base frequency (thường là `Config.rope_theta`)
- `head_dim`: Số chiều của attention head (`Config.head_dim`)
- `scaling_factor`: Hệ số scale (`Config.rope_scaling_factor`)
- `initial_context_length`: Độ dài context ban đầu (`Config.initial_context_length`)
- `ntk_beta`, `ntk_alpha`: Tham số cho NTK (Neural Tangent Kernel) scaling
- `concentration_out`: Output concentration factor
- `inv_freq_out`: Output inverse frequencies (độ dài `head_dim/2`)

**Hoạt động**:
1. Tính toán các frequency ban đầu: `freq[i] = base^(i/head_dim)`
2. Nếu `scaling_factor > 1.0`: Áp dụng YaRN concentration và NTK scaling
3. Nếu không: Sử dụng frequency thông thường
4. Trả về concentration factor và inverse frequencies

### 2. compute_cos_sin_getp
```cpp
void compute_cos_sin_getp(int pos, float base, int head_dim, float scaling_factor,
                     float initial_context_length, float ntk_beta,
                     float ntk_alpha, float *cos_out, float *sin_out)
```

**Mục đích**: Tính toán giá trị cos và sin cho RoPE tại vị trí cụ thể.

**Tham số**:
- `pos`: Vị trí trong sequence
- Các tham số khác tương tự `compute_concentration_and_inv_freq_getp`
- `cos_out`, `sin_out`: Output arrays cho cos và sin values (độ dài `head_dim/2`)

**Hoạt động**:
1. Gọi `compute_concentration_and_inv_freq_getp` để lấy concentration và inv_freq
2. Với mỗi dimension j: tính `val = pos * inv_freq[j]`
3. Tính `cos_out[j] = cos(val) * concentration` và `sin_out[j] = sin(val) * concentration`

### 3. warm_up
```cpp
void warm_up(Transformer *transformer, Tokenizer *tokenizer)
```

**Mục đích**: Khởi tạo và chuẩn bị các tài nguyên cần thiết trước khi inference.

**Tham số**:
- `transformer`: Pointer tới struct Transformer
- `tokenizer`: Pointer tới struct Tokenizer

**Hoạt động**:
1. Pre-compute tất cả cos/sin values cho RoPE cho mọi vị trí trong sequence
2. Cấp phát memory cho `cos_vals` và `sin_vals` arrays
3. Điền sẵn các giá trị để tránh tính toán lặp lại trong quá trình inference

**Lưu ý**: Đây là optimization quan trọng - thay vì tính cos/sin mỗi lần forward, ta tính trước một lần.

### 4. finish
```cpp
void finish(Transformer *transformer, Tokenizer *tokenizer)
```

**Mục đích**: Giải phóng tài nguyên và dọn dẹp sau khi inference hoàn tất.

**Tham số**:
- `transformer`: Pointer tới struct Transformer
- `tokenizer`: Pointer tới struct Tokenizer

**Hoạt động**: Giải phóng memory được cấp phát trong `warm_up`, unload model, etc.

### 5. rmsnorm_getp
```cpp
void rmsnorm_getp(float *o, float *x, float *weight, int size)
```

**Mục đích**: Thực hiện RMSNorm (Root Mean Square Normalization).

**Tham số**:
- `o`: Output array (size elements)
- `x`: Input array (size elements)
- `weight`: Scale weights từ `TransformerWeights` (size elements)
- `size`: Số lượng elements (thường là `Config.hidden_dim`)

**Hoạt động**:
1. Tính sum of squares: `ss = Σ(x[j]^2) / size`
2. Chuẩn hóa: `ss = 1/sqrt(ss + 1e-5)`
3. Apply scaling: `o[j] = weight[j] * (ss * x[j])`

### 6. softmax_getp
```cpp
void softmax_getp(float *x, int size)
```

**Mục đích**: Áp dụng softmax function in-place.

**Tham số**:
- `x`: Input/output array (size elements)
- `size`: Số lượng elements

**Hoạt động**:
1. Tìm giá trị max để numerical stability
2. Tính exponential và sum: `x[i] = exp(x[i] - max_val)`
3. Normalize: `x[i] /= sum`

### 7. matmul_getp
```cpp
void matmul_getp(float *xout, float *x, float *w, int n, int d)
```

**Mục đích**: Thực hiện matrix multiplication được tối ưu hóa với OpenMP.

**Tham số**:
- `xout`: Output vector (d elements)
- `x`: Input vector (n elements)
- `w`: Weight matrix (d×n, stored row-major)
- `n`: Input features
- `d`: Output features

**Hoạt động**: Tính `xout = w @ x` với parallel processing sử dụng OpenMP.

### 8. accumulate
```cpp
void accumulate(float *a, float *b, float factor, int size)
```

**Mục đích**: Thực hiện scaled accumulation: `a += b * factor`.

**Tham số**:
- `a`: Target array (size elements)
- `b`: Source array (size elements)
- `factor`: Scaling factor
- `size`: Số lượng elements

### 9. apply_rotary_emb_getp
```cpp
void apply_rotary_emb_getp(float *x, float *cos, float *sin, int n_heads, int head_dim)
```

**Mục đích**: Áp dụng Rotary Position Embedding lên query hoặc key vectors.

**Tham số**:
- `x`: Input/output tensor (n_heads × head_dim)
- `cos`, `sin`: Pre-computed cos/sin values (head_dim/2 elements)
- `n_heads`: Số attention heads (`Config.n_attn_heads` hoặc `Config.n_kv_heads`)
- `head_dim`: Số chiều mỗi head (`Config.head_dim`)

**Hoạt động**:
1. Với mỗi head h và dimension i < head_dim/2:
2. Lấy x1 = x[h×head_dim + i] và x2 = x[h×head_dim + head_dim/2 + i]
3. Áp dụng rotation: 
   - `o1 = x1 * cos[i] - x2 * sin[i]`
   - `o2 = x2 * cos[i] + x1 * sin[i]`

### 10. sdpa_getp
```cpp
void sdpa_getp(Transformer *transformer, unsigned long long l, int pos)
```

**Mục đích**: Thực hiện Scaled Dot-Product Attention.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `l`: Layer index (0 đến `Config.n_layers-1`)
- `pos`: Position trong sequence

**Hoạt động**:
1. Tính attention scores giữa query và tất cả keys đến vị trí hiện tại
2. Áp dụng sliding window mask nếu được enable
- Chú ý nếu áp dụng chỉ áp dụng ở các layer có index chẵn
```c++
   // Apply sliding window mask if enabled
      if (p->sliding_window > 0 && (l % 2 == 0)) {
        score += s->mask[pos * p->seq_len + t];
      }
```
3. Thêm attention sink scores
4. Softmax scores
5. Weighted sum của values

### 11. attention_getp
```cpp
void attention_getp(Transformer *transformer, float *x, unsigned long long l, int pos)
```

**Mục đích**: Thực hiện toàn bộ attention mechanism cho một layer.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `x`: Input/output activations (`Config.hidden_dim` elements)
- `l`: Layer index
- `pos`: Position trong sequence

**Hoạt động**:
1. RMSNorm input
2. Tính QKV projections với matmul + bias
3. Tách Q, K, V từ QKV output
4. Áp dụng RoPE cho Q và K
5. Gọi `sdpa_getp` để tính attention
6. Output projection với matmul + bias
7. Residual connection

### 12. swiglu_getp
```cpp
void swiglu_getp(float *x, float *gate, float *up, float *gate_up, int intermediate_dim, float swiglu_getp_limit)
```

**Mục đích**: Thực hiện SwiGLU activation function.

**Tham số**:
- `x`: Unused trong implementation này
- `gate`: Gate values (`Config.intermediate_dim` elements)
- `up`: Up values (`Config.intermediate_dim` elements)
- `gate_up`: Output (`Config.intermediate_dim` elements)
- `intermediate_dim`: `Config.intermediate_dim`
- `swiglu_getp_limit`: `Config.swiglu_limit`

**Hoạt động**:
1. Clamp gate và up values trong giới hạn [-limit, limit]
2. Áp dụng SiLU cho gate: `gate *= 1/(1 + exp(-alpha * gate))`
3. Nhân với up (có thêm bias +1): `gate_up[i] = gate * (up + 1)`

### 13. mlp_getp
```cpp
void mlp_getp(Transformer *transformer, float *x, float *w_mlp1, float *b_mlp1, float *w_mlp2, float *b_mlp2, float expert_w)
```

**Mục đích**: Thực hiện MLP computation cho một expert cụ thể.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `x`: Input (unused)
- `w_mlp1`, `b_mlp1`: Weights và bias cho layer 1 (gate_up projection)
- `w_mlp2`, `b_mlp2`: Weights và bias cho layer 2 (down projection)
- `expert_w`: Weight của expert này từ router

**Hoạt động**:
1. First linear: `mlp1_out = w_mlp1 @ t + b_mlp1`
2. Tách thành gate và up components
3. Áp dụng SwiGLU activation
4. Second linear: `tb2 = w_mlp2 @ gate_up + b_mlp2`
5. Accumulate vào expert aggregation với weight `expert_w`

### 14. moe_getp
```cpp
void moe_getp(Transformer *transformer, float *x, unsigned long long l, int pos)
```

**Mục đích**: Thực hiện Mixture of Experts cho một layer.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `x`: Input/output activations
- `l`: Layer index
- `pos`: Position (unused)

**Hoạt động**:
1. RMSNorm input
2. Tính router scores: `router_score = w_router @ t + b_router`
3. Select top-k experts
4. Softmax router scores
5. Với mỗi selected expert: gọi `mlp_getp`
6. Residual connection với aggregated expert outputs

### 15. forward_getp
```cpp
float *forward_getp(Transformer *transformer, int token, int pos)
```

**Mục đích**: Thực hiện forward pass cho một token tại vị trí cụ thể.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `token`: Token ID
- `pos`: Position trong sequence

**Trả về**: Pointer tới logits array (`Config.vocab_size` elements)

**Hoạt động**:
1. Copy token embedding vào activation buffer
2. Với mỗi layer: 
   - Gọi `attention_getp`
   - Gọi `moe_getp`
3. Final RMSNorm
4. Classifier projection để tạo logits

### 16. simple_getp_generate
```cpp
long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps)
```

**Mục đích**: Generate text từ input prompt.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `tokenizer`: Pointer tới Tokenizer struct
- `sampler`: Pointer tới Sampler struct
- `input_seq`: Input prompt string
- `output_tokens`: Buffer để lưu generated tokens
- `steps`: Số steps tối đa để generate (`Requests.max_seq_len`)

**Trả về**: Số tokens đã generate

**Hoạt động**:
1. Encode input prompt thành tokens
2. Generation loop:
   - Forward pass để lấy logits
   - Nếu vẫn trong prompt: dùng prompt token tiếp theo
   - Nếu không: sample token từ logits
   - Lưu generated token
   - Kiểm tra termination condition
   - Decode và print token (debugging - sẽ được remove)

### 17. inference
```cpp
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests)
```

**Mục đích**: Thực hiện inference cho nhiều requests.

**Tham số**:
- `transformer`: Pointer tới Transformer struct
- `tokenizer`: Pointer tới Tokenizer struct
- `sampler`: Pointer tới Sampler struct
- `requests`: Pointer tới Requests struct chứa tất cả requests

**Trả về**: Tổng số tokens đã generate

**Hoạt động**:
1. Iterate qua tất cả requests trong batch
2. Với mỗi request: gọi `simple_getp_generate`
3. Accumulate tổng số tokens generated

## Tối ưu hóa so với run.cpp

1. **Pre-computed RoPE**: Tính trước cos/sin values trong `warm_up`
2. **OpenMP parallelization**: Sử dụng trong `matmul_getp` và `sdpa_getp`
3. **Memory layout optimization**: Optimized access patterns
4. **Reduced function call overhead**: Inline nhiều operations
5. **Batch processing**: Xử lý nhiều requests cùng lúc

## Global Variables

- `float *cos_vals`: Pre-computed cosine values cho RoPE
- `float *sin_vals`: Pre-computed sine values cho RoPE

Cả hai arrays có kích thước `(Config.head_dim/2) * Config.seq_len` elements.
