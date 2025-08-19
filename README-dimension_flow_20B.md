# GPT-OSS Model: Complete Dimension Flow (Batch Size = 4)

## Model Configuration
```
vocab_size: 201,088
hidden_dim: 2,880  
n_experts: 32
experts_per_token: 4
intermediate_dim: 2,880
n_layers: 24
head_dim: 64
n_attn_heads: 64
n_kv_heads: 8
max_seq_len: 2,048
sliding_window: 128
```

---

## Input Processing Flow

```
INPUT TEXT BATCH (4 samples)
┌─────────────────────────────────────────┐
│ Sample 1: "Hello world how are you"     │ → length: 5
│ Sample 2: "AI is amazing technology"    │ → length: 4  
│ Sample 3: "Deep learning models"        │ → length: 3
│ Sample 4: "Transformers are powerful"   │ → length: 3
└─────────────────────────────────────────┘
                    ↓ TOKENIZATION
┌─────────────────────────────────────────┐
│ Token IDs (max length = 5, padded)      │
│ [1234, 5678, 9012, 3456, 7890]          │ Sample 1
│ [2345, 6789, 0123, 4567,    0]          │ Sample 2 (+ 1 pad)
│ [3456, 7890, 1234,    0,    0]          │ Sample 3 (+ 2 pads)  
│ [4567, 8901, 2345,    0,    0]          │ Sample 4 (+ 2 pads)
└─────────────────────────────────────────┘
                SHAPE: [4, 5]
                    ↓
```

---

## 1. TOKEN EMBEDDING LAYER

```
EMBEDDING LOOKUP
┌─────────────────────────────────────────┐
│ Embedding Matrix: [201088, 2880]        │
│                                         │
│ Input:  [4, 5] token IDs                │
│         ↓ embedding_lookup              │
│ Output: [4, 5, 2880] embeddings         │
└─────────────────────────────────────────┘

VISUALIZATION (simplified to show concept):
Token IDs → Embedding Vectors

[1234] → [0.23, -0.45, 0.67, ..., 0.89] (2880 dims)
[5678] → [0.12, 0.78, -0.34, ..., 0.56] (2880 dims)
[9012] → [-0.23, 0.45, 0.89, ..., 0.12] (2880 dims)
[3456] → [0.34, -0.67, 0.23, ..., 0.78] (2880 dims)
[7890] → [0.56, 0.12, -0.89, ..., 0.34] (2880 dims)

TENSOR SHAPE: [4, 5, 2880]
```

---

## 2. TRANSFORMER LAYER (Repeated 24 times)

### Layer Input: [4, 5, 2880]

```
┌══════════════════════════════════════════════════════════════════┐
║                      TRANSFORMER LAYER                           ║
╠══════════════════════════════════════════════════════════════════╣
║  INPUT: [4, 5, 2880]                                             ║
║    ↓                                                             ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ RMSNorm 1                                                  │  ║
║  │ Input:  [4, 5, 2880]                                       │  ║
║  │ Output: [4, 5, 2880] (normalized)                          │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║    ↓                                                             ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ GROUPED QUERY ATTENTION                                    │  ║
║  │                                                            │  ║
║  │ PROJECTION WEIGHTS:                                        │  ║
║  │ • W_Q: [2880, 64×64] = [2880, 4096]                        │  ║
║  │ • W_K: [2880, 8×64]  = [2880, 512]                         │  ║
║  │ • W_V: [2880, 8×64]  = [2880, 512]                         │  ║
║  │ • W_O: [2880, 2880]                                        │  ║
║  │                                                            │  ║
║  │ FORWARD PASS:                                              │  ║
║  │ Input: [4, 5, 2880]                                        │  ║
║  │   ↓ Linear projections                                     │  ║ 
║  │ Q: [4, 5, 2880] → [4, 5, 4096] → [4, 5, 64, 64]            │  ║
║  │ K: [4, 5, 2880] → [4, 5, 512]  → [4, 5, 8, 64]             │  ║
║  │ V: [4, 5, 2880] → [4, 5, 512]  → [4, 5, 8, 64]             │  ║
║  │   ↓ Transpose for attention                                │  ║
║  │ Q: [4, 64, 5, 64]  (64 query heads)                        │  ║
║  │ K: [4, 8, 5, 64]   (8 key heads)                           │  ║
║  │ V: [4, 8, 5, 64]   (8 value heads)                         │  ║
║  │                                                            │  ║
║  │ GROUPED ATTENTION COMPUTATION:                             │  ║
║  │ Group 1: Q[0:8]   × K[0] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 2: Q[8:16]  × K[1] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 3: Q[16:24] × K[2] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 4: Q[24:32] × K[3] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 5: Q[32:40] × K[4] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 6: Q[40:48] × K[5] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 7: Q[48:56] × K[6] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │ Group 8: Q[56:64] × K[7] → Attn_scores: [4, 8, 5, 5]       │  ║
║  │                                                            │  ║
║  │ SLIDING WINDOW ATTENTION (window=128):                     │  ║
║  │ Each token can only attend to 128 previous tokens          │  ║
║  │ For seq_len=5: all tokens can attend to all (5 < 128)      │  ║
║  │                                                            │  ║
║  │ Attention Output per group: [4, 8, 5, 64]                  │  ║
║  │ Concatenate all groups: [4, 64, 5, 64] → [4, 5, 4096]      │  ║
║  │ Output projection: [4, 5, 4096] × [4096, 2880] → [4,5,2880]│  ║
║  └────────────────────────────────────────────────────────────┘  ║
║    ↓ Residual connection                                         ║
║  [4, 5, 2880] + [4, 5, 2880] = [4, 5, 2880]                      ║
║    ↓                                                             ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ RMSNorm 2                                                  │  ║
║  │ Input:  [4, 5, 2880]                                       │  ║
║  │ Output: [4, 5, 2880] (normalized)                          │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║    ↓                                                             ║
║  ┌────────────────────────────────────────────────────────────┐  ║
║  │ MIXTURE OF EXPERTS (MoE)                                   │  ║
║  │                                                            │  ║
║  │ ROUTER:                                                    │  ║
║  │ Input: [4, 5, 2880]                                        │  ║
║  │ Router weights: [2880, 32]                                 │  ║
║  │ Router logits: [4, 5, 32] (probabilities for 32 experts)   │  ║
║  │   ↓ Top-K selection (K=4)                                  │  ║
║  │ Selected experts per token: 4 out of 32                    │  ║
║  │                                                            │  ║
║  │ EXPERT COMPUTATION (for each selected expert):             │  ║
║  │ SwiGLU Architecture:                                       │  ║
║  │ • Gate:   [2880, 2880] × [4, 5, 2880] = [4, 5, 2880]       │  ║
║  │ • Up:     [2880, 2880] × [4, 5, 2880] = [4, 5, 2880]       │  ║
║  │ • SiLU:   gate = silu(gate)                                │  ║
║  │ • Mult:   gate × up = [4, 5, 2880]                         │  ║
║  │ • Down:   [2880, 2880] × [4, 5, 2880] = [4, 5, 2880]       │  ║
║  │                                                            │  ║
║  │ EXPERT MIXING:                                             │  ║
║  │ weight1 × expert1_output + weight2 × expert2_output +      │  ║
║  │ weight3 × expert3_output + weight4 × expert4_output        │  ║
║  │ = [4, 5, 2880]                                             │  ║
║  │                                                            │  ║
║  │ LOAD BALANCING: Ensure experts are used roughly equally    │  ║
║  └────────────────────────────────────────────────────────────┘  ║
║    ↓ Residual connection                                         ║
║  [4, 5, 2880] + [4, 5, 2880] = [4, 5, 2880]                      ║
║    ↓                                                             ║
║  OUTPUT: [4, 5, 2880]                                            ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. FINAL LAYERS

```
AFTER 24 TRANSFORMER LAYERS: [4, 5, 2880]
                    ↓
┌─────────────────────────────────────────┐
│ FINAL RMSNorm                           │
│ Input:  [4, 5, 2880]                    │
│ Output: [4, 5, 2880] (normalized)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ LINEAR OUTPUT LAYER                     │
│ Weight matrix: [2880, 201088]           │
│ Input:  [4, 5, 2880]                    │
│ Output: [4, 5, 201088] (logits)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ NEXT TOKEN PREDICTION                   │
│ Take last position: [4, 201088]         │
│ Apply softmax: [4, 201088] probs        │
│ Sample next token: [4, 1]               │
└─────────────────────────────────────────┘
```

---

## 4. GENERATION LOOP (Autoregressive)

```
GENERATION PROCESS:
Initial sequence: [4, 5] 
Step 1: Generate token 6 → [4, 6]
Step 2: Generate token 7 → [4, 7] 
...
Continue until <EOS> token or max_length

SLIDING WINDOW (128 tokens):
When sequence > 128:
- Only keep last 128 tokens for attention
- Older tokens are "forgotten" 
- Memory usage stays constant

KV CACHE OPTIMIZATION:
Cached tensors per layer:
• K cache: [4, 8, current_seq_len, 64]   
• V cache: [4, 8, current_seq_len, 64]

Memory savings from GQA:
• MHA would need: [4, 64, current_seq_len, 64] × 2
• GQA only needs: [4, 8, current_seq_len, 64] × 2  
• 8x memory reduction for KV cache!
```

---

## 5. MEMORY ANALYSIS

```
PARAMETER COUNT BREAKDOWN:
┌─────────────────────────────────────────┐
│ Token Embedding: 201088 × 2880          │ = 578,813,440
│ Each Transformer Layer:                 │
│   • Attention:                          │
│     - W_Q: 2880 × 4096                  │ = 11,796,480
│     - W_K: 2880 × 512                   │ = 1,474,560  
│     - W_V: 2880 × 512                   │ = 1,474,560
│     - W_O: 2880 × 2880                  │ = 8,294,400
│   • RMSNorm weights: 2880 × 2           │ = 5,760
│   • MoE (32 experts):                   │
│     - Router: 2880 × 32                 │ = 92,160
│     - Each expert: 3 × (2880 × 2880)    │ = 24,883,200
│     - Total experts: 32 × 24,883,200    │ = 796,262,400
│ × 24 layers                             │
│ Final norm + output: 2880 + 2880×201088 │ = 578,816,320
└─────────────────────────────────────────┘

TOTAL: ~20B parameters

ACTIVATION MEMORY (batch_size=4, seq_len=5):
┌─────────────────────────────────────────┐
│ Embeddings: [4, 5, 2880]                │ = 57,600 floats
│ Each layer activations:                 │
│   • Attention intermediate              │ ≈ 200,000 floats  
│   • MoE intermediate                    │ ≈ 150,000 floats
│ × 24 layers                             │ = 8,400,000 floats
│ KV Cache (inference):                   │
│   • Per layer: [4, 8, seq, 64] × 2      │
│   • 24 layers × growing sequence        │
└─────────────────────────────────────────┘
```

---

## Key Insights:

1. **GQA Efficiency**: 64 Q-heads but only 8 KV-heads reduces memory by 8x
2. **MoE Scaling**: Only 4/32 experts active per token (12.5% utilization)
3. **Sliding Window**: Attention limited to 128 tokens for memory efficiency
4. **Parameter Distribution**: Most parameters in MoE experts (~80%)
5. **Activation Peaks**: MoE layers have highest activation memory

---

## Deep Dive: `sdpa_getp` Function - Core Attention Implementation

### Hàm `sdpa_getp` (Scaled Dot-Product Attention) - Giải thích chi tiết

`sdpa_getp` là trái tim của attention mechanism, implement **Grouped Query Attention (GQA)** với sliding window và attention sinks. Đây là hàm quan trọng nhất quyết định hiệu năng và chất lượng của model.

### 📍 **Function Signature & Parameters**
```cpp
void sdpa_getp(Transformer *transformer, unsigned long long l, int pos)
```
- `transformer`: Pointer đến model state
- `l`: Layer hiện tại (0-23)
- `pos`: Vị trí token hiện tại trong sequence

### 🔧 **1. Initialization & Key Variables**
```cpp
Config *p = &transformer->config;
TransformerWeights *w = &transformer->weights;
RunState *s = &transformer->state;

int head_dim = p->head_dim;          // 64 - dimension mỗi head
int hidden_dim = p->hidden_dim;      // 2880 - total hidden dimension  
int kv_dim = p->head_dim * p->n_kv_heads;  // 64 * 8 = 512 - KV dimension
int kv_mul = p->n_attn_heads / p->n_kv_heads; // 64/8 = 8 - GQA multiplier
int loff = l * p->seq_len * kv_dim;  // layer offset trong KV cache
```

**Ý nghĩa các biến:**
- `kv_mul = 8`: Mỗi KV head phục vụ 8 query heads (GQA efficiency)
- `loff`: Offset để locate KV cache của layer `l` trong memory
- `kv_dim = 512`: Total dimension cho tất cả KV heads (8 × 64)

### 🚀 **2. Parallel Head Processing**
```cpp
#pragma omp parallel for private(h)
for (h = 0; h < p->n_attn_heads; h++) {  // 64 attention heads
    float *q = s->q + h * head_dim;      // Query vector cho head h [64 dims]
    float *att = s->att + h * p->seq_len; // Attention scores buffer [2048 dims]
```

**OpenMP Parallelization**: 64 heads được xử lý song song trên multiple CPU cores, tăng throughput đáng kể.

### 🧮 **3. Attention Score Computation (Q·K)**
```cpp
for (int t = 0; t <= pos; t++) {  // Duyệt tất cả positions từ 0 đến pos
    // GQA Magic: Shared KV heads
    float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    
    // Dot product Q·K
    double score = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        score += q[i] * k[i];  // Element-wise multiplication và sum
    }
    score /= sqrtf(head_dim);  // Scale by sqrt(64) = 8.0 for stability
```

**GQA Key Mapping**:
```
Query Head → Key Head Mapping:
Q[0-7]   → K[0]   (h/8 = 0)
Q[8-15]  → K[1]   (h/8 = 1)  
Q[16-23] → K[2]   (h/8 = 2)
Q[24-31] → K[3]   (h/8 = 3)
Q[32-39] → K[4]   (h/8 = 4)
Q[40-47] → K[5]   (h/8 = 5)
Q[48-55] → K[6]   (h/8 = 6)
Q[56-63] → K[7]   (h/8 = 7)
```

### 🪟 **4. Sliding Window Masking**
```cpp
// Apply sliding window mask nếu enabled
if (p->sliding_window > 0 && (l % 2 == 0)) {
    score += s->mask[pos * p->seq_len + t];
}
att[t] = score;  // Store attention score
```

**Sliding Window Logic**:
- Chỉ áp dụng cho **even layers** (`l % 2 == 0`) 
- Giới hạn attention trong window **128 tokens**
- `s->mask[i]` = `-∞` cho tokens ngoài window, `0` cho tokens trong window
- Giảm complexity từ O(n²) xuống O(n×128)

### 🎯 **5. Attention Sinks**
```cpp
// Thêm attention sink score
att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
```

**Attention Sinks Explained**:
- **Learned parameters** không phụ thuộc vào input
- Cho phép model "attend" đến global information
- Bypass sliding window limitation
- Critical cho long sequences

### 📊 **6. Softmax Normalization**
```cpp
softmax_getp(att, pos + 2);  // Softmax over [0..pos+1] scores
```

**Softmax Range**: `pos + 2` elements bao gồm:
- Positions `[0..pos]`: Actual token positions
- Position `pos + 1`: Attention sink
- Total: `pos + 2` elements

### 🔄 **7. Weighted Value Sum (Attention·V)**
```cpp
float *tb = s->tb + h * head_dim;  // Output buffer cho head h [64 dims]
memset(tb, 0, head_dim * sizeof(float));  // Reset to zero

for (int t = 0; t <= pos; t++) {
    // GQA: Shared value head
    float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
    
    // Weighted accumulation: output += attention_weight × value_vector  
    accumulate(tb, v, att[t], head_dim);
}
```

**Output Computation**:
```
output[h] = Σ(t=0 to pos) attention[h][t] × value[h/8][t]
```

### 🧠 **Memory Layout Analysis**

#### **Traditional Multi-Head Attention (MHA)**:
```
Memory per layer:
├── Q: [64 heads × 64 dim] = 4,096 parameters
├── K: [64 heads × 64 dim] = 4,096 parameters  
├── V: [64 heads × 64 dim] = 4,096 parameters
└── Total KV: 8,192 parameters
```

#### **Grouped Query Attention (GQA)**:
```
Memory per layer:
├── Q: [64 heads × 64 dim] = 4,096 parameters
├── K: [8 heads × 64 dim]  = 512 parameters
├── V: [8 heads × 64 dim]  = 512 parameters  
└── Total KV: 1,024 parameters (8x smaller!)
```

### 📈 **Performance Characteristics**

#### **Time Complexity**:
- **Without sliding window**: O(pos × head_dim × n_heads) = O(pos × 64 × 64)
- **With sliding window**: O(min(pos, 128) × head_dim × n_heads) = O(128 × 64 × 64)
- **Attention sinks**: O(1) additional computation

#### **Memory Complexity**:
```
KV Cache per layer:
├── Keys:   [8 heads × pos × 64] = 512 × pos floats
├── Values: [8 heads × pos × 64] = 512 × pos floats  
├── Total:  1,024 × pos floats per layer
└── 24 layers: 24,576 × pos floats total
```

### 🔧 **Optimization Techniques**

#### **1. OpenMP Parallelization**
```cpp
#pragma omp parallel for private(h)
```
- Mỗi thread xử lý một subset của 64 heads
- Scaling gần như linear với số CPU cores
- Critical cho real-time inference

#### **2. Memory Access Optimization**
```cpp
// Sequential access pattern cho cache efficiency
float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
```

#### **3. GQA Memory Sharing**
- 8 query heads share 1 KV head
- Giảm 87.5% memory cho KV cache
- Minimal quality degradation

### 🎯 **Visual Flow Representation**

```
SDPA_GETP EXECUTION FLOW:
┌─────────────────────────────────────────────────────────────┐
│ Input: Q[64×64], K[8×64], V[8×64], pos=current_position     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FOR each of 64 attention heads (parallel):                 │
│   1. Get Q[h] vector (64 dims)                             │
│   2. FOR each position t from 0 to pos:                    │
│      • Get K[h/8][t] (shared key head)                     │
│      • Compute score = Q[h] · K[h/8][t] / √64              │
│      • Apply sliding window mask (even layers only)        │
│   3. Add attention sink score                              │
│   4. Apply softmax normalization                           │
│   5. FOR each position t from 0 to pos:                    │
│      • Get V[h/8][t] (shared value head)                   │
│      • Accumulate: output[h] += attention[t] × V[h/8][t]   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Output: attention_output[64×64] = concatenated head outputs │
└─────────────────────────────────────────────────────────────┘
```

### 🔍 **Critical Implementation Details**

#### **1. Numerical Stability**
```cpp
score /= sqrtf(head_dim);  // Prevents exploding gradients
```

#### **2. Causal Masking**
```cpp
for (int t = 0; t <= pos; t++)  // Only attend to past + current
```

#### **3. Memory Safety**
```cpp
memset(tb, 0, head_dim * sizeof(float));  // Clean slate cho mỗi head
```

### 💡 **Key Takeaways**

1. **GQA Efficiency**: 8x memory reduction với minimal quality loss
2. **Sliding Window**: Constant memory complexity for attention
3. **Attention Sinks**: Global context retention
4. **Parallel Processing**: Linear scaling với CPU cores
5. **Causal Attention**: Autoregressive text generation support

Hàm `sdpa_getp` là masterpiece of engineering, combining efficiency (GQA + sliding window) với expressiveness (attention sinks) để achieve sota performance on resource-constrained environments.

---


# Quy tắc sinh token trong batch có padding

## Tổng quan
Khi xử lý batch các sequence có độ dài khác nhau, việc sinh token mới phải tuân theo 3 nguyên tắc cốt lõi để đảm bảo tính nhất quán và hiệu quả.

## 🔄 Quy tắc 1: Xử lý padding khi sinh token
**Cơ chế hoạt động:**
- Các sequence ngắn hơn trong batch được **bổ sung padding token `<pad>`**
- Khi model sinh token mới, nó sẽ **thay thế vị trí `<pad>` đầu tiên** trong sequence
- Quá trình tiếp tục cho đến khi gặp điều kiện dừng

**Minh họa:**
```
Trạng thái ban đầu:
Sequence 1: [A, B, C, <pad>, <pad>]
Sequence 2: [X, Y, Z, W, <pad>]

Sau 1 lần sinh token:
Sequence 1: [A, B, C, D, <pad>]
Sequence 2: [X, Y, Z, W, Q]
```

## ⏹️ Quy tắc 2: Xử lý token kết thúc `<eos>`
**Cơ chế hoạt động:**
- Khi sequence nào sinh ra `<eos>`, sequence đó **ngừng sinh token**
- Model không tạo thêm token cho sequence đã hoàn thành
- Các vị trí còn lại giữ nguyên trạng thái `<eos>` và padding

**Minh họa:**
```
Sequence 1: [A, B, C, <eos>, <pad>, <pad>]  ← Đã hoàn thành
Sequence 2: [X, Y, Z, W, Q, <pad>]         ← Tiếp tục sinh
```

## 📏 Quy tắc 3: Xử lý giới hạn độ dài tối đa
**Cơ chế hoạt động:**
- Sequence đạt `max_length` mà chưa có `<eos>` sẽ **bị cắt ngắn**
- Kết quả được coi là hoàn tất dù không có token kết thúc
- Đây là cơ chế bảo vệ để tránh sinh token vô hạn

**Minh họa (với max_length=6):**
```
Sequence 1: [A, B, C, D, E, F]              ← Bị cắt tại giới hạn
Sequence 2: [X, Y, Z, <eos>, <pad>, <pad>]  ← Kết thúc tự nhiên
```

## 📝 Tóm tắt quan trọng
1. **Padding → Token mới**: Ghi đè padding bằng token được sinh
2. **Gặp `<eos>` → Dừng**: Sequence hoàn thành và không sinh thêm
3. **Đạt max_length → Cắt**: Dừng bắt buộc dù chưa có `<eos>`

## 🎯 Lưu ý thực tế
- Các quy tắc này đảm bảo batch processing hiệu quả
- Giúp tránh lãng phí tài nguyên tính toán
- Đảm bảo kết quả có độ dài phù hợp cho ứng dụng






