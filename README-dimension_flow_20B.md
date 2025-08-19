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