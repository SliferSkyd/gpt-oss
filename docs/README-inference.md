# GPT-OSS Inference Flow Documentation

## Overview

This document provides a detailed explanation of the inference flow in the gpt-oss high-performance inference system. The implementation follows a transformer architecture with mixture-of-experts (MoE) layers, attention mechanisms with sliding windows, and RoPE positional encoding.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Structures](#data-structures)
3. [Inference Pipeline](#inference-pipeline)
4. [Execution Modes](#execution-modes)
5. [Key Algorithms](#key-algorithms)
6. [Memory Management](#memory-management)
7. [Performance Optimizations](#performance-optimizations)

## Architecture Overview

The gpt-oss model implements a transformer architecture with the following key components:

### Model Architecture
- **Transformer Layers**: Variable number of transformer blocks (e.g., 48 for 20B model)
- **Mixture of Experts (MoE)**: Each layer contains multiple expert networks
- **Attention Mechanism**: Multi-query attention with grouped query attention (GQA)
- **Positional Encoding**: RoPE (Rotary Position Embedding) with YaRN scaling
- **Sliding Window**: Optional sliding window attention for long sequences

### Key Features
- **Expert Routing**: Top-k expert selection per token
- **Attention Sinks**: Learnable attention sink values
- **SwiGLU Activation**: Custom activation function with clamping
- **RMSNorm**: Root Mean Square normalization layers

## Data Structures

### Config Structure
```c
typedef struct {
    // Model dimensions
    int vocab_size;           // Vocabulary size
    int hidden_dim;           // Model dimension
    int head_dim;            // Attention head dimension
    int n_attn_heads;        // Number of attention heads
    int n_kv_heads;          // Number of key-value heads (for MQA/GQA)
    int n_layers;            // Number of transformer layers
    
    // MoE configuration
    int n_experts;           // Number of experts per layer
    int experts_per_token;   // Top-k experts selected per token
    int intermediate_dim;    // FFN intermediate dimension
    
    // Sequence configuration
    int seq_len;             // Maximum sequence length
    int initial_context_length; // Initial context window
    int sliding_window;      // Sliding window size (0 = disabled)
    
    // Model-specific parameters
    float rope_theta;        // RoPE base frequency
    float rope_scaling_factor; // YaRN scaling factor
    float swiglu_limit;      // SwiGLU clamping limit
} Config;
```

### TransformerWeights Structure
Contains all model parameters memory-mapped from the binary file:

- **Embeddings**: `token_embedding_table`, `out` (unembedding)
- **Normalization**: `rms_attn_w`, `rms_ffn_w`, `rms_out_w`
- **Attention**: `w_qkv`, `b_qkv`, `w_o`, `b_o`, `attn_sinks`
- **MoE**: `w_router`, `b_router`, `w_mlp1`, `b_mlp1`, `w_mlp2`, `b_mlp2`

### RunState Structure
Runtime buffers for forward pass computations:

- **Activations**: `x` (current state), `t` (normalized state)
- **Attention**: `q`, `k`, `v`, `att`, `key_cache`, `value_cache`
- **MoE**: `router_score`, `topk_v`, `topk_i`, `e_agg`
- **Intermediate**: Various temporary buffers for computations

## Inference Pipeline

### 1. Initialization Phase

```
1. Load model checkpoint (config + weights)
2. Memory map weights from binary file
3. Allocate runtime state buffers
4. Initialize tokenizer from binary file
5. Setup sampler with temperature/top-p parameters
```

### 2. Forward Pass (`forward()` function)

The core inference happens in the `forward()` function in `run.cpp:528`:

#### Step 1: Token Embedding
```c
// Copy token embedding into activation buffer
float *content_row = w->token_embedding_table + token * hidden_dim;
memcpy(x, content_row, hidden_dim * sizeof(*x));
```

#### Step 2: Transformer Layers Loop
For each transformer layer (0 to n_layers-1):

##### 2.1 Attention Block
```
a) RMSNorm on input: rmsnorm(s->t, x, w->rms_attn_w + l * hidden_dim)
b) QKV projection: matmul + bias addition
c) Split into Q, K, V tensors
d) Apply RoPE positional encoding
e) Multi-head attention computation:
   - Compute attention scores (Q·K^T / √d_k)
   - Apply sliding window mask (if enabled)
   - Add attention sink scores
   - Apply softmax normalization
   - Weighted sum with values (attention·V)
f) Output projection: matmul with w_o + bias
g) Residual connection: x += attention_output
```

##### 2.2 MoE (Mixture of Experts) Block
```
a) RMSNorm on residual: rmsnorm(s->t, x, w->rms_ffn_w + l * hidden_dim)
b) Router computation: matmul(router_score, t, w_router) + bias
c) Top-k expert selection: topk(topk_v, topk_i, router_score, experts_per_token)
d) Expert weight normalization: softmax(topk_v)
e) For each selected expert:
   - Gate/Up projection: matmul + bias
   - SwiGLU activation with clamping
   - Down projection: matmul + bias
   - Weighted aggregation based on router scores
f) Residual connection: x += moe_output
```

#### Step 3: Final Processing
```
a) Final RMSNorm: rmsnorm(s->t, x, w->rms_out_w)
b) Output projection: matmul(logits, t, w->out)
c) Return logits for next token prediction
```

### 3. Sampling Phase

After getting logits from forward pass:
```
1. Apply temperature scaling
2. Apply top-p (nucleus) sampling
3. Sample next token from probability distribution
4. Update KV cache with current token's key/value
```

## Execution Modes

### 1. Generate Mode (`generate()`)
Single prompt completion:
```
1. Tokenize input prompt
2. For each position in sequence:
   a) Run forward pass
   b) Sample next token
   c) Append to sequence
   d) Check for EOS token
3. Decode and output generated text
```

### 2. Chat Mode (`chat()`)
Interactive conversation:
```
1. Setup system prompt (if provided)
2. Interactive loop:
   a) Read user input
   b) Format as chat message
   c) Run generation
   d) Display response
   e) Continue conversation
```

### 3. Batch Evaluation Mode (`getp()`)
Batch processing for evaluation:
```
1. Read requests from input file
2. For each request:
   a) Tokenize input
   b) Run forward pass
   c) Store generated token IDs
3. Write all outputs to file
```

## Key Algorithms

### 1. RoPE (Rotary Position Embedding)
```c
void apply_rotary_emb(float *x, float *cos_vals, float *sin_vals, 
                      int n_heads, int head_dim) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float x1 = x[h * head_dim + i];
            float x2 = x[h * head_dim + head_dim/2 + i];
            float c = cos_vals[i];
            float s = sin_vals[i];
            
            x[h * head_dim + i] = x1 * c - x2 * s;
            x[h * head_dim + head_dim/2 + i] = x1 * s + x2 * c;
        }
    }
}
```

### 2. SwiGLU Activation
```c
// SwiGLU with clamping: gate * silu(up)
for (int i = 0; i < intermediate_dim; i++) {
    float gate_val = clamp(s->gate[i], -swiglu_limit, swiglu_limit);
    float up_val = clamp(s->up[i], -swiglu_limit, swiglu_limit);
    
    // SiLU(x) = x * sigmoid(x)
    gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));
    s->gate_up[i] = gate_val * up_val;
}
```

### 3. Sliding Window Attention
```c
// Apply sliding window mask during attention computation
if (p->sliding_window > 0 && (l % 2 == 0)) {
    if (pos - t >= p->sliding_window) {
        att[t] = -INFINITY;  // Mask out distant tokens
    }
}
```

### 4. Top-K Expert Selection
```c
void topk(float *values, int *indices, float *scores, 
          int n_experts, int k) {
    // Find k experts with highest router scores
    // Used for MoE routing decisions
}
```

## Memory Management

### Memory Layout
The system uses memory mapping for efficient weight loading:

```
Binary File Layout:
├── Config struct (header)
├── Token embeddings (vocab_size × hidden_dim)
├── Output embeddings (vocab_size × hidden_dim)
├── Layer normalization weights
├── Attention weights (QKV, O projections)
├── Attention biases
├── Router weights and biases
├── MLP weights and biases (for all experts)
└── Final normalization weights
```

### Buffer Allocation
- **Static**: Model weights (memory-mapped, read-only)
- **Dynamic**: Runtime state buffers (allocated per inference)
- **Cached**: Key-value cache (seq_len × kv_dim per layer)

### Memory Access Patterns
- **Sequential**: Token embeddings, layer processing
- **Random**: Expert selection, attention computation
- **Cached**: KV cache for autoregressive generation

## Performance Optimizations

### Current Optimizations
1. **Memory Mapping**: Zero-copy weight loading
2. **OpenMP**: Parallel attention head computation
3. **Cache Efficiency**: Contiguous memory layouts
4. **SIMD-friendly**: Float32 operations throughout

### Potential GPU Optimizations
1. **Custom Kernels**: HIP-based attention, MLP kernels
2. **Memory Coalescing**: Optimized memory access patterns
3. **Tensor Parallelism**: Multi-GPU expert parallelism
4. **Kernel Fusion**: Combined operations for efficiency

### Optimization Points in Code

#### Attention Computation (`run.cpp:597-642`)
```c
#pragma omp parallel for private(h)
for (h = 0; h < p->n_attn_heads; h++) {
    // Parallel attention head computation
    // Opportunity for GPU kernel optimization
}
```

#### MoE Expert Processing (`run.cpp:677-730`)
```c
for (int e = 0; e < n_experts; e++) {
    if (in_topk) {
        // Expert computation
        // Opportunity for parallel expert execution
    }
}
```

#### Matrix Operations
```c
void matmul(float *xout, float *x, float *w, int n, int d) {
    // Core matrix multiplication
    // Primary target for GPU acceleration
}
```

## Integration Points

### For GPU Acceleration
The main optimization opportunities lie in:

1. **`run.cpp:528-750`**: Forward pass implementation
2. **Matrix operations**: `matmul()`, attention computations
3. **Memory management**: KV cache, expert routing
4. **Custom kernels**: Attention, MoE, normalization layers

### Development Guidelines
- **Protected files**: Do not modify `run.cpp`, `getp_eval.cpp`, `Makefile`
- **Integration point**: Use `run_exec.cpp` for new implementations
- **Testing**: Use 7M model for development, 20B/120B for evaluation
- **Correctness first**: Validate against CPU baseline before optimizing

---

This inference flow documentation provides the foundation for understanding and optimizing the gpt-oss inference system. The modular design allows for targeted optimizations while maintaining correctness and compatibility with the existing codebase.