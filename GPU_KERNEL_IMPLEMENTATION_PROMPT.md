# GPU Kernel Implementation Prompt for GPT-OSS 20B

## Context
I have a GPT-OSS 20B MoE model that needs GPU optimization. The basic GPU infrastructure is set up with memory allocation and BF16 conversion. Now I need to implement all the actual GPU kernels.

## Current Setup
- AMD GPU with HIP (not CUDA), 64GB VRAM
- Model: 20B MoE with 24 layers, 32 experts, 4 experts/token
- Data types: CPU uses FP32, GPU uses BF16
- Batch size: 2
- Infrastructure files already created:
  - `include/gpu_utils.hpp` - GPU memory structures and utilities
  - `include/gpu_kernels.hpp` - Kernel function declarations
  - `getp-csrc/gpu_utils.cpp` - Memory management implementation
  - `getp-csrc/gpu_kernels.cpp` - Kernel placeholders
  - `getp-csrc/getp_run_gpu.cpp` - GPU forward pass orchestration

## Required GPU Kernels to Implement

### 1. BLAS Operations
- **matmul_batch_gpu**: Batched matrix multiplication (xout = x @ w^T)
  - Input: x[batch_size, n], w[d, n]
  - Output: xout[batch_size, d]
  - Currently has basic implementation, needs optimization

### 2. Attention Components

#### a. QKV Projection
- Project input through W_qkv and add bias
- Split into Q, K, V tensors
- Handle different n_attn_heads and n_kv_heads (MQA)

#### b. RoPE (Rotary Position Embedding)
- Apply rotary embeddings to Q and K
- Use precomputed cos/sin values
- Handle YaRN scaling parameters

#### c. SDPA (Scaled Dot-Product Attention)
- Compute attention scores: Q @ K^T / sqrt(head_dim)
- Apply sliding window mask if enabled
- Softmax over sequence dimension
- Attention weights @ V
- Support for attention sinks

#### d. Attention Output
- Concatenate heads
- Project through W_o and add bias

### 3. MoE (Mixture of Experts) Components

#### a. Router
- Compute router scores: x @ W_router + b_router
- Top-k selection (k=experts_per_token=4)
- Softmax normalization of selected expert weights

#### b. Expert Processing
- Gather tokens for each expert
- Process through expert MLPs:
  - gate_up = x @ W_mlp1 + b_mlp1
  - Split into gate and up
  - gate = swiglu(gate)
  - expert_out = (gate * up) @ W_mlp2 + b_mlp2
- Scatter results back weighted by router scores

### 4. Activation Functions

#### a. RMSNorm
- Compute RMS: sqrt(mean(x^2) + eps)
- Normalize and scale with learned weights
- Need efficient reduction for large hidden_dim

#### b. SwiGLU
- Clamp values to [-swiglu_limit, swiglu_limit]
- Apply SiLU: x * sigmoid(x)
- Element-wise multiplication with up tensor

### 5. KV Cache Management
- Update key_cache and value_cache at current position
- Handle batch dimension properly
- Efficient memory access patterns

## Optimization Requirements

### Memory Access Patterns
- Coalesce global memory accesses
- Use shared memory for reductions
- Minimize bank conflicts
- Tile computations for cache efficiency

### Parallelization Strategy
- Block dimensions for different tensor sizes
- Warp-level primitives for reductions
- Stream parallelism where applicable

### Numerical Considerations
- BF16 arithmetic throughout
- Mixed precision for reductions if needed
- Numerical stability for softmax and normalization

## CPU Reference Implementations
Check these files for CPU logic to mirror:
- `getp-csrc/DNN/attention.cpp` - Attention logic
- `getp-csrc/DNN/moe.cpp` - MoE routing and expert processing
- `getp-csrc/DNN/rmsnorm.cpp` - RMSNorm
- `getp-csrc/DNN/rope.cpp` - RoPE
- `getp-csrc/DNN/softmax.cpp` - Softmax
- `getp-csrc/DNN/swiglu.cpp` - SwiGLU activation
- `getp-csrc/BLAS.cpp` - Matrix operations

## Testing Requirements
Each kernel should:
1. Match CPU output within BF16 precision tolerance (~5% relative error)
2. Handle edge cases (sequence boundaries, padding)
3. Work correctly with batch processing
4. Synchronize properly (hipDeviceSynchronize where needed)

## Implementation Order (Suggested)
1. Fix and optimize matmul_batch_gpu
2. Implement rmsnorm_batch_gpu properly
3. Complete rope_batch_gpu
4. Implement full attention mechanism
5. Implement MoE components
6. Integration and testing

## Important Notes
- Use HIP API (not CUDA): `__hip_bfloat16`, `hipMalloc`, etc.
- Target architecture: gfx90a (MI200 series)
- Don't use external BLAS libraries - custom kernels only
- Maintain the existing function signatures in gpu_kernels.hpp
- The forward_batch_gpu function should orchestrate all kernels

Please implement all these GPU kernels with proper optimization for the AMD GPU architecture.