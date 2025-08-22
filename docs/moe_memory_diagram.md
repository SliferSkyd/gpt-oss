# ASCII Diagram: MoE Memory Layout và Flow

## Memory Layout Overview

```
╭──────────────────────────────────────────────────────────────────────────────────────╮
│                              GPU MEMORY LAYOUT                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐        │
│  │   Input Tensors     │  │    Weight Tensors   │  │  Output Tensors     │        │
│  │    (FP32)           │  │    (BF16)           │  │    (FP32)           │        │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                           INPUT TENSORS (FP32)                                 ││
│  ├─────────────────────────────────────────────────────────────────────────────────┤│
│  │ d_x          │ [batch_size, hidden_dim]     │ Input activations              ││
│  │ d_t          │ [batch_size, hidden_dim]     │ Temp buffer (after RMSNorm)   ││
│  │ d_router_score│ [batch_size, n_experts]     │ Router logits                  ││
│  │ d_topk_v     │ [batch_size, experts_per_token] │ Top-K expert weights       ││
│  │ d_topk_i     │ [batch_size, experts_per_token] │ Top-K expert indices       ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                         WEIGHT TENSORS (BF16)                                  ││
│  ├─────────────────────────────────────────────────────────────────────────────────┤│
│  │ d_rms_ffn_w  │ [n_layers, hidden_dim]       │ RMSNorm weights               ││
│  │ d_w_router   │ [n_layers, hidden_dim, n_experts] │ Router weights            ││
│  │ d_b_router   │ [n_layers, n_experts]        │ Router biases                 ││
│  │ d_w_mlp1     │ [n_layers,n_experts,2*inter_dim,hidden_dim] │ Gate/Up proj   ││
│  │ d_b_mlp1     │ [n_layers,n_experts,2*inter_dim] │ Gate/Up biases            ││
│  │ d_w_mlp2     │ [n_layers,n_experts,hidden_dim,inter_dim] │ Down projection  ││
│  │ d_b_mlp2     │ [n_layers,n_experts,hidden_dim] │ Down biases               ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                         EXPERT BUFFERS (FP32)                                  ││
│  ├─────────────────────────────────────────────────────────────────────────────────┤│
│  │ d_expert_input_buffer  │ [BATCH_SIZE, hidden_dim]     │ Gathered inputs        ││
│  │ d_expert_output_buffer │ [BATCH_SIZE, hidden_dim]     │ Expert outputs         ││
│  │ d_expert_indices       │ [BATCH_SIZE*K]               │ Original batch indices ││
│  │ d_expert_weights       │ [BATCH_SIZE*K]               │ Router weights         ││
│  │ d_batch_count          │ [1]                          │ Token count per expert ││
│  │ d_mlp1_out            │ [BATCH_SIZE, 2*inter_dim]     │ Gate/Up outputs        ││
│  │ d_gate                │ [BATCH_SIZE, inter_dim]       │ Gate activations       ││
│  │ d_up                  │ [BATCH_SIZE, inter_dim]       │ Up activations         ││
│  │ d_gate_up             │ [BATCH_SIZE, inter_dim]       │ SwiGLU outputs         ││
│  │ d_e_agg               │ [BATCH_SIZE, hidden_dim]      │ Final aggregation      ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

## MoE Flow Diagram

```
╭──────────────────────────────────────────────────────────────────────────────────────╮
│                                MoE EXECUTION FLOW                                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT: d_x [batch_size, hidden_dim]                                                │
│    │                                                                                │
│    ▼                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 1: RMSNorm                                                                ││
│  │ ┌─────────────┐    d_rms_ffn_w     ┌─────────────┐                            ││
│  │ │   d_x       │ ────────────────▶  │    d_t      │                            ││
│  │ │[B,H]        │    [layers,H]      │  [B,H]      │                            ││
│  │ └─────────────┘                    └─────────────┘                            ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│    │                                                                                │
│    ▼                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 2: Router Computation                                                     ││
│  │ ┌─────────────┐    d_w_router      ┌─────────────────┐                        ││
│  │ │    d_t      │ ────────────────▶  │ d_router_score  │                        ││
│  │ │  [B,H]      │ [layers,H,E] +bias │   [B,E]         │                        ││
│  │ └─────────────┘                    └─────────────────┘                        ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│    │                                                                                │
│    ▼                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 3: Top-K Selection & Softmax                                             ││
│  │ ┌─────────────────┐   topk_kernel   ┌─────────────┐  ┌─────────────┐          ││
│  │ │ d_router_score  │ ──────────────▶ │  d_topk_v   │  │  d_topk_i   │          ││
│  │ │     [B,E]       │                 │ [B,K]       │  │   [B,K]     │          ││
│  │ └─────────────────┘                 └─────────────┘  └─────────────┘          ││
│  │                                           │                                     ││
│  │                                           ▼ softmax                            ││
│  │                                     ┌─────────────┐                            ││
│  │                                     │  d_topk_v   │                            ││
│  │                                     │[B,K] (norm) │                            ││
│  │                                     └─────────────┘                            ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│    │                                                                                │
│    ▼                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 4: Expert Processing Loop (for each expert_id = 0 to n_experts-1)       ││
│  │                                                                               ││
│  │  ┌───────────────────────────────────────────────────────────────────────────┐ ││
│  │  │ STEP 4.1: GATHER PHASE                                                   │ ││
│  │  │ ┌─────────┐ ┌─────────┐ ┌─────────┐  gather_kernel  ┌─────────────────┐ │ ││
│  │  │ │   d_t   │ │d_topk_i │ │d_topk_v │ ──────────────▶ │d_expert_input_  │ │ ││
│  │  │ │  [B,H]  │ │ [B,K]   │ │ [B,K]   │                 │buffer [N,H]     │ │ ││
│  │  │ └─────────┘ └─────────┘ └─────────┘                 └─────────────────┘ │ ││
│  │  │                                                     ┌─────────────────┐ │ ││
│  │  │                                                     │d_expert_indices │ │ ││
│  │  │                                                     │d_expert_weights │ │ ││
│  │  │                                                     │d_batch_count[1] │ │ ││
│  │  │                                                     └─────────────────┘ │ ││
│  │  └───────────────────────────────────────────────────────────────────────────┘ ││
│  │    │                                                                         ││
│  │    ▼                                                                         ││
│  │  ┌───────────────────────────────────────────────────────────────────────────┐ ││
│  │  │ STEP 4.2: COMPUTE PHASE                                                  │ ││
│  │  │                                                                           │ ││
│  │  │ ┌─────────────────┐  d_w_mlp1   ┌─────────────────┐                     │ ││
│  │  │ │d_expert_input_  │ ──────────▶ │   d_mlp1_out    │                     │ ││
│  │  │ │buffer [N,H]     │ [L,E,2I,H]  │   [N,2I]        │                     │ ││
│  │  │ └─────────────────┘             └─────────────────┘                     │ ││
│  │  │                                           │                              │ ││
│  │  │                                           ▼ split + bias                │ ││
│  │  │                      ┌─────────────┐  ┌─────────────┐                  │ ││
│  │  │                      │   d_gate    │  │    d_up     │                  │ ││
│  │  │                      │   [N,I]     │  │   [N,I]     │                  │ ││
│  │  │                      └─────────────┘  └─────────────┘                  │ ││
│  │  │                                │            │                          │ ││
│  │  │                                └────────────┘                          │ ││
│  │  │                                       │ SwiGLU                         │ ││
│  │  │                                       ▼                                │ ││
│  │  │                                ┌─────────────┐                         │ ││
│  │  │                                │  d_gate_up  │                         │ ││
│  │  │                                │   [N,I]     │                         │ ││
│  │  │                                └─────────────┘                         │ ││
│  │  │                                       │                                │ ││
│  │  │                                       ▼ d_w_mlp2                      │ ││
│  │  │                                ┌─────────────────┐                     │ ││
│  │  │                                │d_expert_output_ │                     │ ││
│  │  │                                │buffer [N,H]     │                     │ ││
│  │  │                                └─────────────────┘                     │ ││
│  │  └───────────────────────────────────────────────────────────────────────────┘ ││
│  │    │                                                                         ││
│  │    ▼                                                                         ││
│  │  ┌───────────────────────────────────────────────────────────────────────────┐ ││
│  │  │ STEP 4.3: SCATTER PHASE                                                  │ ││
│  │  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐             │ ││
│  │  │ │d_expert_output_ │ │d_expert_indices │ │d_expert_weights │             │ ││
│  │  │ │buffer [N,H]     │ │     [N]         │ │     [N]         │             │ ││
│  │  │ └─────────────────┘ └─────────────────┘ └─────────────────┘             │ ││
│  │  │           │                   │                   │                     │ ││
│  │  │           └───────────────────┼───────────────────┘                     │ ││
│  │  │                               ▼ scatter_kernel                          │ ││
│  │  │                        ┌─────────────────┐                              │ ││
│  │  │                        │    d_e_agg      │ (accumulate with weights)    │ ││
│  │  │                        │    [B,H]        │                              │ ││
│  │  │                        └─────────────────┘                              │ ││
│  │  └───────────────────────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│    │                                                                                │
│    ▼                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 5: Residual Connection                                                    ││
│  │ ┌─────────────┐    ┌─────────────┐     ┌─────────────┐                       ││
│  │ │    d_x      │ +  │   d_e_agg   │  =  │    d_x      │                       ││
│  │ │   [B,H]     │    │   [B,H]     │     │   [B,H]     │ (OUTPUT)              ││
│  │ └─────────────┘    └─────────────┘     └─────────────┘                       ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

## Legend:
- B = batch_size (8)
- H = hidden_dim 
- E = n_experts
- K = experts_per_token
- I = intermediate_dim
- L = n_layers
- N = actual number of tokens routed to current expert (≤ B*K)
- FP32 = 32-bit floating point
- BF16 = 16-bit brain floating point

## Về max_tokens trong context MoE:

**max_tokens ≠ tokenizer max_tokens!**

Trong implementation này:
- **Thực tế**: Expert buffers được allocate với size `BATCH_SIZE * hidden_dim`
- **max_tokens** trong diagram chỉ là khái niệm lý thuyết = `BATCH_SIZE * experts_per_token`
- **N** là số tokens thực tế được route đến expert hiện tại (≤ max_tokens)

**Ví dụ cụ thể**:
- BATCH_SIZE = 8  
- experts_per_token = 2
- Lý thuyết max_tokens = 8 * 2 = 16 tokens có thể được route đến 1 expert
- Thực tế: Buffer được allocate cho 8 tokens (BATCH_SIZE)
- N có thể là 0-8 tokens thực tế được route đến expert đó

**Tại sao buffer chỉ cần BATCH_SIZE?**
- Mỗi token trong batch chỉ có thể xuất hiện 1 lần trong expert buffer  
- Dù token đó chọn nhiều experts, nó vẫn chỉ cần 1 slot trong mỗi expert buffer
- Gather phase copy từng token vào compact buffer dựa trên expert routing

## Key Features:
1. **Memory Efficient**: Uses bfloat16 for weights to save 50% memory
2. **Dynamic Batching**: Only processes tokens routed to each expert
3. **Gather-Compute-Scatter**: Efficient expert processing pattern
4. **Atomic Operations**: Safe aggregation with multiple experts per token
