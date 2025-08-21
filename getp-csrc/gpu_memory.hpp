#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <memory>
#include <unordered_map>

// BF16 type definitions
using bf16 = __hip_bfloat16;
using bf16x2 = __hip_bfloat162;

// GPU Memory Management Layer
class GPUMemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        int device_id;
        bool is_bf16;
    };
    
    std::unordered_map<std::string, MemoryBlock> allocations;
    std::vector<int> device_ids;
    int num_gpus;
    
public:
    GPUMemoryManager();
    ~GPUMemoryManager();
    
    // Allocation methods
    void* allocate(const std::string& name, size_t size_bytes, int device_id, bool is_bf16 = true);
    void* allocate_multi_gpu(const std::string& name, size_t size_per_gpu, bool is_bf16 = true);
    void deallocate(const std::string& name);
    void deallocate_all();
    
    // Memory transfer methods
    void copy_host_to_device(const std::string& name, const float* host_data, size_t num_elements, bool convert_to_bf16 = true);
    void copy_device_to_host(const std::string& name, float* host_data, size_t num_elements);
    void copy_device_to_device(const std::string& src, const std::string& dst, size_t num_elements);
    
    // Utility methods
    void* get_ptr(const std::string& name) const;
    int get_device_id(const std::string& name) const;
    size_t get_size(const std::string& name) const;
    void set_device(int device_id);
    
    // Precision conversion utilities
    static void convert_fp32_to_bf16(const float* fp32, bf16* bf16_out, size_t num_elements);
    static void convert_bf16_to_fp32(const bf16* bf16_in, float* fp32, size_t num_elements);
};

// Model weight distribution across GPUs
struct GPUModelWeights {
    // Token embeddings - replicated across all GPUs for fast access
    bf16* token_embedding_table;  // [vocab_size, hidden_dim]
    
    // Per-layer weights distributed across GPUs
    struct LayerWeights {
        // Attention weights - layer parallelism
        bf16* rms_attn_w;     // [hidden_dim]
        bf16* w_qkv;          // [qkv_dim, hidden_dim]
        bf16* b_qkv;          // [qkv_dim]
        bf16* w_o;            // [hidden_dim, head_dim * n_heads]
        bf16* b_o;            // [hidden_dim]
        bf16* attn_sinks;     // [n_heads]
        
        // MoE weights - expert parallelism
        bf16* rms_ffn_w;      // [hidden_dim]
        bf16* w_router;       // [hidden_dim, n_experts]
        bf16* b_router;       // [n_experts]
        
        // Expert weights distributed across GPUs (4 experts per GPU)
        bf16* w_mlp1;         // [local_experts, 2*intermediate_dim, hidden_dim]
        bf16* w_mlp2;         // [local_experts, hidden_dim, intermediate_dim]
        bf16* b_mlp1;         // [local_experts, 2*intermediate_dim]
        bf16* b_mlp2;         // [local_experts, hidden_dim]
        
        int device_id;        // GPU this layer is on
        int expert_start_idx; // Starting expert index for this GPU
        int num_local_experts; // Number of experts on this GPU
    };
    
    std::vector<LayerWeights> layers;
    
    // Output weights - replicated for final projection
    bf16* rms_out_w;      // [hidden_dim]
    bf16* out;            // [vocab_size, hidden_dim]
};

// GPU Run State for batched inference
struct GPURunState {
    // Batch-aware activations [batch_size, ...]
    float* x;              // Current activations [batch, hidden_dim] - FP32 for accuracy
    float* residual;       // Residual connection [batch, hidden_dim]
    
    // Attention intermediates
    float* qkv;           // [batch, qkv_dim]
    float* q;             // [batch, n_heads, head_dim]
    float* k;             // [batch, n_kv_heads, head_dim]
    float* v;             // [batch, n_kv_heads, head_dim]
    float* att_scores;    // [batch, n_heads, seq_len]
    float* att_output;    // [batch, hidden_dim]
    
    // MoE intermediates  
    float* router_logits; // [batch, n_experts]
    float* router_probs;  // [batch, n_experts]
    int* expert_indices;  // [batch, experts_per_token]
    float* expert_weights;// [batch, experts_per_token]
    float* expert_input;  // [batch * experts_per_token, hidden_dim]
    float* expert_output; // [batch * experts_per_token, hidden_dim]
    
    // MLP intermediates
    bf16* mlp1_out;       // [batch * experts_per_token, 2*intermediate_dim] - BF16 for memory
    bf16* gate;           // [batch * experts_per_token, intermediate_dim]
    bf16* up;             // [batch * experts_per_token, intermediate_dim]
    float* mlp_output;    // [batch, hidden_dim]
    
    // KV Cache - distributed across layers
    bf16* key_cache;      // [n_layers, batch, seq_len, kv_dim]
    bf16* value_cache;    // [n_layers, batch, seq_len, kv_dim]
    
    // Output
    float* logits;        // [batch, vocab_size]
    
    // Metadata
    int* positions;       // [batch] - current position in sequence
    int* sequence_lengths;// [batch] - total sequence length
    bool* finished;       // [batch] - whether sequence is done
    
    int batch_size;
    int max_seq_len;
};

// Initialize GPU memory and weights
void initialize_gpu_memory(GPUModelWeights& gpu_weights, 
                          GPURunState& gpu_state,
                          const Config* config,
                          GPUMemoryManager& memory_mgr,
                          int batch_size);

// Load weights from CPU to GPU with BF16 conversion
void load_weights_to_gpu(GPUModelWeights& gpu_weights,
                        const TransformerWeights* cpu_weights,
                        const Config* config,
                        GPUMemoryManager& memory_mgr);

// Allocate run state buffers on GPU
void allocate_gpu_run_state(GPURunState& gpu_state,
                           const Config* config,
                           GPUMemoryManager& memory_mgr,
                           int batch_size);

// Free GPU memory
void free_gpu_memory(GPUModelWeights& gpu_weights,
                    GPURunState& gpu_state,
                    GPUMemoryManager& memory_mgr);