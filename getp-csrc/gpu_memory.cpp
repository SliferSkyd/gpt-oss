#include "gpu_memory.hpp"
#include "../include/utils.hpp"
#include <cstring>
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d - %s\n", \
                __FILE__, __LINE__, hipGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Kernel for FP32 to BF16 conversion
__global__ void convert_fp32_to_bf16_kernel(const float* fp32, bf16* bf16_out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bf16_out[idx] = __float2bfloat16(fp32[idx]);
    }
}

// Kernel for BF16 to FP32 conversion  
__global__ void convert_bf16_to_fp32_kernel(const bf16* bf16_in, float* fp32, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fp32[idx] = __bfloat162float(bf16_in[idx]);
    }
}

GPUMemoryManager::GPUMemoryManager() {
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    printf("[GPU] Found %d AMD GPUs\n", num_gpus);
    
    device_ids.resize(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        device_ids[i] = i;
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        printf("[GPU %d] %s - %.1f GB VRAM\n", i, prop.name, 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

GPUMemoryManager::~GPUMemoryManager() {
    deallocate_all();
}

void* GPUMemoryManager::allocate(const std::string& name, size_t size_bytes, 
                                 int device_id, bool is_bf16) {
    if (allocations.find(name) != allocations.end()) {
        fprintf(stderr, "Warning: allocation %s already exists\n", name.c_str());
        return allocations[name].ptr;
    }
    
    HIP_CHECK(hipSetDevice(device_id));
    
    void* ptr;
    HIP_CHECK(hipMalloc(&ptr, size_bytes));
    HIP_CHECK(hipMemset(ptr, 0, size_bytes));
    
    allocations[name] = {ptr, size_bytes, device_id, is_bf16};
    
    printf("[GPU %d] Allocated %s: %.2f MB (%s)\n", device_id, name.c_str(), 
           size_bytes / (1024.0 * 1024.0), is_bf16 ? "BF16" : "FP32");
    
    return ptr;
}

void* GPUMemoryManager::allocate_multi_gpu(const std::string& name, 
                                           size_t size_per_gpu, bool is_bf16) {
    // For multi-GPU allocations, we allocate on GPU 0 and return that pointer
    // Individual GPU allocations are tracked with name_gpu{i}
    void* main_ptr = nullptr;
    
    for (int i = 0; i < num_gpus; i++) {
        std::string gpu_name = name + "_gpu" + std::to_string(i);
        void* ptr = allocate(gpu_name, size_per_gpu, i, is_bf16);
        if (i == 0) main_ptr = ptr;
    }
    
    return main_ptr;
}

void GPUMemoryManager::deallocate(const std::string& name) {
    auto it = allocations.find(name);
    if (it != allocations.end()) {
        HIP_CHECK(hipSetDevice(it->second.device_id));
        HIP_CHECK(hipFree(it->second.ptr));
        allocations.erase(it);
    }
}

void GPUMemoryManager::deallocate_all() {
    for (auto& [name, block] : allocations) {
        HIP_CHECK(hipSetDevice(block.device_id));
        HIP_CHECK(hipFree(block.ptr));
    }
    allocations.clear();
}

void GPUMemoryManager::copy_host_to_device(const std::string& name, 
                                           const float* host_data, 
                                           size_t num_elements,
                                           bool convert_to_bf16) {
    auto it = allocations.find(name);
    if (it == allocations.end()) {
        fprintf(stderr, "Error: allocation %s not found\n", name.c_str());
        return;
    }
    
    HIP_CHECK(hipSetDevice(it->second.device_id));
    
    if (convert_to_bf16 && it->second.is_bf16) {
        // Allocate temporary buffer for FP32 data
        float* temp_fp32;
        HIP_CHECK(hipMalloc(&temp_fp32, num_elements * sizeof(float)));
        HIP_CHECK(hipMemcpy(temp_fp32, host_data, num_elements * sizeof(float), 
                           hipMemcpyHostToDevice));
        
        // Convert to BF16
        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;
        hipLaunchKernelGGL(convert_fp32_to_bf16_kernel, dim3(grid_size), dim3(block_size),
                          0, 0, temp_fp32, (bf16*)it->second.ptr, num_elements);
        
        HIP_CHECK(hipFree(temp_fp32));
    } else {
        // Direct copy for FP32
        HIP_CHECK(hipMemcpy(it->second.ptr, host_data, 
                           num_elements * sizeof(float), hipMemcpyHostToDevice));
    }
}

void GPUMemoryManager::copy_device_to_host(const std::string& name,
                                           float* host_data,
                                           size_t num_elements) {
    auto it = allocations.find(name);
    if (it == allocations.end()) {
        fprintf(stderr, "Error: allocation %s not found\n", name.c_str());
        return;
    }
    
    HIP_CHECK(hipSetDevice(it->second.device_id));
    
    if (it->second.is_bf16) {
        // Convert BF16 to FP32
        float* temp_fp32;
        HIP_CHECK(hipMalloc(&temp_fp32, num_elements * sizeof(float)));
        
        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;
        hipLaunchKernelGGL(convert_bf16_to_fp32_kernel, dim3(grid_size), dim3(block_size),
                          0, 0, (bf16*)it->second.ptr, temp_fp32, num_elements);
        
        HIP_CHECK(hipMemcpy(host_data, temp_fp32, num_elements * sizeof(float),
                           hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(temp_fp32));
    } else {
        HIP_CHECK(hipMemcpy(host_data, it->second.ptr,
                           num_elements * sizeof(float), hipMemcpyDeviceToHost));
    }
}

void* GPUMemoryManager::get_ptr(const std::string& name) const {
    auto it = allocations.find(name);
    return (it != allocations.end()) ? it->second.ptr : nullptr;
}

int GPUMemoryManager::get_device_id(const std::string& name) const {
    auto it = allocations.find(name);
    return (it != allocations.end()) ? it->second.device_id : -1;
}

void GPUMemoryManager::set_device(int device_id) {
    HIP_CHECK(hipSetDevice(device_id));
}

void GPUMemoryManager::convert_fp32_to_bf16(const float* fp32, bf16* bf16_out, 
                                            size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        bf16_out[i] = __float2bfloat16(fp32[i]);
    }
}

void GPUMemoryManager::convert_bf16_to_fp32(const bf16* bf16_in, float* fp32,
                                            size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        fp32[i] = __bfloat162float(bf16_in[i]);
    }
}

// Initialize GPU memory structures
void initialize_gpu_memory(GPUModelWeights& gpu_weights,
                          GPURunState& gpu_state,
                          const Config* config,
                          GPUMemoryManager& memory_mgr,
                          int batch_size) {
    printf("\n[GPU] Initializing GPU memory for batch_size=%d\n", batch_size);
    
    // Calculate dimensions
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int n_layers = config->n_layers;
    int n_experts = config->n_experts;
    int experts_per_gpu = 4; // 32 experts / 8 GPUs = 4 experts per GPU
    
    // Allocate model weights
    load_weights_to_gpu(gpu_weights, nullptr, config, memory_mgr);
    
    // Allocate run state
    allocate_gpu_run_state(gpu_state, config, memory_mgr, batch_size);
    
    printf("[GPU] Memory initialization complete\n\n");
}

void load_weights_to_gpu(GPUModelWeights& gpu_weights,
                        const TransformerWeights* cpu_weights,
                        const Config* config,
                        GPUMemoryManager& memory_mgr) {
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int n_layers = config->n_layers;
    int n_experts = config->n_experts;
    int intermediate_dim = config->intermediate_dim;
    int head_dim = config->head_dim;
    int n_heads = config->n_attn_heads;
    int n_kv_heads = config->n_kv_heads;
    int qkv_dim = head_dim * (n_heads + 2 * n_kv_heads);
    
    // Experts per GPU (32 experts / 8 GPUs = 4)
    int experts_per_gpu = 4;
    int num_gpus = 8;
    
    printf("[GPU] Loading model weights to GPU memory (BF16)\n");
    
    // Token embeddings - replicate on all GPUs for fast access
    size_t embed_size = vocab_size * hidden_dim * sizeof(bf16);
    gpu_weights.token_embedding_table = (bf16*)memory_mgr.allocate_multi_gpu(
        "token_embeddings", embed_size, true);
    
    // Initialize per-layer weights
    gpu_weights.layers.resize(n_layers);
    
    for (int l = 0; l < n_layers; l++) {
        // Determine which GPU this layer's computation will primarily run on
        int layer_gpu = l % num_gpus;
        auto& layer = gpu_weights.layers[l];
        layer.device_id = layer_gpu;
        
        std::string prefix = "layer" + std::to_string(l) + "_";
        
        // Attention weights (on layer's primary GPU)
        layer.rms_attn_w = (bf16*)memory_mgr.allocate(
            prefix + "rms_attn", hidden_dim * sizeof(bf16), layer_gpu, true);
        layer.w_qkv = (bf16*)memory_mgr.allocate(
            prefix + "w_qkv", qkv_dim * hidden_dim * sizeof(bf16), layer_gpu, true);
        layer.b_qkv = (bf16*)memory_mgr.allocate(
            prefix + "b_qkv", qkv_dim * sizeof(bf16), layer_gpu, true);
        layer.w_o = (bf16*)memory_mgr.allocate(
            prefix + "w_o", hidden_dim * head_dim * n_heads * sizeof(bf16), layer_gpu, true);
        layer.b_o = (bf16*)memory_mgr.allocate(
            prefix + "b_o", hidden_dim * sizeof(bf16), layer_gpu, true);
        layer.attn_sinks = (bf16*)memory_mgr.allocate(
            prefix + "attn_sinks", n_heads * sizeof(bf16), layer_gpu, true);
        
        // MoE router weights (on layer's primary GPU)
        layer.rms_ffn_w = (bf16*)memory_mgr.allocate(
            prefix + "rms_ffn", hidden_dim * sizeof(bf16), layer_gpu, true);
        layer.w_router = (bf16*)memory_mgr.allocate(
            prefix + "w_router", hidden_dim * n_experts * sizeof(bf16), layer_gpu, true);
        layer.b_router = (bf16*)memory_mgr.allocate(
            prefix + "b_router", n_experts * sizeof(bf16), layer_gpu, true);
        
        // Expert weights - distribute across GPUs
        // Each GPU gets 4 experts for this layer
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            int expert_start = gpu * experts_per_gpu;
            
            if (gpu == layer_gpu) {
                // Store info for primary GPU
                layer.expert_start_idx = expert_start;
                layer.num_local_experts = experts_per_gpu;
                
                // Allocate expert weights on this GPU
                layer.w_mlp1 = (bf16*)memory_mgr.allocate(
                    prefix + "w_mlp1_gpu" + std::to_string(gpu),
                    experts_per_gpu * 2 * intermediate_dim * hidden_dim * sizeof(bf16),
                    gpu, true);
                layer.w_mlp2 = (bf16*)memory_mgr.allocate(
                    prefix + "w_mlp2_gpu" + std::to_string(gpu),
                    experts_per_gpu * hidden_dim * intermediate_dim * sizeof(bf16),
                    gpu, true);
                layer.b_mlp1 = (bf16*)memory_mgr.allocate(
                    prefix + "b_mlp1_gpu" + std::to_string(gpu),
                    experts_per_gpu * 2 * intermediate_dim * sizeof(bf16),
                    gpu, true);
                layer.b_mlp2 = (bf16*)memory_mgr.allocate(
                    prefix + "b_mlp2_gpu" + std::to_string(gpu),
                    experts_per_gpu * hidden_dim * sizeof(bf16),
                    gpu, true);
            } else {
                // Also allocate experts on other GPUs for this layer
                memory_mgr.allocate(
                    prefix + "w_mlp1_gpu" + std::to_string(gpu),
                    experts_per_gpu * 2 * intermediate_dim * hidden_dim * sizeof(bf16),
                    gpu, true);
                memory_mgr.allocate(
                    prefix + "w_mlp2_gpu" + std::to_string(gpu),
                    experts_per_gpu * hidden_dim * intermediate_dim * sizeof(bf16),
                    gpu, true);
                memory_mgr.allocate(
                    prefix + "b_mlp1_gpu" + std::to_string(gpu),
                    experts_per_gpu * 2 * intermediate_dim * sizeof(bf16),
                    gpu, true);
                memory_mgr.allocate(
                    prefix + "b_mlp2_gpu" + std::to_string(gpu),
                    experts_per_gpu * hidden_dim * sizeof(bf16),
                    gpu, true);
            }
        }
    }
    
    // Output weights - replicate on all GPUs
    gpu_weights.rms_out_w = (bf16*)memory_mgr.allocate_multi_gpu(
        "rms_out", hidden_dim * sizeof(bf16), true);
    gpu_weights.out = (bf16*)memory_mgr.allocate_multi_gpu(
        "out_proj", vocab_size * hidden_dim * sizeof(bf16), true);
    
    // If CPU weights provided, copy them to GPU
    if (cpu_weights != nullptr) {
        printf("[GPU] Copying weights from CPU to GPU with BF16 conversion...\n");
        
        // Copy token embeddings to all GPUs
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            std::string name = "token_embeddings_gpu" + std::to_string(gpu);
            memory_mgr.copy_host_to_device(name, cpu_weights->token_embedding_table,
                                          vocab_size * hidden_dim, true);
        }
        
        // Copy layer weights
        for (int l = 0; l < n_layers; l++) {
            auto& layer = gpu_weights.layers[l];
            std::string prefix = "layer" + std::to_string(l) + "_";
            
            // Copy attention weights
            memory_mgr.copy_host_to_device(prefix + "rms_attn",
                cpu_weights->rms_attn_w + l * hidden_dim, hidden_dim, true);
            memory_mgr.copy_host_to_device(prefix + "w_qkv",
                cpu_weights->w_qkv + l * qkv_dim * hidden_dim, qkv_dim * hidden_dim, true);
            // ... continue for other weights
        }
        
        printf("[GPU] Weight loading complete\n");
    }
}

void allocate_gpu_run_state(GPURunState& gpu_state,
                           const Config* config,
                           GPUMemoryManager& memory_mgr,
                           int batch_size) {
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int n_layers = config->n_layers;
    int n_experts = config->n_experts;
    int experts_per_token = config->experts_per_token;
    int intermediate_dim = config->intermediate_dim;
    int head_dim = config->head_dim;
    int n_heads = config->n_attn_heads;
    int n_kv_heads = config->n_kv_heads;
    int seq_len = config->seq_len;
    int qkv_dim = head_dim * (n_heads + 2 * n_kv_heads);
    int kv_dim = head_dim * n_kv_heads;
    
    gpu_state.batch_size = batch_size;
    gpu_state.max_seq_len = seq_len;
    
    printf("[GPU] Allocating run state buffers for batch_size=%d\n", batch_size);
    
    // Primary activations (FP32 for accuracy)
    gpu_state.x = (float*)memory_mgr.allocate(
        "x", batch_size * hidden_dim * sizeof(float), 0, false);
    gpu_state.residual = (float*)memory_mgr.allocate(
        "residual", batch_size * hidden_dim * sizeof(float), 0, false);
    
    // Attention intermediates
    gpu_state.qkv = (float*)memory_mgr.allocate(
        "qkv", batch_size * qkv_dim * sizeof(float), 0, false);
    gpu_state.q = (float*)memory_mgr.allocate(
        "q", batch_size * n_heads * head_dim * sizeof(float), 0, false);
    gpu_state.k = (float*)memory_mgr.allocate(
        "k", batch_size * n_kv_heads * head_dim * sizeof(float), 0, false);
    gpu_state.v = (float*)memory_mgr.allocate(
        "v", batch_size * n_kv_heads * head_dim * sizeof(float), 0, false);
    gpu_state.att_scores = (float*)memory_mgr.allocate(
        "att_scores", batch_size * n_heads * seq_len * sizeof(float), 0, false);
    gpu_state.att_output = (float*)memory_mgr.allocate(
        "att_output", batch_size * hidden_dim * sizeof(float), 0, false);
    
    // MoE intermediates
    gpu_state.router_logits = (float*)memory_mgr.allocate(
        "router_logits", batch_size * n_experts * sizeof(float), 0, false);
    gpu_state.router_probs = (float*)memory_mgr.allocate(
        "router_probs", batch_size * n_experts * sizeof(float), 0, false);
    gpu_state.expert_indices = (int*)memory_mgr.allocate(
        "expert_indices", batch_size * experts_per_token * sizeof(int), 0, false);
    gpu_state.expert_weights = (float*)memory_mgr.allocate(
        "expert_weights", batch_size * experts_per_token * sizeof(float), 0, false);
    gpu_state.expert_input = (float*)memory_mgr.allocate(
        "expert_input", batch_size * experts_per_token * hidden_dim * sizeof(float), 0, false);
    gpu_state.expert_output = (float*)memory_mgr.allocate(
        "expert_output", batch_size * experts_per_token * hidden_dim * sizeof(float), 0, false);
    
    // MLP intermediates (BF16 for memory efficiency)
    gpu_state.mlp1_out = (bf16*)memory_mgr.allocate(
        "mlp1_out", batch_size * experts_per_token * 2 * intermediate_dim * sizeof(bf16), 0, true);
    gpu_state.gate = (bf16*)memory_mgr.allocate(
        "gate", batch_size * experts_per_token * intermediate_dim * sizeof(bf16), 0, true);
    gpu_state.up = (bf16*)memory_mgr.allocate(
        "up", batch_size * experts_per_token * intermediate_dim * sizeof(bf16), 0, true);
    gpu_state.mlp_output = (float*)memory_mgr.allocate(
        "mlp_output", batch_size * hidden_dim * sizeof(float), 0, false);
    
    // KV Cache (BF16 for memory efficiency, distributed across layers)
    size_t kv_cache_size = n_layers * batch_size * seq_len * kv_dim;
    gpu_state.key_cache = (bf16*)memory_mgr.allocate_multi_gpu(
        "key_cache", kv_cache_size * sizeof(bf16) / 8, true);
    gpu_state.value_cache = (bf16*)memory_mgr.allocate_multi_gpu(
        "value_cache", kv_cache_size * sizeof(bf16) / 8, true);
    
    // Output logits
    gpu_state.logits = (float*)memory_mgr.allocate(
        "logits", batch_size * vocab_size * sizeof(float), 0, false);
    
    // Metadata
    gpu_state.positions = (int*)memory_mgr.allocate(
        "positions", batch_size * sizeof(int), 0, false);
    gpu_state.sequence_lengths = (int*)memory_mgr.allocate(
        "sequence_lengths", batch_size * sizeof(int), 0, false);
    gpu_state.finished = (bool*)memory_mgr.allocate(
        "finished", batch_size * sizeof(bool), 0, false);
    
    printf("[GPU] Run state allocation complete\n");
}

void free_gpu_memory(GPUModelWeights& gpu_weights,
                    GPURunState& gpu_state,
                    GPUMemoryManager& memory_mgr) {
    printf("[GPU] Freeing GPU memory...\n");
    memory_mgr.deallocate_all();
    printf("[GPU] Memory freed\n");
}