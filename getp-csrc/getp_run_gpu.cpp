// GPU-accelerated version of getp_run.cpp for GPT-OSS 20B model
// This file integrates GPU kernels while maintaining compatibility with run.cpp interface

#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include "gpu_memory.hpp"
#include "gpu_kernels.hpp"
#include <cassert>
#include "../include/model.hpp"
#include <cstring>
#include <chrono>

#ifndef GETP_RUN_GPU
#define GETP_RUN_GPU

// Global GPU resources
static GPUMemoryManager* g_memory_mgr = nullptr;
static GPUModelWeights* g_gpu_weights = nullptr;
static GPURunState* g_gpu_state = nullptr;
static bool g_gpu_initialized = false;

// Initialize GPU resources (called once at startup)
void initialize_gpu_resources(Transformer* transformer, int batch_size) {
    if (g_gpu_initialized) return;
    
    printf("\n=== Initializing GPU Resources ===\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create memory manager
    g_memory_mgr = new GPUMemoryManager();
    
    // Allocate GPU structures
    g_gpu_weights = new GPUModelWeights();
    g_gpu_state = new GPURunState();
    
    // Initialize GPU memory
    initialize_gpu_memory(*g_gpu_weights, *g_gpu_state,
                         &transformer->config, *g_memory_mgr, batch_size);
    
    // Load weights to GPU (with BF16 conversion)
    printf("\n[GPU] Loading model weights from CPU to GPU...\n");
    load_weights_to_gpu(*g_gpu_weights, &transformer->weights,
                       &transformer->config, *g_memory_mgr);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("[GPU] Initialization complete in %.2f seconds\n\n", duration.count() / 1000.0);
    
    g_gpu_initialized = true;
}

// Cleanup GPU resources
void cleanup_gpu_resources() {
    if (!g_gpu_initialized) return;
    
    printf("\n[GPU] Cleaning up GPU resources...\n");
    
    if (g_memory_mgr && g_gpu_weights && g_gpu_state) {
        free_gpu_memory(*g_gpu_weights, *g_gpu_state, *g_memory_mgr);
    }
    
    delete g_gpu_state;
    delete g_gpu_weights;
    delete g_memory_mgr;
    
    g_gpu_initialized = false;
    printf("[GPU] Cleanup complete\n");
}

// GPU-accelerated forward pass
float* forward_batch_getp_gpu(Transformer* transformer, RunState* s,
                              int* tokens, int batch_size) {
    Config* p = &transformer->config;
    
    // Initialize GPU on first call
    if (!g_gpu_initialized) {
        initialize_gpu_resources(transformer, BATCH_SIZE);
    }
    
    // Verify batch size
    if (batch_size > g_gpu_state->batch_size) {
        fprintf(stderr, "Error: batch_size %d exceeds allocated size %d\n",
                batch_size, g_gpu_state->batch_size);
        return nullptr;
    }
    
    // Run GPU forward pass
    float* gpu_logits = gpu_forward_batch(*g_gpu_weights, *g_gpu_state,
                                         *g_memory_mgr, tokens, batch_size, p);
    
    // Copy logits back to CPU buffer
    size_t logits_size = batch_size * p->vocab_size * sizeof(float);
    g_memory_mgr->copy_device_to_host("logits", s->logits, batch_size * p->vocab_size);
    
    return s->logits;
}

// Fallback to CPU implementation with GPU acceleration where possible
float* forward_batch_getp_hybrid(Transformer* transformer, RunState* s,
                                 int* tokens, int batch_size) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    
    // Use GPU if available, otherwise fall back to CPU
    if (g_gpu_initialized) {
        return forward_batch_getp_gpu(transformer, s, tokens, batch_size);
    }
    
    // CPU fallback (original implementation)
    int hidden_dim = p->hidden_dim;
    
    // Copy token embeddings into x_batch
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float* x_b = s->x + b * hidden_dim;
        float* content_row = w->token_embedding_table + tokens[b] * hidden_dim;
        memcpy(x_b, content_row, hidden_dim * sizeof(float));
    }
    
    // Forward through all layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_batch_getp(transformer, s, l, batch_size);
        moe_batch_getp(transformer, s, l, batch_size);
    }
    
    rmsnorm_batch_getp(s->x, s->x, w->rms_out_w, batch_size, hidden_dim);
    
    // Classifier into logits
    matmul_batch_getp(s->logits, s->x, w->out, batch_size, hidden_dim, p->vocab_size);
    
    return s->logits;
}

// Main entry point - chooses between GPU and CPU implementation
float* forward_batch_getp(Transformer* transformer, RunState* s,
                         int* tokens, int batch_size) {
    // Check if GPU acceleration is available
    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) {
        // Use GPU implementation
        return forward_batch_getp_gpu(transformer, s, tokens, batch_size);
    } else {
        // Fall back to CPU implementation
        printf("[Warning] No GPUs found, using CPU implementation\n");
        return forward_batch_getp_hybrid(transformer, s, tokens, batch_size);
    }
}

// Batched generation with GPU acceleration
long long batched_generate(Transformer* transformer, Tokenizer* tokenizer,
                           Sampler* sampler, Requests* requests) {
    Config* p = &transformer->config;
    long long total_tokens_generated = 0;
    
    // Initialize GPU resources at the start
    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    if (gpu_count > 0 && !g_gpu_initialized) {
        initialize_gpu_resources(transformer, BATCH_SIZE);
    }
    
    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE) {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                    ? (requests->num_reqs - req_start)
                                    : BATCH_SIZE;
        
        RunState* s = &transformer->state;
        
        // Batch state arrays
        static int prompt_tokens[BATCH_SIZE][4096];
        static int prompt_lens[BATCH_SIZE];
        static int positions[BATCH_SIZE];
        static bool finished[BATCH_SIZE];
        static int current_tokens[BATCH_SIZE];
        
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char* input_seq = get_str_req_ptr(requests, req_idx);
            
            // Encode prompt
            encode(tokenizer, input_seq, 1, 0, prompt_tokens[b],
                   &prompt_lens[b], p->initial_context_length);
            
            if (prompt_lens[b] < 1) {
                fprintf(stderr, "Error: prompt too short for request %d\n", req_idx);
                prompt_lens[b] = 1;
                prompt_tokens[b][0] = 1; // BOS token
            }
            
            // Initialize sequence state
            positions[b] = 0;
            finished[b] = false;
            current_tokens[b] = prompt_tokens[b][0];
        }
        
        // Generation loop with GPU acceleration
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;
        
        auto gen_start = std::chrono::high_resolution_clock::now();
        int total_steps = 0;
        
        for (int step = 0; step < max_steps && alive > 0; step++) {
            // Forward pass (GPU or CPU)
            float* logits = forward_batch_getp(transformer, s,
                                              current_tokens, current_batch_size);
            
            // Sample next tokens (on CPU)
            for (int b = 0; b < current_batch_size; b++) {
                if (finished[b]) continue;
                
                int req_idx = req_start + b;
                int pos = positions[b];
                float* logits_b = logits + b * p->vocab_size;
                
                int next_token;
                if (pos < prompt_lens[b] - 1) {
                    // Still processing prompt
                    next_token = prompt_tokens[b][pos + 1];
                } else {
                    // Generate new token
                    next_token = sample(sampler, logits_b);
                    
                    // Save generated token
                    int* output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1);
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        output_tokens[gen_pos] = next_token;
                        total_tokens_generated++;
                    }
                }
                
                // Check for termination
                if (next_token == 1 || pos >= max_steps - 1) {
                    --alive;
                    finished[b] = true;
                    int* output_tokens = get_tok_gen_ptr(requests, req_idx);
                    int gen_pos = pos - (prompt_lens[b] - 1) + 1;
                    if (gen_pos >= 0 && gen_pos < requests->max_seq_len) {
                        output_tokens[gen_pos] = -1; // End marker
                    }
                    continue;
                }
                
                // Update for next iteration
                positions[b]++;
                current_tokens[b] = next_token;
            }
            total_steps++;
        }
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);
        float tokens_per_sec = (float)total_tokens_generated / (gen_duration.count() / 1000.0);
        
        // Print results
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char* input_seq = get_str_req_ptr(requests, req_idx);
            int* output_tokens = get_tok_gen_ptr(requests, req_idx);
            
            // Print prompt
            safe_printf(input_seq);
            printf("!");
            
            // Decode and print generated tokens
            int last_prompt_token = prompt_tokens[b][prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0; ; ++i) {
                int token = output_tokens[i];
                if (token == -1) break;
                
                const char* piece = decode_piece(tokenizer, prev_token, token);
                safe_printf(piece);
                prev_token = token;
            }
            printf("\n");
            printf("[GPU] Request %d: Generated %d tokens (%.1f tok/s)\n",
                   req_idx, positions[b] - prompt_lens[b] + 1, tokens_per_sec);
        }
        fflush(stdout);
    }
    
    return total_tokens_generated;
}

// Main inference function
long long inference(Transformer* transformer, Tokenizer* tokenizer,
                   Sampler* sampler, Requests* requests) {
    auto start = std::chrono::high_resolution_clock::now();
    
    long long tokens = batched_generate(transformer, tokenizer, sampler, requests);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("\n[GPU] Total inference time: %.2f seconds\n", duration.count() / 1000.0);
    printf("[GPU] Total tokens generated: %lld\n", tokens);
    printf("[GPU] Average throughput: %.2f tokens/second\n",
           tokens * 1000.0 / duration.count());
    
    // Cleanup GPU resources
    cleanup_gpu_resources();
    
    return tokens;
}

#endif // GETP_RUN_GPU