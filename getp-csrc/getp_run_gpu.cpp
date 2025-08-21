// GPU-optimized version of getp_run.cpp

#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include "../include/model.hpp"
#include "../include/gpu_utils.hpp"
#include "../include/gpu_kernels.hpp"

#ifndef GETP_RUN_GPU
#define GETP_RUN_GPU

// Global GPU structures
GPUWeights gpu_weights;
GPURunState gpu_state;
bool gpu_initialized = false;

// GPU cos/sin values for RoPE
float *gpu_cos_vals = nullptr;
float *gpu_sin_vals = nullptr;

void initialize_gpu_model(Transformer *transformer) {
    if (gpu_initialized) return;
    
    Config *p = &transformer->config;
    
    // Allocate GPU memory
    allocate_gpu_weights(&gpu_weights, p);
    allocate_gpu_runstate(&gpu_state, p);
    
    // Copy weights to GPU
    copy_weights_to_gpu(&gpu_weights, &transformer->weights, p);
    
    // Copy RoPE values to GPU
    CHECK_HIP(hipMalloc(&gpu_cos_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&gpu_sin_vals, (p->head_dim / 2) * p->seq_len * sizeof(float)));
    CHECK_HIP(hipMemcpy(gpu_cos_vals, cos_vals, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(gpu_sin_vals, sin_vals, (p->head_dim / 2) * p->seq_len * sizeof(float), hipMemcpyHostToDevice));
    
    gpu_initialized = true;
    printf("[GPU] Model initialized with BF16 precision\n");
}

float *forward_batch_getp_gpu(Transformer *transformer, RunState *s,
                              int *tokens, int batch_size) {
    Config *p = &transformer->config;
    
    // Initialize GPU if not done
    initialize_gpu_model(transformer);
    
    // Run forward pass on GPU
    bf16 *gpu_logits = forward_batch_gpu(&gpu_weights, &gpu_state, p, 
                                         tokens, batch_size, positions);
    
    // Copy logits back to CPU
    copy_logits_from_gpu(s->logits, gpu_logits, batch_size, p->vocab_size);
    
    return s->logits;
}

// Test function to compare GPU vs CPU output
void test_gpu_vs_cpu(Transformer *transformer, int *test_tokens, int batch_size) {
    Config *p = &transformer->config;
    RunState *s = &transformer->state;
    
    printf("\n=== GPU vs CPU Test ===\n");
    
    // Run CPU version
    float *cpu_logits = forward_batch_getp(transformer, s, test_tokens, batch_size);
    float cpu_first = cpu_logits[0];
    printf("CPU first logit: %f\n", cpu_first);
    
    // Run GPU version
    float *gpu_logits = forward_batch_getp_gpu(transformer, s, test_tokens, batch_size);
    float gpu_first = gpu_logits[0];
    printf("GPU first logit: %f\n", gpu_first);
    
    // Compare
    float diff = fabsf(cpu_first - gpu_first);
    float rel_error = diff / (fabsf(cpu_first) + 1e-6f);
    printf("Absolute difference: %e\n", diff);
    printf("Relative error: %.2f%%\n", rel_error * 100);
    
    if (rel_error < 0.05f) {  // 5% tolerance for BF16
        printf("✓ Test PASSED - Results match within tolerance\n");
    } else {
        printf("✗ Test FAILED - Results differ significantly\n");
    }
    printf("======================\n\n");
}

long long batched_generate_gpu(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, Requests *requests) {
    Config *p = &transformer->config;
    long long total_tokens_generated = 0;
    
    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE) {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                    ? (requests->num_reqs - req_start)
                                    : BATCH_SIZE;
        
        RunState *s = &transformer->state;
        
        // Initialize batch
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            
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
        
        // Run test on first batch
        if (req_start == 0) {
            test_gpu_vs_cpu(transformer, current_tokens, current_batch_size);
        }
        
        // Generation loop
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;
        for (int step = 0; step < max_steps && alive > 0; step++) {
            // Forward pass on GPU
            float *logits = forward_batch_getp_gpu(transformer, s,
                                                   current_tokens, current_batch_size);
            
            // Sample next tokens
            for (int b = 0; b < current_batch_size; b++) {
                if (finished[b]) continue;
                
                int req_idx = req_start + b;
                int pos = positions[b];
                float *logits_b = logits + b * p->vocab_size;
                
                int next_token;
                if (pos < prompt_lens[b] - 1) {
                    // Still processing prompt
                    next_token = prompt_tokens[b][pos + 1];
                } else {
                    // Generate new token
                    next_token = sample(sampler, logits_b);
                    
                    // Save generated token
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
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
                    int *output_tokens = get_tok_gen_ptr(requests, req_idx);
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
        }
        
        // Print results for the batch
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            int *output_tokens = get_tok_gen_ptr(requests, req_idx);
            
            // Print the original prompt string
            safe_printf(input_seq);
            printf("!");
            
            // Decode and print the newly generated tokens
            int last_prompt_token = prompt_tokens[b][prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0; ; ++i) {
                int token = output_tokens[i];
                if (token == -1) break;
                
                const char *piece = decode_piece(tokenizer, prev_token, token);
                safe_printf(piece);
                prev_token = token;
            }
            printf("\n");
            printf("[GPU DEBUG] Request %d: Generated %d tokens\n", 
                   req_idx, positions[b] - prompt_lens[b] + 1);
        }
        fflush(stdout);
    }
    
    return total_tokens_generated;
}

long long inference_gpu(Transformer *transformer, Tokenizer *tokenizer,
                       Sampler *sampler, Requests *requests) {
    return batched_generate_gpu(transformer, tokenizer, sampler, requests);
}

// Override the original inference function to use GPU
long long inference(Transformer *transformer, Tokenizer *tokenizer,
                   Sampler *sampler, Requests *requests) {
    printf("[INFO] Using GPU-accelerated inference with BF16\n");
    return inference_gpu(transformer, tokenizer, sampler, requests);
}

// Cleanup function
void cleanup_gpu() {
    if (gpu_initialized) {
        free_gpu_weights(&gpu_weights);
        free_gpu_runstate(&gpu_state);
        if (gpu_cos_vals) hipFree(gpu_cos_vals);
        if (gpu_sin_vals) hipFree(gpu_sin_vals);
        gpu_initialized = false;
    }
}

// Modified finish function
void finish(Transformer *transformer, Tokenizer *tokenizer) {
    cleanup_gpu();
    // Add any other cleanup here
}

#endif // GETP_RUN_GPU