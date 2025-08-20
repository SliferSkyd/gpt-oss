// TODO: Modify this file to optimize end-to-end throughput


#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>
#include "../include/model.hpp"


#ifndef GETP_RUN
#define GETP_RUN




float *forward_batch_getp(Transformer *transformer, RunState *s,
                          int *tokens, int batch_size) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;

    int hidden_dim = p->hidden_dim;

    // Copy token embeddings into x_batch
#pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        float *x_b = s->x + b * hidden_dim;
        float *content_row = w->token_embedding_table + tokens[b] * hidden_dim;
        memcpy(x_b, content_row, hidden_dim * sizeof(float));
    }

    // Forward through all layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        attention_batch_getp(transformer, s, l, batch_size);
        moe_batch_getp(transformer, s, l, batch_size);
    }
    rmsnorm_batch_getp(s->x, s->x,
                       w->rms_out_w, batch_size, hidden_dim);

    // Classifier into logits
    matmul_batch_getp(s->logits, s->x, w->out, batch_size, hidden_dim, p->vocab_size);

    return s->logits;
}

long long batched_generate(Transformer *transformer, Tokenizer *tokenizer,
                           Sampler *sampler, Requests *requests) {
    Config *p = &transformer->config;
    long long total_tokens_generated = 0;

    // Process requests in batches
    for (int req_start = 0; req_start < requests->num_reqs; req_start += BATCH_SIZE) {
        int current_batch_size = (req_start + BATCH_SIZE > requests->num_reqs)
                                     ? (requests->num_reqs - req_start)
                                     : BATCH_SIZE;

        RunState *s = &transformer->state;

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

        // Generation loop
        int max_steps = requests->max_seq_len;
        int alive = current_batch_size;
        for (int step = 0; step < max_steps && alive > 0; step++) {
            // Forward pass for active sequences
            float *logits = forward_batch_getp(transformer, s,
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
                if (next_token == 1 || pos >= max_steps - 1) { // BOS token or max length
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

        // After the generation loop for the batch is complete, print everything.
        for (int b = 0; b < current_batch_size; b++) {
            int req_idx = req_start + b;
            const char *input_seq = get_str_req_ptr(requests, req_idx);
            int *output_tokens = get_tok_gen_ptr(requests, req_idx);

            // 1. Print the original prompt string
            safe_printf(input_seq);
            printf("!");
            // 2. Decode and print the newly generated tokens
            int last_prompt_token = prompt_tokens[b][prompt_lens[b] - 1];
            int prev_token = last_prompt_token;
            for (int i = 0; ; ++i) {
                int token = output_tokens[i];
                if (token == -1) break; // Stop at our end-of-sequence marker

                const char *piece = decode_piece(tokenizer, prev_token, token);
                safe_printf(piece);
                prev_token = token;
            }
            printf("\n");
            printf("[DEBUG] Request %d: Generated %d tokens\n", req_idx, positions[b] - prompt_lens[b] + 1);
        }
        // Flush stdout once after printing all results for the batch
        fflush(stdout);
    }

    return total_tokens_generated;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
    return batched_generate(transformer, tokenizer, sampler, requests);
}

#endif // GETP_RUN