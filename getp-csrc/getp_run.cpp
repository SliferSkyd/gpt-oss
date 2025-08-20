// TODO: Modify this file to optimize end-to-end throughput


#include "../tokenizer.hpp"
#include "getp_eval.cpp"
#include <cassert>

#include "../model.hpp"


#ifndef GETP_RUN
#define GETP_RUN

float *cos_vals, *sin_vals;
int **prompt_tokens;
int *current_tokens;
bool *finished;
int *positions;
int *prompt_lens;
float *expert_input_buffer; // Buffer for MoE expert inputs

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head Adapted from
    // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
    // RoPE with YaRN scaling adapted from Python code

    Config *p = &transformer->config;
    RunState *s = &transformer->state;

    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    cos_vals =
        reinterpret_cast<float *>(malloc((p->head_dim / 2) * p->seq_len * sizeof(float)));
    sin_vals =
        reinterpret_cast<float *>(malloc((p->head_dim / 2) * p->seq_len * sizeof(float)));
    for (int pos = 0; pos < p->seq_len; ++pos)
        compute_cos_sin_getp(pos, p->rope_theta, p->head_dim, p->rope_scaling_factor,
                           p->initial_context_length, ntk_beta, ntk_alpha, cos_vals + (pos * p->head_dim / 2),
                           sin_vals + (pos * p->head_dim / 2));

    // re-allocate run state (more space for batching)
    malloc_batch_run_state(s, p);

    prompt_tokens = (int**)malloc(BATCH_SIZE * sizeof(int*));
    current_tokens = (int*)malloc(BATCH_SIZE * sizeof(int));

    for (int b = 0; b < BATCH_SIZE; b++) {
        prompt_tokens[b] = (int*)malloc((p->seq_len + 3) * sizeof(int));
    }

    finished = (bool*)malloc(BATCH_SIZE * sizeof(bool));
    positions = (int*)malloc(BATCH_SIZE * sizeof(int));
    prompt_lens = (int*)malloc(BATCH_SIZE * sizeof(int));
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
    // Do not inference here
    // You should handle the finish process
    // TODO:
    // - Memory deallocation
    // - Unload model
    // - ...
}




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

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
    // Inference here

    const char *empty_prompt = "";
    if (input_seq == NULL) {
        input_seq = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                       sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
           transformer->config.initial_context_length);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;                   // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                   // position in the sequence
    while (pos < steps) {

        // forward_getp the transformer to get logits for the next token
        float *logits = forward_getp(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt
            // token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
            // save the output token, it will be printed to file
            if(pos >= num_prompt_tokens)
                output_tokens[pos - num_prompt_tokens] = next;
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits
        // sequences
        if (next == 1) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        // should be removed
        const char *piece = decode_piece(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);

        token = next;
    }

    // should be removed
    printf("\n");

    // Marker for end of sequence
    output_tokens[pos - num_prompt_tokens] = -1;

    free(prompt_tokens);

    return pos - num_prompt_tokens;
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