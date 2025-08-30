#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

// Helper functions for expert distribution across GPUs
__device__ __host__ inline bool is_expert_local(int expert_id, int gpu_id, int n_experts) {
    int n_local_experts = n_experts / 2;
    int expert_start_idx = (gpu_id % 2 == 0) ? 0 : n_local_experts;
    return (expert_id >= expert_start_idx) && (expert_id < expert_start_idx + n_local_experts);
}

__device__ __host__ inline int get_local_expert_start(int gpu_id, int n_experts) {
    int n_local_experts = n_experts / 2;
    return (gpu_id % 2 == 0) ? 0 : n_local_experts;
}

__device__ __host__ inline int get_local_expert_count(int n_experts) {
    return n_experts / 2;
}

__global__ void split_gate_up_kernel(float *gate, float *up, const float *mlp1_out,
                                     const __hip_bfloat16 *bias, int batch_size, int intermediate_dim)
{
    size_t idx = 1LL * blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / intermediate_dim;
    size_t dim_idx = idx % intermediate_dim;

    if (batch_idx >= batch_size || dim_idx >= intermediate_dim)
        return;

    size_t mlp1_idx = 1LL * batch_idx * 2 * intermediate_dim;
    // Convert bfloat16 bias to fp32 on-the-fly
    float bias_gate_fp32 = __bfloat162float(bias[2 * dim_idx]);
    float bias_up_fp32 = __bfloat162float(bias[2 * dim_idx + 1]);

    gate[idx] = mlp1_out[mlp1_idx + 2 * dim_idx] + bias_gate_fp32;
    up[idx] = mlp1_out[mlp1_idx + 2 * dim_idx + 1] + bias_up_fp32;
}



__global__ void topk_kernel(float *topk_values, int *topk_indices, const float *scores,
                            int batch_size, int n_experts, int k)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    const float *batch_scores = scores + 1LL * batch_idx * n_experts;
    float *batch_topk_v = topk_values + 1LL * batch_idx * k;
    int *batch_topk_i = topk_indices + 1LL * batch_idx * k;

    // Simple selection sort for top-k (works well for small k)
    for (int i = 0; i < k; i++)
    {
        float max_val = -INFINITY;
        int max_idx = -1;

        for (int j = 0; j < n_experts; j++)
        {
            bool already_selected = false;
            for (int prev = 0; prev < i; prev++)
            {
                if (batch_topk_i[prev] == j)
                {
                    already_selected = true;
                    break;
                }
            }

            if (!already_selected && batch_scores[j] > max_val)
            {
                max_val = batch_scores[j];
                max_idx = j;
            }
        }

        batch_topk_v[i] = max_val;
        batch_topk_i[i] = max_idx;
    }
}



// "Gather" kernel: Finds tokens for an expert and creates a compact list.
__global__ void gather_expert_inputs_kernel(const float *d_t, const int *topk_i, const float *topk_v,
                                            int expert_id, int batch_size, int hidden_dim, int experts_per_token,
                                            float *expert_input_buffer, int *expert_indices, float *expert_weights,
                                            int *d_batch_count)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size)
        return;

    // Check if this token 'b' selected the current 'expert_id'
    for (int k = 0; k < experts_per_token; ++k)
    {
        int topk_idx = b * experts_per_token + k;
        if (topk_i[topk_idx] == expert_id)
        {
            // This token is routed to this expert.
            // Atomically get a unique index for this token in the compact buffer.
            int compact_idx = atomicAdd(d_batch_count, 1);

            // Store the original batch index and weight for the scatter step.
            expert_indices[compact_idx] = b;
            expert_weights[compact_idx] = topk_v[topk_idx];

            // Copy the hidden state from d_t into the compact input buffer.
            // This is a strided copy, which is slow. For max performance,
            // a second kernel could re-format this into a dense matrix.
            // For logic matching, this is correct.
            const float *src = d_t + b * hidden_dim;
            float *dst = expert_input_buffer + compact_idx * hidden_dim;
            for (int i = 0; i < hidden_dim; ++i)
            {
                dst[i] = src[i];
            }
            // break; // Token found its expert, move to next token
        }
    }
}

// "Scatter" kernel: Adds the expert outputs back to the final aggregation buffer.
__global__ void scatter_expert_outputs_kernel(float *d_e_agg, const float *expert_output_buffer,
                                              const int *expert_indices, const float *expert_weights,
                                              int batch_count, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count * hidden_dim)
        return;

    int compact_idx = idx / hidden_dim;
    int dim = idx % hidden_dim;

    // Get the original batch index and the expert's router weight
    int original_batch_idx = expert_indices[compact_idx];
    float weight = expert_weights[compact_idx];

    // Calculate the destination address in the main aggregation buffer
    float *dst = d_e_agg + original_batch_idx * hidden_dim + dim;
    float value = expert_output_buffer[idx];

    // Atomically add the weighted result. This is crucial because multiple
    // experts (if experts_per_token > 1) write to the same d_e_agg location.
    atomicAdd(dst, value * weight);
}



// NEW: Improved MoE implementation with better expert routing
__global__ void expert_routing_kernel(float *expert_weights, const int *topk_indices,
                                      const float *topk_values, int batch_size,
                                      int experts_per_token, int n_experts)
{
    int batch_idx = blockIdx.x;
    int expert_slot = blockIdx.y;

    if (batch_idx >= batch_size || expert_slot >= experts_per_token)
        return;

    int expert_id = topk_indices[batch_idx * experts_per_token + expert_slot];
    float weight = topk_values[batch_idx * experts_per_token + expert_slot];

    expert_weights[batch_idx * n_experts + expert_id] = weight;
}

// NEW KERNEL for correct MoE aggregation
__global__ void aggregate_expert_output_kernel(float *d_e_agg, const float *expert_output,
                                               const int *topk_indices, const float *topk_values,
                                               int current_expert_id, int batch_size, int hidden_dim,
                                               int experts_per_token)
{
    // Each thread handles one dimension of one token in the batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_dim)
        return;

    int b = idx / hidden_dim; // Get the batch index for this thread

    // Check if the current expert (current_expert_id) was selected for this token (b)
    for (int k = 0; k < experts_per_token; ++k)
    {
        int expert_slot_idx = b * experts_per_token + k;
        if (topk_indices[expert_slot_idx] == current_expert_id)
        {
            // This token uses this expert. Add the weighted output to the aggregation buffer.
            float weight = topk_values[expert_slot_idx];
            d_e_agg[idx] += weight * expert_output[idx];

            // Since top-k indices are unique for a token, we can stop after finding the match
            break;
        }
    }
}

/**
 * @brief Stage 1: Counts the number of tokens assigned to each expert in parallel.
 *
 * This kernel launches one thread per token. Each thread iterates through its
 * top-k expert choices and atomically increments the counter for each chosen expert.
 * Contention is low as it's distributed across all expert counters.
 *
 * @param topk_i Device pointer to the top-k expert indices for each token.
 * @param d_expert_counts Device pointer to an array of size n_experts (pre-filled with zeros).
 * @param batch_size Total number of tokens.
 * @param experts_per_token The 'k' in top-k.
 */
__global__ void count_tokens_per_expert_kernel(const int *topk_i, int *d_expert_counts,
                                               int batch_size, int experts_per_token)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size)
    {
        return;
    }

    // Each token contributes to the count of its assigned experts
    for (int k = 0; k < experts_per_token; ++k)
    {
        int expert_id = topk_i[token_idx * experts_per_token + k];
        // This atomic is low-contention because updates are spread across n_experts counters
        atomicAdd(&d_expert_counts[expert_id], 1);
    }
}

/**
 * @brief Stage 2: Gathers/permutes expert inputs into a compact buffer using a
 * block-per-token strategy for coalesced memory access.
 *
 * This kernel is the high-performance replacement for the original gather_expert_inputs_kernel.
 * It uses a full thread block to process each token, allowing the `hidden_dim` vector
 * to be copied in a fully parallel and coalesced manner.
 *
 * @param d_t The source hidden states (batch_size, hidden_dim).
 * @param topk_i The top-k expert indices for each token.
 * @param topk_v The top-k expert weights for each token.
 * @param d_expert_offsets The starting index for each expert in the compact buffer.
 * @param d_expert_write_idx A temporary counter for each expert to get a local index.
 * @param batch_size Total number of tokens.
 * @param hidden_dim Dimension of the hidden state.
 * @param experts_per_token The 'k' in top-k.
 * @param expert_input_buffer Destination compact buffer for hidden states.
 * @param expert_indices Destination buffer for original token indices for scattering.
 * @param expert_weights Destination buffer for router weights for scattering.
 */
__global__ void permute_expert_inputs_kernel(const float *d_t, const int *topk_i, const float *topk_v,
                                             const int *d_expert_offsets, int *d_expert_write_idx,
                                             int batch_size, int hidden_dim, int experts_per_token,
                                             float *expert_input_buffer, int *expert_indices, float *expert_weights)
{
    // Each BLOCK processes one token to enable parallel copying
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size)
    {
        return;
    }

    // Use shared memory to communicate the calculated destination index to all threads in the block.
    // Allocate enough space for all experts_per_token entries
    extern __shared__ int destination_indices[];

    // The first few threads handle the logic for each of the token's expert choices
    if (threadIdx.x < experts_per_token)
    {
        int k = threadIdx.x;
        int topk_flat_idx = token_idx * experts_per_token + k;
        int expert_id = topk_i[topk_flat_idx];

        // Atomically get the local write position within this expert's designated data block
        int local_idx = atomicAdd(&d_expert_write_idx[expert_id], 1);

        // Calculate the final destination index in the large compact buffer
        int compact_idx = d_expert_offsets[expert_id] + local_idx;

        // Store metadata needed for the later scatter step
        expert_indices[compact_idx] = token_idx;
        expert_weights[compact_idx] = topk_v[topk_flat_idx];

        // Share the destination index with all threads in this block
        destination_indices[k] = compact_idx;
    }

    // Synchronize to ensure destination_indices is visible to all threads in the block
    __syncthreads();

    // Now, all threads in the block cooperate to copy the hidden state for each expert choice.
    // This loop ensures we handle all `experts_per_token` assignments for the current token.
    for (int k = 0; k < experts_per_token; ++k)
    {
        const float *src = d_t + token_idx * hidden_dim;
        float *dst = expert_input_buffer + destination_indices[k] * hidden_dim;

        // This is the coalesced copy: each thread copies a different element of the hidden_dim vector
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }
}

/**
 * @brief Partitions tokens from the gathered buffer into local and remote sets.
 * 
 * After initial gathering, this kernel separates tokens based on their assigned experts:
 * - Tokens for local experts stay on this GPU
 * - Tokens for remote experts are prepared for peer transfer
 * 
 * @param local_buffer Output buffer for tokens assigned to local experts
 * @param send_buffer Output buffer for tokens to send to peer GPU
 * @param local_indices Original batch indices for local tokens
 * @param send_indices Original batch indices for tokens to send
 * @param local_weights Router weights for local tokens
 * @param send_weights Router weights for tokens to send
 * @param gathered_input Input buffer from gather operation
 * @param gathered_indices Original indices from gather
 * @param gathered_weights Router weights from gather
 * @param expert_indices Expert IDs for each gathered token
 * @param d_local_write_idx Atomic counter for local buffer writes
 * @param d_send_write_idx Atomic counter for send buffer writes
 * @param total_tokens Total number of gathered tokens
 * @param hidden_dim Dimension of hidden states
 * @param n_local_experts Number of experts on this GPU
 * @param expert_start_idx First expert ID handled by this GPU
 */
__global__ void partition_tokens_kernel(
    float* local_buffer, float* send_buffer,
    int* local_indices, int* send_indices,
    float* local_weights, float* send_weights,
    const float* gathered_input, const int* gathered_indices,
    const float* gathered_weights, const int* expert_indices,
    int* d_local_write_idx, int* d_send_write_idx,
    int total_tokens, int hidden_dim, int n_local_experts, int expert_start_idx)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    int expert_id = expert_indices[token_idx];
    bool is_local = (expert_id >= expert_start_idx) && 
                   (expert_id < expert_start_idx + n_local_experts);
    
    // Get write position using atomic operation
    int write_pos;
    float* dest_buffer;
    int* dest_indices;
    float* dest_weights;
    
    if (is_local) {
        write_pos = atomicAdd(d_local_write_idx, 1);
        dest_buffer = local_buffer;
        dest_indices = local_indices;
        dest_weights = local_weights;
    } else {
        write_pos = atomicAdd(d_send_write_idx, 1);
        dest_buffer = send_buffer;
        dest_indices = send_indices;
        dest_weights = send_weights;
    }
    
    // Copy metadata
    if (threadIdx.x == 0) {
        dest_indices[write_pos] = gathered_indices[token_idx];
        dest_weights[write_pos] = gathered_weights[token_idx];
    }
    
    // Parallel copy of hidden state vector
    const float* src = gathered_input + token_idx * hidden_dim;
    float* dst = dest_buffer + write_pos * hidden_dim;
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/**
 * @brief Merges tokens received from peer GPU into the final input buffer.
 * 
 * This kernel combines locally partitioned tokens with those received from
 * the peer GPU, organizing them according to global expert offsets.
 * 
 * @param final_buffer Destination buffer for merged tokens
 * @param final_indices Destination for merged token indices
 * @param final_weights Destination for merged router weights
 * @param local_buffer Buffer containing local tokens
 * @param local_indices Indices of local tokens
 * @param local_weights Weights of local tokens
 * @param recv_buffer Buffer containing tokens from peer
 * @param recv_indices Indices of received tokens
 * @param recv_weights Weights of received tokens
 * @param d_expert_offsets Global offsets for each expert
 * @param d_expert_write_idx Write positions for each expert
 * @param local_count Number of local tokens
 * @param recv_count Number of received tokens
 * @param hidden_dim Dimension of hidden states
 * @param n_experts Total number of experts
 */
__global__ void merge_tokens_kernel(
    float* final_buffer, int* final_indices, float* final_weights,
    const float* local_buffer, const int* local_indices, const float* local_weights,
    const float* recv_buffer, const int* recv_indices, const float* recv_weights,
    const int* d_expert_offsets, int* d_expert_write_idx,
    int local_count, int recv_count, int hidden_dim, int n_experts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = local_count + recv_count;
    
    if (tid >= total_count) return;
    
    // Determine source buffer and index
    bool from_local = (tid < local_count);
    int src_idx = from_local ? tid : (tid - local_count);
    
    const float* src_buffer = from_local ? local_buffer : recv_buffer;
    const int* src_indices = from_local ? local_indices : recv_indices;
    const float* src_weights = from_local ? local_weights : recv_weights;
    
    // Copy token data
    const float* src = src_buffer + src_idx * hidden_dim;
    float* dst = final_buffer + tid * hidden_dim;
    
    for (int i = 0; i < hidden_dim; i++) {
        dst[i] = src[i];
    }
    
    // Copy metadata
    final_indices[tid] = src_indices[src_idx];
    final_weights[tid] = src_weights[src_idx];
}

/**
 * @brief Partitions expert outputs back to their originating GPUs.
 * 
 * After MLP processing, this kernel separates outputs:
 * - Outputs for tokens that originated locally stay on this GPU
 * - Outputs for tokens from the peer GPU are prepared for return transfer
 * 
 * @param local_output Buffer for outputs of locally-originated tokens
 * @param send_output Buffer for outputs to send back to peer
 * @param expert_output Output from expert MLPs
 * @param token_origins Flags indicating token origin (0=local, 1=peer)
 * @param original_indices Original batch indices for tokens
 * @param d_local_write_idx Atomic counter for local output writes
 * @param d_send_write_idx Atomic counter for send buffer writes
 * @param total_tokens Total number of processed tokens
 * @param hidden_dim Dimension of hidden states
 */
__global__ void partition_outputs_kernel(
    float* local_output, float* send_output,
    const float* expert_output, const bool* token_origins,
    const int* original_indices, int* local_indices, int* send_indices,
    int* d_local_write_idx, int* d_send_write_idx,
    int total_tokens, int hidden_dim)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    bool is_local_origin = !token_origins[token_idx];
    
    // Get write position
    int write_pos;
    float* dest_buffer;
    int* dest_indices;
    
    if (is_local_origin) {
        write_pos = atomicAdd(d_local_write_idx, 1);
        dest_buffer = local_output;
        dest_indices = local_indices;
    } else {
        write_pos = atomicAdd(d_send_write_idx, 1);
        dest_buffer = send_output;
        dest_indices = send_indices;
    }
    
    // Store original index for final scatter
    if (threadIdx.x == 0) {
        dest_indices[write_pos] = original_indices[token_idx];
    }
    
    // Parallel copy of output vector
    const float* src = expert_output + token_idx * hidden_dim;
    float* dst = dest_buffer + write_pos * hidden_dim;
    
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/**
 * @brief Merges outputs received back from peer GPU.
 * 
 * Final step before scatter: combines locally-processed outputs with
 * those returned from the peer GPU.
 * 
 * @param final_output Destination for all outputs
 * @param local_output Outputs from locally-processed tokens
 * @param recv_output Outputs received from peer
 * @param local_indices Indices for local outputs
 * @param recv_indices Indices for received outputs
 * @param final_indices Combined indices for scatter operation
 * @param local_count Number of local outputs
 * @param recv_count Number of received outputs
 * @param hidden_dim Dimension of hidden states
 */
__global__ void merge_outputs_kernel(
    float* final_output, int* final_indices,
    const float* local_output, const float* recv_output,
    const int* local_indices, const int* recv_indices,
    int local_count, int recv_count, int hidden_dim)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_count = local_count + recv_count;
    
    if (tid >= total_count) return;
    
    // Determine source
    bool from_local = (tid < local_count);
    int src_idx = from_local ? tid : (tid - local_count);
    
    const float* src_buffer = from_local ? local_output : recv_output;
    const int* src_indices = from_local ? local_indices : recv_indices;
    
    // Copy output data
    const float* src = src_buffer + src_idx * hidden_dim;
    float* dst = final_output + tid * hidden_dim;
    
    for (int i = 0; i < hidden_dim; i++) {
        dst[i] = src[i];
    }
    
    // Copy index for scatter
    final_indices[tid] = src_indices[src_idx];
}
