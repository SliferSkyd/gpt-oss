#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"

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

// ============ helpers: map a block-idx (m-tile) -> expert ============

__device__ __forceinline__ int map_tile_to_expert(int tileIdx,
                                                  const int* __restrict__ mtile_prefix,
                                                  int n_experts) {
    // binary search on prefix (mtile_prefix[0]=0, ..., [n_experts]=total)
    int lo = 0, hi = n_experts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (mtile_prefix[mid+1] <= tileIdx) lo = mid + 1;
        else hi = mid;
    }
    return lo; // expert id
}

// Variant of store that adds bf16 bias (same col bias for 4 rows)
template<bool InteriorStore>
__device__ inline void store_c_tile_addbias(
    float* __restrict__ C, const f32x4& acc,
    const __hip_bfloat16* __restrict__ bias,
    int M, int N, int m0, int n0, int wave_m, int wave_n, int lane)
{
    const int rowBase = m0 + wave_m * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n * WN + lane_row(lane);

    float b = 0.f;
    if constexpr (InteriorStore) {
        b = __bfloat162float(bias[col]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            C[(size_t)(rowBase + i) * N + col] = acc[i] + b;
        }
    } else {
        if (col < N) {
            b = __bfloat162float(bias[col]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = rowBase + i;
                if (row < M) C[(size_t)row * N + col] = acc[i] + b;
            }
        }
    }
}

// ==== GROUPED MLP1 (MXFP4) across all experts: FIXED LAYER-RELATIVE DEQUANT ====
__global__ void grouped_mlp1_mxfp4_kernel(
    float* __restrict__ C,                     // mlp1_out [sum_tokens, 2*D]
    const float* __restrict__ A,               // expert_input_buffer [sum_tokens, H]
    const uint8_t* __restrict__ Wp_layer,      // layer base (packed)
    const float* __restrict__ Sc_layer,        // layer base (scales)
    const int* __restrict__ expert_offsets,    // [n_experts]
    const int* __restrict__ expert_counts,     // [n_experts]
    const int* __restrict__ mtile_prefix,      // [n_experts+1]
    int n_experts,
    int K,                                     // H
    int N,                                     // 2*D
    size_t seg_elems,                          // per-expert elems in W (N*K) -- used for indexing
    size_t /*seg_packed_bytes*/,               // kept for ABI; unused here
    size_t /*seg_blocks*/                      // kept for ABI; unused here
){
    // which expert does this m-tile belong to?
    const int mTileGlobal = blockIdx.y;
    const int e = map_tile_to_expert(mTileGlobal, mtile_prefix, n_experts);
    const int mTileLocal  = mTileGlobal - mtile_prefix[e];

    const int M_e = expert_counts[e];
    if (M_e == 0) return;

    // tile origins
    const int m0 = mTileLocal * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    uint16_t* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    uint16_t* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    uint16_t* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // per-expert row/col bases into the *layer* tensor
    // layer is laid out as [n_experts, N, K] flattened, so expert e starts at:
    const size_t e_elem_base = (size_t)e * seg_elems;            // element offset
    const size_t layer_elems = (size_t)n_experts * seg_elems;    // handy for bounds in dequant

    // slice pointers for A and C for this expert
    const float* A_e = A + (size_t)expert_offsets[e] * K;
    float*       C_e = C + (size_t)expert_offsets[e] * N;

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // --- preload k-slice 0 (double-buffered) ---
    {
        // A: [M_e, K] -> sA0 [BLOCK_M x BLOCK_K]
        for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
            const int r  = idx / BLOCK_K;         // row in tile
            const int ck = idx % BLOCK_K;         // k within slice
            const int gm = m0 + r;
            const int gk = ck;
            float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
            sA0[r * BLOCK_K + ck] = f32_to_bf16_bits(a);
        }
        // B: dequant from layer base using layer-relative element indices
        // store as sB0 [BLOCK_N x BLOCK_K] (note the col-major-ish packing used by mfma path)
        for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
            const int cN = idx / BLOCK_K;         // N-col within the tile
            const int rK = idx % BLOCK_K;         // K within the slice
            const int gn = n0 + cN;
            const int gk = rK;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp_layer, Sc_layer,
                                   e_elem_base + (size_t)gn * K + gk, layer_elems)
                : 0.f;
            sB0[cN * BLOCK_K + rK] = f32_to_bf16_bits(wb);
        }
    }
    __syncthreads();

    uint16_t* currA = sA0; uint16_t* nextA = sA1;
    uint16_t* currB = sB0; uint16_t* nextB = sB1;

    // --- main K loop (double buffering on shared mem) ---
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;

            // preload next A-slice
            for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
                const int r  = idx / BLOCK_K;
                const int ck = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + ck;
                float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
                nextA[r * BLOCK_K + ck] = f32_to_bf16_bits(a);
            }
            // preload next B-slice from layer base
            for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
                const int cN = idx / BLOCK_K;
                const int rK = idx % BLOCK_K;
                const int gn = n0 + cN;
                const int gk = kBase + rK;
                float wb = (gk < K && gn < N)
                    ? dequantize_mxfp4(Wp_layer, Sc_layer,
                                       e_elem_base + (size_t)gn * K + gk, layer_elems)
                    : 0.f;
                nextB[cN * BLOCK_K + rK] = f32_to_bf16_bits(wb);
            }
        }

        // MFMA on current tiles
        const int ldA = BLOCK_K, ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;
        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads(); // before swapping the buffers
        uint16_t* tA = currA; currA = nextA; nextA = tA;
        uint16_t* tB = currB; currB = nextB; nextB = tB;
    }

    // store C (guarded for edges)
    const bool interior = (m0 + BLOCK_M) <= M_e && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_mfma<true>(C_e, acc, M_e, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C_e, acc, M_e, N, m0, n0, wave_m, wave_n, lane);
    }
}


// ============ fused bias+SiLU(gate)*up over concatenated tokens ============
__global__ void bias_swiglu_epilogue_kernel(
    const float* __restrict__ mlp1_out,     // [sum_tokens, 2*D]
    const __hip_bfloat16* __restrict__ b1,  // layer base, per-expert segments [n_experts, 2*D]
    const int* __restrict__ expert_offsets, // [n_experts]
    const int* __restrict__ expert_counts,  // [n_experts]
    int n_experts,
    float* __restrict__ gate_up,            // [sum_tokens, D]
    int D,                                  // intermediate_dim
    int sum_tokens,
    float clamp_limit,
    float alpha_silu = 1.702f)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sum_tokens * D;
    if (idx >= total) return;

    int t = idx / D;     // token id in concatenated buffer
    int d = idx % D;

    // map token -> expert (binary search over offsets)
    int lo = 0, hi = n_experts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int start = expert_offsets[mid];
        int end   = start + expert_counts[mid];
        if (t >= end) lo = mid + 1;
        else if (t < start) hi = mid;
        else { lo = mid; break; }
    }
    int e = lo;

    const int t_local = t - expert_offsets[e];
    const float bg = __bfloat162float(b1[(size_t)e * (2*D) + 2*d + 0]);
    const float bu = __bfloat162float(b1[(size_t)e * (2*D) + 2*d + 1]);

    float g = mlp1_out[(size_t)t * (2*D) + (2*d + 0)] + bg;
    float u = mlp1_out[(size_t)t * (2*D) + (2*d + 1)] + bu;

    // fast, stable-ish SiLU approximation with clamp
    g = fminf(fmaxf(g, -clamp_limit), clamp_limit);
    u = fminf(fmaxf(u, -clamp_limit), clamp_limit);

    g *= (1.0f / (1.0f + expf(-alpha_silu * g)));
    g *= (u + 1.0f);
    gate_up[(size_t)t * D + d] = g;
}
// ==== GROUPED MLP2 (MXFP4) across all experts + bias: FIXED LAYER-RELATIVE DEQUANT ====
__global__ void grouped_mlp2_mxfp4_bias_kernel(
    float* __restrict__ C,                      // expert_output_buffer [sum_tokens, H]
    const float* __restrict__ A,                // gate_up [sum_tokens, D]
    const uint8_t* __restrict__ Wp_layer,       // layer base (packed)
    const float* __restrict__ Sc_layer,         // layer base (scales)
    const __hip_bfloat16* __restrict__ b2_layer,// bias per expert [n_experts, H]
    const int* __restrict__ expert_offsets,     // [n_experts]
    const int* __restrict__ expert_counts,      // [n_experts]
    const int* __restrict__ mtile_prefix,       // [n_experts+1]
    int n_experts,
    int K,                                      // D
    int N,                                      // H
    size_t seg_elems,                           // per-expert elems in W (N*K)
    size_t /*seg_packed_bytes*/,                // kept for ABI; unused here
    size_t /*seg_blocks*/                       // kept for ABI; unused here
){
    const int mTileGlobal = blockIdx.y;
    const int e = map_tile_to_expert(mTileGlobal, mtile_prefix, n_experts);
    const int mTileLocal  = mTileGlobal - mtile_prefix[e];

    const int M_e = expert_counts[e];
    if (M_e == 0) return;

    const int m0 = mTileLocal * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    uint16_t* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    uint16_t* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    uint16_t* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // layer-relative bases for this expert
    const size_t e_elem_base = (size_t)e * seg_elems;
    const size_t layer_elems = (size_t)n_experts * seg_elems;

    const float*        A_e  = A + (size_t)expert_offsets[e] * K;
    float*              C_e  = C + (size_t)expert_offsets[e] * N;
    const __hip_bfloat16* b2_e = b2_layer + (size_t)e * N;

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // --- preload k-slice 0 ---
    {
        for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
            const int r  = idx / BLOCK_K;
            const int ck = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = ck;
            float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
            sA0[r * BLOCK_K + ck] = f32_to_bf16_bits(a);
        }
        for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
            const int cN = idx / BLOCK_K;
            const int rK = idx % BLOCK_K;
            const int gn = n0 + cN;
            const int gk = rK;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp_layer, Sc_layer,
                                   e_elem_base + (size_t)gn * K + gk, layer_elems)
                : 0.f;
            sB0[cN * BLOCK_K + rK] = f32_to_bf16_bits(wb);
        }
    }
    __syncthreads();

    uint16_t* currA = sA0; uint16_t* nextA = sA1;
    uint16_t* currB = sB0; uint16_t* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;

            for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
                const int r  = idx / BLOCK_K;
                const int ck = idx % BLOCK_K;
                const int gm = m0 + r;
                const int gk = kBase + ck;
                float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
                nextA[r * BLOCK_K + ck] = f32_to_bf16_bits(a);
            }
            for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
                const int cN = idx / BLOCK_K;
                const int rK = idx % BLOCK_K;
                const int gn = n0 + cN;
                const int gk = kBase + rK;
                float wb = (gk < K && gn < N)
                    ? dequantize_mxfp4(Wp_layer, Sc_layer,
                                       e_elem_base + (size_t)gn * K + gk, layer_elems)
                    : 0.f;
                nextB[cN * BLOCK_K + rK] = f32_to_bf16_bits(wb);
            }
        }

        const int ldA = BLOCK_K, ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;
        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        uint16_t* tA = currA; currA = nextA; nextA = tA;
        uint16_t* tB = currB; currB = nextB; nextB = tB;
    }

    // store + add bf16 bias per output column
    const bool interior = (m0 + BLOCK_M) <= M_e && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_addbias<true>(C_e, acc, b2_e, M_e, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_addbias<false>(C_e, acc, b2_e, M_e, N, m0, n0, wave_m, wave_n, lane);
    }
}
