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

// ============ GROUPED MLP1 (MXFP4) across all experts ============
// A: expert_input [sum_tokens, K] (FP32, packed by expert using offsets)
// Wp/scales: per-layer base; each expert is a contiguous segment
// Output: mlp1_out [sum_tokens, 2*intermediate_dim] (FP32)
__global__ void grouped_mlp1_mxfp4_kernel(
    float* __restrict__ C,                     // mlp1_out
    const float* __restrict__ A,               // expert_input_buffer
    const uint8_t* __restrict__ Wp_layer,      // layer base (packed)
    const float* __restrict__ Sc_layer,        // layer base (scales)
    const int* __restrict__ expert_offsets,    // [n_experts]
    const int* __restrict__ expert_counts,     // [n_experts]
    const int* __restrict__ mtile_prefix,      // [n_experts+1] prefix of m-tiles
    int n_experts,
    int K,                                     // hidden_dim
    int N,                                     // 2*intermediate_dim
    size_t seg_elems,                          // per-expert elems in W (N*K)
    size_t seg_packed_bytes,                   // per-expert bytes in Wp
    size_t seg_blocks)                         // per-expert #scale blocks
{
    const int mTileGlobal = blockIdx.y;
    const int e = map_tile_to_expert(mTileGlobal, mtile_prefix, n_experts);

    const int mTileLocal = mTileGlobal - mtile_prefix[e];
    const int M_e = expert_counts[e];

    if (M_e == 0) return;

    // tile bases (relative to expert)
    const int m0 = mTileLocal * BLOCK_M;
    const int n0 = blockIdx.x * BLOCK_N;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // Per-expert bases
    const float*  A_e  = A + (size_t)expert_offsets[e] * K;
    float*        C_e  = C + (size_t)expert_offsets[e] * N;
    const uint8_t*Wp_e = Wp_layer + e * seg_packed_bytes;
    const float*  Sc_e = Sc_layer + e * seg_blocks;

    f32x4 acc = {0.f,0.f,0.f,0.f};

    // Preload k-slice 0
    {
        for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gm = m0 + r;
            const int gk = c;
            float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
            sA0[r * BLOCK_K + c] = f32_to_bf16_bits(a);
        }
        for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K; // N-col within block
            const int r = idx % BLOCK_K; // K within slice
            const int gn = n0 + c;
            const int gk = r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp_e, Sc_e, (size_t)gn * K + gk, seg_elems)
                : 0.f;
            sB0[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
        }
    }
    __syncthreads();

    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K, c = idx % BLOCK_K;
                const int gm = m0 + r, gk = kBase + c;
                float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
                nextA[r * BLOCK_K + c] = f32_to_bf16_bits(a);
            }
            for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K, r = idx % BLOCK_K;
                const int gn = n0 + c, gk = kBase + r;
                float wb = (gk < K && gn < N)
                    ? dequantize_mxfp4(Wp_e, Sc_e, (size_t)gn * K + gk, seg_elems)
                    : 0.f;
                nextB[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
            }
        }

        const int ldA = BLOCK_K, ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;
        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    const bool interior = (m0 + BLOCK_M) <= M_e && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_mfma<true>(C_e, acc, M_e, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_mfma<false>(C_e, acc, M_e, N, m0, n0, wave_m, wave_n, lane);
    }
}

__device__ __forceinline__ float fast_expf(float x) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __expf(x); // fast SFU path on ROCm
#else
    return expf(x);
#endif
}

__global__ void bias_swiglu_epilogue_kernel(
    const float* __restrict__ mlp1_out,     // [sum_tokens, 2*D]
    const __hip_bfloat16* __restrict__ b1,  // [n_experts, 2*D]
    const int* __restrict__ expert_offsets, // [n_experts]
    const int* __restrict__ expert_counts,  // [n_experts]
    int n_experts,
    float* __restrict__ gate_up,            // [sum_tokens, D]
    int D,
    int sum_tokens,
    float clamp_limit,
    float alpha_silu /* = 1.702f */)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)sum_tokens * D;
    if (idx >= total) return;

    const int d = (int)(idx % D);

    // ---- warp-cooperative token/expert resolution ----
    const int lane = threadIdx.x & (warpSize - 1);
    const size_t warp_first = idx - lane;
    const int t0 = (int)(warp_first / D);
    const int t1 = (int)((warp_first + (warpSize - 1)) / D);

    int t, e;
    if (t0 == t1) {
        // Same token across the whole wave: binary-search once in lane 0
        t = t0;
        int lo = 0, hi = n_experts;
        if (lane == 0) {
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                const int start = expert_offsets[mid];
                const int end   = start + expert_counts[mid];
                if (t >= end)       lo = mid + 1;
                else if (t < start) hi = mid;
                else { lo = mid; break; }
            }
        }
        // broadcast t and e
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        t = __shfl(t, 0);
        e = __shfl(lo, 0);
#else
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        e = __shfl_sync(0xFFFFFFFF, lo, 0);
#endif
    } else {
        // Fallback (rare if D is large): per-thread search
        t = (int)(idx / D);
        int lo = 0, hi = n_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            const int start = expert_offsets[mid];
            const int end   = start + expert_counts[mid];
            if (t >= end)       lo = mid + 1;
            else if (t < start) hi = mid;
            else { lo = mid; break; }
        }
        e = lo;
    }

    // ---- precompute bases to cut address math & lifetimes ----
    const size_t t2D = (size_t)t * (2 * (size_t)D);
    const size_t e2D = (size_t)e * (2 * (size_t)D);

    // bias in bf16 -> f32
    const float bg = __bfloat162float(b1[e2D + (size_t)(2*d + 0)]);
    const float bu = __bfloat162float(b1[e2D + (size_t)(2*d + 1)]);

    // load + bias
    float gx = mlp1_out[t2D + (size_t)(2*d + 0)] + bg;
    float uy = mlp1_out[t2D + (size_t)(2*d + 1)] + bu;

    // clamp early to keep ranges tight (reduces VGPR lifetimes)
    gx = fminf(fmaxf(gx, -clamp_limit), clamp_limit);
    uy = fminf(fmaxf(uy, -clamp_limit), clamp_limit);

    // SiLU(x) = x * sigmoid(alpha*x)
    const float sig = 1.0f / (1.0f + fast_expf(-alpha_silu * gx));
    gx *= sig; // reuse gx register as swish

    // swish * (uy + 1) using FMA to save one temp & shrink live ranges
    gate_up[(size_t)t * (size_t)D + d] = fmaf(gx, uy, gx);
}


// ============ GROUPED MLP2 (MXFP4) across all experts (+bias add) ============
__global__ void grouped_mlp2_mxfp4_bias_kernel(
    float* __restrict__ C,                     // expert_output_buffer
    const float* __restrict__ A,               // gate_up
    const uint8_t* __restrict__ Wp_layer,      // layer base (packed)
    const float* __restrict__ Sc_layer,        // layer base (scales)
    const __hip_bfloat16* __restrict__ b2_layer,// layer base bias [n_experts, hidden_dim]
    const int* __restrict__ expert_offsets,    // [n_experts]
    const int* __restrict__ expert_counts,     // [n_experts]
    const int* __restrict__ mtile_prefix,      // [n_experts+1]
    int n_experts,
    int K,                                     // intermediate_dim
    int N,                                     // hidden_dim
    size_t seg_elems,                          // per-expert elems in W (N*K)
    size_t seg_packed_bytes,                   // per-expert bytes in Wp
    size_t seg_blocks)                         // per-expert #scale blocks
{
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
    auto* sA0 = reinterpret_cast<uint16_t*>(smemRaw);
    auto* sA1 = sA0 + (BLOCK_M * BLOCK_K);
    auto* sB0 = sA1 + (BLOCK_M * BLOCK_K);
    auto* sB1 = sB0 + (BLOCK_K * BLOCK_N);

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    const float*  A_e   = A + (size_t)expert_offsets[e] * K;
    float*        C_e   = C + (size_t)expert_offsets[e] * N;
    const uint8_t*Wp_e  = Wp_layer + e * seg_packed_bytes;
    const float*  Sc_e  = Sc_layer + e * seg_blocks;
    const __hip_bfloat16* b2_e = b2_layer + (size_t)e * N;

    f32x4 acc = {0.f,0.f,0.f,0.f};

    // preload slice 0
    {
        for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
            const int r = idx / BLOCK_K, c = idx % BLOCK_K;
            const int gm = m0 + r, gk = c;
            float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
            sA0[r * BLOCK_K + c] = f32_to_bf16_bits(a);
        }
        for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
            const int c = idx / BLOCK_K, r = idx % BLOCK_K;
            const int gn = n0 + c, gk = r;
            float wb = (gk < K && gn < N)
                ? dequantize_mxfp4(Wp_e, Sc_e, (size_t)gn * K + gk, seg_elems) : 0.f;
            sB0[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
        }
    }
    __syncthreads();

    auto* currA = sA0; auto* nextA = sA1;
    auto* currB = sB0; auto* nextB = sB1;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        if (k0 + BLOCK_K < K) {
            const int kBase = k0 + BLOCK_K;
            for (int idx = linearT; idx < BLOCK_M*BLOCK_K; idx += threadsPerBlock) {
                const int r = idx / BLOCK_K, c = idx % BLOCK_K;
                const int gm = m0 + r, gk = kBase + c;
                float a = (gm < M_e && gk < K) ? A_e[(size_t)gm * K + gk] : 0.f;
                nextA[r * BLOCK_K + c] = f32_to_bf16_bits(a);
            }
            for (int idx = linearT; idx < BLOCK_K*BLOCK_N; idx += threadsPerBlock) {
                const int c = idx / BLOCK_K, r = idx % BLOCK_K;
                const int gn = n0 + c, gk = kBase + r;
                float wb = (gk < K && gn < N)
                    ? dequantize_mxfp4(Wp_e, Sc_e, (size_t)gn * K + gk, seg_elems) : 0.f;
                nextB[c * BLOCK_K + r] = f32_to_bf16_bits(wb);
            }
        }

        const int ldA = BLOCK_K, ldB = BLOCK_K;
        const int aRowBase = wave_m * WM;
        const int bColBase = wave_n * WN;
        bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);

        __syncthreads();
        auto* tA = currA; currA = nextA; nextA = tA;
        auto* tB = currB; currB = nextB; nextB = tB;
    }

    const bool interior = (m0 + BLOCK_M) <= M_e && (n0 + BLOCK_N) <= N;
    if (interior) {
        store_c_tile_addbias<true>(C_e, acc, b2_e, M_e, N, m0, n0, wave_m, wave_n, lane);
    } else {
        store_c_tile_addbias<false>(C_e, acc, b2_e, M_e, N, m0, n0, wave_m, wave_n, lane);
    }
}

// ===== Helper: find expert for a given mtile id =====
__device__ inline int find_expert_from_mtile(const int* __restrict__ mtile_prefix, int E, int mtile_id) {
    // small E (<=64) -> linear is fine; switch to binary if you want
    for (int e = 0; e < E; ++e) {
        if (mtile_id < mtile_prefix[e+1]) return e;
    }
    return E - 1;
}

// ===== Grouped MLP1 (BF16 weights), W per expert: [N=2D, K=H] row-major =====
__global__ void grouped_mlp1_bf16_kernel(
    float* __restrict__ C,                     // [total_tokens, 2D]
    const float* __restrict__ A,               // [total_tokens, H]
    const __hip_bfloat16* __restrict__ W1,     // layer base: [E, 2D, H] row-major
    const int* __restrict__ expert_offsets,    // [E]
    const int* __restrict__ expert_counts,     // [E]
    const int* __restrict__ mtile_prefix,      // [E+1] (built with BLOCK_M tiles)
    int E, int K, int N)                       // K=H, N=2D
{
    // FMA micro-tile coverage (matches your matmul config)
    const int SUB_M = TB_Y * TM;   // rows per threadblock from FMA micro-tiles
    const int SUB_N = TB_X * TN;   // cols per threadblock from FMA micro-tiles

    // Map (blockIdx.y) -> (expert e, local BLOCK_M-sized M tile)
    const int mtile_id   = blockIdx.y;
    const int e          = find_expert_from_mtile(mtile_prefix, E, mtile_id);
    const int first_tile = mtile_prefix[e];
    const int tile_m_in_e= mtile_id - first_tile;

    // Expert-local M range for this BLOCK_M tile
    const int m0_block = expert_offsets[e] + tile_m_in_e * BLOCK_M;  // row start in A/C
    const int m_left   = expert_counts[e] - tile_m_in_e * BLOCK_M;   // rows left in this expert
    if (m_left <= 0) return;
    const int M_bound_block = m0_block + m_left; // exclusive upper bound for expert rows

    // Column origin for the MFMA block
    const int n0_block = blockIdx.x * BLOCK_N;

    // Thread coords (FMA layout)
    const int tx = threadIdx.x; // 0..TB_X-1
    const int ty = threadIdx.y; // 0..TB_Y-1

    // Shared mem (float) sized for one FMA subtile (SUB_M x SUB_N)
    extern __shared__ float smem[];
    float* sA0 = smem;
    float* sA1 = sA0 + (SUB_M * BK);
    float* sB0 = sA1 + (SUB_M * BK);
    float* sB1 = sB0 + (BK * SUB_N);

    // Helpers for cooperative loads
    const int threadsPerBlock = TB_X * TB_Y;
    const int linearT         = ty * TB_X + tx;

    // Base pointer into this expert’s weight matrix [N, K] row-major
    const __hip_bfloat16* __restrict__ W_e = W1 + (size_t)e * (size_t)N * (size_t)K;

    // Loop over MFMA block in FMA sub-tiles along M and N
    for (int m_off = 0; m_off < BLOCK_M; m_off += SUB_M) {
        const int m0 = m0_block + m_off;
        if (m0 >= M_bound_block) break;               // no more rows in expert
        const int M_bound_sub = min(M_bound_block, m0 + SUB_M);

        for (int n_off = 0; n_off < BLOCK_N; n_off += SUB_N) {
            const int n0 = n0_block + n_off;
            if (n0 >= N) continue;                    // off the right edge

            // Per-thread micro-tile origin for this subtile
            const int rowBase = m0 + ty * TM;
            const int colBase = n0 + tx * TN;

            // Accumulators (double for stability, like your matmul)
            double acc[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = 0.0;

            // Cooperative loaders (global -> shared), FMA layout
            auto loadA = [&](float* dst, int kBase) {
                // A is row-major [M x K], we want [SUB_M x BK] slice
                for (int idx = linearT; idx < SUB_M * BK; idx += threadsPerBlock) {
                    const int r  = idx / BK;   // 0..SUB_M-1
                    const int c  = idx % BK;   // 0..BK-1
                    const int gm = m0 + r;
                    const int gk = kBase + c;
                    const float a = (gm < M_bound_sub && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                    dst[r * BK + c] = a;
                }
            };
            auto loadB = [&](float* dst, int kBase) {
                // W_e is row-major [N x K]; build sB row-major [BK x SUB_N] for unit-stride kk
                for (int idx = linearT; idx < BK * SUB_N; idx += threadsPerBlock) {
                    const int r  = idx / SUB_N; // 0..BK-1 (k within slice)
                    const int c  = idx % SUB_N; // 0..SUB_N-1 (n within subtile)
                    const int gk = kBase + r;
                    const int gn = n0   + c;
                    const float b = (gk < K && gn < N) ? __bfloat162float(W_e[(size_t)gn * K + gk]) : 0.0f;
                    dst[r * SUB_N + c] = b;
                }
            };

            // Preload first K-slice
            loadA(sA0, /*kBase=*/0);
            loadB(sB0, /*kBase=*/0);
            __syncthreads();

            float* currA = sA0; float* nextA = sA1;
            float* currB = sB0; float* nextB = sB1;

            // Main K loop in BK chunks (BK = your matmul BK)
            for (int k0 = 0; k0 < K; k0 += BK) {
                if (k0 + BK < K) {
                    loadA(nextA, k0 + BK);
                    loadB(nextB, k0 + BK);
                }

#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float aFrag[TM];
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        const int r_local = ty * TM + i;             // 0..SUB_M-1
                        aFrag[i] = currA[r_local * BK + kk];
                    }
                    float bFrag[TN];
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int c_local = tx * TN + j;             // 0..SUB_N-1
                        bFrag[j] = currB[kk * SUB_N + c_local];
                    }
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            acc[i][j] += (double)aFrag[i] * bFrag[j];
                }

                __syncthreads();
                float* tA = currA; currA = nextA; nextA = tA;
                float* tB = currB; currB = nextB; nextB = tB;
            }

            // Store back (masked to expert rows/valid cols)
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int gm = rowBase + i;
                if (gm >= M_bound_sub) break;
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int gn = colBase + j;
                    if (gn < N) C[(size_t)gm * N + gn] = (float)acc[i][j];
                }
            }
        } // n_off
    } // m_off
}

// ===== Grouped MLP2 (BF16) + fused bias add; W per expert: [N=H, K=D] row-major =====
__global__ void grouped_mlp2_bf16_bias_kernel(
    float* __restrict__ C,                      // [total_tokens, H]
    const float* __restrict__ A,                // [total_tokens, D]  (A = gate_up)
    const __hip_bfloat16* __restrict__ W2,      // layer base: [E, H, D] row-major
    const __hip_bfloat16* __restrict__ b2,      // layer base: [E, H]
    const int* __restrict__ expert_offsets,     // [E]
    const int* __restrict__ expert_counts,      // [E]
    const int* __restrict__ mtile_prefix,       // [E+1] (built with BLOCK_M tiles)
    int E, int K, int N)                        // K=D, N=H
{
    const int SUB_M = TB_Y * TM;
    const int SUB_N = TB_X * TN;

    const int mtile_id   = blockIdx.y;
    const int e          = find_expert_from_mtile(mtile_prefix, E, mtile_id);
    const int first_tile = mtile_prefix[e];
    const int tile_m_in_e= mtile_id - first_tile;

    const int m0_block   = expert_offsets[e] + tile_m_in_e * BLOCK_M;
    const int m_left     = expert_counts[e] - tile_m_in_e * BLOCK_M;
    if (m_left <= 0) return;
    const int M_bound_block = m0_block + m_left;

    const int n0_block   = blockIdx.x * BLOCK_N;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    extern __shared__ float smem[];
    float* sA0 = smem;
    float* sA1 = sA0 + (SUB_M * BK);
    float* sB0 = sA1 + (SUB_M * BK);
    float* sB1 = sB0 + (BK * SUB_N);

    const __hip_bfloat16* __restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16* __restrict__ b_e = b2 + (size_t)e * (size_t)N;

    const int threadsPerBlock = TB_X * TB_Y;
    const int linearT         = ty * TB_X + tx;

    for (int m_off = 0; m_off < BLOCK_M; m_off += SUB_M) {
        const int m0 = m0_block + m_off;
        if (m0 >= M_bound_block) break;
        const int M_bound_sub = min(M_bound_block, m0 + SUB_M);

        for (int n_off = 0; n_off < BLOCK_N; n_off += SUB_N) {
            const int n0 = n0_block + n_off;
            if (n0 >= N) continue;

            const int rowBase = m0 + ty * TM;
            const int colBase = n0 + tx * TN;

            double acc[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = 0.0;

            auto loadA = [&](float* dst, int kBase) {
                for (int idx = linearT; idx < SUB_M * BK; idx += threadsPerBlock) {
                    const int r  = idx / BK;
                    const int c  = idx % BK;
                    const int gm = m0 + r;
                    const int gk = kBase + c;
                    const float a = (gm < M_bound_sub && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
                    dst[r * BK + c] = a;
                }
            };
            auto loadB = [&](float* dst, int kBase) {
                for (int idx = linearT; idx < BK * SUB_N; idx += threadsPerBlock) {
                    const int r  = idx / SUB_N;
                    const int c  = idx % SUB_N;
                    const int gk = kBase + r;
                    const int gn = n0   + c;
                    const float b = (gk < K && gn < N) ? __bfloat162float(W_e[(size_t)gn * K + gk]) : 0.0f;
                    dst[r * SUB_N + c] = b;
                }
            };

            loadA(sA0, 0);
            loadB(sB0, 0);
            __syncthreads();

            float* currA = sA0; float* nextA = sA1;
            float* currB = sB0; float* nextB = sB1;

            for (int k0 = 0; k0 < K; k0 += BK) {
                if (k0 + BK < K) {
                    loadA(nextA, k0 + BK);
                    loadB(nextB, k0 + BK);
                }

#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float aFrag[TM];
#pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        const int r_local = ty * TM + i;
                        aFrag[i] = currA[r_local * BK + kk];
                    }
                    float bFrag[TN];
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int c_local = tx * TN + j;
                        bFrag[j] = currB[kk * SUB_N + c_local];
                    }
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j)
                            acc[i][j] += (double)aFrag[i] * bFrag[j];
                }

                __syncthreads();
                float* tA = currA; currA = nextA; nextA = tA;
                float* tB = currB; currB = nextB; nextB = tB;
            }

            // Add bias (per output column) on store
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gn    = colBase + j;
                const float bia = (gn < N) ? __bfloat162float(b_e[gn]) : 0.f;
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int gm = rowBase + i;
                    if (gm < M_bound_sub && gn < N) {
                        C[(size_t)gm * N + gn] = (float)acc[i][j] + bia;
                    }
                }
            }
        } // n_off
    } // m_off
}
