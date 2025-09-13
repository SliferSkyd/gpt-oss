__global__ void route_and_pack_fused_kernel(
    const float*  __restrict__ x,              // [B, H]
    int B, int H,
    const int*    __restrict__ topk_i,         // [B, K]
    const float*  __restrict__ topk_v,         // [B, K]
    int K,
    const int*    __restrict__ expert_offsets, // [E] (exclusive prefix)
    int E,
    int*          __restrict__ local_ids,      // [B, K] (out)
    float*        __restrict__ local_wts,      // [B, K] (out)
    float*        __restrict__ expert_in)      // [sum_tokens, H] (out)
{
    const int e = blockIdx.x;
    if (e >= E) return;

    const int T = blockDim.x;     // threads per block
    const int tid = threadIdx.x;

    extern __shared__ int smem[];
    int* flags   = smem;          // [T]
    int* excl    = flags + T;     // [T]  (inclusive scan buffer)
    int* kidx    = excl  + T;     // [T]  (which k matched e, or -1)
    int* toklist = kidx  + T;     // [T]  (selected token indices in this chunk)
    int* cmplist = toklist + T;   // [T]  (their compact indices)
    int* carry_shared = cmplist + T; // [1]  (shared carry value)


    // int carry = 0;                // how many tokens for expert e packed so far
    if (tid == 0) carry_shared[0] = 0;  // Initialize shared carry
    __syncthreads();

    // Process tokens in tiles of T to support B > T
    for (int base = 0; base < B; base += T) {

        // ---- 1) Flag tokens in this chunk and remember which k matched ----
        int t_global = base + tid;
        int f = 0, kk = -1;
        if (t_global < B) {
            const int off = t_global * K;
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                if (topk_i[off + i] == e) { f = 1; kk = i; break; }
            }
        }
        flags[tid] = f;
        kidx[tid]  = kk;
        __syncthreads();

        // ---- 2) Inclusive scan on flags (Hillis–Steele in-place) ----
        excl[tid] = flags[tid];
        __syncthreads();
        for (int ofs = 1; ofs < T; ofs <<= 1) {
            int v = (tid >= ofs) ? excl[tid - ofs] : 0;
            __syncthreads();
            excl[tid] += v;
            __syncthreads();
        }
        const int rank_local   = (tid == 0) ? 0 : excl[tid - 1];   // exclusive rank within this chunk
        const int chunk_total  = excl[T - 1];                      // total selected in this chunk

        // ---- 3) For selected tokens: compute compact_idx, store lists, fill ids/wts ----
        if (flags[tid]) {
            const int compact_idx = expert_offsets[e] + carry_shared[0] + rank_local;
            const int lid = t_global * K + kidx[tid];

            // per-(b,k) mapping; exactly one expert block writes each entry
            local_ids[lid] = compact_idx;
            local_wts[lid] = topk_v[lid];

            toklist[rank_local] = t_global;
            cmplist[rank_local] = compact_idx;
        }
        __syncthreads();

        // ---- 4) Cooperative copy of all rows selected in this chunk ----
        // Every thread helps copy each row (good coalescing: i strides by T)
        for (int j = 0; j < chunk_total; ++j) {
            const int tkn = toklist[j];
            const int cmp = cmplist[j];
            const float* __restrict__ src = x + (size_t)tkn * H;
            float*       __restrict__ dst = expert_in + (size_t)cmp * H;
            for (int i = tid; i < H; i += T) {
                dst[i] = src[i];
            }
        }
        __syncthreads();

        // ---- 5) Advance global carry for this expert ----
        if (tid == 0) carry_shared[0] += chunk_total;
        __syncthreads();
    }
}

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

__global__ void reduce_tokenwise_expert_outputs(
    float * __restrict__ e_agg,                 // [B, H]  (output)
    const float * __restrict__ expert_output,   // [sum_tokens, H]
    const int   * __restrict__ local_ids,       // [B, Ktok]
    const float * __restrict__ local_wts,       // [B, Ktok]
    int batch_size, int H, int Ktok)
{
    const int b = blockIdx.x; // token
    int d = threadIdx.x + blockIdx.y * blockDim.x; // hidden dim

    if (b >= batch_size || d >= H) return;

    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < Ktok; ++k) {
        const int ci = local_ids[b * Ktok + k];   // compact idx for (b,k)
        if (ci >= 0) {
            acc += local_wts[b * Ktok + k] * expert_output[(size_t)ci * H + d];
        }
    }
    e_agg[(size_t)b * H + d] = acc;
}


__global__ void permute_expert_inputs_kernel(
    const float *d_t, const int *topk_i, const float *topk_v,
    const int *d_expert_offsets, int *d_expert_write_idx,
    int batch_size, int hidden_dim, int experts_per_token,
    float *expert_input_buffer, int *expert_indices, float *expert_weights,
    // NEW: per-token stable mapping
    int *local_ids, float *local_wts)
{
    int token_idx = blockIdx.x;
    if (token_idx >= batch_size) return;

    extern __shared__ int destination_indices[]; // size: experts_per_token

    // One thread per k handles the index math
    if (threadIdx.x < experts_per_token) {
        const int k = threadIdx.x;
        const int topk_flat_idx = token_idx * experts_per_token + k;
        const int expert_id = topk_i[topk_flat_idx];

        // Local position inside expert's block (order here does not matter
        // anymore, we will remember the compact index explicitly)
        const int local_idx  = atomicAdd(&d_expert_write_idx[expert_id], 1);
        const int compact_idx = d_expert_offsets[expert_id] + local_idx;

        // Metadata for later scatter
        expert_indices[compact_idx] = token_idx;
        expert_weights[compact_idx] = topk_v[topk_flat_idx];

        // NEW: remember the exact compact slot for (token, k)
        local_ids[topk_flat_idx] = compact_idx;
        local_wts[topk_flat_idx] = topk_v[topk_flat_idx];

        destination_indices[k] = compact_idx;
    }
    __syncthreads();

    // Coalesced copy of hidden vector into each expert slot
    for (int k = 0; k < experts_per_token; ++k) {
        const float *src = d_t + (size_t)token_idx * hidden_dim;
        float *dst = expert_input_buffer + (size_t)destination_indices[k] * hidden_dim;
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            dst[i] = src[i];
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


#ifndef WAVES_M_MLP
#define WAVES_M_MLP 4
#endif
#ifndef WAVES_N_MLP
#define WAVES_N_MLP 4
#endif

#ifndef WAVES_K_MLP
#define WAVES_K_MLP 1
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M_MLP = WM * WAVES_M_MLP;        // e.g. 64 if WAVES_M_MLP=4
constexpr int BLOCK_N_MLP = WN * WAVES_N_MLP;        // e.g. 64 if WAVES_N_MLP=4
constexpr int WAVES_PER_BLOCK_MLP = WAVES_M_MLP * WAVES_N_MLP;
constexpr int BLOCK_K_MLP = WK * WAVES_K_MLP;        // e.g. 16 if WAVES_K_MLP=1

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

// Dequantize two MXFP4 elements and pack to one u32 (2×bf16)
__device__ __forceinline__ uint32_t deq2_pack_bf16_u32(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    size_t idx0, size_t idx1, int K, int N)
{
    const float f0 = dequantize_mxfp4_block32(packed, scales, idx0);
    float f1 = 0.f;
    if (idx1 < (size_t)K * (size_t)N) {
        f1 = dequantize_mxfp4_block32(packed, scales, idx1);
    }
    return pack2_bf16_bits_f32(f0, f1);
}


// Two outputs from one packed byte + one pre-expanded f32 scale
__device__ __forceinline__ uint32_t deq2_pack_bf16_u32_fscale(
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ scales_f32,
    size_t even_idx, // must be even: even_idx and even_idx+1 share the byte/scale
    int K, int N)
{
    const size_t blk = even_idx >> 5;            // /32
    const uint8_t b  = packed[even_idx >> 1];    // /2
    const uint8_t nib0 =  b        & 0xF;
    const uint8_t nib1 = (b >> 4)  & 0xF;
    const float X = scales_f32[blk];

    // fast magnitude map without branches
    const float mag0 = mxfp4_mag_from_code(nib0 & 7);
    const float mag1 = mxfp4_mag_from_code(nib1 & 7);
    const float v0 = ((nib0 & 8) ? -mag0 : mag0) * X;
    const float v1 = ((nib1 & 8) ? -mag1 : mag1) * X;

    return pack2_bf16_bits_f32(v0, v1);
}

// Load B tile: global (MXFP4) -> LDS (bf16 col-major).
// Layout in LDS matches your mfma path: [BLOCK_K_MLP x BLOCK_N], ldB = BLOCK_K_MLP.
template<int LD_B>
__device__ inline void copy_B_tile_vec_MLP_MXFP4(
    uint32_t* __restrict__ dst_u32,           // writes u32 (2×bf16) into LDS
    const uint8_t* __restrict__ W_packed_e,   // per-expert base (packed nibbles)
    const uint8_t* __restrict__ S_e8m0_e,     // per-expert base (E8M0 scales per 32)
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int BN_ = BLOCK_N_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerCol = BK_ >> 1;           // 2 elems per u32
    const int totalPairs  = BN_ * pairsPerCol;

    // We store column-major in LDS: column c lives at dst_u32 + (c * LD_B)/2 (u32 indexing)
    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int c  = t / pairsPerCol;         // 0..BN_-1
        const int p  = t % pairsPerCol;         // 0..pairsPerCol-1
        const int gn = n0 + c;                  // global N index (column)
        const int gk = kBase + (p << 1);        // row index in K (two per u32)

        uint32_t out = 0u;
        if (gn < N && gk < K) {
            // Flattened row-major index over the per-expert matrix [N x K]:
            // idx = gn*K + gk  (your CPU side stored [E, N, K] row-major, and packed by that)
            const size_t idx0 = (size_t)gn * (size_t)K + (size_t)gk;
            const size_t idx1 = idx0 + 1;   // tail-safe handled inside deq2_pack
            out = deq2_pack_bf16_u32(W_packed_e, S_e8m0_e, idx0, idx1, K, N);
        }
        // write into LDS in column-major: (c * LD_B) / 2 is u32 stride
        reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = out;
    }
}

// Load B tile: global (MXFP4) -> LDS (bf16 col-major) using f32 scales.
// Stores into LDS as u32 (two bf16 packed). LD_B is LDS stride in elements.
template<int LD_B>
__device__ inline void copy_B_tile_vec_MLP_MXFP4_f32(
    uint32_t* __restrict__ dst_u32,           // writes u32 (2×bf16) into LDS
    const uint8_t* __restrict__ W_packed_e,   // per-expert packed nibble base
    const float*   __restrict__ S_f32_e,      // per-expert f32 scales base
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int BN_ = BLOCK_N_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerCol = BK_ >> 1;           // 2 elems per u32
    const int totalPairs  = BN_ * pairsPerCol;

    // LDS column-major: column c lives at dst_u32 + (c * LD_B)/2 (u32 addressing)
    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int c  = t / pairsPerCol;         // 0..BN_-1
        const int p  = t % pairsPerCol;         // 0..pairsPerCol-1
        const int gn = n0 + c;                  // global column (N)
        const int gk = kBase + (p << 1);        // row in K, even

        uint32_t out = 0u;
        if (gn < N && gk < K) {
            // linear index over row-major [N x K]
            const size_t idx0 = (size_t)gn * (size_t)K + (size_t)gk;   // even
            out = deq2_pack_bf16_u32_fscale(W_packed_e, S_f32_e, idx0, K, N);
        }
        reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = out;
    }
}


// Vectorized A load: read float2 (64b), convert to 2×bf16 in a single u32
template<bool Aligned, int LD_A>
__device__ inline void copy_A_tile_vec_MLP(uint32_t* __restrict__ dst_u32,
                                       const float* __restrict__ A,
                                       int m_start, int M_bound, int K, int kBase,
                                       int linearT, int threadsPerBlock) {
    constexpr int BM_ = BLOCK_M_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerRow = BK_ >> 1;
    const int totalPairs  = BM_ * pairsPerRow;

    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int r  = t / pairsPerRow;
        const int p  = t % pairsPerRow;
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);

        uint32_t val = 0u;
        if (gm < M_bound && gk < K) {
            const size_t base = (size_t)gm * K + gk;
            if constexpr (Aligned) {
                // 64-bit aligned path if A is 8B-aligned and K multiple of 2
                const float2 v = *reinterpret_cast<const float2*>(&A[base]);
                val = pack2_bf16_bits_f32(v.x, v.y);
            } else {
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits_f32(a0, a1);
            }
        }
        reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)r * LD_A >> 1))[p] = val;
    }
}

#ifndef PAD_K_MLP
#define PAD_K_MLP 0   // try 2 or 8 if you see LDS conflicts
#endif
static_assert((BLOCK_K_MLP % 2) == 0, "BLOCK_K_MLP must be even (packs 2×bf16).");
static_assert((PAD_K_MLP % 2) == 0, "PAD_K_MLP must be even (u32 pair addressing).");


// NOTE: we accept *f32* scale layer pointers now.
__global__ void grouped_mlp1_mxfp4_kernel(
    float* __restrict__ C,                 // [sum_tokens, 2D]
    const float* __restrict__ A,           // [sum_tokens, H]  (float), row-major
    const uint8_t* __restrict__ W1_packed_layer,    // [E, ceil((2D*H)/2)]
    const float*   __restrict__ S1_scales_f32_layer,// [E, ceil((2D*H)/32)]
    const int* __restrict__ expert_offsets,         // [E]
    const int* __restrict__ expert_counts,          // [E]
    const int* __restrict__ tile2expert,            // [num_mtiles]
    const int* __restrict__ tile2local,             // [num_mtiles]
    int /*E*/, int K /*=H*/, int N /*=2D*/)
{
    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M_MLP;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K_MLP + PAD_K_MLP;
    const int ldB = BLOCK_K_MLP + PAD_K_MLP;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N_MLP);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;

    const size_t seg_elems        = (size_t)N * (size_t)K;
    const size_t seg_packed_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks       = (seg_elems + 31) / 32;

    const uint8_t* __restrict__ W_e_packed = W1_packed_layer      + (size_t)e * seg_packed_bytes;
    const float*   __restrict__ S_f32_e     = S1_scales_f32_layer  + (size_t)e * seg_blocks;

    const bool alignedA = (((uintptr_t)A & 0x7)==0) && ((K & 1)==0);

    if (alignedA) copy_A_tile_vec_MLP<true,  /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec_MLP<false, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(sB0_u32, W_e_packed, S_f32_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    const int BK_ = BLOCK_K_MLP;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_) {
        const int kNext = k0 + BK_;
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec_MLP<true,  ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
        if (kNext < Kmain) {
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    if (has_tail) {
        if (alignedA) copy_A_tile_vec_MLP<true,  ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK) {
            bf16x4 avec = make_a_vec_k(nextA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(nextB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }

    const int rowBase = m_start + wave_m * WM + lane_group(lane) * 4;
    const int col     = n0 + wave_n * WN + lane_row(lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if (row < M_bound && col < N) {
            C[(size_t)row * N + col] = acc[i];
        }
    }
}

__global__ void grouped_mlp2_mxfp4_bias_kernel(
    float* __restrict__ C,                   // [sum_tokens, H]
    const float* __restrict__ A,             // [sum_tokens, D]  (float), row-major
    const uint8_t* __restrict__ W2_packed_layer,    // [E, ceil((H*D)/2)]
    const float*   __restrict__ S2_scales_f32_layer,// [E, ceil((H*D)/32)]
    const __hip_bfloat16* __restrict__ b2,          // [E, H] bf16
    const int* __restrict__ expert_offsets, const int* __restrict__ expert_counts,
    const int* __restrict__ tile2expert,     const int* __restrict__ tile2local,
    int /*E*/, int K /*=D*/, int N /*=H*/)
{
    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M_MLP;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K_MLP + PAD_K_MLP;
    const int ldB = BLOCK_K_MLP + PAD_K_MLP;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N_MLP);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;

    const size_t seg_elems        = (size_t)N * (size_t)K;
    const size_t seg_packed_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks       = (seg_elems + 31) / 32;

    const uint8_t* __restrict__ W_e_packed = W2_packed_layer      + (size_t)e * seg_packed_bytes;
    const float*   __restrict__ S_f32_e     = S2_scales_f32_layer  + (size_t)e * seg_blocks;
    const __hip_bfloat16* __restrict__ b_e  = b2 + (size_t)e * (size_t)N;

    const bool alignedA = (((uintptr_t)A & 0x7)==0) && ((K & 1)==0);

    if (alignedA) copy_A_tile_vec_MLP<true,  /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec_MLP<false, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(sB0_u32, W_e_packed, S_f32_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    const int BK_ = BLOCK_K_MLP;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_) {
        const int kNext = k0 + BK_;
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec_MLP<true,  ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK) {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }

        __syncthreads();
        if (kNext < Kmain) {
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    if (has_tail) {
        if (alignedA) copy_A_tile_vec_MLP<true,  ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK) {
            bf16x4 avec = make_a_vec_k(nextA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(nextB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }

    const int col  = n0 + wave_n * WN + lane_row(lane);
    const float bias = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;
    const int rowBase = m_start + wave_m * WM + lane_group(lane) * 4;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + i;
        if (row < M_bound && col < N) {
            C[(size_t)row * N + col] = acc[i] + bias;
        }
    }
}