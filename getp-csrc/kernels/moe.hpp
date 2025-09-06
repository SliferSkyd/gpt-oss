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
#define WAVES_M_MLP 1
#endif
#ifndef WAVES_N_MLP
#define WAVES_N_MLP 8
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M_MLP = WM * WAVES_M_MLP;        // e.g. 64 if WAVES_M_MLP=4
constexpr int BLOCK_N_MLP = WN * WAVES_N_MLP;        // e.g. 64 if WAVES_N_MLP=4
constexpr int WAVES_PER_BLOCK_MLP = WAVES_M_MLP * WAVES_N_MLP;

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


// Optional tiny padding to reduce LDS bank conflicts (0 keeps behavior identical)
#ifndef PAD_K_MLP
#define PAD_K_MLP 0   // try 2 or 8 if you see LDS conflicts
#endif

// ===== Grouped MLP1 (BF16 weights), W per expert: [N=2D, K=H] row-major =====
__global__ void grouped_mlp1_bf16_kernel(
    float* __restrict__ C, const float* __restrict__ A,
    const __hip_bfloat16* __restrict__ W1,
    const int* __restrict__ expert_offsets, const int* __restrict__ expert_counts,
    const int* __restrict__ tile2expert,
    const int* __restrict__ tile2local,
    int /*E*/, int K, int N)
{
    // ----- Block/tile mapping -----
    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M_MLP;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    // ----- Thread topology -----
    const int lane   = threadIdx.x;                 // 0..63
    const int wave   = threadIdx.y;                 // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    // ----- Shared memory (double-buffered) -----
    // We keep u16 views for MFMA helpers and u32 views for vectorized (2×bf16) writes.
    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K + PAD_K_MLP;           // row-major leading dim in LDS
    const int ldB = BLOCK_K + PAD_K_MLP;           // col-major leading dim in LDS

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
    const __hip_bfloat16* __restrict__ W_e = W1 + (size_t)e * (size_t)N * (size_t)K;

    // ----- helpers -----
    auto pack2_bf16_bits = [] __device__ (float a0, float a1) -> uint32_t {
        const __hip_bfloat16 b0 = __float2bfloat16(a0);
        const __hip_bfloat16 b1 = __float2bfloat16(a1);
        return (uint32_t(hipbf16_to_bits(b1)) << 16) | uint32_t(hipbf16_to_bits(b0));
    };

    // A-tile copy: [BM x BK] at K-base kBase -> LDS row-major with ldA
    auto copy_A_tile = [&] __device__ (uint32_t* __restrict__ dst_u32, int kBase) {
        constexpr int BM_ = BLOCK_M_MLP;
        constexpr int BK_ = BLOCK_K;
        const int pairsPerRow = BK_ >> 1;          // 2 bf16 per u32
        const int totalPairs  = BM_ * pairsPerRow;

        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int r  = t / pairsPerRow;       // row in tile
            const int p  = t % pairsPerRow;       // pair index along K
            const int gm = m_start + r;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gm < M_bound && gk < K) {
                const size_t base = (size_t)gm * K + gk;
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits(a0, a1);
            }
            // row-major with padding: row base is &sA[r*ldA], store pair p
            uint32_t* row = reinterpret_cast<uint32_t*>(dst_u32 + (r * ldA >> 1));
            row[p] = val;
        }
    };

    // B-tile copy: [BK x BN] at K-base kBase -> LDS col-major with ldB
    auto copy_B_tile = [&] __device__ (uint32_t* __restrict__ dst_u32, int kBase) {
        constexpr int BN_ = BLOCK_N_MLP;
        constexpr int BK_ = BLOCK_K;
        const int pairsPerCol = BK_ >> 1;
        const int totalPairs  = BN_ * pairsPerCol;

        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int c  = t / pairsPerCol;       // column in tile
            const int p  = t % pairsPerCol;       // pair along K (two rows)
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gn < N && gk < K) {
                const size_t base = (size_t)gn * K + gk;  // W_e is [N,K] row-major
                if (gk + 1 < K) {
                    // two contiguous bf16 -> single aligned/unaligned u32
                    val = *reinterpret_cast<const uint32_t*>(&W_e[base]);
                } else {
                    const __hip_bfloat16 w0 = W_e[base];
                    val = uint32_t(hipbf16_to_bits(w0));  // hi half zero
                }
            }
            // col-major with padding: col base is &sB[c*ldB], store pair p
            uint32_t* col = reinterpret_cast<uint32_t*>(dst_u32 + (c * ldB >> 1));
            col[p] = val;
        }
    };

    // ----- Preload K-slab 0 -----
    copy_A_tile(sA0_u32, 0);
    copy_B_tile(sB0_u32, 0);
    __syncthreads();

    // ping-pong buffers (u16 views feed your MFMA helpers)
    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;

    // also keep u32 views for the next loads
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    // constants used by the hot loop
    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    // Split K: main (BLOCK_K-aligned) + single tail
    const int BK_ = BLOCK_K;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

    // ----- Main K loop: no K-branching in the copy path -----
#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_) {
        const int kNext = k0 + BK_;

        if (kNext < Kmain) {
            copy_A_tile(nextA32, kNext);
            copy_B_tile(nextB32, kNext);
        }

        // consume curr
        {
            bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
            bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
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

    // ----- Tail slab (0 < K - Kmain < BLOCK_K) -----
    if (has_tail) {
        copy_A_tile(nextA32, Kmain);
        copy_B_tile(nextB32, Kmain);
        __syncthreads();

        bf16x4 avec = make_a_vec(nextA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(nextB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);
        __syncthreads();
    }

    // ----- Store (masked) -----
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
// Optional tiny padding to reduce LDS bank conflicts (0 keeps behavior identical)
#ifndef PAD_K_MLP
#define PAD_K_MLP 0   // try 2 or 8 if you see LDS conflicts; must be even
#endif

static_assert((BLOCK_K % 2) == 0, "BLOCK_K must be even (packs 2×bf16).");
static_assert((PAD_K_MLP % 2) == 0, "PAD_K_MLP must be even (u32 pair addressing).");

// ===== Grouped MLP2 (BF16 weights) + fused bias, W per expert: [N=H, K=D], row-major =====
__global__ void grouped_mlp2_bf16_bias_kernel(
    float* __restrict__ C, const float* __restrict__ A,
    const __hip_bfloat16* __restrict__ W2, const __hip_bfloat16* __restrict__ b2,
    const int* __restrict__ expert_offsets, const int* __restrict__ expert_counts,
    const int* __restrict__ tile2expert,
    const int* __restrict__ tile2local,
    int E, int K, int N)
{
    // ----- Block/tile mapping -----
    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M_MLP;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    // ----- Thread topology -----
    const int lane   = threadIdx.x;                 // 0..63
    const int wave   = threadIdx.y;                 // 0..(WAVES_PER_BLOCK-1)
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    // ----- Shared memory (double-buffered) -----
    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K + PAD_K_MLP;            // row-major leading dim in LDS
    const int ldB = BLOCK_K + PAD_K_MLP;            // col-major leading dim in LDS

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BLOCK_M_MLP * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BLOCK_N_MLP);

    // 32-bit views for vectorized (2×bf16) LDS writes
    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16* __restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16* __restrict__ b_e = b2 + (size_t)e * (size_t)N;

    // ----- helpers -----
    auto pack2_bf16_bits = [] __device__ (float a0, float a1) -> uint32_t {
        const __hip_bfloat16 b0 = __float2bfloat16(a0);
        const __hip_bfloat16 b1 = __float2bfloat16(a1);
        return (uint32_t(hipbf16_to_bits(b1)) << 16) | uint32_t(hipbf16_to_bits(b0));
    };

    // A-tile copy: [BM x BK] at K-base kBase -> LDS row-major with ldA (2×bf16/u32)
    auto copy_A_tile = [&] __device__ (uint32_t* __restrict__ dst_u32, int kBase) {
        constexpr int BM_ = BLOCK_M_MLP;
        constexpr int BK_ = BLOCK_K;
        const int pairsPerRow = BK_ >> 1;               // bf16*2 per u32
        const int totalPairs  = BM_ * pairsPerRow;

        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int r  = t / pairsPerRow;
            const int p  = t % pairsPerRow;
            const int gm = m_start + r;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gm < M_bound && gk < K) {
                const size_t base = (size_t)gm * K + gk;
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits(a0, a1);
            }
            // row-major with padding
            uint32_t* row = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)r * ldA >> 1));
            row[p] = val;
        }
    };

    // B-tile copy: [BK x BN] at K-base kBase -> LDS col-major with ldB (2×bf16/u32)
    auto copy_B_tile = [&] __device__ (uint32_t* __restrict__ dst_u32, int kBase) {
        constexpr int BN_ = BLOCK_N_MLP;
        constexpr int BK_ = BLOCK_K;
        const int pairsPerCol = BK_ >> 1;
        const int totalPairs  = BN_ * pairsPerCol;

        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int c  = t / pairsPerCol;           // column in the BN tile
            const int p  = t % pairsPerCol;           // pair along K
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gn < N && gk < K) {
                const size_t base = (size_t)gn * K + gk;  // W_e is [N,K] row-major
                if (gk + 1 < K) {
                    val = *reinterpret_cast<const uint32_t*>(&W_e[base]);
                } else {
                    const __hip_bfloat16 w0 = W_e[base];
                    val = uint32_t(hipbf16_to_bits(w0));  // hi half zero
                }
            }
            // col-major with padding
            uint32_t* col = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * ldB >> 1));
            col[p] = val;
        }
    };

    // ----- Preload K-slab 0 -----
    copy_A_tile(sA0_u32, 0);
    copy_B_tile(sB0_u32, 0);
    __syncthreads();

    // ping-pong buffers (u16 views feed your MFMA helpers)
    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;

    // also keep u32 views for the next loads
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    // constants used by the hot loop
    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    // Split K: main (BLOCK_K-aligned) + single tail
    const int BK_ = BLOCK_K;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

    // ----- Main K loop: no K-branching in the copy path -----
#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_) {
        const int kNext = k0 + BK_;

        if (kNext < Kmain) {
            copy_A_tile(nextA32, kNext);
            copy_B_tile(nextB32, kNext);
        }

        // consume curr
        {
            bf16x4 avec = make_a_vec(currA, ldA, aRowBase, lane);
            bf16x4 bvec = make_b_vec(currB, ldB, bColBase, lane);
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

    // ----- Tail slab (0 < K - Kmain < BLOCK_K) -----
    if (has_tail) {
        copy_A_tile(nextA32, Kmain);
        copy_B_tile(nextB32, Kmain);
        __syncthreads();

        bf16x4 avec = make_a_vec(nextA, ldA, aRowBase, lane);
        bf16x4 bvec = make_b_vec(nextB, ldB, bColBase, lane);
        acc = mfma_16x16x16_bf16(avec, bvec, acc);
        __syncthreads();
    }

    // ----- Store with fused bias -----
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
