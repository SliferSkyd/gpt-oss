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
    float *expert_input_buffer,
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

    int carry = 0;                // how many tokens for expert e packed so far

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
            const int compact_idx = expert_offsets[e] + carry + rank_local;
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
        if (tid == 0) carry += chunk_total;
        __syncthreads();
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



// ============================================================================
// Templated copy helpers (BM/BN are compile-time so loops vectorize nicely)
// ============================================================================

template<bool Aligned, int LD_A, int BM, int BK>
__device__ inline void copy_A_tile_vec_tiled(
    uint32_t* __restrict__ dst_u32,
    const float* __restrict__ A,
    int m_start, int M_bound, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int pairsPerRow = BK >> 1;
    constexpr int totalPairs  = BM * pairsPerRow;

    for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
        const int r  = t / pairsPerRow;       // 0..BM-1
        const int p  = t % pairsPerRow;       // 0..BK/2-1
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);

        uint32_t val = 0u;
        if (gm < M_bound && gk < K) {
            const size_t base = (size_t)gm * K + gk;
            if constexpr (Aligned) {
                const float2 v = *reinterpret_cast<const float2*>(&A[base]);
                // (pack2_bf16_bits_f32 is from your GEMM path)
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

template<bool Use128b, int LD_B, int BN, int BK>
__device__ inline void copy_B_tile_vec_tiled(
    uint32_t* __restrict__ dst_u32,
    const __hip_bfloat16* __restrict__ W,   // [N,K] row-major
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int pairsPerCol = BK >> 1;

    if constexpr (Use128b) {
        constexpr int quadPerCol = BK / 8;   // 8 bf16 per 128b
        constexpr int totalQuads = BN * quadPerCol;
        for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
            const int c   = t / quadPerCol;  // 0..BN-1
            const int q   = t % quadPerCol;  // which 8-elem chunk
            const int gn  = n0 + c;
            const int gk8 = kBase + (q << 3);

            uint4 v = {0,0,0,0};
            if (gn < N && (gk8 + 7) < K) {
                const uint4* src = reinterpret_cast<const uint4*>(&W[(size_t)gn * K + gk8]);
                v = *src;
            } else {
                __hip_bfloat16 tmp[8] = {};
                for (int i=0;i<8 && (gk8+i)<K && gn<N;i++) tmp[i] = W[(size_t)gn*K + gk8 + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0],p[1],p[2],p[3]);
            }
            uint32_t* col = reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1));
            const int off = q << 2;
            col[off + 0] = v.x; col[off + 1] = v.y; col[off + 2] = v.z; col[off + 3] = v.w;
        }
    } else {
        constexpr int totalPairs = BN * pairsPerCol;
        for (int t = linearT; t < totalPairs; t += threadsPerBlock) {
            const int c  = t / pairsPerCol;
            const int p  = t % pairsPerCol;
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gn < N && gk < K) {
                const size_t base = (size_t)gn * K + gk;
                if (gk + 1 < K) val = *reinterpret_cast<const uint32_t*>(&W[base]);
                else            val = uint32_t(hipbf16_to_bits(W[base]));
            }
            reinterpret_cast<uint32_t*>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = val;
        }
    }
}

// ============================================================================
// Templated grouped MLP1 kernel: C = A·W1  (no bias)
// Supports per-wave multi-tiles: TW_M × TW_N (each is >=1)
// ============================================================================

template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=8, int WAVES_K=2,
    int TW_M=1,  int TW_N=2,
    int PAD_K=0
>
__global__ void grouped_mlp1_bf16_kernel_tiled(
    float* __restrict__ C,                 // [sum_tokens, 2*D]
    const float* __restrict__ A,           // [sum_tokens, H]
    const __hip_bfloat16* __restrict__ W1, // [N=2*D, K=H] row-major
    const int* __restrict__ expert_offsets,// [E]
    const int* __restrict__ expert_counts, // [E]
    const int* __restrict__ tile2expert,   // [cur_tiles]
    const int* __restrict__ tile2local,    // [cur_tiles]
    int E, int K, int N)
{
    static_assert(WM==16 && WN==16 && WK==16, "MFMA kernel assumes 16x16x16 bf16 tiles.");
    constexpr int LANE_PER_WAVE = 64;

    constexpr int BLOCK_M = WM * WAVES_M;             // per-block (one buffer)
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK      = WK * WAVES_K;

    // expanded per-stage tile (what we actually load/compute)
    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M;   // base of this "logical" block
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BN;   // grid.x in units of BN

    const int lane   = threadIdx.x;                           // 0..63
    const int wave   = threadIdx.y;                           // 0..(WAVES_M*WAVES_N-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // ------------------ LDS double buffers ------------------
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BM * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BM * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BN);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    // ------------------ accumulators ------------------
    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm)
    #pragma unroll
    for (int tn=0; tn<TW_N; ++tn)
        acc[tm][tn] = {0.f,0.f,0.f,0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16* __restrict__ W_e = W1 + (size_t)e * (size_t)N * (size_t)K;

    // alignment guards
    const bool alignedA = (((uintptr_t)A  & 0x7)==0) && ((K & 1)==0);
    const bool use128bW = (((uintptr_t)W_e& 0xF)==0) && ((K & 7)==0);

    // -------- preload k=0 --------
    if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);
    else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    // per-wave bases for sub-tiles (inside BM×BN)
    auto aRowBase_tm = [&](int tm){ return (wave_m + tm*WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn){ return (wave_n + tn*WAVES_N) * WN; };

    const int Kmain = (K / BK) * BK;
    const bool has_tail = (Kmain < K);

    // -------- main loop over K --------
    #pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK) {
        const int kNext = k0 + BK;

        // prefetch next
        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
            else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        // consume current (TW_M × TW_N MFMA calls per kk)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
        if (kNext < Kmain) {
            uint16_t* tA = currA; currA = nextA; nextA = tA;
            uint16_t* tB = currB; currB = nextB; nextB = tB;
            nextA32 = reinterpret_cast<uint32_t*>(nextA);
            nextB32 = reinterpret_cast<uint32_t*>(nextB);
        }
    }

    // -------- tail --------
    if (has_tail) {
        if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);
        else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(nextA, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(nextB, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // -------- stores (masked on borders) --------
    const bool interior =
        (m_left >= BLOCK_M * TW_M) &&
        ((n0 + BN) <= N);

    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm) {
        const int rowBase = m_start + (wave_m + tm*WAVES_M) * WM + lane_group(lane) * 4;
        #pragma unroll
        for (int tn=0; tn<TW_N; ++tn) {
            const int col = n0 + (wave_n + tn*WAVES_N) * WN + lane_row(lane);
            #pragma unroll
            for (int i=0;i<4;++i) {
                const int row = rowBase + i;
                if (interior) {
                    C[(size_t)row * N + col] = acc[tm][tn][i];
                } else {
                    if (row < M_bound && col < N)
                        C[(size_t)row * N + col] = acc[tm][tn][i];
                }
            }
        }
    }
}

// ============================================================================
// Templated grouped MLP2 kernel: C = A·W2 + b2
// Supports TW_M × TW_N per-wave multi-tiles
// ============================================================================

template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=8, int WAVES_K=2,
    int TW_M=1,  int TW_N=2,
    int PAD_K=0
>
__global__ void grouped_mlp2_bf16_bias_kernel_tiled(
    float* __restrict__ C,                 // [sum_tokens, H]
    const float* __restrict__ A,           // [sum_tokens, D]
    const __hip_bfloat16* __restrict__ W2, // [N=H, K=D] row-major
    const __hip_bfloat16* __restrict__ b2, // [N=H] bf16
    const int* __restrict__ expert_offsets,// [E]
    const int* __restrict__ expert_counts, // [E]
    const int* __restrict__ tile2expert,   // [cur_tiles]
    const int* __restrict__ tile2local,    // [cur_tiles]
    int E, int K, int N)
{
    static_assert(WM==16 && WN==16 && WK==16, "MFMA kernel assumes 16x16x16 bf16 tiles.");
    constexpr int LANE_PER_WAVE = 64;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK      = WK * WAVES_K;

    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left   = expert_counts[e]  - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BN;

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // ------------------ LDS ------------------
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;

    uint16_t* sA0_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sA1_u16 = sA0_u16 + (BM * ldA);
    uint16_t* sB0_u16 = sA1_u16 + (BM * ldA);
    uint16_t* sB1_u16 = sB0_u16 + (ldB * BN);

    uint32_t* sA0_u32 = reinterpret_cast<uint32_t*>(sA0_u16);
    uint32_t* sA1_u32 = reinterpret_cast<uint32_t*>(sA1_u16);
    uint32_t* sB0_u32 = reinterpret_cast<uint32_t*>(sB0_u16);
    uint32_t* sB1_u32 = reinterpret_cast<uint32_t*>(sB1_u16);

    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm=0; tm<TW_M; ++tm)
    #pragma unroll
    for (int tn=0; tn<TW_N; ++tn)
        acc[tm][tn] = {0.f,0.f,0.f,0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16* __restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16* __restrict__ b_e = b2 + (size_t)e * (size_t)N;

    const bool alignedA = (((uintptr_t)A  & 0x7)==0) && ((K & 1)==0);
    const bool use128bW = (((uintptr_t)W_e& 0xF)==0) && ((K & 7)==0);

    if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);
    else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t* currA = sA0_u16; uint16_t* nextA = sA1_u16;
    uint16_t* currB = sB0_u16; uint16_t* nextB = sB1_u16;
    uint32_t* nextA32 = sA1_u32;
    uint32_t* nextB32 = sB1_u32;

    auto aRowBase_tm = [&](int tm){ return (wave_m + tm*WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn){ return (wave_n + tn*WAVES_N) * WN; };

    const int Kmain = (K / BK) * BK;
    const bool has_tail = (Kmain < K);

    #pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK) {
        const int kNext = k0 + BK;

        if (kNext < Kmain) {
            if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
            else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
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
        if (alignedA) copy_A_tile_vec_tiled<true,  ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else          copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        if (use128bW) copy_B_tile_vec_tiled<true,  ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);
        else          copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(nextA, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(nextB, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm=0; tm<TW_M; ++tm)
            #pragma unroll
            for (int tn=0; tn<TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // stores (+bias)
    const bool interior =
        (m_left >= BLOCK_M * TW_M) &&
        ((n0 + BN) <= N);

    #pragma unroll
    for (int tn=0; tn<TW_N; ++tn) {
        const int col = n0 + (wave_n + tn*WAVES_N) * WN + lane_row(lane);
        const float bias = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;

        #pragma unroll
        for (int tm=0; tm<TW_M; ++tm) {
            const int rowBase = m_start + (wave_m + tm*WAVES_M) * WM + lane_group(lane) * 4;

            #pragma unroll
            for (int i=0;i<4;++i) {
                const int row = rowBase + i;
                const float v = acc[tm][tn][i] + bias;
                if (interior) {
                    C[(size_t)row * N + col] = v;
                } else {
                    if (row < M_bound && col < N)
                        C[(size_t)row * N + col] = v;
                }
            }
        }
    }
}

// ================================
// MLP1 launcher (templated)
// ================================
template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=8, int WAVES_K=2,
    int TW_M=1,  int TW_N=2,
    int PAD_K=0
>
inline void mlp1(
    float* C, const float* A, const __hip_bfloat16* W1,
    const int* expert_offsets, const int* expert_counts,
    const int* tile2expert, const int* tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM==16 && WN==16 && WK==16, "bf16 MFMA uses 16x16x16 tiles");

    constexpr int LANE_PER_WAVE   = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK      = WK * WAVES_K;

    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    dim3 grid((N + BN - 1) / BN, cur_tiles);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(2 * BM * ldA + 2 * ldB * BN);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp1_bf16_kernel_tiled");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp1_bf16_kernel_tiled<
            WM,WN,WK, WAVES_M,WAVES_N,WAVES_K, TW_M,TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W1, expert_offsets, expert_counts,
        tile2expert, tile2local, E, K, N);
    HIP_CHECK(hipGetLastError());
}

// ================================
// MLP2 launcher (templated)
// ================================
template<
    int WM=16, int WN=16, int WK=16,
    int WAVES_M=1, int WAVES_N=8, int WAVES_K=2,
    int TW_M=1,  int TW_N=2,
    int PAD_K=0
>
inline void mlp2(
    float* C, const float* A, const __hip_bfloat16* W2, const __hip_bfloat16* b2,
    const int* expert_offsets, const int* expert_counts,
    const int* tile2expert, const int* tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM==16 && WN==16 && WK==16, "bf16 MFMA uses 16x16x16 tiles");

    constexpr int LANE_PER_WAVE   = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK      = WK * WAVES_K;

    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    dim3 grid((N + BN - 1) / BN, cur_tiles);
    dim3 block(LANE_PER_WAVE, WAVES_PER_BLOCK);

    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(2 * BM * ldA + 2 * ldB * BN);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp2_bf16_bias_kernel_tiled");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp2_bf16_bias_kernel_tiled<
            WM,WN,WK, WAVES_M,WAVES_N,WAVES_K, TW_M,TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W2, b2, expert_offsets, expert_counts,
        tile2expert, tile2local, E, K, N);
    HIP_CHECK(hipGetLastError());
}