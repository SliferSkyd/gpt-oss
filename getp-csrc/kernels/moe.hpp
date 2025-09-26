#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "../config.hpp"
#include "../memory/mxfp4.hpp"

const int WM = 16;
const int WN = 16;
const int WK = 16;
constexpr int LANE_PER_WAVE = 64;

#ifndef WAVES_M_MLP
#define WAVES_M_MLP 4
#endif
#ifndef WAVES_N_MLP
#define WAVES_N_MLP 8
#endif

#ifndef WAVES_K_MLP
#define WAVES_K_MLP 4
#endif

static_assert(WM == 16 && WN == 16 && WK == 16, "This MFMA microkernel assumes 16x16x16 bf16 tiles.");

constexpr int BLOCK_M_MLP = WM * WAVES_M_MLP; // e.g. 64 if WAVES_M_MLP=4
constexpr int BLOCK_N_MLP = WN * WAVES_N_MLP; // e.g. 64 if WAVES_N_MLP=4
constexpr int WAVES_PER_BLOCK_MLP = WAVES_M_MLP * WAVES_N_MLP;
constexpr int BLOCK_K_MLP = WK * WAVES_K_MLP; // e.g. 16 if WAVES_K_MLP=1
#ifndef PAD_K_MLP
#define PAD_K_MLP 8 // try 2 or 8 if you see LDS conflicts
#endif
static_assert((BLOCK_K_MLP % 2) == 0, "BLOCK_K_MLP must be even (packs 2×bf16).");
static_assert((PAD_K_MLP % 2) == 0, "PAD_K_MLP must be even (u32 pair addressing).");

// ============================================================================
// Templated copy helpers (BM/BN are compile-time so loops vectorize nicely)
// ============================================================================

template <bool Aligned, int LD_A, int BM, int BK>
__device__ inline void copy_A_tile_vec_tiled(
    uint32_t *__restrict__ dst_u32,
    const float *__restrict__ A,
    int m_start, int M_bound, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int pairsPerRow = BK >> 1;
    constexpr int totalPairs = BM * pairsPerRow;

    for (int t = linearT; t < totalPairs; t += threadsPerBlock)
    {
        const int r = t / pairsPerRow; // 0..BM-1
        const int p = t % pairsPerRow; // 0..BK/2-1
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);

        uint32_t val = 0u;
        if (gm < M_bound && gk < K)
        {
            const size_t base = (size_t)gm * K + gk;
            if constexpr (Aligned)
            {
                const float2 v = *reinterpret_cast<const float2 *>(&A[base]);
                // (pack2_bf16_bits_f32 is from your GEMM path)
                val = pack2_bf16_bits_f32(v.x, v.y);
            }
            else
            {
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits_f32(a0, a1);
            }
        }
        reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)r * LD_A >> 1))[p] = val;
    }
}

template <bool Use128b, int LD_B, int BN, int BK>
__device__ inline void copy_B_tile_vec_tiled(
    uint32_t *__restrict__ dst_u32,
    const __hip_bfloat16 *__restrict__ W, // [N,K] row-major
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int pairsPerCol = BK >> 1;

    if constexpr (Use128b)
    {
        constexpr int quadPerCol = BK / 8; // 8 bf16 per 128b
        constexpr int totalQuads = BN * quadPerCol;
        for (int t = linearT; t < totalQuads; t += threadsPerBlock)
        {
            const int c = t / quadPerCol; // 0..BN-1
            const int q = t % quadPerCol; // which 8-elem chunk
            const int gn = n0 + c;
            const int gk8 = kBase + (q << 3);

            uint4 v = {0, 0, 0, 0};
            if (gn < N && (gk8 + 7) < K)
            {
                const uint4 *src = reinterpret_cast<const uint4 *>(&W[(size_t)gn * K + gk8]);
                v = *src;
            }
            else
            {
                __hip_bfloat16 tmp[8] = {};
                for (int i = 0; i < 8 && (gk8 + i) < K && gn < N; i++)
                    tmp[i] = W[(size_t)gn * K + gk8 + i];
                const uint32_t *p = reinterpret_cast<const uint32_t *>(tmp);
                v = make_uint4(p[0], p[1], p[2], p[3]);
            }
            uint32_t *col = reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1));
            const int off = q << 2;
            col[off + 0] = v.x;
            col[off + 1] = v.y;
            col[off + 2] = v.z;
            col[off + 3] = v.w;
        }
    }
    else
    {
        constexpr int totalPairs = BN * pairsPerCol;
        for (int t = linearT; t < totalPairs; t += threadsPerBlock)
        {
            const int c = t / pairsPerCol;
            const int p = t % pairsPerCol;
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);

            uint32_t val = 0u;
            if (gn < N && gk < K)
            {
                const size_t base = (size_t)gn * K + gk;
                if (gk + 1 < K)
                    val = *reinterpret_cast<const uint32_t *>(&W[base]);
                else
                    val = uint32_t(hipbf16_to_bits(W[base]));
            }
            reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = val;
        }
    }
}

__device__ __forceinline__ uint32_t deq2_pack_bf16_u32(
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ scales,
    size_t idx0, size_t idx1, int K, int N)
{
    const float f0 = dequantize_mxfp4_block32(packed, scales, idx0);
    float f1 = 0.f;
    if (idx1 < (size_t)K * (size_t)N)
    {
        f1 = dequantize_mxfp4_block32(packed, scales, idx1);
    }
    return pack2_bf16_bits_f32(f0, f1);
}

// Two outputs from one packed byte + one pre-expanded f32 scale
__device__ __forceinline__ uint32_t deq2_pack_bf16_u32_fscale(
    const uint8_t *__restrict__ packed,
    const float *__restrict__ scales_f32,
    size_t even_idx, // must be even: even_idx and even_idx+1 share the byte/scale
    int K, int N)
{
    const size_t blk = even_idx >> 5;        // /32
    const uint8_t b = packed[even_idx >> 1]; // /2
    const uint8_t nib0 = b & 0xF;
    const uint8_t nib1 = (b >> 4) & 0xF;
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
template <int LD_B>
__device__ inline void copy_B_tile_vec_MLP_MXFP4(
    uint32_t *__restrict__ dst_u32,         // writes u32 (2×bf16) into LDS
    const uint8_t *__restrict__ W_packed_e, // per-expert base (packed nibbles)
    const uint8_t *__restrict__ S_e8m0_e,   // per-expert base (E8M0 scales per 32)
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int BN_ = BLOCK_N_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerCol = BK_ >> 1; // 2 elems per u32
    const int totalPairs = BN_ * pairsPerCol;

    // We store column-major in LDS: column c lives at dst_u32 + (c * LD_B)/2 (u32 indexing)
    for (int t = linearT; t < totalPairs; t += threadsPerBlock)
    {
        const int c = t / pairsPerCol;   // 0..BN_-1
        const int p = t % pairsPerCol;   // 0..pairsPerCol-1
        const int gn = n0 + c;           // global N index (column)
        const int gk = kBase + (p << 1); // row index in K (two per u32)

        uint32_t out = 0u;
        if (gn < N && gk < K)
        {
            // Flattened row-major index over the per-expert matrix [N x K]:
            // idx = gn*K + gk  (your CPU side stored [E, N, K] row-major, and packed by that)
            const size_t idx0 = (size_t)gn * (size_t)K + (size_t)gk;
            const size_t idx1 = idx0 + 1; // tail-safe handled inside deq2_pack
            out = deq2_pack_bf16_u32(W_packed_e, S_e8m0_e, idx0, idx1, K, N);
        }
        // write into LDS in column-major: (c * LD_B) / 2 is u32 stride
        reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = out;
    }
}

// Load B tile: global (MXFP4) -> LDS (bf16 col-major) using f32 scales.
// Stores into LDS as u32 (two bf16 packed). LD_B is LDS stride in elements.
template <int LD_B>
__device__ inline void copy_B_tile_vec_MLP_MXFP4_f32(
    uint32_t *__restrict__ dst_u32,         // writes u32 (2×bf16) into LDS
    const uint8_t *__restrict__ W_packed_e, // per-expert packed nibble base
    const float *__restrict__ S_f32_e,      // per-expert f32 scales base
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    constexpr int BN_ = BLOCK_N_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerCol = BK_ >> 1; // 2 elems per u32
    const int totalPairs = BN_ * pairsPerCol;

    // LDS column-major: column c lives at dst_u32 + (c * LD_B)/2 (u32 addressing)
    for (int t = linearT; t < totalPairs; t += threadsPerBlock)
    {
        const int c = t / pairsPerCol;   // 0..BN_-1
        const int p = t % pairsPerCol;   // 0..pairsPerCol-1
        const int gn = n0 + c;           // global column (N)
        const int gk = kBase + (p << 1); // row in K, even

        uint32_t out = 0u;
        if (gn < N && gk < K)
        {
            // linear index over row-major [N x K]
            const size_t idx0 = (size_t)gn * (size_t)K + (size_t)gk; // even
            out = deq2_pack_bf16_u32_fscale(W_packed_e, S_f32_e, idx0, K, N);
        }
        reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = out;
    }
}

// Vectorized A load: read float2 (64b), convert to 2×bf16 in a single u32
template <bool Aligned, int LD_A>
__device__ inline void copy_A_tile_vec_MLP(uint32_t *__restrict__ dst_u32,
                                           const float *__restrict__ A,
                                           int m_start, int M_bound, int K, int kBase,
                                           int linearT, int threadsPerBlock)
{
    constexpr int BM_ = BLOCK_M_MLP;
    constexpr int BK_ = BLOCK_K_MLP;
    const int pairsPerRow = BK_ >> 1;
    const int totalPairs = BM_ * pairsPerRow;

    for (int t = linearT; t < totalPairs; t += threadsPerBlock)
    {
        const int r = t / pairsPerRow;
        const int p = t % pairsPerRow;
        const int gm = m_start + r;
        const int gk = kBase + (p << 1);

        uint32_t val = 0u;
        if (gm < M_bound && gk < K)
        {
            const size_t base = (size_t)gm * K + gk;
            if constexpr (Aligned)
            {
                // 64-bit aligned path if A is 8B-aligned and K multiple of 2
                const float2 v = *reinterpret_cast<const float2 *>(&A[base]);
                val = pack2_bf16_bits_f32(v.x, v.y);
            }
            else
            {
                const float a0 = A[base];
                const float a1 = (gk + 1 < K) ? A[base + 1] : 0.0f;
                val = pack2_bf16_bits_f32(a0, a1);
            }
        }
        reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)r * LD_A >> 1))[p] = val;
    }
}

__device__ inline bf16x4 make_a_vec_k(const uint16_t *__restrict__ sA,
                                      int ldA, int aRowBase, int kOff, int lane)
{
    const int r = aRowBase + lane_row(lane);
    const int grp = lane_group(lane);
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        v[i] = sA[r * ldA + (kOff + grp * 4 + i)];
    }
    return v;
}

__device__ inline bf16x4 make_b_vec_k(const uint16_t *__restrict__ sB,
                                      int ldB, int bColBase, int kOff, int lane)
{
    const int col = bColBase + lane_row(lane);
    const int grp = lane_group(lane);
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        v[i] = sB[col * ldB + (kOff + grp * 4 + i)];
    }
    return v;
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

__global__ void reduce_tokenwise_expert_outputs(
    float *__restrict__ e_agg,               // [B, H]  (output)
    const float *__restrict__ expert_output, // [sum_tokens, H]
    const int *__restrict__ local_ids,       // [B, Ktok]
    const float *__restrict__ local_wts,     // [B, Ktok]
    int batch_size, int H, int Ktok)
{
    const int b = blockIdx.x;                      // token
    int d = threadIdx.x + blockIdx.y * blockDim.x; // hidden dim

    if (b >= batch_size || d >= H)
        return;

    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < Ktok; ++k)
    {
        const int ci = local_ids[b * Ktok + k]; // compact idx for (b,k)
        if (ci >= 0)
        {
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
    if (token_idx >= batch_size)
        return;

    extern __shared__ int destination_indices[]; // size: experts_per_token

    // One thread per k handles the index math
    if (threadIdx.x < experts_per_token)
    {
        const int k = threadIdx.x;
        const int topk_flat_idx = token_idx * experts_per_token + k;
        const int expert_id = topk_i[topk_flat_idx];

        // Local position inside expert's block (order here does not matter
        // anymore, we will remember the compact index explicitly)
        const int local_idx = atomicAdd(&d_expert_write_idx[expert_id], 1);
        const int compact_idx = d_expert_offsets[expert_id] + local_idx;

        // NEW: remember the exact compact slot for (token, k)
        local_ids[topk_flat_idx] = compact_idx;
        local_wts[topk_flat_idx] = topk_v[topk_flat_idx];

        destination_indices[k] = compact_idx;
    }
    __syncthreads();

    // Coalesced copy of hidden vector into each expert slot
    for (int k = 0; k < experts_per_token; ++k)
    {
        const float *src = d_t + (size_t)token_idx * hidden_dim;
        float *dst = expert_input_buffer + (size_t)destination_indices[k] * hidden_dim;
        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        {
            dst[i] = src[i];
        }
    }
}

__global__ void route_and_pack_fused_kernel(
    const float *__restrict__ x, // [B, H]
    int B, int H,
    const int *__restrict__ topk_i,   // [B, K]
    const float *__restrict__ topk_v, // [B, K]
    int K,
    const int *__restrict__ expert_offsets, // [E] (exclusive prefix)
    int E,
    int *__restrict__ local_ids,   // [B, K] (out)
    float *__restrict__ local_wts, // [B, K] (out)
    float *__restrict__ expert_in) // [sum_tokens, H] (out)
{
    const int e = blockIdx.x;
    if (e >= E)
        return;

    const int T = blockDim.x; // threads per block
    const int tid = threadIdx.x;

    extern __shared__ int _smem[];
    int *flags = _smem;              // [T]
    int *excl = flags + T;           // [T]  (inclusive scan buffer)
    int *kidx = excl + T;            // [T]  (which k matched e, or -1)
    int *toklist = kidx + T;         // [T]  (selected token indices in this chunk)
    int *cmplist = toklist + T;      // [T]  (their compact indices)
    int *carry_shared = cmplist + T; // [1]  (shared carry value)

    if (tid == 0)
        carry_shared[0] = 0; // Initialize shared carry
    __syncthreads();

    // Process tokens in tiles of T to support B > T
    for (int base = 0; base < B; base += T)
    {

        // ---- 1) Flag tokens in this chunk and remember which k matched ----
        int t_global = base + tid;
        int f = 0, kk = -1;
        if (t_global < B)
        {
            const int off = t_global * K;
#pragma unroll
            for (int i = 0; i < K; ++i)
            {
                if (topk_i[off + i] == e)
                {
                    f = 1;
                    kk = i;
                    break;
                }
            }
        }
        flags[tid] = f;
        kidx[tid] = kk;
        __syncthreads();

        // ---- 2) Inclusive scan on flags (Hillis–Steele in-place) ----
        excl[tid] = flags[tid];
        __syncthreads();
        for (int ofs = 1; ofs < T; ofs <<= 1)
        {
            int v = (tid >= ofs) ? excl[tid - ofs] : 0;
            __syncthreads();
            excl[tid] += v;
            __syncthreads();
        }
        const int rank_local = (tid == 0) ? 0 : excl[tid - 1]; // exclusive rank within this chunk
        const int chunk_total = excl[T - 1];                   // total selected in this chunk

        // ---- 3) For selected tokens: compute compact_idx, store lists, fill ids/wts ----
        if (flags[tid])
        {
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
        for (int j = 0; j < chunk_total; ++j)
        {
            const int tkn = toklist[j];
            const int cmp = cmplist[j];
            const float *__restrict__ src = x + (size_t)tkn * H;
            float *__restrict__ dst = expert_in + (size_t)cmp * H;
            for (int i = tid; i < H; i += T)
            {
                dst[i] = src[i];
            }
        }
        __syncthreads();

        // ---- 5) Advance global carry for this expert ----
        if (tid == 0)
            carry_shared[0] += chunk_total;
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

__device__ __forceinline__ float fast_expf(float x)
{
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __expf(x); // fast SFU path on ROCm
#else
    return expf(x);
#endif
}
// Variant of store that adds bf16 bias (same col bias for 4 rows)
template <bool InteriorStore>
__device__ inline void store_c_tile_addbias(
    float *__restrict__ C, const f32x4 &acc,
    const __hip_bfloat16 *__restrict__ bias,
    int M, int N, int m0, int n0, int wave_m, int wave_n, int lane)
{
    const int rowBase = m0 + wave_m * WM + lane_group(lane) * 4;
    const int col = n0 + wave_n * WN + lane_row(lane);

    float b = 0.f;
    if constexpr (InteriorStore)
    {
        b = __bfloat162float(bias[col]);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            C[(size_t)(rowBase + i) * N + col] = acc[i] + b;
        }
    }
    else
    {
        if (col < N)
        {
            b = __bfloat162float(bias[col]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int row = rowBase + i;
                if (row < M)
                    C[(size_t)row * N + col] = acc[i] + b;
            }
        }
    }
}

__global__ void bias_swiglu_epilogue_kernel(
    const float *__restrict__ mlp1_out,     // [sum_tokens, 2*D]
    const __hip_bfloat16 *__restrict__ b1,  // [n_experts, 2*D]
    const int *__restrict__ expert_offsets, // [n_experts]
    const int *__restrict__ expert_counts,  // [n_experts]
    int n_experts,
    float *__restrict__ gate_up, // [sum_tokens, D]
    int D,
    int sum_tokens,
    float clamp_limit,
    float alpha_silu /* = 1.702f */)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)sum_tokens * D;
    if (idx >= total)
        return;

    const int d = (int)(idx % D);

    // ---- warp-cooperative token/expert resolution ----
    const int lane = threadIdx.x & (warpSize - 1);
    const size_t warp_first = idx - lane;
    const int t0 = (int)(warp_first / D);
    const int t1 = (int)((warp_first + (warpSize - 1)) / D);

    int t, e;
    if (t0 == t1)
    {
        // Same token across the whole wave: binary-search once in lane 0
        t = t0;
        int lo = 0, hi = n_experts;
        if (lane == 0)
        {
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                const int start = expert_offsets[mid];
                const int end = start + expert_counts[mid];
                if (t >= end)
                    lo = mid + 1;
                else if (t < start)
                    hi = mid;
                else
                {
                    lo = mid;
                    break;
                }
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
    }
    else
    {
        // Fallback (rare if D is large): per-thread search
        t = (int)(idx / D);
        int lo = 0, hi = n_experts;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            const int start = expert_offsets[mid];
            const int end = start + expert_counts[mid];
            if (t >= end)
                lo = mid + 1;
            else if (t < start)
                hi = mid;
            else
            {
                lo = mid;
                break;
            }
        }
        e = lo;
    }

    // ---- precompute bases to cut address math & lifetimes ----
    const size_t t2D = (size_t)t * (2 * (size_t)D);
    const size_t e2D = (size_t)e * (2 * (size_t)D);

    // bias in bf16 -> f32
    const float bg = __bfloat162float(b1[e2D + (size_t)(2 * d + 0)]);
    const float bu = __bfloat162float(b1[e2D + (size_t)(2 * d + 1)]);

    // load + bias
    float gx = mlp1_out[t2D + (size_t)(2 * d + 0)] + bg;
    float uy = mlp1_out[t2D + (size_t)(2 * d + 1)] + bu;

    // clamp early to keep ranges tight (reduces VGPR lifetimes)
    gx = fminf(fmaxf(gx, -clamp_limit), clamp_limit);
    uy = fminf(fmaxf(uy, -clamp_limit), clamp_limit);

    // SiLU(x) = x * sigmoid(alpha*x)
    const float sig = 1.0f / (1.0f + fast_expf(-alpha_silu * gx));
    gx *= sig; // reuse gx register as swish

    // swish * (uy + 1) using FMA to save one temp & shrink live ranges
    gate_up[(size_t)t * (size_t)D + d] = fmaf(gx, uy, gx);
}

// NOTE: we accept *f32* scale layer pointers now.
__global__ void grouped_mlp1_mxfp4_kernel(
    float *__restrict__ C,                         // [sum_tokens, 2D]
    const float *__restrict__ A,                   // [sum_tokens, H]  (float), row-major
    const uint8_t *__restrict__ W1_packed_layer,   // [E, ceil((2D*H)/2)]
    const float *__restrict__ S1_scales_f32_layer, // [E, ceil((2D*H)/32)]
    const int *__restrict__ expert_offsets,        // [E]
    const int *__restrict__ expert_counts,         // [E]
    const int *__restrict__ tile2expert,           // [num_mtiles]
    const int *__restrict__ tile2local,            // [num_mtiles]
    int /*E*/, int K /*=H*/, int N /*=2D*/)
{
    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left = expert_counts[e] - tile_m * BLOCK_M_MLP;
    if (m_left <= 0)
        return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K_MLP + PAD_K_MLP;
    const int ldB = BLOCK_K_MLP + PAD_K_MLP;

    uint16_t *sA0_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sA1_u16 = sA0_u16 + (BLOCK_M_MLP * ldA);
    uint16_t *sB0_u16 = sA1_u16 + (BLOCK_M_MLP * ldA);
    uint16_t *sB1_u16 = sB0_u16 + (ldB * BLOCK_N_MLP);

    uint32_t *sA0_u32 = reinterpret_cast<uint32_t *>(sA0_u16);
    uint32_t *sA1_u32 = reinterpret_cast<uint32_t *>(sA1_u16);
    uint32_t *sB0_u32 = reinterpret_cast<uint32_t *>(sB0_u16);
    uint32_t *sB1_u32 = reinterpret_cast<uint32_t *>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;

    const size_t seg_elems = (size_t)N * (size_t)K;
    const size_t seg_packed_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks = (seg_elems + 31) / 32;

    const uint8_t *__restrict__ W_e_packed = W1_packed_layer + (size_t)e * seg_packed_bytes;
    const float *__restrict__ S_f32_e = S1_scales_f32_layer + (size_t)e * seg_blocks;

    const bool alignedA = (((uintptr_t)A & 0x7) == 0) && ((K & 1) == 0);

    if (alignedA)
        copy_A_tile_vec_MLP<true, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else
        copy_A_tile_vec_MLP<false, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(sB0_u32, W_e_packed, S_f32_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t *currA = sA0_u16;
    uint16_t *nextA = sA1_u16;
    uint16_t *currB = sB0_u16;
    uint16_t *nextB = sB1_u16;
    uint32_t *nextA32 = sA1_u32;
    uint32_t *nextB32 = sB1_u32;

    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    const int BK_ = BLOCK_K_MLP;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_)
    {
        const int kNext = k0 + BK_;
        if (kNext < Kmain)
        {
            if (alignedA)
                copy_A_tile_vec_MLP<true, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else
                copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

#pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK)
        {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
        if (kNext < Kmain)
        {
            uint16_t *tA = currA;
            currA = nextA;
            nextA = tA;
            uint16_t *tB = currB;
            currB = nextB;
            nextB = tB;
            nextA32 = reinterpret_cast<uint32_t *>(nextA);
            nextB32 = reinterpret_cast<uint32_t *>(nextB);
        }
    }

    if (has_tail)
    {
        if (alignedA)
            copy_A_tile_vec_MLP<true, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else
            copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK)
        {
            bf16x4 avec = make_a_vec_k(nextA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(nextB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }

    const int rowBase = m_start + wave_m * WM + lane_group(lane) * 4;
    const int col = n0 + wave_n * WN + lane_row(lane);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int row = rowBase + i;
        if (row < M_bound && col < N)
        {
            C[(size_t)row * N + col] = acc[i];
        }
    }
}

__global__ void grouped_mlp2_mxfp4_bias_kernel(
    float *__restrict__ C,                         // [sum_tokens, H]
    const float *__restrict__ A,                   // [sum_tokens, D]  (float), row-major
    const uint8_t *__restrict__ W2_packed_layer,   // [E, ceil((H*D)/2)]
    const float *__restrict__ S2_scales_f32_layer, // [E, ceil((H*D)/32)]
    const __hip_bfloat16 *__restrict__ b2,         // [E, H] bf16
    const int *__restrict__ expert_offsets, const int *__restrict__ expert_counts,
    const int *__restrict__ tile2expert, const int *__restrict__ tile2local,
    int /*E*/, int K /*=D*/, int N /*=H*/)
{
    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M_MLP;
    const int m_left = expert_counts[e] - tile_m * BLOCK_M_MLP;
    if (m_left <= 0)
        return;

    const int n0 = blockIdx.x * BLOCK_N_MLP;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N_MLP;
    const int wave_n = wave % WAVES_N_MLP;

    extern __shared__ uint8_t smemRaw[];
    const int ldA = BLOCK_K_MLP + PAD_K_MLP;
    const int ldB = BLOCK_K_MLP + PAD_K_MLP;

    uint16_t *sA0_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sA1_u16 = sA0_u16 + (BLOCK_M_MLP * ldA);
    uint16_t *sB0_u16 = sA1_u16 + (BLOCK_M_MLP * ldA);
    uint16_t *sB1_u16 = sB0_u16 + (ldB * BLOCK_N_MLP);

    uint32_t *sA0_u32 = reinterpret_cast<uint32_t *>(sA0_u16);
    uint32_t *sA1_u32 = reinterpret_cast<uint32_t *>(sA1_u16);
    uint32_t *sB0_u32 = reinterpret_cast<uint32_t *>(sB0_u16);
    uint32_t *sB1_u32 = reinterpret_cast<uint32_t *>(sB1_u16);

    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;

    const size_t seg_elems = (size_t)N * (size_t)K;
    const size_t seg_packed_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks = (seg_elems + 31) / 32;

    const uint8_t *__restrict__ W_e_packed = W2_packed_layer + (size_t)e * seg_packed_bytes;
    const float *__restrict__ S_f32_e = S2_scales_f32_layer + (size_t)e * seg_blocks;
    const __hip_bfloat16 *__restrict__ b_e = b2 + (size_t)e * (size_t)N;

    const bool alignedA = (((uintptr_t)A & 0x7) == 0) && ((K & 1) == 0);

    if (alignedA)
        copy_A_tile_vec_MLP<true, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else
        copy_A_tile_vec_MLP<false, /*LD_A=*/ldA>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(sB0_u32, W_e_packed, S_f32_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t *currA = sA0_u16;
    uint16_t *nextA = sA1_u16;
    uint16_t *currB = sB0_u16;
    uint16_t *nextB = sB1_u16;
    uint32_t *nextA32 = sA1_u32;
    uint32_t *nextB32 = sB1_u32;

    const int aRowBase = wave_m * WM;
    const int bColBase = wave_n * WN;

    const int BK_ = BLOCK_K_MLP;
    const int Kmain = (K / BK_) * BK_;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK_)
    {
        const int kNext = k0 + BK_;
        if (kNext < Kmain)
        {
            if (alignedA)
                copy_A_tile_vec_MLP<true, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else
                copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

#pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK)
        {
            bf16x4 avec = make_a_vec_k(currA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(currB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }

        __syncthreads();
        if (kNext < Kmain)
        {
            uint16_t *tA = currA;
            currA = nextA;
            nextA = tA;
            uint16_t *tB = currB;
            currB = nextB;
            nextB = tB;
            nextA32 = reinterpret_cast<uint32_t *>(nextA);
            nextB32 = reinterpret_cast<uint32_t *>(nextB);
        }
    }

    if (has_tail)
    {
        if (alignedA)
            copy_A_tile_vec_MLP<true, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else
            copy_A_tile_vec_MLP<false, ldA>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        copy_B_tile_vec_MLP_MXFP4_f32</*LD_B=*/ldB>(nextB32, W_e_packed, S_f32_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BLOCK_K_MLP; kk += WK)
        {
            bf16x4 avec = make_a_vec_k(nextA, ldA, aRowBase, kk, lane);
            bf16x4 bvec = make_b_vec_k(nextB, ldB, bColBase, kk, lane);
            acc = mfma_16x16x16_bf16(avec, bvec, acc);
        }
        __syncthreads();
    }

    const int col = n0 + wave_n * WN + lane_row(lane);
    const float bias = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;
    const int rowBase = m_start + wave_m * WM + lane_group(lane) * 4;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int row = rowBase + i;
        if (row < M_bound && col < N)
        {
            C[(size_t)row * N + col] = acc[i] + bias;
        }
    }
}

// ============================================================================
// Templated grouped MLP1 kernel: C = A·W1  (no bias)
// Supports per-wave multi-tiles: TW_M × TW_N (each is >=1)
// ============================================================================

template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
__global__ void grouped_mlp1_bf16_kernel_tiled(
    float *__restrict__ C,                  // [sum_tokens, 2*D]
    const float *__restrict__ A,            // [sum_tokens, H]
    const __hip_bfloat16 *__restrict__ W1,  // [N=2*D, K=H] row-major
    const int *__restrict__ expert_offsets, // [E]
    const int *__restrict__ expert_counts,  // [E]
    const int *__restrict__ tile2expert,    // [cur_tiles]
    const int *__restrict__ tile2local,     // [cur_tiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA kernel assumes 16x16x16 bf16 tiles.");
    constexpr int LANE_PER_WAVE = 64;

    constexpr int BLOCK_M = WM * WAVES_M; // per-block (one buffer)
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK = WK * WAVES_K;

    // expanded per-stage tile (what we actually load/compute)
    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M; // base of this "logical" block
    const int m_left = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0)
        return;

    const int n0 = blockIdx.x * BN; // grid.x in units of BN

    const int lane = threadIdx.x; // 0..63
    const int wave = threadIdx.y; // 0..(WAVES_M*WAVES_N-1)
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // ------------------ LDS double buffers ------------------
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;

    uint16_t *sA0_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sA1_u16 = sA0_u16 + (BM * ldA);
    uint16_t *sB0_u16 = sA1_u16 + (BM * ldA);
    uint16_t *sB1_u16 = sB0_u16 + (ldB * BN);

    uint32_t *sA0_u32 = reinterpret_cast<uint32_t *>(sA0_u16);
    uint32_t *sA1_u32 = reinterpret_cast<uint32_t *>(sA1_u16);
    uint32_t *sB0_u32 = reinterpret_cast<uint32_t *>(sB0_u16);
    uint32_t *sB1_u32 = reinterpret_cast<uint32_t *>(sB1_u16);

    // ------------------ accumulators ------------------
    f32x4 acc[TW_M][TW_N];
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
            acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16 *__restrict__ W_e = W1 + (size_t)e * (size_t)N * (size_t)K;

    // alignment guards
    const bool alignedA = (((uintptr_t)A & 0x7) == 0) && ((K & 1) == 0);
    const bool use128bW = (((uintptr_t)W_e & 0xF) == 0) && ((K & 7) == 0);

    // -------- preload k=0 --------
    if (alignedA)
        copy_A_tile_vec_tiled<true, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else
        copy_A_tile_vec_tiled<false, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    if (use128bW)
        copy_B_tile_vec_tiled<true, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);
    else
        copy_B_tile_vec_tiled<false, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t *currA = sA0_u16;
    uint16_t *nextA = sA1_u16;
    uint16_t *currB = sB0_u16;
    uint16_t *nextB = sB1_u16;
    uint32_t *nextA32 = sA1_u32;
    uint32_t *nextB32 = sB1_u32;

    // per-wave bases for sub-tiles (inside BM×BN)
    auto aRowBase_tm = [&](int tm)
    { return (wave_m + tm * WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn)
    { return (wave_n + tn * WAVES_N) * WN; };

    const int Kmain = (K / BK) * BK;
    const bool has_tail = (Kmain < K);

// -------- main loop over K --------
#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK)
    {
        const int kNext = k0 + BK;

        // prefetch next
        if (kNext < Kmain)
        {
            if (alignedA)
                copy_A_tile_vec_tiled<true, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else
                copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            if (use128bW)
                copy_B_tile_vec_tiled<true, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
            else
                copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

// consume current (TW_M × TW_N MFMA calls per kk)
#pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase_tm(tm), kk, lane);

#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase_tn(tn), kk, lane);

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
        if (kNext < Kmain)
        {
            uint16_t *tA = currA;
            currA = nextA;
            nextA = tA;
            uint16_t *tB = currB;
            currB = nextB;
            nextB = tB;
            nextA32 = reinterpret_cast<uint32_t *>(nextA);
            nextB32 = reinterpret_cast<uint32_t *>(nextB);
        }
    }

    // -------- tail --------
    if (has_tail)
    {
        if (alignedA)
            copy_A_tile_vec_tiled<true, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else
            copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        if (use128bW)
            copy_B_tile_vec_tiled<true, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);
        else
            copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(nextA, ldA, aRowBase_tm(tm), kk, lane);

#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(nextB, ldB, bColBase_tn(tn), kk, lane);

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // -------- stores (masked on borders) --------
    const bool interior =
        (m_left >= BLOCK_M * TW_M) &&
        ((n0 + BN) <= N);

#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    {
        const int rowBase = m_start + (wave_m + tm * WAVES_M) * WM + lane_group(lane) * 4;
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
        {
            const int col = n0 + (wave_n + tn * WAVES_N) * WN + lane_row(lane);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int row = rowBase + i;
                if (interior)
                {
                    C[(size_t)row * N + col] = acc[tm][tn][i];
                }
                else
                {
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

template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
__global__ void grouped_mlp2_bf16_bias_kernel_tiled(
    float *__restrict__ C,                  // [sum_tokens, H]
    const float *__restrict__ A,            // [sum_tokens, D]
    const __hip_bfloat16 *__restrict__ W2,  // [N=H, K=D] row-major
    const __hip_bfloat16 *__restrict__ b2,  // [N=H] bf16
    const int *__restrict__ expert_offsets, // [E]
    const int *__restrict__ expert_counts,  // [E]
    const int *__restrict__ tile2expert,    // [cur_tiles]
    const int *__restrict__ tile2local,     // [cur_tiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA kernel assumes 16x16x16 bf16 tiles.");
    constexpr int LANE_PER_WAVE = 64;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK = WK * WAVES_K;

    constexpr int BM = BLOCK_M * TW_M;
    constexpr int BN = BLOCK_N * TW_N;

    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0)
        return;

    const int n0 = blockIdx.x * BN;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    // ------------------ LDS ------------------
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;

    uint16_t *sA0_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sA1_u16 = sA0_u16 + (BM * ldA);
    uint16_t *sB0_u16 = sA1_u16 + (BM * ldA);
    uint16_t *sB1_u16 = sB0_u16 + (ldB * BN);

    uint32_t *sA0_u32 = reinterpret_cast<uint32_t *>(sA0_u16);
    uint32_t *sA1_u32 = reinterpret_cast<uint32_t *>(sA1_u16);
    uint32_t *sB0_u32 = reinterpret_cast<uint32_t *>(sB0_u16);
    uint32_t *sB1_u32 = reinterpret_cast<uint32_t *>(sB1_u16);

    f32x4 acc[TW_M][TW_N];
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
            acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16 *__restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16 *__restrict__ b_e = b2 + (size_t)e * (size_t)N;

    const bool alignedA = (((uintptr_t)A & 0x7) == 0) && ((K & 1) == 0);
    const bool use128bW = (((uintptr_t)W_e & 0xF) == 0) && ((K & 7) == 0);

    if (alignedA)
        copy_A_tile_vec_tiled<true, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);
    else
        copy_A_tile_vec_tiled<false, ldA, BM, BK>(sA0_u32, A, m_start, M_bound, K, 0, linearT, threadsPerBlock);

    if (use128bW)
        copy_B_tile_vec_tiled<true, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);
    else
        copy_B_tile_vec_tiled<false, ldB, BN, BK>(sB0_u32, W_e, n0, N, K, 0, linearT, threadsPerBlock);

    __syncthreads();

    uint16_t *currA = sA0_u16;
    uint16_t *nextA = sA1_u16;
    uint16_t *currB = sB0_u16;
    uint16_t *nextB = sB1_u16;
    uint32_t *nextA32 = sA1_u32;
    uint32_t *nextB32 = sB1_u32;

    auto aRowBase_tm = [&](int tm)
    { return (wave_m + tm * WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn)
    { return (wave_n + tn * WAVES_N) * WN; };

    const int Kmain = (K / BK) * BK;
    const bool has_tail = (Kmain < K);

#pragma unroll 1
    for (int k0 = 0; k0 < Kmain; k0 += BK)
    {
        const int kNext = k0 + BK;

        if (kNext < Kmain)
        {
            if (alignedA)
                copy_A_tile_vec_tiled<true, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);
            else
                copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, kNext, linearT, threadsPerBlock);

            if (use128bW)
                copy_B_tile_vec_tiled<true, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
            else
                copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, kNext, linearT, threadsPerBlock);
        }

#pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(currA, ldA, aRowBase_tm(tm), kk, lane);

#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(currB, ldB, bColBase_tn(tn), kk, lane);

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
        if (kNext < Kmain)
        {
            uint16_t *tA = currA;
            currA = nextA;
            nextA = tA;
            uint16_t *tB = currB;
            currB = nextB;
            nextB = tB;
            nextA32 = reinterpret_cast<uint32_t *>(nextA);
            nextB32 = reinterpret_cast<uint32_t *>(nextB);
        }
    }

    if (has_tail)
    {
        if (alignedA)
            copy_A_tile_vec_tiled<true, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);
        else
            copy_A_tile_vec_tiled<false, ldA, BM, BK>(nextA32, A, m_start, M_bound, K, Kmain, linearT, threadsPerBlock);

        if (use128bW)
            copy_B_tile_vec_tiled<true, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);
        else
            copy_B_tile_vec_tiled<false, ldB, BN, BK>(nextB32, W_e, n0, N, K, Kmain, linearT, threadsPerBlock);

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WK)
        {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(nextA, ldA, aRowBase_tm(tm), kk, lane);

#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(nextB, ldB, bColBase_tn(tn), kk, lane);

#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    // stores (+bias)
    const bool interior =
        (m_left >= BLOCK_M * TW_M) &&
        ((n0 + BN) <= N);

#pragma unroll
    for (int tn = 0; tn < TW_N; ++tn)
    {
        const int col = n0 + (wave_n + tn * WAVES_N) * WN + lane_row(lane);
        const float bias = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;

#pragma unroll
        for (int tm = 0; tm < TW_M; ++tm)
        {
            const int rowBase = m_start + (wave_m + tm * WAVES_M) * WM + lane_group(lane) * 4;

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int row = rowBase + i;
                const float v = acc[tm][tn][i] + bias;
                if (interior)
                {
                    C[(size_t)row * N + col] = v;
                }
                else
                {
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
template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
inline void mlp1(
    float *C, const float *A, const __hip_bfloat16 *W1,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");

    constexpr int LANE_PER_WAVE = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK = WK * WAVES_K;

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
            WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W1, expert_offsets, expert_counts,
        tile2expert, tile2local, E, K, N);
    HIP_CHECK(hipGetLastError());
}

// ================================
// MLP2 launcher (templated)
// ================================
template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
inline void mlp2(
    float *C, const float *A, const __hip_bfloat16 *W2, const __hip_bfloat16 *b2,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");

    constexpr int LANE_PER_WAVE = 64;
    constexpr int WAVES_PER_BLOCK = WAVES_M * WAVES_N;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK = WK * WAVES_K;

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
            WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W2, b2, expert_offsets, expert_counts,
        tile2expert, tile2local, E, K, N);
    HIP_CHECK(hipGetLastError());
}

// --- compact score/index pair for reductions ---
struct __align__(8) ScorePick
{
    float v;
    int i;
};

__device__ __forceinline__ bool better_pick(ScorePick a, ScorePick b, float eps)
{
    // epsilon-stable: prefer larger; if ~equal within eps*max(|.|), prefer smaller index
    float thr = eps * fmaxf(fabsf(a.v), fabsf(b.v));
    if (a.v > b.v + thr)
        return true;
    if (fabsf(a.v - b.v) <= thr && a.i < b.i)
        return true;
    return false;
}

// block-wide reduction to best pick; uses shared memory array of ScorePick size=blockDim.x
__device__ __forceinline__ ScorePick reduce_block_best(ScorePick local, ScorePick *smem, float eps)
{
    const int tid = threadIdx.x;
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            ScorePick r = smem[tid + s];
            if (better_pick(r, smem[tid], eps))
                smem[tid] = r;
        }
        __syncthreads();
    }
    return smem[0]; // valid at tid==0
}

// Kernel: one block per token. Reads router_score once to smem (with bf16 bias).
// Uses a visited bitmap to avoid mutating the score array for K rounds.
template <int KSEL>
__global__ void route_select_softmax_kernel(
    const float *__restrict__ router_score,       // [B, E]
    const __hip_bfloat16 *__restrict__ bias_bf16, // [E]
    int n_experts,
    float *__restrict__ topk_v_out, // [B, KSEL]
    int *__restrict__ topk_i_out)   // [B, KSEL]
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    const float *s_in = router_score + (size_t)b * n_experts;
    float *v_out = topk_v_out + (size_t)b * KSEL;
    int *i_out = topk_i_out + (size_t)b * KSEL;

    // Shared memory layout:
    // [ scores(E floats) | visited(E bytes) | scratch(blockDim.x ScorePick) | sel_idx(K) | sel_val(K) ]
    extern __shared__ unsigned char smem_raw[];
    float *scores = reinterpret_cast<float *>(smem_raw);
    unsigned char *visited = reinterpret_cast<unsigned char *>(scores + n_experts);

    // align the rest to 8 bytes for ScorePick
    uintptr_t p = reinterpret_cast<uintptr_t>(visited + n_experts);
    p = (p + 7u) & ~uintptr_t(7u);
    ScorePick *scratch = reinterpret_cast<ScorePick *>(p);
    int *sel_idx = reinterpret_cast<int *>(scratch + blockDim.x);
    float *sel_val = reinterpret_cast<float *>(sel_idx + KSEL);

    // 1) load scores (+bf16 bias) and clear visited flags
    for (int i = tid; i < n_experts; i += blockDim.x)
    {
        scores[i] = s_in[i] + __bfloat162float(bias_bf16[i]);
        visited[i] = 0;
    }
    if (tid < KSEL)
    {
        sel_idx[tid] = -1;
        sel_val[tid] = -INFINITY;
    }
    __syncthreads();

    // 2) K selections with epsilon-stable tie-break
    constexpr float EPS = 1e-6f;
#pragma unroll
    for (int sel = 0; sel < KSEL; ++sel)
    {
        // local scan
        ScorePick best = {-INFINITY, n_experts};
        for (int i = tid; i < n_experts; i += blockDim.x)
        {
            if (!visited[i])
            {
                ScorePick cand = {scores[i], i};
                if (better_pick(cand, best, EPS))
                    best = cand;
            }
        }
        // reduce to block best
        ScorePick blk = reduce_block_best(best, scratch, EPS);

        if (tid == 0)
        {
            sel_idx[sel] = blk.i;
            sel_val[sel] = blk.v;
            if (blk.i < n_experts)
                visited[blk.i] = 1; // mark used
        }
        __syncthreads();
    }

    // 3) softmax over the K selected only
    if (tid == 0)
    {
        float mx = sel_val[0];
#pragma unroll
        for (int i = 1; i < KSEL; ++i)
            mx = fmaxf(mx, sel_val[i]);
        float ex[KSEL];
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < KSEL; ++i)
        {
            ex[i] = expf(sel_val[i] - mx);
            sum += ex[i];
        }
        const float inv = 1.f / sum;
#pragma unroll
        for (int i = 0; i < KSEL; ++i)
        {
            v_out[i] = ex[i] * inv; // normalized
            i_out[i] = sel_idx[i];
        }
    }
}

// Host launcher (specialized for KSEL=4 to match your configs)
inline void run_route_select_softmax_fused(
    const float *router_score,       // [B, E]
    const __hip_bfloat16 *bias_bf16, // [E]
    int B, int E, int K,
    float *topk_v_out, int *topk_i_out,
    hipStream_t stream)
{
    if (K != 4)
    {
        fprintf(stderr, "[route_select] Unsupported experts_per_token=%d. Add another specialization.\n", K);
        abort();
    }

    // pick threads = nextPow2(E) clamped to [64, 512]
    auto next_pow2 = [](int x)
    { int v=1; while (v<x) v<<=1; return v; };
    int threads = next_pow2(E);
    if (threads < 64)
        threads = 64;
    if (threads > 512)
        threads = 512;

    // shared memory:
    // E*f32 + E*u8 + threads*sizeof(ScorePick) + K*(sizeof(int)+sizeof(float))
    const size_t shmem_bytes =
        (size_t)E * sizeof(float) +
        (size_t)E * sizeof(unsigned char) +
        (size_t)threads * sizeof(ScorePick) +
        (size_t)K * (sizeof(int) + sizeof(float));

    dim3 grid(B), block(threads);
    route_select_softmax_kernel<4><<<grid, block, shmem_bytes, stream>>>(
        router_score, bias_bf16, E, topk_v_out, topk_i_out);
    HIP_CHECK(hipGetLastError());
}


// ---- A: global fp32 -> LDS bf16 (vectorized) ----
// Loads float4 (16B) then packs to two u32 (4 bf16).
template <int BLOCK_M, int BLOCK_K, int LD_A>
__device__ inline void copy_A_tile_vec128_fp32(uint32_t *__restrict__ dst_u32,
                                               const float *__restrict__ A,
                                               int m_start, int M, int K, int kBase,
                                               int linearT, int threadsPerBlock)
{
    static_assert((BLOCK_K % 4) == 0, "copy_A_tile_vec128_fp32: BLOCK_K must be multiple of 4 (floats per 16B).");

    // 16B unit along K = 4 floats = 4 bf16 -> 2 u32 pairs
    constexpr int quadsPerRow = BLOCK_K / 4;
    const int totalQuads = BLOCK_M * quadsPerRow;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock)
    {
        const int r = t / quadsPerRow; // row within BLOCK_M
        const int q = t % quadsPerRow; // 16B unit index along K
        const int gm = m_start + r;
        const int gk4 = kBase + (q << 2); // float index (4 per 16B)

        uint32_t p0 = 0u, p1 = 0u;
        if (gm < M)
        {
            const size_t base = (size_t)gm * K + gk4;
            if (gk4 + 3 < K && is_aligned_16B(&A[base]))
            {
                const float4 v = *reinterpret_cast<const float4 *>(&A[base]); // 16B coalesced
                p0 = pack2_bf16_bits_f32(v.x, v.y);
                p1 = pack2_bf16_bits_f32(v.z, v.w);
            }
            else
            {
                // tail/unaligned path
                float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int i = 0; i < 4 && (gk4 + i) < K; ++i)
                    tmp[i] = A[base + i];
                p0 = pack2_bf16_bits_f32(tmp[0], tmp[1]);
                p1 = pack2_bf16_bits_f32(tmp[2], tmp[3]);
            }
        }
        // write 2×u32 pairs for this 16B chunk
        // dst_u32 is in u32-pair addressing (LD_A is in bf16, so >>1 gives pairs per row)
        uint32_t *row = reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)r * LD_A >> 1));
        const int off = (q << 1);
        row[off + 0] = p0;
        row[off + 1] = p1;
    }
}


// ===================================
// Grouped MLP1 (bf16 weights, no bias) in unified-MFMA style
// C = A * W1   for a subset of rows selected by (expert_offsets, tile2local)
// ===================================
template <
    // MFMA micro-tile
    int WM, int WN, int WK,
    // Block waves (logical)
    int WAVES_M, int WAVES_N, int WAVES_K,
    // Tiles per wave (multi-tile per wave)
    int TW_M, int TW_N,
    // Padding along K in LDS
    int PAD_K_MC>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2) void grouped_mlp1_bf16_kernel_unified(
    float *__restrict__ C,                    // [sum_tokens, N]
    const float *__restrict__ A,              // [sum_tokens, K]
    const __hip_bfloat16 *__restrict__ Wbf16, // [E, N, K] row-major per expert (flat Base + e*N*K)
    const int *__restrict__ expert_offsets,   // [E]
    const int *__restrict__ expert_counts,    // [E]
    const int *__restrict__ tile2expert,      // [num_mtiles]
    const int *__restrict__ tile2local,       // [num_mtiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "This kernel assumes 16x16x16 bf16 MFMA.");
    static_assert((WAVES_M % TW_M) == 0 && (WAVES_N % TW_N) == 0, "TW_* must divide WAVES_*");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    constexpr int WAVES_PER_BLOCK_E = WAVES_M_E * WAVES_N_E;
    static_assert((BLOCK_K % 2) == 0, "BLOCK_K must be even");
    // static_assert((PAD_K_MC % 2) == 0, "PAD_K_MC must be even");

    // ---- which expert / which M tile am I serving? ----
    const int mtile_id = blockIdx.y; // your scheduler sets this
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M; // base row (M) for this logical tile
    const int m_left = expert_counts[e] - tile_m * BLOCK_M;   // rows left from this base
    if (m_left <= 0)
        return;

    const int n0 = blockIdx.x * BLOCK_N; // standard N tile origin
    const int m0 = m_start;              // origin in M for this logical block

    // ---- standard unified indexing within the block ----
    const int lane = threadIdx.x; // 0..63
    const int wave = threadIdx.y; // 0..WAVES_PER_BLOCK_E-1
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    const int aBase0 = tile_m0 * WM;
    const int bBase0 = tile_n0 * WN;

    // ===== Single-buffer LDS =====
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t *sA_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sB_u16 = sA_u16 + (BLOCK_M * ldA);

    uint32_t *sA_u32 = reinterpret_cast<uint32_t *>(sA_u16);
    uint32_t *sB_u32 = reinterpret_cast<uint32_t *>(sB_u16);

    // accumulators
    f32x4 acc[TW_M][TW_N];
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
            acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // Expert base for W
    const __hip_bfloat16 *__restrict__ W_e = Wbf16 + (size_t)e * (size_t)N * (size_t)K;

    // Decide vectorization once
    const bool alignedA = (((uintptr_t)A & 0xF) == 0) && ((K & 3) == 0);   // 16B & K%4==0 (wider path)
    const bool use128bW = (((uintptr_t)W_e & 0xF) == 0) && ((K & 7) == 0); // 16B & K%8==0

    // Iterate K in chunks: copy -> sync -> compute
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        copy_A_tile_vec128_fp32<BLOCK_M, BLOCK_K, ldA>(
            sA_u32, A, m0, m_start + m_left, K, k0, linearT, threadsPerBlock);

        copy_B_tile_vec128_bf16<BLOCK_N, BLOCK_K, ldB>(
            sB_u32, W_e, n0, N, K, k0, linearT, threadsPerBlock);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK)
        {
            bf16x4 avec[TW_M];
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
            {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase, kk, lane);
            }
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // Stores
    const bool interior =
        ((m0 + BLOCK_M) <= (m_start + m_left)) &&
        ((n0 + BLOCK_N) <= N);
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    {
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
        {
            const int wmt = tile_m0 + tm;
            const int wnt = tile_n0 + tn;
            if (interior)
                store_c_tile<WM, WN, true>(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
            else
                store_c_tile<WM, WN, false>(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
        }
    }
}

// ================================
// MLP1 launcher (bf16, no bias)
// ================================
template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K_MC = 16 // good default to avoid LDS bank conflicts
    >
inline void mlp1_optimized(
    float *C, const float *A, const __hip_bfloat16 *W1, // W1 is [E,N,K] flat
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;

    // grid.x over N, grid.y over "logical tiles" (expert-local M tiles)
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, cur_tiles);
    dim3 block(64, (WAVES_M / TW_M) * (WAVES_N / TW_N));

    // single-buffer shared memory
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(BLOCK_M * ldA + ldB * BLOCK_N);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp1_bf16_kernel_unified");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp1_bf16_kernel_unified<
            WM, WN, WK,
            WAVES_M, WAVES_N, WAVES_K,
            TW_M, TW_N,
            PAD_K_MC>),
        grid, block, shmem_bytes, s,
        C, A, W1,
        expert_offsets, expert_counts,
        tile2expert, tile2local,
        E, K, N);
    HIP_CHECK(hipGetLastError());
}


template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
__global__ void grouped_mlp2_bf16_bias_kernel_tiled_optimized(
    float *__restrict__ C,                  // [sum_tokens, H]
    const float *__restrict__ A,            // [sum_tokens, D]
    const __hip_bfloat16 *__restrict__ W2,  // [N=H, K=D] row-major
    const __hip_bfloat16 *__restrict__ b2,  // [N=H] bf16
    const int *__restrict__ expert_offsets, // [E]
    const int *__restrict__ expert_counts,  // [E]
    const int *__restrict__ tile2expert,    // [cur_tiles]
    const int *__restrict__ tile2local,     // [cur_tiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA 16x16x16 bf16.");
    constexpr int LANE_PER_WAVE = 64;

    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BK      = WK * WAVES_K;

    constexpr int BM = BLOCK_M * TW_M;   // rows per block
    constexpr int BN = BLOCK_N * TW_N;   // cols per block

    // ----- which expert / tile -----
    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];

    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left   = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0       = blockIdx.x * BN;
    const int lane     = threadIdx.x;     // 0..63
    const int wave     = threadIdx.y;     // 0..(WAVES_M*WAVES_N-1)
    const int wave_m   = wave / WAVES_N;
    const int wave_n   = wave % WAVES_N;

    // ---- LDS (single buffer) ----
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;   // bf16 pitch
    constexpr int ldB = BK + PAD_K;

    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BM * ldA);
    const size_t sB_off   = (sA_bytes + 15) & ~size_t(15);  // 16B align B tile

    uint16_t* sA_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sB_u16 = reinterpret_cast<uint16_t*>(smemRaw + sB_off);

    uint32_t* sA_u32 = reinterpret_cast<uint32_t*>(sA_u16);
    uint32_t* sB_u32 = reinterpret_cast<uint32_t*>(sB_u16);

    // accumulators
    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn)
        acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16* __restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16* __restrict__ b_e = b2 + (size_t)e * (size_t)N;

    auto aRowBase_tm = [&](int tm){ return (wave_m + tm * WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn){ return (wave_n + tn * WAVES_N) * WN; };

    // ---- K loop (single buffer): copy -> sync -> compute -> sync ----
    for (int k0 = 0; k0 < K; k0 += BK) {
        copy_A_tile_vec128_fp32_tiled_2<ldA, BM, BK>(
            sA_u32, A, m_start, M_bound, K, k0, linearT, threadsPerBlock);
        copy_B_tile_vec128_bf16_tiled_2<ldB, BN, BK>(
            sB_u32, W_e, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // ---- stores (+bias) ----
    const bool interior = (m_left >= BLOCK_M * TW_M) && ((n0 + BN) <= N);

    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn) {
        const int col  = n0 + (wave_n + tn * WAVES_N) * WN + lane_row(lane);
        const float bs = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;

        #pragma unroll
        for (int tm = 0; tm < TW_M; ++tm) {
            const int rowBase = m_start + (wave_m + tm * WAVES_M) * WM + lane_group(lane) * 4;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = rowBase + i;
                const float v = acc[tm][tn][i] + bs;
                if (interior) {
                    C[(size_t)row * N + col] = v;
                } else {
                    if (row < M_bound && col < N) C[(size_t)row * N + col] = v;
                }
            }
        }
    }
}


template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
inline void mlp2_optimized(
    float *C, const float *A, const __hip_bfloat16 *W2, const __hip_bfloat16 *b2,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");

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

    const size_t sA_bytes  = sizeof(uint16_t) * (size_t)(BM * ldA);
    const size_t sB_bytes  = sizeof(uint16_t) * (size_t)(BN * ldB);
    const size_t sB_off    = (sA_bytes + 15) & ~size_t(15); // 16B align
    const size_t shmem_bytes = sB_off + sB_bytes;

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp2_bf16_bias_kernel_tiled(vec128,singlebuf)");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp2_bf16_bias_kernel_tiled_optimized<
            WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W2, b2,
        expert_offsets, expert_counts,
        tile2expert, tile2local,
        E, K, N);
    HIP_CHECK(hipGetLastError());
}



__device__ __forceinline__ void deq32_stream_store_from_u128(
    uint4 v128, float X, uint32_t* __restrict__ dst_u32)
{
    const float mtab[8] = {0.f, 0.5f*X, 1.f*X, 1.5f*X, 2.f*X, 3.f*X, 4.f*X, 6.f*X};
    auto emit4 = [&](uint32_t w){
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint8_t byte = (uint8_t)((w >> (k * 8)) & 0xFF);
            const uint8_t n0 =  byte       & 0x0F;
            const uint8_t n1 = (byte >> 4) & 0x0F;
            const float v0 = (n0 & 8) ? -mtab[n0 & 7] : mtab[n0 & 7];
            const float v1 = (n1 & 8) ? -mtab[n1 & 7] : mtab[n1 & 7];
            *dst_u32++ = pack2_bf16_bits_f32(v0, v1);
        }
    };
    emit4(v128.x); emit4(v128.y); emit4(v128.z); emit4(v128.w);
}


// ---- A (global bf16) -> LDS (bf16; write u32 pairs) ----
// One 16B unit = 8 bf16 => 4 u32 pairs.
template<int BLOCK_M, int BLOCK_K, int LD_A>
__device__ __forceinline__ void copy_A_tile_vec128_bf16(
    uint32_t* __restrict__ dst_u32,
    const __hip_bfloat16* __restrict__ A, // [M,K] row-major
    int m_start, int M, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be multiple of 8 bf16 (16B).");
    constexpr int quadsPerRow = BLOCK_K / 8;   // 8 bf16 per 16B
    const int totalQuads      = BLOCK_M * quadsPerRow;

    // Entire tile's K-range is fully in-bounds?
    const bool fullK = (kBase + BLOCK_K) <= K;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int r  = t / quadsPerRow;       // 0..BLOCK_M-1
        const int q  = t % quadsPerRow;       // 16B unit along K
        const int gm = m_start + r;           // global row

        // LDS row base (u32-pair addressing: LD_A is in bf16 elems)
        uint32_t* __restrict__ row_out = dst_u32 + (((size_t)r * LD_A) >> 1);
        uint32_t* __restrict__ out     = row_out + (q << 2); // 4 u32 per 16B

        uint4 v = {0,0,0,0};

        if (gm < M) {
            const size_t rowBase     = (size_t)gm * K + kBase;
            const __hip_bfloat16* gp = A + rowBase;

            // Fast path: entire K-slab fits and the row start is 16B-aligned.
            if (fullK && is_aligned_16B(gp)) {
                const uint4* p128 = reinterpret_cast<const uint4*>(gp);
                v = p128[q];
            } else {
                // Per-chunk path (handles partial K and/or misalignment gracefully).
                const int gk8 = kBase + (q << 3); // bf16 index (×8)
                if ((gk8 + 7) < K) {
                    const __hip_bfloat16* chunk = gp + (q << 3);
                    if (is_aligned_16B(chunk)) {
                        v = *reinterpret_cast<const uint4*>(chunk);
                    } else {
                        __hip_bfloat16 tmp[8] = {};
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int kk = gk8 + i;
                            if (kk < K) tmp[i] = A[(size_t)gm * K + kk];
                        }
                        const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                        v = make_uint4(p[0], p[1], p[2], p[3]);
                    }
                }
                // else: keep zeros
            }
        }
        // Always write (zero-fill for out-of-bounds gm/q).
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    }
}


// ---- B: global bf16 -> LDS bf16 (vectorized) ----
// Loads uint4 (16B = 8 bf16) and stores as 4×u32 pairs.
template <int BLOCK_N, int BLOCK_K, int LD_B>
__device__ __forceinline__ void copy_B_tile_vec128_bf16(
    uint32_t *__restrict__ dst_u32,
    const __hip_bfloat16 *__restrict__ W,   // [N,K] row-major
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BLOCK_K % 8) == 0, "copy_B_tile_vec128_bf16: BLOCK_K must be multiple of 8 (bf16 per 16B).");

    constexpr int quadsPerCol = BLOCK_K / 8; // 8 bf16 per 16B
    const int totalQuads      = BLOCK_N * quadsPerCol;

    const bool fullK = (kBase + BLOCK_K) <= K;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int c  = t / quadsPerCol;      // col within BLOCK_N
        const int q  = t % quadsPerCol;      // 16B unit along K
        const int gn = n0 + c;

        // LDS col base (u32-pair addressing: LD_B is in bf16 elems)
        uint32_t* __restrict__ col_out = dst_u32 + (((size_t)c * LD_B) >> 1);
        uint32_t* __restrict__ out     = col_out + (q << 2); // 4 u32 per 16B

        uint4 v = {0,0,0,0};

        if (gn < N) {
            const size_t rowBase     = (size_t)gn * K + kBase;
            const __hip_bfloat16* gp = W + rowBase;

            if (fullK && is_aligned_16B(gp)) {
                const uint4* p128 = reinterpret_cast<const uint4*>(gp);
                v = p128[q];
            } else {
                const int gk8 = kBase + (q << 3);
                if ((gk8 + 7) < K) {
                    const __hip_bfloat16* chunk = gp + (q << 3);
                    if (is_aligned_16B(chunk)) {
                        v = *reinterpret_cast<const uint4*>(chunk);
                    } else {
                        __hip_bfloat16 tmp[8] = {};
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int kk = gk8 + i;
                            if (kk < K) tmp[i] = W[(size_t)gn * K + kk];
                        }
                        const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                        v = make_uint4(p[0], p[1], p[2], p[3]);
                    }
                }
            }
        }
        // Always write (zero-fill for out-of-bounds gn/q).
        out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
    }
}

// ---- A (global bf16) -> LDS (bf16; write u32 pairs), tiled form used by MLP2 ----
template<int LD_A, int BM, int BK>
__device__ inline void copy_A_tile_vec128_bf16_tiled_2(
    uint32_t* __restrict__ dst_u32,
    const __hip_bfloat16* __restrict__ A, // [M,K] row-major
    int m_start, int M_bound, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    static_assert((BK % 8) == 0, "BK must be multiple of 8 bf16 (16B).");
    constexpr int quadsPerRow = BK / 8;       // 8 bf16 per 16B
    const int totalQuads      = BM * quadsPerRow;

    for (int t = linearT; t < totalQuads; t += threadsPerBlock) {
        const int r   = t / quadsPerRow;      // 0..BM-1
        const int q   = t % quadsPerRow;      // 16B unit along K
        const int gm  = m_start + r;          // global row
        const int gk8 = kBase + (q << 3);     // bf16 index (×8)

        uint4 v = {0,0,0,0};
        if (gm < M_bound) {
            const size_t base = (size_t)gm * K + gk8;
            if ((gk8 + 7) < K && is_aligned_16B(&A[base])) {
                v = *reinterpret_cast<const uint4*>(&A[base]);  // coalesced 16B
            } else {
                __hip_bfloat16 tmp[8] = {};
                #pragma unroll
                for (int i=0;i<8 && (gk8+i)<K;++i) tmp[i] = A[base + i];
                const uint32_t* p = reinterpret_cast<const uint32_t*>(tmp);
                v = make_uint4(p[0], p[1], p[2], p[3]);
            }
        }
        uint32_t* row = dst_u32 + ((size_t)r * LD_A >> 1);
        const int off = (q << 2);
        row[off + 0] = v.x;
        row[off + 1] = v.y;
        row[off + 2] = v.z;
        row[off + 3] = v.w;
    }
}


template <
    int WM, int WN, int WK,
    int WAVES_M, int WAVES_N, int WAVES_K,
    int TW_M, int TW_N,
    int PAD_K_MC>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2)
void grouped_mlp1_bf16_kernel_unified(
    float *__restrict__ C,                        // [sum_tokens, N] (fp32)
    const __hip_bfloat16 *__restrict__ A,         // [sum_tokens, K] (bf16)  <-- changed
    const __hip_bfloat16 *__restrict__ Wbf16,     // [E, N, K] (bf16)
    const int *__restrict__ expert_offsets,       // [E]
    const int *__restrict__ expert_counts,        // [E]
    const int *__restrict__ tile2expert,          // [num_mtiles]
    const int *__restrict__ tile2local,           // [num_mtiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "This kernel assumes 16x16x16 bf16 MFMA.");
    static_assert((WAVES_M % TW_M) == 0 && (WAVES_N % TW_N) == 0, "TW_* must divide WAVES_*");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    constexpr int WAVES_PER_BLOCK_E = WAVES_M_E * WAVES_N_E;
    static_assert((BLOCK_K % 2) == 0, "BLOCK_K must be even");
    // static_assert((PAD_K_MC % 2) == 0, "PAD_K_MC must be even");

    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];

    const int m_start = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left  = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0  = blockIdx.x * BLOCK_N;
    const int m0  = m_start;

    const int lane = threadIdx.x; // 0..63
    const int wave = threadIdx.y; // 0..WAVES_PER_BLOCK_E-1
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    const int aBase0 = tile_m0 * WM;
    const int bBase0 = tile_n0 * WN;

    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t *sA_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sB_u16 = sA_u16 + (BLOCK_M * ldA);

    uint32_t *sA_u32 = reinterpret_cast<uint32_t *>(sA_u16);
    uint32_t *sB_u32 = reinterpret_cast<uint32_t *>(sB_u16);

    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn)
        acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const __hip_bfloat16 *__restrict__ W_e = Wbf16 + (size_t)e * (size_t)N * (size_t)K;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // A: bf16→LDS (128b)
        copy_A_tile_vec128_bf16<BLOCK_M, BLOCK_K, ldA>(
            sA_u32, A, m0, m_start + m_left, K, k0, linearT, threadsPerBlock);
        // B: bf16→LDS (128b)
        copy_B_tile_vec128_bf16<BLOCK_N, BLOCK_K, ldB>(
            sB_u32, W_e, n0, N, K, k0, linearT, threadsPerBlock);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase, kk, lane);
            }
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase, kk, lane);
            }
            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    const bool interior =
        ((m0 + BLOCK_M) <= (m_start + m_left)) && ((n0 + BLOCK_N) <= N);

    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TW_N; ++tn) {
            const int wmt = tile_m0 + tm;
            const int wnt = tile_n0 + tn;
            if (interior)
                store_c_tile<WM, WN, true >(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
            else
                store_c_tile<WM, WN, false>(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
        }
    }
}


template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K_MC = 16>
inline void mlp1_optimized(
    float *C,
    const __hip_bfloat16 *A,                  // <-- bf16 input
    const __hip_bfloat16 *W1,                 // [E,N,K] bf16
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, cur_tiles);
    dim3 block(64, (WAVES_M / TW_M) * (WAVES_N / TW_N));

    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(BLOCK_M * ldA + ldB * BLOCK_N);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp1_bf16_kernel_unified");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp1_bf16_kernel_unified<
            WM, WN, WK,
            WAVES_M, WAVES_N, WAVES_K,
            TW_M, TW_N,
            PAD_K_MC>),
        grid, block, shmem_bytes, s,
        C, A, W1,
        expert_offsets, expert_counts,
        tile2expert, tile2local,
        E, K, N);
    HIP_CHECK(hipGetLastError());
}


template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
__global__ void grouped_mlp2_bf16_bias_kernel_tiled_optimized(
    float *__restrict__ C,                       // [sum_tokens, H]
    const __hip_bfloat16 *__restrict__ A,        // [sum_tokens, D] (bf16)  <-- changed
    const __hip_bfloat16 *__restrict__ W2,       // [N=H, K=D] bf16
    const __hip_bfloat16 *__restrict__ b2,       // [N=H] bf16
    const int *__restrict__ expert_offsets,
    const int *__restrict__ expert_counts,
    const int *__restrict__ tile2expert,
    const int *__restrict__ tile2local,
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA 16x16x16 bf16.");
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
    const int m_left   = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0       = blockIdx.x * BN;
    const int lane     = threadIdx.x;
    const int wave     = threadIdx.y;
    const int wave_m   = wave / WAVES_N;
    const int wave_n   = wave % WAVES_N;

    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BK + PAD_K;
    constexpr int ldB = BK + PAD_K;

    const size_t sA_bytes = sizeof(uint16_t) * (size_t)(BM * ldA);
    const size_t sB_off   = (sA_bytes + 15) & ~size_t(15);

    uint16_t* sA_u16 = reinterpret_cast<uint16_t*>(smemRaw);
    uint16_t* sB_u16 = reinterpret_cast<uint16_t*>(smemRaw + sB_off);

    uint32_t* sA_u32 = reinterpret_cast<uint32_t*>(sA_u16);
    uint32_t* sB_u32 = reinterpret_cast<uint32_t*>(sB_u16);

    f32x4 acc[TW_M][TW_N];
    #pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn)
        acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const int M_bound = m_start + m_left;
    const __hip_bfloat16* __restrict__ W_e = W2 + (size_t)e * (size_t)N * (size_t)K;
    const __hip_bfloat16* __restrict__ b_e = b2 + (size_t)e * (size_t)N;

    auto aRowBase_tm = [&](int tm){ return (wave_m + tm * WAVES_M) * WM; };
    auto bColBase_tn = [&](int tn){ return (wave_n + tn * WAVES_N) * WN; };

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A: bf16→LDS (128b)
        copy_A_tile_vec128_bf16_tiled_2<ldA, BM, BK>(
            sA_u32, A, m_start, M_bound, K, k0, linearT, threadsPerBlock);
        // B: bf16→LDS (128b)
        copy_B_tile_vec128_bf16_tiled_2<ldB, BN, BK>(
            sB_u32, W_e, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            bf16x4 avec[TW_M];
            bf16x4 bvec[TW_N];

            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
                avec[tm] = make_a_vec_k<WM>(sA_u16, ldA, aRowBase_tm(tm), kk, lane);

            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                bvec[tn] = make_b_vec_k<WN>(sB_u16, ldB, bColBase_tn(tn), kk, lane);

            #pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            #pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
                acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }
        __syncthreads();
    }

    const bool interior = (m_left >= BLOCK_M * TW_M) && ((n0 + BN) <= N);

    #pragma unroll
    for (int tn = 0; tn < TW_N; ++tn) {
        const int col  = n0 + (wave_n + tn * WAVES_N) * WN + lane_row(lane);
        const float bs = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;

        #pragma unroll
        for (int tm = 0; tm < TW_M; ++tm) {
            const int rowBase = m_start + (wave_m + tm * WAVES_M) * WM + lane_group(lane) * 4;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = rowBase + i;
                const float v = acc[tm][tn][i] + bs;
                if (interior) {
                    C[(size_t)row * N + col] = v;
                } else {
                    if (row < M_bound && col < N) C[(size_t)row * N + col] = v;
                }
            }
        }
    }
}

template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K = 0>
inline void mlp2_optimized(
    float *C,
    const __hip_bfloat16 *A,                  // <-- bf16 input
    const __hip_bfloat16 *W2,
    const __hip_bfloat16 *b2,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "bf16 MFMA uses 16x16x16 tiles");

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

    const size_t sA_bytes  = sizeof(uint16_t) * (size_t)(BM * ldA);
    const size_t sB_bytes  = sizeof(uint16_t) * (size_t)(BN * ldB);
    const size_t sB_off    = (sA_bytes + 15) & ~size_t(15);
    const size_t shmem_bytes = sB_off + sB_bytes;

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp2_bf16_bias_kernel_tiled(vec128,singlebuf)");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp2_bf16_bias_kernel_tiled_optimized<
            WM, WN, WK, WAVES_M, WAVES_N, WAVES_K, TW_M, TW_N, PAD_K>),
        grid, block, shmem_bytes, s,
        C, A, W2, b2,
        expert_offsets, expert_counts,
        tile2expert, tile2local,
        E, K, N);
    HIP_CHECK(hipGetLastError());
}



// ==== REPLACEMENT: fused routing/packing that WRITES BF16 expert input ====
__global__ void route_and_pack_fused_kernel_bf16(
    const float *__restrict__ x, // [B, H] (fp32 source)
    int B, int H,
    const int *__restrict__ topk_i,   // [B, K]
    const float *__restrict__ topk_v, // [B, K]
    int K,
    const int *__restrict__ expert_offsets, // [E] (exclusive prefix)
    int E,
    int *__restrict__ local_ids,       // [B, K] (out)
    float *__restrict__ local_wts,     // [B, K] (out)
    __hip_bfloat16 *__restrict__ expert_in_bf16) // [sum_tokens, H] (out, bf16)
{
    const int e = blockIdx.x;
    if (e >= E) return;

    const int T  = blockDim.x;
    const int tid = threadIdx.x;

    extern __shared__ int _smem[];
    int *flags = _smem;              // [T]
    int *excl = flags + T;           // [T]
    int *kidx = excl + T;            // [T]
    int *toklist = kidx + T;         // [T]
    int *cmplist = toklist + T;      // [T]
    int *carry_shared = cmplist + T; // [1]

    if (tid == 0) carry_shared[0] = 0;
    __syncthreads();

    for (int base = 0; base < B; base += T)
    {
        // 1) flag tokens for this expert + remember which k matched
        const int t_global = base + tid;
        int f = 0, kk = -1;
        if (t_global < B)
        {
            const int off = t_global * K;
#pragma unroll
            for (int i = 0; i < K; ++i)
            {
                if (topk_i[off + i] == e) { f = 1; kk = i; break; }
            }
        }
        flags[tid] = f;
        kidx[tid]  = kk;
        __syncthreads();

        // 2) inclusive scan (Hillis–Steele)
        excl[tid] = flags[tid];
        __syncthreads();
        for (int ofs = 1; ofs < T; ofs <<= 1)
        {
            int v = (tid >= ofs) ? excl[tid - ofs] : 0;
            __syncthreads();
            excl[tid] += v;
            __syncthreads();
        }
        const int rank_local = (tid == 0) ? 0 : excl[tid - 1];
        const int chunk_total = excl[T - 1];

        // 3) for selected tokens: compute compact_idx, write ids/wts and lists
        if (flags[tid])
        {
            const int compact_idx = expert_offsets[e] + carry_shared[0] + rank_local;
            const int lid = t_global * K + kidx[tid];

            local_ids[lid] = compact_idx;
            local_wts[lid] = topk_v[lid];

            toklist[rank_local] = t_global;
            cmplist[rank_local] = compact_idx;
        }
        __syncthreads();

        // 4) cooperative copy rows src(fp32)->dst(bf16) with per-thread striding
        for (int j = 0; j < chunk_total; ++j)
        {
            const int tkn = toklist[j];
            const int cmp = cmplist[j];
            const float *__restrict__ src = x + (size_t)tkn * H;
            __hip_bfloat16 *__restrict__ dst = expert_in_bf16 + (size_t)cmp * H;

            // vector-friendly, but scalar convert is fine; compiler will unroll
            for (int i = tid; i < H; i += T)
            {
                dst[i] = __float2bfloat16(src[i]);
            }
        }
        __syncthreads();

        // 5) advance expert-local carry
        if (tid == 0) carry_shared[0] += chunk_total;
        __syncthreads();
    }
}

// ========== templated HYBRID MXFP4 B tile copy (row-major N×K -> LDS col-major) ==========
template <int BLOCK_N, int BLOCK_K, int LD_B>
__device__ inline void copy_B_tile_vec_MXFP4_hybrid(
    uint32_t *__restrict__ dst_u32,         // writes u32 (2×bf16) into LDS
    const uint8_t *__restrict__ W_packed_e, // [N,K] row-major (2 nibbles/byte)
    const float   *__restrict__ S_f32_e,    // [ceil(N*K/32)] one f32 per 32 elems
    int n0, int N, int K, int kBase,
    int linearT, int threadsPerBlock)
{
    if constexpr (BLOCK_K >= 64) {
        constexpr int vecsPerCol = BLOCK_K / 32;
        constexpr int totalVecs  = BLOCK_N * vecsPerCol;

        for (int t = linearT; t < totalVecs; t += threadsPerBlock)
        {
            const int c   = t / vecsPerCol;          // 0..BLOCK_N-1
            const int v32 = t % vecsPerCol;          // which 32-chunk along K
            const int gn  = n0 + c;
            const int gk  = kBase + (v32 << 5);      // multiple of 32

            uint32_t *col = reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1));
            const int offPairs = ((gk - kBase) >> 1);

            if (gn < N && gk < K) {
                const size_t idx0   = (size_t)gn * (size_t)K + (size_t)gk;
                const size_t byteOf = idx0 >> 1;     // 2 nibbles/byte
                const size_t blk32  = idx0 >> 5;     // /32 for scale
                const float  X      = S_f32_e[blk32];

                if ((gk + 31) < K && is_aligned_16B(&W_packed_e[byteOf])) {
                    const uint4 v = *reinterpret_cast<const uint4 *>(&W_packed_e[byteOf]);
                    deq32_stream_store_from_u128(v, X, col + offPairs);
                } else {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const int kEven = gk + (j << 1);
                        col[offPairs + j] = (kEven < K)
                          ? deq2_pack_bf16_u32_fscale(W_packed_e, S_f32_e, idx0 + (j << 1), K, N)
                          : 0u;
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) col[offPairs + j] = 0u;
            }
        }
    } else {
        // BK==32 path: coalesced p-major dequant
        constexpr int pairsPerCol = BLOCK_K >> 1;
        constexpr int totalPairs  = BLOCK_N * pairsPerCol;

        for (int t = linearT; t < totalPairs; t += threadsPerBlock)
        {
            const int c  = t / pairsPerCol;             // 0..BLOCK_N-1
            const int p  = t % pairsPerCol;             // 0..pairsPerCol-1
            const int gn = n0 + c;
            const int gk = kBase + (p << 1);

            uint32_t out = 0u;
            if (gn < N && gk < K) {
                const size_t idx0 = (size_t)gn * (size_t)K + (size_t)gk;
                out = deq2_pack_bf16_u32_fscale(W_packed_e, S_f32_e, idx0, K, N);
            }
            reinterpret_cast<uint32_t *>(dst_u32 + ((size_t)c * LD_B >> 1))[p] = out;
        }
    }
}


// ========== NEW: unified MXFP4 kernel that LOADS A IN BF16 ==========
template <
    // MFMA micro-tile
    int WM, int WN, int WK,
    // Block waves (logical)
    int WAVES_M, int WAVES_N, int WAVES_K,
    // Tiles per wave (multi-tile per wave)
    int TW_M, int TW_N,
    // Padding along K in LDS
    int PAD_K_MC>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2)
void grouped_mlp1_mxfp4_kernel_unified_A_bf16(
    float *__restrict__ C,                        // [sum_tokens, N]
    const __hip_bfloat16 *__restrict__ A_bf16,    // [sum_tokens, K]  <<< bf16 inputs
    const uint8_t *__restrict__ W_packed_layer,   // [E, ceil(N*K/2)]
    const float *__restrict__ S_scales_f32_layer, // [E, ceil(N*K/32)]
    const int *__restrict__ expert_offsets,       // [E]
    const int *__restrict__ expert_counts,        // [E]
    const int *__restrict__ tile2expert,          // [num_mtiles]
    const int *__restrict__ tile2local,           // [num_mtiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA 16x16x16");
    static_assert((WAVES_M % TW_M) == 0 && (WAVES_N % TW_N) == 0, "TW must divide WAVES");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    static_assert((BLOCK_K % 2) == 0, "BLOCK_K even");
    static_assert((PAD_K_MC % 2) == 0, "PAD_K_MC even");

    // which expert / M tile
    const int mtile_id = blockIdx.y;
    const int e = tile2expert[mtile_id];
    const int tile_m = tile2local[mtile_id];
    const int m_start = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N;
    const int m0 = m_start;

    // unified indexing
    const int lane = threadIdx.x; // 0..63
    const int wave = threadIdx.y; // 0..WAVES_M_E*WAVES_N_E-1
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    const int aBase0 = tile_m0 * WM;
    const int bBase0 = tile_n0 * WN;

    // single-buffer LDS
    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t *sA_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sB_u16 = sA_u16 + (BLOCK_M * ldA);

    uint32_t *sA_u32 = reinterpret_cast<uint32_t *>(sA_u16);
    uint32_t *sB_u32 = reinterpret_cast<uint32_t *>(sB_u16);

    // accumulators
    f32x4 acc[TW_M][TW_N];
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
            acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT = wave * blockDim.x + lane;

    // per-expert segment bases
    const size_t seg_elems = (size_t)N * (size_t)K;
    const size_t seg_bytes = (seg_elems + 1) / 2;
    const size_t seg_blocks = (seg_elems + 31) / 32;
    const uint8_t *__restrict__ W_e_packed = W_packed_layer + (size_t)e * seg_bytes;
    const float *__restrict__ S_f32_e = S_scales_f32_layer + (size_t)e * seg_blocks;

    const int M_bound = m_start + m_left;

    // K loop
    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        // A: bf16 -> LDS bf16 (128b vectorized)
        copy_A_tile_vec128_bf16<BLOCK_M, BLOCK_K, ldA>(
            sA_u32, A_bf16, m0, M_bound, K, k0, linearT, threadsPerBlock);

        // B: MXFP4 -> bf16 (hybrid)
        copy_B_tile_vec_MXFP4_hybrid<BLOCK_N, BLOCK_K, ldB>(
            sB_u32, W_e_packed, S_f32_e, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK)
        {
            bf16x4 avec[TW_M];
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
            {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k(sA_u16, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn)
            {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k(sB_u16, ldB, bColBase, kk, lane);
            }
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // stores (identical to your previous path)
    const bool interior =
        ((m0 + BLOCK_M) <= (m_start + m_left)) &&
        ((n0 + BLOCK_N) <= N);
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    {
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
        {
            const int wmt = tile_m0 + tm;
            const int wnt = tile_n0 + tn;
            if (interior)
                store_c_tile<WM, WN, true>(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
            else
                store_c_tile<WM, WN, false>(acc[tm][tn], C, m_start + m_left, N, m0, n0, wmt, wnt, lane);
        }
    }
}

// ========== NEW: launcher for the bf16-A unified kernel ==========
template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K_MC = 16>
inline void mlp1_mxfp4_optimized_unified_A_bf16(
    float *C, const __hip_bfloat16 *A_bf16,
    const uint8_t *W1_packed, const float *S1_scales_f32,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;

    // grid over N and the logical M tiles
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, cur_tiles);
    dim3 block(64, (WAVES_M / TW_M) * (WAVES_N / TW_N)); // effective waves

    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(BLOCK_M * ldA + ldB * BLOCK_N);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp1_mxfp4_kernel_unified_A_bf16");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp1_mxfp4_kernel_unified_A_bf16<
            WM, WN, WK,
            WAVES_M, WAVES_N, WAVES_K,
            TW_M, TW_N,
            PAD_K_MC>),
        grid, block, shmem_bytes, s,
        C, A_bf16, W1_packed, S1_scales_f32,
        expert_offsets, expert_counts,
        tile2expert, tile2local,
        E, K, N);
    HIP_CHECK(hipGetLastError());
}


// ==== NEW: SwiGLU epilogue that writes BF16 ====
__global__ void bias_swiglu_epilogue_kernel_bf16(
    const float *__restrict__ mlp1_out,     // [sum_tokens, 2*D]
    const __hip_bfloat16 *__restrict__ b1,  // [n_experts, 2*D]
    const int *__restrict__ expert_offsets, // [n_experts]
    const int *__restrict__ expert_counts,  // [n_experts]
    int n_experts,
    __hip_bfloat16 *__restrict__ gate_up_bf16, // [sum_tokens, D]  <<< bf16 out
    int D,
    int sum_tokens,
    float clamp_limit,
    float alpha_silu /* = 1.702f */)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)sum_tokens * D;
    if (idx >= total) return;

    const int d = (int)(idx % D);

    // ---- warp-cooperative token/expert resolution (same as fp32 path) ----
    const int lane = threadIdx.x & (warpSize - 1);
    const size_t warp_first = idx - lane;
    const int t0 = (int)(warp_first / D);
    const int t1 = (int)((warp_first + (warpSize - 1)) / D);

    int t, e;
    if (t0 == t1) {
        t = t0;
        int lo = 0, hi = n_experts;
        if (lane == 0) {
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                const int start = expert_offsets[mid];
                const int end = start + expert_counts[mid];
                if (t >= end) lo = mid + 1;
                else if (t < start) hi = mid;
                else { lo = mid; break; }
            }
        }
    #if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        t = __shfl(t, 0);
        e = __shfl(lo, 0);
    #else
        t = __shfl_sync(0xFFFFFFFF, t, 0);
        e = __shfl_sync(0xFFFFFFFF, lo, 0);
    #endif
    } else {
        t = (int)(idx / D);
        int lo = 0, hi = n_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            const int start = expert_offsets[mid];
            const int end = start + expert_counts[mid];
            if (t >= end) lo = mid + 1;
            else if (t < start) hi = mid;
            else { lo = mid; break; }
        }
        e = lo;
    }

    const size_t t2D = (size_t)t * (2 * (size_t)D);
    const size_t e2D = (size_t)e * (2 * (size_t)D);

    const float bg = __bfloat162float(b1[e2D + (size_t)(2 * d + 0)]);
    const float bu = __bfloat162float(b1[e2D + (size_t)(2 * d + 1)]);

    float gx = mlp1_out[t2D + (size_t)(2 * d + 0)] + bg;
    float uy = mlp1_out[t2D + (size_t)(2 * d + 1)] + bu;

    gx = fminf(fmaxf(gx, -clamp_limit), clamp_limit);
    uy = fminf(fmaxf(uy, -clamp_limit), clamp_limit);

    const float sig = 1.0f / (1.0f + fast_expf(-alpha_silu * gx));
    gx *= sig;

    const float out = fmaf(gx, uy, gx);
    gate_up_bf16[(size_t)t * (size_t)D + d] = __float2bfloat16(out);
}

// ========== NEW: unified MXFP4 MLP2 kernel that LOADS A IN BF16 ==========
template <
    int WM, int WN, int WK,
    int WAVES_M, int WAVES_N, int WAVES_K,
    int TW_M, int TW_N,
    int PAD_K_MC>
__global__ __launch_bounds__(64 * ((WAVES_M / TW_M) * (WAVES_N / TW_N)), 2)
void grouped_mlp2_mxfp4_bias_kernel_unified_A_bf16(
    float *__restrict__ C,                         // [sum_tokens, N]
    const __hip_bfloat16 *__restrict__ A_bf16,     // [sum_tokens, K]  <<< bf16 inputs
    const uint8_t *__restrict__ W_packed_layer,    // [E, ceil(N*K/2)]
    const float   *__restrict__ S_scales_f32_layer,// [E, ceil(N*K/32)]
    const __hip_bfloat16 *__restrict__ b2,         // [E, N] bf16
    const int *__restrict__ expert_offsets,        // [E]
    const int *__restrict__ expert_counts,         // [E]
    const int *__restrict__ tile2expert,           // [num_mtiles]
    const int *__restrict__ tile2local,            // [num_mtiles]
    int E, int K, int N)
{
    static_assert(WM == 16 && WN == 16 && WK == 16, "MFMA 16x16x16");
    static_assert((WAVES_M % TW_M) == 0 && (WAVES_N % TW_N) == 0, "TW must divide WAVES");
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;
    constexpr int WAVES_M_E = WAVES_M / TW_M;
    constexpr int WAVES_N_E = WAVES_N / TW_N;
    static_assert((BLOCK_K % 2) == 0, "BLOCK_K even");
    static_assert((PAD_K_MC % 2) == 0, "PAD_K_MC even");

    const int mtile_id = blockIdx.y;
    const int e        = tile2expert[mtile_id];
    const int tile_m   = tile2local[mtile_id];
    const int m_start  = expert_offsets[e] + tile_m * BLOCK_M;
    const int m_left   = expert_counts[e] - tile_m * BLOCK_M;
    if (m_left <= 0) return;

    const int n0 = blockIdx.x * BLOCK_N;
    const int m0 = m_start;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int wave_m_eff = wave / WAVES_N_E;
    const int wave_n_eff = wave % WAVES_N_E;

    const int tile_m0 = wave_m_eff * TW_M;
    const int tile_n0 = wave_n_eff * TW_N;

    const int aBase0 = tile_m0 * WM;
    const int bBase0 = tile_n0 * WN;

    extern __shared__ uint8_t smemRaw[];
    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;

    uint16_t *sA_u16 = reinterpret_cast<uint16_t *>(smemRaw);
    uint16_t *sB_u16 = sA_u16 + (BLOCK_M * ldA);

    uint32_t *sA_u32 = reinterpret_cast<uint32_t *>(sA_u16);
    uint32_t *sB_u32 = reinterpret_cast<uint32_t *>(sB_u16);

    f32x4 acc[TW_M][TW_N];
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
            acc[tm][tn] = {0.f, 0.f, 0.f, 0.f};

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearT         = wave * blockDim.x + lane;

    const size_t seg_elems  = (size_t)N * (size_t)K;
    const size_t seg_bytes  = (seg_elems + 1) / 2;
    const size_t seg_blocks = (seg_elems + 31) / 32;
    const uint8_t *__restrict__ W_e_packed = W_packed_layer    + (size_t)e * seg_bytes;
    const float   *__restrict__ S_f32_e    = S_scales_f32_layer + (size_t)e * seg_blocks;
    const __hip_bfloat16 *__restrict__ b_e = b2 + (size_t)e * (size_t)N;

    const int M_bound = m_start + m_left;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K)
    {
        // A: bf16 -> LDS bf16 (128b vectorized)
        copy_A_tile_vec128_bf16<BLOCK_M, BLOCK_K, ldA>(
            sA_u32, A_bf16, m_start, M_bound, K, k0, linearT, threadsPerBlock);

        // B: MXFP4 -> bf16
        copy_B_tile_vec_MXFP4_hybrid<BLOCK_N, BLOCK_K, ldB>(
            sB_u32, W_e_packed, S_f32_e, n0, N, K, k0, linearT, threadsPerBlock);

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WK)
        {
            bf16x4 avec[TW_M];
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm) {
                const int aRowBase = aBase0 + tm * WM;
                avec[tm] = make_a_vec_k(sA_u16, ldA, aRowBase, kk, lane);
            }
            bf16x4 bvec[TW_N];
#pragma unroll
            for (int tn = 0; tn < TW_N; ++tn) {
                const int bColBase = bBase0 + tn * WN;
                bvec[tn] = make_b_vec_k(sB_u16, ldB, bColBase, kk, lane);
            }
#pragma unroll
            for (int tm = 0; tm < TW_M; ++tm)
#pragma unroll
                for (int tn = 0; tn < TW_N; ++tn)
                    acc[tm][tn] = mfma_16x16x16_bf16(avec[tm], bvec[tn], acc[tm][tn]);
        }

        __syncthreads();
    }

    // Stores with bias (same as your fp32-A version)
#pragma unroll
    for (int tm = 0; tm < TW_M; ++tm)
    {
        const int rowBase = m_start + (aBase0 + tm * WM) + lane_group(lane) * 4;
#pragma unroll
        for (int tn = 0; tn < TW_N; ++tn)
        {
            const int col = n0 + (bBase0 + tn * WN) + lane_row(lane);
            const float bias = (col < N) ? __bfloat162float(b_e[col]) : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int row = rowBase + i;
                if (row < M_bound && col < N)
                    C[(size_t)row * N + col] = acc[tm][tn][i] + bias;
            }
        }
    }
}

// ========== NEW: launcher for the bf16-A unified MLP2 ==========
template <
    int WM = 16, int WN = 16, int WK = 16,
    int WAVES_M = 1, int WAVES_N = 8, int WAVES_K = 2,
    int TW_M = 1, int TW_N = 2,
    int PAD_K_MC = 16>
inline void mlp2_mxfp4_bias_optimized_unified_A_bf16(
    float *C, const __hip_bfloat16 *A_bf16,     // <<< bf16 A
    const uint8_t *W_packed, const float *S_scales_f32,
    const __hip_bfloat16 *bias_bf16,
    const int *expert_offsets, const int *expert_counts,
    const int *tile2expert, const int *tile2local,
    int E, int K, int N, int cur_tiles, hipStream_t s = nullptr)
{
    constexpr int BLOCK_M = WM * WAVES_M;
    constexpr int BLOCK_N = WN * WAVES_N;
    constexpr int BLOCK_K = WK * WAVES_K;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, cur_tiles);
    dim3 block(64, (WAVES_M / TW_M) * (WAVES_N / TW_N));

    constexpr int ldA = BLOCK_K + PAD_K_MC;
    constexpr int ldB = BLOCK_K + PAD_K_MC;
    const size_t shmem_bytes =
        sizeof(uint16_t) * (size_t)(BLOCK_M * ldA + ldB * BLOCK_N);

#ifdef assert_smem_or_die
    assert_smem_or_die(shmem_bytes, "grouped_mlp2_mxfp4_bias_kernel_unified_A_bf16");
#endif

    hipLaunchKernelGGL(
        (grouped_mlp2_mxfp4_bias_kernel_unified_A_bf16<
            WM, WN, WK,
            WAVES_M, WAVES_N, WAVES_K,
            TW_M, TW_N,
            PAD_K_MC>),
        grid, block, shmem_bytes, s,
        /*C*/ C,
        /*A*/ A_bf16,
        /*W*/ W_packed,
        /*S*/ S_scales_f32,
        /*b*/ bias_bf16,
        /*routing*/ expert_offsets, expert_counts,
        /*tiles*/   tile2expert, tile2local,
        /*E,K,N*/ E, K, N);
    HIP_CHECK(hipGetLastError());
}

