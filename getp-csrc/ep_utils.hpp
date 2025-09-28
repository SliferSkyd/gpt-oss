// ep_utils.hpp
#pragma once
#include <hip/hip_runtime.h>
#include "nccl.hpp"      // for MAX_TP
#include "tp_ring.hpp"   // TPGroup, tp_group_barrier
#include "config.hpp"
#include <cstdio>

// Partition experts evenly and contiguously across ranks of the TP group.
static inline void ep_owner_map(const TPGroup& tp, int E,
                                int &E_loc, int &E_start)
{
    const int TP = tp.tp_size;
    const int r  = tp.rank_in_group;
    const int base = E / TP, rem = E % TP;
    E_loc   = base + (r < rem ? 1 : 0);
    E_start = r * base + (r < rem ? r : rem);
}

// Owner rank (group-local 0..TP-1) for global expert e.
static inline int ep_owner_rank(int e, int TP, int rem, int base)
{
    // inverse of the contiguous split above
    // find smallest r such that e < r*base + min(r,rem) + (base + (r<rem))
    // O(TP) loop is fine for small TP (<=8)
    int acc = 0;
    for (int r = 0; r < TP; ++r) {
        const int len = base + (r < rem ? 1 : 0);
        if (e < acc + len) return r;
        acc += len;
    }
    return TP-1; // fallback
}

static inline void ep_global_to_local(int e, int E_start, int &e_local)
{
    e_local = e - E_start; // caller must ensure 0 <= e_local < E_loc
}

// Build contiguous [0..TP) device send sizes from expert counts per token.
// K is small (typically 2).
static inline void ep_compute_peer_counts(const TPGroup& tp,
                                          int E, const int* __restrict__ topk_i, int Bgrp, int K,
                                          int counts_peer[MAX_TP])
{
    const int TP = tp.tp_size;
    const int base = E / TP, rem = E % TP;
    for (int r = 0; r < TP; ++r) counts_peer[r] = 0;

    for (int t = 0; t < Bgrp; ++t) {
        const int* ti = topk_i + t*K;
        for (int k = 0; k < K; ++k) {
            const int e = ti[k];
            const int r = ep_owner_rank(e, TP, rem, base);
            counts_peer[r] += 1;
        }
    }
}

// Prefix sums for up to MAX_TP
static inline void ep_prefix_sum_small(int TP, int arr[MAX_TP])
{
    int run = 0;
    for (int i = 0; i < TP; ++i) { int v = arr[i]; arr[i] = run; run += v; }
}

// Returns peer device id from group-local rank
static inline int ep_peer_dev(const TPGroup& tp, int r_local) { return tp.group_devs[r_local]; }
