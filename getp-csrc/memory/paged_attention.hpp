#pragma once

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "../config.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Use the HIP_CHECK from utils.hpp if available, otherwise define fallback
#ifndef HIP_CHECK
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            std::exit(1); \
        } \
    } while(0)
#endif
// In paged_attention.hpp

#pragma once

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "../config.hpp"

#include <hip/hip_runtime.h>

// Use the HIP_CHECK from utils.hpp if available, otherwise define fallback
#ifndef HIP_CHECK
#define HIP_CHECK(call) \
    do { \
        hipError_t err__ = (call); \
        if (err__ != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
            std::exit(1); \
        } \
    } while(0)
#endif

// -------------------- PagedAttentionManager (manager-only change) --------------------
class PagedAttentionManager {
private:
    // Host/device lookup tables used by kernels (fixed size = BATCH_SIZE * PAGES_PER_SEQ)
    float** h_block_table;
    float** d_block_table;

    // Per-seq ring pool (allocated lazily when sliding-window is enabled)
    // Size: [BATCH_SIZE * ring_pages]. Null until ring mode is enabled.
    float** ring_slots = nullptr;

    int     kv_dim;
    size_t  table_size_bytes;

    // Ring mode is enabled automatically the first time free_past_blocks() is called
    bool    ring_mode  = false;
    int     ring_pages = 0;   // number of pages kept by the sliding window

    static inline int div_floor(int a, int b) { return a >= 0 ? (a / b) : ((a - (b-1)) / b); }

public:
    // Keep the same ctor signature so build sites don't change
    explicit PagedAttentionManager(int kv_dim)
        : kv_dim(kv_dim)
    {
        table_size_bytes = (size_t)BATCH_SIZE * (size_t)PAGES_PER_SEQ * sizeof(float*);

        h_block_table = (float**)std::malloc(table_size_bytes);
        if (!h_block_table) {
            std::fprintf(stderr, "Failed to allocate host block table\n");
            std::exit(EXIT_FAILURE);
        }
        std::memset(h_block_table, 0, table_size_bytes);

        HIP_CHECK(hipMalloc((void**)&d_block_table, table_size_bytes));
        HIP_CHECK(hipMemset(d_block_table, 0, table_size_bytes));
    }

    ~PagedAttentionManager() {
        // Free ring slots if they exist
        if (ring_slots) {
            for (int s = 0; s < BATCH_SIZE * ring_pages; ++s) {
                if (ring_slots[s] != nullptr) {
                    HIP_CHECK(hipFree(ring_slots[s]));
                    ring_slots[s] = nullptr;
                }
            }
            std::free(ring_slots);
            ring_slots = nullptr;
        }

        // Free any remaining non-ring pages referenced in the table
        // (avoid double-free by skipping pointers that are in ring_slots)
        for (int i = 0; i < BATCH_SIZE * PAGES_PER_SEQ; ++i) {
            float* p = h_block_table[i];
            if (!p) continue;

            bool is_ring = false;
            if (ring_mode && ring_slots) {
                for (int r = 0; r < ring_pages; ++r) {
                    // check membership in this seq’s ring set
                    int seq = i / PAGES_PER_SEQ;
                    if (ring_slots[seq * ring_pages + r] == p) { is_ring = true; break; }
                }
            }
            if (!is_ring) HIP_CHECK(hipFree(p));
            h_block_table[i] = nullptr;
        }

        if (h_block_table) std::free(h_block_table);
        if (d_block_table) HIP_CHECK(hipFree(d_block_table));
    }

    // Logical index into the (fixed-size) table that kernels also use.
    inline int get_page_id(int seq_id, int token_pos) {
        if (seq_id < 0 || seq_id >= BATCH_SIZE) return -1;
        if (token_pos < 0) return -1;
        const int logical_page = token_pos / PAGE_SIZE;
        if (logical_page >= PAGES_PER_SEQ) return -1;  // guard against OOB
        return seq_id * PAGES_PER_SEQ + logical_page;
    }

    // Allocate a page (or reuse a ring slot) and point the table entry at it.
    void extend_new_block(int seq_id, int token_pos) {
        const int block_id = get_page_id(seq_id, token_pos);
        if (block_id < 0) return;

        // Sliding-window ON => reuse one of ring_pages slots for this seq
        if (ring_mode) {
            // Lazy ring pool allocation
            if (!ring_slots) {
                ring_slots = (float**)std::malloc((size_t)BATCH_SIZE * ring_pages * sizeof(float*));
                if (!ring_slots) { std::fprintf(stderr, "Failed to alloc ring slots\n"); std::exit(EXIT_FAILURE); }
                std::memset(ring_slots, 0, (size_t)BATCH_SIZE * ring_pages * sizeof(float*));
            }
            const int logical_page = token_pos / PAGE_SIZE;
            const int slot         = logical_page % ring_pages;                 // select ring slot
            float*& slot_ptr       = ring_slots[seq_id * ring_pages + slot];

            if (!slot_ptr) {
                const size_t elems = 2ull * PAGE_SIZE * (size_t)kv_dim;         // K + V rows
                HIP_CHECK(hipMalloc((void**)&slot_ptr, elems * sizeof(float)));
            }
            // Map this logical page to the chosen ring slot
            h_block_table[block_id] = slot_ptr;
            return;
        }

        // Full-history mode (odd layer): allocate once per logical page
        if (h_block_table[block_id] == nullptr) {
            const size_t elems = 2ull * PAGE_SIZE * (size_t)kv_dim;             // K + V rows
            HIP_CHECK(hipMalloc((void**)&h_block_table[block_id], elems * sizeof(float)));
        }
    }

    // Host tells us the current pos & window; we:
    //  * On first call: enable ring mode sized to window (even layers),
    //  * Free pages older than the window (for pre-ring allocations),
    //  * For odd layers (never called), nothing changes: they keep full history.
    void free_past_blocks(int seq_id, int current_pos, int window_size) {
        if (window_size <= 0 || current_pos < window_size) return;

        // Enable ring mode on first call
        if (!ring_mode) {
            ring_mode  = true;
            ring_pages = std::max(1, (window_size + PAGE_SIZE - 1) / PAGE_SIZE);
            // Free everything strictly older than the window immediately
            const int window_start_token = current_pos - window_size;
            int last_stale_page_idx = div_floor(window_start_token - 1, PAGE_SIZE);
            last_stale_page_idx = std::min(last_stale_page_idx, PAGES_PER_SEQ - 1);
            for (int page = 0; page <= last_stale_page_idx; ++page) {
                const int block_id = seq_id * PAGES_PER_SEQ + page;
                float* p = h_block_table[block_id];
                if (p) { HIP_CHECK(hipFree(p)); h_block_table[block_id] = nullptr; }
            }
            return;
        }

        // Already in ring mode: free any pre-ring pages that finally slid out
        const int window_start_token = current_pos - window_size;
        int last_stale_page_idx = div_floor(window_start_token - 1, PAGE_SIZE);
        if (last_stale_page_idx < 0) return;
        last_stale_page_idx = std::min(last_stale_page_idx, PAGES_PER_SEQ - 1);

        for (int page = 0; page <= last_stale_page_idx; ++page) {
            const int block_id = seq_id * PAGES_PER_SEQ + page;
            float* p = h_block_table[block_id];
            if (!p) continue;

            // Don't free ring slots. Just clear stale table entries.
            bool is_ring = false;
            if (ring_slots) {
                for (int r = 0; r < ring_pages; ++r) {
                    if (ring_slots[seq_id * ring_pages + r] == p) { is_ring = true; break; }
                }
            }
            if (!is_ring) { HIP_CHECK(hipFree(p)); }
            h_block_table[block_id] = nullptr;
        }
    }

    // Copy the whole mapping to device (unchanged kernel interfaces)
    void sync_to_device() {
        HIP_CHECK(hipMemcpy(d_block_table, h_block_table, table_size_bytes, hipMemcpyHostToDevice));
    }

    float** get_device_block_table() { return d_block_table; }
};
