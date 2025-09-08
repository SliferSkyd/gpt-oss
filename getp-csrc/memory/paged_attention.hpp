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

class PagedAttentionManager {
private:
    float** h_block_table;
    float** d_block_table;
    int kv_dim;
    size_t table_size_bytes;

public:
    PagedAttentionManager(int kv_dim) : kv_dim(kv_dim) {
        // Use sizeof(float*) – pointer size is the same, but make it consistent
        table_size_bytes = (size_t)BATCH_SIZE * (size_t)PAGES_PER_SEQ * sizeof(float*);

        // Host table
        h_block_table = (float**)malloc(table_size_bytes);
        if (!h_block_table) { fprintf(stderr,"Failed to alloc host block table\n"); exit(EXIT_FAILURE); }
        memset(h_block_table, 0, table_size_bytes);

        // Device table
        HIP_CHECK(hipMalloc((void**)&d_block_table, table_size_bytes));
        HIP_CHECK(hipMemset(d_block_table, 0, table_size_bytes)); // start clean on device too
    }

    ~PagedAttentionManager() {
        for (int i = 0; i < BATCH_SIZE * PAGES_PER_SEQ; ++i) {
            if (h_block_table[i] != nullptr) HIP_CHECK(hipFree(h_block_table[i]));
        }
        if (h_block_table) free(h_block_table);
        if (d_block_table) HIP_CHECK(hipFree(d_block_table));
    }

    int get_page_id(int seq_id, int token_pos) {
        if (seq_id < 0 || seq_id >= BATCH_SIZE) return -1;
        if (token_pos < 0 || token_pos >= PAGES_PER_SEQ * PAGE_SIZE) return -1;
        return seq_id * PAGES_PER_SEQ + token_pos / PAGE_SIZE;
    }

    // Allocate a page large enough for K and V, both as float (FP32)
    void extend_new_block(int seq_id, int token_pos) {
        const int block_id = get_page_id(seq_id, token_pos);
        if (block_id < 0) return;
        if (h_block_table[block_id] != nullptr) return;

        const size_t elems = 2ull * PAGE_SIZE * (size_t)kv_dim; // K + V
        HIP_CHECK(hipMalloc((void**)&h_block_table[block_id], elems * sizeof(float)));
    }

    void free_past_blocks(int seq_id, int current_pos, int window_size) {
        if (window_size <= 0 || current_pos < window_size) return;

        const int window_start_token = current_pos - window_size;
        int last_stale_page_idx = (window_start_token - 1) / PAGE_SIZE;
        if (last_stale_page_idx < 0) return;

        last_stale_page_idx = min(last_stale_page_idx, PAGES_PER_SEQ - 1);
        for (int page_offset = 0; page_offset <= last_stale_page_idx; ++page_offset) {
            const int block_id = seq_id * PAGES_PER_SEQ + page_offset;
            if (block_id < 0 || block_id >= BATCH_SIZE * PAGES_PER_SEQ) continue;
            if (h_block_table[block_id] != nullptr) {
                HIP_CHECK(hipFree(h_block_table[block_id]));
                h_block_table[block_id] = nullptr;
            }
        }
    }

    void sync_to_device() {
        HIP_CHECK(hipMemcpy(d_block_table, h_block_table, table_size_bytes, hipMemcpyHostToDevice));
    }

    float** get_device_block_table() { return d_block_table; }
};
