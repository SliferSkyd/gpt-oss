#pragma once

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "config.hpp"
#include "kernels/attention.hpp"

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
    // Pointers for both host and device
    __hip_bfloat16** h_block_table; // Host-side table for management
    __hip_bfloat16** d_block_table; // Device-side table for kernels
    int kv_dim;
    size_t table_size_bytes;

public:
    PagedAttentionManager(int kv_dim) {
        this->kv_dim = kv_dim;
        this->table_size_bytes = BATCH_SIZE * PAGES_PER_SEQ * sizeof(__hip_bfloat16*);

        // 1. Allocate host-side table and initialize to null
        h_block_table = (__hip_bfloat16**)malloc(table_size_bytes);
        if (h_block_table == nullptr) {
            fprintf(stderr, "Failed to allocate host memory for block table\n");
            exit(EXIT_FAILURE);
        }
        memset(h_block_table, 0, table_size_bytes);

        // 2. Allocate device-side table
        HIP_CHECK(hipMalloc((void **)&d_block_table, table_size_bytes));
    }

    ~PagedAttentionManager() {
        // Free all allocated physical blocks on the GPU
        for (int i = 0; i < BATCH_SIZE * PAGES_PER_SEQ; ++i) {
            if (h_block_table[i] != nullptr) {
                hipFree(h_block_table[i]);
            }
        }
        // Free the tables themselves
        if (h_block_table) free(h_block_table);
        if (d_block_table) HIP_CHECK(hipFree(d_block_table));
    }

    int get_page_id(int seq_id, int token_pos) {
        if (seq_id < 0 || seq_id >= BATCH_SIZE) return -1;
        if (token_pos < 0 || token_pos >= PAGES_PER_SEQ * PAGE_SIZE) return -1;
        return seq_id * PAGES_PER_SEQ + token_pos / PAGE_SIZE;
    }

    void extend_new_block(int seq_id, int token_pos) { 
        int block_id = get_page_id(seq_id, token_pos);
        if (block_id == -1) return;
        // Check and allocate on the host-side table
        if (h_block_table[block_id] != nullptr) return; // already allocated

        // Allocate a new physical block and store its pointer in the HOST table
        HIP_CHECK(hipMalloc((void **)&h_block_table[block_id], 2 * PAGE_SIZE * kv_dim * sizeof(__hip_bfloat16)));
    }

    // New function to sync host table to device
    void sync_to_device() {
        HIP_CHECK(hipMemcpy(d_block_table, h_block_table, table_size_bytes, hipMemcpyHostToDevice));
    }

    __hip_bfloat16** get_device_block_table() {
        return d_block_table;
    }
};