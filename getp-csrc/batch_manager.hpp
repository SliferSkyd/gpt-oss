#pragma once

#include "config.hpp"

// Page-based KV cache block management
struct PagedBlock {
    int block_id;
    int ref_count;  // Reference count for sharing
    bool is_free;
};

// Page table entry for mapping logical pages to physical blocks
struct PageTableEntry {
    int physical_block_id;
    bool is_allocated;
};

// Per-sequence page management - simplified to avoid template issues
struct SequencePageInfo {
    PageTableEntry page_table[MAX_PAGES_PER_SEQ];  // Maps logical page -> physical block
    int num_pages;                                  // Current number of allocated pages
    int max_pages;                                 // Maximum pages needed for this sequence
};

// Memory pool manager for paged attention
class PagedAttentionManager {
private:
    PagedBlock block_pool[MAX_BLOCKS];             // Pool of physical blocks
    int free_blocks[MAX_BLOCKS];                   // Array of free block IDs  
    int free_blocks_count;                         // Number of free blocks
    SequencePageInfo sequence_pages[BATCH_SIZE];   // Per-sequence page info
    
    // GPU memory pointers
    __hip_bfloat16 *paged_key_cache;        // (MAX_BLOCKS, PAGE_SIZE, kv_dim)
    __hip_bfloat16 *paged_value_cache;      // (MAX_BLOCKS, PAGE_SIZE, kv_dim)
    int *block_table;                     // (BATCH_SIZE, MAX_PAGES_PER_SEQ) - maps seq page to physical block
    
    int kv_dim;
    int n_layers;

public:
    PagedAttentionManager() 
        : paged_key_cache(nullptr), paged_value_cache(nullptr), block_table(nullptr),
        free_blocks_count(0), kv_dim(0), n_layers(0) {
        // Initialize all blocks as free
        for (int i = 0; i < MAX_BLOCKS; i++) {
            block_pool[i].block_id = i;
            block_pool[i].ref_count = 0;
            block_pool[i].is_free = true;
            free_blocks[i] = i;
        }
        free_blocks_count = MAX_BLOCKS;
        
        // Initialize sequence page info
        for (int i = 0; i < BATCH_SIZE; i++) {
            sequence_pages[i].num_pages = 0;
            sequence_pages[i].max_pages = 0;
            for (int j = 0; j < MAX_PAGES_PER_SEQ; j++) {
                sequence_pages[i].page_table[j].physical_block_id = -1;
                sequence_pages[i].page_table[j].is_allocated = false;
            }
        }
    }

    ~PagedAttentionManager() {
        if (paged_key_cache) {
            HIP_CHECK(hipFree(paged_key_cache));
        }
        if (paged_value_cache) {
            HIP_CHECK(hipFree(paged_value_cache));
        }
        if (block_table) {
            HIP_CHECK(hipFree(block_table));
        }
    }
    
    // Initialize the memory pool
    void initialize(int kv_dim, int n_layers) {
        this->kv_dim = kv_dim;
        this->n_layers = n_layers;
        
        // Calculate memory sizes for paged KV cache
        size_t key_cache_size = MAX_BLOCKS * PAGE_SIZE * kv_dim * n_layers * sizeof(__hip_bfloat16);
        size_t value_cache_size = MAX_BLOCKS * PAGE_SIZE * kv_dim * n_layers * sizeof(__hip_bfloat16);
        size_t block_table_size = BATCH_SIZE * MAX_PAGES_PER_SEQ * sizeof(int);
        
        printf("Initializing Paged Attention:\n");
        printf("  - Max blocks: %d\n", MAX_BLOCKS);
        printf("  - Page size: %d tokens\n", PAGE_SIZE);
        printf("  - KV cache size: %.2f MB\n", (key_cache_size + value_cache_size) / (1024.0 * 1024.0));
        printf("  - Block table size: %.2f KB\n", block_table_size / 1024.0);
        
        // Allocate GPU memory for paged KV cache
        HIP_CHECK(hipMalloc((void**)&paged_key_cache, key_cache_size));
        HIP_CHECK(hipMalloc((void**)&paged_value_cache, value_cache_size));
        HIP_CHECK(hipMalloc((void**)&block_table, block_table_size));
        
        // Initialize memory to zero
        HIP_CHECK(hipMemset(paged_key_cache, 0, key_cache_size));
        HIP_CHECK(hipMemset(paged_value_cache, 0, value_cache_size));
        HIP_CHECK(hipMemset(block_table, -1, block_table_size)); // -1 indicates unallocated
    }
    
    // Allocate pages for a new sequence
    bool allocate_pages(int seq_id, int required_pages) {
        if (seq_id >= BATCH_SIZE || required_pages <= 0 || required_pages > MAX_PAGES_PER_SEQ) {
            return false;
        }
        
        if (free_blocks_count < required_pages) {
            printf("Warning: Not enough free blocks for sequence %d (need %d, have %d)\n", 
                    seq_id, required_pages, free_blocks_count);
            return false;
        }
            
        SequencePageInfo& seq_info = sequence_pages[seq_id];
        seq_info.num_pages = required_pages;
        seq_info.max_pages = required_pages;
        
        // Allocate physical blocks for each logical page
        for (int page = 0; page < required_pages; page++) {
            int block_id = allocate_block();
            if (block_id == -1) {
                // Rollback partial allocation
                for (int rollback = 0; rollback < page; rollback++) {
                    free_block(seq_info.page_table[rollback].physical_block_id);
                    seq_info.page_table[rollback].physical_block_id = -1;
                    seq_info.page_table[rollback].is_allocated = false;
                }
                seq_info.num_pages = 0;
                return false;
            }
            
            seq_info.page_table[page].physical_block_id = block_id;
            seq_info.page_table[page].is_allocated = true;
        }
        
        return true;
    }
    
    // Free pages for a completed sequence
    void free_pages(int seq_id) {
        if (seq_id >= BATCH_SIZE) return;
    
        SequencePageInfo& seq_info = sequence_pages[seq_id];
        
        // Free all allocated pages
        for (int page = 0; page < seq_info.num_pages; page++) {
            if (seq_info.page_table[page].is_allocated) {
                free_block(seq_info.page_table[page].physical_block_id);
                seq_info.page_table[page].physical_block_id = -1;
                seq_info.page_table[page].is_allocated = false;
            }
        }
        
        seq_info.num_pages = 0;
        seq_info.max_pages = 0;
    }
    
    // Extend pages for a sequence (during generation)
    bool extend_pages(int seq_id, int additional_pages) {
        if (seq_id >= BATCH_SIZE || additional_pages <= 0) {
            return false;
        }
        
        SequencePageInfo& seq_info = sequence_pages[seq_id];
        int new_total = seq_info.num_pages + additional_pages;
        
        if (new_total > MAX_PAGES_PER_SEQ || free_blocks_count < additional_pages) {
            return false;
        }
        
        // Allocate additional blocks
        for (int i = 0; i < additional_pages; i++) {
            int page_idx = seq_info.num_pages + i;
            int block_id = allocate_block();
            if (block_id == -1) {
                // Rollback partial extension
                for (int rollback = seq_info.num_pages; rollback < page_idx; rollback++) {
                    free_block(seq_info.page_table[rollback].physical_block_id);
                    seq_info.page_table[rollback].physical_block_id = -1;
                    seq_info.page_table[rollback].is_allocated = false;
                }
                return false;
            }
            
            seq_info.page_table[page_idx].physical_block_id = block_id;
            seq_info.page_table[page_idx].is_allocated = true;
        }
        
        seq_info.num_pages = new_total;
        if (seq_info.max_pages < new_total) {
            seq_info.max_pages = new_total;
        }
        
        return true;
    }
    
    // Get the block table for GPU kernels
    int* get_block_table() { return block_table; }
    
    // Get KV cache pointers for GPU kernels
    __hip_bfloat16* get_key_cache() { return paged_key_cache; }
    __hip_bfloat16* get_value_cache() { return paged_value_cache; }
    
    // Get memory usage statistics
    int get_free_blocks() const { return free_blocks_count; }
    int get_total_blocks() const { return MAX_BLOCKS; }
    
    // Update block table on GPU for a batch
    void update_block_table_gpu(int* active_seq_ids, int num_active) {
        // Prepare CPU block table data
        int host_block_table[BATCH_SIZE * MAX_PAGES_PER_SEQ];
        memset(host_block_table, -1, sizeof(host_block_table));
        
        // Fill in the block table for active sequences
        for (int i = 0; i < num_active; i++) {
            int seq_id = active_seq_ids[i];
            if (seq_id >= BATCH_SIZE) continue;
            
            SequencePageInfo& seq_info = sequence_pages[seq_id];
            for (int page = 0; page < seq_info.num_pages; page++) {
                if (seq_info.page_table[page].is_allocated) {
                    host_block_table[seq_id * MAX_PAGES_PER_SEQ + page] = 
                        seq_info.page_table[page].physical_block_id;
                }
            }
        }
        
        // Copy to GPU
        HIP_CHECK(hipMemcpy(block_table, host_block_table, 
                        BATCH_SIZE * MAX_PAGES_PER_SEQ * sizeof(int),
                        hipMemcpyHostToDevice));
    }
    
private:
    int allocate_block() {
        if (free_blocks_count == 0) {
            return -1;  // No free blocks
        }
        
        int block_id = free_blocks[free_blocks_count - 1];
        free_blocks_count--;
        
        block_pool[block_id].is_free = false;
        block_pool[block_id].ref_count = 1;
        
        return block_id;
    }

    void free_block(int block_id) {
        if (block_id < 0 || block_id >= MAX_BLOCKS) return;
    
        PagedBlock& block = block_pool[block_id];
        block.ref_count--;
        
        if (block.ref_count <= 0) {
            block.is_free = true;
            block.ref_count = 0;
            free_blocks[free_blocks_count] = block_id;
            free_blocks_count++;
        }
    }
};