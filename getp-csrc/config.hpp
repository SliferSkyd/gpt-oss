#pragma once

#define BATCH_SIZE 512LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

// Paged Attention Configuration
#define PAGE_SIZE 16LL           // Number of tokens per page
#define MAX_PAGES_PER_SEQ 128LL          // Maximum pages per sequence (2048 tokens / 16)
#define MAX_BLOCKS 8192LL * 4               // Total number of blocks in memory pool