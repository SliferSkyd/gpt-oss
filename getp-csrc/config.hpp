#pragma once

#define BATCH_SIZE 32LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define PAGE_SIZE 16
#define INVALID_PAGE -1
#define MAX_BLOCKS 2048
#define PAGES_PER_SEQ 64