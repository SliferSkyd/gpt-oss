#pragma once

#define BATCH_SIZE 256LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024

#define MAX_GPUS 8

#define PAGE_SIZE 16
#define PAGES_PER_SEQ 64