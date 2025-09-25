#pragma once

size_t BATCH_SIZE = 1024LL;
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024

#define MAX_GPUS 8

int TENSOR_PARALLEL_SIZE = 2;   

#define RING_TILE_BYTES (8u << 20)  // 8 MiB tile; tune 4–32 MiB depending on your platform


#ifndef BF16_KEY_TOKENS
#define BF16_KEY_TOKENS 64
#endif

#ifndef BF16_VALUE_TOKENS
#define BF16_VALUE_TOKENS 64
#endif