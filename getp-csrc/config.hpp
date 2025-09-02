#pragma once

#define BATCH_SIZE 32LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024
#define N_MLP_STREAMS 8 // tune me (4–8 is usually good)

#define MAX_GPUS 8

// FP8 KV Cache Configuration
#define USE_FP8_KV_CACHE 1  // Enable FP8 quantization for KV cache (0=BF16, 1=FP8)
#define FP8_MIXED_PRECISION_TOKENS 8  // Keep first N tokens in BF16 for accuracy