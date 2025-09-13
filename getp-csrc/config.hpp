#pragma once

#define BATCH_SIZE 1280LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024

#define MAX_GPUS 8

#define TENSOR_PARALLEL_SIZE 2   // change as you like
