#pragma once

#define BATCH_SIZE 64LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024
#define N_MLP_STREAMS 8 // tune me (4–8 is usually good)

#define MAX_GPUS 8