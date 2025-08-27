#pragma once

#define BATCH_SIZE 960LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024
#define N_MLP_STREAMS 4 // tune me (4–8 is usually good)
