#pragma once

#define BATCH_SIZE 512LL
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 64

#define MAX_SEQ_LEN 1024

#define MAX_GPUS 8

#define TENSOR_PARALLEL_SIZE 4   // change as you like

#ifndef RING_TILE_BYTES
// 8 MiB tile; tune 4–32 MiB depending on your platform
#define RING_TILE_BYTES (8u << 20)
#endif