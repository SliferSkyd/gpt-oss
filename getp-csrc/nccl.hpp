#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#ifndef MAX_TP
#define MAX_TP 8
#endif

#ifndef HIP_CHECK
#define HIP_CHECK(stmt) do {                                       \
  hipError_t err = (stmt);                                         \
  if (err != hipSuccess) {                                         \
    fprintf(stderr, "HIP error %s at %s:%d\n",                     \
            hipGetErrorString(err), __FILE__, __LINE__);           \
    abort();                                                       \
  }                                                                \
} while(0)
#endif

// Simple per-device view of a tensor-parallel (TP) group.
// One TPGroup is created per GPU (device) and initialized while that device
// is current. It records the group's device ids, this device's rank in the
// group, and a p2p reachability matrix (indexed by group-local ranks).
struct TPGroup {
  int world_size{1};            // total GPUs visible to the process
  int tp_size{1};               // GPUs per TP group
  int world_rank{0};            // this device id (0..world_size-1)
  int group_id{0};              // world_rank / tp_size
  int rank_in_group{0};         // world_rank % tp_size

  int  group_devs[MAX_TP]{};    // device ids in this TP group
  bool p2p[MAX_TP][MAX_TP]{};   // p2p[i][j] => can device i access device j

  hipStream_t comm{nullptr};    // lightweight comm stream for helper copies
};

// Compute p2p reachability for the whole group, and enable peer access
// FROM this device TO each peer it can reach. This should be called with
// the current device already set to tp.world_rank.
inline void tp_enable_p2p(TPGroup &tp) {
  // Fill reachability matrix (independent of current device)
  for (int i = 0; i < tp.tp_size; ++i) {
    for (int j = 0; j < tp.tp_size; ++j) {
      if (i == j) { tp.p2p[i][j] = true; continue; }
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, tp.group_devs[i], tp.group_devs[j]));
      tp.p2p[i][j] = (can != 0);
    }
  }

  // Enable peer access from *this* device to its peers when possible.
  int saved = -1;
  HIP_CHECK(hipGetDevice(&saved));
  if (saved != tp.world_rank) HIP_CHECK(hipSetDevice(tp.world_rank));

  for (int j = 0; j < tp.tp_size; ++j) {
    if (j == tp.rank_in_group) continue;
    if (!tp.p2p[tp.rank_in_group][j]) continue;

    hipError_t e = hipDeviceEnablePeerAccess(tp.group_devs[j], 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "[TP] hipDeviceEnablePeerAccess(%d -> %d) failed: %s\n",
              tp.world_rank, tp.group_devs[j], hipGetErrorString(e));
      // don't abort; we’ll fall back to host-bounce in user code
    }
  }

  if (saved != tp.world_rank) HIP_CHECK(hipSetDevice(saved));
}

// Initialize TPGroup for the current device.
inline void tp_init(TPGroup &tp, int world_size, int world_rank, int tp_size) {
  tp.world_size    = world_size;
  tp.world_rank    = world_rank;
  tp.tp_size       = tp_size <= 0 ? 1 : tp_size;
  tp.group_id      = world_rank / tp.tp_size;
  tp.rank_in_group = world_rank % tp.tp_size;

  // Fill group device ids (contiguous partitioning).
  const int base = tp.group_id * tp.tp_size;
  for (int i = 0; i < tp.tp_size; ++i) tp.group_devs[i] = base + i;

  // Create a non-blocking helper stream on this device.
  HIP_CHECK(hipStreamCreateWithFlags(&tp.comm, hipStreamNonBlocking));

  // Compute reachability and enable peer access from this device.
  tp_enable_p2p(tp);
}

// Destroy helper stream. (No device memory owned here.)
inline void tp_free(TPGroup &tp) {
  if (tp.comm) {
    HIP_CHECK(hipStreamDestroy(tp.comm));
    tp.comm = nullptr;
  }
}

// tp_no_nccl.hpp (add)
#include <atomic>

#ifndef MAX_TP_GROUPS
#define MAX_TP_GROUPS 8
#endif

struct TPBarrierState {
  std::atomic<int> count[MAX_TP_GROUPS];
  std::atomic<uint64_t> epoch[MAX_TP_GROUPS];
  TPBarrierState() {
    for (int i = 0; i < MAX_TP_GROUPS; ++i) { count[i].store(0); epoch[i].store(0); }
  }
};
inline TPBarrierState& tp_barrier_state() {
  static TPBarrierState s;
  return s;
}

// Barrier for one TP group
inline void tp_group_barrier(const TPGroup& tp) {
  auto& S = tp_barrier_state();
  const int gid = tp.group_id % MAX_TP_GROUPS;
  const int need = tp.tp_size;

  const uint64_t my_epoch = S.epoch[gid].load(std::memory_order_acquire);
  const int old = S.count[gid].fetch_add(1, std::memory_order_acq_rel);

  if (old + 1 == need) {
    // last in: reset and advance epoch
    S.count[gid].store(0, std::memory_order_release);
    S.epoch[gid].fetch_add(1, std::memory_order_acq_rel);
  } else {
    // spin until epoch advances
    while (S.epoch[gid].load(std::memory_order_acquire) == my_epoch) {
      __asm__ __volatile__ ("" ::: "memory"); // tiny pause
    }
  }
}
