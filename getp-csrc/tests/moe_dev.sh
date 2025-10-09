#!/bin/bash
#SBATCH -J gpt_inference
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --signal=USR1@60       # cảnh báo 60s trước khi hết giờ/preempt

set -euo pipefail

################################
# 0) Modules & ENV
################################
module load rocm || true       # thường Makefile sẽ gọi hipcc từ ROCm
# module load gcc cmake ninja || true   # nếu Makefile cần

################################
# 1) Đường dẫn & tuỳ chọn
################################
# Thư mục project có Makefile & code (chạy make tại đây)
PROJ_DIR="/nfs/gpu_trainee/getp02/gpt-oss/getp-csrc/tests"  # <-- SỬA: đường dẫn project của bạn
CONTROL_ENV="$PROJ_DIR/control.env"
STOP_FILE="$PROJ_DIR/STOP"

# Binary mặc định sau khi make
BIN="$PROJ_DIR/bench_moe"      # có thể override trong control.env

# Runtime options cho bench_moe (override được trong control.env)
RUN_OPTS="--iters 50 --warmup 5"

# Build options (override được trong control.env)
MAKE_JOBS="${MAKE_JOBS:-8}"
MAKE_ARGS="${MAKE_ARGS:-}"     # ví dụ: AMD_ARCH=gfx90a, ROCM_PATH=/opt/rocm

# Polling & idle
POLL_SEC=5
IDLE_TIMEOUT_MIN=120           # tự thoát nếu không chạy gì quá X phút

mkdir -p logs

################################
# 2) Helpers
################################
log() { echo "[$(date +'%F %T')] $*"; }

# Dò thay đổi source/Makefile bằng chữ ký
src_sig() {
  find "$PROJ_DIR" -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.hip.cpp' -o -name 'Makefile' -o -name '*.mk' \) \
    -print0 | sort -z | xargs -0 sha1sum 2>/dev/null | sha1sum | awk '{print $1}'
}

# Nạp control.env (cho phép override biến)
load_control_env() {
  if [[ -f "$CONTROL_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$CONTROL_ENV"; set +a
    RUN_OPTS="${RUN_OPTS:-${RUN_ARGS:-$RUN_OPTS}}"
    BIN="${BIN:-$PROJ_DIR/bench_moe}"
    MAKE_JOBS="${MAKE_JOBS:-8}"
    MAKE_ARGS="${MAKE_ARGS:-}"
    POLL_SEC="${POLL_SEC:-$POLL_SEC}"
    IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-$IDLE_TIMEOUT_MIN}"
  fi
}

# Backoff khi build lỗi (tránh spam)
BUILD_FAILS=0
build_bin() {
  log ">> make clean && make bench_moe  (jobs=$MAKE_JOBS $MAKE_ARGS)"
  make -C "$PROJ_DIR" clean || true
  if ! MAKEFLAGS="-j$MAKE_JOBS" make -C "$PROJ_DIR" bench_moe $MAKE_ARGS; then
    BUILD_FAILS=$((BUILD_FAILS+1))
    local S=$(( 2 ** (BUILD_FAILS>5?5:BUILD_FAILS) ))
    log "!! Build failed (count=$BUILD_FAILS). Backoff ${S}s, chờ bạn sửa…"
    sleep "$S"
    return 1
  fi
  BUILD_FAILS=0
  # Tìm binary nếu path khác
  if [[ ! -x "$BIN" ]]; then
    for cand in "$PROJ_DIR/bench_moe" "$PROJ_DIR/build/bench_moe" "$PROJ_DIR/bin/bench_moe"; do
      [[ -x "$cand" ]] && BIN="$cand" && break
    done
    if [[ ! -x "$BIN" ]]; then
      local cand; cand=$(find "$PROJ_DIR" -maxdepth 3 -type f -name bench_moe -perm -u+x | head -n1 || true)
      [[ -n "${cand:-}" ]] && BIN="$cand"
    fi
  fi
  if [[ ! -x "$BIN" ]]; then
    log "!! Không tìm thấy binary bench_moe sau khi build."
    return 1
  fi
  log ">> BUILD DONE: $BIN"
}

run_bin() {
  log ">> RUN: $BIN $RUN_OPTS"
  # Dùng srun để SLURM quản lý step
  srun -n 1 "$BIN" $RUN_OPTS || return $?
}

# Traps để thoát gọn
trap 'log "Got USR1 (preempt/timeout soon) -> exit."; exit 0' USR1
trap 'log "Got TERM/INT -> exit."; exit 0' TERM INT

################################
# 3) Main loop
################################
log "===== MOE DEV SESSION START ====="
log "PROJ_DIR=$PROJ_DIR"
log "CONTROL_ENV=$CONTROL_ENV"

LAST_SIG="$(src_sig || echo none)"
LAST_RUN_TS="$(date +%s)"
REBUILD_NEEDED=1
HAS_RUN_SINCE_BUILD=1

load_control_env

# Build ban đầu
if build_bin; then
  REBUILD_NEEDED=0
  HAS_RUN_SINCE_BUILD=0
else
  log "Initial build failed. Sẽ chờ bạn sửa hoặc UPDATED=1…"
fi

while true; do
  [[ -f "$STOP_FILE" ]] && { log "STOP found. Bye."; exit 0; }

  load_control_env

  # Cờ điều khiển nóng
  if [[ "${UPDATED:-0}" == "1" || "${FORCE_REBUILD:-0}" == "1" ]]; then
    REBUILD_NEEDED=1
  fi

  # Dò thay đổi source/Makefile
  CUR_SIG="$(src_sig || echo none)"
  if [[ "$CUR_SIG" != "$LAST_SIG" ]]; then
    log "Source/Makefile changed."
    REBUILD_NEEDED=1
    LAST_SIG="$CUR_SIG"
  fi

  # Rebuild nếu cần
  if [[ "$REBUILD_NEEDED" == "1" ]]; then
    if build_bin; then
      REBUILD_NEEDED=0
      HAS_RUN_SINCE_BUILD=0
      # reset UPDATED=0 nếu có
      if [[ -f "$CONTROL_ENV" ]]; then
        grep -q '^UPDATED=' "$CONTROL_ENV" && sed -i 's/^UPDATED=.*/UPDATED=0/' "$CONTROL_ENV" || true
      fi
    else
      log "Build failed. Chờ thay đổi hoặc UPDATED=1 rồi thử lại…"
    fi
  fi

  # Quyết định chạy: chạy 1 lần sau mỗi build, hoặc khi FORCE_RUN=1
  if [[ "$REBUILD_NEEDED" == "0" && ( "$HAS_RUN_SINCE_BUILD" == "0" || "${FORCE_RUN:-0}" == "1" ) ]]; then
    if run_bin; then
      LAST_RUN_TS="$(date +%s)"
      HAS_RUN_SINCE_BUILD=1
      # reset FORCE_RUN nếu có
      if [[ -f "$CONTROL_ENV" ]]; then
        grep -q '^FORCE_RUN=' "$CONTROL_ENV" && sed -i 's/^FORCE_RUN=.*/FORCE_RUN=0/' "$CONTROL_ENV" || true
      fi
    else
      log "Run failed (exit=$?). Chờ bạn sửa hoặc FORCE_RUN=1…"
    fi
  fi

  # Idle timeout → trả GPU
  NOW="$(date +%s)"; ELAP_MIN=$(( (NOW - LAST_RUN_TS)/60 ))
  if (( ELAP_MIN >= IDLE_TIMEOUT_MIN )); then
    log "Idle >= ${IDLE_TIMEOUT_MIN}min → exit to free GPU."
    exit 0
  fi

  # Đợi thay đổi: ưu tiên inotifywait; fallback sleep
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -qq -e modify,create,delete,move \
      -r "$PROJ_DIR" "$CONTROL_ENV" "$STOP_FILE" || true
  else
    sleep "$POLL_SEC"
  fi
done
