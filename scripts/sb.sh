#!/bin/bash
#SBATCH -J run
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --signal=USR1@60       # báo trước 60s khi sắp hết giờ / preempt

set -euo pipefail

################################
# 0) Modules & ENV (tùy cluster)
################################
# Ví dụ ROCm cho AMD GPU; bỏ/đổi theo cụm của bạn
module load rocm || true
# module load gcc cmake ninja || true

################################
# 1) Mặc định (override bởi control.env)
################################
CONTROL_ENV="$HOME/control.env"   # <-- đặt control.env ở đâu tùy ý (nên dùng path tuyệt đối)
STOP_FILE="$HOME/STOP-KEEPER"     # tạo file này để dừng job sạch

POLL_SEC=5                        # chu kỳ polling nếu không có inotifywait
IDLE_TIMEOUT_MIN=0                # 0=tắt tự thoát; >0 = idle X phút sẽ tự thoát

RUN_ON_START=0                    # 1 = chạy ngay khi job start (không cần UPDATED)
WRAP_WITH_SRUN=1                  # 1 = dùng srun bao RUN_CMD
SRUN_ARGS="-n 1"                  # thêm args tùy ý (vd --cpus-per-task=8)

RUN_LOG_DIR="$HOME/runlogs"       # log riêng cho mỗi lần chạy
ENV_EXPORT=""                     # export biến môi trường trước khi chạy, ví dụ: 'export AMD_ARCH=gfx90a'

################################
# 2) Helpers
################################
log() { echo "[$(date +'%F %T')] $*"; }

load_control_env() {
  if [[ -f "$CONTROL_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$CONTROL_ENV"; set +a
    # Cho phép override defaults
    POLL_SEC="${POLL_SEC:-5}"
    IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-0}"
    RUN_ON_START="${RUN_ON_START:-0}"
    WRAP_WITH_SRUN="${WRAP_WITH_SRUN:-1}"
    SRUN_ARGS="${SRUN_ARGS:-"-n 1"}"
    RUN_LOG_DIR="${RUN_LOG_DIR:-$RUN_LOG_DIR}"
    ENV_EXPORT="${ENV_EXPORT:-$ENV_EXPORT}"
  fi
}

reset_updated_flag() {
  # Luôn đưa UPDATED về 0 để mỗi lần bạn muốn chạy lại thì bật lại UPDATED=1
  if [[ -f "$CONTROL_ENV" ]]; then
    if grep -q '^UPDATED=' "$CONTROL_ENV"; then
      sed -i 's/^UPDATED=.*/UPDATED=0/' "$CONTROL_ENV"
    else
      printf '\nUPDATED=0\n' >> "$CONTROL_ENV"
    fi
  fi
}

run_once() {
  mkdir -p "$RUN_LOG_DIR"
  local ts; ts="$(date +'%Y%m%d_%H%M%S')"
  local rlog="$RUN_LOG_DIR/run_${SLURM_JOB_ID}_${ts}.log"

  # WORK_DIR & RUN_CMD bắt buộc phải có trong control.env
  if [[ -z "${WORK_DIR:-}" || -z "${RUN_CMD:-}" ]]; then
    log "!! WORK_DIR hoặc RUN_CMD chưa được cấu hình trong control.env"
    echo "[ERROR] WORK_DIR/RUN_CMD missing" > "$rlog"
    return 2
  fi
  if [[ ! -d "$WORK_DIR" ]]; then
    log "!! WORK_DIR không tồn tại: $WORK_DIR"
    echo "[ERROR] WORK_DIR not found: $WORK_DIR" > "$rlog"
    return 3
  fi

  log ">> WORK_DIR=$WORK_DIR"
  log ">> RUN_CMD=$RUN_CMD"
  log ">> Log file: $rlog"

  (
    set +e
    cd "$WORK_DIR"
    # Xuất biến môi trường chung nếu có
    if [[ -n "${ENV_EXPORT:-}" ]]; then
      eval "$ENV_EXPORT"
    fi
    # Thực thi command
    if [[ "${WRAP_WITH_SRUN:-1}" -eq 1 ]]; then
      srun $SRUN_ARGS bash -lc "$RUN_CMD" &> "$rlog"
    else
      bash -lc "$RUN_CMD" &> "$rlog"
    fi
    rc=$?
    set -e
    exit $rc
  )
  local rc=$?

  if [[ $rc -eq 0 ]]; then
    log ">> RUN OK (rc=0). Output: $rlog"
  else
    log "!! RUN FAIL (rc=$rc). Xem log: $rlog"
  fi
  return $rc
}

trap 'log "Got USR1 (preempt/timeout soon) -> exit."; exit 0' USR1
trap 'log "Got TERM/INT -> exit."; exit 0' TERM INT

################################
# 3) Main loop
################################
log "===== RESOURCE KEEPER START ====="
log "CONTROL_ENV=$CONTROL_ENV"
mkdir -p "$(dirname "$CONTROL_ENV")" logs "$RUN_LOG_DIR"

load_control_env

LAST_RUN_TS="$(date +%s)"
HAS_RUN_THIS_CYCLE=1

# Chạy ngay khi start nếu bạn bật RUN_ON_START=1 trong control.env
if [[ "${RUN_ON_START:-0}" -eq 1 ]]; then
  HAS_RUN_THIS_CYCLE=0
fi

while true; do
  [[ -f "$STOP_FILE" ]] && { log "STOP file found: $STOP_FILE. Bye."; exit 0; }

  load_control_env

  # Khi UPDATED=1 -> cho phép chạy 1 lần
  if [[ "${UPDATED:-0}" == "1" ]]; then
    HAS_RUN_THIS_CYCLE=0
  fi

  if [[ "$HAS_RUN_THIS_CYCLE" -eq 0 ]]; then
    if run_once; then
      LAST_RUN_TS="$(date +%s)"
      HAS_RUN_THIS_CYCLE=1
      reset_updated_flag     # ✅ luôn reset về 0 sau khi chạy
    else
      LAST_RUN_TS="$(date +%s)"
      HAS_RUN_THIS_CYCLE=1
      reset_updated_flag     # ✅ lỗi cũng reset về 0 để không lặp vô hạn
      log "Run failed. Sửa cấu hình/code rồi đặt UPDATED=1 để thử lại."
    fi
  fi

  # Idle auto-exit (nếu bật)
  if [[ "${IDLE_TIMEOUT_MIN:-0}" -gt 0 ]]; then
    now="$(date +%s)"
    elap=$(( (now - LAST_RUN_TS)/60 ))
    if (( elap >= IDLE_TIMEOUT_MIN )); then
      log "Idle >= ${IDLE_TIMEOUT_MIN}min → exit to free resources."
      exit 0
    fi
  fi

  # Đợi thay đổi control.env/STOP: prefer inotify
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -qq -e modify,create,delete,move "$CONTROL_ENV" "$STOP_FILE" 2>/dev/null || true
  else
    sleep "${POLL_SEC:-5}"
  fi
done
