#!/bin/bash
#SBATCH -J gpt_inference
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --signal=USR1@60       # cảnh báo 60s trước khi hết giờ / preempt

set -euo pipefail

################################
# 0) Modules & ENV (tùy cluster)
################################
module load rocm || true       # giữ hipcc/ROCm cho AMD GPU nếu cần
# module load gcc cmake ninja || true

################################
# 1) Defaults (override bởi control.env)
################################
CONTROL_ENV="/nfs/gpu_trainee/getp03/tmp1/gpt-oss/scripts/control.env"     # <-- path tuyệt đối tới control.env
STOP_JOB_FILE="$HOME/STOP-KEEPER"   # touch file này để KẾT THÚC job (trả tài nguyên)

POLL_SEC=3                          # chu kỳ polling nếu không có inotifywait
IDLE_TIMEOUT_MIN=0                  # 0 = không auto-exit khi idle
RUN_ON_START=0                      # 1 = tự chạy ngay khi job start
WRAP_WITH_SRUN=1                    # 1 = chạy RUN_CMD dưới srun
SRUN_ARGS="-n 1"                    # thêm --cpus-per-task=... nếu muốn
RUN_LOG_DIR="$HOME/runlogs"         # nơi lưu log mỗi lần chạy
ENV_EXPORT=""                       # export biến trước khi chạy (vd: 'export AMD_ARCH=gfx90a')

# Hành vi khi UPDATED=1: "restart" (mặc định) hoặc "start"
ON_UPDATED="restart"

################################
# 2) Helpers
################################
log() { echo "[$(date +'%F %T')] $*"; }

# nạp lại control.env (cho phép override các default trên)
load_control_env() {
  if [[ -f "$CONTROL_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$CONTROL_ENV"; set +a
    POLL_SEC="${POLL_SEC:-3}"
    IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-0}"
    RUN_ON_START="${RUN_ON_START:-0}"
    WRAP_WITH_SRUN="${WRAP_WITH_SRUN:-1}"
    SRUN_ARGS="${SRUN_ARGS:-"-n 1"}"
    RUN_LOG_DIR="${RUN_LOG_DIR:-$RUN_LOG_DIR}"
    ENV_EXPORT="${ENV_EXPORT:-$ENV_EXPORT}"
    ON_UPDATED="${ON_UPDATED:-restart}"
  fi
}

# reset một flag (nếu có) về 0
reset_flag() {
  local key="$1"
  [[ -f "$CONTROL_ENV" ]] || return 0
  if grep -q "^${key}=" "$CONTROL_ENV"; then
    sed -i "s/^${key}=.*/${key}=0/" "$CONTROL_ENV"
  else
    printf "\n%s=0\n" "$key" >> "$CONTROL_ENV"
  fi
}

# reset nhiều flag
reset_flags() {
  for k in "$@"; do reset_flag "$k"; done
}

# file lưu PID của tiến trình srun/bash background
pid_file() { echo "$RUN_LOG_DIR/run_${SLURM_JOB_ID}.pid"; }
# symlink tiện xem log gần nhất
last_log_link() { echo "$RUN_LOG_DIR/last_run_${SLURM_JOB_ID}.log"; }

is_running() {
  local pf; pf="$(pid_file)"
  if [[ -f "$pf" ]]; then
    local pid; pid="$(cat "$pf" 2>/dev/null || true)"
    [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
    return $?
  fi
  return 1
}

start_run() {
  if is_running; then
    log ">> RUN is already running. Skip start."
    return 0
  fi
  mkdir -p "$RUN_LOG_DIR"
  local ts rlog pf
  ts="$(date +'%Y%m%d_%H%M%S')"
  rlog="$RUN_LOG_DIR/run_${SLURM_JOB_ID}_${ts}.log"
  pf="$(pid_file)"

  if [[ -z "${WORK_DIR:-}" || -z "${RUN_CMD:-}" ]]; then
    log "!! WORK_DIR hoặc RUN_CMD chưa cấu hình trong control.env"
    echo "[ERROR] WORK_DIR/RUN_CMD missing" > "$rlog"
    return 2
  fi
  if [[ ! -d "$WORK_DIR" ]]; then
    log "!! WORK_DIR không tồn tại: $WORK_DIR"
    echo "[ERROR] WORK_DIR not found: $WORK_DIR" > "$rlog"
    return 3
  fi

  log ">> START RUN"
  log "   WORK_DIR=$WORK_DIR"
  log "   RUN_CMD=$RUN_CMD"
  log "   log: $rlog"

  (
    set +e
    cd "$WORK_DIR"
    if [[ -n "${ENV_EXPORT:-}" ]]; then
      eval "$ENV_EXPORT"
    fi
    if [[ "${WRAP_WITH_SRUN:-1}" -eq 1 ]]; then
      # chạy nền dưới srun để giữ tài nguyên đúng step
      srun $SRUN_ARGS bash -lc "$RUN_CMD" &> "$rlog" &
    else
      bash -lc "$RUN_CMD" &> "$rlog" &
    fi
    echo $! > "$pf"
    ln -sf "$(basename "$rlog")" "$(last_log_link)" 2>/dev/null || true
    wait $!   # CHÚ Ý: đừng để bg ở đây; bắt PID xong thì retourne ngay
  ) &
  # tiến trình giám sát ở nền; bản thân sbatch vòng lặp vẫn rảnh
  return 0
}

stop_run() {
  local pf; pf="$(pid_file)"
  if ! is_running; then
    log ">> No running process. Nothing to stop."
    [[ -f "$pf" ]] && rm -f "$pf"
    return 0
  fi
  local pid; pid="$(cat "$pf" 2>/dev/null || true)"
  [[ -z "${pid:-}" ]] && { rm -f "$pf"; return 0; }

  log ">> STOP RUN (pid=$pid)"
  # gửi tín hiệu mềm -> cứng -> kill -9 nếu cần
  kill -SIGINT "$pid" 2>/dev/null || true
  for t in 1 2 3 4 5; do
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -SIGTERM "$pid" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pf"
  log ">> STOP done."
  return 0
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

LAST_ACTION_TS="$(date +%s)"

# Tùy chọn chạy ngay khi job start
if [[ "${RUN_ON_START:-0}" -eq 1 ]]; then
  start_run || true
  LAST_ACTION_TS="$(date +%s)"
fi

while true; do
  [[ -f "$STOP_JOB_FILE" ]] && { log "STOP file found → exit job."; exit 0; }

  load_control_env

  # 1) Ưu tiên STOP
  if [[ "${STOP:-0}" == "1" ]]; then
    stop_run || true
    reset_flag STOP
    LAST_ACTION_TS="$(date +%s)"
  fi

  # 2) RESTART nếu yêu cầu
  if [[ "${RESTART:-0}" == "1" ]]; then
    stop_run || true
    start_run || true
    reset_flag RESTART
    LAST_ACTION_TS="$(date +%s)"
  fi

  # 3) UPDATED → theo ON_UPDATED: restart (mặc định) hoặc start
  if [[ "${UPDATED:-0}" == "1" ]]; then
    if [[ "${ON_UPDATED:-restart}" == "restart" ]]; then
      stop_run || true
      start_run || true
    else
      start_run || true
    fi
    reset_flag UPDATED
    LAST_ACTION_TS="$(date +%s)"
  fi

  # 4) START nếu yêu cầu và chưa chạy
  if [[ "${START:-0}" == "1" ]]; then
    start_run || true
    reset_flag START
    LAST_ACTION_TS="$(date +%s)"
  fi

  # 5) Auto-restart khi process tự thoát (tuỳ chọn)
  if [[ "${AUTORESTART_ON_EXIT:-0}" == "1" ]]; then
    if ! is_running; then
      # nếu đã từng start mà bây giờ không còn → restart
      # (tránh spam: chỉ restart nếu không có STOP/RESTART/UPDATED/START đang chờ)
      start_run || true
      LAST_ACTION_TS="$(date +%s)"
    fi
  fi

  # 6) Idle auto-exit (nếu bật)
  if [[ "${IDLE_TIMEOUT_MIN:-0}" -gt 0 ]]; then
    now="$(date +%s)"
    elap=$(( (now - LAST_ACTION_TS)/60 ))
    if (( elap >= IDLE_TIMEOUT_MIN )); then
      log "Idle >= ${IDLE_TIMEOUT_MIN}min → exit to free resources."
      exit 0
    fi
  fi

  # 7) Đợi thay đổi control.env / STOP_JOB_FILE
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -qq -e modify,create,delete,move "$CONTROL_ENV" "$STOP_JOB_FILE" 2>/dev/null || true
  else
    sleep "${POLL_SEC:-3}"
  fi
done
Collapse



