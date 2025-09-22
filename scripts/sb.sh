#!/bin/bash
#SBATCH -J gpt_inference
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --signal=USR1@60       # báo trước 60s khi sắp hết giờ / preempt

# An toàn bash
set -euo pipefail

################################
# 0) Modules & ENV (tùy cluster)
################################
# Ví dụ: giữ hipcc/rocm sẵn cho dự án AMD GPU
module load rocm || true
# module load gcc cmake ninja || true

################################
# 1) Cấu hình khởi tạo (có thể override bởi control.env)
################################
# File điều khiển (đường dẫn tuyệt đối khuyến nghị)
CONTROL_ENV="/nfs/gpu_trainee/getp02/gpt-oss/scripts/control.env"   # <-- bạn đặt file ở đâu tùy ý
STOP_FILE="/nfs/gpu_trainee/getp02/gpt-oss/scripts/STOP-KEEPER"     # tạo file này để dừng job sạch

# Tham số vòng lặp
POLL_SEC=5                        # chu kỳ polling nếu không có inotifywait
IDLE_TIMEOUT_MIN=0                # 0 = tắt idle auto-exit; >0 = tự thoát nếu không chạy gì quá X phút

# Mặc định, **chỉ chạy khi UPDATED=1**. Nếu muốn chạy ngay khi start, set RUN_ON_START=1 trong control.env
RUN_ON_START=0

# Nếu muốn chạy câu lệnh bằng srun (nên), để 1
WRAP_WITH_SRUN=1
SRUN_ARGS="-n 1"                  # có thể thêm --cpus-per-task=… tại đây nếu muốn

# Thư mục log riêng cho mỗi lần chạy
RUN_LOG_DIR="$HOME/runlogs"

# Bạn có thể xuất biến môi trường chung cho toàn bộ phiên bằng ENV_EXPORT trong control.env
# Ví dụ: ENV_EXPORT='export OMP_NUM_THREADS=8; export MYFLAG=1'
ENV_EXPORT=""

################################
# 2) Helpers
################################
log() { echo "[$(date +'%F %T')] $*"; }

# Nạp lại control.env (cho phép override biến ở trên)
load_control_env() {
  if [[ -f "$CONTROL_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; source "$CONTROL_ENV"; set +a
    # Cho phép override các default (nếu người dùng không set thì giữ nguyên)
    POLL_SEC="${POLL_SEC:-5}"
    IDLE_TIMEOUT_MIN="${IDLE_TIMEOUT_MIN:-0}"
    RUN_ON_START="${RUN_ON_START:-0}"
    WRAP_WITH_SRUN="${WRAP_WITH_SRUN:-1}"
    SRUN_ARGS="${SRUN_ARGS:-"-n 1"}"
    RUN_LOG_DIR="${RUN_LOG_DIR:-$RUN_LOG_DIR}"
    ENV_EXPORT="${ENV_EXPORT:-$ENV_EXPORT}"
  fi
}

# Chạy một lần theo WORK_DIR + RUN_CMD hiện tại
run_once() {
  mkdir -p "$RUN_LOG_DIR"
  local ts; ts="$(date +'%Y%m%d_%H%M%S')"
  local rlog="$RUN_LOG_DIR/run_${SLURM_JOB_ID}_${ts}.log"

  if [[ -z "${WORK_DIR:-}" || -z "${RUN_CMD:-}" ]]; then
    log "!! WORK_DIR hoặc RUN_CMD chưa được cấu hình trong control.env"
    return 2
  fi
  if [[ ! -d "$WORK_DIR" ]]; then
    log "!! WORK_DIR không tồn tại: $WORK_DIR"
    return 3
  fi

  log ">> WORK_DIR=$WORK_DIR"
  log ">> RUN_CMD=$RUN_CMD"
  log ">> Log file: $rlog"

  # Thực thi
  (
    set +e
    cd "$WORK_DIR"
    # Xuất ENV_EXPORT nếu có
    if [[ -n "${ENV_EXPORT:-}" ]]; then
      eval "$ENV_EXPORT"
    fi
    # Chạy lệnh
    if [[ "${WRAP_WITH_SRUN:-1}" -eq 1 ]]; then
      # Dùng một shell login để hiểu && ; || ; biến; v.v.
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
    log ">> RUN OK (rc=0). Output đã lưu: $rlog"
  else
    log "!! RUN FAIL (rc=$rc). Xem log: $rlog"
  fi
  return $rc
}

# Bắt tín hiệu để thoát gọn
trap 'log "Got USR1 (preempt/timeout soon) -> exit."; exit 0' USR1
trap 'log "Got TERM/INT -> exit."; exit 0' TERM INT

################################
# 3) Main loop
################################
log "===== RESOURCE KEEPER START ====="
log "CONTROL_ENV=$CONTROL_ENV"
mkdir -p "$(dirname "$CONTROL_ENV")" logs "$RUN_LOG_DIR"

# Nạp cấu hình lần đầu
load_control_env

# Cờ trạng thái
LAST_RUN_TS="$(date +%s)"
HAS_RUN_THIS_CYCLE=1   # để chỉ chạy 1 lần mỗi khi UPDATED=1

# Nếu muốn chạy ngay khi khởi động
if [[ "${RUN_ON_START:-0}" -eq 1 ]]; then
  HAS_RUN_THIS_CYCLE=0
fi

while true; do
  # Cho phép dừng thủ công
  [[ -f "$STOP_FILE" ]] && { log "STOP file found: $STOP_FILE. Bye."; exit 0; }

  # Luôn nạp lại cấu hình (có thể đổi WORK_DIR/RUN_CMD…)
  load_control_env

  # Nếu UPDATED=1 -> mở khóa cho phép chạy 1 lần trong cycle này
  if [[ "${UPDATED:-0}" == "1" ]]; then
    HAS_RUN_THIS_CYCLE=0
  fi

  # Khi được phép chạy (UPDATED=1 hoặc RUN_ON_START bật & chưa chạy)
  if [[ "$HAS_RUN_THIS_CYCLE" -eq 0 ]]; then
    if run_once; then
      LAST_RUN_TS="$(date +%s)"
      HAS_RUN_THIS_CYCLE=1
      # reset UPDATED=0 trong control.env nếu có dòng này
      if [[ -f "$CONTROL_ENV" ]]; then
        grep -q '^UPDATED=' "$CONTROL_ENV" && sed -i 's/^UPDATED=.*/UPDATED=0/' "$CONTROL_ENV" || true
      fi
    else
      # Nếu chạy lỗi, vẫn đóng cycle (để tránh loop liên tục). Lần chạy kế tiếp cần UPDATED=1 hoặc FORCE_RUN=1
      LAST_RUN_TS="$(date +%s)"
      HAS_RUN_THIS_CYCLE=1
      # Bạn có thể bật FORCE_RUN để chạy lại ngay mà không chờ sửa:
      # sed -i 's/^UPDATED=.*/UPDATED=1/' control.env
    fi
  fi

  # Idle timeout (nếu cấu hình >0)
  if [[ "${IDLE_TIMEOUT_MIN:-0}" -gt 0 ]]; then
    now="$(date +%s)"
    elap=$(( (now - LAST_RUN_TS)/60 ))
    if (( elap >= IDLE_TIMEOUT_MIN )); then
      log "Idle >= ${IDLE_TIMEOUT_MIN}min → exit to free resources."
      exit 0
    fi
  fi

  # Đợi thay đổi: ưu tiên inotifywait; nếu không có thì sleep
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -qq -e modify,create,delete,move "$CONTROL_ENV" "$STOP_FILE" 2>/dev/null || true
  else
    sleep "${POLL_SEC:-5}"
  fi
done
