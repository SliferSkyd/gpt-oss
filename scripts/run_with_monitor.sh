#!/bin/bash
#SBATCH --job-name=gpt_inference_monitor
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Tạo thư mục logs nếu chưa có
mkdir -p logs

# Lấy hostname của node được assign
NODE=$(hostname)
echo "Running on node: $NODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting at: $(date)"

# Function để monitor GPU
monitor_gpu() {
    local log_file="logs/gpu_monitor_${SLURM_JOB_ID}.log"
    echo "Starting GPU monitoring - logging to $log_file"
    
    while true; do
        {
            echo "================================================================"
            echo "           GPU Memory Usage on $NODE"
            echo "                $(date)"
            echo "================================================================"
            printf "%-8s %-12s %-12s %-12s %-8s\n" "GPU ID" "Used (GB)" "Total (GB)" "Free (GB)" "Usage%"
            echo "--------+------------+------------+------------+--------"
            
            rocm-smi --showmeminfo vram | awk '
            /GPU\[[0-9]+\].*VRAM Total Memory \(B\):/ {
                match($0, /GPU\[([0-9]+)\]/, gpu_arr)
                gpu_id = gpu_arr[1]
                match($0, /([0-9]+)$/, total_arr)
                total_bytes = total_arr[1]
                
                getline
                match($0, /([0-9]+)$/, used_arr)
                used_bytes = used_arr[1]
                
                used_gb = used_bytes / (1024*1024*1024)
                total_gb = total_bytes / (1024*1024*1024)
                free_gb = total_gb - used_gb
                usage_pct = (used_bytes / total_bytes) * 100
                
                printf "%-8s %-12.2f %-12.2f %-12.2f %-7.1f%%\n", 
                       "GPU " gpu_id, used_gb, total_gb, free_gb, usage_pct
            }'
            echo "================================================================"
            echo
        } >> "$log_file"
        
        sleep 5
    done
}

# Function để chạy inference
run_inference() {
    echo "Starting inference task..."
    ./scripts/run.sh run \
        --checkpoint ${MODELBIN_ROOT}/gpt-oss-20b.bin \
        -m getp \
        -i data/input.txt \
        -o data/output.txt -n 100
        # -n 64 \
    
    local exit_code=$?
    echo "Inference completed with exit code: $exit_code"
    return $exit_code
}

# Bắt đầu GPU monitoring ở background
monitor_gpu &
MONITOR_PID=$!

echo "GPU monitoring started with PID: $MONITOR_PID"

# Chạy inference task
run_inference
INFERENCE_EXIT_CODE=$?

# Dừng monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo "Job completed at: $(date)"
echo "Final GPU status:"
rocm-smi

# Exit với code của inference task
exit $INFERENCE_EXIT_CODE