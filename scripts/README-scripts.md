# Submit job
sbatch run_with_monitor.sh

# Theo dõi job
squeue -u $USER

# Xem logs realtime
tail -f logs/inference_JOBID.out    # Output chính
tail -f logs/gpu_monitor_JOBID.log  # GPU monitoring
