#!/bin/bash
# profile_with_slurm.sh
# Model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

# Thiết lập output directory
OUTPUT_DIR="profiling_gpt_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== GPT-OSS Profiling with Slurm ==="
echo "Output directory: $OUTPUT_DIR"

# Chạy với slurm và rocprof
srun -N 1 --gres=gpu:1 \
rocprof --hip-trace --hsa-trace --sys-trace --stats \
        --timestamp on --basenames off --obj-tracking on \
        --roctx-trace --roctx-rename \
        -o $OUTPUT_DIR/gpt_profile.csv \
        ./run "${MODELBIN_ROOT}/gpt-oss-7m.bin" -m getp -n 20 -i data/input.txt -o data/output.txt

echo "Profiling completed!"
echo "Files generated in: $OUTPUT_DIR/"
ls -la $OUTPUT_DIR/
echo