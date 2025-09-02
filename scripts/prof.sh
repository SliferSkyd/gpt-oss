#!/bin/bash
# profile_with_slurm.sh
# Model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

srun --gres=gpu:1 rocprof --hip-trace   ./run "${MODELBIN_ROOT}/gpt-oss-20b.bin"   -n 50 -m getp -i data/input.txt -o data/output.txt