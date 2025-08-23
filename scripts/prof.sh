#!/bin/bash
# profile_with_slurm.sh
# Model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

# Thiết lập output directory
rocprofv2 --hip-trace --hsa-trace --roctx-trace -o trace.json srun ./run
rocprofv2 \
  --hip-api --hsa-api \
  --hip-trace --hsa-trace \
  --roctx-trace \
  -o trace \
  srun ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp -i data/input.txt -o data/output.txt