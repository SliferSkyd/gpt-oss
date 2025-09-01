#!/bin/bash
# profile_with_slurm.sh
# Model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

srun -N 1 --nodelist=MV-DZ-MI250-02 --gres=gpu:1 rocprofv2 \
  --hip-api --hsa-api \
  --hip-trace --hsa-trace \
  --roctx-trace \
  --plugin perfetto \
  -d trace_out -o trace \
  ./run ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp -i data/input.txt -o data/output.txt
