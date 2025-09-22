export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
srun -N 1 --gres=gpu:8 python evaluation.py -m 20b