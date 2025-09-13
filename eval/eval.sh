export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
srun -N 1 --gres=gpu:2 python evaluation.py -m 120b