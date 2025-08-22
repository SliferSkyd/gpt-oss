export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

make runomp

srun -N 1 --gres=gpu:1 ./scripts/run.sh run --checkpoint ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp -n 20 -i data/input.txt -o data/output.txt 

pip install numpy pandas matplotlib seaborn

python3 scripts/analyze.py --file times.csv

