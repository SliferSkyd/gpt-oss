export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"

make runomp

sbatch -N 1 --gres=gpu:1 ./scripts/run_with_monitor.sh

srun -N 1 --gres=gpu:1 ./scripts/run.sh run --checkpoint ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp -n 20 -i data/input.txt -o data/output.txt 

pip install numpy pandas matplotlib seaborn

python3 scripts/analyze.py --file logs/times_job_JOBID.csv

tail -f logs/inference_JOBID.out    # Output chính
tail -f logs/gpu_monitor_JOBID.log  # GPU monitoring