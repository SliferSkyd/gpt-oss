# uv venv --python=3.12
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
# pip install --upgrade evaluate bert-score nltk pandas tiktoken jsonschema


srun -N 1 --gres=gpu:4 python scripts/eval.py -p data/input.txt -s data/output.txt -r data/ground_truth.txt --bs-batch 4