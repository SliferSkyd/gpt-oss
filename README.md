# gpt-oss High-Performance Inference

> Multi-GPU HIP inference stack for OpenAI’s GPT-OSS 20B/120B models. This project ranked **1st of 14 teams** in the Moreh GPU Training Program final by delivering the highest measured throughput on both 20B and 120B model.

## Overview

This project implements a high-performance inference serving system for OpenAI's gpt-oss large language models, specifically targeting the 20B and 120B parameter versions. Built as an extension of llama2.c (a pure C inference engine), this system is adapted to support gpt-oss architectures and optimized for maximum throughput on AMD GPU hardware.


## Highlights
- **Throughput-first design:** Scale to 8×MI250, achieving ~47K and 20.5K tokens/s for both 20B and 120B deployments as documented in `docs/final-project-presentation.pdf`.
- **Multi-level parallelism:** Hybrid data + tensor parallel execution, with custom ring collectives that shard MoE weights across GPUs while keeping attention local.
- **Memory usage optimization:** MXFP4 quantization for MLP weights, bf16 run state, and an int8 KV cache (first tokens kept in bf16) balance memory scale with accuracy, able to infer up to 12K tokens at the same time.
- **Hand-written HIP kernels:** Flash decoding attention, fused SwiGLU, GPU sampler, and MFMA matmul kernels tuned for MI250 bandwidth and compute.
- **Quality preserved:** Final evaluation matched reference quality (BERTScore 0.96 for both models; METEOR 0.385 / 0.304 for 20B / 120B, passing the predefined threshold) while dramatically improving throughput.

## Model & Hardware Targets
| Model | GPUs | Parallelism Topology | Batch Size | Inputs / Run | Observed Runtime |
|-------|------|----------------------|------------|--------------|-------------------|
| GPT-OSS 20B | 8×AMD MI250 | 2× tensor × 4× data | 1,536 | 12,288 | 5–6 minutes |
| GPT-OSS 120B | 8×AMD MI250 | 4× tensor × 2× data | 1,856 | 14,848 | ~20 minutes |

## Architecture & Techniques

### Multi-Level Parallelism
- **Data Parallel (DP):** Prompts distributed across GPUs with final aggregation to keep communication minimal.
- **Tensor Parallel (TP) for MoE only:** MLP weights are partitioned across TP ranks; each rank processes a shard, performs all-gather after RMSNorm, and reduce-scatter after the fused MLPs. Attention stays local to avoid oversized communication.
- **Ring-based collectives:** Custom HIP implementations (`getp-csrc/tp_ring.hpp`) all-gather and reduce-scatter bf16 payloads while balancing link utilization.

### Batching & Scheduler
- **Continuous batching:** Maintains high utilization as requests finish early; padded tokens are avoided during matmul scheduling.
- **Adaptive kernel choices:** Switches between 16×16×16 and 32×32×8 MFMA implementations based on per-expert token counts to maximize arithmetic intensity.

### Memory Efficiency
- **MXFP4 MoE weights:** Packed uint8 weights with per-block scales staged to bf16 in ping-pong buffers to hide dequant overhead.
- **Int8 KV cache with bf16 “recent” window:** The first tokens of each sequence stay in bf16 (configurable via `KV_BF16_KEEP_TOKENS`), later tokens are compressed to int8 with per-token scales.
- **bf16 activations & communication:** Run state, router outputs, and TP collectives are bf16 to cut bandwidth and memory footprints.

### Kernel & Runtime Optimizations
- **Flash decoding attention:** Sequence-split softmax with online reduction keeps shared-memory pressure low.
- **Vectorized loads/stores:** Q/K/V transactions aligned to 16 bytes; bypass shared memory when caches can serve data.
- **Fused kernels:** Matmul+bias, SwiGLU, and GPU-side sampling reduce launch overhead.
- **OpenMP on CPU fallbacks:** Retains optimized CPU path (extended from `llama2.c`) for debugging and profiling.

### Quality Controls
- **Accuracy metrics:** Pipeline validated with METEOR and BERTScore (see performance slide).
- **Tokenizer parity:** Custom tokenizer binary validated against OpenAI’s o200k_harmony via `test_tokenizer.py`.

## Performance & Quality
| Benchmark | Configuration | #inputs | Throughput | METEOR / BERTscore |
|-----------|---------------|--------|--------|-------------|
| 20B | 8×MI250, 2×TP × 4×DP | 12,288 | 46793 tok/s | 0.385 / 0.96 |
| 120B | 8×MI250, 4×TP × 2×DP | 14,848 | 20549 tok/s | 0.304 / 0.96 |

## Getting Started

### Prerequisites
- C/C++ compiler with C++17 support
- ROCm toolchain (HIP, `hipcc`) for AMD MI250
- GNU Make
- Python 3.10 for tokenizer utilities

### Environment Setup
```bash
export GPT_OSS_REPO_ROOT="/path/to/gpt-oss"   # update to your local path
cd "$GPT_OSS_REPO_ROOT"

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Model assets mounted in the training cluster
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
```

### Model & Tokenizer Assets
- `gpt-oss-7M.bin` – functional smoke tests
- `gpt-oss-20B.bin`, `gpt-oss-120B.bin` – production checkpoints under `${MODELBIN_ROOT}`
- Generate tokenizer binary (`tokenizer.bin`) from OpenAI’s o200k_harmony vocab:
  ```bash
  make tokenizer-bin
  ```

## Build Targets
| Command | Purpose |
|---------|---------|
| `make run` | Debug-friendly CPU build |
| `make runfast` | Optimized CPU build (`-O3`) |
| `make runomp` | OpenMP-enabled build (recommended baseline) |
| `make rundebug` | Instrumented build for profilers |
| `make decode` | Build the batch-output decoder |
| `make tokenizer-test` | Build the tokenizer validation binary |

## Running Inference

### Standalone (local GPU)
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" \
  -m generate \
  -i "What is the capital of France?"
```

### Chat Loop
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat
```

### Batch Evaluation (getp)
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" \
  -m getp \
  -f data/input.txt \
  -o data/output.txt
```

### Decode Batch Output
```bash
make decode
./decode -i data/output.txt
```

### Tokenizer Validation
```bash
make tokenizer-test
./test_tokenizer -t tokenizer.bin -i "Hello world"
python3 test_tokenizer.py --bin ./test_tokenizer --tok tokenizer.bin --prompt data/input.txt
```

### SLURM Example
```bash
srun --gres=gpu:<N> ./run <model.bin> [options]
```

## Profiling & Monitoring
- Submit combined inference + GPU monitor jobs:  
  `sbatch -N 1 --gres=gpu:1 ./scripts/run_with_monitor.sh`
- Live logs: `tail -f logs/inference_JOBID.out` and `tail -f logs/gpu_monitor_JOBID.log`
- Parse timing CSVs: `python3 scripts/analyze.py --file logs/times_job_JOBID.csv`

## Repository Layout
```
gpt-oss/
├── run.cpp                  # CPU reference / debug path (extended llama2.c)
├── getp-csrc/               # HIP implementation for high-throughput inference
│   ├── kernels/             # Attention, MoE, RMSNorm, matmul, swiglu, etc.
│   ├── memory/              # MXFP4 packing & staging utilities
│   └── tp_ring.hpp          # Custom tensor-parallel ring collectives
├── tokenizer.cpp/.hpp       # Tokenizer implementation + CLI test harness
├── decode.cpp               # Batch output decoder
├── scripts/                 # Launch helpers, monitoring, analysis
├── data/                    # Sample prompts / outputs
├── docs/                    # Detailed design notes & presentation slides
└── README.md
```

## Documentation
- `docs/final-project-presentation.pdf` – final deck (architecture, metrics, throughput scoreboard)
- `docs/README-architecture.md` – model internals & configuration
- `docs/README-inference.md` – CPU forward-pass walkthrough
- `docs/README-getp_run.md` – GPU pipeline deep dive
- `docs/moe_memory_diagram.md` – MoE routing & buffer diagrams
- `docs/performance_tracking.md` – throughput milestones

---

Built on top of `llama2.c`, this project delivers a from-scratch HIP inference stack that reaches production-level throughput while staying within the competition constraints (no external GPU libraries) and maintaining model quality.
