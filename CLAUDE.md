# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance inference engine for GPT-OSS models (20B & 120B parameters) optimized for AMD MI250 GPUs. The goal is to maximize throughput (tokens/second) while maintaining numerical correctness.

## Core Constraints

- **Read-Only Files:** Do NOT modify `run.cpp`, `getp-csrc/getp_eval.cpp`, or root `Makefile`
- **Custom Kernels Only:** All GPU kernels must be written from scratch in HIP (no rocBLAS, rocFFT, or MIOpen)
- **Working Directory:** All modifications confined to `getp-csrc/` and its subdirectories
- **Primary Metric:** Throughput (tokens/sec) - justify all optimizations by this metric

## Build Commands

```bash
# Environment setup
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
export OMP_NUM_THREADS=8

# Primary build (recommended for performance)
make runomp  # OpenMP parallel build with -O3 optimization

# Alternative builds
make run      # Basic debug build
make runfast  # Optimized build without OpenMP
make rundebug # Debug build for profiling
```

## Execution & Testing

```bash
# Run inference (batch evaluation mode)
./scripts/run.sh run --checkpoint ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp

# Performance profiling
python3 scripts/analyze.py --file times.csv

# GPU profiling with ROCm
rocprofv2 --hip-trace --hsa-trace --roctx-trace -o trace.json srun ./run

# Evaluation against reference
python3 eval/eval.py
```

## Technical Specifications

### Hardware Environment
- **Target:** 8x AMD MI250 GPUs (64GB HBM2e each)
- **Compiler:** hipcc with `--offload-arch=gfx90a`
- **C++ Standard:** C++17

### Model Architecture
- **Type:** Decoder-only Transformer with pre-layer RMSNorm
- **Attention:** Grouped-Query Attention (GQA) with alternating full/sliding-window (128 tokens)
- **FFN:** SwiGLU Mixture-of-Experts (MoE), top-4 selection
- **Dimensions:** 2,880 embedding, 64 heads (45 dim), 200K vocab, 131K context
- **Quantization:** MoE weights MXFP4 (stored as FP32)

### Model Variants
| Model | Layers | Experts | Active Params | Total Params |
|-------|--------|---------|---------------|---------------|
| 20B   | 24     | 32      | ~3.6B         | ~20B          |
| 120B  | 36     | 128     | ~5.1B         | ~120B         |

## Codebase Structure

```
getp-csrc/                # Main working directory
├── getp_run.cpp         # Main GPU-accelerated engine
├── getp_eval.cpp        # Evaluation harness (READ-ONLY)
├── config.hpp           # GPU kernel configuration
├── utils.hpp            # Utility functions
├── batch_manager.hpp    # Continuous batching
├── kernels/             # Custom HIP kernels
│   ├── attention.hpp    # Attention
│   ├── matmul.hpp       # Matrix multiplication
│   ├── moe.hpp          # Mixture-of-Experts
│   ├── rmsnorm.hpp      # RMS normalization
│   ├── softmax.hpp      # Softmax
│   ├── rope.hpp         # Rotary positional embedding
│   └── swiglu.hpp       # SwiGLU activation
└── memory/
    └── mxfp4.hpp        # MXFP4 quantization
```

## Implemented Features

1. **Continuous Batching** - Dynamic batch processing
2. **Paged Attention** - Memory-efficient attention mechanism
3. **Mixed Precision Inference** - bfloat16 optimizations
4. **Custom HIP Kernels** - All operations implemented from scratch

## Development Workflow

1. **Build:** `make runomp`
2. **Run:** `./scripts/run.sh run --checkpoint ${MODELBIN_ROOT}/gpt-oss-20b.bin -m getp`
3. **Profile:** `python3 scripts/analyze.py --file times.csv`
4. **Analyze:** Review kernel performance in `times.csv`
5. **Optimize:** Focus on bottleneck kernels in `getp-csrc/kernels/`

## Performance Analysis Tools

- `scripts/analyze.py` - Kernel performance breakdown
- `scripts/prof.sh` - ROCm profiling wrapper
- `scripts/run_with_monitor.sh` - GPU utilization monitoring
- `eval/eval.py` - Correctness validation against references

## Key Development Notes

- All GPU operations must maintain numerical correctness
- Optimize for throughput (tokens/sec) as primary metric
- Test changes against reference implementation in `run.cpp`
- Use `rocprof` for detailed GPU performance analysis
- Monitor GPU memory usage and utilization during runs