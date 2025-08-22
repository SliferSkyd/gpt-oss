# Final Project: High-Performance Inference for gpt-oss

## Overview

This project implements a high-performance inference serving system for OpenAI's gpt-oss large language models, specifically targeting the 20B and 120B parameter versions. Built as an extension of llama2.c (a pure C inference engine), this system is adapted to support gpt-oss architectures and optimized for maximum throughput on AMD GPU hardware.

The system provides both interactive (chat) and non-interactive modes, supporting single prompt generation and batch evaluation workflows. The implementation focuses on correctness as the primary objective, followed by achieving the highest possible throughput (tokens/sec) through various optimization and parallelization techniques.

## Key Objectives

- **Primary Goal**: Correctness of inference implementation
- **Secondary Goal**: Maximum throughput optimization (tokens/sec)
- **Modes Supported**:
  - Interactive chat mode
  - Single prompt generation (`generate`)
  - Batch evaluation (`getp`)
- **Model Support**: gpt-oss 20B and 120B parameter models
- **Foundation**: Extended from llama2.c pure C inference engine

## Technical Stack & Constraints

### Language & Environment
- **Language**: C/C++
- **Hardware**: Single-node system with up to 8 AMD MI250 GPUs
- **Compiler**: C/C++ compiler with C++17 support

### Parallelization Stack
- **CPU Parallelization**: OpenMP or pthreads
- **GPU Programming**: HIP for AMD GPU programming
- **Critical Constraint**: All GPU kernels must be written from scratch - no pre-existing GPU libraries allowed

### Codebase Rules
- **Protected Files**: Do not modify `run.cpp`, `getp_csrc/getp_eval.cpp`, or `Makefile`
- **Integration Point**: New functionality should be integrated via `run_exec.cpp`
- **Baseline**: CPU-only implementation provided as foundation

## Setup and Installation

### Prerequisites

1. **C/C++ Compiler** with C++17 support
2. **Make** build system
3. **ROCm Toolkit** (for HIP GPU programming)
4. **Python 3.10** (for tokenizer utilities)

### Environment Setup

```bash
# Set up project paths
export GPT_OSS_REPO_ROOT="/path/to/your/gpt-oss"  # Update to your path
cd $GPT_OSS_REPO_ROOT

# Create and activate Python virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set model paths
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
```

### Model Setup

Pre-compiled model binaries are located in `/mnt/getp/final-project/modelbin/`:

- `gpt-oss-7M.bin` - Debug model for testing
- `gpt-oss-20B.bin` - 20 billion parameter model
- `gpt-oss-120B.bin` - 120 billion parameter model

### Tokenizer Setup

The project uses OpenAI's o200k_harmony tokenizer. Generate the tokenizer binary:

```bash
make tokenizer-bin
```

## Compilation

The project supports multiple build configurations:

```bash
# Basic build (debug, unoptimized)
make run

# Optimized build with -O3
make runfast

# OpenMP parallel build with optimization
make runomp

# Debug build for profiling tools
make rundebug
```

**Recommended for performance**: Use `make runomp` for optimized parallel execution.

### Additional Utilities

```bash
# Build decoder utility for getp output
make decode

# Build tokenizer test binary
make tokenizer-test
```

## Usage

### Running on Slurm

Execute the program within a Slurm environment:

```bash
srun --gres=gpu:<N> ./run <model.bin> [options]
```

Where `<N>` is the number of GPUs to allocate (1-8).

### Command Examples

#### Single Prompt Generation
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "What is the capital of France?"
```

#### Interactive Chat Mode
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat
```

#### Batch Evaluation
```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -f data/input.txt -o data/output.txt
```

#### Decoding Batch Output
For `getp` mode, convert token indices to readable text:

```bash
make decode
./decode -i data/output.txt
```

### OpenMP Configuration

For multi-threaded CPU execution, set the number of threads:

```bash
export OMP_NUM_THREADS=8
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "Hello world"
```

### Tokenizer Testing

Test tokenizer compatibility:

```bash
# C++ tokenizer test
make tokenizer-test
./test_tokenizer -t tokenizer.bin -i "Hello world"
# Expected output: 13225 2375

# Verify compatibility with tiktoken
python3 test_tokenizer.py \
  --bin ./test_tokenizer \
  --tok ./tokenizer.bin \
  --verbose \
  --prompt data/input.txt
```

## Project Structure

```
gpt-oss/
├── run.cpp                    # Main inference engine
├── tokenizer.cpp/hpp          # Tokenizer implementation
├── decode.cpp                 # Output decoder utility
├── getp-csrc/                 # Batch evaluation code 
│   ├── getp_eval.cpp
│   └── getp_run.cpp
├── export_tokenizer_bin.py    # Tokenizer binary generator
├── test_tokenizer.cpp/py      # Tokenizer testing utilities
├── data/                      # Sample input/output files
├── Makefile                   # Build configuration 
└── README.md                  # This file
```

## Performance Optimization Areas

The project allows optimization in the following areas:
- Custom GPU kernel implementation using HIP
- Memory management and data layout optimization
- CPU parallelization using OpenMP/pthreads
- Model sharding across multiple GPUs
- Custom attention and MLP layer implementations

## References

*Note: Include all links from the references slide of your presentation here.*

---

**University Final Project** | **High-Performance GPU Computing** | **C/C++ Implementation**