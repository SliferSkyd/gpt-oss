#!/bin/bash

echo "=== Building GPU-accelerated GPT-OSS ==="

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please ensure ROCm is installed."
    exit 1
fi

# Compile GPU utilities
echo "Compiling GPU utilities..."
hipcc -c getp-csrc/gpu_utils.cpp -o gpu_utils.o --offload-arch=gfx90a -I. --std=c++17

# Compile GPU kernels
echo "Compiling GPU kernels..."
hipcc -c getp-csrc/gpu_kernels.cpp -o gpu_kernels.o --offload-arch=gfx90a -I. --std=c++17

# Link everything together
echo "Linking..."
hipcc -o run_gpu run.cpp tokenizer.cpp gpu_utils.o gpu_kernels.o \
    --offload-arch=gfx90a --std=c++17 -O3 -fopenmp \
    -DUSE_GPU -include getp-csrc/getp_run_gpu.cpp

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "To run tests:"
    echo "  ./run_gpu <model_file> -i <input_file>"
    echo ""
    echo "The first batch will automatically run a GPU vs CPU comparison test."
else
    echo "Build failed!"
    exit 1
fi