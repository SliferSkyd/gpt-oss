# GPU Implementation Guide for GPT-OSS 20B

## Overview

This document provides a comprehensive guide for the GPU-accelerated implementation of the GPT-OSS 20B model, optimized for 8 AMD GPUs with BF16 precision.

## Architecture Design

### 1. Memory Layout Strategy

#### BF16/FP32 Hybrid Approach
- **Model Weights**: Stored in BF16 on GPU (50% memory reduction)
- **Activations**: 
  - Critical paths (x, residual): FP32 for accuracy
  - Intermediate buffers (MLP outputs): BF16 for memory efficiency
- **KV Cache**: BF16 to maximize sequence length capacity

#### Memory Distribution (per GPU - 64GB VRAM)
```
Model Weights (BF16):     ~20GB
- Embeddings:             3.6GB (replicated)
- Per-layer weights:      2.5GB × 3 layers = 7.5GB
- Expert weights:         2.0GB × 4 experts = 8GB
- Output projection:      0.9GB

Run State (Mixed):        ~12GB
- Activations (FP32):     4GB
- KV Cache (BF16):        6GB
- Intermediates:          2GB

Available Buffer:         ~32GB (for larger batches)
```

### 2. GPU Distribution Scheme

#### Layer-Expert Hybrid Parallelism
```
GPU Assignment:
- GPU 0-7: Each handles 3 layers (24 layers / 8 GPUs)
- Expert Distribution: 4 experts per GPU (32 experts / 8 GPUs)

Layer 0:  GPU 0 (experts 0-3,   4-7,   8-11,  12-15, 16-19, 20-23, 24-27, 28-31)
Layer 1:  GPU 1 (experts 0-3,   4-7,   8-11,  12-15, 16-19, 20-23, 24-27, 28-31)
...
Layer 23: GPU 7 (experts 0-3,   4-7,   8-11,  12-15, 16-19, 20-23, 24-27, 28-31)
```

#### Communication Pattern
- **Attention**: Layer-local computation (no communication)
- **MoE**: All-to-all communication for expert dispatch
- **Pipeline**: Sequential layer processing with overlap

### 3. Implementation Phases

## Phase 1: GPU Memory Management ✅

### Files Created:
1. **gpu_memory.hpp/cpp**: Memory management layer
   - BF16/FP32 conversion utilities
   - Multi-GPU allocation tracking
   - Efficient memory transfers

### Key Features:
- Automatic BF16 conversion during weight loading
- Device-aware memory allocation
- Memory pool management for efficiency

## Phase 2: Kernel Structure ✅

### Files Created:
1. **gpu_kernels.hpp/cpp**: Complete kernel implementations
   - Embedding lookup
   - RMSNorm
   - Attention components (QKV, RoPE, MHA)
   - MoE router and expert dispatch
   - Expert MLP with SwiGLU
   - Output projection

### Optimization Techniques:
- Shared memory utilization for reductions
- Warp-level primitives for efficiency
- Fused operations where possible

## Phase 3: Integration ✅

### Files Created:
1. **getp_run_gpu.cpp**: GPU-accelerated inference
   - Seamless integration with existing interface
   - Automatic GPU detection and fallback
   - Batch processing optimization

### Features:
- Drop-in replacement for CPU implementation
- Maintains FP32 interface compatibility
- Efficient batch processing

## Phase 4: Testing Framework ✅

### Files Created:
1. **test_gpu_kernels.cpp**: Comprehensive testing
   - Kernel validation against CPU reference
   - Performance benchmarking
   - Memory leak detection

### Test Coverage:
- Individual kernel correctness
- End-to-end inference validation
- Performance regression testing

## Usage Instructions

### 1. Compilation

Add to your Makefile:
```makefile
# GPU compilation flags
GPU_FLAGS = -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include
GPU_LIBS = -L/opt/rocm/lib -lhipblas -lrocblas -lamdhip64

# Build GPU version
getp_gpu: getp-csrc/getp_run_gpu.cpp getp-csrc/gpu_memory.cpp getp-csrc/gpu_kernels.cpp
	hipcc $(GPU_FLAGS) -O3 -o $@ $^ $(GPU_LIBS)

# Build tests
test_gpu: test_gpu_kernels.cpp getp-csrc/gpu_memory.cpp getp-csrc/gpu_kernels.cpp
	hipcc $(GPU_FLAGS) -O3 -o $@ $^ $(GPU_LIBS)
```

### 2. Running the Model

The GPU implementation automatically activates when GPUs are detected:
```bash
./run model.bin tokenizer.bin "Your prompt here"
```

### 3. Testing

Run kernel tests:
```bash
./test_gpu
```

Run with custom batch size:
```bash
./test_gpu 4  # Batch size 4
```

## Performance Optimization Guide

### 1. Kernel Optimizations

#### Priority Order:
1. **GEMM Operations** (70% of compute)
   - Use hipBLAS/rocBLAS for optimal performance
   - Implement tensor cores utilization

2. **Attention Mechanism** (15% of compute)
   - Flash Attention algorithm
   - Sliding window optimization

3. **MoE Routing** (10% of compute)
   - Efficient top-k selection
   - Batched expert dispatch

### 2. Memory Optimizations

#### Strategies:
- **Memory Pooling**: Pre-allocate buffers
- **Stream Overlap**: Hide memory transfers
- **Fusion**: Combine kernels to reduce memory traffic

### 3. Multi-GPU Optimizations

#### Communication Reduction:
- **Expert Locality**: Keep expert computation local
- **Pipeline Parallelism**: Overlap computation and communication
- **Gradient Checkpointing**: For training (if needed)

## Benchmarking Results (Expected)

### Single Token Generation
```
Metric          CPU (FP32)    GPU (BF16)    Speedup
------------------------------------------------------
Latency         250ms         15ms          16.7x
Throughput      4 tok/s       66 tok/s      16.5x
Memory Use      80GB          40GB          2.0x
```

### Batch Processing (batch_size=8)
```
Metric          CPU (FP32)    GPU (BF16)    Speedup
------------------------------------------------------
Latency         2000ms        45ms          44.4x
Throughput      4 tok/s       177 tok/s     44.3x
Memory Use      85GB          45GB          1.9x
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**
   - Reduce batch size
   - Enable memory pooling
   - Check for memory leaks

2. **Precision Errors**
   - Increase tolerance in critical paths
   - Use FP32 for accumulation
   - Validate against CPU reference

3. **Performance Issues**
   - Profile with rocprof
   - Check GPU utilization
   - Optimize kernel launch configuration

## Next Steps

### Immediate Optimizations:
1. Implement Flash Attention
2. Add hipBLAS for GEMM operations
3. Optimize expert dispatch pattern
4. Implement pipeline parallelism

### Advanced Features:
1. Dynamic batching
2. Continuous batching
3. PagedAttention for KV cache
4. Quantization support (INT8/INT4)

## Questions & Answers

### Q1: Should run state buffers be FP32 or BF16?
**A:** Use hybrid approach:
- FP32 for critical accumulations (x, residual, attention scores)
- BF16 for memory-intensive buffers (KV cache, MLP intermediates)

### Q2: How to handle float32 interface with BF16 internally?
**A:** Implement conversion layer:
- Convert FP32→BF16 during weight loading
- Keep interface buffers in FP32
- Convert at kernel boundaries

### Q3: Optimal expert distribution strategy?
**A:** Hybrid approach:
- Static assignment (4 experts/GPU) for load balance
- Dynamic routing for tokens to minimize communication
- Local expert caching for frequently used experts

### Q4: Layer-wise vs expert-wise parallelism?
**A:** Use both:
- Layer-wise for pipeline parallelism
- Expert-wise within each layer for MoE
- This maximizes both memory and compute utilization

## Conclusion

This implementation provides a solid foundation for GPU acceleration of the GPT-OSS 20B model. The modular design allows for incremental optimization while maintaining compatibility with the existing codebase. Focus on implementing the immediate optimizations listed above for maximum performance gains.