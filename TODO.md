# TODO list

**Target Models:** 20B (24 layers, 32 experts) & 120B (36 layers, 128 experts)  
**Hardware:** 8x AMD MI250 GPUs  
**Optimization Goals:** Maximize throughput (tokens/sec) while maintaining correctness  

---

## Phase 1: Foundation & Refactoring 🏗️

### Code Architecture Refactoring
- [ ] **Create modular C++ architecture from existing Python implementation** (https://github.com/openai/gpt-oss)
---

## Phase 2: Memory & Compute Analysis
### Layer-wise Memory Profiling
- [ ] **Calculate precise memory requirements for each layer**

## Phase 3: Multi-GPU Parallelization Strategy 🔥
### Design Tensor/Data/Model/Expert Parallelism
- [ ] **Design for 20B model**
- [ ] **Design for 120B model**
---

## Phase 4: Implement FlashAttention, PagedAttention
- [ ] **FlashAttention Implementation**
- [ ] **PagedAttention System**
---

