
# Performance Tracking

| ID | Optimization | Tok/s | Note |
|----|--------------|-------|------|
| 1 | Baseline | 2 | Initial implementation on CPU without optimizations |
| 2 | Move all to GPU | 3 | Move all weight, runstate to run on 1 GPU, all FP32 no batching |
| 3 | Optimze matmul tensor core, quantize Moe weight MXFP4, Continous batching | 300 | Batchsize 512 |