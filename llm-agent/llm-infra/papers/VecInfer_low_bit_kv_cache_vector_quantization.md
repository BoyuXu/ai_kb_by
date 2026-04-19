# VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization
> 来源：arXiv:2510.06175 | 领域：llm-infra | 学习日期：20260419

## 核心方法
1. **离群值抑制**：Smooth变换 + Hadamard变换处理Key Cache中的离群值
2. **向量量化（Vector Quantization）**：对KV Cache进行VQ压缩
3. **融合CUDA Kernel**：计算与反量化融合，最小化内存访问开销
4. **极低比特压缩**：实现低于4-bit的KV Cache压缩

## 面试考点
- Q: KV Cache中为什么会有离群值？
  - A: 某些attention head对特定维度的activation值极端（如sink tokens），影响均匀量化精度
