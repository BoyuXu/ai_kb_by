# PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation

> 来源：arXiv 2026 | 领域：llm-infra | 学习日期：20260408

## 问题定义

LLM 权重量化中，outlier weights 导致量化误差大。

**核心问题**：如何将权重分布转换为更适合量化的形式？

## 核心方法与创新点

**PolarQuant 三阶段流程**：

1. **Block-wise Normalization**：
   - 将权重块归一化到单位超球面
   - 消除尺度差异

2. **Walsh-Hadamard Rotation**：
   - Hadamard 矩阵旋转：$W' = HW$
   - 将不均匀分布转化为近高斯分布
   - 关键发现：Hadamard 旋转贡献了 98% 的质量改进

3. **Gaussian-Matched Quantization**：
   - 对近高斯分布使用最优量化码本
   - Lloyd-Max 量化器在高斯假设下可解析求解

## 关键结果

- 4-bit 量化下接近无损
- Hadamard 旋转是关键组件

## 面试考点

- Hadamard 矩阵性质：正交、可快速计算
- 为什么旋转能改善量化？（均匀化分布）
