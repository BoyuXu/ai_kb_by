# Token-Selective Propagation for Efficient LLM Inference

> 来源：arXiv 2026 | 领域：llm-infra | 学习日期：20260408

## 问题定义

Transformer 中，不是所有 token 都同等重要：
- 信息扩散现象：关键信息集中在少数 token 中
- 全量计算浪费大量资源

**核心问题**：如何选择性地传播关键 token，减少计算开销？

## 核心方法与创新点

1. **Information Diffusion Analysis**：
   - 分析注意力矩阵发现信息集中规律
   - 少数 "sink token" 承载大部分信息

2. **Dynamic Token Selection**：
   - 每层动态选择需要计算的 token 子集
   - 非关键 token 直接跳过

3. **KV Cache Compression**：
   - 选择性 token 传播自然压缩 KV Cache
   - 与 LazyLLM 等方法互补

## 关键结果

- 单 GPU 端到端延迟降低 1.88×
- 长上下文场景加速更明显

## 面试考点

- Token 重要性度量方法
- 动态稀疏与静态稀疏的区别
