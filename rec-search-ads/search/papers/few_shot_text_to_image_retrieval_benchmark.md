# Few Shots Text to Image Retrieval: Benchmark and Optimization

> 来源：arXiv 2026 | 领域：search | 学习日期：20260408

## 问题定义

文本到图像检索中，组合查询和 OOD 查询表现差。

## 核心方法与创新点

1. **FSIR-BD 基准数据集**：
   - 38,353 图像，303 查询
   - 平均每查询 37 个正样本

2. **Few-Shot 优化方法**：
   - 利用 1-few 参考示例改进检索
   - 兼容任意预训练图像编码器

## 工程启示

- 少样本学习在视觉检索中的应用前景
