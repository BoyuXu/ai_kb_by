# MF-QAT: Multi-Format Quantization-Aware Training for Elastic Inference

> 来源：arXiv 2026 | 领域：llm-infra | 学习日期：20260408

## 问题定义

不同硬件支持不同量化格式（INT8, FP8, MXINT, MXFP 等），单一格式模型无法跨平台部署。

**核心问题**：如何训练一个模型，使其在多种量化格式下都表现良好？

## 核心方法与创新点

1. **Multi-Format QAT**：
   - 训练时同时模拟多种量化格式
   - 模型学会对多种精度鲁棒的权重分布

2. **Slice-and-Scale Conversion**：
   - 运行时无缝转换到目标精度格式
   - 无需重新训练或校准

3. **Elastic Precision Scaling**：
   - 根据硬件能力自适应选择最优精度
   - 从 FP16 到 INT4 弹性伸缩

## 工程启示

- 解决异构硬件部署的量化格式碎片化问题
- 一次训练，多格式部署
