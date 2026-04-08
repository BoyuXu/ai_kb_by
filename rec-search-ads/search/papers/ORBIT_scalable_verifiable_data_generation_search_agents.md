# ORBIT: Scalable and Verifiable Data Generation for Search Agents

> 来源：arXiv 2026 | 领域：search | 学习日期：20260408

## 问题定义

训练搜索 Agent 需要大量高质量推理数据，但商业 API 成本高昂。

**核心问题**：如何低成本生成可验证的搜索训练数据？

## 核心方法与创新点

1. **模块化数据生成框架**：
   - Seed 创建 → QA 对生成 → 双重验证
   - 使用开源工具，不依赖付费 API

2. **Self + External Verification**：
   - 自验证：模型检查自身生成一致性
   - 外验证：搜索引擎验证答案正确性

3. **多领域覆盖**：
   - 15 个领域，20K 推理密集查询
   - 每个查询附带可验证短答案

## 关键结果

- 单台笔记本即可运行
- 数据质量接近商业 API 生成

## 工程启示

- 低成本数据生成是搜索 Agent 民主化的关键
