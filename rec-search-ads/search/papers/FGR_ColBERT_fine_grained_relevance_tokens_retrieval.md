# FGR-ColBERT: Identifying Fine-Grained Relevance Tokens During Retrieval

> 来源：arXiv 2026 | 领域：search | 学习日期：20260408

## 问题定义

ColBERT 实现高效多向量检索，但无法指出哪些 token 贡献了相关性。

**核心问题**：如何在检索阶段同时预测 token 级相关性信号？

## 核心方法与创新点

1. **LLM 蒸馏监督**：
   - 从 Gemma 2 蒸馏 token 级相关性标注
   - 245× 更小的模型获得推理级别性能

2. **联合训练**：
   - 文档级检索蒸馏 + token 级相关性预测
   - 两个任务共享编码器

3. **极低延迟开销**：
   - 仅 1.12× 延迟增加
   - 保持 99% 原始检索效果

## 关键结果

- MS MARCO token 级 F1：64.5
- 检索效果几乎无损

## 面试考点

- 多向量检索中的可解释性
- LLM 蒸馏到小模型的效率优势
