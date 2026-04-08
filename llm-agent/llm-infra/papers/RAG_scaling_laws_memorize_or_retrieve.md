# To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining

> 来源：arXiv 2026 | 领域：llm-infra | 学习日期：20260408

## 问题定义

LLM 预训练需要海量数据来"记忆"知识，RAG 提供了"检索"替代方案：
- 记忆：大模型 + 大语料
- 检索：小模型 + 检索库

**核心问题**：给定固定预算，如何在模型大小、预训练数据量和检索库大小之间最优分配？

## 核心方法与创新点

**三维缩放流形（3D Scaling Manifold）**：

$$L(N, D, R) = f(N, D) + g(N, R)$$

其中 N=模型参数, D=预训练 token 数, R=检索语料库大小

1. **Scale-Dependent Crossover**：
   - 存在临界规模点：超过该点，检索比增加训练数据更高效
   - 小模型更受益于检索增强

2. **Optimal Data Budget Allocation**：
   - 给定总预算，最优分配预训练计算和检索存储

## 关键结果

- 检索在中小规模模型上 ROI 最高
- 超大模型的内部知识足够丰富，检索边际收益递减

## 面试考点

- RAG vs 大模型记忆的 scaling law 分析
- 计算预算最优分配策略
