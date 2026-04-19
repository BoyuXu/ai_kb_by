# RAG Performance Prediction for Question Answering

> arXiv:2604.07985 | 2026-04 | Or Dado, David Carmel | 领域：RAG评估

## 一句话总结

研究如何预测 RAG 对 QA 任务的增益（相对于不用 RAG），提出一种基于 question-passage-answer 三元组语义关系建模的监督预测器，实现最优预测质量。

## 问题定义

**核心问题**：给定一个问题，预测使用 RAG 是否比不使用 RAG 效果更好（以及好多少）。

这对实际系统很有价值：
- 不必对每个 query 都走 RAG 流程（节省延迟和成本）
- 可以做自适应路由：简单问题直接回答，需要外部知识的问题走 RAG

## 预测器分类

### Pre-retrieval Predictors
- 在检索前基于 query 本身预测
- 借鉴 ad-hoc retrieval 中的 Query Performance Prediction (QPP)
- 特征：query 复杂度、query 与语料库的统计关系等

### Post-retrieval Predictors
- 在检索后基于 query + retrieved passages 预测
- 借鉴 IR 中的 post-retrieval QPP 方法
- 特征：检索文档的相关性分布、文档间一致性等

### Post-generation Predictors（本文重点）
- 在生成答案后，利用 question + passages + answer 三元组
- **Novel Predictor**：建模三者之间的语义关系
  - 答案与 passages 的一致性（grounding 程度）
  - 答案与 question 的语义匹配度
  - passages 对答案的支撑度

## 核心发现

1. Post-generation 预测器 > Post-retrieval > Pre-retrieval（信息量递增）
2. 新提出的三元组语义关系预测器取得最佳效果
3. QPP 方法从 IR 迁移到 RAG 场景是可行的，但需要适配

## 实际意义

```
Query 进入
  ↓
Pre-retrieval predictor → 低不确定性 → 直接 LLM 回答（跳过 RAG）
  ↓ 高不确定性
执行 RAG pipeline
  ↓
Post-generation predictor → 高置信度 → 返回答案
  ↓ 低置信度
fallback 策略（拒绝回答 / 人工审核）
```

## 与其他工作的关系

- RAG 系统评估维度见 [[2026-04-09_rag_systems_evolution|RAG 系统演进]]
- Query Performance Prediction 是 IR 经典问题，本文将其扩展到 RAG 场景
