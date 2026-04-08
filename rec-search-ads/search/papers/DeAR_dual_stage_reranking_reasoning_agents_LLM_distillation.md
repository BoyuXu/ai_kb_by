# DeAR: Dual-Stage Document Reranking with Reasoning Agents

> 来源：arXiv 2025 | 领域：search | 学习日期：20260408

## 问题定义

LLM 重排序成本高，且缺乏多粒度推理能力。

## 核心方法与创新点

1. **Pointwise Scoring（逐点评分）**：
   - 从 13B LLaMA 蒸馏 token 级信号到 3-8B 学生
   - 高效逐文档评分

2. **Listwise Reasoning（列表级推理）**：
   - LoRA 适配器 + 20K GPT-4o CoT 排列
   - 同时考虑文档间相对关系

3. **双阶段协同**：
   - Pointwise 初筛 → Listwise 精排
   - 兼顾效率和质量

## 关键结果

- NovelEval: +3.09 nDCG@10（超越 GPT-4）
- Natural Questions: 54.29 Top-1 Accuracy

## 面试考点

- Pointwise vs Listwise vs Pairwise 重排序
- LLM 蒸馏在重排序中的应用
