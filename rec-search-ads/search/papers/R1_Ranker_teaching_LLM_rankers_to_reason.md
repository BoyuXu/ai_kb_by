# R1-Ranker: Teaching LLM Rankers to Reason

> 来源：arXiv 2025 | 领域：search | 学习日期：20260408

## 问题定义

LLM 排序器缺乏深层推理能力，依赖表面模式匹配。

## 核心方法与创新点

1. **DRanker（Direct Ranker）**：
   - 一次性完整排序
   - 适用于延迟敏感场景

2. **IRanker（Iterative Ranker）**：
   - 迭代消除式排序
   - 逐步奖励鼓励更深层推理
   - 类似辩论赛淘汰机制

3. **RL Training with Step-wise Rewards**：
   - 每个消除步骤都有奖励信号
   - 推理过程可解释

## 关键结果

- IRanker-3B SOTA：9 个排序数据集平均 +15.7%
- 超越更大的 7B 模型
- 零样本 OOD 提升 9%

## 面试考点

- 排序中的强化学习 vs 监督学习
- 迭代消除 vs 一次性排序的 trade-off
