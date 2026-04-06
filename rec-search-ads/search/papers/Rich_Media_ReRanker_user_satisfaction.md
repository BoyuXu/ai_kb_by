# Rich-Media Re-Ranker: A User Satisfaction-Driven LLM Re-ranking Framework

**Date:** 2025-2026 | **Application:** 工业级搜索系统

## 核心问题
现有重排方法的两大局限：
1. 多维用户意图建模不足
2. 忽视富媒体侧信息（视觉感知信号等）

## 系统架构

### Query Planner
- 分析 session 内的查询精炼序列，捕获真实搜索意图
- 将查询分解为清晰互补的子查询，扩大用户潜在意图覆盖

### VLM-based Visual Evaluator
- 对候选结果的富媒体侧信息（封面图等视觉内容）进行评估
- 生成视觉质量信号

### LLM-based Re-ranker
综合多个维度进行整体评估：
- 内容相关性和质量
- 信息增益（Information Gain）
- 信息新颖性（Information Novelty）
- 封面图视觉呈现

### Multi-task Reinforcement Learning
增强 VLM 评估器和 LLM 重排器的场景适应性。

## 工业部署
大规模工业搜索系统部署，在线用户参与率显著提升。

## 面试考点
- 用户满意度多维建模的挑战？
- VLM 信号如何与文本信号融合？
- Multi-task RL 如何提升场景适应性？

**Tags:** #search #reranking #llm #rich-media #user-satisfaction
