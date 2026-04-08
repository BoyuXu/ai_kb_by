# ARTIST: Agentic Reasoning and Tool Integration for LLMs via RL

> 来源：arXiv 2025 | 领域：llm-infra/agent | 学习日期：20260408

## 问题定义

LLM 使用工具时面临：何时调用、调用哪个、如何处理结果。

**核心问题**：如何通过强化学习训练 LLM 自主掌握工具使用策略？

## 核心方法与创新点

1. **Outcome-Based RL**：
   - 不监督中间步骤，只奖励最终结果
   - 让模型自由探索最优工具使用策略

2. **Multi-Turn Reasoning Chain**：
   - 多轮推理-调用-观察循环
   - 自适应迭代直到找到答案

3. **Adaptive Tool Selection**：
   - 模型学会判断何时需要工具、何时可直接回答
   - 避免不必要的工具调用

## 关键结果

- 最难数学任务上比基础模型提升 22%
- 学会自我纠正错误的工具调用

## 面试考点

- RL 在工具使用学习中的优势 vs SFT
- Outcome-based vs Process-based reward
