# O-Researcher: Open Ended Deep Research via Multi-Agent Distillation and Agentic RL

**ArXiv:** 2601.03743 | **Date:** 2026-01

## 核心问题
闭源和开源 LLM 在深度研究任务上的性能差距，主要源于高质量训练数据的获取差异。

## 核心方案
自动化合成高质量研究级指令数据的框架：
1. **多 Agent 工作流**：协作 AI agents 模拟复杂工具集成推理，端到端生成多样、高保真训练数据
2. **两阶段训练策略**：
   - 阶段1：监督微调（SFT）
   - 阶段2：新型强化学习（Agentic RL），最大化模型对齐和能力

## 关键创新
- 不依赖闭源模型的直接蒸馏，而是通过多 Agent 协作模拟生成数据
- Agentic RL：专为 agent 场景设计的 RL 方法

## 性能
在主要深度研究基准上达到开源模型新 SOTA，多个模型规模均有效。

## 面试考点
- Multi-agent distillation vs 单模型蒸馏的区别？
- Agentic RL 如何定义奖励？
- 深度研究任务的评估挑战？

**Tags:** #llm-infra #agent #deep-research #multi-agent #distillation
