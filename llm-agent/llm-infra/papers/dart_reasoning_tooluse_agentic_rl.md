# DART: Reasoning and Tool-use Compete in Agentic RL

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2602.00994
- **Date**: 2026-04

## 核心贡献

揭示推理和工具使用在联合训练中产生冲突梯度，提出DART框架分离解决。

## 关键技术

- **LEAS**: 线性效应归因系统，量化推理与工具使用的干扰
- **DART框架**: 推理和工具使用token路由到独立LoRA适配器
- **独立梯度路径**: 避免能力间的梯度冲突

## 核心结果

- 7个工具增强QA benchmark平均EM +6.35%
- 显著优于联合训练基线
- 证实了推理与工具使用的训练冲突

## 面试考点

多能力LLM训练中的梯度冲突，LoRA路由策略，Agent RL训练方法论。
