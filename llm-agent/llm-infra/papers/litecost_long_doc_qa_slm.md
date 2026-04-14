# LiteCoST: Long-Document QA with Chain-of-Structured-Thought

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2603.29232
- **Date**: 2026-04

## 核心贡献

RL增强的框架，微调3B-7B小模型生成结构化输出，在长文档QA上达到GPT-4o水平。

## 关键技术

- **CoST模板**: 结构化思维链模板指导输出格式
- **两阶段微调**: SFT + GRPO（三重奖励）
- **小模型蒸馏**: 将大模型能力蒸馏到3B/7B模型

## 核心结果

- 3B SLM接近GPT-4o-mini性能
- 7B模型达到GPT-4o级别质量
- 延迟降低2-4倍

## 面试考点

知识蒸馏，结构化输出生成，GRPO奖励设计，长文档处理策略。
