# RLVR: Reinforcement Learning from Verifiable Rewards for LLM Training

**ArXiv:** 2506.14245 | **Date:** 2025-06

## 核心问题
RLVR 通过基于答案正确性的可验证奖励训练 LLM，但其内在激励机制尚不明确。

## 核心贡献
提供理论框架，解释 RLVR 的激励机制：即使奖励仅基于答案正确性（不直接奖励推理过程），RLVR 也能隐式激励正确的思维链（CoT）推理。

## 关键发现
- RLVR 在训练早期就能激励正确推理
- 可扩展数学和代码任务的推理边界
- 引入新评估指标 **CoT-Pass@K**

## 算法基础
基于 **GRPO**（Group Relative Policy Optimization），DeepSeek-R1 采用的核心算法。

## 实践意义
- 不需要人工标注推理过程，仅需可验证的答案标签
- 对数学、代码等有客观答案的任务效果显著
- 开源模型通过 RLVR 可大幅提升推理能力

## 面试考点
- RLVR 与 RLHF 的核心区别？
- 为什么可验证奖励比人类偏好奖励更稳定？
- GRPO vs PPO 的优势？

**Tags:** #llm-infra #rlvr #reinforcement-learning #reasoning #grpo
