# OneRec-Think: Chain-of-Thought Reasoning in Generative Recommendation

> 来源：arXiv 2025 (快手) | 领域：llm-infra/rec-sys | 学习日期：20260408

## 问题定义

生成式推荐模型直接输出推荐结果，但缺乏可解释的推理过程。

**核心问题**：如何在生成式推荐中引入 CoT 推理，同时保持工业级推荐性能？

## 核心方法与创新点

1. **Itemic Alignment（物品文本对齐）**：
   - 将物品 ID 与文本描述在同一表示空间对齐
   - 支持基于文本的推理

2. **Reasoning Scaffolding（推理激活）**：
   - 生成显式推荐理由
   - 使用推荐特定奖励函数优化推理质量

3. **Reasoning Enhancement（推理增强）**：
   - 迭代优化推理链
   - 确保推理与最终推荐一致

## 关键结果

- 快手工业部署：APP 停留时间 +0.159%
- 生成可解释的推荐理由

## 工程启示

- CoT 推理在工业推荐系统中可行且有效
- 推理过程可用于用户画像理解和内容理解
