# MiniMax-Text-01: Scaling Foundation Models with Lightning Attention

**ArXiv:** 2501.08313 | **Date:** 2025-01

## 核心贡献
MiniMax-01 系列（MiniMax-Text-01 + MiniMax-VL-01），通过 Lightning Attention 实现超长上下文，同时使用 MoE 扩展模型容量。

## 模型规格
- **总参数**：456B（32 experts MoE）
- **激活参数**：45.9B per token
- **上下文窗口**：训练 1M tokens，推理可外推至 4M tokens

## 核心技术：Lightning Attention
- 线性注意力机制变体，实现 O(n) 复杂度
- 高效计算-通信重叠（computation-communication overlap）
- 与 MoE 集成优化并行策略

## 多模态扩展
MiniMax-VL-01：在 MiniMax-Text-01 基础上，用 512B 视觉-语言 tokens 继续训练

## 性能对标
- 匹配 GPT-4o 和 Claude-3.5-Sonnet
- 上下文窗口比竞品长 **20-32 倍**

## 工业意义
超长上下文对文档处理、代码库理解、长对话等场景价值极大，Lightning Attention 是关键基础设施创新。

## 面试考点
- Linear Attention vs Standard Attention 的权衡？
- MoE 如何与长上下文 Attention 协同？
- 4M token 外推如何做到？

**Tags:** #llm-infra #long-context #moe #attention #scaling
