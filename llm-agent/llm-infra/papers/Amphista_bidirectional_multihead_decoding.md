# Amphista: Bi-directional Multi-head Decoding for Accelerating LLM Inference

**ArXiv:** 2406.13170 | **Venue:** NAACL 2025 | **Date:** 2024-06

## 核心问题
LLM 自回归解码缺乏并行性，推理速度慢。

## 核心贡献
Amphista 是增强版 speculative decoding 框架，在 Medusa 基础上引入双向注意力机制，允许不同草稿头（drafting heads）之间交互。

## 关键技术
- **Auto-embedding Block**：双向 Transformer Encoder，带位置编码，每个 head 可 attend 到其他 head，提升协作预测能力
- **Staged Adaptation Layers**：多个因果 Transformer Decoder 层，在两个阶段适配 base LLM 的隐藏状态和采样 token，确保适配特征包含更丰富的上下文信息
- 非自回归（Non-autoregressive）风格并行推理

## 性能
Vicuna 33B 上：
- vs 自回归解码：**2.75×** 加速
- vs Medusa：**1.40×** 加速
- 无损生成质量（lossless）

## 与 Medusa 对比
Medusa 各草稿头独立预测，Amphista 通过双向注意力让各头协作，提升 speculation 准确率。

## 面试考点
- Speculative decoding 原理：draft + verify
- 为什么双向注意力能提升 speculation 准确率？
- 如何保持生成质量 lossless？

**Tags:** #llm-infra #inference #speculative-decoding #efficiency
