# GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm

**ArXiv:** 2602.01865 | **Date:** 2026-02 | **Org:** Baidu

## 核心理念
借鉴 LLM 的 scaling 成功经验，以序列为核心构建 CTR 预估范式（Generative Ranking for Ads at Baidu）。

## 架构创新

### Sequence-First 范式
将用户-广告交互建模为自回归事件序列，捕获时序依赖，类似 LLM 的 next-token prediction。

### CamA 注意力机制（Causal Action-aware Multi-channel Attention）
- 有效捕获用户行为序列中的时序动态
- 建模特定 action 信号（点击、购买、跳过等）
- 多通道注意力融合稀疏特征和稠密 Transformer

## 线上 A/B 测试（百度大规模部署）
- 收入：**+3.05%**
- CTR：**+3.49%**

## 扩展性
展现出类似 LLM 的 scaling 特性：随交互序列加长，模型表达力单调且近似线性提升。

## 面试考点
- Sequence-First vs Feature-Interaction-First 的区别？
- 为什么 CTR 模型也能从 LLM scaling 规律中获益？
- 如何设计适合 CTR 的 causal attention？

**Tags:** #ads #ctr #sequence-modeling #llm-inspired #scaling #baidu
