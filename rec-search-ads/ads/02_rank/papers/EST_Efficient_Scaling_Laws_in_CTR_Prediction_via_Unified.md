# EST: Efficient Scaling Laws in CTR Prediction via Unified Modeling

> 来源：arxiv | 日期：20260316 | 领域：ads

## 问题定义

CTR 预估模型（Wide&Deep、DCN、DeepFM 等）通过增大参数量提升效果，但存在 **Scaling 效率低** 的问题：
- 参数量从 100M → 1B，效果提升 <5%，但计算成本增加 10x。
- 不同模块（Embedding、MLP、Attention）的 Scaling 效率不同，无差别增大浪费资源。
- 缺乏类似 GPT/BERT 的 Scaling Law 指导 CTR 模型设计。

EST 研究 CTR 预估的 Scaling Law，并提出统一建模架构，找到最优 Scaling 路径。

## 核心方法与创新点

1. **Scaling Law 实证研究**：
   - 系统测量 CTR 模型在不同参数量（1M-10B）下的 AUC 提升曲线。
   - 发现 CTR 模型遵循 **Power Law**：AUC ∝ N^α（N=参数量，α≈0.05-0.07）。
   - 关键发现：**Embedding 层 Scaling 效率远高于 MLP**，同样参数量下 embedding 维度提升 > 层数增加。

2. **统一建模架构（Unified Model）**：
   - 将用户行为序列、用户画像、item 特征统一用 **Transformer Encoder** 处理。
   - 废弃独立的 FM/DNN 组件，用统一 attention 建模所有特征交叉。
   - 参数共享：跨场景（App/Web/Mobile）共享底层 Transformer，上层加场景特定 head。

3. **High-Efficiency Scaling 路径**：
   - 根据 Scaling Law 实验，给出最优配置：优先扩大 embedding dim，其次增加 attention heads，最后增加 MLP 层数。

## 实验结论

- 在 1B 参数规模下，EST 相比同规模 DCNv2 AUC +0.15%（CTR 领域显著提升）。
- FLOPs 相同条件下，EST 比独立组件架构效率提升 **2.3x**（更高 AUC per FLOP）。
- Scaling Law 预测误差 < 0.5%，可用于工程资源规划。

## 工程落地要点

- CTR 模型 0.1% AUC 提升 ≈ 线上 1-3% RPM 提升，是工业界重要优化目标。
- 统一 Transformer 架构训练成本高，需混合精度训练（FP16/BF16）+ 梯度 checkpointing。
- Embedding 参数是模型最大开销（通常占 80%+），用 Hash Trick + 低秩分解压缩。
- 建议先用小模型验证 Scaling Law 斜率 α，再外推到大模型预测效果，避免资源浪费。

## 常见考点

- Q: 什么是 Scaling Law？在 LLM 和 CTR 中有何不同？
  A: Scaling Law 描述模型性能与参数量/数据量/计算量的幂律关系（Kaplan et al. 2020）。LLM 的 α 约为 0.07（参数量翻倍，loss 降低约 5%）；CTR 模型 α 更小（约 0.05），因为 CTR 数据的噪声上限（贝叶斯误差）更高，大模型收益递减更快。

- Q: CTR 模型中的特征交叉有哪些方式？
  A: (1) 显式交叉：FM（内积）、DCN（cross network）、CIN（向量卷积）；(2) 隐式交叉：MLP（多层非线性）；(3) Attention 交叉：Transformer self-attention 建模 all-pair 特征交互（本文）。

- Q: 为什么 Embedding 的 Scaling 效率高于 MLP？
  A: Embedding 直接扩大特征表示空间，让模型能区分更细粒度的用户/item 差异。MLP 增大只是扩展非线性组合能力，但受限于 embedding 表达力。信息瓶颈在 embedding 层。
