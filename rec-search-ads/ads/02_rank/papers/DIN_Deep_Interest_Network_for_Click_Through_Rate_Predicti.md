# DIN: Deep Interest Network for Click-Through Rate Prediction

> 来源：arxiv (KDD 2018, Alibaba) | 日期：20260322 | 领域：广告系统（经典工作补充）

## 问题定义

传统 CTR 模型（Wide&Deep, DeepFM）对用户历史行为序列做简单 pooling（avg/sum），丢失了与目标广告相关的局部兴趣信号。如何让模型动态关注与当前广告最相关的历史行为？

## 核心方法与创新点

- **兴趣激活机制（Attention-based Interest Activation）**：
  - 对用户历史行为序列中每个 item，计算与目标广告的相关性权重（attention score）
  - 加权聚合历史行为，得到用户对目标广告的"激活兴趣"表示
- **Attention Score 计算**：
  ```
  score(h_i, a) = sigmoid(W * [h_i; a; h_i - a; h_i ⊙ a])
  ```
  不使用 softmax（局部激活而非全局竞争）
- **数据自适应激活函数（Dice）**：对 PReLU 改进，根据数据分布自适应调整激活函数的控制点
- **Mini-Batch Aware Regularization**：针对广告系统的极稀疏特征（长尾商品 ID），只对 batch 内出现的参数做正则化

## 实验结论

- 淘宝广告点击率预估：AUC 提升 0.6%（工业界 0.1% 即显著）
- 相比 Wide&Deep：参数量仅增加 5%，AUC 提升 0.4%
- 消融：Attention 机制贡献最大（0.3% AUC），Dice 贡献 0.1%

## 工程落地要点

- **序列长度**：DIN 原文使用最近 50 个行为，实际部署可根据计算资源选择 50-200
- **Attention 计算复杂度**：O(L) 其中 L 为序列长度，适合实时 serving
- **特征工程**：用户行为序列需去噪（过滤误点击、机器行为）
- **扩展方向**：DIEN（引入 GRU 捕捉兴趣演化）、MIDN（多兴趣分解）、SIM（超长序列检索）

## 常见考点

1. **Q：DIN 的 Attention 为什么不用 softmax？**
   A：用户对不同类目的兴趣可以同时激活（不是互斥的），softmax 强制归一化会抑制多兴趣；sigmoid 允许每个历史行为独立评分

2. **Q：DIN vs Transformer Self-Attention 的区别？**
   A：DIN 是目标广告 vs 历史行为的 cross-attention（单向）；Transformer 是历史行为序列内的 self-attention（全连接）。DIN 更轻量，更适合实时推断

3. **Q：如何处理用户超长行为序列（1000+）？**
   A：SIM（Search-based Interest Model）：先检索（基于目标广告属性从长序列中检索 TopK 相关行为），再 DIN Attention
