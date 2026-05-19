# 推荐系统 Scaling Laws 综合 — 2026-05-19

> 综合 3 篇当日学习论文：EST、Performance Law for SR、Scaling Law of Large Sequential Recommendation Models

## 一、技术演进

NLP 的 Chinchilla scaling law 揭示了"在固定算力下，模型参数与训练 token 的最优比例约为 1:20"。推荐系统侧的 scaling law 研究比 NLP 晚约两年，且面临三个根本差异：

1. **任务结构差异：** 推荐是 hidden state → next item 的检索问题，而非语言建模的 next token；user behavior 的语义稀疏度远高于自然语言
2. **延迟约束差异：** 推荐排序须毫秒级，无法直接堆模型规模
3. **数据特性差异：** 用户行为序列高度长尾，重复 / 冗余样本比例高

围绕这三大差异，2024–2026 推荐 scaling 研究经历了三个阶段：

| 阶段 | 代表工作 | 核心论点 |
|------|---------|----------|
| 验证存在性 | Scaling Law of Large SR Models (2023.11) | 纯 ID 序列即可观察到幂律 |
| 揭示数据质量 | Performance Law for SR (2024.12) | 数据"质"比"量"更关键，用 ApEn 度量 |
| 解决工程化 | EST (2026.02) | 无损统一序列 + 稀疏 attention 实现可服务的 scaling |

## 二、核心公式

**1. 经典 Chinchilla 形式（NLP）：**

L(N, D) = E + A/N^α + B/D^β

其中 N 是参数数、D 是 token 数、α≈0.34、β≈0.28。

**2. 推荐侧 Performance Law（Predictive Models in SR）：**

HR@k(N, D, ApEn) = E + A/N^α + B/D^β + C·g(ApEn)

引入 Approximate Entropy 作为数据质量项，**g(ApEn) 单调递增** — 即同等 (N, D) 下，更高熵的数据带来更高 HR。

**3. EST 的 attention 复杂度：**

- 标准 self-attention：O(L²·d)
- LCA + CSA：O(L·k·d)，k 为动态选择的高信号 behavior 数量

由此使 transformer 在 L≥10⁴ 的超长用户序列上仍能毫秒级 serve。

## 三、工业实践

**HSTU / SIM / OneRec 等大模型推荐的工程范式：**

1. **超长序列 + 稀疏 attention：** 不再用 SIM-Hard 的两阶段检索，而是把全序列送入模型，靠 attention 稀疏化（CSA、Linear Attention、Hyena）控制 FLOPs
2. **算力 / 数据 / 模型规模联合优化：** 借鉴 Chinchilla，用 Performance Law 指导"是该买更多卡 / 该扩数据 / 该加参数"
3. **数据质量预筛：** 用 ApEn / 困惑度 / 多样性 score 过滤掉冗余样本，等效于"用一半 token 训出同等模型"
4. **统一序列表征：** EST 范式 — 所有特征拼成 single sequence，避免传统 hierarchical pooling 的信息损失

**两条工业落地路线：**

- **生成式路线（OneRec / TIGER）：** 把推荐转成 next-Semantic-ID 生成，直接 inherit LLM 的 scaling
- **稠密路线（HSTU / EST）：** 保留 transformer 排序架构，靠 attention 改造与统一序列实现 scaling

## 四、面试考点

1. 推荐 scaling law 与 LLM scaling law 的本质差异是什么？为什么纯 ID 任务也能观察到 scaling？
2. Performance Law 中 ApEn 度量数据质量的直觉？为什么数据量到一定规模后边际收益递减？
3. EST 的 LCA 与 CSA 分别解决什么问题？为什么"无损统一序列"是关键？
4. HSTU、OneRec、EST 三大范式的差异与适用场景？
5. 工业上如何在固定算力预算下决定"扩参 vs 扩数据 vs 提质"？

## 参考

- [EST: Towards Efficient Scaling Laws in CTR via Unified Modeling](https://arxiv.org/abs/2602.10811)
- [Predictive Models in Sequential Rec: Performance Laws & Data Quality](https://arxiv.org/abs/2412.00430)
- [Scaling Law of Large Sequential Recommendation Models](https://arxiv.org/abs/2311.11351)
