# Col-Bandit：零样本查询时剪枝用于后期交互检索

> 来源：Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval | 日期：20260318 | 领域：search

## 问题定义

**后期交互模型（Late-Interaction）** 如 ColBERT：用 bi-encoder 生成 query 和 document 的 token-level embedding，在检索时做 MaxSim 操作（每个 query token 找最相似的 doc token 取最大值，再求和）。效果接近 cross-encoder，速度远快于 cross-encoder。

问题：ColBERT 的 MaxSim 需要对所有 query token 都做计算，但实际上：
- 不同 query 的 token 重要性差异巨大（"What"、"the" 等无信息词贡献有限）。
- 对不同候选文档，有效的 query token 也不同。

能否在**不需要训练**（零样本）的情况下，在查询时动态剪枝无用 query token，在保持效果的同时大幅降低计算量？

## 核心方法与创新点

1. **Bandit 框架**：将 query token 剪枝建模为 Multi-Armed Bandit 问题：
   - 每个 query token 是一个"臂"（arm）。
   - 每步选择一个 token，观察其对 score 排名的"贡献"（reward）。
   - 用 UCB（Upper Confidence Bound）策略在探索（尝试新 token）和利用（用高 reward token）之间平衡。

2. **零样本收益估计**：无需训练数据，用 token 的 IDF（Inverse Document Frequency）和 self-attention 权重作为先验估计收益，初始化 bandit 的置信区间。

3. **早停条件**：当 top-K 排名在最近若干步内未发生变化，认为已收敛，停止剪枝迭代。

4. **集成到 PLAID**：ColBERT 的工业级实现 PLAID 已有候选过滤机制，Col-Bandit 作为插件在 PLAID 候选上做进一步精排，兼容现有部署。

## 实验结论

- 在 MS-MARCO 和 BEIR 数据集，query token 使用率降至 40-60%（剪枝掉 40-60% token 计算量）。
- MRR@10 和 NDCG@10 损失 < 0.5%（统计不显著）。
- 端到端推理速度提升约 1.5-2x（在 GPU 批处理场景）。
- 零样本 vs 有监督剪枝方法：效果相近（差约 1%），但无需标注数据。

## 工程落地要点

- **Token IDF 预计算**：需要在目标语料上统计 token IDF，离线预计算一次即可；更新频率低（月级别）。
- **批处理适配**：剪枝后不同 query 的有效 token 数不同，GPU 批处理时需要 padding 或动态批次分组，避免计算效率退化。
- **与精排的配合**：Col-Bandit 作为 ColBERT 精排阶段的加速组件，不影响第一阶段 ANN 召回。
- **调参**：UCB 的探索系数 c 需在效果-速度之间调参，通常 c=1-2 为最优区间。

## 面试考点

**Q: ColBERT 的 MaxSim 操作是什么？**
- 对每个 query token embedding，在 document 的所有 token embedding 中找余弦相似度最大的，取该最大值；所有 query token 的最大值求和得到最终 score。捕捉细粒度 token 级匹配。

**Q: Late-Interaction vs Cross-Encoder vs Bi-Encoder 的比较？**
- Bi-encoder：最快（O(1) 检索），精度中等；Cross-encoder：最慢（O(N·L²)），精度最高；Late-interaction（ColBERT）：中等速度（O(N·q·d)），精度接近 cross-encoder。

**Q: Bandit 算法的 UCB 策略原理？**
- `UCB = mean_reward + c * sqrt(log(t) / n_i)`，t 是总步数，n_i 是第 i 个臂被选次数；未被充分探索的臂有更大的置信上界，引导探索；随时间收敛到最优臂。
