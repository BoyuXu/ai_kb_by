# OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation
> 来源：arxiv/2312.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
工业广告推荐系统中，生成（召回）和排序（精排）通常是两个独立模型：
- 各自维护独立参数，特征工程重复
- 召回阶段信息损失导致排序阶段质量上限受限
- 多阶段优化目标不一致，存在 reward hacking
- 系统维护复杂，迭代成本高

## 核心方法与创新点
**OneRanker**：单一模型同时完成生成（序列推荐/候选生成）和排序（精排评分）。

**双头架构：**
```
Shared Encoder: h = Transformer(user_features, item_features, context)
Generation Head: P(next_item | h) = Softmax(h · E^T)  # 自回归生成
Ranking Head:   score = MLP(h_user ⊕ h_item)           # 点积/MLP打分
```

**联合训练目标：**
```
L = α·L_gen + β·L_rank + γ·L_aux
L_gen  = -log P(clicked_item | history)   # 生成式 next-item 预测
L_rank = BCE(score, label)                # 二分类 CTR
L_aux  = 辅助任务（停留时长、转化等）
```

**特征共享策略：**
- 用户序列 Transformer 层全共享
- 任务特定层分叉（top-k 层为任务专属）
- Cross-attention 融合候选 item 特征

## 实验结论
- 线上 A/B 测试（某头部广告平台）：
  - CTR +1.8%，CVR +2.3%，RPM +2.1%
- 离线评估：
  - 召回 Recall@50：+4.2%（vs 独立召回模型）
  - 精排 AUC：+0.003（vs 独立精排模型）
- 模型效率：参数量减少 35%，推理 QPS 提升 1.5x

## 工程落地要点
1. **分阶段推理**：召回阶段用 Generation Head + ANN，排序阶段用 Ranking Head
2. **梯度冲突处理**：使用梯度外科手术（Gradient Surgery）或 PCGrad 缓解多任务梯度冲突
3. **样本空间不一致**：召回样本（全库负采样）≠ 排序样本（曝光未点击），需分别构建训练集
4. **在线更新**：实时日志流更新序列特征，全量参数异步更新
5. **延迟预算**：Generation Head 用 FAISS 近似检索，保证召回 latency <10ms

## 常见考点
**Q1: 生成头和排序头梯度冲突如何解决？**
A: PCGrad（投影梯度）或 GradNorm（动态权重平衡）。具体做法：当两任务梯度方向夹角>90°，将冲突梯度投影到正交方向，只保留非冲突分量。

**Q2: OneRanker 召回和精排的样本如何统一？**
A: 不完全统一，而是共享底层参数。召回用全库随机负采样训练 Gen Head；精排用曝光样本训练 Rank Head。两个训练集可以混合 batch，也可以交替 mini-batch。

**Q3: 相比两阶段独立模型，OneRanker 最大优势是什么？**
A: 消除了 Stage Gap（召回模型看不到精排特征的信息差）。共享 Encoder 使召回和精排基于相同的特征表示，召回出来的候选在精排阶段评分更一致。

**Q4: 如何衡量生成和排序两个任务的贡献？**
A: Ablation：①只用 Rank Head（传统精排）②只用 Gen Head（纯生成召回）③联合训练（OneRanker）。通常联合训练在端到端指标上优于任一单独任务。

**Q5: 工业部署中 OneRanker 的主要挑战？**
A: 延迟：生成式召回比 ANN 向量检索慢 3-5x；需要 speculative decoding 或分层 beam search 加速。同时模型更大，GPU 资源成本更高。
