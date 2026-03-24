# 推荐系统排序范式演进：从 LR 到 Transformer 到 LLM

> 📚 参考文献
> - [Model Calibration Deep Dive](../../rec-sys/papers/model_calibration_deep_dive.md) — 模型校准（Model Calibration）完整学习笔记
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/20260321_gems-breaking-the-long-sequence-barrier-in-generative-recommendation-with-a-multi-stream-decoder.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Reg4Rec Reasoning-Enhanced Generative Model For La](../../rec-sys/papers/20260323_reg4rec_reasoning-enhanced_generative_model_for_la.md) — REG4Rec: Reasoning-Enhanced Generative Model for Large-Sc...
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Gems Long Sequence Generative Rec](../../rec-sys/papers/20260322_gems_long_sequence_generative_rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Onerec Unifying Retrieve And Rank With Generative ](../../rec-sys/papers/20260323_onerec_unifying_retrieve_and_rank_with_generative_.md) — OneRec: Unifying Retrieve and Rank with Generative Recomm...


> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：DIN, DIEN, BST, DCN-V2, OneRec-Think, CADET 系列

---

## 🎯 核心洞察（5条）

1. **排序模型的核心任务是特征交叉**：从手工交叉（LR+特征工程）到自动交叉（FM/DeepFM）再到深度交叉（DCN-V2），本质是让模型发现有效的特征组合
2. **用户行为序列建模是精排提升的最大杠杆**：DIN → DIEN → BST → SIM，每一代都在更好地利用用户历史行为，AUC 提升 0.5-1.5%
3. **Attention 机制的引入是分水岭**：DIN 用 target attention 让模型"看向"与候选物品相关的历史行为，AUC 提升显著且工程可行
4. **多任务排序成为标配**：单一 CTR 模型已不够，工业系统同时预估 CTR/CVR/时长/完播率等 5-10 个目标，MMoE/PLE 是基础架构
5. **LLM 正在重塑排序**：CADET（Decoder-Only CTR）和 OneRec-Think（推理增强排序）代表两条路径——把排序模型替换为 LLM 或让 LLM 辅助理解用户意图

---

## 📈 技术演进脉络

```
LR + 手工特征交叉（2010-2013）
  → FM 自动二阶交叉（2013-2015）
    → Wide&Deep / DeepFM 深度+宽度（2016-2017）
      → DIN/DIEN 序列建模（2018-2019）
        → BST / Transformer 排序（2019-2021）
          → DCN-V2 / FiBiNet 高阶交叉（2020-2022）
            → 多任务 MMoE/PLE（2018-2022 持续）
              → LLM-enhanced 排序（2024-2026）
```

**关键转折点**：
- **FM（2010）**：自动学习特征交叉，终结了手工特征工程的苦力活
- **DIN（2018）**：引入 target attention，模型"知道该看用户的哪些历史行为"
- **BST（2019）**：Transformer 进入推荐排序，self-attention 捕捉行为序列内部依赖

---

## 🔗 跨文献共性规律

| 规律 | 体现论文/系统 | 说明 |
|------|-------------|------|
| 越来越长的行为序列 | DIN(50)→SIM(5000+) | 更多历史 = 更好的用户理解，关键是计算效率 |
| 显式交叉仍有价值 | DCN-V2, CAN | 即使有深度网络，显式的交叉层仍能提升效果 |
| Embedding 维度持续增大 | 8D→64D→128D | 更大的 Embedding 表达力更强，但存储和通信成本也更高 |
| 多目标成为基础设施 | MMoE→PLE→STAR | 排序模型必须同时服务多个业务目标 |

---

## 🎓 面试考点（7条）

### Q1: DIN 的 target attention 怎么工作？
**30秒答案**：DIN 不对用户所有历史行为等权求和，而是用候选物品作为 query，对历史行为做 attention 加权——"你正在看一双鞋，那么你过去买鞋的行为比买书的行为更重要"。权重 = softmax(MLP(concat(candidate, behavior)))。
**追问方向**：DIN 和 DIEN 的区别？答：DIEN 加了 GRU 建模行为序列的时序演化，捕捉兴趣漂移。

### Q2: Wide&Deep 的设计动机？
**30秒答案**：Wide 部分（LR）负责"记忆"已知的特征组合（如"用户 A 喜欢物品 B"）；Deep 部分（DNN）负责"泛化"到未见过的组合。两者互补，Google Play 首次验证。
**追问方向**：DeepFM 比 Wide&Deep 好在哪？答：DeepFM 的 Wide 部分用 FM 替代 LR，自动学习二阶交叉，无需手动设计 cross features。

### Q3: DCN-V2 的交叉层做了什么？
**30秒答案**：每层做 `x_{l+1} = x_0 ⊙ (W_l × x_l + b_l) + x_l`，即用初始特征 x_0 和当前层 x_l 做逐元素乘法交叉，堆叠 L 层实现 2^L 阶交叉。比 DCN-V1 增加了 W 矩阵（Mix 模式）使交叉更灵活。
**追问方向**：为什么不直接用更深的 MLP？答：MLP 学习交叉是隐式的，DCN 的显式交叉收敛更快且可解释。

### Q4: CTR 预估模型的离线评估指标？
**30秒答案**：①AUC（全局区分能力）；②GAUC（分组 AUC，按用户分组计算再加权平均，消除用户活跃度差异）；③LogLoss（校准性，预测概率 vs 实际点击率的差异）；④NDCG（排序质量）。
**追问方向**：AUC 高但 LogLoss 差怎么办？答：说明排序对但概率值不准，需要 calibration（Platt Scaling / Isotonic Regression）。

### Q5: 多任务排序中目标权重怎么设？
**30秒答案**：①手动调权：基于业务优先级设置 `total_loss = w1*L_ctr + w2*L_cvr + w3*L_duration`；②Uncertainty Weighting：每个任务的权重正比于 1/σ²（不确定性越大权重越小）；③Pareto 搜索：自动找到多目标最优权重组合。
**追问方向**：实际中最常用？答：先手动粗调，再用 Uncertainty Weighting 微调。

### Q6: Transformer 在推荐排序中的应用方式？
**30秒答案**：BST 用 Transformer 编码用户行为序列（self-attention 捕捉行为间依赖），然后用 candidate item 做 target attention 提取相关信号。与 NLP 的区别：序列更短（50-500 vs 1000+）、位置编码用时间间隔。
**追问方向**：Transformer 比 GRU/LSTM 好在哪？答：并行训练更快，长距离依赖捕捉更好，但推理成本更高。

### Q7: LLM 怎么用于推荐排序？
**30秒答案**：两条路——①LLM 作为特征增强（LEARN/PLUM：LLM 生成用户/物品语义特征，离线蒸馏到排序模型）；②LLM 作为排序主干（CADET：Decoder-Only 架构直接做 CTR 预估，用 LoRA 适配）。
**追问方向**：LLM 排序的延迟问题？答：特征增强路线无额外延迟；主干替换路线需要量化+蒸馏，目前仍是实验阶段。

---

## 🌐 知识体系连接

- **上游依赖**：特征工程、Embedding 表示学习、Transformer/Attention 机制
- **下游应用**：重排策略、多目标平衡、在线 A/B 测试
- **相关 synthesis**：std_rec_recall_evolution.md, std_rec_feature_engineering.md, std_cross_long_sequence.md
- **相关论文笔记**：rec-sys/01_ctr_models_deep_dive.md, rec-sys/05_ranking_deep.md
