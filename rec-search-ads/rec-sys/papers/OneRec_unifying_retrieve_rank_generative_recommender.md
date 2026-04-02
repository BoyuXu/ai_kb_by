# OneRec: Unifying Retrieve and Rank with Generative Recommender
> 来源：arXiv:2502.18965 | 领域：rec-sys | 学习日期：20260330

## 问题定义
传统推荐 pipeline 将召回（retrieve）和排序（rank）分为独立子系统，两阶段分别优化导致目标不一致。OneRec 提出单一生成式模型，将"召回+排序"统一为 autoregressive 序列生成任务，一次前向输出有序推荐列表。

## 核心方法与创新点
1. **Decoder-Only 架构**：基于 Causal LM（GPT-style），将推荐历史作为 context，自回归生成物品 ID token 序列。
2. **物品 ID 离散化**：物品通过语义聚类（K-means on item embedding）映射为多层次离散 code（类似 SoundStream/EnCodec 的 RVQ 结构），每个物品用 1-4 个 token 表示。
3. **Beam Search 即排序**：Beam Search 过程产生的 log-prob 直接作为排序 score，无需额外 scoring pass。
4. **训练目标**：交叉熵 loss 在 item token 序列上，加上 ListMLE（listwise 排序 loss）对全序列质量约束。
5. **快手工业验证**：在数亿 DAU 的短视频推荐平台上实测，证明工业级可行性。

## 实验结论
- 快手线上 A/B：用户活跃时长 +2.3%，完播率 +1.8%
- 离线 NDCG@10 超越 DSSM+SASRec 两阶段基线 7.2%
- 端到端系统延迟 ~150ms（含 Beam Search），通过 Speculative Decoding 可降至 ~80ms

## 工程落地要点
- 候选空间裁剪：全量 Beam Search 不可行（物品>千万），需先用轻量过滤缩小候选到 1-10 万
- Beam 宽度权衡：32-64 是精度/延迟的 sweet spot
- 物品 ID 的 code 层次数影响 recall 和精度：层次越多，表达越精细但生成步骤越多
- 需要专用推理引擎（TensorRT/vLLM 定制）支持 item token 空间约束的 Beam Search

## 常见考点
- Q: OneRec 如何保证生成物品的多样性？
  - A: Beam Search 天然保留多条路径；加入 diversity penalty（相似 item score 惩罚）；训练时加入 negative sampling 避免热门物品垄断
- Q: 为什么生成式推荐的延迟比双塔+ANN 高？
  - A: 双塔 ANN ~10ms；生成式每次 Beam Search 需多步 forward（O(L×beam) 次），L 是物品 token 数
- Q: 生成式推荐和 BERT4Rec 等序列模型的区别？
  - A: BERT4Rec 用 masked LM 做 next item prediction（pointwise），生成式直接 autoregressive 输出有序列表（listwise），天然支持多步规划
