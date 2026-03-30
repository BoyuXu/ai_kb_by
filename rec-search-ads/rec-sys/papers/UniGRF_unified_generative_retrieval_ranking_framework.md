# UniGRF: Unified Generative Retrieval and Ranking Framework
> 来源：arXiv:2504.16454 | 领域：rec-sys | 学习日期：20260330

## 问题定义
推荐系统传统上将召回（Retrieval）和排序（Ranking）作为独立阶段，存在目标不一致、误差传播、维护成本高等问题。UniGRF 用单一生成式模型统一召回和排序，以 Beam Search 生成有序物品 ID 列表，实现端到端优化。

## 核心方法与创新点
1. **Generative Retrieval as Sequence Generation**：将物品 ID 映射为语义码本（Semantic ID），用自回归模型生成有序物品序列，Beam Search 的 score 直接作为排序依据。
2. **Semantic ID 构建**：用 RQ-VAE（Residual Quantization VAE）将物品语义压缩为层次化 codebook token，保证相似物品有相近 ID 前缀（树形组织）。
3. **联合训练目标**：同时优化生成准确性（cross-entropy on item ID tokens）和排序质量（listwise ranking loss），两目标通过动态权重平衡。
4. **增量更新**：新物品只需更新 codebook 末端，无需重训全模型，解决实时物品入库问题。

## 实验结论
- 电商场景：Recall@50 提升 5.3%，NDCG@10 提升 6.1%（对比两阶段 DSSM+DIN）
- 冷启动物品效果提升最显著（+12% Recall），得益于语义 ID 的语义泛化
- 统一模型推理一次完成召回+排序，总延迟减少 ~30%（省去两次模型推理+候选传输）

## 工程落地要点
- Semantic ID 质量是关键，RQ-VAE 需在完整物品语料上预训练
- Beam Search 宽度决定召回质量 vs 延迟 tradeoff（建议 beam=50-200）
- 新物品在 codebook 中的位置分配需实时系统支持
- 生成式模型对物品空间大小敏感（>1000万商品需 Hierarchical Beam Search）

## 面试考点
- Q: 生成式召回和双塔模型召回的核心区别？
  - A: 双塔：user/item 向量内积，ANN 检索；生成式：直接 autoregressive 生成 item ID，天然排序，但受限于 Beam 宽度
- Q: RQ-VAE 在生成式推荐中的作用？
  - A: 将连续 item embedding 量化为离散 code，使 item ID 具有语义结构（相似 item 共享前缀），提升泛化
- Q: 统一召回排序模型如何处理实时更新问题？
  - A: Embedding 层实时更新（Parameter Server），Semantic ID 增量扩展，定期全量重训 codebook
