# MFLI: Multifaceted Learnable Index for Large-scale Recommendation

> 来源：https://arxiv.org/abs/2602.16124 | 领域：推荐系统 | 学习日期：20260331

## 问题定义

大规模推荐中ANN检索依赖固定量化索引（PQ/IVF），索引构建与模型训练分离导致检索质量受限。

## 核心方法与创新点

1. **可学习索引**：索引结构（聚类中心、量化码本）与推荐模型联合端到端训练
2. **多面索引**：每个item学习多个语义facet的索引编码，支持多角度召回
3. **残差量化+对比学习**：

$$
L = L_{rec} + \lambda_1 L_{index} + \lambda_2 L_{contrast}
$$

4. **Beam Search解码**：在线推理时beam search在索引树上高效检索

## 实验结论

在工业数据集上，MFLI相比传统IVFPQ索引Recall@100提升15%，相比TIGER等生成式检索提升5-8%。

## 工程落地要点

- 索引构建可离线进行，不影响在线服务
- 多facet索引天然支持多路召回融合
- 需定期re-index以应对新item
- 可与向量数据库（Milvus/Faiss）集成

## 面试考点

1. **可学习索引vs传统ANN索引？** 端到端优化避免目标不一致
2. **多facet的直觉？** 一个商品可从品类、价格、风格等多角度被检索
3. **生成式检索与传统检索的trade-off？** 生成式更灵活但训练更复杂
