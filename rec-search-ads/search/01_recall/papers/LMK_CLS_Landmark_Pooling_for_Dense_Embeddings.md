# LMK > CLS: Landmark Pooling for Dense Embeddings

> 来源：arXiv | 日期：20260317

## 问题定义

密集检索（Dense Retrieval）中，文档通常通过 BERT 等编码器的 **[CLS] token 表征**为单一向量。[CLS] 的局限：
1. [CLS] 需要通过全层注意力聚合所有 token 信息，对长文档效果差
2. 预训练任务（MLM）不是为 [CLS] 作为全局表征优化的
3. 固定位置的 [CLS] 对位置信息不敏感

**Landmark Token（LMK）**：本文提出用**可学习的 Landmark token**（插入文档关键位置）替代或补充 [CLS] 作为全局表征。

## 核心方法与创新点

1. **Landmark Token 插入**
   - 在文档中按固定间隔（如每 64 tokens）插入特殊 `[LMK]` token
   - 多个 LMK token 覆盖文档不同段落，各自聚合局部信息
   - 最终文档向量 = 所有 LMK token 表征的池化（Mean/Max/Attention-weighted）

2. **局部注意力聚合**
   - 每个 LMK token 主要关注其周围窗口的 token（局部感受野）
   - 减少长程注意力噪声，对长文档更有效

3. **与 [CLS] 的对比**
   - [CLS]：全局感受野，随文档长度增加稀释
   - LMK：多点局部感受野，文档长度增加时通过增加 LMK 数量维持覆盖密度

4. **训练适配**
   - 在 Bi-Encoder 框架内端到端训练
   - LMK token 的 embedding 随机初始化，通过对比学习（InfoNCE）优化

## 实验结论

- MSMARCO: LMK pooling nDCG@10 比 [CLS] 提升约 1.5%
- BEIR 长文档子集（Robust04, ArguAna）: 提升约 3~5%（长文档优势显著）
- 推理时间：LMK 数量少时（2~4个）与 [CLS] 相当；数量多时略慢但效果更好

## 工程落地要点

1. **LMK 数量选择**：文档长度 ≤512 tokens 时用 2~4 个 LMK；长文档（>2048 tokens）可用 8~16 个
2. **存储代价**：多 LMK 时文档表征维度扩大 K 倍，存储和 ANN 检索代价相应增加
3. **与 Late-Interaction 结合**：LMK pooling 是 Single-Vector，存储比 ColBERT 小得多
4. **微调策略**：建议先在 LMK token 位置做 masked language modeling 预热，再做检索微调

## 常见考点

- **Q: 为什么 [CLS] 对长文档效果差？**
  A: Transformer 的全局注意力使 [CLS] 需要聚合所有 token 的信息，文档越长，[CLS] 的注意力越分散，难以捕获关键信息。注意力矩阵的稀疏性在长文档中更加严重。

- **Q: Landmark Pooling 和 Mean Pooling 有什么区别？**
  A: Mean Pooling 对所有 token 均等加权，包含大量停用词和噪声 token；Landmark Pooling 用特殊的可学习 token 选择性聚合，每个 LMK 覆盖一个局部段落，更结构化，且 LMK 有自己的语义学习空间。

- **Q: 如何在 ANN 检索中支持多向量文档表征？**
  A: 1) 每个 LMK 向量单独建索引，查询时取各向量检索结果的并集；2) 多向量聚合为单向量后建索引（牺牲部分精度换效率）；3) MaxSim 操作（类似 ColBERT）对查询向量和文档多向量计算。
