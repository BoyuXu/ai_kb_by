# LMK > CLS：地标池化用于稠密 Embedding

> 来源：LMK > CLS: Landmark Pooling for Dense Embeddings | 日期：20260318 | 领域：search

## 问题定义

稠密检索（Dense Retrieval）通常用 BERT 的 [CLS] token embedding 表示整段文本。但 [CLS] 的问题：
1. [CLS] 被设计用于 NSP（下一句预测）任务，并非天然适合语义表示。
2. 长文档时 [CLS] 难以聚合所有信息（注意力稀释）。
3. [CLS] 的表示高度依赖预训练目标，对检索任务的对齐度有限。

**Landmark Pooling（地标池化）** 提出：用文档中的**关键 token**（"地标"）的 embedding 做池化，替代 [CLS]，获得更具代表性的文档 embedding。

## 核心方法与创新点

1. **地标选择**：定义地标为对文档语义最重要的 token，选择标准：
   - **自注意力权重**：在 [CLS] 的注意力视角下，权重最高的 K 个 token。
   - **TF-IDF 权重**：文档中 IDF 最高的 K 个 token（关键词）。
   - **学习型选择**：训练一个轻量 selector 网络预测每个 token 的"地标分数"。

2. **地标池化操作**：选出 K 个地标 token 后，对其 embedding 做加权平均（权重 = 地标分数），得到文档 embedding。

3. **对比学习微调**：用 contrastive loss（正样本 = 相关文档，负样本 = 随机 + 难负例）微调地标选择器和 transformer，使地标 embedding 空间对检索任务对齐。

4. **多地标 embedding**：可选：保留 K 个地标 embedding（而非池化为一个），用 multi-vector 检索（类似 ColBERT），兼顾粒度和效率。

## 实验结论

- 在 MS-MARCO Passage Retrieval，相比 CLS-based DPR，MRR@10 提升约 1.5-2%。
- 在长文档检索（Natural Questions, TriviaQA），提升约 3-4%（长文档场景 CLS 退化更严重）。
- K=4-8 的地标数量是最优区间，更多地标边际收益递减。
- 地标选择方式：学习型 > TF-IDF > 注意力权重（在标注数据充足时）。

## 工程落地要点

- **索引兼容性**：单地标 embedding 与标准 ANN 索引（FAISS、Milvus）完全兼容；多地标 embedding 需要支持 multi-vector 检索的索引（如 PLAID）。
- **地标 selector 开销**：轻量 selector（1-2 层 MLP）的推理开销可忽略，不影响 encoding 速度。
- **文档切分策略**：超长文档仍需切分为段落（chunk），地标池化在段落级别应用效果最好。
- **与 instruction tuning 结合**：可以用 instruction（"为以下文档找最重要的句子"）引导 LLM 选择地标，效果更好但推理成本更高。

## 面试考点

**Q: 为什么 CLS token 不是最优的文档表示？**
- BERT 预训练中 CLS 用于 NSP 任务，并非专为语义匹配设计；在长文档中注意力分散，信息聚合能力有限；对检索任务需要微调才能对齐。

**Q: 稠密检索和稀疏检索（BM25）各有什么优势？**
- BM25：精确关键词匹配，计算高效，无需 GPU，对罕见词/术语效果好；Dense：语义匹配，处理同义词和改述，需 GPU 但已大量部署。混合检索（BM25 + Dense）通常最优。

**Q: Mean Pooling vs CLS Pooling，哪个更好？**
- 经验：sentence-transformers 发现 mean pooling 通常优于 CLS，因为它聚合了所有 token 的信息；但对于长文档，mean pooling 也会被无关 token 噪声干扰，地标池化是折中方案。
