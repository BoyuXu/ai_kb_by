# 稀疏检索 vs 稠密检索：从 BM25 到 SPLADE，搜索召回的两条腿

> 📚 参考文献
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-Unified-Eva...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-unified-evaluation-framework.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-a-unified-evaluation-framework-for-large-scale-product-search.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Vs Sparse Retrieval Eval](../../search/papers/20260322_dense_vs_sparse_retrieval_eval.md) — Dense Retrieval vs Sparse Retrieval: Unified Evaluation F...
> - [Sparse Meets Dense Unified Generative Recommend...](../../search/papers/20260323_sparse_meets_dense_unified_generative_recommendatio.md) — Sparse Meets Dense: Unified Generative Recommendations wi...
> - [Splade V3 Sparse Retrieval](../../search/papers/20260322_splade_v3_sparse_retrieval.md) — SPLADE-v3: Advancing Sparse Retrieval with Deep Language ...
> - [Dpr-Dense-Retrieval](../../search/papers/20260316_dpr-dense-retrieval.md) — 稠密检索 DPR：原理、训练与工程实践
> - [Splade-V3-Advancing-Sparse-Retrieval-With-Deep-...](../../search/papers/20260321_splade-v3-advancing-sparse-retrieval-with-deep-language-models.md) — SPLADE-v3: Advancing Sparse Retrieval with Deep Language ...


**一句话**：搜索召回有两种基本路线——数数词（稀疏/词汇检索）和算语义（稠密/向量检索）——SPLADE-v3 是把两者优点融合的新代表。

**类比**：
- BM25（稀疏）：图书馆卡片目录，你搜"苹果手机"，查哪些书包含这两个词，词频越高排越前。快、精确、不懂"iPhone 也是苹果手机"。
- Dense（稠密）：语义相似的书放在同一个房间，你进去自然能找到相关书。懂语义、但找房间要时间（ANN）。
- SPLADE：图书管理员先帮你把"苹果手机"扩展成"iPhone、智能手机、iOS 设备..."再去查卡片目录。两全其美。

**核心机制（SPLADE）**：
```
1. 输入文本 → BERT MLM Head
2. 对词汇表每个 token 计算权重：w_t = log(1 + ReLU(BERT_output_t))
3. 得到稀疏向量（大多数词权重=0，只有约 120/30K 词有权重）
4. 用 FLOP 正则化强制稀疏度
5. 存入倒排索引（和 BM25 一样的基础设施！）
```

**SPLADE-v3 的改进**：
- 双正则化（FLOP + Saturation）：更精细控制稀疏度，避免高频词权重饱和
- MarginMSE + KL 联合蒸馏：从 CrossEncoder teacher 更高效传递知识
- INT8 量化感知训练：延迟降 40%，精度损失 <0.3% MRR

**三种检索方式对比**：
| 维度 | BM25（稀疏）| SPLADE-v3（神经稀疏）| Dense DPR/BGE |
|------|-----------|-----------------|-------------|
| 索引类型 | 倒排索引 | 倒排索引（稀疏向量）| ANN（HNSW/Faiss）|
| 存储大小 | 小 | 中（约 Dense 1/7）| 大（高维向量）|
| 查询延迟 | ~3ms | ~8ms | ~12ms |
| 同义词理解 | ❌ | ✅ 词汇扩展 | ✅ 语义向量 |
| 精确词匹配 | ✅ 强 | ✅ 强 | ❌ 弱 |
| BEIR 零样本 | 43.0 | 52.3 | ~50-54（DPR~41）|
| 冷启动/新领域 | 好 | 好 | 差（需微调）|
| 可解释性 | ✅ 词权重可看 | ✅ 扩展词可看 | ❌ 黑盒 |

**和今日 Dense 召回（Dense vs Sparse 对比论文）的连接**：
今日另一篇 "Dense vs Sparse Retrieval Unified Evaluation" 的核心结论：两种方法在不同查询类型上互补——精确商品 SKU 搜索 BM25/SPLADE 更好，模糊概念搜索 Dense 更好 → 混合是工业最优解。

**工业最佳实践**：
- 三层架构：SPLADE 召回 Top200 → Dense Reranker 精排 → CrossEncoder 最终 Top10
- 延迟分配：SPLADE ~8ms（召回）+ Dense ~15ms（重排）+ CE ~20ms（精排）= P99 ~50ms
- 存储：SPLADE 约 Dense 索引 1/7，可作为主召回省内存
- 增量更新：倒排索引天然支持增量添加新文档（Dense ANN 需要重建或用 HNSW 动态插入）

**面试考点**：
- Q: SPLADE 和 BM25 核心区别？ → BM25 字面词频；SPLADE 通过 MLM Head 做词汇扩展，"car"→也给"automobile"权重
- Q: 什么场景用 SPLADE，什么场景用 Dense？ → SPLADE：延迟严格（<10ms）、有精确匹配需求（品牌词/SKU）、存储受限；Dense：长尾语义查询、跨语言、多模态
- Q: 稀疏度为什么重要？ → 倒排索引高效查询依赖稀疏性（激活 token 越少，AND/OR 代价越低）；过稠密退化为全扫描

**演进脉络**：`TF-IDF → BM25 (词频统计) → SPLADE-v1 (BERT+MLM权重) → SPLADE-v2 (蒸馏优化) → SPLADE-v3 (异步流水线+QAT+双正则)`
