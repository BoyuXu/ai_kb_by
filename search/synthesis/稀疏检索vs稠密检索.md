# 稀疏检索 vs 稠密检索：从 BM25 到 SPLADE，搜索召回的两条腿

> 📚 参考文献
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-Unified-Eva...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Vs Sparse Retrieval Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_Unified_Evaluation_Fr.md) — Dense Retrieval vs Sparse Retrieval: Unified Evaluation F...
> - [Sparse Meets Dense Unified Generative Recommend...](../../search/papers/Sparse_Meets_Dense_Unified_Generative_Recommendations_wit.md) — Sparse Meets Dense: Unified Generative Recommendations wi...
> - [Splade V3 Sparse Retrieval](../../search/papers/SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M.md) — SPLADE-v3: Advancing Sparse Retrieval with Deep Language ...
> - [Dpr-Dense-Retrieval](../../search/papers/dpr_dense_retrieval.md) — 稠密检索 DPR：原理、训练与工程实践
> - [Splade-V3-Advancing-Sparse-Retrieval-With-Deep-...](../../search/papers/SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M.md) — SPLADE-v3: Advancing Sparse Retrieval with Deep Language ...


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


## 📐 核心公式与原理

### 1. BM25
$$BM25(q,d) = \sum_{t \in q} IDF(t) \cdot \frac{tf \cdot (k_1+1)}{tf + k_1(1-b+b\frac{|d|}{avgdl})}$$
- 经典稀疏检索评分

### 2. Dense Retrieval
$$score = E_q^T E_d$$
- 双塔编码器的向量内积

### 3. ColBERT MaxSim
$$score = \sum_i \max_j E_q^i \cdot E_d^j$$
- 每个 query token 找最相似的 doc token

### Q1: 搜索系统的评估指标有哪些？
**30秒答案**：离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。

### Q2: 稠密检索的训练数据构造？
**30秒答案**：正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。

### Q3: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。

### Q4: 向量检索的工程挑战？
**30秒答案**：①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。

### Q5: RAG 系统的常见问题和解决方案？
**30秒答案**：①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。

### Q6: E5 和 BGE 嵌入模型的区别？
**30秒答案**：E5（微软）：通用文本嵌入，支持 instruct 前缀。BGE-M3（BAAI）：多语言+多粒度+多功能（dense+sparse+ColBERT 三合一）。BGE-M3 更全面但模型更大。

### Q7: 搜索系统的 Query 分析流水线？
**30秒答案**：①Tokenization/分词→②拼写纠错→③实体识别→④意图分类→⑤Query 改写/扩展→⑥同义词映射。每一步都可以用 LLM 替代或增强，但要注意延迟约束。

### Q8: 搜索相关性标注的方法？
**30秒答案**：①人工标注（5 级相关性）：金标准但成本高；②点击日志推断：点击=相关（有噪声）；③LLM 标注：用 GPT-4 做自动标注（便宜但需校准）。实践中混合使用。

### Q9: 个性化搜索和通用搜索的区别？
**30秒答案**：通用搜索：同一 query 返回相同结果。个性化搜索：结合用户历史偏好调整排序。方法：用户 embedding 作为额外特征输入排序模型。风险：过度个性化导致信息茧房。

### Q10: 搜索系统的 freshness（时效性）怎么做？
**30秒答案**：①时间衰减因子：较新文档加权；②实时索引更新：新文档分钟级可搜；③时效性意图识别：检测「最新」「今天」等时效性 query。电商搜索中 freshness 影响较小，新闻搜索中至关重要。
