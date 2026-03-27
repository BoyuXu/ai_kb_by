# 稀疏检索 vs 密集检索：怎么选，怎么融合

> 📚 参考文献
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-Unified-Eva...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Splade-V3-Advancing-Sparse-Retrieval-With-Deep-...](../../search/papers/SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M.md) — SPLADE-v3: Advancing Sparse Retrieval with Deep Language ...
> - [Dense Vs Sparse Retrieval Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_Unified_Evaluation_Fr.md) — Dense Retrieval vs Sparse Retrieval: Unified Evaluation F...
> - [Splade-V3 New Baselines For Splade](../../search/papers/SPLADE_v3_New_Baselines_for_SPLADE.md) — SPLADE-v3: New Baselines for SPLADE
> - [Dense Passage Retrieval For Open-Domain Questio...](../../search/papers/Dense_Passage_Retrieval_for_Open_Domain_Question_Answerin.md) — Dense Passage Retrieval for Open-Domain Question Answerin...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../search/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...


**一句话**：稀疏检索（BM25/SPLADE）像「关键词控」——精确匹配词汇，快且可解释；密集检索（DPR/E5）像「语义理解者」——懂同义词但慢且黑盒。现代系统通常两者都要。

**类比**：你去图书馆找书。稀疏检索像用目录索引——输入「机器学习」，直接找书名包含「机器学习」的书，精准快速。密集检索像问图书管理员——你说「我想学 AI 算法」，管理员理解你的意图，推荐《统计学习方法》《深度学习》等，语义匹配但可能偏。最佳体验是两者结合。


## 📐 核心公式与原理

### 1. BM25
$$
BM25(q,d) = \sum_{t \in q} IDF(t) \cdot \frac{tf \cdot (k_1+1)}{tf + k_1(1-b+b\frac{|d|}{avgdl})}
$$
- 经典稀疏检索评分

### 2. Dense Retrieval
$$
score = E_q^T E_d
$$
- 双塔编码器的向量内积

### 3. ColBERT MaxSim
$$
score = \sum_i \max_j E_q^i \cdot E_d^j
$$
- 每个 query token 找最相似的 doc token

---

## SPLADE-v3 的核心突破

### 稀疏检索进化史
```
BM25（TF-IDF+词频平滑）→ 词汇精确匹配，无法处理同义词
    ↓
DeepCT（2020）→ 用 BERT 加权词汇重要性，仍是关键词级别
    ↓
SPLADE（2021）→ BERT MLM head 生成语义稀疏向量（突破性）
    ↓
SPLADE-v2（2022）→ 文档扩展（doc2query），稀疏控制
    ↓
SPLADE-v3（今日）→ DeBERTa 基座 + 蒸馏 + 量化 = 接近 Dense 效果 + 稀疏效率
```

### SPLADE 核心原理（3 行理解）
1. 用 BERT 的 MLM 头，给词汇表中每个词生成权重
2. 用 ReLU + log 确保稀疏（大多数词权重 = 0）
3. 检索时做稀疏向量点积 = 等同于倒排索引查找（BM25 级别速度）

---

## 场景选型指南（今日实证数据）

| 查询类型 | 最优方案 | 原因 |
|---------|---------|------|
| 精确品牌查询（"耐克 Air Max 90"） | Sparse（BM25） | 词汇精确匹配无敌 |
| 语义探索查询（"适合婚礼的礼服"） | Dense（E5/BGE） | 理解语义意图 |
| 长尾低频 query | 混合检索（RRF 融合） | 单一方法都不稳定 |
| 百亿级商品，延迟 <10ms | Sparse / 量化 Dense | 延迟优先 |
| 多语言 / 低资源语言 | Dense（跨语言迁移） | Sparse 词汇不对齐 |

---

## 工程落地：混合检索的标准做法

```
用户 query
    ↓
┌─────────────────┬───────────────────┐
│  Sparse 召回     │   Dense 召回       │
│  (SPLADE/BM25)  │   (bi-encoder ANN) │
│  top-100        │   top-100          │
└─────────────────┴───────────────────┘
         ↓ RRF 融合（Reciprocal Rank Fusion）
     合并排序 top-200
         ↓ Cross-Encoder 精排（ColBERT / Reranker）
     最终 top-10
```

**RRF 公式**：`score(d) = Σ 1/(k + rank_i(d))`（k 通常取 60）

---

## 和已有知识的连接

- → [Dense vs Sparse 统一评估框架（今日）]：电商场景实证数据
- → [SPLADE-v3 tech deep-dive（今日）]：DeBERTa + 蒸馏细节
- → [20260320 Hybrid Retrieval Evolution]：RAG 场景的混合召回设计
- → [Query Reformulation（今日）]：query 改写可改善 Sparse 召回效果

---

## 面试考点

**Q：SPLADE 如何实现「语义稀疏」？为什么不直接用 BM25？**  
答：BM25 只匹配原始词汇，无法理解同义词（「跑鞋」vs「运动鞋」）。SPLADE 用 BERT 的 MLM Head 对每个 token 生成词汇表全维度权重，再用 log(1+ReLU(w)) 强制稀疏。这样「跑鞋」查询时，模型可能在「sneaker」「athletic shoe」「运动」等维度也生成权重，实现语义扩展。检索时仍是倒排索引，速度接近 BM25。

**Q：RRF 融合为什么常用 k=60？**  
答：k=60 是经验值，作用是平滑排名差异——rank=1 和 rank=2 的差距不那么大（1/61 vs 1/62），避免高排名结果权重过高。k 越小越激进（头部优势大），k 越大越平滑。实际工业中 k 可通过 dev set 调优。

**Q：什么时候 Sparse 比 Dense 好？**  
答：（1）精确品牌/型号查询；（2）低延迟要求场景（sparse 比 ANN 快 3-5×）；（3）可解释性要求（sparse 知道哪些词贡献了分数）；（4）数据量少时（sparse 不需要标注训练数据）。

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
