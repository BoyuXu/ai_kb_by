# Learning to Rank：搜索排序的三大范式

> 📚 参考文献
> - [Document Re-Ranking With Llm From Listwise To Pair](../../search/papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/Query_as_Anchor_Scenario_Adaptive_User_Representation_via.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../search/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../search/papers/Generative_Query_Expansion_for_E_Commerce_Search_at_Scale.md) — Generative Query Expansion for E-Commerce Search at Scale
> - [Hybrid-Search-Llm-Re-Ranking](../../search/papers/Hybrid_Search_with_LLM_Re_ranking_for_Enhanced_Retrieval.md) — Hybrid Search with LLM Re-ranking for Enhanced Retrieval ...
> - [Multimodal-Visual-Document-Retrieval-Survey](../../search/papers/Unlocking_Multimodal_Document_Intelligence_Visual_Documen.md) — Unlocking Multimodal Document Intelligence: Visual Docume...


> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：LambdaMART, RankNet, ListNet, BERT Reranker 系列


## 📐 核心公式与原理

### 1. NDCG
$$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG = \sum_{i=1}^K \frac{2^{rel_i}-1}{\log_2(i+1)}$$
- 搜索排序核心评估指标

### 2. Cross-Encoder
$$score = \text{MLP}(\text{BERT}_{CLS}([q;d]))$$
- Query-Doc 联合编码

### 3. Query Likelihood
$$P(q|d) = \prod_{t \in q} P(t|d)$$
- 概率语言模型检索

---

## 🎯 核心洞察（4条）

1. **三大范式各有优劣**：Pointwise（逐个打分，简单但忽略文档间关系）→ Pairwise（文档对比较，平衡）→ Listwise（整列表优化，理论最优但训练复杂）
2. **LambdaMART 仍是非深度学习的 SOTA**：基于 GBDT 的排序模型，在特征工程好的情况下效果可以比肩甚至超过深度模型
3. **BERT Reranker 改变了精排**：将 query-document 拼接输入 BERT 做分类/回归，精度大幅提升但延迟也大幅增加
4. **NDCG 是排序评估的标准指标**：Normalized Discounted Cumulative Gain 综合考虑了相关性级别和位置折扣

---

## 📈 技术演进脉络

```
BM25 启发式排序（~2005）→ RankSVM/RankNet Pairwise（2005-2010）
  → LambdaMART GBDT（2010-2018）→ BERT Reranker（2019-2022）
    → LLM Listwise Reranker（2023+）
```

---

## 🎓 面试考点（5条）

### Q1: Pointwise/Pairwise/Listwise 的区别？
**30秒答案**：Pointwise 将排序转化为回归/分类（预测每个文档的相关性分数）；Pairwise 转化为文档对的比较（A 比 B 更相关？）；Listwise 直接优化整个列表的排序指标（如 NDCG）。Pointwise 最简单，Listwise 理论最优。

### Q2: LambdaMART 的核心思想？
**30秒答案**：在 MART（梯度提升树）基础上，梯度不是 loss 对预测值的导数，而是 λ 梯度——直接反映"交换两个文档位置对 NDCG 的影响"。这样模型直接优化排序指标而非替代 loss。

### Q3: NDCG 的计算方式？
**30秒答案**：`DCG@K = Σ(2^rel_i - 1) / log2(i+1)`，i 是排名位置，rel_i 是相关性标签。NDCG = DCG / IDCG（理想排序的 DCG），归一化到 [0,1]。关键：位置越靠后折扣越大，相关性越高收益越大。

### Q4: BERT Reranker vs 双塔模型？
**30秒答案**：双塔（Bi-Encoder）：query 和 doc 独立编码再内积，快但交互弱；BERT Reranker（Cross-Encoder）：query-doc 拼接后过 BERT，慢但交互充分。通常双塔做召回/粗排，BERT 做精排。

### Q5: 搜索排序的特征类型？
**30秒答案**：①Query 特征（长度、意图类型）；②文档特征（PageRank、长度、新鲜度）；③Query-Doc 交互特征（BM25 分数、term overlap、语义相似度）；④用户特征（历史点击偏好）；⑤上下文特征（设备、时间、地域）。

---


### Q6: 搜索系统的评估指标有哪些？
**30秒答案**：离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。

### Q7: 稠密检索的训练数据构造？
**30秒答案**：正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。

### Q8: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。

### Q9: 向量检索的工程挑战？
**30秒答案**：①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。

### Q10: RAG 系统的常见问题和解决方案？
**30秒答案**：①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。
## 🌐 知识体系连接

- **上游依赖**：GBDT/XGBoost、BERT/Transformer、特征工程
- **下游应用**：搜索引擎排序、电商搜索、RAG 重排序
- **相关 synthesis**：检索三角形深析.md, 推荐系统排序范式演进.md
