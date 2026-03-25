# 搜索 Reranker 演进：从 LambdaMART 到 LLM

> 📚 参考文献
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/Query_as_Anchor_Scenario_Adaptive_User_Representation_via.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../search/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../search/papers/Generative_Query_Expansion_for_E_Commerce_Search_at_Scale.md) — Generative Query Expansion for E-Commerce Search at Scale
> - [Colbert V3 Efficient Neural Retrieval With Late...](../../search/papers/ColBERT_v3_Efficient_Neural_Retrieval_with_Late_Interacti.md) — ColBERT v3: Efficient Neural Retrieval with Late Interaction
> - [Document Re-Ranking With Llm From Listwise To Pair](../../search/papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dllm-Searcher-Adapting-Diffusion-Large-Language...](../../search/papers/DLLM_Searcher_Adapting_Diffusion_Large_Language_Model_for.md) — DLLM-Searcher: Adapting Diffusion Large Language Model fo...


> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：monoT5, RankGPT, ColBERT Reranker, Cross-Encoder 系列


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

1. **Reranker 是搜索精度的"最后一公里"**：在 top-100 候选上重排序，NDCG@10 提升 5-15%，投入产出比极高
2. **Cross-Encoder 是精度标杆**：query-doc 拼接输入 BERT 做全交互，精度最高但延迟大（每个 doc 需要一次 BERT 推理）
3. **LLM Reranker 正在崛起**：RankGPT 用 GPT-4 做 listwise 重排序（一次性排序整个列表），在多个基准上超过 Cross-Encoder
4. **蒸馏是 Reranker 落地的关键**：用 LLM/大 Cross-Encoder 作为 teacher，蒸馏到小模型（如 MiniLM）做线上推理

---

## 🎓 面试考点（4条）

### Q1: Cross-Encoder Reranker 和 Bi-Encoder 的本质区别？
**30秒答案**：Bi-Encoder 将 query 和 doc 独立编码为向量再算内积（交互少但快）；Cross-Encoder 将 [query; doc] 拼接后过整个 Transformer（全交互但慢 100x）。Bi-Encoder 做召回，Cross-Encoder 做精排/重排。

### Q2: 怎么控制 Reranker 延迟？
**30秒答案**：①限制重排候选数（top-50~100）；②模型蒸馏（大模型→小模型，如 BERT-base→MiniLM）；③INT8 量化；④GPU batch 推理。目标延迟 <50ms。

### Q3: LLM Reranker 的两种范式？
**30秒答案**：①Pointwise：逐个文档打分（"这个文档与 query 相关吗？评分 1-10"）；②Listwise：一次性排序（"将这 20 个文档按相关性排序"）。Listwise 更强但受 context length 限制。

### Q4: Reranker 训练数据怎么构建？
**30秒答案**：①人工标注（query-doc 相关性标签 0-3 级）；②蒸馏标签（用 LLM 打分作为 soft label）；③点击数据（搜索日志中被点击的 doc 为正例）；④Hard Negative Mining（检索到但未被点击的作为难负例）。

---


### Q5: 搜索系统的评估指标有哪些？
**30秒答案**：离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。

### Q6: 稠密检索的训练数据构造？
**30秒答案**：正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。

### Q7: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。

### Q8: 向量检索的工程挑战？
**30秒答案**：①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。

### Q9: RAG 系统的常见问题和解决方案？
**30秒答案**：①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。

### Q10: E5 和 BGE 嵌入模型的区别？
**30秒答案**：E5（微软）：通用文本嵌入，支持 instruct 前缀。BGE-M3（BAAI）：多语言+多粒度+多功能（dense+sparse+ColBERT 三合一）。BGE-M3 更全面但模型更大。
## 🌐 知识体系连接

- **上游依赖**：BERT/LLM、Knowledge Distillation
- **下游应用**：搜索引擎精排、RAG 系统、电商搜索
- **相关 synthesis**：检索三角形深析.md, LearningToRank搜索排序三大范式.md
