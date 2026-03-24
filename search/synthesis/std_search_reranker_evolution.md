# 搜索 Reranker 演进：从 LambdaMART 到 LLM

> 📚 参考文献
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-a-unified-evaluation-framework-for-large-scale-product-search.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/20260321_query-as-anchor-scenario-adaptive-user-representation-via-large-language-model-for-search.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../search/papers/20260321_intent-aware-neural-query-reformulation-for-behavior-aligned-product-search.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../search/papers/20260323_generative_query_expansion_for_e-commerce_search_a.md) — Generative Query Expansion for E-Commerce Search at Scale
> - [Colbert V3 Efficient Neural Retrieval With Late...](../../search/papers/20260323_colbert_v3_efficient_neural_retrieval_with_late_int.md) — ColBERT v3: Efficient Neural Retrieval with Late Interaction
> - [Document Re-Ranking With Llm From Listwise To Pair](../../search/papers/20260323_document_re-ranking_with_llm_from_listwise_to_pair.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dllm-Searcher-Adapting-Diffusion-Large-Language...](../../search/papers/20260321_dllm-searcher-adapting-diffusion-large-language-model-for-search-agents.md) — DLLM-Searcher: Adapting Diffusion Large Language Model fo...


> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：monoT5, RankGPT, ColBERT Reranker, Cross-Encoder 系列

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

## 🌐 知识体系连接

- **上游依赖**：BERT/LLM、Knowledge Distillation
- **下游应用**：搜索引擎精排、RAG 系统、电商搜索
- **相关 synthesis**：std_search_retrieval_triangle.md, std_search_learning_to_rank.md
