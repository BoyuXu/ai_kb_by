# 系统设计面试：推荐/搜索/广告系统架构要点

> 📚 参考文献
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/20260321_query-as-anchor-scenario-adaptive-user-representation-via-large-language-model-for-search.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-a-unified-evaluation-framework-for-large-scale-product-search.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Tbgrecall A Generative Retrieval Model For E-Co...](../../ads/papers/20260323_tbgrecall_a_generative_retrieval_model_for_e-commer.md) — TBGRecall: A Generative Retrieval Model for E-commerce Re...
> - [Counterfactual-Learning-For-Unbiased-Ad-Ranking...](../../ads/papers/20260321_counterfactual-learning-for-unbiased-ad-ranking-in-industrial-search-systems.md) — Counterfactual Learning for Unbiased Ad Ranking in Indust...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dllm-Searcher-Adapting-Diffusion-Large-Language...](../../search/papers/20260321_dllm-searcher-adapting-diffusion-large-language-model-for-search-agents.md) — DLLM-Searcher: Adapting Diffusion Large Language Model fo...


> 创建：2026-03-24 | 领域：跨域 | 类型：综合分析
> 来源：工业实践, System Design Interview 系列

---

## 🎯 核心洞察（4条）

1. **三大系统的架构高度相似**：都是"召回→粗排→精排→重排"的漏斗结构，区别在于每层的模型选择和业务约束
2. **延迟预算是架构设计的硬约束**：搜索（<200ms）、推荐（<100ms）、广告（<50ms for DSP），每层的延迟分配决定了模型复杂度上限
3. **特征服务是最容易出问题的环节**：训练/推理特征不一致、实时特征延迟、特征存储容量是三大工程痛点
4. **离线/近线/在线三层计算架构**：离线（天级，全量训练+全量特征预计算）→ 近线（分钟级，实时特征更新+模型增量更新）→ 在线（毫秒级，模型推理+出价决策）

---

## 🎓 面试考点（6条）

### Q1: 设计一个推荐系统，你怎么分层？
**30秒答案**：①召回层（<5ms，多路召回取 top-3000）；②粗排（<10ms，轻量模型筛到 300）；③精排（<30ms，复杂深度模型排到 50）；④重排（<5ms，多样性+业务规则出最终 10-20 条）。总延迟 <100ms。

### Q2: 推荐/搜索系统的实时性要求？
**30秒答案**：①用户行为实时反馈（<5s，点了一个商品后下次刷新推荐应该变化）；②特征实时更新（<1min，实时 CTR 统计）；③模型更新（小时级增量训练 or FTRL 在线学习）。

### Q3: 大规模推荐系统的存储架构？
**30秒答案**：①特征存储：Redis/Memcached（实时特征）+ HBase/Cassandra（用户画像）；②模型存储：Model Server + GPU 集群；③向量索引：Faiss/Milvus（召回向量）；④日志存储：Kafka → Flink → 数据仓库。

### Q4: 搜索系统 vs 推荐系统的架构差异？
**30秒答案**：①搜索有显式 query，推荐没有（用用户画像+上下文代替）；②搜索的召回强调"相关性"，推荐强调"个性化"；③搜索有 Query 理解模块（纠错/NER/意图分类），推荐没有；④搜索的排序更注重相关性，推荐更注重点击率。

### Q5: 如何处理系统的高可用和降级？
**30秒答案**：①模型服务降级：复杂模型超时 → 回退到简单模型/规则；②特征降级：Redis 不可用 → 用默认特征；③召回降级：个性化召回失败 → 热门/编辑推荐兜底；④限流：QPS 过高时降级到缓存结果。

### Q6: 日请求量 10 亿的推荐系统需要多少机器？
**30秒答案**：假设 10 亿请求/天 ≈ 12K QPS。精排用 GPU（A100 每卡 ~1K QPS）→ 需要 ~12 台 A100；召回用 CPU（每台 ~2K QPS）→ 需要 ~6 台。加上冗余和各层，总计约 50-100 台服务器。

---

## 🌐 知识体系连接

- **上游依赖**：分布式系统、GPU/CPU 架构、缓存/存储系统
- **下游应用**：面试系统设计题、架构方案评审
- **相关 synthesis**：std_rec_ranking_evolution.md, std_ads_rtb_architecture.md, std_search_retrieval_triangle.md
