# 搜索系统知识库导航 🔍

## 📊 领域概览

| 分类 | 文档数 | 描述 |
|------|--------|------|
| **Papers** (学术论文笔记) | 72篇 | arXiv 论文、学术研究、算法基础 |
| **Practices** (工业实践案例) | 0篇 | 大厂系统设计、生产案例（待补充） |
| **Synthesis** (提炼总结) | 2篇 | 搜索系统演进、框架总结 |
| **总计** | 74篇 | - |

---

## 🚀 快速导航

### 📚 按学习阶段查找

1. **新手入门** → [搜索排序系统概览](./synthesis/01_search_ranking.md)
2. **深度学习基础** → [密集检索 (DPR)](./papers/20260316_dpr-dense-retrieval.md) | [稀疏检索 (BM25)](./papers/20260316_bm25-semantic-hybrid-retrieval.md)
3. **LLM 时代前沿** → [LLM for IR 综述](./papers/20260316_llm-for-ir-survey.md) | [RAG 检索优化](./papers/20260316_rag-retrieval-optimization.md)
4. **系统架构** → [LLM 集成框架](./synthesis/llm_integration_framework.md)

### 🎯 按研究方向查找

#### 💾 检索 (Retrieval)
- **基础** 
  - [BM25 & 混合检索](./papers/20260316_bm25-semantic-hybrid-retrieval.md)
  - [Dense Passage Retrieval (DPR)](./papers/20260316_dpr-dense-retrieval.md)
  - [Multi-Lingual Embeddings (BGE-M3)](./papers/20260320_BGE-M3-Multi-Lingual-Embeddings.md)
  
- **最新进展**
  - [SPLADE v3 - 稀疏检索新突破](./papers/20260322_splade_v3_sparse_retrieval.md)
  - [E5-Mistral 7B - 双语嵌入](./papers/20260323_e5-mistral-7b-instruct_improving_text_embeddings_wi.md)
  - [GTE - 通用文本嵌入](./papers/20260323_gte_towards_general_text_embeddings_with_multi-stag.md)

#### 🔄 排序 & 重排 (Ranking & Reranking)
- [ColBERT v2 - 后期交互检索](./papers/20260313_colbert_v2.md)
- [LLM 重排与排序](./papers/20260323_rankllm_reranking_with_large_language_models.md)
- [缩放律与重排](./papers/20260318_ScalingLawsReranking.md)
- [基于扩散模型的重排](./papers/20260321_dllm-searcher-adapting-diffusion-large-language-model-for-search-agents.md)

#### 🧠 图神经网络 & 图检索
- [GRIT - 图召回](./papers/20260313_grit_graph_recall.md)
- [RETE - 时间图](./papers/20260313_rete_temporal_graph.md)
- [知识图谱增强搜索](./papers/20260323_knowledge_graph_enhanced_product_search_bridging_s.md)

#### 🌍 多模态 & 地理感知
- [多模态文档检索综述](./papers/20260317_multimodal-document-retrieval-survey.md)
- [地理感知嵌入](./papers/20260322_location_aware_embedding_geotargeting.md)
- [视频检索 (Landmark)](./papers/20260316_llandmark-multimodal-video-retrieval.md)

#### 🤖 LLM & RAG 集成
- [RAG 检索优化](./papers/20260316_rag-retrieval-optimization.md)
- [LLM for IR 综述](./papers/20260316_llm-for-ir-survey.md)
- [查询改写与优化](./papers/20260319_rewritegen-query-optimization-rl.md)
- [多智能体 RAG](./papers/20260319_ma-rag-multi-agent-cot-reasoning.md)
- [意图感知查询改写](./papers/20260321_intent-aware-neural-query-reformulation-for-behavior-aligned-product-search.md)

#### 📍 产品搜索 & 场景化
- [产品搜索意图识别](./papers/20260321_intent-aware-neural-query-reformulation-for-behavior-aligned-product-search.md)
- [查询作为锚点 (用户表示)](./papers/20260322_query_as_anchor_llm_user_repr.md)
- [生成式搜索引擎优化](./papers/geo_generative_engine_optimization.md)

#### 🔧 工程优化
- [BM25S - 快速稀疏搜索](./papers/20260323_bm25s_orders_of_magnitude_faster_lexical_search_via.md)
- [ColBERT Serve - 内存映射评分](./papers/20260320_ColBERT-serve-Efficient-Memory-Mapped-Scoring.md)
- [FlashRAG - 检索工具包](./papers/20260323_flashrag_a_modular_toolkit_for_efficient_retrieval-.md)

#### 📋 特殊场景
- [法律文本检索 (LegalMALR)](./papers/20260317_legalmalr-chinese-statute-retrieval.md)
- [WSDM 2026 多语言赛](./papers/20260317_naver-wsdm-2026-multilingual.md)
- [Tip-of-Tongue 检索](./papers/20260318_TipOfTongueRetrieval.md)

---

## 📚 完整文档列表

### 📄 Papers (72篇 学术论文笔记)

#### 基础与综述
- [BM25 & 语义混合检索](./papers/20260316_bm25-semantic-hybrid-retrieval.md)
- [DPR - 密集通道检索](./papers/20260316_dpr-dense-retrieval.md)
- [LLM for IR 综述](./papers/20260316_llm-for-ir-survey.md)
- [多模态文档检索综述](./papers/20260317_multimodal-document-retrieval-survey.md)
- [多模态视觉文档检索综述](./papers/20260319_multimodal-visual-document-retrieval-survey.md)

#### 检索方法与模型
- [稀疏检索 vs 密集检索 统一评估](./papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md)
- [SPLADE v3 - 稀疏检索新基准](./papers/20260323_splade-v3_new_baselines_for_splade.md)
- [SPLADE v3 - 推进稀疏检索](./papers/20260321_splade-v3-advancing-sparse-retrieval-with-deep-language-models.md)
- [E5-Mistral 7B - 文本嵌入改进](./papers/20260323_e5-mistral-7b-instruct_improving_text_embeddings_wi.md)
- [GTE - 通用文本嵌入多阶段](./papers/20260323_gte_towards_general_text_embeddings_with_multi-stag.md)
- [BGE-M3 - 多语言嵌入](./papers/20260320_BGE-M3-Multi-Lingual-Embeddings.md)
- [密集检索对话搜索](./papers/20260320_Dense-Passage-Retrieval-Conversational-Search.md)
- [LeSeR - 词法语义检索](./papers/20260320_LeSeR-Lexical-Semantic-Retrieval.md)

#### 排序与重排
- [ColBERT v2 - 后期交互检索](./papers/20260313_colbert_v2.md)
- [ColBERT v3 - 高效后期交互](./papers/20260323_colbert_v3_efficient_neural_retrieval_with_late_int.md)
- [RankLLM - LLM 重排](./papers/20260323_rankllm_reranking_with_large_language_models.md)
- [LLM 文档重排 - 列表到成对](./papers/20260323_document_re-ranking_with_llm_from_listwise_to_pair.md)
- [缩放律与重排](./papers/20260318_ScalingLawsReranking.md)
- [密集 vs 稀疏检索评估](./papers/20260322_dense_vs_sparse_retrieval_eval.md)

#### 图与高级检索
- [GRIT - 图召回](./papers/20260313_grit_graph_recall.md)
- [RETE - 时间图](./papers/20260313_rete_temporal_graph.md)
- [知识图谱增强产品搜索](./papers/20260323_knowledge_graph_enhanced_product_search_bridging_s.md)

#### 多模态与地理
- [Landmark 多模态视频检索](./papers/20260316_llandmark-multimodal-video-retrieval.md)
- [LMK-CLS - Landmark 池化](./papers/20260318_LandmarkPooling.md)
- [Location-Aware 嵌入 地理定向](./papers/20260321_location-aware-embedding-for-geotargeting-in-sponsored-search-advertising.md)
- [地理定向位置感知嵌入](./papers/20260322_location_aware_embedding_geotargeting.md)
- [混合搜索 LLM 重排](./papers/20260320_Hybrid-Search-LLM-Re-ranking.md)

#### LLM & RAG
- [RAG 检索优化](./papers/20260316_rag-retrieval-optimization.md)
- [多模态搜索](./papers/20260316_multimodal-search.md)
- [查询改写生成 & RL 优化](./papers/20260319_rewritegen-query-optimization-rl.md)
- [多智能体 RAG - CoT 推理](./papers/20260319_ma-rag-multi-agent-cot-reasoning.md)
- [SUNAR - 语义不确定性检索](./papers/20260319_sunar-semantic-uncertainty-retrieval.md)
- [DRAMA - 多样化数据增强密集检索](./papers/20260319_drama-diverse-augmentation-dense-retrieval.md)
- [LURE - RAG 重排](./papers/20260317_lure-rag-reranking.md)
- [W-RAG - 弱监督密集检索](./papers/20260320_W-RAG-Weakly-Supervised-Dense-Retrieval.md)
- [LEGALMALR - 中文法律检索](./papers/20260317_legalmalr-chinese-statute-retrieval.md)
- [LegalMALR - 多智能体中文法律检索](./papers/20260319_legalmalr-multi-agent-chinese-statute-retrieval.md)
- [LEGALMALR - 中文法规检索](./papers/20260319_legalmalr-chinese-statute-retrieval.md)

#### 产品与应用场景
- [意图感知查询改写 - 行为对齐产品搜索](./papers/20260321_intent-aware-neural-query-reformulation-for-behavior-aligned-product-search.md)
- [查询作为锚点 场景自适应用户表示](./papers/20260321_query-as-anchor-scenario-adaptive-user-representation-via-large-language-model-for-search.md)
- [意图感知神经查询改写](./papers/20260322_intent_aware_query_reformulation.md)
- [查询作为锚点 LLM 用户表示](./papers/20260322_query_as_anchor_llm_user_repr.md)
- [生成式查询扩展 电子商务搜索](./papers/20260323_generative_query_expansion_for_e-commerce_search_a.md)
- [统一生成搜索和推荐](./papers/20260323_sparse_meets_dense_unified_generative_recommendatio.md)
- [单一搜索初步探索](./papers/20260323_onesearch_a_preliminary_exploration_of_the_unified_.md)

#### 提示与辅助
- [提示提示技巧](./papers/20260318_LURE_RAG.md)
- [Tip-of-Tongue 检索](./papers/20260318_TipOfTongueRetrieval.md)
- [ColBandit - 零-shot 剪枝](./papers/20260317_col-bandit-zero-shot-pruning.md)
- [LLM 用户表示](./papers/20260320_Dense-Passage-Retrieval-Conversational-Search.md)

#### 数据与合成
- [合成数据重排](./papers/20260313_synthetic_data_reranker.md)
- [提示增强重排](./papers/20260313_hint_augmented_reranking.md)
- [大规模推理嵌入](./papers/20260313_large_reasoning_embedding.md)

#### 竞赛与最新
- [NAVER WSDM 2026 多语言](./papers/20260317_naver-wsdm-2026-multilingual.md)
- [Wukong - 大规模推荐缩放律](./papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md)
- [FlashRAG - 模块化高效检索工具包](./papers/20260323_flashrag_a_modular_toolkit_for_efficient_retrieval_.md)

#### 工程优化
- [BM25S - 快速词法搜索](./papers/20260323_bm25s_orders_of_magnitude_faster_lexical_search_via.md)
- [ColBERT Serve - 高效内存映射评分](./papers/20260320_ColBERT-serve-Efficient-Memory-Mapped-Scoring.md)
- [DLLM Searcher - 扩散 LLM 搜索](./papers/20260321_dllm-searcher-adapting-diffusion-large-language-model-for-search-agents.md)
- [DLLM Searcher 扩散 LLM](./papers/20260322_dllm_searcher_diffusion_llm_search.md)

#### 其他
- [搜索算法知识库](./papers/搜索算法知识库.md)
- [LLM 项目创意](./papers/llm_project_ideas.md)
- [地理生成引擎优化](./papers/geo_generative_engine_optimization.md)

### 🏢 Practices (0篇 工业实践案例 - 待补充)

当前知识库缺少大厂工业实践案例，建议补充：
- Google Search System（AI Overview）
- Perplexity AI Architecture
- ChatGPT Browsing 功能
- DuckDuckGo LLM 集成
- Meta Search Infrastructure
- Alibaba 搜索系统

### 📖 Synthesis (2篇 提炼总结)

- [搜索排序系统概览](./synthesis/01_search_ranking.md) - 搜索系统从传统到 LLM 的演进
- [LLM 集成框架](./synthesis/llm_integration_framework.md) - LLM 在搜索中的应用框架

---

## 💡 使用指南

### 对于不同角色

**🎓 学生/初学者**
1. 先读 [搜索排序系统概览](./synthesis/01_search_ranking.md)
2. 理解基础概念（BM25、DPR、排序）
3. 深入学习感兴趣的方向

**🔬 研究者**
1. 按研究方向查找 Papers
2. 对比最新论文（2026年最新）
3. 理解缩放律、LLM 集成等前沿

**👨‍💼 工程师**
1. 查看 [LLM 集成框架](./synthesis/llm_integration_framework.md)
2. 参考工业实践案例（待补充）
3. 实现系统组件

**🤖 LLM 时代**
1. 重点阅读 [LLM for IR 综述](./papers/20260316_llm-for-ir-survey.md)
2. 理解 RAG、重排、查询改写
3. 学习 LLM 系统集成

---

## 🔗 关键论文快速链接

| 方向 | 关键论文 |
|------|---------|
| **基础检索** | [DPR](./papers/20260316_dpr-dense-retrieval.md) \| [BM25](./papers/20260316_bm25-semantic-hybrid-retrieval.md) |
| **排序重排** | [ColBERT v3](./papers/20260323_colbert_v3_efficient_neural_retrieval_with_late_int.md) \| [RankLLM](./papers/20260323_rankllm_reranking_with_large_language_models.md) |
| **LLM集成** | [LLM for IR](./papers/20260316_llm-for-ir-survey.md) \| [RAG优化](./papers/20260316_rag-retrieval-optimization.md) |
| **多模态** | [多模态综述](./papers/20260317_multimodal-document-retrieval-survey.md) |
| **前沿** | [Wukong缩放律](./papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md) |

---

## 📈 知识树状图

```
搜索系统
├── 🔍 检索 (Retrieval)
│   ├── 稀疏 (BM25, SPLADE)
│   ├── 密集 (DPR, E5, GTE, BGE)
│   └── 混合 (Hybrid Search)
│
├── 🔄 排序 (Ranking)
│   ├── 传统排序
│   ├── 神经排序 (ColBERT)
│   └── LLM 重排
│
├── 🧠 高级特性
│   ├── 图检索 (GRIT, RETE)
│   ├── 多模态 (Landmark, 视频)
│   └── 地理感知
│
└── 🤖 LLM 时代
    ├── RAG 系统
    ├── 查询改写
    ├── 多智能体
    └── 场景化应用
```

---

## 📝 最后更新

- **最后更新**: 2026-03-24
- **总文档数**: 74 篇
- **近期更新**: 添加 2026 年最新论文（WSDM、稀疏检索突破）

> 💡 **提示**: 点击侧边栏快速导航不同研究方向！
