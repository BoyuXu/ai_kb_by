# 生成式范式在推荐/广告/搜索中的统一视角

> 📚 参考文献
> - [Action Is All You Need Dual-Flow Generative Ran...](../../ads/papers/20260323_action_is_all_you_need_dual-flow_generative_ranking.md) — Action is All You Need: Dual-Flow Generative Ranking Netw...
> - [Diffgrm Diffusion Generative Rec](../../rec-sys/papers/20260322_diffgrm_diffusion_generative_rec.md) — DiffGRM: Diffusion-based Generative Recommendation Model
> - [Bm25-Semantic-Hybrid-Retrieval](../../search/papers/20260316_bm25-semantic-hybrid-retrieval.md) — BM25 与语义检索融合：Hybrid Retrieval 最佳实践
> - [Tbgrecall A Generative Retrieval Model For E-Co...](../../ads/papers/20260323_tbgrecall_a_generative_retrieval_model_for_e-commer.md) — TBGRecall: A Generative Retrieval Model for E-commerce Re...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Diffgrm-Diffusion-Based-Generative-Recommendati...](../../rec-sys/papers/20260321_diffgrm-diffusion-based-generative-recommendation-model.md) — DiffGRM: Diffusion-based Generative Recommendation Model
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/20260321_deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...


> 创建：2026-03-24 | 领域：跨域 | 类型：综合分析
> 来源：TIGER, ActionPiece, DiffGRM, Generative Retrieval, TBGRecall

---

## 🎯 核心洞察（5条）

1. **三个领域同时经历"判别→生成"范式转换**：推荐从"打分排序"到"生成下一个物品"，搜索从"匹配检索"到"生成答案"，广告从"竞价排序"到"生成广告创意"
2. **Semantic ID 是连接三个领域的桥梁**：物品/文档/广告都可以编码为 Semantic Token 序列，统一的自回归框架就能处理推荐、检索、广告匹配
3. **自回归 vs 扩散是两条技术路线**：AR（TIGER/ActionPiece）逐 token 生成，优势是成熟可控；Diffusion（DiffGRM）并行去噪，优势是全局建模
4. **落地挑战集中在延迟和可控性**：生成式方法推理延迟 10-100x 于传统方法，且输出不完全可控（可能生成不存在的 ID）
5. **LLM 重塑了生成式推荐/搜索的可能性**：大语言模型天然具备生成能力，微调后可以直接做推荐（prompt："根据用户历史推荐"）和搜索（生成回答+引用）

---

## 📈 技术演进脉络

```
推荐：CF/打分排序 → 双塔向量召回 → Semantic ID 生成式召回（TIGER 2023）→ 扩散生成（DiffGRM 2025）
搜索：BM25/向量检索 → RAG（检索+生成）→ 生成式搜索（直接生成文档ID，2024+）
广告：eCPM 竞价排序 → 生成式 CTR 预估 → 生成式广告创意（2025+）
```

---

## 🔗 跨文献共性规律

| 规律 | 推荐 | 搜索 | 广告 |
|------|------|------|------|
| Semantic ID/Token | 物品→RQ-VAE→token | 文档→BERT→token | 广告→CLIP→token |
| 生成方式 | 自回归/扩散 | RAG/生成式检索 | 生成式CTR/创意生成 |
| 延迟挑战 | 高（需 speculative decoding） | 高（需缓存/蒸馏） | 极高（实时竞价不可用） |
| 当前阶段 | 实验→初步上线 | RAG 已普及 | 概念验证 |

---

## 🎓 面试考点（5条）

### Q1: 判别式 vs 生成式推荐的核心区别？
**30秒答案**：判别式——给每个 user-item pair 打分，从候选集中选最高分；生成式——直接"生成"推荐结果（item 的 Semantic ID 序列），不需要预定义候选集。
**追问方向**：生成式推荐能解决什么判别式不能的问题？答：长尾物品覆盖（不需要在 ANN 索引中被检索到）、跨模态推荐（统一文本/图像/视频的生成框架）。

### Q2: Semantic ID 的生成方式？
**30秒答案**：①RQ-VAE（Residual Quantization）：将物品 embedding 量化为多级 codebook token [C1, C2, C3]，层级越深越精细；②BSQ（Binary Scalar Quantization）：二值化编码。
**追问方向**：Semantic ID 和传统 Item ID 的区别？答：传统 ID 是随机分配的数字，没有语义；Semantic ID 编码了物品的内容语义，相似物品的 ID 相近。

### Q3: 自回归 vs 扩散模型在推荐中的对比？
**30秒答案**：AR（TIGER）：逐 token 生成，与 LLM 兼容，可控性强，但有 exposure bias 和单向依赖限制。Diffusion（DiffGRM）：并行生成所有 token，无单向依赖，但可控性差、推理需要多步去噪。
**追问方向**：哪个更有前景？答：短期看 AR（与 LLM 生态兼容），长期看可能融合（先扩散生成粗方案，再 AR 精细化）。

### Q4: RAG 和生成式搜索的关系？
**30秒答案**：RAG = 检索（传统方法找相关文档）+ 生成（LLM 合成答案），是"检索辅助生成"。生成式搜索 = 模型直接生成答案和来源引用，不需要传统检索环节。RAG 是过渡方案，生成式搜索是终态。
**追问方向**：为什么现在还用 RAG 而非完全生成式？答：LLM 幻觉问题未解决，检索提供的事实锚定仍不可或缺。

### Q5: 生成式广告的可行性？
**30秒答案**：①生成式 CTR 预估：用 LLM 理解广告语义直接预测 CTR，冷启动效果好但延迟高；②生成式广告创意：LLM 根据产品信息自动生成文案/图片，降低广告主门槛；③当前阶段：创意生成已商用，CTR 预估仍在实验。

---

## 🌐 知识体系连接

- **上游依赖**：自回归模型、扩散模型、Semantic ID/RQ-VAE
- **下游应用**：下一代推荐引擎、对话式搜索、智能广告投放
- **相关 synthesis**：std_rec_recall_evolution.md, std_search_hybrid_retrieval.md, std_ads_cold_start.md
