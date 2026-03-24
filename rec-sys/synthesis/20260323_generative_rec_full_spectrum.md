# 生成式推荐的完整技术图谱（2024-2025）
> 📚 参考文献
> - [Spotify Unified Lm Search Rec](../../rec-sys/papers/20260322_spotify_unified_lm_search_rec.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [Gpr Generative Personalized Recommendation With E](../../rec-sys/papers/20260323_gpr_generative_personalized_recommendation_with_e.md) — GPR: Generative Personalized Recommendation with End-to-E...
> - [Spotify Semantic Id Podcast](../../rec-sys/papers/20260322_spotify_semantic_id_podcast.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Mtgrboost Boosting Large-Scale Generative Recommen](../../rec-sys/papers/20260323_mtgrboost_boosting_large-scale_generative_recommen.md) — MTGRBoost: Boosting Large-scale Generative Recommendation...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/20260321_deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify.md) — Deploying Semantic ID-based Generative Retrieval for Larg...


> 知识卡片 | 创建：2026-03-23 | 领域：rec-sys / ads

---

**一句话**：生成式推荐把「从候选集里挑」变成「直接生成推荐列表」，今天的论文展示了这一范式已从召回扩展到排序、再到推理——整条链路都在被生成式重写。

**类比**：传统推荐像「超市货架选货」（先放货→再让用户挑），生成式推荐像「私人订制配送」（厨师直接做出你想要的菜单），OneRec-Think 更像「厨师先想想你的饮食喜好再下单」。

---

## 核心机制：生成式推荐进化三阶段

```
阶段 1 - 生成式召回（2022-2023）
    TBGRecall / TIGER / GPR
    │  做什么：用 Semantic ID 替代 item ID，用 LLM 直接「说出」推荐物品
    │  核心：RQ-VAE 对物品编码 → LLM 解码出物品 ID token 序列
    └─ 局限：只替换召回阶段，排序还用传统模型

阶段 2 - 统一召回+排序（2024）
    OneRec / EteGrec / MTGRBoost
    │  做什么：单一生成模型同时完成召回和排序
    │  核心：Encoder-Decoder + Constrained Beam Search 保证物品合法
    │  快手实践：OneRec 直接从百亿候选中生成 Top-K，无独立排序阶段
    └─ 局限：推理 latency 高（比 pointwise 慢 10-100x）

阶段 3 - 生成式推荐 + 推理（2025）
    OneRec-Think / Reg4Rec / PinRec
    │  做什么：生成列表的同时产生可读推理链 <think>...</think>
    │  核心：推理监督信号来自用户满意度反馈（用户行为反向验证推理质量）
    │  快手数据：用户留存 +0.4%，冷启动用户 +1.2%；推理最优长度 50-100 token
    └─ 当前天花板：推理 token 增加 ~30% 计算开销
```

---

## 横向对比：今日 5 种生成式推荐变体

| 模型 | 核心创新 | 适用场景 | 关键数字 |
|------|---------|---------|---------|
| TBGRecall | 树状 beam search 约束召回 | 电商召回 | Recall@50 +8% |
| GPR | 个性化 RQ-VAE 编码 | 长尾物品召回 | 冷启动 +15% |
| COBRA | 稀疏+稠密联合生成 | 混合检索场景 | NDCG +6% |
| OneRec | 召回排序一体化 | 短视频（快手） | Recall@K +5-8% |
| OneRec-Think | 推理增强生成 | 复杂意图理解 | 留存 +0.4% |

---

## 工业落地桥梁

**论文假设 vs 工业现实**：
- 论文：生成式推荐可以完全替代传统 pipeline
- 工业：通常先替换召回层，排序仍用 LTR（因 latency SLA 严格）
- 快手是少数生产环境全生成式的：OneRec 支撑主要流量

**工程核心问题**：
1. **Latency**：自回归解码 O(K × L) 步，K=列表长度，L=每物品token数 → 用 Speculative Decoding / CUDA 优化
2. **合法性**：生成的 token 可能不是真实物品 → Trie/前缀树约束 Beam Search
3. **新物品覆盖**：新上架物品没有 Semantic ID → 每日重建 RQ-VAE 索引（Spotify 方案）
4. **推理内容安全**：OneRec-Think 的推理链需内容审核层（避免推理暴露隐私/偏见）

---

## 技术演进脉络

```
BERT4Rec (2019) → 序列推荐，掩码预测
    ↓
SASRec (2018-2020) → Self-Attention 序列推荐
    ↓
P5 (2022) → 统一 NLP 框架，文本输入输出
    ↓
TIGER / TBGRecall (2023) → Semantic ID + 生成式召回
    ↓
OneRec (2024) → 端到端统一召回排序
    ↓
OneRec-Think (2025) → 推理增强，用户意图深度理解
    ↓（预测）
Agentic Recommendation → 多轮对话 + 主动澄清意图
```

---

## 面试考点

1. **Q: 生成式推荐和传统协同过滤的本质区别？**
   A: 传统 CF 是「相似度查找」（近邻），生成式是「条件生成」（语言模型），后者可以捕获更复杂的序列依赖和语义

2. **Q: Semantic ID 为什么比 raw item ID 效果好？**
   A: raw ID 是独热，无语义；Semantic ID（RQ-VAE编码）共享 codebook，相似物品有相似 ID 前缀，能泛化到新物品

3. **Q: 生成式推荐中如何防止生成不存在的物品？**
   A: Constrained Beam Search + Trie Tree，只允许已知物品ID的合法token前缀路径

4. **Q: OneRec-Think 的推理为什么对新用户帮助更大？**
   A: 新用户行为数据稀少，推理可以引入常识先验（"新用户喜欢热门内容"），弥补协同过滤信号不足

5. **Q: 快手如何解决 OneRec 的在线延迟问题？**
   A: 多级缓存、批推理、KV Cache 复用、Speculative Decoding，将 P99 延迟控制在 SLA 内
