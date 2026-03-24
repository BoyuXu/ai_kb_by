# 用户行为序列建模：从 DIN 到 SIM 到 LLM

> 📚 参考文献
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/20260321_gems-breaking-the-long-sequence-barrier-in-generative-recommendation-with-a-multi-stream-decoder.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Gems Long Sequence Generative Rec](../../rec-sys/papers/20260322_gems_long_sequence_generative_rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Spotify Unified Lm Search Rec](../../rec-sys/papers/20260322_spotify_unified_lm_search_rec.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Din-Deep-Interest-Network](../../rec-sys/papers/20260317_din-deep-interest-network.md) — DIN：深度兴趣网络（Deep Interest Network）
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...


> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：DIN, DIEN, BST, SIM, ETA, GEMs, HSTU 系列

---

## 🎯 核心洞察（4条）

1. **用户行为序列是推荐最有价值的信号**：比用户画像（年龄/性别）信息密度高 10x+，行为序列包含了用户的真实兴趣和意图变化
2. **Target Attention 是序列建模的核心技术**：不同于 NLP 的 self-attention，推荐中用候选物品（target item）作为 query 来 attend 用户历史行为，提取"与当前候选最相关的历史信号"
3. **兴趣多样性需要多向量表示**：用户不是"一个人"而是"多个兴趣的集合"——喜欢数码也喜欢美食，MIND/ComiRec 用多个向量分别表示不同兴趣
4. **行为序列长度从 50→5000+ 是不可逆趋势**：更长的序列 = 更全面的用户理解，但计算效率是核心挑战

---

## 📈 技术演进脉络

```
平均池化用户行为（~2016）
  → DIN target attention（2018，~50 行为）
    → DIEN GRU+兴趣演化（2019，~100 行为）
      → BST Transformer 序列建模（2019，~150 行为）
        → MIND 多兴趣向量（2019，召回场景）
          → SIM 两阶段超长序列（2020，5000+ 行为）
            → HSTU 沙漏 Transformer（2023，10000+ 行为）
              → LLM 序列理解（2025+）
```

---

## 🎓 面试考点（6条）

### Q1: DIN vs DIEN vs BST 的核心区别？
**30秒答案**：DIN——target attention 加权历史行为（无序列建模）；DIEN——GRU 建模兴趣演化序列（捕捉兴趣漂移）；BST——Transformer self-attention 捕捉行为间依赖关系（最强但最耗算力）。
**追问方向**：三者哪个工业最常用？答：DIN 变体最多（简单有效），BST 在算力允许时效果最好。

### Q2: 多兴趣模型（MIND/ComiRec）的设计思路？
**30秒答案**：用户兴趣不是单一向量——MIND 用 capsule network 将行为聚类为 K 个兴趣向量；ComiRec 用可控多兴趣提取，K 个向量分别检索后合并。召回阶段特别有效。
**追问方向**：K 怎么确定？答：通常 4-8，太少兴趣覆盖不全，太多冗余且检索成本线性增长。

### Q3: SIM 怎么解决超长序列的计算问题？
**30秒答案**：两阶段——①General Search Unit (GSU)：用简单规则（类目匹配+时间衰减）从 5000+ 行为中选 top-200 相关行为；②Exact Search Unit (ESU)：对选出的 200 条行为做标准 target attention。
**追问方向**：GSU 的筛选规则会丢失重要行为吗？答：可能。ETA 用 SimHash 近似最近邻代替规则筛选，减少信息丢失。

### Q4: 行为序列中的时间信息怎么建模？
**30秒答案**：①时间间隔编码：将行为间的时间间隔离散化为 Embedding（<1h, 1-6h, 6-24h, 1-7d, >7d）；②时间衰减：近期行为权重 = exp(-λ×time_gap)；③位置编码：用行为发生的绝对时间/相对顺序做位置编码。
**追问方向**：推荐中的位置编码和 NLP 有什么不同？答：推荐中时间间隔不均匀（不像文本 token 等距），所以时间间隔编码比固定位置编码更适合。

### Q5: 实时行为 vs 历史行为的融合？
**30秒答案**：实时行为（最近 10 分钟）通过 Flink 流式计算更新特征，历史行为从离线存储读取。融合方式：①拼接（实时特征 concat 历史 embedding）；②gate 机制（动态决定实时 vs 历史的权重）。

### Q6: LLM 在用户行为建模中的角色？
**30秒答案**：①行为序列理解：将用户行为转化为自然语言描述（"用户最近看了3部科幻电影"），LLM 提取深层意图；②行为生成：预测用户下一步可能的行为序列。
**追问方向**：LLM 处理行为序列的延迟问题？答：离线生成用户兴趣标签（语义特征），在线作为额外特征输入到传统模型。

---

## 🌐 知识体系连接

- **上游依赖**：Attention 机制、RNN/Transformer、实时特征系统
- **下游应用**：CTR/CVR 预估、召回、用户画像
- **相关 synthesis**：std_rec_ranking_evolution.md, std_cross_long_sequence.md, std_rec_feature_engineering.md
