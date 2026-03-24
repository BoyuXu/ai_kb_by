# 推荐系统召回范式演进：从协同过滤到生成式召回

> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：DSSM, TIGER, ActionPiece, DiffGRM, GEMs, SIM, ULM (Spotify)

---

## 🎯 核心洞察（5条）

1. **召回的核心矛盾是覆盖率 vs 延迟**：从亿级候选中选出万级候选，模型越复杂精度越高但延迟越大，工业系统通常要求 <5ms
2. **多路召回是工业标配**：不存在单一完美的召回方法，实际系统同时使用 4-8 路召回（协同过滤 + 向量召回 + 图召回 + 热门召回 + 规则召回），最后 merge + 去重
3. **生成式召回是下一代范式**：TIGER/ActionPiece 将"检索"转化为"生成"——模型直接自回归生成物品的 Semantic ID，不再需要 ANN 索引
4. **双塔模型仍是主力**：尽管生成式召回概念先进，但双塔（DSSM）+ HNSW/Faiss 的工程成熟度、可解释性、延迟优势使其在 2025 年仍是工业主流
5. **负采样策略决定召回质量**：随机负采样 → 热度加权负采样 → 批内负采样 → 难负例挖掘，采样策略对 AUC 的影响可达 1-3%

---

## 📈 技术演进脉络

```
ItemCF/UserCF（2000s）
  → 矩阵分解 MF/SVD++（2009-2014）
    → 双塔 DSSM/YoutubeDNN（2015-2018）
      → 图召回 PinSage/EGES（2018-2020）
        → 超长序列召回 SIM/ETA（2020-2022）
          → 生成式召回 TIGER/ActionPiece（2023-2025）
            → 扩散模型召回 DiffGRM（2025+）
```

**关键转折点**：
- **DSSM 双塔（2013）**：将召回从"精确匹配"变为"语义匹配"，user/item 各编码为向量，cosine 相似度检索
- **ANN 工程成熟（2017-2019）**：HNSW/Faiss 使亿级向量检索 <1ms 成为现实，双塔 + ANN 成为工业标配
- **Semantic ID（2023）**：TIGER 首次证明可以用自回归模型生成物品 token，召回不再需要 ANN 索引

---

## 🔗 跨文献共性规律

| 规律 | 体现论文/系统 | 说明 |
|------|-------------|------|
| 从匹配到生成的范式转换 | TIGER, ActionPiece | 召回从"找最像的"变为"生成想要的" |
| 离线计算 + 在线检索的分离 | 双塔, 图召回 | item embedding 离线算好，在线只算 user embedding + ANN |
| 多粒度兴趣建模 | SIM, GEMs | 用户不是单一兴趣，需要多个向量/多流捕捉不同兴趣 |
| 负采样决定模型质量 | 全部向量召回方法 | "你告诉模型什么是负例"直接决定模型学到什么 |

---

## 🎓 面试考点（7条）

### Q1: 双塔模型（DSSM）的优劣势？
**30秒答案**：优势——user/item 独立编码，item embedding 离线预算一次在线复用，延迟极低（<1ms）；劣势——user 和 item 只在最后一层做内积交互，无法捕捉细粒度特征交叉。
**追问方向**：怎么弥补交互不足？答：精排阶段用 cross-attention 弥补，或用 ColBERT 式 late interaction。

### Q2: 多路召回怎么融合？
**30秒答案**：每路召回返回 top-K（通常 100-500），合并去重后约 2000-5000 候选。融合方式：①简单合并 + 去重 + 截断；②RRF（Reciprocal Rank Fusion）按排名倒数加权；③学习融合（训练一个轻量模型给每路打权重）。
**追问方向**：每路取多少个？答：按离线评估的各路召回率动态分配配额。

### Q3: 向量检索 ANN 有哪些方案？
**30秒答案**：①HNSW（图检索，精度最高，内存大）；②IVF-PQ（倒排 + 乘积量化，内存小速度快）；③ScaNN（Google，量化+剪枝）。选择依据：内存预算决定用 PQ 还是全精度，QPS 需求决定用图还是倒排。
**追问方向**：十亿级向量怎么存？答：分片 + 流式加载，或用 Milvus/Pinecone 等分布式向量数据库。

### Q4: SIM 怎么解决超长序列召回？
**30秒答案**：SIM 两阶段——第一阶段用简单规则（类目匹配/时间衰减）从用户 10000+ 行为中筛出 top-200 相关行为；第二阶段用 target attention（类 DIN）精细建模。
**追问方向**：ETA 和 SIM 的区别？答：ETA 用 SimHash 近似检索替代 SIM 的规则筛选，更快但精度略低。

### Q5: Semantic ID 和 生成式召回的核心思想？
**30秒答案**：用 RQ-VAE 将物品编码为多级 token（如 [A3, B7, C2]），推荐模型自回归生成这些 token，解码回物品。好处：不需要 ANN 索引，长尾物品天然被覆盖。
**追问方向**：生成式召回的延迟怎么解决？答：Speculative Decoding、Parallel Decoding、或缓存高频序列前缀。

### Q6: 负采样策略怎么选？
**30秒答案**：①随机负采样：最简单但效率低；②热度加权（P∝popularity^0.75）：避免总选冷门负例；③批内负采样（In-batch）：同 batch 内其他正样本作负例，高效但存在热度偏差；④难负例挖掘：用上一轮模型检索 top-K 中未点击的作为负例。
**追问方向**：批内负采样的热度偏差怎么修正？答：logQ correction，用 log(item_frequency) 修正 logits。

### Q7: 图召回（PinSage/EGES）的适用场景？
**30秒答案**：适合有明确图结构的场景（社交网络、商品知识图谱）。PinSage 用 GraphSAGE 在 Pin-Board 二部图上做 node embedding；EGES（阿里）在 item-item 共现图上做 embedding，引入 side information 增强冷启动。
**追问方向**：图召回 vs 向量召回哪个更好？答：互补——图召回更擅长发现结构关系（同一店铺、同一品牌），向量召回更擅长语义相似性。

---

## 🌐 知识体系连接

- **上游依赖**：Embedding 表示学习、ANN 索引算法、序列建模（Transformer/SSM）
- **下游应用**：粗排/精排模型输入、多路召回融合策略
- **相关 synthesis**：std_rec_ranking_evolution.md, std_rec_feature_engineering.md, std_search_hybrid_retrieval.md
- **相关论文笔记**：rec-sys/06_industry_recall_papers.md, synthesis/20260321_semantic_id_generative_retrieval.md
