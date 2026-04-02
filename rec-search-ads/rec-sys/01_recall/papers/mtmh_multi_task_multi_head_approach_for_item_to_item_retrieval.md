# MTMH: Multi-Task Multi-Head Approach for Item-to-Item Retrieval
> 来源：arxiv/2310.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
Item-to-Item（I2I）检索（"看了又看"/"买了又买"）面临：
- 单一相似度不够：语义相似 ≠ 协同相似 ≠ 购买共现相似
- 不同场景需要不同相似度维度：首页推荐 vs 购物车 vs 搜索结果页
- 多目标难以在单模型中统一
- 训练信号稀疏：I2I 正样本数量远少于 U2I

## 核心方法与创新点
**MTMH（Multi-Task Multi-Head）**：多任务多头 I2I 检索模型。

**架构设计：**
```python
# 共享 item Encoder
item_emb = SharedEncoder(item_features)  # title, category, price, etc.

# 多个任务头，每个头学习不同的相似度空间
head_semantic = Linear(item_emb, d_out)    # 语义相似
head_cf       = Linear(item_emb, d_out)    # 协同过滤（共现）
head_purchase  = Linear(item_emb, d_out)   # 购买共现
head_session  = Linear(item_emb, d_out)    # session 内共现

# 检索时根据场景选择对应头
query_emb = head_k(encoder(query_item))
candidates = ANN_search(query_emb, index_k)  # 场景 k 对应的索引
```

**多任务训练：**
```
L = Σ_k w_k · InfoNCE_k(anchor, pos_k, neg_k)
InfoNCE_k = -log exp(sim(q_k, p_k)/τ) / Σ_j exp(sim(q_k, n_kj)/τ)
```

**Hard Negative Mining：**
- 语义头：用 TF-IDF 相似但用户不点击的 item
- CF 头：同类目但共现低的 item
- BM25/in-batch/cross-head hard negative

## 实验结论
- 京东 I2I 检索系统 A/B：
  - 点击率 +2.8%，加购率 +1.9%
- 离线 Recall@100：
  - 语义相似：+6.3% vs 单头模型
  - 协同相似：+4.1%
- 多样性：不同场景使用不同头，多样性显著提升

## 工程落地要点
1. **多索引管理**：每个头对应独立的 FAISS 索引，定期重建
2. **场景路由**：根据场景（首页/购物车/搜索）动态选择检索头
3. **负采样策略**：不同任务头使用不同负采样方式（语义用随机，CF用热门）
4. **在线融合**：多个头的检索结果合并后统一排序（线性加权或 RankFusion）
5. **冷启动**：新 item 无协同数据时，仅使用语义头

## 常见考点
**Q1: I2I 检索为什么要多头设计？**
A: 不同场景的相似度需求不同：购物车场景需要购买共现相似（常一起买）；首页"看了又看"需要语义相似；Session 内推荐需要 session 共现。单一相似度无法满足所有场景，多头允许每个场景选最合适的相似度空间。

**Q2: 多任务训练时任务权重 w_k 如何设置？**
A: ①均等权重：简单但忽略任务难度差异 ②Uncertainty Weighting：基于 homoscedastic uncertainty 自动学习权重 ③GradNorm：动态平衡各任务梯度范数。实践中先用均等，再用 GradNorm 微调。

**Q3: I2I 模型的负样本构建有哪些策略？**
A: ①随机负采样（简单，但质量差）②in-batch 负采样（效率高，但易有假负例）③Hard Negative：BM25/语义相近但无共现的 item ④Popularity-based：用热门 item 作为负例（缓解流行度偏差）。

**Q4: 如何评估 I2I 检索质量？**
A: 离线：Recall@K（正样本是历史共现 item）、Hit Rate；在线：I2I 模块的点击率、加购率、购买率；多样性：ILD（Item List Diversity）；覆盖率：有 I2I 结果的 item 比例。

**Q5: I2I 与 U2I 检索的主要区别？**
A: U2I：Query 是用户，需要个性化，正样本是用户点击/购买 item；I2I：Query 是 item，需要 item 间相似度，正样本是共现 item。I2I 样本更稀疏但个性化无关，适合做离线 item 图。

## 模型架构详解

### 候选编码
- **Item 表示**：Semantic ID（层次化离散编码）或稠密向量 Embedding
- **编码方式**：RQ-VAE（残差量化）/ K-Means 聚类 / 端到端学习的 Token 序列
- **多模态融合**：文本/图片/行为信号的统一表示空间

### 检索机制
- **生成式检索**：自回归解码器逐步生成 Item Token 序列
- **向量检索**：双塔编码 + ANN 索引（HNSW/IVF-PQ）
- **混合召回**：多路检索结果的统一评分与去重

### 训练策略
- **正样本**：用户交互（点击/购买/收藏）
- **负采样**：In-batch Negatives + 难负例挖掘
- **对比学习**：InfoNCE Loss 拉近正样本、推远负样本
- **课程学习**：从简单到困难逐步增加负例难度

## 与相关工作对比

| 维度 | 生成式召回 | 双塔向量召回 | 传统倒排 |
|------|-----------|------------|---------|
| 冷启动 | 好（内容特征） | 中（需行为） | 差 |
| 索引维护 | 无需显式索引 | 需 ANN 索引 | 需倒排表 |
| 推理延迟 | 中（自回归） | 低（一次编码） | 低 |
| 可扩展性 | 亿级 | 亿级 | 百万级 |
| 多模态 | 原生支持 | 需要适配 | 困难 |

## 面试深度追问

- **Q: Semantic ID 的设计思路和优势？**
  A: 将 Item 映射为离散 Token 序列（类似自然语言），使推荐问题转化为序列生成。优势：1) 天然支持自回归生成；2) 层次化结构（粗→细）提升检索效率；3) 避免连续向量的 ANN 近似误差。

- **Q: 生成式召回如何处理新物品？**
  A: 1) 内容特征驱动的 Semantic ID 分配（新物品基于属性分配 Token）；2) 增量学习更新 Codebook；3) 备用的 Content-based 召回通道兜底。

- **Q: 多路召回的融合策略？**
  A: 1) 统一打分：所有通道候选用同一模型重新打分；2) 配额分配：各通道按历史表现分配固定配额；3) 加权融合：考虑通道多样性的加权排序。

- **Q: 如何衡量召回质量？**
  A: 离线：Recall@K, HR@K, NDCG@K。在线：端到端 CTR/GMV 提升 + 召回覆盖率 + 新颖性。注意 K 值要与下游排序的候选集大小匹配。
