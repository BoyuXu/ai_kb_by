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
