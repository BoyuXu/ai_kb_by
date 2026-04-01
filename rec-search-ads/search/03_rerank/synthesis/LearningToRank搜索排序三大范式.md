# Learning to Rank：搜索排序的三大范式

> 📚 参考文献
> - [Document Re-Ranking With Llm From Listwise To Pair](../papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../01_recall/papers/Query_as_Anchor_Scenario_Adaptive_User_Representation_via.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../04_query/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../04_query/papers/Generative_Query_Expansion_for_E_Commerce_Search_at_Scale.md) — Generative Query Expansion for E-Commerce Search at Scale
> - [Hybrid-Search-Llm-Re-Ranking](../papers/Hybrid_Search_with_LLM_Re_ranking_for_Enhanced_Retrieval.md) — Hybrid Search with LLM Re-ranking for Enhanced Retrieval ...
> - [Multimodal-Visual-Document-Retrieval-Survey](../../01_recall/papers/Unlocking_Multimodal_Document_Intelligence_Visual_Documen.md) — Unlocking Multimodal Document Intelligence: Visual Docume...

> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：LambdaMART, RankNet, ListNet, BERT Reranker 系列

## 📐 核心公式与原理

### 📐 1. DCG / NDCG 推导

$$
\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}, \qquad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}
$$

**推导步骤：**

1. **CG（Cumulative Gain）**：仅累加相关性，不考虑位置：$\text{CG}@K = \sum_{i=1}^K rel_i$，缺点是 rank 1 和 rank K 权重相同

2. **DCG（折扣）**：用 $\log_2(i+1)$ 对位置 $i$ 折扣，第 1 位权重为 $1/\log_2 2 = 1$，第 2 位为 $1/\log_2 3 \approx 0.63$，以此类推。相关性增益 $2^{rel_i}-1$ 对高等级相关性（如 rel=3 → 7 分）给予指数级奖励，区分"相关"与"高度相关"。

3. **IDCG（Ideal DCG）**：将所有文档按真实相关度从高到低排列的 DCG，即理论最优 DCG

4. **NDCG 归一化**：NDCG = DCG / IDCG，范围 $[0,1]$；NDCG=1 表示排序与理想排序完全一致

5. **不可微的问题**：NDCG 涉及排序操作，对模型参数不可微，无法直接梯度下降优化——这是 RankNet/LambdaRank 等方法存在的根本动机。

**符号说明：**
- $rel_i \in \{0,1,2,3\}$：位置 $i$ 处文档的相关性分级（人工标注）
- $K$：评估截断深度（如 NDCG@10）
- $\text{IDCG}@K$：理想排序的 DCG（相关文档按相关度从高到低排列）

---

### 📐 2. LambdaRank / LambdaMART 推导

LambdaRank 的核心思想：**直接定义梯度**（虚拟梯度），而非从损失函数求导。

对文档对 $(i, j)$（$i$ 的相关性高于 $j$），LambdaRank 的梯度定义为：

$$
\lambda_{ij} = \frac{-1}{1 + e^{s_i - s_j}} \cdot |\Delta\text{NDCG}_{ij}|
$$

$$
\lambda_i = \sum_{j:(i,j) \in \mathcal{P}} \lambda_{ij} - \sum_{j:(j,i) \in \mathcal{P}} \lambda_{ji}
$$

**推导步骤：**

1. **RankNet 的出发点**：RankNet 用交叉熵损失最小化文档对的排序错误：
   $$\mathcal{L}_{\text{RankNet}} = \sum_{(i,j)} \log(1 + e^{-(s_i - s_j)})$$
   对 $s_i$ 的梯度为 $\lambda_{ij}^{\text{RankNet}} = -\sigma(-\Delta_{ij}) = \frac{-1}{1+e^{s_i - s_j}}$

2. **LambdaRank 的关键改进**：将 RankNet 梯度乘以 $|\Delta\text{NDCG}_{ij}|$（交换文档 $i,j$ 的位置后 NDCG 的变化量）：
   - 如果交换 $i, j$ 导致 NDCG 大幅下降（即 $i$ 排在前面很重要），则惩罚更大
   - 如果交换导致 NDCG 变化极小（两个文档都不重要），则惩罚很小

3. **$|\Delta\text{NDCG}|$ 的计算**（对序列长为 $N$ 的查询）：
   $$|\Delta\text{NDCG}_{ij}| = \left|\frac{1}{\text{IDCG}}\left(\frac{2^{rel_i}-1}{\log_2(\text{rank}_i+1)} + \frac{2^{rel_j}-1}{\log_2(\text{rank}_j+1)} - \frac{2^{rel_i}-1}{\log_2(\text{rank}_j+1)} - \frac{2^{rel_j}-1}{\log_2(\text{rank}_i+1)}\right)\right|$$

4. **LambdaMART**：以 LambdaRank 梯度作为 GBDT（MART）的训练目标，每棵树拟合 $\lambda_i$，是非深度学习排序模型的 SOTA。

**符号说明：**
- $s_i, s_j$：模型对文档 $i, j$ 的打分
- $\lambda_{ij}$：文档对 $(i,j)$ 给文档 $i$ 的梯度贡献
- $|\Delta\text{NDCG}_{ij}|$：交换 $i, j$ 位置后的 NDCG 变化量（衡量该对的重要性）
- $\mathcal{P}$：所有文档对集合，其中 $i$ 的真实相关性高于 $j$

**直观理解：** LambdaRank 说的是："不是所有的排错都同等严重——把 rank 1 的相关文档排到 rank 10 比把 rank 9 的排到 rank 10 损失大得多。" $|\Delta\text{NDCG}|$ 系数让模型更关注"高位排错"的纠正，自然地优化了 NDCG 这个不可微目标。

---

### 3. Query Likelihood 语言模型检索

$$
P(q \mid d) = \prod_{t \in q} P(t \mid d), \quad P(t \mid d) = (1-\lambda) \frac{tf_{t,d}}{|d|} + \lambda P(t \mid \mathcal{C})
$$

**Jelinek-Mercer 平滑**：$\lambda \in [0,1]$ 控制文档语言模型与语料库背景语言模型的插值比例，解决文档中词频为零的问题（smoothing）。这是 BM25 之前经典统计检索的标准方法。

---

## 🎯 核心洞察（4条）

1. **三大范式各有优劣**：Pointwise（逐个打分，简单但忽略文档间关系）→ Pairwise（文档对比较，平衡）→ Listwise（整列表优化，理论最优但训练复杂）
2. **LambdaMART 仍是非深度学习的 SOTA**：基于 GBDT 的排序模型，在特征工程好的情况下效果可以比肩甚至超过深度模型
3. **BERT Reranker 改变了精排**：将 query-document 拼接输入 BERT 做分类/回归，精度大幅提升但延迟也大幅增加
4. **NDCG 是排序评估的标准指标**：Normalized Discounted Cumulative Gain 综合考虑了相关性级别和位置折扣

---

## 📈 技术演进脉络

```
BM25 启发式排序（~2005）→ RankSVM/RankNet Pairwise（2005-2010）
  → LambdaMART GBDT（2010-2018）→ BERT Reranker（2019-2022）
    → LLM Listwise Reranker（2023+）
```

---

## 🎓 面试考点（5条）

### Q1: Pointwise/Pairwise/Listwise 的区别？
**30秒答案**：Pointwise 将排序转化为回归/分类（预测每个文档的相关性分数）；Pairwise 转化为文档对的比较（A 比 B 更相关？）；Listwise 直接优化整个列表的排序指标（如 NDCG）。Pointwise 最简单，Listwise 理论最优。

### Q2: LambdaMART 的核心思想？
**30秒答案**：在 MART（梯度提升树）基础上，梯度不是 loss 对预测值的导数，而是 λ 梯度——直接反映"交换两个文档位置对 NDCG 的影响"。这样模型直接优化排序指标而非替代 loss。

### Q3: NDCG 的计算方式？
**30秒答案**：`DCG@K = Σ(2^rel_i - 1) / log2(i+1)`，i 是排名位置，rel_i 是相关性标签。NDCG = DCG / IDCG（理想排序的 DCG），归一化到 [0,1]。关键：位置越靠后折扣越大，相关性越高收益越大。

### Q4: BERT Reranker vs 双塔模型？
**30秒答案**：双塔（Bi-Encoder）：query 和 doc 独立编码再内积，快但交互弱；BERT Reranker（Cross-Encoder）：query-doc 拼接后过 BERT，慢但交互充分。通常双塔做召回/粗排，BERT 做精排。

### Q5: 搜索排序的特征类型？
**30秒答案**：①Query 特征（长度、意图类型）；②文档特征（PageRank、长度、新鲜度）；③Query-Doc 交互特征（BM25 分数、term overlap、语义相似度）；④用户特征（历史点击偏好）；⑤上下文特征（设备、时间、地域）。

---

### Q6: 搜索系统的评估指标有哪些？
**30秒答案**：离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。

### Q7: 稠密检索的训练数据构造？
**30秒答案**：正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。

### Q8: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。

### Q9: 向量检索的工程挑战？
**30秒答案**：①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。

### Q10: RAG 系统的常见问题和解决方案？
**30秒答案**：①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。
## 🌐 知识体系连接

- **上游依赖**：GBDT/XGBoost、BERT/Transformer、特征工程
- **下游应用**：搜索引擎排序、电商搜索、RAG 重排序
- **相关 synthesis**：检索三角形深析.md, 推荐系统排序范式演进.md
