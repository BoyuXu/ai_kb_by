# 搜索 Query 理解：从分词到意图识别到 LLM 改写

> 📚 参考文献
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Hybrid-Search-Llm-Re-Ranking](../../03_rerank/papers/Hybrid_Search_with_LLM_Re_ranking_for_Enhanced_Retrieval.md) — Hybrid Search with LLM Re-ranking for Enhanced Retrieval ...
> - [Multimodal-Visual-Document-Retrieval-Survey](../../01_recall/papers/Unlocking_Multimodal_Document_Intelligence_Visual_Documen.md) — Unlocking Multimodal Document Intelligence: Visual Docume...
> - [Legalmalr-Multi-Agent-Chinese-Statute-Retrieval](../../03_rerank/papers/LegalMALR_Multi_Agent_Query_Understanding_LLM_Based_Reran.md) — LegalMALR: Multi-Agent Query Understanding and LLM-Based ...
> - [Document Re-Ranking With Llm From Listwise To Pair](../../03_rerank/papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dense-Passage-Retrieval-Conversational-Search](../../01_recall/papers/Dense_Passage_Retrieval_in_Conversational_Search.md) — Dense Passage Retrieval in Conversational Search

> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：Query Rewriting, Intent Classification, NER, LLM-based QU 系列

## 📐 核心公式与原理

### 1. NDCG

$$
NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG = \sum_{i=1}^K \frac{2^{rel_i}-1}{\log_2(i+1)}
$$

- 搜索排序核心评估指标

### 2. Cross-Encoder

$$
score = \text{MLP}(\text{BERT}}_{\text{{CLS}}([q;d]))
$$

- Query-Doc 联合编码

### 3. Query Likelihood

$$
P(q|d) = \prod_{t \in q} P(t|d)
$$

- 概率语言模型检索

---

## 🎯 核心洞察（4条）

1. **Query 理解是搜索质量的第一关**：用户输入的 query 往往模糊、有错别字、表达不精确，QU 模块的好坏直接决定召回质量的上限
2. **QU 模块的四层架构**：纠错 → 分词/NER → 意图分类 → 改写/扩展，每层都可以用传统方法或深度模型
3. **LLM 正在统一 QU 的各个子模块**：传统方法需要纠错模型+NER模型+意图分类模型+改写模型（4 个独立模型），LLM 一个 prompt 就能完成全部 QU 任务
4. **Query 改写对 RAG 至关重要**：RAG 系统中用户 query 直接用于检索，如果 query 表达不好，检索结果就差。Query 改写将口语化/多轮 query 转化为结构化检索 query

---

## 📈 技术演进脉络

```
规则分词 + 同义词词典（~2010）
  → 统计方法（CRF NER + SVM 意图分类, 2010-2016）
    → 深度模型（BERT NER + Seq2Seq 改写, 2018-2022）
      → LLM 统一 QU（GPT/Claude 做纠错+NER+意图+改写, 2023+）
```

---

## 🎓 常见考点（5条）

### Q1: Query 理解包含哪些模块？
**30秒答案**：①拼写纠错（编辑距离/语言模型）；②分词+实体识别（品牌名/型号/属性）；③意图分类（导航型/信息型/事务型/多意图）；④同义词扩展（"手机"→"手机,智能手机,移动电话"）；⑤Query 改写（多轮对话上下文融合、口语→结构化）。

### Q2: 搜索意图分类的类型？
**30秒答案**：①导航型（找特定网站，如"淘宝"）；②信息型（找信息，如"什么是机器学习"）；③事务型（想完成操作，如"买 iPhone 16"）；④多意图（如"苹果"——水果还是手机？）。分类结果影响后续排序策略。

### Q3: Query 改写在 RAG 中怎么做？
**30秒答案**：①HyDE（Hypothetical Document Embeddings）：LLM 先生成一个"假设性回答"，用回答的 embedding 检索；②Multi-Query：将原始 query 改写为多个角度的 sub-queries，分别检索后合并；③Step-back Prompting：将具体问题抽象化再检索。

### Q4: 实体链接（Entity Linking）的作用？
**30秒答案**：将 query 中的实体提及映射到知识库中的标准实体。如"苹果14 pro"→ 标准实体"iPhone 14 Pro"。作用：①精确匹配产品库；②消歧义（"苹果"→Apple Inc. 还是水果）；③关联属性信息。

### Q5: LLM 做 Query 理解的优劣势？
**30秒答案**：优势——一个模型做全部 QU 任务，零样本能力强，对口语化/多语言 query 理解好。劣势——延迟高（50-200ms vs 传统 <5ms）、成本高、不可控（可能改写出不相关的内容）。
**追问方向**：怎么平衡？答：简单 query 用传统方法，复杂/模糊 query 才调用 LLM。

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

- **上游依赖**：NLP 基础（NER/分词）、LLM、知识图谱
- **下游应用**：搜索召回质量、RAG 系统、对话式搜索
- **相关 synthesis**：检索三角形深析.md, 混合检索融合_多路召回实践.md


## 📐 核心公式直观理解

### Intent Classification

$$
P(\text{intent} | q) = \text{softmax}(W \cdot \text{BERT}_{\text{CLS}}(q) + b)
$$

**直观理解**：搜索意图分为导航型（"淘宝官网"）、信息型（"什么是量子计算"）、交易型（"买 iPhone 16"）。意图决定了后续检索策略——导航型直接返回 URL，信息型返回知识卡片，交易型返回商品列表。

### Query Segmentation

$$
P(\text{seg} | q) = \arg\max_{s \in \text{segments}} \prod_{i} P(w_i | w_{<i}, s)
$$

**直观理解**：中文 query "深度学习推荐系统"应该切成"深度学习/推荐系统"而非"深度/学习/推荐/系统"。正确分词直接影响检索质量——错误分词会导致关键短语被拆散，失去语义。

### Query Autocomplete

$$
P(q | q_{1:k}) = \prod_{t=k+1}^{T} P(q_t | q_{1:t-1})
$$

**直观理解**：用户输入"推荐系"后，模型预测最可能的完整 query（"推荐系统"、"推荐系统论文"）。本质是条件语言模型——但不是通用 LM，而是在历史 query 日志上训练的，偏向高频 query。



---
## Query 理解的工程决策：各方法的适用边界

### 从例子看 Query 理解的难度

| Query | 字面含义 | 真实意图 | 挑战 |
|-------|---------|---------|------|
| "苹果" | 水果/品牌？ | 80% 手机，15% 水果，5% 公司 | 词义消歧 |
| "便宜手机" | 字面即意图 | 但"便宜"阈值因人而异 | 个性化理解 |
| "怎么选显卡" | 购买意图 | 实际是知识类 query | 意图识别 |
| "iPad pro 和 MacBook 哪个好" | 比较意图 | 需要结构化对比结果 | 结果类型理解 |
| "推荐一个适合去日本的行李箱" | 购买+场景意图 | 需要理解旅行场景 | 上下文推断 |

### Query 改写的收益与风险

**收益**：解决长尾 query 召回率低的问题
- 长尾 query（"适合骑手用的防水蓝牙耳机"）可能直接召回 0 结果
- 改写为 "防水蓝牙耳机" → 有结果

**风险**：改写过度导致语义偏移
- "白色连衣裙" → 改写为 "连衣裙" → 召回了大量非白色 → 用户投诉
- 规则：改写的泛化度不能超过 1 层（不能跨两个属性同时删除）

### LLM Query 理解 vs 传统 NLU 的实战对比

| 维度 | 传统 NLU（BERT fine-tuned）| LLM（7B+）|
|------|--------------------------|---------|
| 延迟 | < 5ms | 50-200ms |
| 准确率（标准意图）| 92% | 95% |
| 准确率（新兴 query）| 60%（需重新训练）| 85%（zero-shot）|
| 维护成本 | 高（每季度重标注）| 低（prompt 调整即可）|
| 适用策略 | P50 高频 query（量大）| P99 长尾/新 query（精度优先）|

实践结论：**LLM 做兜底，传统 NLU 做主力**（80/20 分工）。
