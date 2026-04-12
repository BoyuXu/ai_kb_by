# 搜索 Reranker 演进：从 LambdaMART 到 LLM

> 📚 参考文献
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../01_recall/papers/Query_as_Anchor_Scenario_Adaptive_User_Representation_via.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../04_query/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../04_query/papers/Generative_Query_Expansion_for_E_Commerce_Search_at_Scale.md) — Generative Query Expansion for E-Commerce Search at Scale
> - [Colbert V3 Efficient Neural Retrieval With Late...](../../01_recall/papers/ColBERT_v3_Efficient_Neural_Retrieval_with_Late_Interacti.md) — ColBERT v3: Efficient Neural Retrieval with Late Interaction
> - [Document Re-Ranking With Llm From Listwise To Pair](../papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dllm-Searcher-Adapting-Diffusion-Large-Language...](../../04_query/papers/DLLM_Searcher_Adapting_Diffusion_Large_Language_Model_for.md) — DLLM-Searcher: Adapting Diffusion Large Language Model fo...

> 创建：2026-03-24 | 领域：搜索 | 类型：综合分析
> 来源：monoT5, RankGPT, ColBERT Reranker, Cross-Encoder 系列

## 架构总览

```mermaid
graph TB
    subgraph "范式1：Cross-Encoder"
        CE1["输入: [CLS] Query [SEP] Doc"] --> CE2[BERT全交互]
        CE2 --> CE3[相关性分数]
        CE4[✅ 精度最高<br/>❌ O(N)推理，延迟高]
    end
    subgraph "范式2：LLM Pointwise"
        LP1["Prompt: 该文档与Query相关吗？"] --> LP2[LLM判断 Yes/No]
        LP2 --> LP3[按概率排序]
        LP4[monoT5 / RankLLaMA]
    end
    subgraph "范式3：LLM Listwise"
        LL1["Prompt: 对以下文档按相关性排序"] --> LL2[LLM直接输出排列]
        LL2 --> LL3[滑动窗口处理长列表]
        LL4[RankGPT / LRL]
    end
    CE3 --> R[最终排序结果]
    LP3 --> R
    LL3 --> R
```

## 📐 核心公式与原理

### 1. NDCG

$$
NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG = \sum_{i=1}^K \frac{2^{rel_i}-1}{\log_2(i+1)}
$$

- 搜索排序核心评估指标

### 2. Cross-Encoder

$$
score = \text{MLP}(\text{BERT_{CLS}([q;d]))
$$

- Query-Doc 联合编码

### 3. Query Likelihood

$$
P(q|d) = \prod_{t \in q} P(t|d)
$$

- 概率语言模型检索

---

## 🎯 核心洞察（4条）

1. **Reranker 是搜索精度的"最后一公里"**：在 top-100 候选上重排序，NDCG@10 提升 5-15%，投入产出比极高
2. **Cross-Encoder 是精度标杆**：query-doc 拼接输入 BERT 做全交互，精度最高但延迟大（每个 doc 需要一次 BERT 推理）
3. **LLM Reranker 正在崛起**：RankGPT 用 GPT-4 做 listwise 重排序（一次性排序整个列表），在多个基准上超过 Cross-Encoder
4. **蒸馏是 Reranker 落地的关键**：用 LLM/大 Cross-Encoder 作为 teacher，蒸馏到小模型（如 MiniLM）做线上推理

---

## 🎓 常见考点（4条）

### Q1: Cross-Encoder Reranker 和 Bi-Encoder 的本质区别？
**30秒答案**：Bi-Encoder 将 query 和 doc 独立编码为向量再算内积（交互少但快）；Cross-Encoder 将 [query; doc] 拼接后过整个 Transformer（全交互但慢 100x）。Bi-Encoder 做召回，Cross-Encoder 做精排/重排。

### Q2: 怎么控制 Reranker 延迟？
**30秒答案**：①限制重排候选数（top-50~100）；②模型蒸馏（大模型→小模型，如 BERT-base→MiniLM）；③INT8 量化；④GPU batch 推理。目标延迟 <50ms。

### Q3: LLM Reranker 的两种范式？
**30秒答案**：①Pointwise：逐个文档打分（"这个文档与 query 相关吗？评分 1-10"）；②Listwise：一次性排序（"将这 20 个文档按相关性排序"）。Listwise 更强但受 context length 限制。

### Q4: Reranker 训练数据怎么构建？
**30秒答案**：①人工标注（query-doc 相关性标签 0-3 级）；②蒸馏标签（用 LLM 打分作为 soft label）；③点击数据（搜索日志中被点击的 doc 为正例）；④Hard Negative Mining（检索到但未被点击的作为难负例）。

---

### Q5: 搜索系统的评估指标有哪些？
**30秒答案**：离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。

### Q6: 稠密检索的训练数据构造？
**30秒答案**：正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。

### Q7: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。

### Q8: 向量检索的工程挑战？
**30秒答案**：①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。

### Q9: RAG 系统的常见问题和解决方案？
**30秒答案**：①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。

### Q10: E5 和 BGE 嵌入模型的区别？
**30秒答案**：E5（微软）：通用文本嵌入，支持 instruct 前缀。BGE-M3（BAAI）：多语言+多粒度+多功能（dense+sparse+ColBERT 三合一）。BGE-M3 更全面但模型更大。
---

## 🆕 2026-03-26 更新：Rank-R1 — 推理增强重排序

### 新参考文献
> - [Rank-R1](../papers/rank_r1_enhancing_reasoning_in_llm_based_document_rerankers.md) — Enhancing Reasoning in LLM-based Document Rerankers via RL

### 核心创新：CoT + RL 的推理重排序

**问题**：标准 LLM Reranker 直接输出相关性分数，对复杂查询（多跳推理/领域知识密集）缺乏中间推理过程，排序精度低。

**Rank-R1 方法**：
```python
# 标准 LLM Reranker（无推理）
score = LLM(f"Query: {q}\nDoc: {d}\nScore:")

# Rank-R1（带 CoT 推理链）
output = LLM(f"""
Query: {q}, Doc: {d}

<think>
1. 这个 query 需要什么信息？
2. 这篇文档提供了什么内容？
3. query 需求和文档内容的匹配度如何？
</think>
相关性分数：
""")
# RL 优化：用 NDCG 改善量作为奖励信号，训练模型输出更准确的推理链
```

**为什么 RL 而非 SFT？**
- SFT 需要人工标注推理链（成本极高）
- RL 只需相关性标签（0/1/多级），自动从奖励学习推理策略
- 奖励：NDCG 提升量（列表级），梯度传播到整个推理链

**与 DeepSeek-R1 的技术迁移关系**：
```
DeepSeek-R1（推理 LLM）→ Rank-R1（推理 Reranker）
共同点：
  - 相同技术：CoT 格式 + RL 优化（GRPO/PPO）
  - 相同原理：推理链让注意力聚焦于相关信息
差异：
  - R1：奖励 = 数学答案正确性（0/1）
  - Rank-R1：奖励 = NDCG 改善量（连续值）
```

**BRIGHT 基准评测结果**（推理密集型检索）：
- 标准 LLM Reranker NDCG@10：~0.35
- Rank-R1 NDCG@10：~0.43（+23%，复杂推理查询提升更显著）

### 工程落地注意事项
1. **延迟**：CoT 输出更长（2-5x tokens），Reranker 延迟从 100ms → 300-500ms
2. **解决方案**：只对 Top-20 候选使用 Rank-R1，其余用轻量 Cross-Encoder
3. **蒸馏**：用 Rank-R1 生成推理链标注 → 蒸馏到 7B 模型，延迟降回 100ms
4. **适用场景**：复杂长尾查询（推理密集），不必对所有查询都用推理链

---

## 🌐 知识体系连接

- **上游依赖**：BERT/LLM、Knowledge Distillation、DeepSeek-R1/GRPO
- **下游应用**：搜索引擎精排、RAG 系统、电商搜索
- **相关 synthesis**：检索三角形深析.md, LearningToRank搜索排序三大范式.md
- **跨域连接**：[强化学习跨域统一视角](../../../cross-domain/synthesis/强化学习跨域统一视角_LLM推理到广告出价.md)

## 📐 核心公式直观理解

### MonoBERT（Pointwise Reranker）

$$
P(\text{relevant} | q, d) = \sigma(W \cdot \text{BERT}_{CLS}([q; d]) + b)
$$

**直观理解**：把 (query, document) 对输入 BERT，用 CLS token 的输出做二分类（相关/不相关）。Pointwise 方法简单直接但忽略了文档间的比较关系——给两篇文档打了 0.8 和 0.7 分，但它们之间的排序关系可能和真实不一致。

### DuoBERT（Pairwise Reranker）

$$
P(d_i \succ d_j | q) = \sigma(W \cdot \text{BERT}_{CLS}([q; d_i; d_j]) + b)
$$

**直观理解**：两两比较"哪个更好"，比绝对打分更稳定——人类也更擅长说"A 比 B 好"而非给 A 打多少分。代价是 $O(n^2)$ 次比较，工业中用 bubble sort 策略（只比较相邻 pair）降低到 $O(n \log n)$。

### ColBERT 的 MaxSim 操作

$$
\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} q_i^T d_j
$$

**直观理解**：query 的每个 token 找到 document 中"最匹配"的那个 token 算分，所有 query token 的分数求和。比 dense retrieval（全局一个向量）精细——能捕捉"query 中某个关键词在文档中有精确对应"的信号。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
