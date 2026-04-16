# LLM 增强信息检索与 RAG 技术进展（2025）

> 综合总结 | 领域：search | 学习日期：20260404

## 主题概述

2025年信息检索的核心趋势：**LLM 驱动的 Embedding 质量飞跃**、**Agentic 检索（推理-检索交织）**、**RAG 系统深度优化**。

---

## 一、LLM 作为 Embedding 模型

**LLM-Embedder** 的技术路线：

双向 Attention 改造：

$$e_{\text{text}} = h_L(\text{[EOS]}) \text{ with bidirectional attention mask}$$

多任务对比学习：

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{e^{s(q, d^+)/\tau}}{\sum_{j} e^{s(q, d_j^-)/\tau}}$$

**GTE-Qwen2** 跨语言对齐：

$$\mathcal{L}_{\text{cross-lingual}} = -\log \frac{e^{s(q_{\text{zh}}, d_{\text{en}}^+)/\tau}}{\sum_j e^{s(q_{\text{zh}}, d_j)/\tau}}$$

Matryoshka Representation：截断 Embedding 维度仍有效，256d 只损失 2.1% 性能，速度 +4x。

---

## 二、Agentic 检索

**Search-o1** 推理-检索交织：

自适应触发：

$$\text{trigger} = \mathbb{1}[\text{Var}(P(\text{tokens})) > \theta]$$

迭代 RAG 流程：

```
Reasoning → [知识缺口] → <search>query</search> → 
[检索证据] → 更新推理 → [新知识缺口] → ...
```

RL 奖励函数（兼顾质量和效率）：

$$R = R_{\text{correctness}} - \alpha \cdot N_{\text{searchCalls}}$$

---

## 三、统一重排与生成

**RankRAG** 一模两用：

$$\text{RankRAG}(q, \mathcal{D}) = \text{Gen}\left(q, \text{Top-K}_{d \in \mathcal{D}}[\text{Rank}(q, d)]\right)$$

通过联合训练将 Reranker 和 Generator 合并，减少 35% 推理延迟，NDCG +4.2%。

---

## 四、查询改写与多路融合

HyDE（假设文档）：

$$e_{\text{query}} \leftarrow e_{\text{LLM}(\text{"Answer: " + q})}$$

RRF 多路融合：

$$\text{RRF}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{rank}_i(d)}, \quad k=60$$

---

## 五、推理感知检索

**ReasonIR** 推理密集型任务检索：

$$\mathcal{L}_{\text{reason}} = -\log \frac{e^{s(q, d^+)/\tau}}{e^{s(q, d^+)/\tau} + \sum e^{s(q, d^-_{\text{reason-hard}})/\tau}}$$

BRIGHT Benchmark NDCG@10 **+8.3%** vs 最强基线。

---

## 🎓 面试高频 Q&A（10题）

**Q1**: Dense Retrieval vs BM25 的核心对比？  
**A**: BM25：词汇匹配，高精度、可解释、无需训练；Dense：语义匹配，泛化好、跨语言能力强，需要大量标注数据。生产通常混合（Hybrid Search）。

**Q2**: 如何构造 Dense Retrieval 的 Hard Negative？  
**A**: BM25 Top-K（词汇相似但语义不匹配）+ Dense 模型 Top-K（语义相近但答案错误）+ 随机负例（简单）。三者混合，比例约 3:2:1。

**Q3**: RAG 系统的关键评估指标？  
**A**: 检索层：Recall@K、NDCG@K；生成层：EM/F1（事实性）、ROUGE（覆盖率）；系统层：Hallucination Rate（幻觉率）、Faithfulness（答案忠于检索结果）。

**Q4**: Agentic RAG vs 传统 RAG 的适用场景？  
**A**: 传统 RAG：单跳事实问答、固定格式；Agentic RAG：多跳推理、需要中间验证、动态知识整合、不确定需要再检索的任务。

**Q5**: 为什么 HyDE（假设文档）有效？核心假设？  
**A**: Query 和 Document 语言风格不同（疑问句 vs 陈述句），直接用 Query Embedding 检索有 Domain Gap。假设文档语言更接近真实文档，检索精度更高。

**Q6**: 多语言检索模型如何训练跨语言对齐？  
**A**: 大规模跨语言 Parallel 数据（翻译对）对比学习 + 跨语言 (query_zh, doc_en) 对微调。Qwen2 等多语言 LLM 预训练提供强初始化。

**Q7**: RAG 的 Hallucination 来源有哪些？如何缓解？  
**A**: 来源：检索失败（无相关文档）→ 模型编造；检索噪声（相关但错误文档）→ 混淆；过时知识（训练截止）。缓解：Reranking（过滤噪声）+ Faithfulness 约束 + 多轮检索验证。

**Q8**: Query 改写（Query Reformulation）的技术方案？  
**A**: LLM 扩展改写（多表达方式）+ HyDE（假设答案文档）+ 意图分类（压缩/扩展/重表述）+ RRF 多路融合。成本：LLM 调用 latency，建议高频 Query 缓存。

**Q9**: 推理感知检索（ReasonIR）的核心挑战？  
**A**: 传统语义相似 ≠ 推理相关；训练数据稀缺（推理标注昂贵）；多跳误差累积。解法：CoT 合成数据 + 推理路径相关文档作为 Positive。

**Q10**: 如何评估 Embedding 模型的多任务泛化能力？  
**A**: MTEB（Massive Text Embedding Benchmark）：包含分类、聚类、检索、重排、语义相似等 56 个数据集，是当前最全面的 Embedding 评估 Benchmark。

---

## 📚 参考文献

1. LLM-Embedder: Leveraging LLMs for Text Embeddings (2024)
2. Search-o1: Agentic Search-Enhanced Reasoning Models (2025)
3. RankRAG: Unifying Context Ranking with RAG (2024)
4. GTE-Qwen2: Multi-Lingual Embeddings for Dense Retrieval (2024)
5. Query Reformulation with LLMs for Zero-Shot Retrieval (2024)

---

## 相关概念

- [[embedding_everywhere|Embedding 技术全景]]
- [[multi_objective_optimization|多目标优化]]
