# LLM 增强检索与 RAG 技术全景

> 综合文档 | 领域：搜索/RAG | 合并自 3 篇 synthesis | 更新：2026-04-13
> 来源：LLM赋能搜索系统前沿综述.md、LLM增强信息检索与RAG技术进展.md、2026-04-09_rag_systems_evolution.md

---

## 一、核心趋势

2025-2026 年 LLM 增强检索的三大趋势：
1. **LLM 作为 Embedding 模型**：双向 Attention 改造 + 多任务对比学习 → 嵌入质量飞跃
2. **Agentic 检索**：推理-检索交织，从被动 RAG 到主动多步推理检索
3. **RAG 系统深度优化**：从 Naive RAG → Long-Context RAG → Graph RAG → Enterprise RAG

---

## 二、搜索全链路技术图谱

```
用户查询（文本/图片）
        ↓
[Query Understanding]
  - 多模态 LLM 查询理解（图文融合）
  - 意图分类 / 属性抽取 / Query 改写
        ↓
[Retrieval]
  - BM25（稀疏，关键词匹配）
  - Dense Bi-encoder（单向量）
  - Late Interaction ColBERT（多向量）
  - Hybrid Dense + BM25（互补，简单有效）
        ↓
[Reranking]
  - Cross-encoder（精度最高）
  - SLM 蒸馏（LinkedIn: 75x 吞吐提升）
  - Rank-R1（GRPO 推理增强）
        ↓
[Generation / Answer]
  - RAG 生成 / 直接排序结果
```

---

## 三、核心公式

### 3.1 LLM-Embedder 对比学习

$$
\mathcal{L}_{\text{contrastive}} = -\log \frac{e^{s(q, d^+)/\tau}}{\sum_{j} e^{s(q, d_j^-)/\tau}}
$$

- 双向 Attention 改造：$e_{\text{text}} = h_L(\text{[EOS]})$ with bidirectional attention mask
- Matryoshka Representation：截断到 256d 只损失 2.1%，速度 +4x

### 3.2 HyDE（假设文档）

$$
d_{\text{pseudo}} = \text{LLM}(\text{"生成一段回答以下问题的文档："} + q)
$$

- 伪文档虽可能有错，但 embedding 空间中与真相关文档更近（"答案风格"文本）
- 比直接用 query 检索效果好 10-20%

### 3.3 FiD（Fusion-in-Decoder）

$$
P(y | q, d_1, ..., d_K) = \text{Decoder}(\text{Enc}(q, d_1) \oplus ... \oplus \text{Enc}(q, d_K))
$$

- 每个文档和 query 分别编码（encoder 内独立），decoder 中融合
- 比拼接后输入更高效：encoder 可并行

### 3.4 Search-o1 自适应触发

$$
\text{trigger} = \mathbb{1}[\text{Var}(P(\text{tokens})) > \theta]
$$

$$
R = R_{\text{correctness}} - \alpha \cdot N_{\text{searchCalls}}
$$

- 推理-检索交织：[知识缺口] → 检索 → 更新推理 → [新缺口] → ...
- RL 奖励兼顾质量和效率

### 3.5 RankRAG 一模两用

$$
\text{RankRAG}(q, \mathcal{D}) = \text{Gen}\left(q, \text{Top-K}_{d \in \mathcal{D}}[\text{Rank}(q, d)]\right)
$$

- 联合训练 Reranker + Generator，减少 35% 推理延迟，NDCG +4.2%

### 3.6 搜索 Agent 的迭代检索

$$
q_{t+1} = \text{LLM}(q_t, d_{1:K}^{(t)}, \text{"哪些信息还缺？请改写 query"})
$$

- 像研究者"越查越精确"，通常 2-3 轮即可

### 3.7 Hybrid Retrieval 线性插值

$$
S_{hybrid}(q, d) = \alpha \cdot S_{dense}(q, d) + (1-\alpha) \cdot S_{BM25}(q, d)
$$

- $\alpha = 0.5$ 在 BRIGHT 上使 nDCG@10 从 0.289 → 0.372（+28.4%）

### 3.8 知识蒸馏

$$
\mathcal{L}_{KD} = KL(P_{teacher} \| P_{student})
$$

- LinkedIn：8B oracle judge → SLM，75x 吞吐提升
- PyLate：bge-reranker-v2-gemma → ColBERT

---

## 四、RAG 系统演进

### Phase 1: Naive RAG
- 固定 chunk size (100-200 words)，dense retrieval，direct generation
- 问题：Context fragmentation，limited reasoning depth

### Phase 2: Long-Context RAG (LongRAG)
- 4K-token retrieval units（30x 大 chunk），index 22M → 700K
- Answer recall: +19-25%

### Phase 3: Graph-Enhanced RAG (GraphRAG)
- KG extraction → community hierarchy → multi-level summaries
- 3x accuracy on complex reasoning

### Phase 4: Enterprise RAG (RAGFlow)
- Deep document parsing (DeepDoc) for complex layouts
- Multimodal: PDF, Word, Excel, images
- Hybrid retrieval: full-text + vector + PageRank

### 选型建议

| 场景 | 方案 | 关键优势 |
|------|------|---------|
| 简单 QA | LongRAG | 最简架构，长上下文 LLM 足够 |
| 复杂推理 | GraphRAG | 多跳推理，关系理解，3x 精度 |
| 企业文档 | RAGFlow | 复杂格式解析，混合检索 |
| 高效率 | RankRAG | 统一重排+生成，-35% 延迟 |

---

## 五、LLM Ranking 效率技术（LinkedIn 方案）

| 技术 | 作用 | 效果 |
|------|------|------|
| Prefill-Only 推理 | 消除 autoregressive decode 开销 | 核心吞吐提升来源 |
| 结构化剪枝 | 减小模型参数量 | 配合实现 75x |
| 离线摘要压缩 | 减少在线 context 长度 | 降低 prefill 计算量 |
| 共享前缀 KV Cache | 相同 query 候选共享 context | 避免重复计算 |

---

## 六、论文速览

| 论文 | 机构 | 核心贡献 | 关键指标 |
|------|------|---------|---------|
| Semantic Search At LinkedIn | LinkedIn | 75x 推理吞吐；多教师蒸馏 SLM | 75x 吞吐 |
| PyLate | LightOn AI | GTE-ModernColBERT Late Interaction | BEIR avg 54.89 |
| BRIGHT | 多机构 | 推理密集型检索 benchmark | 揭示模型盲区 |
| Rank-R1 | CSIRO/UWaterloo | GRPO RL reranker | BRIGHT avg .205 |
| LongRAG | - | 4K-token chunks + Long-context | +19-25% recall |
| GraphRAG | - | KG + community hierarchy | 3x accuracy |
| Search-o1 | - | 推理-检索交织 | Agentic RAG |
| RankRAG | - | 统一 Reranker+Generator | -35% 延迟 |

---

## 七、面试 Q&A

### Q1: Dense Retrieval vs BM25 核心对比？
BM25：词汇匹配，高精度、可解释、无需训练；Dense：语义匹配，泛化好、跨语言强。生产通常混合。

### Q2: Late Interaction 与 Dense/Cross-Encoder 的本质区别？
Dense 压缩到单向量（lossy）；Cross-Encoder 实时全交互（无法预计算）；Late Interaction 保留每个 token 向量并预计算文档，MaxSim 实现 token-level 匹配。

### Q3: HyDE 为什么有效？
Query 和 Document 语言风格不同（疑问句 vs 陈述句），假设文档更接近真实文档，embedding 空间更近。

### Q4: RAG vs Long-Context LLM？
RAG 82% 优于 direct LLM（U-NIAH），小模型获益更大。趋势：larger chunks + longer context 是最佳实践。

### Q5: GraphRAG 的优势场景？
多跳推理、实体关系理解。代价：graph construction overhead，更复杂 pipeline。

### Q6: Agentic RAG vs 传统 RAG？
传统 RAG：单跳事实问答；Agentic RAG：多跳推理、需要中间验证、动态知识整合。

### Q7: RAG Hallucination 来源？
(1) 检索失败 → 模型编造；(2) 检索噪声 → 混淆；(3) 过时知识。缓解：Reranking + Faithfulness 约束 + 多轮检索验证。

### Q8: Query 改写技术方案？
LLM 扩展改写 + HyDE（假设答案文档）+ 意图分类 + RRF 多路融合。高频 Query 缓存。

### Q9: 如何评估 Embedding 模型泛化能力？
MTEB：56 个数据集，覆盖分类、聚类、检索、重排、语义相似。

### Q10: 全链路 LLM 搜索系统设计？
分层：(1) 召回 BM25+向量 top-100-200；(2) 粗排 300M Cross-Encoder top-20-50；(3) 精排 SLM/Rank-R1 Prefill-Only；(4) QU 层蒸馏小模型；(5) LLM judge 评估；(6) 大模型超时 fallback 到 LTR。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[01_检索范式_稀疏到混合到稠密|检索范式]]
- [[03_推理增强检索与重排|推理增强检索与重排]]
