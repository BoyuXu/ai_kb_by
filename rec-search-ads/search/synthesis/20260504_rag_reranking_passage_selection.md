# RAG Reranking 前沿：Passage Selection、Graph Reranking 与 Listwise 压缩 (2025-2026)

> 覆盖论文：DPS (2508.09497), GraphER (2603.24925), AdaRankLLM (2604.15621), RRK (2604.26483), RAG Fusion at Scale (2603.02153)
> 交叉引用：[[20260503_rag_maturity_and_reasoning_reranking.md]]、[[20260420_retrieval_reranking_distillation.md]]

---

## 1. 技术演进总览

RAG reranking 正经历从「独立打分 + 固定 Top-K」到「全局感知 + 动态选择 + 压缩表示」的范式跃迁：

| 阶段 | 代表方法 | 核心思路 | 局限 |
|------|----------|----------|------|
| Point-wise | BGE-reranker, Cohere | 独立 query-doc 打分 | 忽略段落间依赖 |
| Listwise | RankGPT, RankLLaMA | LLM 对候选列表排序 | 长文本 token 开销大 |
| **Dynamic Selection** | **DPS** | 从排序走向选择：有监督学习段落子集 | 需标注数据 |
| **Graph-enhanced** | **GraphER** | 图结构建模段落间多维近邻关系 | 离线索引开销 |
| **Adaptive Listwise** | **AdaRankLLM** | Passage Dropout + 蒸馏自适应过滤 | 蒸馏依赖教师模型 |
| **Compressed Listwise** | **RRK** | 软压缩 multi-token 表示 + listwise | 压缩可能丢失细节 |
| **Fusion at Scale** | **RAG Fusion** | 多查询 + RRF 融合工业部署分析 | recall 增益被 rerank 抵消 |

---

## 2. 逐篇精读

### 2.1 DPS: Dynamic Passage Selector (2508.09497)

**核心问题**：传统 reranker 用固定 K 截断，小 K 遗漏关键证据、大 K 引入噪声，对多跳推理尤其致命。

**方法**：
- 将 passage selection 建模为**有监督子集选择问题**，而非排序问题
- 模型捕获 inter-passage dependencies（段落间依赖），动态决定选多少、选哪些
- 即插即用（plug-and-play），不需修改 RAG pipeline 其他组件

**形式化**：给定 query $q$ 和候选段落集合 $\mathcal{P} = \{p_1, ..., p_n\}$，DPS 学习选择函数：

$$\mathcal{S}^* = \arg\max_{\mathcal{S} \subseteq \mathcal{P}} f_\theta(q, \mathcal{S})$$

其中 $f_\theta$ 同时建模段落相关性和段落间互补性。

**关键结果**：
- MuSiQue 数据集：F1 比 Qwen3-reranker 高 30.06%，比 RankingGPT 高 15.4%
- 5 个 benchmark 上一致超越 SOTA reranker

**启示**：从 ranking 到 selection 是 paradigm shift——不再是「排好序取前 K」，而是「直接选最优子集」。

---

### 2.2 GraphER: Graph-Based Enrichment and Reranking (2603.24925)

**核心问题**：语义搜索在证据分散于多源时效果下降，纯向量相似度无法捕获结构性关联。

**方法**：
- **离线阶段**：为每个数据对象构建图结构 enrichment，捕获多种 proximity 信号（不仅是语义相似度）
- **在线阶段**：query time 对候选对象做 graph-based reranking
- **无需知识图谱**：与标准 vector store 无缝集成
- **Retriever-agnostic**：不依赖特定检索器

**关键设计**：
- 图节点 = 文档段落，边 = 多维近邻关系（语义、结构、引用等）
- Reranking 时利用图传播信号，将孤立的相关段落通过图路径关联起来

**优势**：
- 延迟开销可忽略（negligible latency overhead）
- 不需要维护独立知识图谱
- 多个检索 benchmark 上效果显著

**与 GraphRAG 的区别**：GraphRAG 依赖显式知识图谱构建（实体-关系三元组），GraphER 仅需隐式图结构，工程成本低得多。

---

### 2.3 AdaRankLLM: Adaptive Listwise Ranking (2604.15621)

**核心问题**：随着 LLM 对噪声的鲁棒性增强，adaptive retrieval 的必要性需要重新审视。

**方法**：
- **Passage Dropout**：训练时随机丢弃部分段落，迫使 ranker 学习在不完整信息下做排序和过滤
- **两阶段渐进蒸馏**（Progressive Distillation）：
  - Stage 1：从强 LLM 蒸馏 listwise ranking 能力
  - Stage 2：蒸馏 adaptive filtering 能力（何时该过滤、何时该保留）
- **数据采样 + 增强**：构造多样化训练样本

**关键发现**：
- 对弱模型：adaptive retrieval 是**关键噪声过滤器**，帮助克服模型局限
- 对强推理模型：adaptive retrieval 变成**性价比优化器**，减少 context 开销但不影响效果
- 3 个数据集、8 个 LLM 上验证：AdaRankLLM 在大多数场景达到最优，且显著降低 context overhead

**形式化**：Passage Dropout 训练目标：

$$\mathcal{L} = \mathbb{E}_{D \sim \text{Dropout}(\mathcal{P})} \left[ \ell_{\text{listwise}}(q, D) + \lambda \cdot \ell_{\text{filter}}(q, D) \right]$$

其中 $\ell_{\text{listwise}}$ 是排序损失，$\ell_{\text{filter}}$ 是过滤决策损失。

---

### 2.4 RRK: Efficient Listwise Reranking with Compressed Documents (2604.26483)

**核心问题**：LLM-based listwise reranking 的 token 开销随文档数和长度线性增长，延迟不可控。

**方法**：
- 将文档压缩为**多 token 固定大小的嵌入表示**（soft compression）
- 直接在压缩表示上做 listwise reranking，跳过原文 token 处理
- 基于 SOTA reranker 蒸馏训练，无需大规模标注数据

**核心公式**：文档 $d_i$ 被压缩为 $m$ 个 token 的表示：

$$\mathbf{E}_i = \text{Compress}(d_i) \in \mathbb{R}^{m \times h}$$

Listwise reranking 直接在压缩表示上执行：

$$\pi^* = \text{Rank}_\theta(q, \{\mathbf{E}_1, ..., \mathbf{E}_n\})$$

**关键结果**：
- 8B 参数模型比 0.6-4B 小模型 reranker **快 3x-18x**
- 效果持平或超越更小的 reranker
- 长文档 benchmark 上优势更加明显（压缩收益更大）

**工程意义**：这是 reranking 效率的突破——大模型反而比小模型快，因为压缩表示大幅减少了实际处理的 token 数。

---

### 2.5 RAG Fusion at Scale: Industry Deployment Lessons (2603.02153)

**核心问题**：多查询检索 + RRF 融合在学术 benchmark 上有效，但在生产环境中表现如何？

**方法**：在 Dell Technologies 企业知识库上评估 RAG Fusion pipeline：
- Multi-query retrieval（多查询生成）
- Reciprocal Rank Fusion (RRF)
- 固定检索深度 + reranking 预算 + 延迟约束

**RRF 公式**：

$$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}$$

其中 $\mathcal{R}$ 是多个检索结果列表，$k$ 通常取 60。

**关键发现（反直觉）**：
1. Retrieval fusion **确实提升了 raw recall**
2. 但这些增益在 reranking + truncation 后**被大幅抵消**
3. Fusion 引入额外冗余和延迟开销
4. 在有 reranking 预算约束的生产环境中，fusion 的 ROI 存疑

**实践启示**：
- 不要盲目加 fusion，先评估 reranking 是否已经足够
- 在延迟敏感场景，fusion 的额外开销可能不值得
- recall 提升 ≠ 最终生成质量提升

---

## 3. 技术对比矩阵

| 维度 | DPS | GraphER | AdaRankLLM | RRK | RAG Fusion |
|------|-----|---------|------------|-----|------------|
| 方法类型 | 子集选择 | 图重排 | 自适应 listwise | 压缩 listwise | 融合策略 |
| 训练需求 | 有监督 | 离线索引 | 蒸馏 | 蒸馏 | 无训练 |
| 延迟影响 | 中等 | 极低 | 中等 | 大幅降低 | 增加 |
| 动态 K | 是 | 否 | 是 | 否 | 否 |
| 段落间依赖 | 显式建模 | 图传播 | 隐式(dropout) | 压缩交互 | RRF融合 |
| 多跳支持 | 强 | 强 | 中 | 中 | 弱 |
| 工程侵入性 | 低(plug-play) | 低(无需KG) | 中(需蒸馏) | 中(需压缩) | 低 |

---

## 4. 面试高频 Q&A

### Q1: RAG 中 reranking 的主要范式有哪些？各自优劣？

**A**: 三大范式：
1. **Point-wise**（BGE-reranker）：独立打分，快但忽略段落间关系
2. **Listwise**（RankGPT）：全局排序，效果好但 token 开销大
3. **Selection-based**（DPS）：直接选子集而非排序，动态 K，最适合多跳场景

新趋势是**压缩 listwise**（RRK）：把文档压缩后再 listwise，兼顾效果和效率。

### Q2: 为什么 RAG Fusion 在生产环境中效果不如预期？

**A**: 三个原因：
1. Recall 增益被 reranking truncation 吸收——reranker 已经能找到好文档
2. 多查询引入冗余文档，浪费 reranking 预算
3. 额外延迟（多次检索 + 融合）在生产 SLO 下不可接受

**关键结论**：先把 reranker 做好，再考虑 fusion。

### Q3: GraphER 与 GraphRAG 有什么本质区别？

**A**:
- **GraphRAG**：需要显式构建知识图谱（实体抽取、关系抽取），维护成本高
- **GraphER**：隐式图结构（段落间多维近邻），无需 KG，与标准 vector store 兼容
- GraphER 更适合工程化落地，GraphRAG 适合知识密集型场景

### Q4: 如何理解从 ranking 到 selection 的范式转变（DPS）？

**A**: 传统 reranker 输出排序，再用固定 K 截断。问题在于：
- K 太小：多跳推理时遗漏关键证据
- K 太大：引入噪声降低生成质量

DPS 直接学习最优子集，同时考虑段落相关性和互补性。本质上是从 $O(n \log n)$ 排序问题变成 $2^n$ 子集选择问题（通过有监督学习近似求解）。

### Q5: RRK 为什么大模型反而比小模型快？

**A**: 因为 RRK 将文档压缩为固定大小的 soft token 表示。处理的实际 token 数不再取决于原始文档长度，而是取决于压缩后的 token 数（固定的）。8B 模型处理少量压缩 token 比 4B 模型处理完整文档 token 快得多。

---

## 5. 开放问题与趋势

1. **Selection vs Ranking 的统一**：能否同时学习排序和选择？
2. **图结构 + 压缩表示的结合**：GraphER 的图信号 + RRK 的压缩表示
3. **Adaptive 的极限**：当 LLM 足够强时（如 o1-级推理模型），reranking 是否还有必要？AdaRankLLM 的发现暗示对强模型，reranking 更多是效率优化而非质量提升
4. **端到端训练**：从检索到生成的端到端优化，而非分离的 retrieve-rerank-generate pipeline

---

*Updated: 2026-05-04 | 概念页关联：[[attention_in_recsys.md]]、[[embedding_everywhere.md]]*
