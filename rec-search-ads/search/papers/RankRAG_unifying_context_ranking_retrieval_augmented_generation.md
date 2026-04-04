# RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation

> 来源：arXiv 2024 | 领域：search | 学习日期：20260404

## 问题定义

RAG 系统中，检索阶段返回的文档并非全部相关，直接喂给 LLM 会：
1. 引入噪声上下文（降低答案质量）
2. 占用宝贵 Context Window（影响推理效率）
3. 分散模型注意力（相关信息被不相关内容淹没）

通常的解法：额外训练一个重排（Reranker）模型，但引入了独立模型的维护成本和延迟。

**核心问题**：能否让同一个 LLM 同时做重排和生成？

## 核心方法与创新点

**RankRAG** 统一重排与生成，一模两用：

1. **统一指令微调**：
   - 同一 LLM 接受两种任务格式：
     - Ranking: `"Is this passage relevant to: {query}? Passage: {doc}. Answer Yes/No."`
     - Generation: `"Answer the question using the context: {docs}. Question: {q}"`
   - 交替训练两种任务（Joint Training）

2. **上下文精炼（Context Refinement）**：
   - 先用 LLM 的 Ranking 模式对检索文档打分
   - 保留 Top-K 相关文档（K << N）
   - 再用 Generation 模式基于精炼上下文生成答案
   
$$\text{RankRAG}(q, \mathcal{D}) = \text{Gen}\left(q, \text{Top-K}_{d \in \mathcal{D}}[\text{Rank}(q, d)]\right)$$

3. **渐进式精炼**：
   - 两阶段：粗粒度过滤（相关/不相关）→ 细粒度重排（相关度打分）
   - 减少最终进入生成阶段的文档数量

4. **数据增强**：
   - 合成训练数据：从 QA 数据集构造干扰文档（负例）
   - Hard Negative：语义相近但答案错误的文档

## 实验结论

- RAG QA（NQ/TriviaQA）: **+4.2%** EM vs Separate Reranker + LLM
- 相关文档精炼率: 过滤 70% 不相关文档，答案质量提升 **+6.8%**
- 推理延迟: 比 Separate Reranker 减少 35%（统一模型推理更快）

## 工程落地要点

- 排序阶段 Batch 处理（同一 Query，多个 Doc 并行）
- 精炼后文档数 K=3-5（平衡覆盖率和噪声）
- 联合训练 Ranking/Generation 比例 1:2（生成数据更丰富）
- 生产中：Ranking 阶段用 vLLM prefix sharing 提速

## 面试考点

1. **Q**: RAG 中为什么需要重排（Reranking）？  
   **A**: 检索（Bi-Encoder）是粗粒度相关性，速度快但精度有限。重排（Cross-Encoder/LLM）是细粒度，利用 query-doc 交互精确判断相关性，过滤噪声文档。

2. **Q**: RankRAG 相比独立 Reranker 的优势？  
   **A**: 统一模型（共享参数），排序能力与生成能力相互增强；减少系统复杂度（一个模型 vs 两个模型）；共享 KV Cache（prefix 复用）减少推理开销。

3. **Q**: 如何构造 Reranking 的训练数据？  
   **A**: Positive：QA 对中包含正确答案的文档；Hard Negative：BM25/Dense 检索到的高排名但无正确答案的文档（最有挑战性）。
