# RAG 检索优化：从 Naive RAG 到 Advanced RAG

> 来源：技术综述 | 日期：20260316 | 领域：search

## 问题定义

基础 RAG（Retrieve-then-Generate）存在诸多问题：
1. **检索不准确**：单次检索可能找不到最相关的 chunk，或检索到噪声文档。
2. **Chunk 边界问题**：固定长度切分导致关键信息被切断。
3. **多跳推理**：需要综合多个文档才能回答的问题，单次检索不够。
4. **上下文窗口瓶颈**：检索太多文档超过 LLM context limit；太少则信息不足。
5. **幻觉风险**：LLM 在检索结果质量差时仍会生成内容。

## 核心方法与创新点

### Pre-Retrieval 优化（检索前）

**1. 查询改写（Query Rewriting）**：
```python
# 用 LLM 生成多个查询变体
expanded_queries = llm.generate(f"""
Generate 3 different search queries for: {user_query}
Focus on: different phrasings, key entities, sub-questions
""")
# 多查询检索并合并结果
results = union([retrieve(q) for q in expanded_queries])
```

**2. HyDE（假设文档扩展）**：
先让 LLM 生成假设答案，用答案 embedding 检索，解决 query 太短的问题。

**3. Step-Back Prompting**：
将具体问题抽象化（"特斯拉 Model 3 续航" → "电动车续航影响因素"），检索更广泛的背景知识。

### Retrieval 优化

**1. Chunk 策略**：
- **固定大小**：简单但可能切断语义。
- **Sentence-level**：按句号分割，语义完整但 chunk 大小不均。
- **Semantic Chunking**：用 embedding 相似度检测语义边界，自适应切分。
- **Parent-Child Chunk**：存储小 chunk 便于精确检索，返回大 chunk 提供上下文。

**2. 重排序（Reranking）**：
检索 Top-50，用 Cross-encoder 精排，只取 Top-5 送 LLM。

**3. 迭代检索（Iterative RAG）**：
```
query → retrieve → generate partial answer → identify missing info
→ new query → retrieve → refine answer → repeat N times
```

### Post-Retrieval 优化（检索后）

**1. 上下文压缩（Contextual Compression）**：
用 LLM 从检索结果中提取仅与 query 相关的句子，减少上下文噪声。

**2. 答案验证（Self-RAG）**：
生成答案时同时生成 **Critique Token**（[Retrieve], [Relevant], [Supported], [Utility]），让模型自我评估是否需要额外检索。

**3. CRAG（Corrective RAG）**：
用评估器判断检索质量，质量差时自动切换到 Web Search 获取额外信息。

## 实验结论

- HyDE 在 NQ 数据集：相比直接 DPR +5% Recall@5。
- Reranking（BGE-reranker）：在 RAG pipeline 中最终答案准确率 +8-12%。
- Self-RAG vs 标准 RAG：幻觉率降低 25%，在开放域 QA 准确率 +7%。

## 工程落地要点

- 生产 RAG pipeline 推荐栈：LangChain/LlamaIndex + Qdrant/Weaviate + BGE-M3（embedding）+ BGE-reranker（rerank）。
- Chunk size 经验：embedding retrieval 用 256-512 token；LLM context 用 512-2048 token（用 parent-child 策略）。
- 评估框架：RAGAS（检索召回率、答案忠实度、答案相关性三维度自动评估）。
- 延迟优化：rerank 是主要瓶颈，用批处理 + GPU 加速，或用轻量 reranker（ms-marco-MiniLM）。

## 常见考点

- Q: RAG 的核心评估指标有哪些？
  A: (1) Context Recall：检索到的文档是否包含答案所需信息；(2) Context Precision：检索到的文档中有多少比例真正有用（避免噪声）；(3) Answer Faithfulness：生成答案是否忠实于检索内容（衡量幻觉）；(4) Answer Relevance：生成答案是否回答了用户问题。RAGAS 框架自动计算这些指标。

- Q: Self-RAG 与标准 RAG 的区别？
  A: Self-RAG 训练 LLM 输出特殊 Reflection Token，在生成过程中动态判断：是否需要检索（[Retrieve]）、检索结果是否相关（[Relevant]/[Irrelevant]）、生成内容是否有支撑（[Supported]）。标准 RAG 是固定流程（检索→生成），Self-RAG 是自适应的。

- Q: 如何处理 RAG 中的长文档？
  A: (1) Hierarchical 切分（章节→段落→句子层级）；(2) Sliding Window（重叠切分保留上下文）；(3) Summary + Detail（先用摘要检索，再获取详细内容）；(4) 长上下文 LLM（128K token 上下文直接塞入，但成本高）；(5) LongRAG（按"retrieval unit"而非固定 chunk 检索）。
