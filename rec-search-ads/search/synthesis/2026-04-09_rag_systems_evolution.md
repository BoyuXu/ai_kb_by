# Synthesis: RAG Systems — 从 Naive RAG 到 Graph RAG
> Date: 2026-04-09 | Papers: LongRAG, GraphRAG, RAGFlow, U-NIAH, RAG Evaluation Survey

## 1. 技术演进 (Technical Evolution)

### Phase 1: Naive RAG
- Fixed chunk size (100-200 words), dense retrieval, direct generation
- 问题：Context fragmentation, limited reasoning depth

### Phase 2: Long-Context RAG (LongRAG)
- 4K-token retrieval units (30x larger), index 22M → 700K
- Light retriever + Heavy reader paradigm
- Answer recall: +19-25% improvement

### Phase 3: Graph-Enhanced RAG (GraphRAG)
- Knowledge graph extraction → community hierarchy → multi-level summaries
- 3x accuracy improvement on complex reasoning
- Multi-hop reasoning across disparate information

### Phase 4: Enterprise RAG (RAGFlow)
- Deep document parsing (DeepDoc) for complex layouts
- Multimodal: PDF, Word, Excel, images
- Hybrid retrieval: full-text + vector + PageRank

## 2. 核心公式 / Core Formulations

### Chunk Size Trade-off
```
Recall ∝ chunk_size (with capable reader)
Precision ∝ 1/chunk_size (with limited reader)
Optimal: chunk_size = f(reader_context_length)
```

### GraphRAG Community Scoring
```
Relevance(community, query) = Σ_entity∈community sim(entity, query) × PageRank(entity)
Answer = LLM(query, top_k_communities)
```

### RAG Evaluation Dimensions (Survey)
```
Quality = w1·Performance + w2·Factuality + w3·Safety + w4·Efficiency
Performance: F1, EM, ROUGE
Factuality: Faithfulness, Hallucination rate
Safety: Adversarial robustness, Toxicity
```

## 3. 工业实践 (Industrial Practices)

| System | Approach | Key Advantage |
|--------|----------|---------------|
| LongRAG | Large chunks + Long-context LLM | Simplicity, -31x index size |
| GraphRAG | Knowledge graph + Community hierarchy | Complex reasoning, 3x accuracy |
| RAGFlow | Deep parsing + Hybrid retrieval | Enterprise docs, multimodal |

### 选型建议
- **简单 QA**: LongRAG (最简架构，长上下文 LLM 足够)
- **复杂推理**: GraphRAG (多跳推理，关系理解)
- **企业文档**: RAGFlow (复杂格式解析，混合检索)

### 评估最佳实践 (U-NIAH + Survey)
- 使用合成数据集避免 pre-training 数据泄露
- Multi-needle 配置测试多文档综合能力
- RAG wins 82.58% vs direct LLM，尤其对小模型

## 4. 面试考点 (Interview Points)

**Q1: RAG vs Long-Context LLM?**
A: RAG 82% 优于 direct LLM (U-NIAH)。小模型获益更大。但长上下文 LLM 架构更简单 (LongRAG)。趋势：larger chunks + longer context 是最佳实践。

**Q2: GraphRAG 的优势场景？**
A: 需要多跳推理、理解实体关系的场景。如：enterprise knowledge base, cross-document analysis。代价：graph construction overhead, 更复杂的 pipeline。

**Q3: RAG 系统如何评估？**
A: 四维度：Performance (F1/EM), Factuality (faithfulness), Safety (robustness), Efficiency (latency)。使用 synthetic data 避免 leakage。关注 error patterns: hallucination, retrieval noise。

**Q4: Chunk size 如何选择？**
A: 取决于 reader 的上下文长度。LongRAG 证明 4K chunks + long-context reader 大幅优于 100-word chunks。关键：chunk size 应与 reader capability 匹配。

---

## 相关概念

- [[embedding_everywhere|Embedding 技术全景]]
