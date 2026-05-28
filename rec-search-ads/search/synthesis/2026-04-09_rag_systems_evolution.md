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

## 5. 2026-04 新进展

### Phase 5: RAG 安全防御 — ProGRank (2603.22934)
- **Corpus Poisoning**：攻击者注入恶意文档到检索库，被检索进 Top-K 后影响生成
- **ProGRank**：training-free 的 reranking 防御，对 query-passage 施加随机扰动，检测梯度不稳定性
- Poison 文档在扰动下梯度方差更大（优化痕迹暴露）
- 即插即用，不修改检索器，支持黑盒 surrogate 模式

### Phase 6: RAG 性能预测 — Predict When RAG Helps (2604.07985)
- **核心问题**：不是所有 query 都需要 RAG，如何预测 RAG 的增益？
- 三层预测器：Pre-retrieval < Post-retrieval < Post-generation
- 最优方案：建模 question-passage-answer 三元组语义关系的监督预测器
- **工业意义**：自适应路由——简单问题跳过 RAG 节省延迟和成本

### Phase 7: Inference-Time Scaling for RAG — REBEL + A-RAG
- 传统 RAG 增加计算量时效果提升有限
- **REBEL** (2504.07104)：多准则重排序（relevance + diversity + specificity），CoT 评估，compute↑→quality↑
- **A-RAG** (2602.03442)：Agent 自主选择检索策略（keyword/semantic/chunk read），multi-hop reasoning
- **RAG Scaling Law**：compute 用在多准则评估 + 自适应检索 + 多跳推理上才有效

---

## 6. 2026-05 新进展：RAG Survey 全景与企业落地

### Phase 8: RAG 综合综述与企业级应用

**Comprehensive RAG Survey [2410.12837] — Gupta et al. (2024.10)**
- 从开放域问答到当代 RAG 的全景综述
- 三维分类: 检索机制 / 生成模型 / 融合策略
- 开放挑战: 可扩展性、偏差、隐私
- 价值: 最完整的 RAG 技术分类框架

**Systematic Review of Key RAG Systems [2507.18910] — Oche et al. (2025.07)**
- 系统回顾从 ODQA 到 SOTA RAG 的演进
- 技术组件深度剖析: 检索器 / seq2seq 生成器 / 融合策略
- 核心贡献: 明确 RAG 如何缓解幻觉和知识过时问题
- Gap 分析: 多模态 RAG、实时更新、评估标准化仍待解决

**Metadata-Driven RAG for Financial QA [2510.24402] — Dadopoulos et al. (2025.10)**
- 金融长文档（SEC filings）上的多阶段 RAG 架构
- 核心创新: **Contextual Chunks** — 将 LLM 生成的元数据嵌入 chunk embedding
- 三阶段优化: pre-retrieval filtering → enriched embedding → cross-encoder reranking
- 在 FinanceBench 上最大增益来自 contextual chunks（元数据直接拼接文本再编码）
- 工业启示: 结构化文档需要元数据感知，不能只靠纯文本 embedding

**Advancing RAG for Structured Enterprise Data [2507.12425] — Cheerla (2025.07)**
- 企业内部数据（HR、报表、表格）的混合 RAG 框架
- 技术栈: all-mpnet-base-v2 + BM25 + SpaCy NER 元数据过滤 + Cross-encoder reranking
- 语义分块 + 表格结构保留（保持行列关系完整性）
- 量化索引 + 人在回路反馈 + 对话记忆
- 结果: Precision@5 提升 15% (90% vs 75%), MRR 提升 16% (0.85 vs 0.69)
- 面试要点: 企业 RAG 的难点在结构化/半结构化数据，不是纯文本

### RAG 架构选型决策树（更新版）
```
Query 类型判断
├── 简单事实 QA → 跳过 RAG，直接 LLM（Phase 6: Predict When RAG Helps）
├── 纯文本文档 QA → LongRAG / Naive RAG
├── 复杂推理 / 多跳 → GraphRAG
├── 企业结构化数据 → Hybrid RAG + 元数据过滤（Phase 8）
│   ├── 表格数据 → 保留行列结构 + semantic chunking
│   └── 金融长文档 → Contextual Chunks + pre-retrieval filtering
├── 安全敏感 → + ProGRank 防御（Phase 5）
└── 高吞吐需求 → + 自适应路由 + A-RAG（Phase 7）
```

### 面试新增 Q&A

**Q8: 企业 RAG 与通用 RAG 的核心差异？**
A: (1) 数据异构性: 表格、PDF、报表混合，不能只靠纯文本分块; (2) 元数据重要性: 文档标题/时间/类型等元数据对检索精度影响巨大; (3) 结构保持: 表格的行列关系不能被分块破坏; (4) 领域术语: 需要 NER + 领域词典辅助检索。

**Q9: Contextual Chunks 的原理和优势？**
A: 用 LLM 为每个 chunk 生成描述性元数据（来源、主题、关键实体），将元数据与原文拼接后再编码为 embedding。优势: embedding 包含了上下文信息，解决了"chunk 脱离文档语境后语义模糊"的问题。

**Q10: RAG 领域的三大开放问题？**
A: (1) 多模态 RAG — 图表/图像/音频的统一检索和理解; (2) 实时知识更新 — 如何增量更新索引而非全量重建; (3) 评估标准化 — 缺乏统一 benchmark，FinanceBench/NaturalQuestions 各有侧重。

---

## 相关概念

- [[embedding_everywhere|Embedding 技术全景]]
- [[ProGRank_Probe_Gradient_Reranking_RAG_Corpus_Poisoning|ProGRank: RAG 安全防御]]
- [[RAG_Performance_Prediction_QA|RAG 性能预测]]
- [[Scaling_RAG_Inference_Time_Compute_Multi_Agent|RAG Inference-Time Scaling]]
- [[20260513_pd_disaggregation_and_kvcache_quant|P/D Disaggregation 与 KV Cache]]
