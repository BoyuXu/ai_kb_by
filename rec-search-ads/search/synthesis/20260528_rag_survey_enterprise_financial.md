# RAG 系统综述与领域应用前沿 — 2026-05-28

> 综合 4 篇论文：RAG 综合综述 (2410.12837)、RAG 系统进展评估 (2507.18910)、金融领域 Metadata-Driven RAG (2510.24402)、企业结构化数据 RAG (2507.12425)

## 一、技术演进

RAG 从 2020 年 Lewis et al. 提出至今，经历了清晰的演进路径：

| 阶段 | 时间 | 代表 | 核心思路 |
|------|------|------|---------|
| Naive RAG | 2020-2022 | RAG (Lewis), REALM | 检索 + 拼接 + 生成，一次检索 |
| Advanced RAG | 2022-2024 | Self-RAG, CRAG, RAPTOR | 多轮检索、自适应检索、树状索引 |
| Modular RAG | 2024-2025 | 各种 pipeline 组合 | 检索/重排/压缩/生成模块化解耦 |
| **Domain-Specific RAG** | 2025-2026 | Metadata-Driven, Enterprise RAG | 领域元数据增强、结构化数据处理 |
| **Agentic RAG** | 2025-2026 | A-RAG, Tool-augmented | LLM 自主决定检索策略 |

两篇综述 (2410.12837, 2507.18910) 共同指出的 **核心挑战**：
1. **幻觉缓解不彻底** — RAG 降低但不消除幻觉，需 attribution + verification
2. **多跳推理** — 需要多次检索 + 推理链协同
3. **结构化数据处理** — 表格/JSON/SQL 的检索与理解
4. **评估标准不统一** — EM/F1/BLEU 无法全面衡量 RAG 质量

## 二、核心公式

**1. RAG 基本范式：**

P(y|x) = ∑_d P(d|x) · P(y|x,d)

其中 P(d|x) 为检索模型（通常 bi-encoder），P(y|x,d) 为生成模型。

**2. Metadata-Enriched Embedding (2510.24402 核心创新)：**

e_chunk = Embed(text_chunk ⊕ metadata_context)

metadata_context 包含：文档标题、章节位置、时间戳、实体标签。

关键发现：将 metadata 直接嵌入 chunk 文本（contextual chunks）比 post-hoc filtering 效果更好。

**3. Hybrid Retrieval Score (2507.12425)：**

score(q, d) = α · sim_dense(q, d) + (1 - α) · BM25(q, d)

使用 all-mpnet-base-v2 做 dense，SpaCy NER 做 metadata-aware filtering，cross-encoder reranking 做精排。

## 三、领域应用深度分析

### 金融领域 RAG (2510.24402 — Metadata-Driven)

**问题特殊性：**
- 财务文档冗长（10-K 报告几百页），相关信息极度稀疏
- 数据高度结构化（表格、数值、交叉引用）
- 精度要求极高（差一个数字意味着完全错误）

**解法分层：**
```
Pre-retrieval: LLM 生成 metadata → 过滤无关文档
Indexing:      Contextual chunks (metadata 嵌入文本)
Retrieval:     Dense + metadata filter
Post-retrieval: Cross-encoder reranking (最关键)
Generation:    带 attribution 的回答
```

**关键结论：** Reranker 是精度的核心保障；但 contextual chunks 对召回提升最大。

### 企业内部数据 RAG (2507.12425)

**挑战：**
- 异构格式（HR 记录、结构化报告、表格文档）
- 表格的行列关系需要显式保留
- 需要 human-in-the-loop 持续反馈

**方案亮点：**
- Semantic chunking（保持文本连贯性）
- 表格专用处理：保留 row-column 关系的特殊 chunk 策略
- Quantized indexing 提升效率
- Conversation memory 支持多轮追问

**结果：** Precision@5 90% (+15%)，Recall@5 87% (+13%)，MRR 0.85 (+16%)

## 四、RAG 系统评估框架

两篇综述提出的评估维度：

| 维度 | 指标 | 说明 |
|------|------|------|
| 检索质量 | Precision@K, Recall@K, MRR | 找到正确文档 |
| 生成质量 | EM, F1, ROUGE | 答案正确性 |
| 忠实度 | Attribution, Faithfulness | 回答是否基于检索内容 |
| 鲁棒性 | Noise Robustness | 检索到噪声文档时的表现 |
| 端到端 | FinanceBench, NQ, TriviaQA | 领域综合评测 |

## 五、工业实践要点

1. **Chunking 策略决定上限** — semantic chunking >> fixed-size chunking，且 metadata 注入 chunk 效果最好
2. **Reranker 是 ROI 最高的组件** — cross-encoder reranking 在各场景下稳定提升 5-15% 精度
3. **领域适配三件套** — 领域 embedding 微调 + 领域 metadata schema + 领域评测集
4. **混合检索仍是工业标配** — Dense + BM25 互补，前者抓语义，后者抓关键词/实体

## 六、面试考点

**Q1: RAG 的三代演进是什么？**
A: Naive RAG（单次检索拼接）→ Advanced RAG（迭代检索、自适应检索、查询改写）→ Modular RAG（模块解耦、可组合 pipeline）。当前趋势是 Domain-Specific RAG（领域元数据增强）和 Agentic RAG（LLM 自主决策检索策略）。

**Q2: 为什么 metadata 注入 chunk 比 post-hoc filter 好？**
A: 因为 embedding 本身会编码 metadata 信息（如"这是2023年Q4财报"），使得语义相似度计算天然包含元数据匹配。Post-hoc filter 是先检索再过滤，可能漏掉语义匹配但 metadata 不完全一致的文档。

**Q3: 金融/企业 RAG 与通用 RAG 的核心区别？**
A: (1) 结构化数据占比高，需要表格感知的 chunking；(2) 精度要求极高，必须有 reranker + attribution；(3) 数据敏感性高，需要 human-in-the-loop 验证。

**Q4: RAG 评估为什么难？**
A: 因为涉及检索和生成两个阶段，且需要同时评估正确性（答对了吗）、忠实度（基于检索内容吗）、完整性（遗漏了吗）。单一指标无法覆盖全部维度。

---

> **关联概念页**: [[embedding_everywhere]] | [[attention_in_recsys]]
> **关联 synthesis**: [[20260519_dense_retrieval_theory_and_agentic_rag]] | [[20260513_agentic_rag_denoising_reranking]] | [[20260504_rag_reranking_passage_selection]]
