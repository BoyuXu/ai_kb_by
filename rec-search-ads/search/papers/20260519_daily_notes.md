# 搜索系统论文笔记 — 2026-05-19

> 来源：MelonEgg 每日学习（automated daily-cron）
> 范围：search 领域 5 篇

---

## 1. Neural Retriever-Reranker Pipelines for RAG over Knowledge Graphs in E-commerce

**来源：** https://arxiv.org/abs/2602.22219 （Abertay University，Dec 2025）
**领域：** Retriever-Reranker × KG-RAG × 电商

**Benchmark：** STaRK Semi-structured Knowledge Base（生产级电商数据集）

**对比内容：** 多种 retriever-reranker pipeline 组合（dense / sparse / hybrid + cross-encoder reranker）在电商 KG-QA 上的表现

**结果：** 比已发布 baseline 高 Hit@1 +20.4%、MRR +14.5%

**关键发现：**
- KG-RAG 的两大挑战：跨多跳连接的 retrieval scaling、generation 时保留上下文关系
- Cross-encoder reranker 是精度提升的核心，但与结构化数据的融合仍 underexplored

**面试考点：** Retriever-Reranker 经典流水线、KG-RAG 与 vanilla RAG 的差异、电商场景的多跳 QA 工程化

---

## 2. On the Theoretical Limitations of Embedding-Based Retrieval

**来源：** https://arxiv.org/abs/2508.21038 （Google DeepMind，Aug 2025）
**领域：** Dense Retrieval × 理论上限

**核心论点：** 把 embedding 维度 d 与"理论上可被某 query 命中的 top-k 子集数量"联系起来 — 该子集数量受 d 的约束（来自学习理论）

**Insight：** 任何固定维度的 single-embedding retriever 都存在"打不到"的 top-k 子集，这不是算法问题而是 representation capacity 问题

**实证：** 在 instruction-following retrieval benchmark 上观察到，要求 model 表征任意 relevance definition 时，单 embedding 表征不够

**工业含义：**
- 高复杂度 query 场景下，应考虑 multi-vector / late interaction / cross-encoder 弥补
- 增加 embedding 维度只能线性缓解，不是根本方案

**面试考点：** Bi-encoder 与 ColBERT / SPLADE 等 late interaction 的差异、Embedding 维度选择的工程权衡、Dense retriever 的失败 case 类型

---

## 3. R^{2k} is Theoretically Large Enough for Embedding-Based Top-k Retrieval

**来源：** https://arxiv.org/abs/2601.20844 （Jan 2026）
**领域：** Dense Retrieval × MED（Minimal Embeddable Dimension）

**与 #2 的关系：** #2 给悲观下界，#3 给乐观上界

**核心定理：** 把 m 个元素和"最多 k 个元素的子集"嵌入向量空间，**2k 维就足够** — 通过 cyclic polytope 构造可实现完美 top-k separation

**适用度量：** ℓ2、inner product、cosine 都有 tight bound

**Insight：** Embedding-based retrieval 的瓶颈是 **learnability**（如何学到那个理想的 query embedding 函数），不是 **geometric** 表征容量

**对比 #2：**
- #2：实际表征是否能区分任意子集 → 受限
- #3：理论上是否存在某个 embedding 能区分 → 2k 足够
- 综合：架构容量够，难点在训练目标与数据

**面试考点：** Cyclic polytope 与 top-k separable 的几何直觉、MED bound 推导思路、Bi-encoder 训练 loss 设计

---

## 4. A Survey of Model Architectures in Information Retrieval

**来源：** https://arxiv.org/abs/2502.14822 （Feb 2025）
**领域：** IR 架构综述

**时间线：** 2019 至今 IR 经历的最大范式转移 — Pre-trained Encoder (BERT) → Decoder-only LLM (2022+)

**两大维度梳理：**
1. **Backbone：** 词袋 → CNN/RNN → BERT/T5 → GPT/LLaMA → Multi-modal Encoder
2. **End-to-end 架构：** Bi-encoder（双塔）→ Cross-encoder → Late Interaction (ColBERT) → Generative Retrieval (DSI)

**核心议题：**
- LLM 的 zero-shot 与复杂推理优势
- 效率与可扩展性的架构优化（FlashAttention、Quantization）
- 多模态 / 多语言鲁棒性
- 自主搜索 agent 等新方向

**面试考点：** IR 范式演进、DSI（Differentiable Search Index）的原理与局限、Encoder-only vs Decoder-only 在检索中的取舍

---

## 5. A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces

**来源：** https://arxiv.org/abs/2602.03442 （Feb 2026）
**领域：** Agentic RAG × Tool-using LLM

**核心理念：** 让 LLM 自主决定怎么检索，而不是预定义流程

**三个检索工具：**
1. **Keyword search：** 关键词级精确匹配
2. **Semantic search：** dense embedding 检索
3. **Chunk read：** 直接读取文档片段（粒度最细）

**三大设计原则（A-RAG 同时满足）：**
1. **Autonomous Strategy：** Agent 根据任务特性动态选检索工具
2. **Iterative Execution：** 多轮检索，按中间结果自适应调整
3. **Interleaved Tool Use：** ReAct 范式（action → observation → reasoning）

**对比：** 单 shot retrieval（DPR/ColBERT）/ workflow RAG（Self-RAG / FLARE）都未能同时满足三原则

**实证：** 多个 open-domain QA benchmark 上以更少的 retrieved tokens 取得更好效果

**面试考点：** RAG 与 Agentic RAG 的边界、ReAct 在检索中的应用、多粒度 retrieval interface 的设计

---

## 当日小结

- **理论双珠：#2 + #3** 形成 dense retriever 表征能力的完整图景 — 单 embedding 的几何上限 + 2k 维理论充足性 + 训练 learnability 是真瓶颈
- **架构综述 #4** 给出 IR 范式演进时间线，可作为后续深读 baseline 列表
- **工业落地：#1 + #5** — 电商 KG-RAG 流水线选型 + Agentic RAG 的工具化检索框架，是当前 RAG 工程化的两条主路径
