# Agentic RAG 架构演进：潜空间推理、去噪导向 IR 与高效重排 (2026)

> 覆盖论文：LatentRAG (2605.06285), LLM-Oriented IR (2605.00505), A-RAG (2602.03442), Biomedical RAG Benchmark (2605.02520), FlashRank (2601.03258)
> 交叉引用：[[20260504_reasoning_retrieval_rag_traces.md]]、[[20260504_rag_reranking_passage_selection.md]]、[[20260503_rag_maturity_and_reasoning_reranking.md]]、[[concepts/embedding_everywhere.md]]、[[concepts/attention_in_recsys.md]]

---

## 1. 技术演进总览

5 篇论文标志 RAG 技术从"多轮文本推理"向"潜空间高效推理 + 去噪优先 + 分层 Agent"的范式跃迁：

```
Agentic RAG 前沿 (2026.05)
|
|-- 推理范式革新
|   |-- LatentRAG: 推理+检索从离散文本迁移到连续潜空间，延迟降 90%
|   |-- A-RAG: 分层检索接口（keyword/semantic/chunk）让 Agent 自主决策检索粒度
|
|-- IR 理论重构
|   |-- LLM-Oriented IR: 去噪成为 IR 第一性原则，四阶段瓶颈框架
|
|-- 检索策略实证
|   |-- Biomedical RAG Benchmark: Cross-Encoder Reranking 最优，Multi-Query 引入噪声
|   |-- FlashRank: 边际效用重排 + 查询扩展双阶段，NDCG@10 提升 5.4%
```

核心趋势：
1. **从文本空间到潜空间**：LatentRAG 证明 Agentic RAG 的推理和子查询生成可以在 hidden states 中完成，无需自回归生成，延迟从秒级降到百毫秒级
2. **去噪成为 IR 第一性原则**：LLM 消费检索结果时对噪声极度敏感，去噪（而非召回率）成为主要瓶颈
3. **Agent 化检索需要分层接口**：A-RAG 证明暴露多粒度检索工具（keyword/semantic/chunk read）比预定义 workflow 更高效
4. **Cross-Encoder Reranking 仍是王道**：生物医学 RAG 基准实证确认 Cross-Encoder 综合最优，盲目扩展查询反而引入噪声

---

## 2. 逐篇精读

### 2.1 LatentRAG: 潜空间推理检索 (2605.06285)

**Problem**: Agentic RAG 通过多轮推理和子查询迭代检索，解决复杂问题。但自回归生成中间思考和子查询带来巨大延迟（通常 5-10 秒/轮）。

**Method**:
- 将推理（thoughts）和子查询（subqueries）从离散语言空间迁移到连续潜空间
- 从 hidden states 直接生成 latent tokens，单次前向传播完成推理+检索
- 将 LLM 与 dense retrieval model 在潜空间对齐，支持端到端联合优化
- 并行潜空间解码机制（parallel latent decoding）将 latent tokens 翻译回自然语言，保持透明性

**Innovation**:
- 首次将 Agentic RAG 的「思考-查询-检索」循环压缩到单次 forward pass
- 潜空间对齐使检索模型无需额外训练即可接受 latent query tokens
- 解码机制保证了可解释性（可以看到模型在"想什么"）

**Results**: 7 个基准数据集上性能与显式 Agentic RAG 相当，推理延迟降低约 **90%**，接近传统单步 RAG 的速度。

**Keywords**: latent space reasoning, agentic RAG, dense retrieval alignment, speculative retrieval

---

### 2.2 LLM-Oriented IR: 去噪优先 (2605.00505)

**Problem**: 传统 IR 为人类用户设计，容忍一定噪声；LLM 消费检索结果时受限于 attention budget，噪声直接导致幻觉和推理失败。

**Method**:
- 提出四阶段 IR 挑战框架：inaccessible → undiscoverable → misaligned → unverifiable
- 论证去噪（denoising）——最大化上下文窗口内的可用证据密度和可验证性——是贯穿全 pipeline 的核心瓶颈
- 重新定义 LLM 时代 IR 的评估标准：从 recall@K 转向 evidence density per token

**Innovation**:
- 首次系统性论证「去噪优先」作为 LLM 导向 IR 的设计原则
- 四阶段框架提供了分析 RAG pipeline 瓶颈的统一视角
- 将 IR 目标从"找到相关文档"重新定义为"最大化可验证证据密度"

**Results**: 理论框架论文，通过多个 RAG 场景案例分析验证了去噪优先视角的解释力。

**Keywords**: denoising-first IR, evidence density, LLM-oriented retrieval, context window optimization

---

### 2.3 A-RAG: 分层检索接口 (2602.03442)

**Problem**: 现有 RAG 系统要么单次检索拼接（naive RAG），要么预定义工作流让模型按步执行（pipeline RAG），都未让模型参与检索决策。

**Method**:
- 暴露三种分层检索工具给 Agent：
  1. **Keyword Search**: 精确匹配关键词级信号
  2. **Semantic Search**: 语义级稠密检索
  3. **Chunk Read**: 读取完整文档块，获取上下文
- Agent 根据任务自主选择检索粒度和策略组合
- 信息天然按多粒度组织（keyword → sentence → chunk），检索接口应匹配这一层次

**Innovation**:
- 首个让 Agent 自主决策检索粒度的 RAG 框架
- 证明暴露多粒度工具比预定义检索 workflow 更有效
- 性能随模型能力提升而自动提升（workflow-free scaling）

**Results**: 多个开放域 QA 基准上一致超越现有方法，且检索 token 数相当或更少。

**Keywords**: hierarchical retrieval, agentic RAG, multi-granularity search, tool-use retrieval

---

### 2.4 Biomedical RAG Benchmark (2605.02520)

**Problem**: RAG 检索策略的选择缺乏受控实验对比，尤其在生物医学等专业领域。

**Method**:
- 固定生成模型（GPT-4o-mini）、向量库（ChromaDB）、embedding（text-embedding-3-small），仅变换检索策略
- 对比 5 种策略：Dense Vector、Hybrid BM25+Dense、Cross-Encoder Reranking、Multi-Query Expansion、MMR
- 250 个 BioASQ QA 对，4 个 DeepEval 指标

**Innovation**:
- 首个严格控制变量的 RAG 检索策略对比研究
- 揭示 Multi-Query Expansion 的反直觉结果：召回导向设计反而降低精度

**Results**:

| 策略 | 综合分 | Contextual Precision | 关键发现 |
|------|--------|---------------------|----------|
| Cross-Encoder Reranking | **0.827** | **0.852** | query-document 交互最优 |
| Hybrid BM25+Dense | 0.79x | 0.80x | 稳健的基线 |
| Dense Vector | 0.78x | 0.79x | 简单有效 |
| MMR | 0.77x | 0.78x | 多样性有限收益 |
| Multi-Query Expansion | 0.75x | 0.671 | **最差精度，噪声引入** |
| No context (ablation) | - | - | Answer relevancy 仅 0.287 |

**Keywords**: RAG benchmark, cross-encoder reranking, biomedical QA, retrieval strategy comparison

---

### 2.5 FlashRank: 快速重排 + 查询扩展 (2601.03258)

**Problem**: RAG 系统面临检索召回与 LLM 上下文窗口限制的矛盾；Cross-Encoder Reranking 虽好但延迟高。

**Method**:
- 两阶段 pipeline：
  1. **LLM-driven Query Expansion (QE)**: 扩展查询提升候选召回
  2. **FlashRank Reranking**: 边际效用重排器，建模文档效用为 relevance + novelty + brevity + cross-encoder evidence 的加权组合
- 动态选择最优证据子集，受 token budget 约束
- 100 个候选文档并行执行 < 60ms

**Innovation**:
- FlashRank 将重排从"排序问题"转化为"效用最大化+token预算约束"问题
- 首次在重排中同时建模 relevance、novelty、brevity 三个维度
- 工程友好：60ms / 100 docs，适合实时金融 RAG

**Results**:
- MS MARCO / BEIR：NDCG@10 提升 **5.4%**
- 生成准确率提升 **6-8%**
- 上下文 token 减少 **35%**
- 响应时间比 Cross-Encoder 快 **22%**

**Keywords**: FlashRank, marginal utility reranking, query expansion, token budget optimization

---

## 3. 横向对比与统一视角

### 3.1 Agentic RAG 的三条路线

| 路线 | 代表 | 核心思路 | 延迟 | 适用场景 |
|------|------|----------|------|----------|
| 显式多轮推理 | ReAct / Self-RAG | 文本推理 + 迭代检索 | 高（秒级/轮） | 复杂多跳问答 |
| 潜空间推理 | **LatentRAG** | Hidden states 直接推理+检索 | 低（接近单步 RAG） | 延迟敏感的复杂问答 |
| 分层工具调用 | **A-RAG** | 暴露多粒度检索工具 | 中 | 信息密度不均匀的语料 |

### 3.2 去噪 vs 召回的权衡

LLM-Oriented IR 和 Biomedical RAG Benchmark 共同指向一个结论：**在 LLM 消费场景下，精度 > 召回**。

- Multi-Query Expansion 提升召回但引入噪声，综合效果最差
- Cross-Encoder Reranking 最大化证据密度，综合最优
- FlashRank 通过 novelty + brevity 建模实现"去噪式重排"

### 3.3 面试高频问题

**Q: Agentic RAG 的主要延迟瓶颈在哪？如何优化？**
A: 瓶颈在自回归生成中间推理和子查询。LatentRAG 方案将推理迁移到潜空间，单次 forward pass 完成，延迟降 90%。

**Q: 为什么 Multi-Query Expansion 在 RAG 中效果不如预期？**
A: 召回导向设计引入的噪声被 LLM 放大为幻觉。LLM-Oriented IR 框架指出：LLM 的 attention budget 有限，evidence density per token 比 recall@K 更重要。

**Q: 如何设计生产级 RAG 的检索策略？**
A: Cross-Encoder Reranking 是安全选择（precision 最高）；如需低延迟可用 FlashRank（60ms/100docs）；如需处理复杂多跳问题，A-RAG 的分层接口比预定义 workflow 更 scalable。

---

## 4. 与现有知识的关联

- **检索范式演进**：本批论文延续了 [[01_检索范式_稀疏到混合到稠密.md]] 的路线，但增加了"潜空间检索"这一新维度
- **重排技术演进**：FlashRank 的效用建模 + token budget 约束是 [[05_Reranker演进与LTR.md]] 和 [[20260504_rag_reranking_passage_selection.md]] 的自然延伸
- **推理增强检索**：LatentRAG 和 A-RAG 将 [[20260504_reasoning_retrieval_rag_traces.md]] 中的推理增强检索推向了新的效率前沿
- **Embedding 对齐**：LatentRAG 的 LLM-retriever 潜空间对齐是 [[concepts/embedding_everywhere.md]] 的新应用场景
