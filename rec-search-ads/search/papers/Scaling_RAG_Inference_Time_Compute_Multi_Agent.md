# Scaling RAG Systems with Inference-Time Compute via Multi-Agent

> 2025-2026 | 领域：Agentic RAG / Inference-Time Scaling | 相关论文：REBEL (2504.07104), A-RAG (2602.03442)

## 一句话总结

传统 RAG 在增加推理时间计算量时效果提升有限；通过多智能体协作和多准则重排序，可以让 RAG 系统的性能随 inference-time compute 有效 scaling。

## 问题背景

**Inference-Time Compute Scaling** 是 2024-2025 的核心趋势（o1, DeepSeek-R1），但在 RAG 场景下：
- 简单地增加检索文档数量 → 噪声增多，效果反而下降
- 简单地让 LLM 思考更久 → 不一定改善检索质量

**核心挑战**：如何让 RAG 系统像 reasoning model 一样，投入更多计算获得更好结果？

## 两条路线

### 路线 1: REBEL — Multi-Criteria Reranking

> arXiv:2504.07104 | Relevance Isn't All You Need

**关键洞见**：传统 reranker 只优化相关性 (relevance)，但 RAG 质量还取决于：
- **Diversity**：检索结果不能都说同一件事
- **Specificity**：文档粒度要匹配问题粒度
- **Freshness**：时效性

**方法**：
- Chain-of-Thought prompting 让 LLM 按多准则评分
- 可选 Multi-Turn 对话式细化
- 投入更多 inference compute → 评估更多文档/更多准则 → 效果持续提升

### 路线 2: A-RAG — Hierarchical Retrieval Interfaces

> arXiv:2602.03442 | Agentic RAG

**关键洞见**：让 agent 自主决定检索策略，而非固定 pipeline。

**三层检索接口**：
1. **Keyword Search**：BM25 精确匹配
2. **Semantic Search**：Dense retrieval 语义匹配
3. **Chunk Read**：细粒度段落读取

**Agent 行为**：
- 自适应选择检索工具（不是固定 pipeline）
- Multi-hop reasoning：一次检索结果不够 → 再次检索
- 性能随计算资源增加稳步提升

## 对比总结

| 维度 | REBEL (Reranking) | A-RAG (Agentic) |
|------|-------------------|-----------------|
| Scaling 机制 | 更多 CoT 评估轮次 | 更多检索-推理循环 |
| 架构改动 | Reranker 替换 | 整体 pipeline 重构 |
| 部署难度 | 低（即插即用） | 高（需要 agent 框架） |
| 多跳问题 | 有限支持 | 天然支持 |
| 计算效率 | 较高（单步） | 较低（多步） |

## 核心价值

RAG 的 inference-time scaling law：
```
RAG_quality ∝ f(compute) 当且仅当 compute 用在正确的地方：
  - 多准则评估（不仅仅是 relevance）
  - 自适应检索（不仅仅是一次检索）
  - 多跳推理（不仅仅是单轮 QA）
```

## 与其他工作的关系

- RAG 系统演进路径见 [[2026-04-09_rag_systems_evolution|RAG 系统演进]]
- Agent 在检索中的应用扩展了 [[检索三角_Dense_Sparse_LateInteraction|检索三角]] 的范围
- Reranking 方向见 [[搜索Reranker演进|搜索Reranker演进]]
