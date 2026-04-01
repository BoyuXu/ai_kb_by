# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2)

> 来源：https://arxiv.org/abs/2502.14802 | 领域：llm-infra | 学习日期：20260401

---

## 问题定义

**核心挑战**：人类智能的核心能力之一是持续学习（Continual Learning）——能持续获取、组织、并利用新知识。而现有 LLM 存在以下根本性问题：

1. **参数化知识静态**：LLM 知识固化在训练权重中，无法动态更新（灾难性遗忘 Catastrophic Forgetting）
2. **RAG 的局限性**：检索增强生成（RAG）虽是主流解决方案，但基于向量相似度的检索无法模拟人类记忆的**动态性**和**关联性**
3. **知识图谱 RAG 的代价**：HippoRAG v1 及类似方法（GraphRAG）引入知识图谱提升关联推理能力，但在基础事实记忆任务上性能大幅下降

**三类记忆任务定义**：
- **Factual Memory（事实记忆）**：精确事实检索，如"X 的 CEO 是谁"
- **Sense-making Memory（语义理解记忆）**：需要理解上下文语义，不只是字面匹配
- **Associative Memory（关联记忆）**：跨文档、跨概念的链式推理，类似人类联想记忆

现有方案的困境：提升关联记忆往往以牺牲事实记忆为代价，无法三者兼顾。

---

## 核心方法与创新点

### HippoRAG 2 架构

基于 HippoRAG v1（Personalized PageRank on 知识图谱）的增强版本，核心改进三个方向：

#### 1. Deeper Passage Integration（更深层段落融合）

HippoRAG v1 将文档抽取为三元组（subject, predicate, object）构建知识图谱，但丢失了大量上下文信息。

HippoRAG 2 改进：
- **保留原始 passage 节点**，与知识图谱实体节点并存
- 建立 passage ↔ 实体双向链接，实现段落级与实体级的联合检索
- 检索路径：Query → Seed Entities → PageRank扩散 → 关联Passages

```
传统 RAG:  Query → [向量检索] → Top-K Passages → LLM
HippoRAG2: Query → [实体识别] → 知识图谱 → [PageRank] → 关联实体 + 原始Passage → LLM
```

#### 2. Personalized PageRank 增强

核心算法继承自 HippoRAG v1，但改进了种子节点选择和 PageRank 参数：

- **PPR（Personalized PageRank）**：以 Query 相关实体为起始节点，在知识图谱上进行带重启的随机游走
- 重启概率 α（通常 0.85）平衡了探索深度与起点偏置
- 收敛后的节点分数反映"与 Query 的关联程度"

**PPR 公式**：

$$
\pi = \alpha \cdot M^T \pi + (1-\alpha) \cdot e_s
$$

其中 $M$ 是图的转移矩阵，$e_s$ 是 Query 种子节点的初始分布，$\alpha$ 为阻尼因子。

#### 3. Online LLM Integration（在线 LLM 增强）

HippoRAG v1 使用 LLM 离线抽取知识图谱，在检索时不调用 LLM。

HippoRAG 2 引入**在线 LLM**：
- 检索得到候选 passages 后，用 LLM 做二次重排序（Re-ranking）
- LLM 判断候选段落与 Query 的深层语义相关性
- 平衡精度与效率：仅对 Top-K 候选调用 LLM，而非全量检索

### 技术栈对比

| 方法 | 事实记忆 | 语义理解 | 关联记忆 | 计算成本 |
|------|---------|---------|---------|---------|
| 标准 RAG | ✅ 强 | ✅ 中 | ❌ 弱 | 低 |
| GraphRAG/KG-RAG | ❌ 退步 | ✅ 中 | ✅ 强 | 高 |
| HippoRAG v1 | ✅ 中 | ✅ 中 | ✅ 强 | 中 |
| **HippoRAG 2** | ✅ **强** | ✅ **强** | ✅ **强** | 中-高 |

---

## 实验结论

**核心指标提升**（相比 SOTA embedding model）：
- **关联记忆任务**：提升 **7%**（MuSiQue, 2WikiMultiHopQA 等多跳 QA）
- **事实记忆任务**：保持或超越标准 RAG 性能（解决了 v1 的事实记忆退步问题）
- **语义理解任务**：全面优于对比系统

**发表状态**：ICML 2025（顶级会议接收）

**关键消融实验发现**：
- Deeper Passage Integration 对事实记忆提升最关键
- Online LLM Re-ranking 对语义理解提升最关键
- PPR 对关联记忆提升最关键（vs 朴素 BFS/DFS）

---

## 工程落地要点

### 1. 系统架构决策

**适合 HippoRAG 2 的场景**：
- 知识库需要频繁更新（新文档持续接入）
- 需要跨文档多跳推理（如法律文献、科学文献）
- 对关联记忆要求高的应用（知识管理、研究辅助）

**不适合的场景**：
- 实时性要求极高（图构建有延迟）
- 纯事实查询（标准 RAG 已足够）
- 资源极度受限（图存储 + LLM 重排有额外开销）

### 2. 知识图谱构建 Pipeline

```python
# HippoRAG 2 离线构建流程
def build_hipporag2_index(documents):
    # Step 1: 实体抽取（用 LLM/NER）
    entities = extract_entities(documents)
    
    # Step 2: 关系抽取 → 三元组
    triples = extract_relations(entities, documents)
    
    # Step 3: 构建知识图谱（NetworkX / Neo4j）
    kg = build_knowledge_graph(triples)
    
    # Step 4: 保留 Passage 节点并建立链接
    for doc in documents:
        passage_node = add_passage_node(kg, doc)
        link_entities_to_passage(kg, entities_in_doc, passage_node)
    
    # Step 5: 预计算 embedding
    embed_all_nodes(kg)
    
    return kg
```

### 3. 检索流程实现

```python
def hipporag2_retrieve(query, kg, top_k=5):
    # Step 1: Query 实体识别（Seed Selection）
    seed_entities = identify_query_entities(query)
    
    # Step 2: 在图上运行 Personalized PageRank
    ppr_scores = personalized_pagerank(kg, seed_entities, alpha=0.85)
    
    # Step 3: 从高分节点中找关联 Passage
    candidate_passages = get_passages_from_high_score_nodes(ppr_scores, top_k * 3)
    
    # Step 4: Online LLM Re-ranking（关键步骤）
    reranked = llm_rerank(query, candidate_passages)
    
    return reranked[:top_k]
```

### 4. 非参数化持续学习的优势

与参数化方法（全量微调、LoRA 微调）对比：
- **无灾难性遗忘**：新文档直接加入图，不影响旧知识
- **可解释性**：推理路径可视化（通过图遍历路径）
- **知识可删除**：直接删除图节点即可实现"遗忘"（compliance/隐私场景）
- **增量更新**：O(新文档) 成本，无需重训模型

---

## 面试考点

**Q1：RAG 为什么难以模拟人类长期记忆？HippoRAG 2 的核心思路是什么？**

A：标准 RAG 基于向量相似度检索，只能捕捉语义相近性，无法处理跨文档关联推理（多跳推理）。人类记忆是关联网络式的——想到 A 会联想到 B、C。HippoRAG 2 用知识图谱 + Personalized PageRank 模拟这种关联传播，同时保留原始 passage 节点解决事实记忆问题，用在线 LLM 重排提升语义理解。三管齐下实现事实/语义/关联三类记忆的全面提升。

**Q2：Personalized PageRank 在 RAG 中如何应用？为何优于普通向量检索？**

A：PPR 以 Query 相关实体为起点，在知识图谱上做带重启随机游走（$\pi = \alpha M^T\pi + (1-\alpha)e_s$），收敛分数反映图上与 Query 的"关联程度"。优势：(1) 能沿关系边跨越语义鸿沟（多跳推理），(2) 天然支持关联传播，(3) 对关联记忆任务提升显著。局限：图构建成本高，实时性略差于向量检索。

**Q3：非参数化持续学习 vs 参数化持续学习（LoRA 微调）的权衡是什么？**

A：非参数化（HippoRAG 2）：无灾难性遗忘、可逆（删节点）、增量成本低、可解释；但检索延迟较高、依赖图质量、无法让模型"内化"知识（还需 RAG 调用）。参数化（LoRA）：知识内化到权重、推理时无额外检索延迟；但存在遗忘风险、更新成本高、不可逆。实践中推荐组合：稳定领域知识用 LoRA 微调，动态更新知识用 HippoRAG 2。

**Q4：HippoRAG 2 如何解决 v1 在事实记忆上的退步问题？**

A：v1 过度依赖知识图谱三元组，丢失了原始文档的细节上下文，导致事实记忆退步。v2 的关键修复是 Deeper Passage Integration：保留原始 passage 节点并与知识图谱实体双向链接。这样检索时既能通过图关系做关联推理，又能回溯到完整段落获取准确事实，两者不再对立。
