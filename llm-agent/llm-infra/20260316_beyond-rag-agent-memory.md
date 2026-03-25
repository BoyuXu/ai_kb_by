# Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation

> 来源：arxiv | 日期：20260316 | 领域：llm-infra

## 问题定义

传统 RAG（检索增强生成）将所有历史交互存为平面 chunk，Agent 需要回忆时进行关键词搜索。问题：
1. **关键词搜索失效**：Agent 的意图可能用某些词表达，但历史信息用不同的词存储（词汇差距）。
2. **信息冗余**：历史对话中大量细节噪声，检索时需要过滤信息。
3. **跨时间聚合困难**：需要综合多次对话中的信息才能回答，单次检索不够。

本文提出 **Decoupling & Aggregation** 范式，将 Agent 记忆解耦为多个维度，支持灵活查询。

## 核心方法与创新点

### 1. 记忆解耦（Memory Decoupling）

将交互记忆分层存储：
```
事实层（Fact Layer）：用户提及的具体信息
  - user_preferences: {"coffee": "espresso", "work_time": "9am-5pm"}
  - user_context: {"location": "SF", "job": "engineer"}

关系层（Relation Layer）：实体间的关系
  - knows(user, alice)
  - purchased(user, laptop)

意图层（Intent Layer）：用户长期目标
  - goal: "learn_ml"
  - interest: ["AI", "research"]

事件层（Event Layer）：时间序列事件
  - event: {date: 2026-03-15, type: "conversation", content: "..."}
```

### 2. 聚合查询（Aggregation Retrieval）

针对不同查询类型：
- **事实查询**："用户喜欢什么咖啡？" → Fact Layer 精确匹配
- **关系查询**："用户认识谁？" → Relation Layer 图遍历
- **意图查询**："用户的长期目标是什么？" → Intent Layer 相似度搜索
- **时序查询**："用户上周做了什么？" → Event Layer 时间范围搜索
- **综合查询**："推荐用户感兴趣的研究论文" → Intent + Relation + Event 多层融合

### 3. 增量更新（Incremental Memory Update）

用简单的规则和 LLM 在线更新内存层，而不是重新处理整个历史：
```python
# 新对话后：
new_facts = LLM_extract_facts(new_dialog)
memory.fact_layer.update(new_facts)

new_relations = LLM_extract_relations(new_dialog)
memory.relation_layer.add_edges(new_relations)

new_event = {date: today, type: "dialog", summary: ...}
memory.event_layer.append(new_event)
```

## 实验结论

- 对比标准 RAG（vector embedding），在 multi-hop 问题上召回率 +35%。
- Agent 通过 Decoupling Memory 完成的任务成功率 +22%（相比无记忆 agent）。
- 内存存储效率高：结构化记忆比扁平 chunk 节省 60% 存储。

## 工程落地要点

- 图数据库（Neo4j/TigerGraph）存储关系层，支持快速图遍历。
- 事实层用 KV 存储（Redis）或结构化数据库（PostgreSQL）。
- 事件层用时间序列 DB（ClickHouse）或简单列表追加。
- 四层模式可以逐步采用：先做 Event（时间序列），再加 Fact（统计），再加 Relation（图），最后 Intent（语义）。

## 面试考点

- Q: Agent 记忆和人类记忆有什么相似和不同？
  A: 相似：都需要选择性遗忘（不是所有信息都重要）、抽象化（记住概念而非细节）、多维度组织。不同：人类记忆是生物化学的，容量有限，容易遗忘；Agent 记忆可以是精确的、永久的、多模态存储。

- Q: 为什么需要意图层（Intent Layer）？
  A: 用户的长期目标是相对稳定的，不会因为每次对话改变。意图层捕捉这种稳定信息，Agent 可以用意图指导决策（"这个推荐符合用户的学习兴趣吗？"），而不需要每次都从历史对话推断。

- Q: 如何处理记忆中的矛盾信息？
  A: 用时间戳和置信度（confidence score）标记事实。新信息覆盖旧信息时，标记为"过期"而非删除。对于矛盾事实，保留两个版本并记录转变时间（用户偏好可能改变），Agent 查询时优先使用最新版本。
