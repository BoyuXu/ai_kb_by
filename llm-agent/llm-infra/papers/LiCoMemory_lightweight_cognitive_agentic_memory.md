# LiCoMemory: Lightweight and Cognitive Agentic Memory

> 来源：arXiv 2025 | 领域：llm-infra/agent | 学习日期：20260408

## 问题定义

长期对话中，Agent 需要高效管理和检索历史信息：
- 传统方法：全文存储 + 向量检索（延迟高、不精确）

**核心问题**：如何构建轻量级但认知有效的 Agent 记忆系统？

## 核心方法与创新点

**CogniGraph 分层图结构**：

1. **实体和关系作为语义索引层**：
   - 不存储原始文本，而是提取实体/关系图
   - 图结构天然支持关联推理

2. **Temporal-Aware Search**：
   - 时间感知的检索策略
   - 近期信息优先，但支持跨时间关联

3. **Hierarchy-Aware Search**：
   - 层级化检索：先定位主题，再深入细节
   - 减少不相关信息的干扰

## 关键结果

- 长期对话基准超越基线
- 更新延迟显著降低

## 面试考点

- 图结构记忆 vs 向量存储的 trade-off
- 时间衰减在记忆检索中的应用
