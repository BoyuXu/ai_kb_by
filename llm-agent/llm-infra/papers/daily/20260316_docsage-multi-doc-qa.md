# DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering

> 来源：arxiv | 日期：20260316 | 领域：llm-infra

## 问题定义

多文档多实体问答（Multi-Doc Multi-Entity QA）：给定多个文档和多个实体，回答关于实体间关系或聚合属性的问题。例如："在这三篇论文中，Smith 和 Johnson 的合作研究有哪些共同点？"

挑战：
1. **信息分散**：所需信息散落在多个文档中，需要跨文档关联。
2. **实体识别与链接**：需要识别实体并关联到相同的现实世界实体（消歧）。
3. **信息组织**：从非结构化文本提取关键信息，组织成结构化表示。

## 核心方法与创新点

1. **信息结构化（Information Structuring）**：
   - 第一步：LLM 从每个文档独立提取关于感兴趣实体的信息，输出为结构化 JSON。
   - 结构包括：实体属性（name, type, age, affiliation 等）、关系（works_with, published_with, cited_by）、事件（participated_in, won, discovered）。

2. **实体链接与融合（Entity Linking & Fusion）**：
   - 用 LLM + embedding 相似度进行实体消歧（多个文档中的"Smith"是否同一人？）。
   - 融合重复实体的信息（来自不同文档的同一实体的属性合并）。

3. **关系图构建（Relation Graph Construction）**：
   - 构建二部图：左侧是实体，右侧是文档，边表示实体在文档中出现。
   - 添加实体-实体关系边（合作、引用等）。

4. **多步推理（Multi-hop Reasoning）**：
   - 对结构化图进行遍历，回答需要跨文档、跨实体的复杂问题。
   - 用 LLM 进行自然语言推理，补充图查询无法表达的逻辑。

## 实验结论

- HotpotQA（跨维基百科文章的多跳推理）：F1 相比 RoBERTa baseline +8.2%。
- 自建多文档学术论文 QA 数据集：准确率 +15.3%（相比 baseline，在提取精度和消歧上优势显著）。
- 实体链接准确率 > 95%（在消歧困难的名字上）。

## 工程落地要点

- 结构化提取的 schema 需要在实际应用前精心设计，涵盖领域特有属性。
- LLM 提取的 JSON 可能格式错误（malformed），需要 parser + fallback 机制。
- 实体消歧成本高（需要 LLM 多次调用比较），用 embedding 相似度预过滤减少消歧次数。
- 构建后的关系图存储在 Neo4j/TigerGraph，支持 Cypher 查询；对于简单场景用 JSON 就足够。

## 常见考点

- Q: 多文档 QA 和标准 RAG 的区别？
  A: 标准 RAG 检索相关文档后，将其拼接进 LLM context，让 LLM 直接生成答案（end-to-end）。多文档 QA 先结构化提取信息，再构建图，最后在图上推理。前者简单但对长文档、复杂关系推理效果差；后者可解释且支持复杂推理，但工程复杂度高。

- Q: 实体链接（Entity Linking）为什么重要？
  A: "Smith" 可能指多个人（Smith as surname），同一人在不同文档可能名字拼写不同。链接错误会导致答案错误（混淆两个不同的 Smith）。准确的链接是多文档推理的基础。

- Q: 如何在 LLM 提取的 JSON 中处理错误/歧义？
  A: (1) Schema validation：检查 JSON 结构是否符合预期；(2) 重试：格式错误时让 LLM 重新提取；(3) Fallback：无法解析时用字符串匹配降级；(4) 置信度评分：让 LLM 同时评估提取的置信度，过低的信息标记为不确定。
