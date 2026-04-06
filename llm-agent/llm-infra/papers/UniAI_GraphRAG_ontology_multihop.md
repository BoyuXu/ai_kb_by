# UniAI-GraphRAG: Ontology-Guided Multi-Dimensional Clustering for Multi-Hop Reasoning

**ArXiv:** 2603.25152 | **Date:** 2026-03

## 核心问题
标准 GraphRAG 在多跳推理、社区完整性和检索准确性方面存在不足。

## 三大核心创新

### 1. Ontology-Guided Knowledge Extraction
用预定义 Schema 引导 LLM 准确识别领域实体和关系，解决通用抽取方法的噪声问题。

### 2. Multi-Dimensional Community Clustering
- **对齐补全（Alignment Completion）**：修复关系不完整问题
- **属性聚类（Attribute-based Clustering）**：基于实体属性的聚类
- **多跳关系聚类（Multi-hop Relationship Clustering）**：捕获跨跳关系

### 3. Dual-Channel Graph Retrieval Fusion
混合图检索 + 社区检索，平衡 QA 准确性和性能。

## 性能
- vs Dify Naive RAG：整体 F1 提升 **22.45%**
- vs Open-LightRAG（SOTA）：提升 **2.77%**

## 面试考点
- GraphRAG vs RAG 的核心差异？
- 本体（Ontology）引导如何提升知识抽取质量？
- 多跳推理中社区结构的作用？

**Tags:** #llm-infra #rag #graphrag #knowledge-graph #multihop
