# From Matching to Generation: A Survey on Generative Information Retrieval

**ArXiv:** 2404.14851 | **Venue:** ACM TOIS 2025 | **Date:** 2024-04

## 概述
系统综述生成式信息检索（GenIR）的发展，从传统相似度匹配向生成范式转变。

## 两大核心方向

### 1. Generative Document Retrieval (GR)
利用生成模型的参数记忆文档，直接生成相关文档标识符进行检索，无需显式索引。
- 代表：DSI（Differentiable Search Index）
- 关键挑战：文档标识符设计（atomic vs semantic）

### 2. Reliable Response Generation
LLM 直接生成用户所需信息（答案），而非返回文档列表。
- 代表：RAG 系列方法
- 关键挑战：幻觉、可靠性、引用准确性

## 技术演进脉络
传统 BM25/Dense Retrieval → GR（端到端文档生成）→ RAG（检索增强生成）→ 生成式搜索（OneSearch 等）

## 工业应用
电商生成搜索（JD、Taobao）、对话式搜索助手、知识密集型 QA。

## 面试考点
- GenIR 相比传统 IR 的核心优势和挑战？
- 文档标识符（DocID）如何设计？
- GR 的 catastrophic forgetting 问题如何缓解？

**Tags:** #search #generative-retrieval #survey #ir #rag
