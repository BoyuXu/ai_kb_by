# CoCR-RAG: Concept-oriented Context Reconstruction for Enhanced Web Q&A

- **Date**: 2026-03
- **Domain**: LLM-Infra/RAG
- **URL**: https://arxiv.org/abs/2603.23989

## 核心贡献

CoCR-RAG通过概念蒸馏算法从AMR(抽象意义表示)中提取核心概念，将多源文档融合重构为统一的信息密集上下文。

## 关键技术

- **AMR语义表示**: 将文本结构化为逻辑图
- **概念蒸馏**: 从AMR中提取本质概念，基于自然语言固有模式，无需超参调优
- **上下文重构**: LLM仅补充必要句子元素，突出核心知识

## 实验结果

- PopQA和EntityQuestions数据集上优于各种基线方法
- 跨多种backbone LLM和上下文重构方法均表现最佳

## 面试考点

- AMR在NLP中的应用
- 多源文档融合的上下文重构方法
- RAG中的上下文压缩与信息保真
