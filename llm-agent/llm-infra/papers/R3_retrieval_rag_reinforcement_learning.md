# R3: Optimizing Retrieval for RAG via Reinforcement Learning

**ArXiv:** 2510.24652 | **Date:** 2025-10

## 核心问题
传统 RAG 检索器针对检索任务本身优化，而非针对 RAG 下游任务，导致次优性能。

## 核心方案
R3（Reinforcement learning for Retrieval in RAG）框架：让检索器在给定 RAG 环境中自主探索并自我改进，最小化人工实验和调优成本。

## 关键技术：增强对比学习（Reinforced Contrastive Learning）
- 训练时探索对比关系数据，而非使用预标注数据集
- 高效实用：仅需 **4 块 GPU**，单日内完成训练

## 性能
- vs 原始检索器：**+5.2%** RAG 性能
- vs SOTA 检索器：**+4.9%**

## 工业意义
无需大量人工标注即可优化检索器以适应特定 RAG 场景，大幅降低部署成本。

## 面试考点
- 为什么直接优化 RAG 下游任务效果更好？
- 对比学习在检索中的应用？
- RL 探索 vs 监督学习的权衡？

**Tags:** #llm-infra #rag #retrieval #reinforcement-learning
