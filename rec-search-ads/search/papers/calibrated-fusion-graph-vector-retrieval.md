# Calibrated Fusion for Heterogeneous Graph-Vector Retrieval in Multi-Hop QA

- **Date**: 2026-03
- **Domain**: Search/Retrieval
- **URL**: https://arxiv.org/abs/2603.28886

## 核心贡献

PhaseGraph方法使用百分位秩归一化(PIT)将向量和图得分映射到统一无量纲尺度，实现稳定融合。

## 关键技术

- **得分校准问题**: 图增强检索将密集相似度与图信号(如PPR)结合，但分布不同不可直接比较
- **PIT归一化**: 百分位秩变换将异质得分映射到[0,1]统一尺度
- **校准融合**: 在归一化后进行融合，不丢弃量级信息

## 实验结果

- MuSiQue: LastHop@5 从75.1%提升到76.5%
- 2WikiMultiHopQA: LastHop@5 从51.7%提升到53.6%
- 基于HippoRAG2风格benchmark

## 面试考点

- 图检索与向量检索的得分融合策略
- 多跳问答中的检索方法
- 异构信号的归一化与校准技术
