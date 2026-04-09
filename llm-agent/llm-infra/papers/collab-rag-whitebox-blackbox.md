# Collab-RAG: White-Box and Black-Box LLM Collaboration for Complex QA

- **Date**: 2025-04
- **Domain**: LLM-Infra/RAG
- **URL**: https://arxiv.org/abs/2504.04915
- **Code**: https://github.com/ritaranx/Collab-RAG

## 核心贡献

White-box SLM分解复杂查询为简单子问题，提升检索精度；Black-box LLM反馈信号改进SLM分解能力。协作训练框架实现双向增强。

## 关键技术

- **查询分解**: SLM将复杂多跳问题分解为简单子问题
- **反馈循环**: Black-box LLM提供反馈改进SLM
- **成本效率**: 仅需affordable black-box LLM监督

## 实验结果

- 5个多跳QA数据集上平均超过基线1.8%-14.2%
- Fine-tuned 3B SLM在问题分解上超越frozen 32B LLM

## 面试考点

- 大小模型协作的RAG设计模式
- 查询分解在多跳推理中的作用
- 模型蒸馏与协作训练的区别
