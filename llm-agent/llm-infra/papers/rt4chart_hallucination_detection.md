# RT4CHART: Retromorphic Testing for Hallucination Detection in RAG

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2603.27752
- **Date**: 2026-04

## 核心贡献

分层验证流水线，将回答分解为可独立验证的声明，逐级验证对抗幻觉。

## 关键技术

- **声明级分解**: 将回答拆解为可独立验证的声明
- **局部到全局验证**: 从细粒度到整体的逐级验证
- **证据归因**: 声明到回答span的显式上下文归因

## 核心结果

- RAGTruth++ F1=0.776，超越最强基线83%
- RAGTruth-Enhance span级F1=47.5%
- 重标注发现1.68倍更多幻觉

## 面试考点

幻觉检测方法，细粒度事实验证，RAG系统可靠性保障。
