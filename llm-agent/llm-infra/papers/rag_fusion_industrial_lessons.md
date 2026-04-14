# Scaling RAG with RAG Fusion: Industrial Deployment Lessons

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2603.02153

## 核心贡献

工业级评估表明RAG融合（多查询+RRF）在生产环境中效果有限。

## 关键发现

- **召回提升被抵消**: 融合提升原始召回，但重排+截断后效果消失
- **Top-k精度下降**: 融合变体不如单查询基线
- **延迟开销**: 无对应效果增益的延迟成本
- **建议**: 需要联合评估检索质量+系统效率+下游影响

## 面试考点

RAG系统设计trade-off，检索融合策略评估，生产环境优化思维。
