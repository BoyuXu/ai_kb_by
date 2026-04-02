# A Lightweight LLM-enhanced Method for CTR Prediction

> 来源：https://arxiv.org/abs/2505.14057 | 领域：计算广告 | 学习日期：20260331

## 问题定义

LLM在CTR预测中语义理解能力强，但直接在线推理延迟和成本不可接受。如何轻量化利用LLM增强CTR？

## 核心方法与创新点

1. **LLM离线知识蒸馏**：LLM离线生成item语义Embedding，作为CTR模型额外特征
2. **跨模态对齐**：LLM Embedding与ID Embedding对比学习对齐

$$
L_{align} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID}) / \tau)}{\sum_j \exp(\text{sim}(e_{LLM}, e_{ID_j}) / \tau)}
$$

3. **轻量融合模块**：简单MLP融合语义与行为特征
4. **渐进式训练**：先预训练对齐，再端到端微调

## 实验结论

AUC提升0.3-0.5%，推理延迟仅增加2ms（vs直接LLM增加200ms+）。

## 工程落地要点

- LLM Embedding预计算存入Feature Store
- 在线推理仅需查表+MLP，开销可控
- 适合现有CTR模型增量升级
- 新item的LLM Embedding可异步生成

## 常见考点

1. **为什么不直接用LLM做CTR？** 延迟和成本不可接受
2. **LLM蒸馏到CTR的关键？** 跨模态对齐保证语义有效传递
3. **LLM Embedding vs传统文本特征？** LLM理解语义更深入
