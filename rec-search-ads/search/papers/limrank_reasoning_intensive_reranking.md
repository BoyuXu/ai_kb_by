# LimRank: Less is More for Reasoning-Intensive Information Reranking

> 来源：https://arxiv.org/abs/2510.23544 | 领域：搜索算法 | 学习日期：20260331

## 问题定义

LLM-based重排模型计算成本极高，输入所有候选文档完整内容导致上下文过长、推理缓慢。

## 核心方法与创新点

1. **选择性上下文压缩**：只保留候选文档中与query最相关的关键段落
2. **渐进式推理**：分批处理候选集，逐步淘汰低质量文档

$$
\text{Score}(d) = \text{LLM}(\text{Compress}(q, d))
$$

3. **推理链简化**：设计更简洁的推理模板
4. **Early Exit**：排序置信度足够高时提前终止

## 实验结论

在BEIR上保持98%的full-context重排质量，推理速度提升3-5倍，token消耗减少60%。

## 工程落地要点

- 压缩策略可预计算
- 分批处理支持流水线并行
- Early Exit适合延迟敏感场景
- 可直接替换现有重排服务

## 面试考点

1. **为什么Less is More？** 冗余信息干扰LLM推理
2. **哪些内容可以压缩？** 基于query-document注意力分数
3. **Early Exit置信度？** 基于top-K排序概率间隔
