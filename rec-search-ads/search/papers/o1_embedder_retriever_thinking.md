# O1 Embedder: Let Retrievers Think Before Action

> 来源：https://arxiv.org/abs/2509.25085 | 领域：搜索算法 | 学习日期：20260331

## 问题定义

传统dense retriever直接将query映射到向量空间，缺乏深层理解。类似o1的「先思考再行动」范式能否提升检索质量？

## 核心方法与创新点

1. **思考链Embedding**：生成query embedding前先进行内部推理
2. **推理感知编码器**：

$$
e_q = \text{Encoder}(\text{Think}(q) \oplus q)
$$

3. **合成推理数据**：LLM生成query推理过程作为训练数据
4. **对比学习+推理监督**：双目标训练

## 实验结论

在BEIR和MTEB上Recall@100提升4-6%，需要推理的查询提升8-12%。

## 工程落地要点

- 推理过程增加encoding时间但可离线预计算
- Query侧推理可在线执行（数量远小于document）
- 与现有ANN索引兼容
- 适合高质量检索场景

## 常见考点

1. **Retriever也需要思考？** 复杂query需理解隐含意图
2. **Think过程做什么？** 分析意图、扩展概念、消除歧义
3. **与query expansion区别？** Think是内部推理，不直接修改query文本
