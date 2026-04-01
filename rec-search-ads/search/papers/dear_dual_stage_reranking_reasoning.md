# DEAR: Dual-stage Document Reranking with Reasoning Agents via LLM Distillation

> 来源：https://arxiv.org/abs/2508.16998 | 领域：搜索算法 | 学习日期：20260331

## 问题定义

LLM-based重排效果好但成本高，小模型快但质量有限。如何通过蒸馏让小模型获得LLM级重排能力？

## 核心方法与创新点

1. **双阶段架构**：Stage 1小模型快速粗排，Stage 2蒸馏模型精排
2. **推理蒸馏**：从大LLM推理链蒸馏排序知识

$$
\mathcal{L}_{distill} = \text{KL}(P_{student}(\text{rank}|q,D) || P_{teacher}(\text{rank}|q,D))
$$

3. **Agent推理框架**：teacher作为推理Agent生成排序理由
4. **选择性蒸馏**：只蒸馏teacher高置信度决策

## 实验结论

Student模型（1.5B参数）达到GPT-4级重排模型95%性能，推理速度快20倍。

## 工程落地要点

- Student可本地部署，无需API调用
- 蒸馏数据可持续积累
- 双阶段与现有pipeline兼容
- 适合延迟敏感搜索场景

## 面试考点

1. **为什么蒸馏而非微调？** 保留LLM推理知识，微调易过拟合
2. **选择性蒸馏好处？** 避免从teacher错误决策中学习
3. **1.5B vs 70B trade-off？** 质量损失5%但速度快20倍
