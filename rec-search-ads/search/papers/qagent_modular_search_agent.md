# Qagent: A Modular Search Agent with Interactive Query Understanding

> 来源：https://arxiv.org/abs/2510.08383 | 领域：搜索算法 | 学习日期：20260331

## 问题定义

复杂搜索任务需要多轮查询理解和改写，传统query understanding模块是静态的，无法根据检索结果动态调整。

## 核心方法与创新点

1. **模块化Agent架构**：Query理解、检索、重排、答案生成四个可组合模块
2. **交互式查询改写**：根据初始检索结果反馈自动决定是否改写

$$a_t = \text{Agent}(q, R_t, \text{History})$$

3. **工具调用机制**：通过API调用不同检索/理解模块
4. **多策略融合**：query扩展、分解、聚焦等自动选择

## 实验结论

复杂QA任务上EM指标提升15-20%，多跳推理问题提升更显著。

## 工程落地要点

- 模块化支持独立升级和AB测试
- Agent决策增加延迟，适合容忍场景
- 可通过缓存减少重复检索
- 与现有搜索引擎API兼容

## 面试考点

1. **搜索Agent vs传统pipeline？** Agent能动态决策和迭代
2. **何时需要query改写？** 初始检索质量低或意图模糊时
3. **评估搜索Agent？** 准确率+延迟+检索轮数
