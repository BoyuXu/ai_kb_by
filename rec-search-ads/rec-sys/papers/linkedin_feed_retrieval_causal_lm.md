# Large Scale Retrieval for LinkedIn Feed Using Causal Language Models

> 来源：https://arxiv.org/abs/2510.14223 | 领域：推荐系统 | 学习日期：20260331

## 问题定义

LinkedIn信息流推荐中，传统双塔召回模型难以建模复杂的用户-内容交互关系，无法有效利用用户行为序列的时序信息。

## 核心方法与创新点

1. **因果语言模型用于召回**：用户历史行为序列建模为自回归序列，预测下一个可能交互的内容
2. **大规模负采样策略**：结合in-batch negatives和hard negatives
3. **两阶段训练**：先预训练序列模型，再微调用于检索任务

$$
P(v_{t+1} | v_1, ..., v_t) = \text{CLM}(v_1, ..., v_t)
$$

4. **Embedding离线索引**：生成的内容表征存入ANN索引，支持毫秒级在线召回

## 实验结论

LinkedIn信息流AB实验中，CLM召回相比双塔模型Recall@500提升8%，线上sessions with engagement提升1.2%。

## 工程落地要点

- 模型需支持增量更新，应对实时内容发布
- Embedding维度与ANN索引效率需权衡
- 用户序列长度截断策略直接影响效果
- 工业部署需要模型蒸馏降低推理成本

## 面试考点

1. **为什么用因果语言模型而非双塔？** CLM能建模序列依赖，捕获时序模式
2. **如何处理内容冷启动？** 多模态特征+内容语义Embedding
3. **亿级规模ANN检索挑战？** 延迟与精度平衡
