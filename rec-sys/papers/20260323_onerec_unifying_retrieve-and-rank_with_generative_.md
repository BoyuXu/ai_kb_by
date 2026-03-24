# OneRec: Unifying Retrieve-and-Rank with Generative Recommender and Alignment
> 来源：https://arxiv.org/search/?query=OneRec+unifying+retrieve+rank+generative+recommender&searchtype=all | 领域：rec-sys | 日期：20260323

## 问题定义
推荐系统中召回（retrieve）与排序（rank）是两个独立优化的阶段，存在目标不一致和信息损失。OneRec尝试用单一生成式模型统一这两个阶段。

## 核心方法与创新点
- 统一生成框架：一个模型同时完成召回和排序
- 迭代偏好对齐：通过强化学习/RLHF方式对齐用户真实偏好
- Encoder-Decoder架构：用用户历史作为context，生成物品ID序列
- Constrained beam search：确保生成的物品ID合法且多样
- 在线对齐：利用用户实时反馈持续更新偏好模型

## 实验结论
相比两阶段基线，OneRec在Recall@K和NDCG指标提升5-8%；偏好对齐后进一步提升2-3%的用户满意度指标。

## 工程落地要点
- 需要构建物品ID的层级编码树，控制beam search搜索空间
- 对齐阶段需要在线收集用户反馈信号（点击、停留时长等）
- 模型更新需要支持增量学习，避免全量重训的时间成本

## 面试考点
1. **Q: 为什么推荐系统需要"对齐"？** A: 模型优化的proxy指标（CTR）与用户真实满意度存在gap，需要对齐
2. **Q: OneRec和传统推荐的核心差异？** A: 统一生成vs分阶段pipeline，减少信息损失和目标不一致
3. **Q: 生成式推荐的beam search如何保证物品合法性？** A: 使用prefix tree/trie约束beam search，只允许合法token序列
4. **Q: 迭代偏好对齐的实现方式？** A: DPO/PPO等RL方法，用用户正负反馈构建偏好数据对
5. **Q: 大规模工业推荐系统中OneRec的主要瓶颈？** A: 推理latency（生成式比pointwise ranking慢10-100x）、物品空间覆盖率
