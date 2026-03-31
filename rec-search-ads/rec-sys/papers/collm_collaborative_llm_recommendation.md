# CoLLM: Collaborative Large Language Model for Recommendation

> 来源：https://arxiv.org/abs/2310.13825 | 领域：推荐系统 | 学习日期：20260331

## 问题定义

LLM缺乏推荐系统中关键的协同过滤信号，仅靠语义理解无法捕获「购买A的用户也购买B」这类协同模式。

## 核心方法与创新点

1. **协同信号注入**：将CF模型（MF/LightGCN）的Embedding作为soft prompt注入LLM
2. **映射网络**：轻量映射网络将CF Embedding转换为LLM的token空间
3. **Prompt设计**：结合用户画像、行为历史和CF Embedding构建推荐Prompt

$$\text{Prompt} = [\text{CF\_Emb}(u); \text{History}(u); \text{Item\_Desc}(i); \text{Task}]$$

4. **LoRA微调**：推荐数据上使用LoRA高效微调LLM

## 实验结论

在MovieLens和Amazon数据集上，CoLLM相比纯LLM推荐方法HR@10提升12-15%，与传统CF方法也有3-5%提升。

## 工程落地要点

- CF Embedding可预计算并缓存，不影响在线推理
- LoRA微调降低训练成本，单GPU即可训练
- 映射网络参数量小，额外推理开销可忽略
- 适合冷启动与长尾场景

## 面试考点

1. **LLM推荐为什么需要协同信号？** 语义相似不等于行为相似
2. **CF Embedding如何注入LLM？** 映射网络转换到token空间作为soft prompt
3. **CoLLM vs InstructRec？** CoLLM显式融合CF信号
