# LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application
> 来源：https://arxiv.org/abs/2408.01024 | 领域：ads | 日期：20260323

## 问题定义
LLM拥有丰富的世界知识，但直接用于工业推荐系统存在效率、实时性和个性化问题。LEARN提出一种知识适配框架，将LLM的知识迁移到轻量级推荐模型，实现实用工业部署。

## 核心方法与创新点
- 知识蒸馏pipeline：LLM生成soft label→学生推荐模型学习
- 语义-协同联合学习：LLM语义特征 + 协同过滤信号联合训练
- 轻量化适配：将LLM知识压缩到几十倍小的推荐模型
- 增量适配：支持新知识的增量更新，无需全量重训LLM

## 实验结论
工业广告场景实验：LEARN相比无LLM知识的基线，CTR AUC提升约0.5%，特别在长尾广告和冷启动广告上提升更显著（约1.5%）；推理速度保持与原模型相当（<5ms）。

## 工程落地要点
- LLM推理可以离线批量完成（每天/每周），不影响在线latency
- 软标签（soft label）包含LLM对物品的语义评分，比one-hot更丰富
- 需要定期更新LLM生成的知识库，保持知识新鲜度

## 面试考点
1. **Q: 为什么不直接用LLM做推荐？** A: 推理太慢（>100ms vs <5ms要求）、个性化不足、实时特征无法输入
2. **Q: 知识蒸馏在推荐中的具体形式？** A: 用LLM的输出概率/embedding作为教师信号，指导学生模型的表示学习
3. **Q: LLM知识如何弥补协同过滤的不足？** A: 提供冷启动物品的语义先验、引入跨域知识、理解长尾概念
4. **Q: 软标签（soft label）vs 硬标签（hard label）的优劣？** A: 软标签包含类间关系信息，训练更稳定，泛化更好；但需要高质量教师模型
5. **Q: LEARN的增量更新如何设计？** A: 只更新与新内容相关的embedding，保持其他参数冻结
