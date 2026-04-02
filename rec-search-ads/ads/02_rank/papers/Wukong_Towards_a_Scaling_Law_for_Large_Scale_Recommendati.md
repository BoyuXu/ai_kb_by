# Wukong: Towards a Scaling Law for Large-Scale Recommendation
> 来源：https://arxiv.org/abs/2403.02545 | 领域：ads | 日期：20260323

## 问题定义
Meta提出Wukong，系统研究推荐系统的Scaling Law：随着模型参数量、训练数据量、计算量的增加，推荐效果如何变化。这是推荐领域首个系统性研究Scaling Law的工作。

## 核心方法与创新点
- 推荐Scaling Law：实验验证参数量从百万到万亿时效果的幂律增长
- Embedding Table Scaling：重点研究Embedding table（推荐模型参数主体）的scaling
- 计算最优（Compute-optimal）：给定计算预算，如何最优分配模型大小和训练步数
- 稀疏-稠密协同scaling：稀疏特征（ID）和稠密特征（MLP）的最优比例

## 实验结论
推荐系统中Scaling Law成立但形式不同：效果随参数量增加呈幂律提升，但Embedding table的边际收益高于MLP；在固定FLOPs下，优先扩大Embedding而非MLP；数据质量比数量更重要。

## 工程落地要点
- 万亿参数Embedding table需要分布式参数服务器存储和查询
- Embedding稀疏访问模式使得通信是主要瓶颈，需要异步更新
- 实际部署需要考虑inference成本，大Embedding增加内存带宽需求

## 常见考点
1. **Q: 推荐系统的Scaling Law与LLM有何不同？** A: 推荐系统的参数主要在Embedding table（稀疏），LLM主要在注意力/FFN（稠密）
2. **Q: 为什么Embedding table更值得扩大？** A: Embedding直接捕获user/item的历史交互信息，是推荐的核心信号
3. **Q: 分布式参数服务器（PS）的工作原理？** A: worker计算梯度→推送到PS→PS更新参数→worker拉取最新参数
4. **Q: 稀疏特征和稠密特征在推荐中各自的作用？** A: 稀疏（ID）：捕获协同过滤信号；稠密（数值/文本）：提供泛化能力
5. **Q: 给定固定成本预算，如何最优设计推荐模型？** A: 根据Wukong的结论：优先扩大Embedding table，MLP适中深度，训练数据质量优先于数量
