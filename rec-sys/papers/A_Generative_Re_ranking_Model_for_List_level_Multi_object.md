# A Generative Re-ranking Model for List-level Multi-objective Optimization at Taobao
> 来源：https://arxiv.org/abs/2505.07197 | 领域：rec-sys | 日期：20260323

## 问题定义
淘宝提出的生成式重排序模型，解决列表级（list-level）多目标优化问题。传统pointwise重排序忽略列表内物品间的交互效应，且多目标优化存在冲突。

## 核心方法与创新点
- 生成式重排：将重排视为序列生成问题，自回归生成最优物品排列
- 列表级建模：考虑位置间物品的互补性、多样性和竞争关系
- 多目标Pareto优化：生成满足多目标Pareto前沿的排列
- 上下文感知：每步生成时考虑已生成物品的上下文信息

## 实验结论
淘宝线上实验：相比PRM（Permutation-based Re-ranking），多目标指标综合提升约2%，用户人均购买数量和GMV均有提升。

## 工程落地要点
- 生成式重排的latency比pointwise大10-50x，需要严格的时间预算控制
- 物品集合大小通常限制在50-200个（重排候选集），控制生成复杂度
- 多目标权重需要根据业务目标动态调整（如促销期间提高GMV权重）

## 面试考点
1. **Q: 列表级重排与pointwise重排的本质区别？** A: 列表级考虑物品间交互（互补、竞争、多样性）；pointwise独立打分忽略上下文
2. **Q: 生成式重排如何处理多目标冲突？** A: Pareto优化、加权scalarization、constrained optimization
3. **Q: 重排阶段的latency如何控制？** A: 限制候选集大小、beam width限制、早停策略、模型蒸馏
4. **Q: 物品列表的多样性如何在生成过程中保证？** A: diversity bonus（对已选类目/风格降权）、maximal marginal relevance（MMR）
5. **Q: 淘宝为什么在重排阶段引入生成式模型？** A: 重排候选集小（~100），生成式的计算代价可接受；列表交互效应在此阶段最显著
