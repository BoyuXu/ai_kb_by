# Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations
> 来源：https://arxiv.org/abs/2503.02453 | 领域：search | 日期：20260323

## 问题定义
生成式推荐中，物品的稀疏（离散token）和稠密（连续embedding）表示各有优缺点。本文提出级联稀疏-稠密统一表示，在生成式推荐框架中融合两种表示的优势。

## 核心方法与创新点
- 级联表示：先生成稀疏token序列（粗粒度），再条件化生成稠密embedding（细粒度）
- 稀疏指导稠密：稀疏token提供类别/主题约束，稠密embedding精化物品选择
- 统一训练：稀疏生成loss + 稠密对比loss联合优化
- 高效索引：稀疏token用树状结构加速，稠密用HNSW

## 实验结论
在多个推荐benchmark，级联稀疏-稠密比纯稀疏生成提升约5%，比纯稠密ANN检索提升约3%；两阶段级联检索比单阶段速度快约4x（稀疏层大幅剪枝候选）。

## 工程落地要点
- 需要维护两套索引（稀疏树+稠密HNSW），工程复杂度增加
- 稀疏层的剪枝比例是关键超参（剪枝太多丢失recall，太少效率无提升）
- 可以根据物品热度动态调整稀疏/稠密的权重

## 常见考点
1. **Q: 稀疏表示（token）和稠密表示（embedding）各自的检索方式？** A: 稀疏：树状结构beam search；稠密：ANN（HNSW/IVF-PQ）近似近邻检索
2. **Q: 为什么级联比并行融合更高效？** A: 级联先用稀疏剪枝（快），再在小候选集上用稠密精排，避免稠密的全量搜索
3. **Q: 生成式推荐中的物品覆盖率如何保证？** A: 增加beam width、添加diversity penalty、专门的长尾增强通道
4. **Q: 稀疏-稠密表示对新物品（冷启动）的处理？** A: 稀疏token可从类目/属性映射（不需要行为数据），稠密embedding从内容生成
5. **Q: 生成式推荐的"幻觉"问题（生成不存在的物品ID）？** A: 约束解码（只允许合法token序列）、后验过滤（检查物品是否在库）
