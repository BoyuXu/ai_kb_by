# Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model
> 来源：arXiv:2510.10127 | 领域：rec-sys | 学习日期：20260419

## 问题定义
生成式推荐中，自回归模型倾向于生成高似然但不一定高质量的推荐（似然陷阱）。模型可能"抄近路"——生成常见/热门物品而非个性化推荐。

## 核心方法
1. **Graph-structured Model**：引入图结构约束打破似然陷阱
2. **一致性训练（Consistent Training）**：确保生成分布与真实分布对齐
3. **图结构先验**：利用物品间的图关系（共现/属性相似）约束生成空间
