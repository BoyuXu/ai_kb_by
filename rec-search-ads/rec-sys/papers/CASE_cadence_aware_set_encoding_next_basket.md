# CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation
> 来源：arXiv:2604.06718 | 领域：rec-sys | 学习日期：20260419

## 问题定义
下一篮子推荐（Next Basket Recommendation）预测用户下次购买的商品集合。复购场景（repurchase）中，用户的购买节奏（cadence）是关键信号——不同商品有不同的复购周期（如牛奶每周、洗衣液每月）。

## 核心方法与创新点
1. **Cadence-Aware（节奏感知）**：显式建模每个商品的复购周期，而非仅依赖序列模式
2. **Set Encoding**：将每次购买篮子视为集合（无序），区别于传统序列建模
3. **大规模部署设计**：面向工业级电商复购场景优化

## 面试考点
- Q: Next Basket Recommendation 和 Sequential Recommendation 的区别？
  - A: NBR预测集合（多个商品），SR预测单个下一次交互；NBR需建模商品间共现关系
