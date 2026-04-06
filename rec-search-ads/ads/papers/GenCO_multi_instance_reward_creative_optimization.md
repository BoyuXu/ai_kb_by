# Generative Modeling with Multi-Instance Reward Learning for E-commerce Creative Optimization

**ArXiv:** 2508.09730 | **Date:** 2025-08

## 核心问题
电商广告创意组合优化（标题+图片+亮点等）的两大挑战：
1. 单独评估创意元素忽略了组合效果
2. 组合空间呈指数级增长，难以搜索

## 方案：GenCO（Generative Creative Optimization）

### 两阶段架构

**阶段1：生成式模型**
- 高效生成多样的创意组合候选集
- 使用强化学习优化探索和精炼策略

**阶段2：多实例学习奖励模型**
- 将组合级奖励（点击等）归因到单个创意元素
- 解决用户反馈稀疏性问题
- 为生成模型提供精准反馈信号

## 工业部署
- 大型电商平台部署
- 显著提升广告收入
- 公开发布大规模工业数据集

## 面试考点
- 为什么创意组合评估比单元素评估更复杂？
- 多实例学习（MIL）如何处理弱标注问题？
- 生成式方法 vs 穷举搜索的优势？

**Tags:** #ads #creative-optimization #generative #multi-instance-learning #e-commerce
