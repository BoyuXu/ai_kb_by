# Robust Autobidding for Noisy Conversion Prediction Models (RobustBid)
> 来源：arXiv:2510.08788 | 领域：ads | 学习日期：20260419

## 问题定义
自动出价（Autobidding）依赖CTR/CVR预估值来计算出价，但预估模型不可避免存在噪声。当预估偏差较大时，可能导致严重的出价错误（过高或过低）。

## 核心方法
1. **鲁棒优化（Robust Optimization）**：在最坏情况下优化出价策略
2. **预估噪声建模**：显式建模CTR/CVR预估的不确定性
3. **工业级验证**：在工业数据集上验证，提升转化率、降低CPC

## 面试考点
- Q: 自动出价系统的核心组件？
  - A: CTR/CVR预估 → 价值评估 → 出价策略（如pacing + throttling）→ 预算约束优化
