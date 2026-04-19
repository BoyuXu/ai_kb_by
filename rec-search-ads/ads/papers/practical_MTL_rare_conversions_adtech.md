# Practical Multi-Task Learning for Rare Conversions in Ad Tech
> 来源：arXiv:2507.20161 | 领域：ads | 学习日期：20260419

## 问题定义
广告系统中深度转化（purchase/subscribe）极为稀疏（CTR~5%, CVR~0.1%, 深度转化~0.01%），传统MTL方法在稀疏label下表现不佳。

## 核心方法
1. **数据驱动任务定义**：不依赖人工定义辅助任务，而是从数据中自动发现有效的中间任务
2. **工业级MTL部署**：在live广告系统中验证，线上/线下一致性提升
3. **处理极端label稀疏**：通过辅助任务的梯度信号缓解稀疏问题

## 面试考点
- Q: ESMM/MMOE/PLE的核心区别？
  - A: ESMM：CVR=CTR×CVR的乘积形式解决sample selection bias；MMOE：多门控专家网络处理任务间差异；PLE：显式分离shared/task-specific专家
