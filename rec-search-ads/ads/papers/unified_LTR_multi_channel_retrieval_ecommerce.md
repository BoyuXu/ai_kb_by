# Unified Learning-to-Rank for Multi-Channel Retrieval in Large-Scale E-Commerce Search
> 来源：arXiv:2602.23530 | 领域：ads | 学习日期：20260419

## 问题定义
大规模电商搜索依赖多个专业检索通道（词法匹配、语义检索、时效性、季节性等），如何有效融合异构通道的候选为统一排序列表？传统RRF/Weighted Interleaving依赖固定全局权重，无法感知query特定的通道效用。

## 核心方法与创新点
1. **Channel-Aware LTR**：将多通道融合建模为channel-aware的排序学习任务
2. **联合多目标优化**：同时优化点击、加购、购买
3. **Channel-specific Objectives**：每个通道有独立优化目标
4. **用户行为信号融合**：融入近期用户行为捕捉短期意图

## 实验结论
- 在线A/B测试：用户转化率提升 **+2.85%**
- p95延迟 < 50ms，满足生产要求
- 已部署于大型电商平台

## 面试考点
- Q: 多通道检索融合的方法？
  - A: ①RRF（倒数排名融合）；②Weighted Interleaving；③Unified LTR（本文）；④级联重排
