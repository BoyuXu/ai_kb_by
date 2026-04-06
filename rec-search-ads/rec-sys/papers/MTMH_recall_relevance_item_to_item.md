# Optimizing Recall or Relevance? Multi-Task Multi-Head Approach for Item-to-Item Retrieval

**ArXiv:** 2506.06239 | **Venue:** KDD 2025 | **Date:** 2025-06

## 核心问题
I2I（Item-to-Item）召回的核心矛盾：
- **Co-engagement 优化的 recall** 过度强调短期共同交互模式
- 忽视语义相关性，导致短期趋势过拟合，牺牲多样性和新兴趣发现

## 方案：MTMH（Multi-Task Multi-Head）

### 多任务学习损失
正式建模 recall 和语义相关性之间的权衡，同时优化两个目标。

### 多头 I2I 检索架构
同时检索两类 item：
1. 高度共同参与（highly co-engaged）的 item
2. 语义相关（semantically relevant）的 item

## 实验结果
vs 先前 SOTA：
- Recall 提升最高 **14.4%**
- 语义相关性提升最高 **56.6%**

## 工业价值
解决推荐系统中"过度局部化"问题，促进多样性发现和用户长期满益。

## 面试考点
- I2I 召回为什么重要？与 U2I 的区别？
- 如何定义和度量语义相关性？
- Recall 和 Relevance 冲突时如何权衡？

**Tags:** #rec-sys #item-to-item #recall #relevance #multi-task #multi-head
