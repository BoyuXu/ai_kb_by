# RASTP: Representation-Aware Semantic Token Pruning for Generative Recommendation

**ArXiv:** 2511.16943 | **Date:** 2025-11 | **Code:** github.com/Yuzt-zju/RASTP

## 核心问题
生成式推荐系统使用语义标识符（SID）表示 item，每个 item 对应多个 SID token，导致输入序列长度大幅增加，计算复杂度和内存消耗显著上升。

## 核心方案：RASTP
直接剪枝输入序列中信息量较少的 token。

### Token 重要性评估（双维度）
1. **语义显著性（Semantic Saliency）**：通过表示幅度（representation magnitude）衡量
2. **注意力中心性（Attention Centrality）**：通过累积注意力权重衡量

### 剪枝策略
结合两个维度综合评分，剪掉低重要性 token，保留关键语义信息。

## 实验结果（Amazon 数据集 × 3）
- 训练时间减少 **26.7%**
- 推荐性能持平或略有提升

## 意义
在生成式推荐效率方向的重要贡献，平衡了 SID 表达能力和计算效率。

## 面试考点
- 生成式推荐中 SID 的设计原则？
- Token Pruning vs 序列截断的区别？
- 如何评估 token 对推荐任务的重要性？

**Tags:** #rec-sys #generative-recommendation #sid #token-pruning #efficiency
