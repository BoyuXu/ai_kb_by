# GLASS: A Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search

**ArXiv:** 2602.05663 | **Date:** 2026-02

## 核心问题
生成式推荐系统难以有效建模用户长期历史序列，现有方法无法充分利用长期兴趣信息。

## 核心架构（三大组件）

### 1. SID-Tier Module
- 将长期交互映射为统一兴趣向量
- 增强初始 SID token（SID1）的预测

### 2. Semantic Hard Search Module
- 基于生成的 SID1 进行语义硬搜索
- 通过门控机制增强细粒度 token 生成（SID2, SID3）
- 有效闭合长期与细粒度兴趣的信息环路

### 3. Sparsity-Aware Augmentation Strategies
处理长序列中的稀疏性问题。

## 实验
在 TAOBAO-MM 和 KuaiRec 数据集上优于基线。

## 技术创新
将 SID 层级化（SID1 → SID2 → SID3），不同粒度对应不同的检索和生成策略。

## 面试考点
- SID 层级结构如何编码 item 语义？
- 长序列推荐中如何避免计算爆炸？
- Semantic Hard Search 如何提升细粒度预测？

**Tags:** #rec-sys #generative-recommendation #long-sequence #sid #semantic-search
