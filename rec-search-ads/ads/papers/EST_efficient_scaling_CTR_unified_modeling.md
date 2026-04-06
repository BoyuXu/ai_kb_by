# EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling

**ArXiv:** 2602.10811 | **Date:** 2026-02 | **Org:** Alibaba

## 核心问题
工业 CTR 预估中的 scaling 瓶颈：现有方法采用用户行为早期聚合，不统一、不完整的建模方式创造了信息瓶颈，丢弃了解锁 scaling 收益所必需的 token 级细粒度信号。

## 方案：EST（Efficiently Scalable Transformer）

### 三大核心技术

**1. Unified Sequence Modeling（统一序列建模）**
将异构输入（用户行为、item 特征、上下文特征）统一在单一 token 序列中处理。

**2. LCA - Lightweight Cross-Attention（轻量交叉注意力）**
专注于最有信息量的交互，减少冗余计算。

**3. CSA - Content Sparse Attention（内容稀疏注意力）**
利用内容相似度进行长行为序列的稀疏建模，几乎无额外开销。

## 实验结果
- 离线评估：SOTA 性能
- 展现出清晰的 **power-law scaling 趋势**（模型容量和计算成本的幂律关系）

## 工业意义
首次系统验证 CTR 模型的 scaling law，为工业 CTR 模型发展方向提供重要参考。

## 面试考点
- CTR 模型为什么之前难以实现 scaling law？
- Sparse attention 如何降低长序列的计算复杂度？
- 统一序列建模 vs 特征分组的优劣？

**Tags:** #ads #ctr #scaling-law #transformer #long-sequence #alibaba
