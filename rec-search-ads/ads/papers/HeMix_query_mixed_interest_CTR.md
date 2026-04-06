# HeMix: Query-Mixed Interest Extraction and Heterogeneous Interaction for Scalable CTR

**ArXiv:** 2602.09387 | **Date:** 2026-02 | **Platform:** AMAP (AutoNavi Map, Alibaba)

## 核心问题
工业推荐系统中 feature interaction 的两大挑战：
1. 稀疏多域输入 + 超长用户行为序列下，context-aware 和 context-independent 意图难以同时建模
2. 现有方法 interaction 机制同质化，次优

## 方案：HeMix

### Query-Mixed Interest Extraction Module
- 通过**动态查询（dynamic queries）**从全局行为序列提取 context-aware 意图
- 通过**固定查询（fixed queries）**从实时行为序列提取 context-independent 意图
- 自适应序列分词（adaptive sequence tokenization）

### HeteroMixer Block（异构混合器）
每个 block 三个阶段：
1. **Multi-token Fusion**：融合多个 token
2. **Heterogeneous Mixed-Token Interaction**：异构混合 token 交互
3. **Group-Aligned Reconstruction**：局部和全局信息整合

## 部署
AMAP（高德地图，数十亿用户）实时推荐系统。

## 面试考点
- Context-aware vs context-independent 用户兴趣的区别？
- 为什么序列分词（tokenization）对长序列建模重要？
- 异构 feature interaction 如何提升 CTR 预估？

**Tags:** #ads #ctr #feature-interaction #interest-extraction #long-sequence #industrial
