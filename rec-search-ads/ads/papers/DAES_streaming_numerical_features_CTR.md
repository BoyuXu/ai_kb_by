# Distribution-Aware End-to-End Embedding for Streaming Numerical Features in CTR

**Platform:** 大型短视频平台（数亿 DAU）| **Domain:** Ads CTR

## 核心问题
CTR 预估中数值特征嵌入的两大痛点：
1. **静态分箱方法**：依赖离线统计，无法适应流式数据分布变化
2. **神经嵌入方法**：丢弃分布信息，损失语义

## 方案：DAES（Distribution-Aware Embedding for Streaming）

### 核心技术

**1. 水库采样分布估计（Reservoir Sampling-based Distribution Estimation）**
高效估计流式数值特征的实时分布，无需全量历史数据。

**2. 两种字段感知分布调制策略（Field-aware Distribution Modulation）**
- 捕获流式分布特征
- 保留字段依赖的语义信息

## 优势
- 相比现有方法显著提升性能
- 适应数据分布的动态变化
- 端到端可训练

## 工业部署
部署于大型短视频平台，数亿 DAU 验证。

## 面试考点
- 数值特征嵌入的主要方法（分箱、归一化、AutoDis 等）？
- 为什么流式场景需要特殊处理？
- 水库采样如何估计分布？

**Tags:** #ads #ctr #numerical-features #streaming #embedding #distribution
