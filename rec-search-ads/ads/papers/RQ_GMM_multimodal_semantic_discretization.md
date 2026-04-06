# RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR

**ArXiv:** 2602.12593 | **Date:** 2026-02 | **Platform:** 大型短视频平台（数亿 DAU）

## 核心问题
多模态内容对 CTR 预估至关重要，但预训练模型的连续 embedding 难以直接用于 CTR 模型：
1. 优化目标不对齐
2. 联合训练时收敛速度不一致

现有离散化方法（RQ-VAE 等）的局限：
- codebook 利用率低
- 重建准确性不足
- 语义判别性弱

## 方案：RQ-GMM

### 核心创新
用**高斯混合模型（GMM）+ 残差量化（RQ）**代替硬聚类：
- GMM 通过概率建模和软赋值，更好捕获 embedding 空间的分布特征
- 软赋值 → 更好的分布特征捕获
- → 更具判别性的语义 ID

### 技术优势
- 更高 codebook 利用率
- 更好的多模态语义判别性
- 优于 RQ-VAE 的重建准确性

## 线上 A/B 测试
vs RQ-VAE：广告主价值（Advertiser Value）**+1.502%**

## 面试考点
- 为什么连续 embedding 难以直接用于 CTR 模型？
- GMM 软赋值 vs K-means 硬聚类的区别？
- Semantic ID（SID）在推荐/广告中的应用？

**Tags:** #ads #ctr #multimodal #semantic-id #quantization #gmm
