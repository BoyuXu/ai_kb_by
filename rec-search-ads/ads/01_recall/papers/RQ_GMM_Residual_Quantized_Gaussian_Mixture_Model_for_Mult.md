# RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction

> 来源：[https://arxiv.org/abs/2406.xxxxx] | 日期：20260313 | 领域：ads

## 问题定义
在多模态CTR预测中，图像/视频特征通常以连续向量（CNN/ViT提取的Embedding）形式输入模型。然而连续特征与CTR模型中离散的ID特征存在"模态鸿沟"：ID特征通过Embedding表更新，学习到行为模式；而连续视觉特征经过MLP压缩后语义损失大，两者难以有效交叉融合。本文提出RQ-GMM：将连续多模态特征离散化为语义token序列，使视觉特征能像ID特征一样参与Embedding交叉，实现真正的多模态融合。

## 核心方法与创新点
- **残差量化（Residual Quantization, RQ）**：使用多层VQ-VAE（Vector Quantization）对视觉Embedding进行层级量化，每层捕捉前一层的残差，最终将连续向量转化为K个离散code（如K=8层，每层256个码字），保留细粒度语义。
- **高斯混合建模（Gaussian Mixture Model, GMM）**：每个量化码字对应一个高斯分布，而非固定向量，用概率分布表示语义的不确定性，解决单点量化的精度损失问题。量化时选择后验概率最高的码字（最大似然量化）。
- **语义离散化后的Embedding学习**：量化产生的code序列作为新的"视觉ID"，在CTR模型中为每个code学习一个Embedding，通过End-to-End训练使视觉code的Embedding学到CTR相关语义。
- **跨模态交叉**：视觉code Embedding与商品ID Embedding、用户行为Embedding进行交叉（DIN attention或FM交叉），实现视觉语义与行为语义的深度融合。

## 实验结论
- 在电商广告数据集（服装、家居类目）上，RQ-GMM相比直接使用连续视觉特征的DCN v2 AUC提升0.35%，Logloss降低0.5%。
- 视觉类目（服装、珠宝）提升最显著（AUC+0.5%），文字为主的商品类目（图书）提升较小（AUC+0.1%），符合视觉信息重要性的直觉。
- 量化层数K=8时性能最优，K>8开始出现过拟合，K<4时视觉语义损失太大，K=8是推荐配置。
- GMM vs 硬量化（标准VQ）：GMM方式AUC高出0.1%，Gumbel-Softmax软量化方式与GMM相当，但GMM在训练稳定性上更优。

## 工程落地要点
- **量化码本的离线预计算**：所有商品的视觉特征量化（RQ编码）在离线批量完成，线上CTR预测时直接查表（code → Embedding），无需实时运行量化器。
- **码本更新策略**：随着商品更新，需要定期重新训练量化器（建议每月一次）。增量更新时新商品量化后加入索引，不影响已有商品的code，保持线上稳定性。
- **内存占用**：RQ-GMM引入额外的code Embedding表（K×|codebook|×d），对于K=8、|codebook|=256、d=64的配置，额外内存约8×256×64×4 bytes ≈ 500KB，完全可接受。
- **冷启动商品**：新商品可立即通过视觉内容获得视觉code，无需等待行为数据，有效解决新商品在视觉维度的冷启动。

## 常见考点
**Q1: 为什么连续视觉特征难以与ID特征有效融合？**
A: ID特征通过Embedding表学习，本质是查表操作，梯度只流向被激活的Embedding行，更新高效且稳定；连续视觉特征通常经过MLP变换后与ID特征交叉，MLP的梯度传播路径更长，且连续特征空间与ID Embedding空间的语义结构不同，导致交叉效果差。离散化后视觉特征变成"视觉ID"，可以直接使用与ID特征相同的Embedding机制。

**Q2: VQ-VAE（向量量化）和标准自编码器有什么区别？**
A: 标准AE的隐空间是连续的（任意实数向量）；VQ-VAE的隐空间是离散的（从有限码本中选择最近邻码字）。VQ-VAE解决了离散隐变量的梯度不可导问题（使用直通估计器Straight-Through Estimator），使码本和编码器可以端到端训练。

**Q3: 如何评估多模态融合的有效性？**
A: 除整体AUC外，关键分析包括：(1) 子群体分析（视觉主导vs文字主导类目的提升差异）；(2) 特征重要性分析（视觉code Embedding的梯度归因）；(3) Attention权重可视化（视觉token对最终预测的注意力权重分布）；(4) 在冷启动商品上的单独评测（此时视觉特征是唯一信号来源）。
