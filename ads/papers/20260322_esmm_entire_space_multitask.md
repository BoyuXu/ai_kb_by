# ESMM: Entire Space Multi-Task Model for Post-Click Conversion Rate Estimation

> 来源：arxiv (SIGIR 2018, Alibaba) | 日期：20260322 | 领域：广告系统（经典工作补充）

## 问题定义

CVR（转化率）预估面临两个核心问题：
1. **样本选择偏差（Sample Selection Bias）**：CVR 模型只在被点击的样本上训练，但需要在全展示空间做预估
2. **数据稀疏（Data Sparsity）**：转化样本量远少于点击样本（通常相差 100x），模型欠拟合

## 核心方法与创新点

- **全空间建模**：在整个展示空间（所有曝光）训练，而非只在点击空间
- **乘法分解**：
  ```
  pCTCVR = pCTR × pCVR
  ```
  联合训练 CTR 和 CVR 任务，通过 pCTCVR 的监督信号反向传播到 CVR 塔
- **共享 Embedding**：CTR 塔和 CVR 塔共享底层用户/商品 Embedding，利用 CTR 的丰富样本迁移知识到 CVR
- **无需引入偏差校正**：通过乘法结构自然解决了样本选择偏差，CVR 的训练在全空间进行

## 实验结论

- 淘宝购买预测：AUC 提升 3.4%（vs 独立 CVR 模型）
- pCTCVR AUC 提升 2.1%
- 数据稀疏场景（低交互商品）提升最显著：CVR AUC +5.2%
- 在线 A/B：GMV 提升 3.8%，广告主 ROI 提升 2.6%

## 工程落地要点

- **Label 构建**：CTR label = 是否点击（全空间样本）；CVR label = 是否转化（全空间，未点击=0）；CTCVR label = 是否点击且转化
- **负样本采样**：全空间负样本量极大，需按 CTR 分层采样平衡正负比例
- **任务权重**：CTCVR loss 和 CTR loss 的权重需调优，建议 1:1 起步
- **扩展**：ESM²（增加收藏、加购中间转化节点）；ESCM²（增加因果约束防止 CVR 高估）

## 面试考点

1. **Q：为什么 CVR 预估会有样本选择偏差？**
   A：训练 CVR 模型只用点击样本，但推理时在所有曝光上预估。点击样本不代表全部曝光空间的分布，导致未被点击的高质量广告的 CVR 被低估

2. **Q：ESMM 如何解决数据稀疏问题？**
   A：共享底层 Embedding 让 CVR 塔借助 CTR 的丰富样本（CTR 样本通常是 CVR 的 10-100 倍）学习更好的表示

3. **Q：ESMM 的局限性？**
   A：假设 CTR 和 CVR 的特征塔结构相同；CVR 预估质量上界受 CTR 塔影响；无法处理延迟转化（购买可能发生在点击后 7 天）
