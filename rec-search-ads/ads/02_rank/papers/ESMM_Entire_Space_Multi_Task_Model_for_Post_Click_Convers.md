# ESMM: Entire Space Multi-Task Model for Post-Click Conversion Rate Estimation

> 来源：SIGIR 2018, Alibaba | 年份：2018 | 领域：ads/02_rank（CVR预估/多任务学习）

## 问题定义

CVR（转化率）预估面临两个核心问题：

1. **样本选择偏差（Sample Selection Bias, SSB）**：
   - CVR 模型传统上只在**点击样本**上训练（因为只有点击后才知道是否转化）
   - 但推理时需要在**所有展示样本**上预估
   - 点击样本 ≠ 全展示空间的分布 → 系统性偏差
   - 类比：只用及格学生的成绩预测全班平均分

2. **数据稀疏（Data Sparsity）**：
   - 转化样本量 ≪ 点击样本量（通常差 10-100 倍）
   - 点击样本量 ≪ 展示样本量（通常差 10-50 倍）
   - CVR 模型可用训练数据极少，模型欠拟合严重

**业务背景**：淘宝广告系统的 eCPM = bid × pCTR × pCVR（CPA 出价），CVR 预估精度直接影响广告主 ROI 和平台收入。

## 模型结构图

```
┌──────────────────────────────────────────────────────────┐
│                    全展示空间（Entire Space）               │
│                                                          │
│   Label: click(0/1)              Label: click∧convert    │
│          ↓                              ↓                │
│   ┌──────┴──────┐              ┌────────┴────────┐      │
│   │  CTR Tower  │              │  pCTCVR = pCTR   │      │
│   │  (Main Task)│              │         × pCVR   │      │
│   │             │              │   (监督信号)       │      │
│   │  pCTR       │──────┐      └────────┬────────┘      │
│   └──────┬──────┘      │               ↑                │
│          ↑             ×          ┌─────┴──────┐        │
│   ┌──────┴──────┐      │         │  CVR Tower  │        │
│   │   CTR MLP   │      └────────→│  (Aux Task) │        │
│   └──────┬──────┘                │  pCVR       │        │
│          ↑                       └──────┬──────┘        │
│   ┌──────┴──────┐                ┌──────┴──────┐        │
│   │  CTR-specific│               │ CVR-specific │        │
│   │  Layers      │               │ Layers       │        │
│   └──────┬──────┘                └──────┬──────┘        │
│          ↑                              ↑                │
│          └──────────────┬───────────────┘                │
│                  ┌──────┴──────┐                         │
│                  │   Shared     │                         │
│                  │  Embedding   │                         │
│                  └──────┬──────┘                         │
│                         ↑                                │
│                  [User, Item, Context Features]           │
└──────────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：概率分解（核心思想）

$$p(y=1, z=1 | x) = p(y=1 | x) \times p(z=1 | y=1, x)$$

即：

$$\text{pCTCVR}(x) = \text{pCTR}(x) \times \text{pCVR}(x)$$

**解释：**
- $x$：展示样本的特征
- $y$：是否点击（click）
- $z$：是否转化（conversion）
- $p(y=1|x) = \text{pCTR}$：点击概率
- $p(z=1|y=1,x) = \text{pCVR}$：点击后转化概率
- $p(y=1,z=1|x) = \text{pCTCVR}$：点击且转化概率

### 公式2：损失函数

$$\mathcal{L} = \mathcal{L}_{CTR} + \mathcal{L}_{CTCVR}$$

$$\mathcal{L}_{CTR} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\text{pCTR}_i) + (1-y_i) \log(1-\text{pCTR}_i)]$$

$$\mathcal{L}_{CTCVR} = -\frac{1}{N} \sum_{i=1}^{N} [y_i z_i \log(\text{pCTCVR}_i) + (1-y_i z_i) \log(1-\text{pCTCVR}_i)]$$

**解释：**
- $N$：全展示空间样本数
- **注意**：没有单独的 $\mathcal{L}_{CVR}$！CVR 通过 pCTCVR = pCTR × pCVR 的梯度反传间接训练
- 两个 loss 都在全展示空间上定义 → 自然解决样本选择偏差

### 公式3：共享 Embedding 的知识迁移

$$e_i^{CTR} = e_i^{CVR} = \text{Embedding}(x_i)$$

**解释：**
- CTR 塔和 CVR 塔共享底层 Embedding
- CTR 有丰富样本（展示即可标注），学到的 Embedding 通过共享迁移到 CVR
- 缓解 CVR 的数据稀疏问题

## 与基线方法对比

| 方法 | 核心区别 | 优势 | 劣势 |
|------|---------|------|------|
| **独立 CVR 模型** | 只在点击空间训练 | 简单直接 | 样本选择偏差 + 数据稀疏 |
| **ESMM** | 全空间 pCTR×pCVR | 解决 SSB + 数据稀疏 | CVR 上界受 CTR 影响 |
| **ESM²** | 加入中间节点(加购/收藏) | 更细粒度的转化路径 | 需要更多标注数据 |
| **ESCM²** | 加因果约束 | 解决 CVR 高估 | 训练更复杂 |
| **MMoE+CVR** | 多专家多任务 | 灵活的任务关系 | 不解决 SSB |
| **DCMT** | 反事实多任务 | 因果角度解决 SSB | 需要因果假设 |

## 实验结论

- **淘宝购买预测**：CVR AUC 提升 3.4%（vs 独立 CVR 模型）
- **pCTCVR AUC**：提升 2.1%
- **低交互商品**（数据稀疏严重）：CVR AUC +5.2%
- **在线 A/B**：GMV 提升 3.8%，广告主 ROI 提升 2.6%
- 共享 Embedding 的贡献：去掉共享后 CVR AUC 下降 1.8%

## 工程落地要点

1. **Label 构建**：CTR label = 是否点击（全空间）；CTCVR label = 是否点击且转化（全空间，未点击的样本 CTCVR=0）
2. **负样本处理**：全空间负样本量极大（曝光未点击），需按 CTR 分层采样平衡正负比例
3. **任务权重**：$\mathcal{L}_{CTR}$ 和 $\mathcal{L}_{CTCVR}$ 的权重建议 1:1 起步，可用 Uncertainty Weighting 自动学习
4. **延迟转化**：购买可能发生在点击后 1-7 天，需要设置归因窗口（attribution window）
5. **数值稳定性**：pCTR × pCVR 两个小数相乘可能下溢，实际用 log 空间计算：$\log(\text{pCTCVR}) = \log(\text{pCTR}) + \log(\text{pCVR})$

## 面试考点

**Q1：为什么 CVR 预估会有样本选择偏差？**
> 训练 CVR 模型只用点击样本（只有点击后才知道是否转化），但推理时在所有展示上预估。点击样本是全展示空间的有偏子集（用户倾向点击感兴趣的），导致 CVR 模型在未点击样本上预估不准。

**Q2：ESMM 的乘法结构 pCTCVR = pCTR × pCVR 如何解决 SSB？**
> pCTCVR 的 label（点击且转化）在全展示空间可直接标注，pCTR 的 label（点击）也在全展示空间标注。通过乘法关系，pCVR 在全空间上通过梯度反传训练，无需在点击空间单独建模。

**Q3：ESMM 如何解决数据稀疏问题？**
> 共享底层 Embedding：CTR 塔的丰富样本（CTR 样本 = 展示样本 >> 转化样本）学到的 Embedding 通过共享传递给 CVR 塔，相当于用 CTR 的"大数据"帮 CVR 的"小数据"学习更好的特征表示。

**Q4：ESMM 的局限性有哪些？**
> ① CVR 预估质量上界受 CTR 塔影响（pCTR 不准则 pCVR 梯度有噪声）② 假设 CTR 和 CVR 塔结构相同（实际可能需要不同结构）③ 无法处理延迟转化 ④ pCTR × pCVR 数值稳定性问题。

**Q5：ESMM 为什么没有单独的 CVR loss？**
> 因为 CVR 的 label（是否转化）只在点击样本上有定义。如果在全空间加 CVR loss，未点击样本的转化 label 是 missing（不是0），直接标为0会引入偏差。通过 pCTCVR loss 间接训练 CVR 塔避免了这个问题。

**Q6：ESMM 后续有哪些改进工作？**
> ① ESM²：加入中间节点（曝光→点击→加购→购买），更细粒度的转化路径建模 ② ESCM²：引入因果约束（反事实正则），解决 CVR 在部分区域被高估的问题 ③ DCMT：从因果推断角度用反事实学习解决 SSB。

**Q7：工业界如何处理延迟转化问题？**
> ① 设定归因窗口（如点击后 7 天内的转化归因给该点击）② 使用"转化回传"机制实时更新 label ③ 在 loss 中对未归因样本降权而非直接标为负样本 ④ 多窗口建模（1天CVR、7天CVR 分别预估）。
