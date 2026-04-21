# Loss 函数全景对比 — 面试向深度总结

> 标签：#loss #cross-entropy #focal-loss #triplet #InfoNCE #BPR #搜广推 #面试

---

## 总对比表

| Loss 名称 | 类别 | 公式速记 | 典型场景 | 优点 | 缺点 |
|-----------|------|---------|---------|------|------|
| Cross Entropy (CE) | 分类 | $-\sum y_i \log \hat{y}_i$ | 多分类、CTR | 梯度清晰，收敛快 | 对类别不平衡敏感 |
| Binary CE (BCE) | 二分类 | $-[y\log p + (1-y)\log(1-p)]$ | CTR/CVR 预估 | 概率输出天然校准 | 正负样本比悬殊时效果差 |
| Focal Loss | 分类 | $-\alpha_t(1-p_t)^\gamma \log p_t$ | 目标检测、CTR 长尾 | 自动降权易分样本 | 超参 γ/α 需调 |
| Label Smoothing CE | 分类 | CE with $y_i = (1-\varepsilon)y_i + \varepsilon/K$ | NLP/CV 大模型 | 防过拟合、提升校准 | 会轻微降低 top-1 置信 |
| MSE | 回归 | $\frac{1}{n}\sum(y-\hat{y})^2$ | 回归、蒸馏 logit | 处处可导 | 对异常值敏感 |
| MAE | 回归 | $\frac{1}{n}\sum\|y-\hat{y}\|$ | 鲁棒回归 | 对异常值鲁棒 | 零点不可导，收敛慢 |
| Huber | 回归 | MSE/MAE 分段 | 鲁棒回归 | 兼顾 MSE+MAE | 需调 δ |
| Triplet Loss | 度量学习 | $\max(0, d_{ap}-d_{an}+m)$ | 人脸/检索 | 直观 | 三元组采样困难 |
| Contrastive (SimCLR) | 度量学习 | NT-Xent with τ | 自监督预训练 | 无需标注 | 需大 batch |
| InfoNCE | 度量学习 | $-\log\frac{e^{sim/\tau}}{\sum e^{sim/\tau}}$ | 双塔召回、CLIP | 多负样本高效 | τ 敏感 |
| ArcFace | 度量学习 | $\cos(\theta + m)$ | 人脸识别 | 角度间隔判别力强 | 需归一化特征 |
| KL Divergence | 蒸馏 | $\sum P\log(P/Q)$ | 知识蒸馏 | 信息论基础扎实 | 非对称 |
| BPR | 排序 | $-\log\sigma(\hat{x}_{ui}-\hat{x}_{uj})$ | 推荐排序 | pairwise 天然适配 | 采样效率问题 |
| Pairwise Hinge | 排序 | $\max(0, m - (s_i - s_j))$ | 搜索排序 | 简单直接 | margin 需调 |
| LambdaRank | 排序 | NDCG-aware gradient | 搜索 LTR | 直接优化排序指标 | 实现复杂 |
| ListNet | 排序 | top-1 概率分布 CE | 搜索 LTR | listwise 全局优化 | 计算量较大 |

---

## 1. 分类 / 回归 Loss

### 1.1 Cross Entropy (CE)

**公式（多分类）：**

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

其中 $\hat{y}_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$。

**梯度直觉：**

对 logit $z_i$ 的梯度为 $\hat{y}_i - y_i$，即预测概率与真实标签的差。这意味着：
- 预测越准 → 梯度越小 → 自动 "放过" 已学好的样本
- 预测越离谱 → 梯度越大 → 纠错力度大

**多分类 vs 二分类：**
- 多分类：softmax + CE，输出 K 个 logit，互斥
- 二分类：sigmoid + BCE，等价于 K=2 的 softmax CE
- 多标签：每个类独立 sigmoid + BCE，非互斥

> **面试一句话：** CE 的梯度就是 $\hat{y} - y$，天然正比于误差大小，所以收敛快。

**值域与梯度：**
- 值域：$[0, +\infty)$。当预测完全正确（$\hat{y}_i = y_i$）时 loss=0；当预测概率趋向 0 时 loss 趋向 $+\infty$。
- 梯度 $\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i \in (-1, 1)$，有界且平滑，不存在梯度爆炸。当预测接近正确时梯度趋向 0，但不会出现梯度消失的训练困难——因为此时 loss 本身已经很小，模型已收敛。
- softmax 的饱和区（某个 logit 远大于其他）不会造成梯度消失，这是 CE + softmax 组合的关键优势，相比 sigmoid + MSE 的组合（后者在饱和区梯度指数衰减）。

**文献：**
- CE 作为损失函数源自信息论，首见于 Shannon (1948) "A Mathematical Theory of Communication"。
- 在神经网络中与 softmax 配合使用，由 Bridle (1990) 系统提出，后经 LeCun et al. (1998) 在 LeNet 中广泛验证。

**原理分析：**
- CE + softmax 的梯度恰好等于 $\hat{y} - y$，这个简洁形式不是巧合，而是指数族分布的 canonical link function 的性质。sigmoid/softmax 是 Bernoulli/Categorical 分布的自然参数化，CE 是对应的负对数似然——两者配对时梯度自然简化。
- 相比 MSE 用于分类的方案（$\sum(y_i - \hat{y}_i)^2$），CE 的梯度在 softmax 饱和区不会指数衰减。MSE + softmax 的梯度包含 $\hat{y}_i(1-\hat{y}_i)$ 因子，当预测极端错误时（$\hat{y}_i \approx 0$）反而梯度消失，造成 "明知大错但改不动" 的困境。CE 完全避免了这个问题。

---

### 1.2 Binary Cross Entropy (BCE)

**公式：**

$$\mathcal{L}_{\text{BCE}} = -\left[ y \log(p) + (1-y) \log(1-p) \right]$$

其中 $p = \sigma(z) = \frac{1}{1+e^{-z}}$。

**Sigmoid + BCE = Logistic Regression：**

这是最经典的组合。CTR 预估中，模型最后一层用 sigmoid + BCE，输出的 $p$ 可直接作为点击概率的校准估计。

**数值稳定性 — log-sum-exp trick：**

直接计算 $\log(\sigma(z))$ 在 $z$ 很大或很小时会溢出。稳定实现：

$$\mathcal{L} = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

PyTorch 的 `F.binary_cross_entropy_with_logits` 内部就是这个实现，**面试中问到数值稳定性必须能写出来**。

> **面试一句话：** BCE 直接接 logit 用 `with_logits` 版本，避免 sigmoid 后 log 的数值爆炸。

**值域与梯度：**
- 值域：$[0, +\infty)$。完美预测时 loss=0，极端错误时趋向 $+\infty$。
- 对 logit $z$ 的梯度为 $\sigma(z) - y$，范围 $(-1, 1)$，有界平滑。
- 与 CE 相同，sigmoid + BCE 组合的梯度不含 $\sigma(z)(1-\sigma(z))$ 因子（该因子在 sigmoid + MSE 中出现），因此在 logit 绝对值很大时不会梯度消失。
- 数值风险集中在 $\log$ 运算：当 $p \to 0$ 时 $\log p \to -\infty$，故必须用 log-sum-exp 稳定实现。

**文献：**
- BCE 是 logistic regression 的原生损失，可追溯至 Cox (1958) "The Regression Analysis of Binary Sequences"。
- 在深度学习 CTR 预估中广泛使用，由 Google Wide & Deep (Cheng et al., 2016) 和 DeepFM (Guo et al., 2017) 确立为搜广推标配。

**原理分析：**
- BCE 的概率解释是 Bernoulli 分布的负对数似然，因此模型输出可直接解释为后验概率 $P(y=1|x)$，天然具有概率校准性。这在 CTR 预估中至关重要——广告竞价系统依赖校准后的点击概率进行出价计算，偏差直接影响收入。
- 相比 pairwise loss（如 BPR），BCE 是 pointwise 的，学到的是绝对概率而非相对排序。这使得不同请求之间的预测值可比较（pairwise loss 做不到），是 eCPM 排序（$\text{pCTR} \times \text{bid}$）的前提条件。

---

### 1.3 Focal Loss

**公式：**

$$\mathcal{L}_{\text{FL}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中：
- $p_t = p$（正样本时）或 $1-p$（负样本时）
- $\alpha_t$：正/负样本权重（典型 α=0.25 给正样本）
- $\gamma$：聚焦参数（典型 γ=2）

**为什么 γ 有效？**

| 样本类型 | $p_t$ | $(1-p_t)^\gamma$（γ=2） | 效果 |
|---------|-------|------------------------|------|
| 易分负样本 | 0.95 | 0.0025 | loss 降到 CE 的 1/400 |
| 难分样本 | 0.5 | 0.25 | loss 仅降到 CE 的 1/4 |
| 极难样本 | 0.1 | 0.81 | 几乎保留全部 loss |

核心思想：**自动将训练注意力从大量易分样本转移到少量难分样本**，不需要手动做采样。

**γ 的选择：**
- γ=0 退化为标准 CE
- γ=1 温和降权
- γ=2 论文推荐，大多数场景 work
- γ=5 极端聚焦，慎用

> **面试一句话：** Focal Loss 通过 $(1-p_t)^\gamma$ 调制因子自动降低易分样本的 loss 权重，解决类别不平衡问题。

**值域与梯度：**
- 值域：$[0, +\infty)$，但实际上界远小于 CE。当 $\gamma=2$ 时，即使完全错误的样本（$p_t \to 0$），loss 也被 $\alpha_t$ 控制在有限范围内。
- 梯度：$\frac{\partial \mathcal{L}}{\partial p_t} = -\alpha_t \left[\gamma(1-p_t)^{\gamma-1}\log(p_t) + \frac{(1-p_t)^\gamma}{p_t}\right]$。对于高置信样本（$p_t \to 1$），梯度被 $(1-p_t)^\gamma$ 压制到接近 0；对于难分样本（$p_t \approx 0.5$），梯度保持在 CE 量级的 $1/4$（$\gamma=2$ 时）。
- 极端情况：$p_t \to 0$ 时梯度不会爆炸（$(1-p_t)^\gamma$ 的抑制和 $1/p_t$ 的放大部分抵消），但 $\gamma$ 过大（>5）时梯度在 $p_t$ 中间区域出现非单调行为，可能导致训练不稳定。

**文献：**
- 首见于 Lin et al. (2017) "Focal Loss for Dense Object Detection"（RetinaNet 论文），由 Facebook AI Research 提出。
- 该论文首次系统性证明了目标检测中类别不平衡是 one-stage 检测器精度低于 two-stage 的根本原因，Focal Loss 使 RetinaNet 首次在精度上超越 Faster R-CNN。

**原理分析：**
- Focal Loss 解决的核心问题是：在极度不平衡的数据中（如目标检测中前景:背景 ≈ 1:1000），大量易分负样本虽然单个 loss 小，但数量巨大，总 loss 主导了梯度方向，导致模型被 "淹没" 在易分样本的梯度中，无法有效学习难分样本。
- 传统解法是 hard negative mining（OHEM），但 OHEM 完全丢弃了易分样本的梯度信号，且需要额外的采样流程。Focal Loss 通过连续的调制因子 $(1-p_t)^\gamma$ 实现了 "软性" mining——易分样本不是被丢弃，而是被降权，保留了梯度的连续性和训练的稳定性。
- $\gamma$ 的数学本质是控制 loss 曲线的 "弯曲程度"：$\gamma=0$ 时为线性（标准 CE），$\gamma$ 越大曲线越 "下凹"，对高置信区域的惩罚越弱。这比手动设置类别权重 $\alpha$ 更精细，因为它是 instance-level 的自适应，而非 class-level 的固定权重。

---

### 1.4 Label Smoothing CE

**公式：**

将 hard label $y_i \in \{0, 1\}$ 替换为 soft label：

$$y_i^{\text{smooth}} = (1 - \varepsilon) \cdot y_i + \frac{\varepsilon}{K}$$

典型 $\varepsilon = 0.1$，$K$ 为类别数。

**为什么有效？**
1. **防止过拟合**：模型不会追求把 logit 推到正无穷
2. **提升校准（calibration）**：预测概率更接近真实后验概率 → 对 [[ctr_calibration]] 很重要
3. **正则化效果**：等价于在 CE 上加了一个 KL(uniform || p) 的正则项

**缺点**：如果 ground truth 确实是 100% 置信的，label smoothing 会引入噪声。知识蒸馏中通常不对 teacher 的 soft label 再做 smoothing。

> **面试一句话：** Label Smoothing 本质是在 CE 上加了均匀分布的 KL 正则，防止模型输出过于自信。

**值域与梯度：**
- 值域：$[0, +\infty)$，但由于目标分布不再是 one-hot，理论最小值不再是 0，而是 $H(y^{\text{smooth}}) = -(1-\varepsilon)\log(1-\varepsilon) - \varepsilon\log(\varepsilon/K) \cdot (K-1)/K > 0$（soft label 自身的熵）。
- 梯度与标准 CE 结构相同（$\hat{y}_i - y_i^{\text{smooth}}$），但目标值从 0/1 变成了 $\varepsilon/K$ 和 $1-\varepsilon+\varepsilon/K$。效果是 logit 不需要推到 $\pm\infty$ 就能达到最小 loss，抑制了权重的无限增长。
- 梯度平滑性优于标准 CE：因为 soft label 使得即使预测正确的类别，仍有微小的 "残余梯度"，相当于持续施加向均匀分布靠拢的力，起到隐式正则作用。

**文献：**
- 首见于 Szegedy et al. (2016) "Rethinking the Inception Architecture"（Inception v2/v3 论文），Google 提出。
- 后被 Vaswani et al. (2017) "Attention Is All You Need" 在 Transformer 中使用（$\varepsilon=0.1$），成为 NLP 大模型的标配训练技巧。

**原理分析：**
- Label Smoothing 解决的根本问题是 CE + one-hot 标签的 "过度自信" 倾向。数学上，CE 在 one-hot 标签下的最优解要求正确类的 logit 趋向 $+\infty$、其他类趋向 $-\infty$——这导致模型权重持续增长，过拟合风险加大，且预测概率极端化，校准性变差。
- Label Smoothing 可以分解为两项：$(1-\varepsilon)\mathcal{L}_{\text{CE}} + \varepsilon \cdot D_{\text{KL}}(U \| \hat{y})$，其中 $U$ 是均匀分布。第二项将预测分布拉向均匀，防止任何类的概率过于极端。
- 在知识蒸馏中不应对 teacher 的 soft label 再做 smoothing，因为 teacher 的概率分布本身已包含类间关系信息（dark knowledge），再 smooth 会破坏这些信息。

---

### 1.5 MSE / MAE / Huber

| Loss | 公式 | 对异常值 | 梯度 | 适用 |
|------|------|---------|------|------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 敏感（平方放大） | $2(\hat{y}-y)$，正比误差 | 标准回归 |
| MAE | $\frac{1}{n}\sum\|y-\hat{y}\|$ | 鲁棒 | $\pm 1$，恒定大小 | 有异常值时 |
| Huber | 分段：小误差用 MSE，大误差用 MAE | 鲁棒 | 小误差正比，大误差恒定 | 兼顾两者 |

**Huber Loss 公式：**

$$L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & \text{if } |a| \le \delta \\ \delta(|a| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

> **面试一句话：** MSE 梯度正比于误差但对 outlier 敏感，MAE 对 outlier 鲁棒但零点不可导且收敛慢，Huber 是两者的平滑过渡。

**值域与梯度：**
- **MSE**：值域 $[0, +\infty)$。梯度 $2(\hat{y}-y)$，随误差线性增长，无上界——大误差产生大梯度，这正是对异常值敏感的原因。误差为 0 时梯度为 0，且处处光滑可导。
- **MAE**：值域 $[0, +\infty)$。梯度为 $\pm 1$（常数），不随误差大小变化——大误差和小误差获得相同的梯度幅值，因此对异常值鲁棒。但在 $\hat{y}=y$ 处不可导（梯度从 -1 跳变为 +1），导致在最优点附近振荡，收敛困难。
- **Huber**：值域 $[0, +\infty)$。在 $|a| \le \delta$ 区域梯度为 $a$（像 MSE），在 $|a| > \delta$ 区域梯度为 $\pm\delta$（像 MAE 但幅值为 $\delta$）。处处连续可导（包括 $|a|=\delta$ 的分界点），兼具 MSE 在小误差区的精细调节和 MAE 在大误差区的鲁棒性。

**文献：**
- **MSE**：作为最小二乘法的基础，可追溯至 Gauss (1809) 和 Legendre (1805)。在神经网络中的使用贯穿整个历史。
- **MAE**：作为 L1 损失同样历史悠久，Boscovich (1757) 首次提出最小绝对偏差回归。
- **Huber Loss**：首见于 Huber (1964) "Robust Estimation of a Location Parameter"，是鲁棒统计学的奠基工作之一。在深度学习中被 DQN (Mnih et al., 2015) 用于 Q 值回归，后在强化学习中广泛使用。

**原理分析：**
- MSE 的核心问题是平方项使梯度随误差线性增长。若数据中存在少量标签错误或异常值（误差为 100），其梯度是正常样本（误差为 1）的 100 倍，单个异常值就能主导整个 batch 的梯度方向。这在现实数据（标注噪声普遍存在）中是严重问题。
- MAE 通过常数梯度彻底解决了异常值问题，但代价是在最优点附近的行为：当模型已经很准（误差很小）时，梯度仍然是 $\pm 1$，"大力出奇迹" 地调整参数，导致在最优解附近来回振荡无法精细收敛。
- Huber 的设计精髓是用 $\delta$ 参数划分 "正常误差" 和 "异常误差" 的边界。$\delta$ 小则更鲁棒但收敛慢，$\delta$ 大则行为接近 MSE。实践中 $\delta$ 通常设为数据误差的中位数量级，但这引入了一个需要调节的超参数。在 DQN 中使用 Huber Loss 是因为 TD error 经常出现大幅波动（目标网络更新导致），Huber 可以防止 Q 值梯度爆炸。

---

### 1.6 决策树：CE vs Focal vs Label Smoothing

```
你的任务是什么？
│
├─ 标准分类（类别均衡）→ 用 CE
│   └─ 模型过于自信 / 需要校准？ → 加 Label Smoothing (ε=0.1)
│
├─ 类别极度不平衡（正:负 < 1:100）
│   ├─ 可以做采样？ → 下采样/上采样 + CE
│   └─ 不想碰数据？ → Focal Loss (γ=2, α=0.25)
│
├─ CTR 预估（搜广推）
│   ├─ 需要概率校准 → BCE + Label Smoothing
│   └─ 正样本极少 → BCE + Focal
│
└─ 知识蒸馏 → soft CE (with temperature) + 硬标签 CE 加权
```

---

## 2. 度量学习 Loss

> 相关阅读：[[contrastive_learning]]、[[embedding_ann]]

### 2.1 Triplet Loss

**公式：**

$$\mathcal{L}_{\text{triplet}} = \max\left(0,\; d(a, p) - d(a, n) + m\right)$$

- $a$: anchor（锚点）
- $p$: positive（同类样本）
- $n$: negative（异类样本）
- $m$: margin（间隔，典型 0.2-1.0）
- $d$: 距离函数（通常 L2 或 cosine）

**Hard Negative Mining（核心中的核心）：**

| 策略 | 定义 | 效果 |
|------|------|------|
| Easy negative | $d(a,n) \gg d(a,p) + m$ | loss=0，无梯度，白算 |
| Semi-hard negative | $d(a,p) < d(a,n) < d(a,p) + m$ | 最有信息量 |
| Hard negative | $d(a,n) < d(a,p)$ | 信息量大但可能导致训练崩塌 |

实践建议：**先用 semi-hard，稳定后逐步引入 hard**。Google FaceNet 论文推荐 batch-all 策略。

> **面试一句话：** Triplet Loss 的效果完全取决于负样本挖掘策略，semi-hard negative 是最优起点。

**值域与梯度：**
- 值域：$[0, +\infty)$。当 $d(a,p) - d(a,n) + m \le 0$ 时 loss=0（hinge 截断），否则等于 $d(a,p) - d(a,n) + m$。
- 梯度行为：在 loss=0 区域梯度为 0（easy triplet 完全不贡献训练信号）；在 loss>0 区域梯度为常数（对距离的偏导为 $\pm 1$）。这意味着训练中大部分 triplet 的梯度为 0——随着训练进行，有效 triplet 比例快速下降，出现 "梯度饥饿" 现象。
- 在 $d(a,p) - d(a,n) + m = 0$ 处不可导（hinge 点），但实践中用 sub-gradient 不影响训练。

**文献：**
- 度量学习中的 triplet loss 概念由 Weinberger & Saul (2009) "Distance Metric Learning for Large Margin Nearest Neighbor Classification" (LMNN) 奠基。
- 在深度学习中大规模使用始于 Schroff et al. (2015) "FaceNet: A Unified Embedding for Face Recognition and Clustering"（Google FaceNet），该论文同时提出了 semi-hard mining 策略。

**原理分析：**
- Triplet Loss 的设计目标是直接在 embedding 空间中构建几何结构：同类样本聚拢、异类样本分离，且间隔不小于 margin $m$。这比分类 loss（CE）更直接地服务于检索任务——检索关心的是 embedding 的相对距离，不需要分类边界。
- 核心瓶颈是 $O(N^3)$ 的三元组空间。即使 batch 内枚举，大部分 triplet 也是 easy 的（loss=0），导致有效梯度信号极度稀疏。这就是为什么 mining 策略比 loss 本身更重要。FaceNet 选择 semi-hard 而非 hard negative 是因为：hardest negative 往往是标注错误或跨类别的极端样本，直接用会导致 embedding 空间坍缩（所有点挤在一起）。
- InfoNCE 后来解决了 triplet loss 的采样效率问题：通过 softmax 同时利用 batch 内所有负样本，而非一次只看一个 negative。

---

### 2.2 Contrastive Loss (SimCLR 风格)

**NT-Xent (Normalized Temperature-scaled Cross Entropy)：**

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- $\text{sim}(u, v) = \frac{u^\top v}{\|u\|\|v\|}$（cosine similarity）
- $\tau$：温度参数（典型 0.05-0.5）
- $2N$：一个 batch 中 $N$ 对正样本，共 $2N$ 个样本

**温度 τ 的作用：**
- τ 小 → 分布更尖锐 → 更关注 hard negative → 但容易不稳定
- τ 大 → 分布更平滑 → 所有负样本均匀看待 → 学习信号弱
- 典型值：SimCLR 用 0.1，CLIP 用 learnable τ（初始 0.07）

> **面试一句话：** SimCLR 的 NT-Xent 本质是在 batch 内做 softmax CE，温度 τ 控制对 hard negative 的关注程度。

**值域与梯度：**
- 值域：$[0, \log(2N-1)]$。下界 0 在正样本 similarity 远大于所有负样本时达到；上界 $\log(2N-1)$ 在所有样本 similarity 相等时达到（等价于随机猜测的 CE loss）。
- 温度 $\tau$ 对梯度的影响：$\tau$ 小时，softmax 分布尖锐，梯度集中在 hardest negative 上，类似 hard mining，但极端时（$\tau < 0.01$）梯度方差极大，训练不稳定；$\tau$ 大时梯度均匀分布在所有负样本上，学习信号弱但稳定。
- 梯度对 cosine similarity 的偏导为 $\frac{1}{\tau}(p_{ij} - \mathbb{1}_{[j=\text{pos}]})$，结构与 CE 一致，$1/\tau$ 相当于梯度的缩放因子。

**文献：**
- NT-Xent 首见于 Chen et al. (2020) "A Simple Framework for Contrastive Learning of Visual Representations"（SimCLR，Google Research）。
- 其前身是 Sohn (2016) "Improved Deep Metric Learning with Multi-class N-pair Loss" 提出的 N-pair loss，NT-Xent 在此基础上加了温度缩放和 L2 归一化。

**原理分析：**
- NT-Xent 相比 triplet loss 的根本改进是：将 $O(N^3)$ 的三元组问题转化为 batch 内的 softmax 分类问题，一次前向传播利用 batch 内所有 $2(N-1)$ 个负样本，梯度信号密度从 triplet 的稀疏跳变到 dense。
- 温度 $\tau$ 的引入解决了 cosine similarity 的动态范围问题：原始 cosine 值在 $[-1, 1]$，softmax 对这个范围的区分度很低。除以 $\tau$（通常 $\tau \ll 1$）将值域放大到 $[-1/\tau, 1/\tau]$，使 softmax 能有效区分相似度的细微差异。
- 需要大 batch 的原因是：batch size 决定了负样本数量，负样本太少时 softmax 的 "分类任务" 过于简单，模型轻松达到低 loss 但学到的表示并不好（类似分类任务中类别数太少导致判别力不足）。SimCLR 原论文用 batch=4096 才达到最佳效果。

---

### 2.3 InfoNCE

**公式：**

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\sum_{i=0}^{K} \exp(\text{sim}(q, k_i) / \tau)}$$

**与互信息的联系：**

InfoNCE 是互信息 $I(X; Y)$ 的一个下界估计。最小化 InfoNCE loss 等价于最大化 query 和正样本之间的互信息。负样本数 $K$ 越大，下界越紧。

**InfoNCE vs Triplet Loss：**

| 维度 | Triplet Loss | InfoNCE |
|------|-------------|---------|
| 负样本数 | 1 | K（通常 256-65536） |
| 理论基础 | 几何直觉 | 互信息下界 |
| 梯度信号 | 来自单个 hardest 三元组 | 来自所有负样本的加权 |
| 工业应用 | 人脸（FaceNet） | 双塔召回（DSSM/CLIP） |

**在搜广推中的应用：**
- 双塔召回模型的训练 loss：user tower + item tower → InfoNCE
- Batch 内随机负采样是 InfoNCE 的简化版
- 需配合 [[embedding_ann]] 做高效检索

> **面试一句话：** InfoNCE 是互信息的下界，负样本越多估计越紧，是双塔召回模型的标准 loss。

**值域与梯度：**
- 值域：$[0, \log(K+1)]$。$K$ 为负样本数量，上界对应均匀分布（随机猜测）。
- 梯度结构与 softmax CE 相同：对 query-positive similarity 的梯度为 $\frac{1}{\tau}(p_{k^+} - 1)$，对 query-negative similarity 的梯度为 $\frac{1}{\tau} p_{k_i}$，其中 $p$ 是 softmax 概率。
- 负样本数 $K$ 对梯度的影响：$K$ 越大，每个负样本分到的 softmax 概率越小，但 hard negative 仍能获得显著概率——这使得 InfoNCE 隐式地实现了 "soft hard mining"。$K$ 太小时所有负样本概率接近 $1/K$，退化为均匀惩罚。

**文献：**
- 首见于 van den Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding"（CPC，DeepMind）。
- 名称 "NCE" 致敬了 Gutmann & Hyvärinen (2010) 的 Noise Contrastive Estimation，但两者数学形式不同。InfoNCE 在搜广推领域被 DSSM (Huang et al., 2013) 的变体广泛使用，CLIP (Radford et al., 2021) 进一步将其推广到多模态对齐。

**原理分析：**
- InfoNCE 提供了互信息 $I(X;Y)$ 的下界：$I(X;Y) \ge \log(K+1) - \mathcal{L}_{\text{InfoNCE}}$。这个下界随 $K$ 增大而变紧，但 $K$ 有限时下界是松的——这意味着即使 InfoNCE loss 降到 0，学到的表示也未必完美捕获了所有互信息，只是在 $K+1$ 路分类任务中做到了完美区分。
- 在双塔召回中，InfoNCE 的 batch 内负采样等价于从 item 的曝光频率分布中采样负样本，引入了流行度偏差：热门 item 被更频繁地选为负样本，模型会过度 "推远" 热门 item。这就是为什么 YouTube DNN、百度 MOBIUS 等系统需要做 $\log Q$ 校正（$\text{logit} - \log p(\text{item})$）来消除采样偏差。
- 相比 triplet loss 一次只用一个负样本，InfoNCE 通过 softmax 同时利用 $K$ 个负样本的信息，梯度效率高出数个量级——这是 InfoNCE 取代 triplet loss 成为工业标准的根本原因。

---

### 2.4 ArcFace

**公式：**

$$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \ne y_i} e^{s \cdot \cos\theta_j}}$$

- $\theta_{y_i}$：特征向量与目标类权重的夹角
- $m$：angular margin（典型 0.5）
- $s$：缩放因子（典型 64）

**为什么角度间隔优于余弦间隔？**

余弦间隔 $\cos\theta - m$ 在不同角度处产生的实际几何间距不同；而角度间隔 $\cos(\theta + m)$ 在角度空间中是恒定的，判别力更均匀。

**演进路线：** SphereFace（乘法角度间隔） → CosFace（余弦间隔） → ArcFace（加法角度间隔），判别力递增。

> **面试一句话：** ArcFace 在角度空间加常数 margin，比余弦空间加 margin 的几何间距更均匀，判别力更强。

**值域与梯度：**
- 值域：$[0, +\infty)$，结构与 softmax CE 相同，但正确类的 logit 被 margin 压低，使得 loss 总体偏大。
- 梯度特性：对 $\theta_{y_i}$ 的梯度包含 $\sin(\theta_{y_i}+m)$ 因子。当 $\theta_{y_i}+m > \pi$ 时（即特征与权重夹角过大+margin 超过 $\pi$），cosine 值不再单调递减，梯度方向反转，导致训练不稳定。实践中通过 clip $\cos(\theta+m)$ 的下界来处理。
- 缩放因子 $s$ 控制梯度幅值：$s$ 太小时 softmax 概率接近均匀，梯度弱且信号含噪；$s$ 太大时 softmax 极端尖锐，loss landscape 变得陡峭。$s=64$ 是在超球面上类别数为千万级时的经验最优。

**文献：**
- 首见于 Deng et al. (2019) "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"（Imperial College London）。
- 演进路线：SphereFace (Liu et al., 2017，乘法角度 margin $\cos(m\theta)$) → CosFace (Wang et al., 2018，余弦 margin $\cos\theta - m$) → ArcFace (加法角度 margin $\cos(\theta+m)$)。ArcFace 在 LFW/MegaFace 等基准上创造了当时的 SOTA。

**原理分析：**
- ArcFace 的核心洞察是：margin 应该加在角度空间而非余弦空间。$\cos\theta - m$（CosFace）的问题是 cosine 函数的非线性导致相同的余弦间隔 $m$ 在不同角度处对应不同的弧度间隔——在 $\theta \approx 0$ 处弧度间隔大，在 $\theta \approx \pi/2$ 处弧度间隔小。$\cos(\theta+m)$ 直接在角度域加常数 $m$，保证了超球面上各方向的判别间隔恒定。
- SphereFace 的乘法 margin $\cos(m\theta)$ 虽然也在角度域操作，但 $m$ 倍的角度要求非常严格（要求 $\theta < \pi/m$），且 $\cos(m\theta)$ 有多个单调区间，需要复杂的 softmax 修正（退火策略）才能稳定训练。ArcFace 的加法形式 $\cos(\theta+m)$ 只需简单的 clip 就能保证单调性，实现简洁得多。
- 特征归一化（$\|f\|=1$）+ 权重归一化（$\|W_j\|=1$）的约束使得 $W_j^\top f = \cos\theta_j$，将 softmax 分类转化为超球面上的角度分类。这消除了特征范数对分类的影响，使模型专注学习角度判别性，是 ArcFace 成功的前提。

---

## 3. 知识蒸馏 Loss

> 相关阅读：`synthesis/llm/06_知识蒸馏技术全景.md`

### 3.1 KL Divergence

**公式：**

$$D_{\text{KL}}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

**蒸馏中的用法：**

$$\mathcal{L}_{\text{distill}} = T^2 \cdot D_{\text{KL}}\left(\text{softmax}(z^T / T) \| \text{softmax}(z^S / T)\right)$$

- $z^T$：teacher logit
- $z^S$：student logit
- $T$：温度（典型 3-20），越大分布越平滑，暗知识越多
- $T^2$ 前乘：补偿温度升高后梯度变小的效应（Hinton 原论文证明）

**非对称性：**

$D_{\text{KL}}(P\|Q) \ne D_{\text{KL}}(Q\|P)$

| 方向 | 含义 | 行为 |
|------|------|------|
| $D_{\text{KL}}(P\|Q)$（前向） | Q 覆盖 P 的所有模式 | mode-covering → Q 分散 |
| $D_{\text{KL}}(Q\|P)$（反向） | Q 选择 P 的某些模式 | mode-seeking → Q 集中 |

蒸馏通常用前向 KL（让 student 覆盖 teacher 的全部知识），但在 RLHF 中用反向 KL（防止 policy 偏离太远）。

> **面试一句话：** 蒸馏用前向 KL 让 student 覆盖 teacher 全部模式；$T^2$ 系数是为了补偿高温下梯度变小。

**值域与梯度：**
- 值域：$[0, +\infty)$。当 $P=Q$ 时 $D_{\text{KL}}=0$；当 $P$ 和 $Q$ 不重叠时 $D_{\text{KL}} \to +\infty$。
- 梯度关键：$\frac{\partial D_{\text{KL}}}{\partial z_i^S} = \frac{1}{T}(q_i - p_i)$，其中 $p_i, q_i$ 是温度 $T$ 下的 softmax 概率。梯度缩放因子 $1/T$ 意味着温度越高梯度越小——这就是为什么要乘 $T^2$：一个 $T$ 补偿梯度缩放，另一个 $T$ 来自 softmax 的链式法则。
- 当 $Q$ 在某个类上的概率为 0 而 $P$ 不为 0 时（$P(i) > 0, Q(i) = 0$），$D_{\text{KL}} \to +\infty$——这是 mode-covering 行为的数学原因：student 不能在 teacher 有概率的地方给 0 概率。

**文献：**
- KL 散度由 Kullback & Leibler (1951) "On Information and Sufficiency" 提出。
- 在知识蒸馏中的标准用法由 Hinton et al. (2015) "Distilling the Knowledge in a Neural Network" 确立，该论文提出了温度缩放 + $T^2$ 补偿的完整框架。

**原理分析：**
- Hinton 蒸馏的核心洞察是 "dark knowledge"：teacher 的 soft probability 包含了类间相似性信息（如 "猫" 的概率 0.7，"豹" 的概率 0.2，"卡车" 的概率 0.001），这些信息在 hard label（one-hot）中完全丢失。温度 $T$ 的作用是放大这些 dark knowledge——$T$ 越高 softmax 越平滑，类间关系越明显。
- 前向 KL $D_{\text{KL}}(P_T \| Q_S)$ 迫使 student 覆盖 teacher 的所有模式（mode-covering），适合蒸馏场景——student 不应该遗漏 teacher 知道的任何知识。反向 KL $D_{\text{KL}}(Q_S \| P_T)$ 允许 student 只选择性地匹配部分模式（mode-seeking），在 RLHF 中用于约束 policy 不偏离太远（此时不需要完全覆盖，只需要在 reference policy 有概率的地方不要走太远）。
- 为什么不直接用 MSE on logits 替代 KL？因为 KL 蒸馏传递的是概率分布级的信息（类间关系），而 MSE 传递的是 logit 数值。两个 logit 向量 $[10, 5, 1]$ 和 $[20, 10, 2]$ 的 MSE 很大，但 softmax 后概率分布几乎相同——KL 蒸馏能正确忽略这种不影响分布的缩放差异。

---

### 3.2 MSE 蒸馏

直接对 logit 或中间层特征做 MSE：

$$\mathcal{L}_{\text{feat}} = \|f^T(x) - g(f^S(x))\|^2$$

其中 $g$ 是可学习的映射层（维度对齐）。适用于特征蒸馏（FitNets）和层间对齐。

**KL vs MSE 蒸馏对比：**

| 维度 | KL 蒸馏 | MSE 蒸馏 |
|------|---------|---------|
| 作用层 | 输出 logit | logit 或中间特征 |
| 信息量 | 分布级（类间关系） | 数值级（绝对值） |
| 温度敏感 | 是 | 否 |
| 常见用法 | Hinton 软标签蒸馏 | FitNets 特征对齐 |

> **面试一句话：** KL 蒸馏传递的是类间关系（dark knowledge），MSE 蒸馏传递的是特征表示。

**值域与梯度：**
- 值域：$[0, +\infty)$，无上界（距离可任意大）。
- 梯度：$\frac{\partial \mathcal{L}}{\partial f^S} = 2g^\top(g(f^S) - f^T)$（含映射层 $g$ 的链式法则），正比于特征差异，方向指向 teacher 特征。
- 特征维度对梯度的影响：高维特征空间中 MSE 的绝对值较大（各维度的误差累加），需要通过 loss 权重系数 $\beta$ 来平衡与 CE loss 的相对大小。

**文献：**
- 特征蒸馏（MSE on intermediate representations）首见于 Romero et al. (2015) "FitNets: Hints for Thin Deep Nets"。
- 后续演进包括 Attention Transfer (Zagoruyko & Komodakis, 2017)、PKT (Passalis & Tefas, 2018) 等，将蒸馏目标从特征值扩展到注意力图、概率分布等形式。

**原理分析：**
- FitNets 的设计动机是：Hinton 蒸馏只利用了 teacher 的最终输出层信息，但 teacher 的中间层特征同样包含丰富的知识（如纹理、形状等层次化表示）。通过在中间层对齐 student 和 teacher 的特征，可以更有效地 "引导" student 网络学习 teacher 的表示方式。
- 映射层 $g$ 的必要性：student 和 teacher 的中间层维度通常不同（student 更窄），直接 MSE 不可行。$g$ 作为一个可学习的线性映射（或浅层网络）负责维度对齐，训练完成后可丢弃。
- MSE 蒸馏与 KL 蒸馏的互补性：KL 蒸馏约束 "输出行为一致"，MSE 蒸馏约束 "内部表示一致"。实践中两者经常组合使用：$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{KL}} + \gamma \mathcal{L}_{\text{feat}}$，分别提供标签监督、输出级蒸馏和特征级蒸馏。

---

## 4. 搜广推特有 Loss

> 相关阅读：[[ctr_calibration]]、[[attention_transformer]]、[[mmoe_multitask]]

### 4.1 BPR (Bayesian Personalized Ranking)

**公式：**

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D_S} \log \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \lambda \|\Theta\|^2$$

- $(u, i, j)$：用户 $u$、正样本 $i$（交互过）、负样本 $j$（未交互）
- $\hat{x}_{ui}$：模型对 $(u,i)$ 的预测分
- $\sigma$：sigmoid 函数

**直觉：** 最大化用户对正样本评分高于负样本评分的后验概率。

**BPR vs Pointwise BCE：**

| 维度 | Pointwise (BCE) | Pairwise (BPR) |
|------|-----------------|----------------|
| 输入 | 单个 (user, item, label) | 三元组 (user, pos, neg) |
| 优化目标 | 预测绝对评分 | 预测相对偏序 |
| 校准性 | 好（直接输出概率） | 差（只学排序） |
| 排序性能 | 一般 | 通常更好 |

> **面试一句话：** BPR 通过 sigmoid(正-负) 最大化 pairwise 排序正确概率，排序效果好但牺牲概率校准。

**值域与梯度：**
- 值域：$(0, +\infty)$。由 $-\log\sigma(\cdot)$ 的性质：当正样本分远高于负样本分时 loss 趋向 0+（但不等于 0）；当负样本分高于正样本分时 loss 无上界。
- 梯度：$\frac{\partial \mathcal{L}}{\partial (\hat{x}_{ui}-\hat{x}_{uj})} = -\sigma(\hat{x}_{uj}-\hat{x}_{ui}) = -(1-\sigma(\hat{x}_{ui}-\hat{x}_{uj}))$。当正样本分 >> 负样本分时，梯度 $\to 0$（类似 easy triplet，不再学习）；当正负分接近或反转时，梯度 $\to -1$（最大修正力度）。
- 与 hinge loss 的关键区别：BPR 的 $-\log\sigma$ 是 smooth 的，处处可导，梯度连续变化；hinge loss 在 margin 处梯度从 0 跳变到常数。BPR 的 smooth 特性使训练更稳定。

**文献：**
- 首见于 Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback"（UAI 2009）。
- 该论文从贝叶斯视角推导了 pairwise 排序的最优准则，用 MAP 估计得到 BPR-OPT 目标函数，是隐式反馈推荐系统的里程碑工作。

**原理分析：**
- BPR 解决的核心问题是隐式反馈的歧义性：用户点击了 item A 但没点 item B，只能推断 A 优于 B（pairwise 偏序），不能推断 A 的绝对评分。Pointwise 方法（如 BCE）强行将未交互 item 标为负样本（label=0），但 "未交互" ≠ "不喜欢"——可能只是没看到。BPR 只建模偏序关系，回避了绝对评分的假设。
- BPR 的 $-\log\sigma$ 形式不是随意选择，而是从 posterior $P(\Theta|>_u) \propto P(>_u|\Theta)P(\Theta)$ 的 MAP 推导自然得出的——其中 $P(i >_u j) = \sigma(\hat{x}_{ui}-\hat{x}_{uj})$。正则项 $\lambda\|\Theta\|^2$ 对应 $P(\Theta)$ 的高斯先验。
- 负采样策略是 BPR 的实践核心：均匀随机负采样效率低（大部分负样本太容易），需要结合流行度加权采样或 dynamic negative sampling 来提升训练效率。

---

### 4.2 Pairwise Hinge Loss

**公式：**

$$\mathcal{L} = \max(0, m - (s_i - s_j))$$

与 BPR 的区别：BPR 用 smooth 的 $-\log\sigma$，hinge loss 用 hard 的 max 截断。Hinge 梯度要么为 0 要么为常数，BPR 梯度是平滑的。实践中 BPR 通常更稳定。

**值域与梯度：**
- 值域：$[0, +\infty)$。当 $s_i - s_j \ge m$ 时 loss=0（满足 margin），否则 loss = $m-(s_i-s_j)$。
- 梯度：loss>0 区域梯度为常数 $\pm 1$；loss=0 区域梯度为 0。在 $s_i - s_j = m$ 处不可导（与 triplet loss 相同的 hinge 特性）。
- 与 BPR 的梯度对比：BPR 的梯度随分差连续变化（sigmoid），对接近 margin 的样本给出更精细的调节；hinge loss 的梯度是 "全有或全无"，一旦满足 margin 就完全停止学习。

**文献：**
- Pairwise hinge loss 在排序学习中的使用可追溯至 Herbrich et al. (2000) "Large Margin Rank Boundaries for Ordinal Regression" 和 RankSVM (Joachims, 2002)。
- 在搜索排序（Learning to Rank）中由 RankSVM 推广，是早期 LTR 方法的代表。

**原理分析：**
- Hinge loss 的设计哲学是 "enough is enough"：一旦正样本分数超过负样本分数 $m$ 以上，就不再优化。这在 SVM 中是最大化间隔的最优策略，但在深度学习排序中可能过于保守——模型可能满足于 "刚好超过 margin" 而不追求更大的分差。
- BPR 的 $-\log\sigma$ 没有这个 "停止学习" 的问题（loss 永远 >0），但代价是即使已经排序很好的 pair 也会持续贡献梯度，可能导致过拟合。实践中 BPR 更常用于推荐系统，而 hinge loss 更常用于搜索排序（文档对标注更可靠，不需要持续调整已正确的排序）。

---

### 4.3 Listwise Loss

#### ListNet

**公式（Top-1 Probability）：**

$$P(y_i) = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$

$$\mathcal{L}_{\text{ListNet}} = -\sum_i P^{\text{true}}(y_i) \log P^{\text{pred}}(y_i)$$

将排序问题转化为 top-1 概率分布的 CE loss。

#### LambdaRank

核心思想：**直接对不可导的排序指标（NDCG）做梯度近似**。

$$\lambda_{ij} = \frac{\partial \mathcal{L}}{\partial s_i} = -\sigma(s_j - s_i) \cdot |\Delta \text{NDCG}_{ij}|$$

$|\Delta \text{NDCG}_{ij}|$ 是交换文档 $i$ 和 $j$ 位置后 NDCG 的变化量。位置敏感的梯度使模型直接优化排序质量。

**Pointwise vs Pairwise vs Listwise 对比：**

| 方法 | 代表 | 输入粒度 | 优化目标 | 排序效果 |
|------|------|---------|---------|---------|
| Pointwise | BCE/MSE | 单条 | 绝对分值/标签 | 一般 |
| Pairwise | BPR/RankSVM | 文档对 | 偏序关系 | 较好 |
| Listwise | LambdaRank/ListNet | 整个列表 | NDCG/MAP | 最好 |

> **面试一句话：** LambdaRank 的核心在于用 ΔNDCG 加权 pairwise 梯度，实现对不可导排序指标的直接优化。

**ListNet 值域与梯度：**
- 值域：$[0, +\infty)$，是两个概率分布之间的 CE，理论下界为 $H(P^{\text{true}})$（真实分布自身的熵）。
- 梯度结构与标准 softmax CE 一致，对 $s_i$ 的梯度为 $P^{\text{pred}}(y_i) - P^{\text{true}}(y_i)$，将预测分布推向真实分布。
- Top-1 probability 的近似使得只有列表头部的文档获得显著梯度，尾部文档梯度接近 0——这与搜索排序中 "头部位置最重要" 的直觉一致。

**ListNet 文献：**
- 首见于 Cao et al. (2007) "Learning to Rank: From Pairwise Approach to Listwise Approach"（ICML 2007）。
- 这是 listwise LTR 方法的开创性工作，首次提出将排列概率分布作为排序 loss 的优化目标。

**ListNet 原理分析：**
- ListNet 的核心创新是将排序问题转化为概率分布匹配问题。完整的排列概率（Plackett-Luce 模型）计算 $n!$ 个排列的概率，计算量不可接受，因此用 top-1 概率近似——只关注 "哪个文档最可能排第一"。这个近似在头部位置准确但尾部位置信息丢失。
- 相比 pairwise 方法（BPR/RankSVM），ListNet 的优势是全局优化：pairwise 只看文档对的局部偏序，可能出现 A>B>C 但 C>A 的循环排序；ListNet 对整个列表建模概率分布，天然避免了偏序不一致。

**LambdaRank 值域与梯度：**
- LambdaRank 不定义显式的 loss 函数，只定义梯度（$\lambda$ 梯度）：$\lambda_{ij} = -\sigma(s_j-s_i) \cdot |\Delta\text{NDCG}_{ij}|$。
- $|\Delta\text{NDCG}_{ij}|$ 使得位置敏感：交换头部文档（位置 1↔2）的 $\Delta$NDCG 远大于尾部文档（位置 99↔100），梯度自动聚焦在排序头部。
- 梯度幅值受 $\sigma(s_j-s_i)$ 控制：当分差已经很大时（排序已正确），sigmoid 项趋向 0，梯度自动衰减。

**LambdaRank 文献：**
- 首见于 Burges et al. (2006) "Learning to Rank with Nonsmooth Cost Functions"（NIPS 2006，Microsoft Research）。
- 后续 LambdaMART (Burges, 2010) 将 LambdaRank 与 GBDT 结合，长期统治搜索排序竞赛（多次获得 Yahoo Learning to Rank Challenge 冠军），至今仍是工业搜索排序的重要基线。

**LambdaRank 原理分析：**
- LambdaRank 解决了 LTR 中的核心矛盾：我们想优化 NDCG/MAP 等排序指标，但这些指标依赖排序位置（整数），对分数不可导（位置关于分数是阶跃函数）。传统 surrogate loss（如 pairwise CE）可导但与 NDCG 的相关性弱。
- LambdaRank 的做法是 "跳过 loss 直接定义梯度"：将 pairwise 梯度乘以 $|\Delta\text{NDCG}|$ 权重，使得对 NDCG 影响大的文档对获得更大的梯度。虽然没有显式 loss（不满足 $\lambda = -\nabla\mathcal{L}$ 对某个 $\mathcal{L}$），但后来 Donmez et al. (2009) 证明这些 $\lambda$ 梯度近似于一个隐式 loss 的梯度，训练收敛有理论保障。
- LambdaMART 之所以在搜索排序中长盛不衰，是因为 GBDT + lambda 梯度的组合恰好适配搜索场景的特点：特征多为离散/稀疏的手工特征（query-document 匹配信号），GBDT 天然擅长处理；而 NDCG-aware 的梯度直接优化业务指标，无需 surrogate loss 的间接转换。

---

## 5. 面试高频问答

### 5.1 每个 Loss 的一句话总结

| Loss | 面试一句话 |
|------|-----------|
| CE | 梯度 = $\hat{y} - y$，天然正比误差 |
| BCE | 用 `with_logits` 版本避免数值爆炸 |
| Focal | $(1-p_t)^\gamma$ 自动降权易分样本，γ=2 是默认 |
| Label Smoothing | 等价于 CE + KL(uniform \|\| p) 正则 |
| MSE | 对 outlier 敏感，梯度正比误差 |
| Huber | MSE 和 MAE 的分段过渡 |
| Triplet | 效果取决于负样本挖掘，semi-hard 最优 |
| InfoNCE | 互信息下界，负样本越多越紧 |
| ArcFace | 角度 margin 几何间距均匀 |
| KL 蒸馏 | 前向 KL mode-covering，$T^2$ 补偿梯度 |
| BPR | sigmoid(正-负) 最大化 pairwise 排序概率 |
| LambdaRank | ΔNDCG 加权 pairwise 梯度 |

---

### 5.2 常见坑点

**坑点 1：BCE 数值不稳定**
```python
# 错误：先 sigmoid 再 log，精度丢失
loss = -y * torch.log(torch.sigmoid(logit))

# 正确：直接用 logit
loss = F.binary_cross_entropy_with_logits(logit, y)
```

**坑点 2：Focal Loss 的 α 方向搞反**
- α=0.25 是给**正样本**的权重（因为正样本少）
- 不是给负样本的！很多实现搞反了

**坑点 3：KL Divergence 的 P/Q 顺序**
- `torch.nn.KLDivLoss` 的输入顺序是 `KLDivLoss(input=log_Q, target=P)`
- 即第一个参数是**模型输出的 log 概率**，第二个是**目标分布**
- 搞反会导致 loss 为负

**坑点 4：蒸馏温度 T 忘乘 $T^2$**
- 温度 $T$ 使 softmax 更平滑，但梯度会缩小 $1/T^2$ 倍
- 不乘 $T^2$ → 蒸馏 loss 权重过小 → student 几乎没学到 teacher 的知识

**坑点 5：Triplet Loss 收敛假象**
- Loss 快速降到 0 **不代表**模型学好了
- 可能只是所有 triplet 都变成 easy → 梯度消失
- 解决：监控有效 triplet 比例，持续做 hard mining

**坑点 6：InfoNCE batch 内负采样偏差**
- Batch 内负采样 = 按热度采样 → 热门 item 被过多当作负样本
- 需要纠偏：减去 $\log p(item)$（流行度校正）
- 参考 YouTube DNN 的采样纠偏策略

---

## 6. CE vs Focal vs Label Smoothing 决策表

| 场景 | CE | Focal | Label Smoothing | 推荐 |
|------|:--:|:-----:|:---------------:|------|
| 类别均衡、标准分类 | ✅ | ❌ 无必要 | 可选 | **CE** |
| 类别不平衡（1:10+） | ❌ 效果差 | ✅ | ❌ 不解决不平衡 | **Focal** |
| 需要概率校准（CTR） | ✅ | ❌ 破坏校准 | ✅ 提升校准 | **CE + Label Smoothing** |
| 大模型防过拟合 | ❌ 易过拟合 | ❌ | ✅ 正则效果 | **Label Smoothing** |
| 目标检测（FCOS 等） | ❌ | ✅ | ❌ | **Focal** |
| 推荐 CTR + 长尾 | ❌ | ✅ + 采样 | 可叠加 | **Focal + 采样** |

**组合使用：** Label Smoothing 和 Focal 可以叠加。先 smooth 标签，再加 focal 调制：

$$\mathcal{L} = -\alpha_t (1-p_t)^\gamma \sum y_i^{\text{smooth}} \log \hat{y}_i$$

但实践中需要仔细调参，不建议作为默认方案。

---

## 参考链接

- [[attention_transformer]] — Transformer 中的 CE + Label Smoothing 实践
- [[contrastive_learning]] — 对比学习 loss 深入展开
- [[mmoe_multitask]] — 多任务场景下的多 loss 加权问题
- [[ctr_calibration]] — CTR 校准与 loss 选择的关系
- [[embedding_ann]] — 度量学习 loss 训练出的 embedding 如何用于 ANN 检索
