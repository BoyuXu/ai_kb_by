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

---

### 4.2 Pairwise Hinge Loss

**公式：**

$$\mathcal{L} = \max(0, m - (s_i - s_j))$$

与 BPR 的区别：BPR 用 smooth 的 $-\log\sigma$，hinge loss 用 hard 的 max 截断。Hinge 梯度要么为 0 要么为常数，BPR 梯度是平滑的。实践中 BPR 通常更稳定。

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
