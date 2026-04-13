# 正则化技术对比 — 面试向深度总结

> 标签：#regularization #dropout #batchnorm #layernorm #RMSNorm #mixup #搜广推 #面试

---

## 总对比表

| 技术 | 类别 | 公式/机制 | 适用场景 | 优点 | 缺点 |
|------|------|-----------|----------|------|------|
| L1 正则 | 显式正则 | $\lambda \sum \|w_i\|$ | 特征选择、稀疏模型 | 产生稀疏解，自动特征选择 | 不可导（0点），优化不稳定 |
| L2 正则 | 显式正则 | $\lambda \sum w_i^2$ | 通用防过拟合 | 平滑解，处处可导 | 不产生精确零，无法做特征选择 |
| Elastic Net | 显式正则 | $\lambda_1 \sum \|w_i\| + \lambda_2 \sum w_i^2$ | 高维相关特征 | 兼具 L1 稀疏 + L2 稳定 | 两个超参需调 |
| Dropout | 隐式正则 | $h' = h \cdot m, \; m \sim \text{Bernoulli}(p)$ | FC 层、Transformer | 集成学习效果，简单有效 | 训练变慢，推理需 scale |
| DropConnect | 隐式正则 | 随机丢弃连接权重 | FC 层 | 比 Dropout 更细粒度 | 实现复杂，收益不明显 |
| DropPath | 隐式正则 | 随机丢弃整条路径/层 | ResNet, ViT | 深层网络训练更稳定 | 需设计存活概率调度 |
| BatchNorm | 归一化 | $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ | CNN, 大 batch | 加速训练，允许更大 lr | 依赖 batch，小 batch 不稳定 |
| LayerNorm | 归一化 | $\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$ | Transformer, RNN | 不依赖 batch，变长序列友好 | 无跨样本归一化 |
| GroupNorm | 归一化 | 通道分组后 LN | 小 batch CV | BN 与 LN 折中 | 需选 group 数 |
| RMSNorm | 归一化 | $\hat{x} = \frac{x}{\text{RMS}(x)}$ | LLaMA, Mistral | 比 LN 快（省去均值计算） | 去掉 re-centering 可能损失表达力 |
| Mixup | 数据增强 | $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ | CV 分类，CTR | 提升泛化，平滑决策边界 | 生成样本可能不自然 |
| CutMix | 数据增强 | 空间区域混合 | CV 分类 | 保留局部结构 | 仅适用图像 |
| Label Smoothing | 数据增强 | $y' = (1-\epsilon)y + \epsilon/K$ | 分类任务 | 防止过度自信，改善校准 | 可能损害 hard example 学习 |

---

## 1. 显式正则化

### 1.1 L1 正则（Lasso）

**公式：**

$$L_{\text{reg}} = L_{\text{original}} + \lambda \sum_{i} |w_i|$$

**核心机制：**
- 在损失函数上叠加权重绝对值的惩罚
- 梯度为 $\text{sign}(w_i)$（常数大小），在 $w=0$ 处不可导
- 优化过程中小权重会被直接压到 0 → **稀疏解**
- 等价于在菱形（diamond）约束区域内求最优解

**为什么 L1 产生稀疏解？**（面试高频）
- 几何解释：L1 约束区域是菱形，菱形的顶点在坐标轴上。等高线与菱形相切时，切点大概率落在顶点（即某些 $w_i = 0$）
- 梯度解释：L1 的梯度（subgradient）大小恒定，不随 $w$ 接近 0 而减小，因此能把小权重一路推到 0

**应用场景：**
- 特征选择：稀疏模型自动过滤不重要特征
- 搜广推大规模稀疏模型中的 Embedding 稀疏化
- 模型压缩：将不重要的连接剪枝

### 1.2 L2 正则（Ridge / Weight Decay）

**公式：**

$$L_{\text{reg}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

**核心机制：**
- 梯度为 $2\lambda w_i$，与权重大小成正比
- 大权重受到大惩罚，小权重受到小惩罚 → 权重趋向均匀小值，但**不会精确为 0**
- 约束区域是圆（球），等高线与圆相切一般不在坐标轴上
- 等价于对权重做高斯先验 $w \sim \mathcal{N}(0, \frac{1}{2\lambda})$

**Weight Decay vs L2 正则：**
- 在 SGD 中完全等价：$w \leftarrow w - \eta(\nabla L + 2\lambda w) = (1 - 2\eta\lambda)w - \eta \nabla L$
- **在 Adam 中不等价！** AdamW 提出 decoupled weight decay，直接在更新后减去 $\lambda w$，而非把 L2 梯度混入自适应学习率
- 面试坑：问「L2 正则和 weight decay 有什么区别？」→ 回答须区分优化器

### 1.3 Elastic Net

**公式：**

$$L_{\text{reg}} = L_{\text{original}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

- 当特征高度相关时，L1 会随机选一个特征，丢弃其余 → 不稳定
- Elastic Net 通过 L2 项将相关特征的权重拉向彼此 → **分组效应（grouping effect）**
- 实践中 $\lambda_1, \lambda_2$ 的比例通过交叉验证选择

### 1.4 L1 vs L2 对比表

| 维度 | L1 正则 | L2 正则 |
|------|---------|---------|
| 约束形状 | 菱形（diamond） | 圆形（circle） |
| 稀疏性 | **产生精确零** | 权重趋近零但不等于零 |
| 特征选择 | 自动选择（内嵌） | 不能 |
| 多重共线性 | 随机选一个 | 均匀缩小 |
| 可导性 | 0 点不可导 | 处处可导 |
| 贝叶斯解释 | Laplace 先验 | Gaussian 先验 |
| 优化 | 需要 proximal gradient | 标准梯度下降 |
| 典型名称 | Lasso | Ridge |
| 搜广推应用 | 大规模稀疏特征选择 | 通用 DNN 防过拟合 |

---

## 2. 隐式正则化 — Dropout 家族

### 2.1 Dropout

**公式（训练时）：**

$$h' = h \cdot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

其中 $p$ 是丢弃概率（通常 $p=0.1 \sim 0.5$）。

**推理时：**

$$h' = h \cdot (1-p)$$

或等价地，训练时使用 **inverted dropout**：$h' = \frac{h \cdot m}{1-p}$，推理时不做任何操作。现代框架均用 inverted dropout。

**为什么 Dropout 有效？**
1. **集成学习视角**：每次 forward pass 相当于训练一个子网络，$n$ 个神经元有 $2^n$ 种子网络，推理时等价于所有子网络的几何平均
2. **打破共适应（co-adaptation）**：防止神经元之间形成固定依赖关系，迫使每个神经元独立学到有用特征
3. **噪声注入视角**：等价于给隐层加乘性噪声，是一种数据增强

**面试一句话：** Dropout 本质是训练指数级子网络的集成，通过 inverted scaling 实现推理时的高效近似。

### 2.2 DropConnect

- 丢弃的不是神经元的输出，而是**权重矩阵中的连接**
- $W' = W \cdot M, \quad M_{ij} \sim \text{Bernoulli}(1-p)$
- 比 Dropout 更细粒度（Dropout 是 DropConnect 的特例：丢弃同一行的所有连接）
- 实践中收益有限，使用较少

### 2.3 DropPath / Stochastic Depth

**机制：** 训练时以一定概率跳过整个残差块（residual block）：

$$x_{l+1} = x_l + b_l \cdot f_l(x_l), \quad b_l \sim \text{Bernoulli}(p_l)$$

**存活概率调度（Survival Probability Schedule）：**
- 线性衰减：浅层存活率高，深层存活率低
- $p_l = 1 - \frac{l}{L}(1 - p_L)$，其中 $p_L$ 是最深层存活率（如 0.8）
- 直觉：深层学到的是更细粒度的特征，偶尔跳过影响较小

**应用：**
- ResNet: 原始 Stochastic Depth 论文
- Vision Transformer (ViT): DeiT, Swin Transformer 中广泛使用
- 效果：加速训练 25%+，同时提升泛化

### 2.4 Dropout 在 Transformer 中的位置

```
Input Embedding
    ↓
[Embedding Dropout] ← 通常 p=0.1
    ↓
┌─ Multi-Head Attention ─┐
│  Q, K, V projection    │
│  Attention weights      │
│  [Attention Dropout]    │ ← 对 softmax 后的 attention 权重做 dropout
│  Output projection      │
└─────────────────────────┘
    ↓ + Residual
  LayerNorm
    ↓
┌─ FFN ───────────────────┐
│  Linear → GELU          │
│  [FFN Dropout]          │ ← FFN 输出做 dropout
│  Linear                 │
└─────────────────────────┘
    ↓ + Residual
  LayerNorm
```

参考 [[attention_transformer]] 了解完整 Transformer 结构。

### 2.5 为什么大模型（LLM）推理时通常不用 Dropout？

- GPT-3、LLaMA 等大模型训练时 **dropout rate = 0**
- 原因：
  1. **模型足够大**，参数冗余本身就是隐式正则
  2. **数据足够多**（万亿 token），过拟合风险低
  3. Dropout 降低训练效率（有效参数利用率下降）
  4. 大规模分布式训练中 dropout 的随机性增加调试难度
- 小模型 fine-tuning 时仍然使用 dropout（如 BERT fine-tune 用 0.1）

---

## 3. 归一化技术（CRITICAL）

### 3.1 BatchNorm (BN)

**公式（训练时）：**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

- $\mu_B, \sigma_B^2$ 在 **mini-batch 维度** 上计算
- $\gamma, \beta$ 是可学习参数（affine transformation），恢复表达能力
- 没有 $\gamma, \beta$ 的话，归一化后每层输出都是标准正态，表达力受限

**推理时：**
- 使用训练期间累积的 **running mean / running variance**（指数移动平均）
- $\mu_{\text{run}} \leftarrow (1-\alpha)\mu_{\text{run}} + \alpha \mu_B$
- 推理时行为**确定性**，无随机性

**BN 为什么有效？**
1. ~~Internal Covariate Shift~~ — 原论文的解释已被质疑
2. **平滑损失曲面**（Sanity Check: How Does BN Help Optimization?, 2018）：BN 使损失函数更平滑（Lipschitz 更小），允许更大学习率
3. **隐式正则化**：batch 统计量引入噪声，类似 dropout 效果

### 3.2 LayerNorm (LN)

**公式：**

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

- $\mu_L, \sigma_L^2$ 在 **特征维度（hidden dimension）** 上计算，与 batch 无关
- 每个样本独立归一化 → 训练和推理行为一致

### 3.3 GroupNorm (GN)

- 将 channels 分成 $G$ 组，每组内做 LayerNorm
- 当 $G=1$ 时退化为 LayerNorm；$G=C$（每个 channel 一组）时退化为 InstanceNorm
- 专为 **小 batch** CV 任务设计（如检测、分割，batch size 常为 1-2）

### 3.4 RMSNorm

**公式：**

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i, \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$$

- 相比 LayerNorm，**去掉了均值中心化（mean subtraction）** 和偏置 $\beta$
- 计算量更小（省一次 mean 计算 + 一次减法）
- 论文实验表明：re-centering（减均值）对 LN 的成功**不是关键因素**，re-scaling（除以标准差）才是核心

**应用：** LLaMA、LLaMA 2、Mistral、Gemma 均使用 RMSNorm 替代 LayerNorm。

### 3.5 BatchNorm vs LayerNorm 详细对比（面试必考）

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch 维度（跨样本） | Feature 维度（单样本内） |
| 统计量 | $\mu, \sigma$ 来自 mini-batch | $\mu, \sigma$ 来自单个样本的特征 |
| Batch 依赖 | **强依赖** | **无依赖** |
| 训练/推理 gap | 有（running stats vs batch stats） | 无 |
| 变长序列 | 不同位置的 batch 统计量语义不一致 | 每个 token 独立归一化，无问题 |
| 小 batch | 统计量噪声大，效果差 | 不受影响 |
| 典型用途 | CNN (ResNet, EfficientNet) | Transformer, RNN |
| 正则化效果 | batch 噪声提供隐式正则 | 无隐式正则效果 |
| 分布式训练 | 需要跨 GPU 同步 BN (SyncBN) | 无需同步 |

**为什么 Transformer 用 LN 不用 BN？**

1. **变长序列问题**：一个 batch 中不同样本长度不同，位置 $t$ 处的 batch 统计量来自不同语义的 token，统计量无意义
2. **小 batch 问题**：NLP 训练 batch size 通常较小（受序列长度限制），BN 统计量噪声大
3. **训练/推理一致性**：LN 无 running stats，训练和推理行为完全一致
4. **自回归生成**：推理时逐 token 生成，batch size = 1，BN 完全无法工作

**为什么 CNN 用 BN 不用 LN？**

1. CNN 中同一 channel 的空间特征共享统计特性（如边缘检测 filter 的激活分布一致），跨样本统计是有意义的
2. CV 任务 batch size 通常较大（32-256），BN 统计量稳定
3. BN 的隐式正则效果对 CV 有益
4. 实验上 BN 在 CNN 中始终优于 LN

### 3.6 RMSNorm vs LayerNorm

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| Mean centering | 有 | **无** |
| 可学习偏置 $\beta$ | 有 | **无** |
| 计算量 | $O(2n)$（mean + var） | $O(n)$（仅 RMS） |
| 效果 | 基准 | **持平或微弱下降** |
| 速度 | 基准 | **快 10-15%** |
| 使用模型 | BERT, GPT-2, GPT-3 | LLaMA, Mistral, Gemma |

大模型中 RMSNorm 的节省在 Transformer 数百层叠加下效果显著。

### 3.7 Pre-Norm vs Post-Norm

**Post-Norm（原始 Transformer）：**
$$x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))$$

**Pre-Norm（GPT-2 及后续主流）：**
$$x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$$

- Pre-Norm 训练更稳定（梯度直接通过残差通路传播）
- Post-Norm 理论上表达力更强，但需要 warmup
- 现代 LLM **全部用 Pre-Norm**

---

## 4. 数据增强作为正则化

### 4.1 Mixup

**公式：**

$$\tilde{x} = \lambda x_i + (1-\lambda)x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda)y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

- $\alpha$ 控制混合程度：$\alpha \to 0$ 时几乎不混合，$\alpha = 1$ 时均匀混合
- 本质：在样本之间做**线性插值**，鼓励模型学到线性行为（更简单的决策边界）
- 等价于一种 **vicinal risk minimization**

### 4.2 CutMix

$$\tilde{x} = M \odot x_i + (1-M) \odot x_j$$

- $M$ 是一个二值 mask，定义一个矩形区域
- 矩形区域从 $x_j$ 裁剪并粘贴到 $x_i$ 上
- 标签按面积比例混合：$\tilde{y} = \lambda y_i + (1-\lambda)y_j$，$\lambda$ = 未被裁剪的面积比例
- 优于 Mixup：**保留了局部空间结构**

### 4.3 Label Smoothing

**公式：**

$$y'_k = \begin{cases} 1 - \epsilon + \epsilon/K & \text{if } k = \text{target} \\ \epsilon/K & \text{otherwise} \end{cases}$$

- $\epsilon$ 通常取 0.1
- 防止模型对正确类别过于自信（softmax 输出趋近 one-hot）
- **改善模型校准（calibration）**：预测概率更接近真实概率 → 与 [[ctr_calibration]] 直接相关
- 副作用：可能影响知识蒸馏（teacher 的 dark knowledge 被平滑掉）

---

## 5. 搜广推中的正则化实践

### 5.1 CTR 模型正则化

| 组件 | 正则手段 | 说明 |
|------|----------|------|
| Embedding 层 | L2 正则 | 防止低频特征 embedding 过拟合 |
| DNN 层 | Dropout (p=0.1~0.3) | 全连接层标准配置 |
| 输出层 | Label Smoothing | 改善校准，CTR 预估校准至关重要 |
| 整体 | Weight Decay | AdamW 中 decoupled weight decay |

**Embedding 正则化特殊处理：**
- 高频特征：标准 L2 即可
- 低频/冷启动特征：更大的 L2 系数，或使用 **embedding 共享/hash trick** 减少参数
- 参见 [[ctr_calibration]] 中校准相关讨论

### 5.2 大规模稀疏模型

- 特征维度可达亿级 → L1 正则做**自动特征选择**，筛掉无用特征
- FTRL（Follow-The-Regularized-Leader）：在线学习中结合 L1 正则的工业标准
- FTRL 的 L1 项使得大量特征权重为 0 → **模型可压缩到极致**

### 5.3 多任务学习中的正则化

- 多任务模型（如 [[mmoe_multitask]]）中，共享层的正则化更重要
- 任务特定 tower 可用不同 dropout rate
- 参见 [[lora_peft]] 中 LoRA 微调的正则化效果

### 5.4 推荐系统特有正则

- **Feature Hashing**：隐式正则，hash 冲突迫使不同特征共享参数
- **Early Stopping**：最常用的隐式正则，按验证集 AUC 停止训练
- **Batch Negative Sampling**：召回模型中用 batch 内负样本，样本量增大等价于正则

---

## 6. 面试高频问答

### Q1: L1 和 L2 的区别？为什么 L1 能产生稀疏解？

**一句话：** L1 的菱形约束区域在坐标轴上有角点，等高线大概率在角点相切（某些 $w_i=0$）；L2 的圆形约束没有角点，切点一般不在轴上。

**坑点：** 不要只说「L1 是绝对值，L2 是平方」，面试官要的是几何/梯度层面的解释。

### Q2: Dropout 的原理？训练和推理有什么区别？

**一句话：** 训练时随机丢弃神经元（等价于训练指数级子网络的集成），推理时用 inverted dropout 保持期望不变。

**坑点：** 说清楚是 inverted dropout（训练时 scale $\frac{1}{1-p}$），而非原始 dropout（推理时 scale $1-p$）。

### Q3: BatchNorm 和 LayerNorm 的区别？为什么 Transformer 用 LN？

**一句话：** BN 跨 batch 归一化、LN 跨 feature 归一化；Transformer 用 LN 因为变长序列 + 小 batch + 无 train/test gap。

**坑点：** 要能画出 BN vs LN 在一个 3D tensor (batch, seq, hidden) 上的归一化维度示意。

### Q4: Weight Decay 和 L2 正则一样吗？

**一句话：** SGD 中等价，**Adam 中不等价**；AdamW 提出 decoupled weight decay，直接减权重而非把 L2 梯度混入自适应学习率。

**坑点：** 这是 AdamW 论文的核心贡献，很多人混淆。

### Q5: RMSNorm 和 LayerNorm 的区别？

**一句话：** RMSNorm 去掉了 mean centering，只做 re-scaling，计算更快，LLaMA 系列用的就是 RMSNorm。

**坑点：** 要知道为什么去掉 mean 还能 work — 论文证明 re-scaling 是归一化的核心，re-centering 不是。

### Q6: Label Smoothing 怎么帮助模型？

**一句话：** 防止 softmax 输出趋向 one-hot（logit 趋向正负无穷），改善校准，使预测概率更可靠。

**坑点：** Label Smoothing 对知识蒸馏有负面影响（teacher 的 soft label 信息被压缩了）。

### Q7: 为什么大模型不用 Dropout？

**一句话：** 模型大 + 数据多 = 过拟合风险低，Dropout 反而浪费参数利用效率和训练速度。

### Q8: Pre-Norm 和 Post-Norm 的区别？

**一句话：** Pre-Norm 先归一化再进子层，梯度通过残差直传；Post-Norm 先做子层再归一化，表达力更强但训练不稳定。

### Q9: 搜广推中最重要的正则化手段是什么？

**一句话：** Early Stopping + L2/Weight Decay + Embedding 正则化。大规模稀疏特征场景下 FTRL + L1 是工业标准。

---

## 7. 总结：如何选择正则化技术？

```
场景判断流程：
├── 模型规模？
│   ├── 大模型 (>1B) → 不需要 Dropout，RMSNorm，Label Smoothing 可选
│   └── 中小模型 → Dropout + Weight Decay + Early Stopping
├── 架构？
│   ├── Transformer → LayerNorm / RMSNorm + Attention Dropout
│   ├── CNN → BatchNorm + DropPath（如 ResNet/ViT）
│   └── MLP/FC → Dropout + L2
├── 数据量？
│   ├── 数据充足 → 轻正则即可
│   └── 数据稀缺 → 强正则 (Dropout + Mixup + Label Smoothing)
└── 特征维度？
    ├── 高维稀疏 → L1 / FTRL
    └── 低维稠密 → L2 / Weight Decay
```

**经验法则：** 正则化是 ensemble 式叠加的 — 实践中通常同时使用 Weight Decay + Dropout + 归一化 + Early Stopping，各司其职。

---

> 相关文档：[[attention_transformer]] | [[mmoe_multitask]] | [[lora_peft]] | [[ctr_calibration]]
