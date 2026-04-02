# DeepFM：深度因子分解机（Deep Factorization Machine）

> 来源：IJCAI 2017, 华为 | 年份：2017 | 领域：ads/02_rank（CTR预估）

## 问题定义

广告 CTR 预估的核心挑战是**特征交叉**。用户-广告交互依赖复杂的跨特征组合（如"女性+25岁+购物App"更容易点击"美妆广告"），手动构造交叉特征耗时且不完备。

**现有方法局限**：
- **FM**：自动学习二阶交叉，但无法建模高阶交互
- **DNN**：能学习高阶交互，但忽略低阶显式交叉（需要足够深度才能"隐式"覆盖二阶交叉）
- **Wide&Deep**：Wide 侧需要领域专家手动设计交叉特征，工程维护成本高

**DeepFM 目标**：统一 FM 的显式低阶交叉和 DNN 的隐式高阶交叉，无需手工特征工程，端到端学习。

## 模型结构图

```
┌─────────────────────────────────────────────────┐
│                  Output: σ(y)                    │
│                      ↑                           │
│                  y = y_FM + y_DNN                │
│              ┌───────┴────────┐                  │
│              ↑                ↑                   │
│     ┌────────┴────────┐  ┌───┴──────────────┐   │
│     │   FM Component   │  │  Deep Component   │  │
│     │                  │  │                   │  │
│     │  y_FM = <w,x>    │  │  a⁰ = [e₁;e₂;…;eₘ]│ │
│     │  + Σᵢ<ⱼ<vᵢ,vⱼ> │  │  a¹ = σ(W¹a⁰+b¹) │  │
│     │    xᵢxⱼ         │  │  a² = σ(W²a¹+b²) │  │
│     │                  │  │  y_DNN = Wᴸaᴸ+bᴸ │  │
│     └────────┬────────┘  └───┬──────────────┘   │
│              ↑                ↑                   │
│              └────────┬───────┘                   │
│                       ↑                           │
│            ┌──────────┴──────────┐               │
│            │  Shared Embedding   │               │
│            │  V ∈ ℝⁿˣᵏ          │               │
│            └──────────┬──────────┘               │
│                       ↑                           │
│            ┌──────────┴──────────┐               │
│            │  Sparse Input (x)   │               │
│            │  [user, item, ctx]  │               │
│            └─────────────────────┘               │
└─────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：FM 组件（显式二阶交叉）

$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

**解释：**
- $w \in \mathbb{R}^n$：一阶权重向量
- $x \in \mathbb{R}^n$：输入特征向量（稀疏高维）
- $v_i \in \mathbb{R}^k$：第 $i$ 个特征的隐向量（embedding）
- $\langle v_i, v_j \rangle$：内积，表示特征 $i$ 和 $j$ 的二阶交互强度
- 复杂度优化：$O(nk)$ 而非 $O(n^2k)$

### 公式2：FM 二阶项计算优化

$$\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^{k} \left[ \left( \sum_{i=1}^{n} v_{i,f} x_i \right)^2 - \sum_{i=1}^{n} v_{i,f}^2 x_i^2 \right]$$

**解释：**
- $f$：隐向量的第 $f$ 维
- 利用 $(a+b)^2 - a^2 - b^2 = 2ab$ 的推广，将 $O(n^2k)$ 降到 $O(nk)$

### 公式3：Deep 组件（隐式高阶交叉）

$$a^{(0)} = [e_1, e_2, \ldots, e_m]$$
$$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)}), \quad l = 0, 1, \ldots, L-1$$
$$y_{DNN} = W^{(L)} a^{(L)} + b^{(L)}$$

**解释：**
- $e_i$：第 $i$ 个 field 的 embedding（与 FM 共享）
- $m$：field 数量
- $\sigma$：ReLU 激活函数
- $L$：DNN 层数（通常 2-3 层）

### 公式4：最终输出

$$\hat{y} = \sigma(y_{FM} + y_{DNN})$$

**解释：**
- $\sigma$：sigmoid 函数，输出 CTR 概率
- FM 和 DNN 的输出直接相加（不是 concat），简单高效

### 核心创新：共享 Embedding

FM 组件和 Deep 组件共享同一套 Embedding 矩阵 $V$：
- FM 梯度和 DNN 梯度同时更新 embedding → 低阶和高阶信号联合优化
- 无需预训练，端到端联合训练
- **vs Wide&Deep**：Wide 侧需要手工交叉特征，DeepFM 完全自动化

## 与基线方法对比

| 方法 | 核心区别 | 优势 | 劣势 |
|------|---------|------|------|
| **LR** | 线性模型 | 简单、可解释 | 无特征交叉能力 |
| **FM** | 二阶交叉 | 自动交叉、参数高效 | 无法建模高阶交互 |
| **FNN** | FM预训练+DNN | 能学高阶 | FM和DNN分开训练，非端到端 |
| **Wide&Deep** | Wide侧手工交叉+DNN | Google验证有效 | Wide侧需手工特征工程 |
| **DeepFM** | FM+DNN共享Embedding | 全自动、端到端 | 仅FM做显式交叉（二阶） |
| **DCN** | Cross Network替代FM | 显式高阶交叉 | Cross Network表达能力受限 |
| **xDeepFM** | CIN显式高阶交叉 | 显式任意阶交叉 | CIN计算复杂度高 |

## 实验结论

- **Criteo 数据集**：AUC 0.8007 vs Wide&Deep 0.8000 vs FM 0.7940
- **华为 App Store**：AUC 提升 0.25%，转化率提升 3.1%（上线效果）
- 训练速度比 Wide&Deep 快（无手工交叉特征处理开销）
- 消融实验：去掉 FM 组件（-0.3% AUC），去掉 Deep 组件（-0.5% AUC），证明两部分互补

## 工程落地要点

1. **Field 划分**：每个语义独立的特征维度作为一个 field（用户ID、广告ID、类目等），每个 field 内的 embedding 维度相同
2. **Embedding 维度**：通常 4-16 维，广告/用户 ID 可更大（16-64），总参数量主要由 embedding 决定
3. **Dense 特征处理**：连续特征需离散化（等频/等宽分桶）后作为 field，或直接拼接到 DNN 输入绕过 FM
4. **正则化策略**：DNN 层用 Dropout（0.3-0.5），Embedding 用 L2 正则防止过拟合
5. **工业变体演进**：xDeepFM（CIN 显式高阶交叉）→ DCN/DCNv2（Cross Network）→ AutoInt（Multi-head Attention 交叉）→ FiBiNET（SENet + Bilinear）

## 面试考点

**Q1：DeepFM 相比 Wide&Deep 的核心优势？**
> FM 侧不需要手工构造交叉特征，且 FM 和 DNN 共享 Embedding，参数更高效，梯度信号更充分。Wide&Deep 的 Wide 侧需要领域知识设计交叉特征，工程维护成本高。

**Q2：FM 中为什么用内积 $\langle v_i, v_j \rangle$ 而不是直接学习 $w_{ij}$？**
> 直接学 $w_{ij}$ 参数量 $O(n^2)$，稀疏数据下大多数 $(i,j)$ 对出现次数极少无法学习。用 latent vector 内积把参数量降到 $O(nk)$，且通过共现泛化到未见过的特征对。

**Q3：DeepFM 中 FM 和 DNN 各学到什么？**
> FM 学习**显式二阶特征交叉**（如性别×品类）；DNN 通过深层非线性变换学习**隐式高阶特征交叉**（多特征的复杂组合模式）。两者互补：FM 精确捕获二阶交叉，DNN 捕获高阶非线性模式。

**Q4：FM 二阶项的计算复杂度优化原理？**
> 利用 $\sum_{i<j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2}[(\sum_i v_i x_i)^2 - \sum_i (v_i x_i)^2]$，将 $O(n^2k)$ 降到 $O(nk)$。本质是利用完全平方展开消除交叉项的双重循环。

**Q5：为什么 DeepFM 选择 FM+DNN 而不是 FM+CNN？**
> CTR 预估的输入是无序的特征集合（不像图像有空间结构），CNN 的局部感受野和平移不变性不适用。DNN 的全连接结构能建模任意特征间的高阶交互。

**Q6：共享 Embedding 的潜在问题是什么？**
> FM 倾向学习二阶交互最优的 embedding，DNN 倾向学习高阶非线性最优的 embedding，两者的最优解可能不同。实际中通过联合训练找到折中解，效果通常优于独立训练，但理论上存在优化冲突。

**Q7：DeepFM 在大规模场景（亿级特征）下的部署挑战？**
> ① Embedding 表占总参数 99%+，需分布式 Parameter Server 存储 ② 稀疏特征查表是主要延迟来源，需要高效的 embedding lookup ③ 在线推理需控制在 10ms 内，DNN 层数不能太深（通常 2-3 层）④ 增量训练 vs 全量训练的策略选择。
