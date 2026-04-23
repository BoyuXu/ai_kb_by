# 深度模型中衡量特征重要度的方法 — 面试向深度总结

> 标签：#feature-importance #interpretability #SHAP #gradient #attention #搜广推 #面试

---

## 总对比表

| 方法 | 类别 | 模型无关？ | 局部/全局 | 计算开销 | 理论保证 | 适用场景 |
|------|------|-----------|----------|---------|---------|---------|
| Gradient × Input | 梯度类 | ❌ | 局部 | 低（一次反向传播） | 弱（饱和区失效） | 快速 debug |
| Integrated Gradients | 梯度类 | ❌ | 局部 | 中（N 次前向+反向） | 强（公理化） | 严格归因 |
| SmoothGrad | 梯度类 | ❌ | 局部 | 中（采样 N 次） | 弱 | 降噪可视化 |
| GradCAM | 梯度类 | ❌ | 局部 | 低 | 弱 | CNN 特征图可视化 |
| Permutation Importance | 扰动类 | ✅ | 全局 | 高（每特征一次推理） | 中（依赖分布） | 特征筛选 |
| LIME | 扰动类 | ✅ | 局部 | 中 | 弱（线性近似） | 样本级解释 |
| KernelSHAP | 扰动类 | ✅ | 局部+全局 | 高 | 强（Shapley 公理） | 金标准归因 |
| DeepSHAP | 扰动类 | ❌ | 局部+全局 | 中 | 强 | 深度模型归因 |
| TreeSHAP | 扰动类 | ❌（树） | 局部+全局 | 低（多项式） | 强 | 树模型归因 |
| Attention Weights | 注意力 | ❌ | 局部 | 低（已有） | 弱（争议大） | 快速探索 |
| Feature Gating | 内置 | ❌ | 全局 | 低（训练副产物） | 中 | 端到端特征选择 |
| L1 Sparse | 内置 | ❌ | 全局 | 低 | 中 | 特征淘汰 |
| AutoFIS | 内置 | ❌ | 全局 | 高（搜索） | 中 | 自动特征交叉选择 |

---

## 方法选择决策树

```
Q1: 需要理论严格性？
├── Yes → SHAP（KernelSHAP / DeepSHAP）
└── No →
    Q2: 解释单个样本 or 全局特征排序？
    ├── 单样本 →
    │   Q3: 模型是否可微？
    │   ├── Yes → Integrated Gradients（首选）/ Gradient × Input（快速）
    │   └── No → LIME
    └── 全局 →
        Q4: 模型类型？
        ├── 树模型 → TreeSHAP / Gini importance
        ├── 深度模型（有 gate）→ gate 权重 + Permutation Importance 交叉验证
        └── 深度模型（无 gate）→ Permutation Importance → SHAP 验证
```

---

## 1. 梯度类方法 (Gradient-based Methods)

核心思想：利用模型对输入特征的梯度来衡量该特征对输出的影响程度。梯度大 → 微小扰动引起输出大变化 → 该特征"重要"。

### 1.1 Gradient × Input

**公式：**

$$\text{Attribution}_i = x_i \cdot \frac{\partial f(x)}{\partial x_i}$$

其中 $f(x)$ 是模型输出（logit 或概率），$x_i$ 是第 $i$ 个特征值。

**直觉：** 纯梯度 $\frac{\partial f}{\partial x_i}$ 表示"特征变一点，输出变多少"，但忽略了特征本身的大小。乘以 $x_i$ 后变成"这个特征当前取值对输出的贡献"。

**优点：**
- 计算极快，一次反向传播搞定
- 实现简单，PyTorch 几行代码

**缺点：**
- **饱和区问题**：ReLU 激活为 0 的区域梯度为 0，但该特征可能仍然重要（只是在当前取值下模型不敏感）
- **不满足 Sensitivity 公理**：两个输入只差一个特征但输出不同，该方法可能给出 0 归因
- 梯度噪声大，结果不稳定

**代码示例：**

```python
x.requires_grad_(True)
output = model(x)
output.backward()
attribution = x * x.grad  # Gradient × Input
```

### 1.2 Integrated Gradients (IG)

> Sundararajan, Taly & Yan, ICML 2017

**动机：** Gradient × Input 的核心缺陷是只看"当前点的梯度"，而忽略了从 baseline 到当前输入的整条路径。IG 通过沿路径积分梯度来修复这个问题。

**公式：**

$$\text{IG}_i(x) = (x_i - x_i') \cdot \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

其中 $x'$ 是 baseline（通常取零向量或训练集均值），$\alpha \in [0, 1]$ 是插值参数。

**实际计算（Riemann 近似）：**

$$\text{IG}_i(x) \approx (x_i - x_i') \cdot \frac{1}{N} \sum_{k=1}^{N} \frac{\partial f\left(x' + \frac{k}{N}(x - x')\right)}{\partial x_i}$$

取 $N = 50 \sim 300$ 步即可。

**公理化保证（核心卖点）：**

1. **Sensitivity（敏感性）**：若 $x$ 和 $x'$ 只在第 $i$ 个特征上不同且输出不同，则 $\text{IG}_i \neq 0$
2. **Implementation Invariance（实现不变性）**：两个功能等价的模型给出相同的归因
3. **Completeness（完备性）**：$\sum_i \text{IG}_i(x) = f(x) - f(x')$，所有特征归因之和恰好等于输出差

**Baseline 选择：**
- 图像：全黑图片
- NLP：padding token 的 embedding
- 推荐系统：全零特征 or 全局均值特征
- Baseline 选择会显著影响结果，是该方法的主要超参

**优点：**
- 唯一满足 Sensitivity + Implementation Invariance 的方法（论文定理证明）
- Completeness 保证归因可加性

**缺点：**
- 计算量 = $N$ 倍前向+反向传播
- Baseline 选择无统一标准
- 对离散特征（如 categorical）不够自然

### 1.3 SmoothGrad

> Smilkov et al., 2017

**动机：** 原始梯度噪声太大（loss landscape 局部震荡），通过多次采样取平均来"平滑"梯度。

**公式：**

$$\hat{M}(x) = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial f(x + \epsilon_k)}{\partial x}, \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2)$$

通常 $\sigma = 0.1 \sim 0.2$ 倍输入范围，$N = 50$。

**本质：** 用高斯噪声对输入做数据增强，然后对梯度取期望。

**优点：** 视觉上更平滑、更易解释
**缺点：** 没有理论保证（只是工程 trick），计算量 $N$ 倍

### 1.4 GradCAM

> Selvaraju et al., ICCV 2017

**公式：**

$$L_{\text{GradCAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

其中 $A^k$ 是第 $k$ 个 feature map，$\alpha_k^c$ 是该 feature map 对类别 $c$ 的重要度权重（梯度全局平均池化）。

**原理推广到非 CV 场景：**
- GradCAM 的核心思想是"对中间层表示求梯度来定位重要区域"
- 推荐系统中可以类比：对 attention 层或 feature interaction 层求梯度，定位哪些特征交叉最重要
- 但实际推荐系统中更常用 IG 或 SHAP

---

## 2. 扰动类方法 (Perturbation-based Methods)

核心思想：通过改变（遮蔽/打乱/替换）某个特征的值，观察模型输出的变化来衡量该特征的重要度。与梯度类方法相比，扰动类方法不依赖模型内部结构，是真正的黑盒方法。

### 2.1 Permutation Importance

> Breiman, 2001（随机森林原始论文就提出了）；Fisher et al. 2019 推广

**算法：**

```
对每个特征 j:
    1. 在验证集上计算原始指标 score_orig (e.g., AUC)
    2. 随机打乱(shuffle)第 j 列的值（破坏该特征与标签的关系）
    3. 计算打乱后的指标 score_perm
    4. importance_j = score_orig - score_perm
```

**公式：**

$$\text{PI}_j = \mathbb{E}[\text{Score}(f, X, y)] - \mathbb{E}[\text{Score}(f, X_{\backslash j}, y)]$$

其中 $X_{\backslash j}$ 表示第 $j$ 列被随机打乱后的数据集。

**优点：**
- 模型无关，适用于任何模型
- 直觉清晰：打乱后性能掉多少 = 这个特征多重要
- 评估的是"去掉该特征信息后的真实影响"

**缺点：**
- **特征相关性陷阱**：若特征 A 和 B 高度相关，打乱 A 后 B 能补偿，导致 A 的 importance 被低估
- 计算量 = 特征数 × 推理成本
- 打乱可能产生 out-of-distribution 样本（例如把身高 180cm 配上体重 40kg）

**改进：** Conditional Permutation Importance — 条件于其他特征的分布来打乱，但计算更昂贵

### 2.2 LIME (Local Interpretable Model-agnostic Explanations)

> Ribeiro et al., KDD 2016

**核心思想：** 在待解释样本 $x$ 的邻域内，用一个可解释的简单模型（线性回归）来近似复杂模型的行为。

**算法：**

$$\xi(x) = \arg\min_{g \in \mathcal{G}} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

其中：
- $f$ 是原始复杂模型
- $g$ 是局部可解释模型（通常是线性模型）
- $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ 是以 $x$ 为中心的权重核，越近权重越大
- $\Omega(g)$ 是复杂度惩罚（如特征个数限制）

**具体步骤：**

1. 在 $x$ 周围采样 $N$ 个扰动样本 $\{z_1, ..., z_N\}$
2. 对每个 $z_i$ 用原始模型预测 $f(z_i)$
3. 用距离核 $\pi_x(z_i)$ 作为样本权重
4. 拟合加权线性回归 $g(z) = w^T z + b$
5. 线性模型的系数 $w_j$ 即为特征 $j$ 的局部重要度

**优点：**
- 模型无关
- 结果直观：每个特征有一个权重，正负表示方向

**缺点：**
- 结果依赖采样策略和核宽度 $\sigma$（不稳定）
- 线性近似在非线性强的区域可能严重失真
- 不满足 Shapley 公理

### 2.3 SHAP (SHapley Additive exPlanations)

> Lundberg & Lee, NeurIPS 2017

**理论基础 — Shapley Value：**

Shapley 值源自合作博弈论，回答"每个玩家对团队总收益的公平贡献是多少"。

$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[f_x(S \cup \{i\}) - f_x(S)\right]$$

其中 $F$ 是全部特征集合，$S$ 是不包含特征 $i$ 的子集，$f_x(S)$ 是只使用 $S$ 中特征时的模型预测期望。

**三大公理（SHAP 的理论优势）：**

1. **Efficiency（有效性）**：$\sum_i \phi_i = f(x) - \mathbb{E}[f(x)]$，归因之和等于预测值与基线的差
2. **Symmetry（对称性）**：若两个特征在所有子集中贡献相同，则归因相同
3. **Linearity（线性性）**：两个模型的组合，归因等于各自归因的线性组合

**KernelSHAP：**

将 Shapley 值计算转化为加权线性回归问题，采样特征子集来近似。

$$\pi(z') = \frac{|F|-1}{\binom{|F|}{|z'|} \cdot |z'| \cdot (|F|-|z'|)}$$

其中 $z' \in \{0, 1\}^{|F|}$ 表示特征子集掩码，$\pi(z')$ 是 SHAP kernel 权重。

- 优点：模型无关
- 缺点：计算量大，$2^{|F|}$ 子集理论上需要全枚举（实际采样近似）

**DeepSHAP：**

结合 DeepLIFT 的反向传播规则和 Shapley 值，在深度网络中高效近似 SHAP 值。

- 利用网络层级结构，逐层传播贡献
- 比 KernelSHAP 快很多（一次反向传播级别）
- 近似误差存在，但实际效果好

**TreeSHAP：**

专为树模型设计，利用树的结构在 $O(TLD^2)$ 时间内精确计算 Shapley 值（$T$ 树数，$L$ 叶子数，$D$ 深度）。

- 精确解，不是近似
- 速度极快，可用于大规模特征分析
- XGBoost / LightGBM 内置支持

**SHAP vs Permutation Importance 核心区别：**

| 维度 | SHAP | Permutation Importance |
|------|------|----------------------|
| 粒度 | 每个样本一个归因向量 | 全局一个分数 |
| 理论 | Shapley 公理 | 无严格公理 |
| 交互效应 | 公平分配交互贡献 | 被相关特征稀释 |
| 计算 | 高（指数级子集） | 中（每特征一次推理） |
| 输出 | 方向 + 大小 | 只有大小 |

---

## 3. 注意力权重 (Attention Weights)

### 3.1 Direct Attention as Importance

在推荐系统中，attention 机制天然产生权重：

**DIN (Deep Interest Network)：**

$$\text{Attention}(e_i, e_{\text{target}}) = \text{softmax}(W \cdot [e_i; e_{\text{target}}; e_i \odot e_{\text{target}}])$$

每个历史行为 $e_i$ 相对于目标 item $e_{\text{target}}$ 有一个注意力权重，可以直接看做"这个历史行为对当前预测有多重要"。

**Transformer Self-Attention：**

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

注意力矩阵 $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 的第 $i$ 行第 $j$ 列 $A_{ij}$ 表示 token $i$ 对 token $j$ 的"关注程度"。

### 3.2 Attention Rollout

多层 Transformer 中，直接看某一层的 attention 只是局部视角。Attention Rollout 将多层 attention 矩阵相乘，得到从输入到输出的"累计注意力"：

$$\hat{A} = \prod_{l=1}^{L} A^{(l)}$$

实际中还会加入残差连接的影响：$\tilde{A}^{(l)} = 0.5 \cdot A^{(l)} + 0.5 \cdot I$（因为残差让每个 token "关注自己"）。

### 3.3 Attention ≠ Explanation（关键争议）

> Jain & Wallace, NAACL 2019: "Attention is not Explanation"

**核心发现：**

1. **Alternative Attention：** 存在完全不同的 attention 分布，但产生几乎相同的输出。如果 attention 是"解释"，那不同的 attention 不应产生相同结果。
2. **Attention vs Gradient：** Attention 权重和 gradient-based attribution 的相关性很低（Kendall τ < 0.5）。
3. **Uniform Attention：** 在一些任务上，均匀分布的 attention 也能达到接近的性能。

> Wiegreffe & Pinter, EMNLP 2019: "Attention is not not Explanation"

**反驳：**
- 找到 alternative attention 不等于原始 attention 没有意义
- Attention 至少提供了模型的一种"合理解读"
- 在某些受限条件下，attention 确实与输入重要度正相关

**工业界共识：**

- Attention 可以作为**快速探索工具**（看看模型大概在关注什么）
- **不能作为严格的特征重要度度量**
- 如果需要严格归因，应使用 IG 或 SHAP
- 推荐系统中 DIN 的 attention 可解释性相对好（因为架构简单、有明确的 target）

---

## 4. 内置方法 (Built-in Feature Selection)

核心思想：在模型架构或训练过程中直接嵌入特征选择机制，让模型自己学习哪些特征重要。

### 4.1 Gate Mechanisms（特征门控）

**原理：** 为每个特征（或特征交叉）学习一个 gate 值 $g_i \in [0, 1]$，训练完成后 $g_i$ 接近 0 的特征可以被安全淘汰。

**DeepFM + Feature Gate：**

$$\hat{e}_i = g_i \cdot e_i, \quad g_i = \sigma(w_g^T e_i + b_g)$$

其中 $e_i$ 是特征 $i$ 的 embedding，$g_i$ 是学出来的门控值。

**FiBiNET（Feature Importance and Bilinear feature Interaction Network）：**

$$g_i = \text{SENet}(e_1, e_2, ..., e_n)_i = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))_i$$

其中 $z = \text{MeanPool}([e_1, ..., e_n])$，本质是用 Squeeze-and-Excitation 网络来为每个特征域学习重要度权重。

**优点：**
- 端到端训练，无需额外计算
- Gate 值本身就是特征重要度的直接度量

**缺点：**
- Gate 值受训练过程影响（初始化、学习率），可能不完全反映"真实"重要度
- 需要修改模型结构

### 4.2 Sparse Regularization（稀疏正则化）

**L1 on Embedding：**

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{i=1}^{n} \|e_i\|_1$$

L1 正则化会将不重要特征的 embedding 向量整体压缩到接近零。训练完成后 $\|e_i\|_1 < \epsilon$ 的特征可以被淘汰。

**Group Lasso（组稀疏）：**

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{i=1}^{n} \|e_i\|_2$$

Group Lasso 使用 L2 范数作为组级惩罚，效果是将整个 embedding 向量一起稀疏化（要么全保留，要么全为零），比逐元素 L1 更适合特征选择。

### 4.3 AutoFIS (Automatic Feature Interaction Selection)

> Liu et al., KDD 2020

**动机：** FM / DeepFM 中所有二阶特征交叉 $\binom{n}{2}$ 个都参与计算，但很多交叉是噪声。AutoFIS 自动选择有用的特征交叉。

**方法：**

1. **搜索阶段（Search）：** 为每个特征交叉 $(i, j)$ 引入架构参数 $\alpha_{ij}$：

$$\hat{y} = \sum_{i<j} \alpha_{ij} \cdot \langle e_i, e_j \rangle + \text{DNN}(x)$$

使用 GRDA（Generalized Regularized Dual Averaging）优化器使 $\alpha_{ij}$ 稀疏化。

2. **重训阶段（Retrain）：** 移除 $\alpha_{ij} = 0$ 的交叉，用剩余交叉重新训练。

**实际效果：** 通常可以移除 50%~70% 的特征交叉，模型效果不降甚至略升。

---

## 5. 树模型对比 (Tree Model Baselines)

树模型的特征重要度是搜广推面试的常见对比项，理解树模型有助于解释"为什么深度模型的特征重要度更难"。

### 5.1 三种经典方法

**Gini Importance（基于不纯度降低）：**

$$\text{Importance}_j = \sum_{t \in \text{nodes using } j} p_t \cdot \Delta \text{Gini}_t$$

每个使用特征 $j$ 做分裂的节点贡献一份不纯度下降量。

**Split Count（分裂次数）：**

$$\text{Importance}_j = \text{特征 } j \text{ 在所有树中被选为分裂特征的次数}$$

最简单但最粗糙。

**Gain（分裂增益总和）：**

$$\text{Importance}_j = \sum_{t \in \text{nodes using } j} \text{Gain}_t$$

XGBoost 默认使用的方法。

### 5.2 为什么深度模型更难

| 维度 | 树模型 | 深度模型 |
|------|-------|---------|
| 特征使用方式 | 显式分裂，一个节点一个特征 | 隐式混合，所有特征同时参与 |
| 非线性来源 | 分段常数（分裂） | 连续非线性（激活函数） |
| 特征交互 | 沿路径的隐式交互 | Embedding + 各种 interaction layer |
| 可解读性 | 每棵树可视化 | 参数空间高维，无法直接解读 |
| 重要度计算 | 精确（结构直接给出） | 需要额外方法（梯度/扰动/SHAP） |

**核心原因：** 树模型中每个特征的"使用"是离散的、可追踪的；深度模型中特征通过 embedding → 线性变换 → 非线性激活 → 交叉层层叠加，单个特征的贡献被"稀释"在整个参数空间中。

---

## 6. 工业实践 (Industrial Practice)

### 6.1 推荐系统中的特征重要度分析

**离线分析流程：**

```
Step 1: 训练完成后，用 Permutation Importance 做全局排序
Step 2: 对 top-K 和 bottom-K 特征用 SHAP 做细粒度分析
Step 3: 对候选淘汰特征做 ablation study（移除重训）
Step 4: 比较 ablation 前后的离线指标（AUC、GAUC、Logloss）
```

**在线验证流程：**

```
Step 1: 离线 ablation 指标持平或微降 < 0.1%
Step 2: 上线 A/B test（去掉该特征的模型 vs 基线）
Step 3: 观察 7 天核心指标（CTR、转化率、GMV）
Step 4: 指标无统计显著下降 → 确认可淘汰
```

### 6.2 特征淘汰的标准流程

```
候选特征池
    │
    ▼
Permutation Importance < 阈值
    │
    ▼
SHAP 分析确认 → 排除"被相关特征掩盖"的情况
    │
    ▼
离线 Ablation Study
    │
    ▼
AUC 下降 < 0.05% ?
    ├── Yes → 在线 A/B Test
    │         │
    │         ▼
    │    核心指标无显著下降？
    │    ├── Yes → 淘汰特征 ✅
    │    └── No → 保留 ❌
    └── No → 保留 ❌
```

### 6.3 常见坑与经验

**坑 1: 特征相关性导致 importance 不稳定**

若特征 A 和 B 的 Pearson 相关系数 > 0.9，Permutation Importance 可能将两者都评为"不重要"（互相补偿）。

**解法：** 先做特征聚类（按相关矩阵），每个 cluster 内选代表特征分析；或使用 SHAP（对交互效应有公平分配）。

**坑 2: 离线 importance ≠ 在线价值**

某些特征离线 importance 很低，但在在线 serving 中对特定人群（如新用户冷启动）至关重要。

**解法：** 分群分析 importance（新用户 vs 老用户 vs 高活 vs 低活），不能只看全局均值。

**坑 3: 训练时特征重要 ≠ 预测时特征重要**

特征可能在训练中帮助模型收敛（正则化效果），但在预测时贡献可忽略。

**解法：** Importance 分析必须在验证集/测试集上做，不在训练集上做。

**坑 4: Embedding 维度影响 importance**

高维 embedding 的特征天然有更大的参数容量，Gradient / SHAP 归因值倾向偏高。

**解法：** 归一化处理，或者对每个特征域独立分析。

**坑 5: 时间维度的特征漂移**

特征重要度会随时间变化（用户行为漂移、业务策略变更）。一次分析的结论不能长期使用。

**解法：** 定期（如月度）重新分析特征重要度，建立监控看板。

---

## 7. 面试 Q&A

### Q1: 深度模型怎么看哪个特征最重要？

**答：** 主要三类方法：

1. **梯度类**：计算模型输出对输入特征的梯度。推荐 Integrated Gradients，有公理化保证。
2. **扰动类**：打乱/遮蔽特征后观察性能变化。全局用 Permutation Importance，样本级用 SHAP。
3. **内置类**：在模型中加入 gate 机制（如 FiBiNET 的 SENet）或 L1 正则化，让模型自己学。

工业中通常的流程是：Permutation Importance 做初筛 → SHAP 做细粒度分析 → Ablation Study 最终验证。

---

### Q2: SHAP 和 Permutation Importance 有什么区别？

**答：**

- **粒度不同**：SHAP 为每个样本的每个特征算一个归因值；PI 为每个特征算一个全局分数。
- **理论基础不同**：SHAP 基于 Shapley 值，满足效率性、对称性、线性性三大公理；PI 没有严格的公理保证。
- **处理特征交互**：SHAP 通过遍历所有特征子集，公平地将交互效应分配给各特征；PI 在特征高度相关时会互相掩盖（A 和 B 相关，打乱 A 后 B 补偿，A 显得不重要）。
- **计算复杂度**：KernelSHAP 指数级（采样近似）；PI 线性于特征数。

---

### Q3: Attention 权重能代表特征重要性吗？

**答：** 不能作为严格度量，但可以作为快速探索工具。

Jain & Wallace (2019) 发现：存在完全不同的 attention 分布但产生相同输出，说明 attention 并非输出的唯一决定因素。

但在特定场景下（如 DIN 的 target attention），attention 确实反映了模型对不同历史行为的关注程度。关键区分：
- **"模型在关注什么"** vs **"这个特征对输出有多大贡献"** 是两个不同的问题
- Attention 回答前者，特征重要度回答后者

---

### Q4: Integrated Gradients 为什么需要 baseline？怎么选？

**答：** IG 衡量的是"从 baseline 到当前输入，每个特征的贡献有多少"。没有 baseline，就没有"从哪里开始"的参考点。

公式 $\text{IG}_i(x) = (x_i - x_i') \int_0^1 \frac{\partial f(x' + \alpha(x-x'))}{\partial x_i} d\alpha$ 中，$x'$ 就是 baseline。

选择原则：baseline 应代表"无信息状态"：
- 图像：全黑/全灰
- 文本：padding embedding
- 推荐特征：全零 or 全局均值
- 关键是 $f(x')$ 应接近模型的默认输出（如平均 CTR）

---

### Q5: 工业界怎么做特征淘汰？

**答：** 四步流程：

1. **粗筛**：Permutation Importance + Gate 权重，找出候选淘汰特征（importance < 阈值）
2. **精分析**：SHAP 确认不是"被相关特征掩盖"，分群分析确认不是"特定人群重要"
3. **离线验证**：去掉特征重训模型，AUC 下降 < 0.05% 视为安全
4. **在线验证**：A/B test 7 天，核心指标无统计显著下降 → 淘汰

注意事项：分群分析（新老用户、不同品类）、定期重做（特征漂移）、不在训练集上做分析。

---

### Q6: 梯度类方法有什么共同缺陷？

**答：**

1. **要求模型可微**：对树模型、规则模型不适用
2. **饱和区问题**：ReLU 的死区、sigmoid 的两端，梯度为 0 不代表特征不重要
3. **只看局部线性近似**：深度模型高度非线性，一阶梯度只是局部信息
4. **离散特征不友好**：categorical 特征的 embedding 梯度物理意义不直观
5. **不考虑特征缺失下的行为**：只看"微小扰动"，不看"完全去掉"

---

### Q7: DeepSHAP 和 KernelSHAP 的区别？

**答：**

- **KernelSHAP**：模型无关，通过采样特征子集 + 加权线性回归近似 Shapley 值。适用于任何模型，但计算慢。
- **DeepSHAP**：利用深度网络的层级结构，结合 DeepLIFT 的反向传播规则逐层传播贡献。速度快很多（一次反向传播级别），但只适用于深度模型，且存在近似误差。

选择建议：
- 深度模型 + 需要大量样本级归因 → DeepSHAP
- 任意模型 + 只需少量样本解释 → KernelSHAP
- 树模型 → TreeSHAP（精确且快速）

---

### Q8: 在推荐系统中，哪种特征重要度方法最实用？

**答：** 推荐系统的实践中，最常用的组合是：

1. **全局筛选**：Permutation Importance（简单高效，一次跑完所有特征排序）
2. **深度分析**：SHAP（对 top/bottom 特征做样本级分析，理解 why）
3. **端到端选择**：Feature Gate / AutoFIS（如果愿意改模型架构）
4. **快速 debug**：Attention 权重（DIN / Transformer 模型看看关注哪些行为）

工业界不会只依赖一种方法，通常是"多方法交叉验证 + 最终以 ablation 和在线 A/B 为准"。

---

### Q9: 为什么说树模型的特征重要度比深度模型更可靠？

**答：** 树模型的特征使用方式是**显式的**：每个节点选一个特征做分裂，分裂次数和增益可以精确统计。深度模型的特征通过 embedding → 线性变换 → 非线性激活 → 多层叠加，所有特征同时参与每一层的计算，单个特征的贡献被"弥散"在参数空间中。

但树模型也有坑：Gini importance 对高基数特征（如 user_id）有偏，会高估其重要度。所以即使是树模型，也推荐用 Permutation Importance 或 TreeSHAP 来代替默认的 Gini importance。

---

### Q10: 特征重要度分析结果不稳定怎么办？

**答：** 不稳定的常见原因和解法：

| 原因 | 解法 |
|------|------|
| 特征高度相关 | 先做特征聚类，每个 cluster 选代表 |
| 采样随机性（LIME/SHAP） | 增加采样次数，多次运行取均值和置信区间 |
| 模型本身不稳定 | 训练多个模型（不同 seed），对 importance 取交集 |
| 数据量不足 | 增大验证集，或用 bootstrap 估计方差 |
| Permutation 产生 OOD 样本 | 用 Conditional Permutation 或 SHAP 替代 |

**实用建议：** 永远不要只看一种方法的结果。至少用两种方法交叉验证，且最终以 ablation study 为准。

---

## 参考文献

- Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
- Ribeiro et al., "Why Should I Trust You? Explaining the Predictions of Any Classifier", KDD 2016
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
- Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017
- Jain & Wallace, "Attention is not Explanation", NAACL 2019
- Wiegreffe & Pinter, "Attention is not not Explanation", EMNLP 2019
- Liu et al., "AutoFIS: Automatic Feature Interaction Selection in Factorization Models", KDD 2020
- Huang et al., "FiBiNET: Combining Feature Importance and Bilinear Feature Interaction", RecSys 2019
- Zhou et al., "Deep Interest Network for Click-Through Rate Prediction", KDD 2018
- Fisher et al., "All Models are Wrong, but Many are Useful", JMLR 2019
