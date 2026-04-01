# Uplift 建模：从因果效应到营销增益

> 标签：#Uplift #因果推断 #ITE #CATE #元学习器 #Qini #广告定向 #营销优化

---

## 1. Uplift 的本质：反事实推断

### 1.1 基本问题设定

**核心悖论**：同一个用户不能同时处于"接受广告"和"不接受广告"两种状态，我们永远无法同时观测到两个潜在结果。

**潜在结果框架（Potential Outcomes / Rubin Causal Model）**：

$$
\tau_i = Y_i(1) - Y_i(0)
$$

- $Y_i(1)$：用户 $i$ 接受广告（treatment=1）后的结果（如是否转化）
- $Y_i(0)$：用户 $i$ 不接受广告（treatment=0）后的结果
- $\tau_i$：个体处理效应（Individual Treatment Effect, ITE）
- 由于"基本因果推断问题"，$\tau_i$ 对单个用户永远不可直接观测

### 1.2 三种因果效应估计

**ATE（平均处理效应）**：

$$
\text{ATE} = \tau = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
$$

在随机对照实验（RCT）中，ATE 可以直接通过处理组均值 - 对照组均值估计。

**ATT（处理组的平均处理效应）**：

$$
\text{ATT} = \mathbb{E}[Y(1) - Y(0) | T = 1]
$$

只关心实际接收广告的用户中，广告的平均效果。

**CATE（条件平均处理效应）**：

$$
\text{CATE} = \tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]
$$

在具有特征 $x$ 的用户群体中，广告的平均效果。**Uplift 建模的核心目标就是估计 CATE**。

### 1.3 随机化假设

CATE 可估计需要以下假设：

1. **SUTVA**（稳定单位处理值假设）：用户 A 是否接受广告不影响用户 B 的结果
2. **无混淆性**（Unconfoundedness）：$\{Y(0), Y(1)\} \perp T | X$，即给定特征 $X$ 后，处理分配独立于潜在结果（随机实验天然满足）
3. **重叠性**（Overlap）：$0 < P(T=1|X) < 1$，即每个特征值的用户都有机会出现在处理组和对照组

---

## 2. 三大元学习器

### 2.1 S-Learner（Single Learner，单模型）

**思路**：将 treatment $T$ 作为一个普通特征，训练单一预测模型：

$$
\hat{\mu}(x, t) = \mathbb{E}[Y | X=x, T=t]
$$

**Uplift 估计**：

$$
\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)
$$

**训练过程**：

```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class SLearner:
    def __init__(self, base_learner=None):
        self.model = base_learner or GradientBoostingClassifier()
    
    def fit(self, X, T, Y):
        # 将 T 拼接到 X 中作为特征
        X_with_t = np.column_stack([X, T])
        self.model.fit(X_with_t, Y)
    
    def predict_uplift(self, X):
        n = len(X)
        # 预测处理组结果
        X_treat = np.column_stack([X, np.ones(n)])
        y_treat = self.model.predict_proba(X_treat)[:, 1]
        
        # 预测对照组结果
        X_ctrl = np.column_stack([X, np.zeros(n)])
        y_ctrl = self.model.predict_proba(X_ctrl)[:, 1]
        
        return y_treat - y_ctrl  # CATE 估计
```

**优点**：实现简单，利用所有数据。

**缺点**：Treatment 信号可能被其他特征淹没（稀释），导致估计偏向 0。树类模型尤其明显，因为 T 只有 0/1 两个取值。

### 2.2 T-Learner（Two Learner，双模型）

**思路**：分别在处理组和对照组数据上训练两个独立模型：

$$
\hat{\mu}_1(x) = \mathbb{E}[Y | X=x, T=1]
$$

$$
\hat{\mu}_0(x) = \mathbb{E}[Y | X=x, T=0]
$$

**Uplift 估计**：

$$
\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

```python
class TLearner:
    def __init__(self, learner_t=None, learner_c=None):
        self.model_t = learner_t or GradientBoostingClassifier()
        self.model_c = learner_c or GradientBoostingClassifier()
    
    def fit(self, X, T, Y):
        # 分处理组/对照组训练
        self.model_t.fit(X[T == 1], Y[T == 1])
        self.model_c.fit(X[T == 0], Y[T == 0])
    
    def predict_uplift(self, X):
        y_treat = self.model_t.predict_proba(X)[:, 1]
        y_ctrl = self.model_c.predict_proba(X)[:, 1]
        return y_treat - y_ctrl
```

**优点**：Treatment 信号直接体现在数据分割，不会被稀释。

**缺点**：两个模型的估计误差会叠加（variance 增大）；当处理组/对照组样本量极不均衡时，少数组的模型精度差。

### 2.3 X-Learner（Cross Learner，交叉学习器）

**思路**：通过交叉预测来估计个体处理效应，解决 T-Learner 在样本不均衡时的问题。

**第一阶段**：训练 T-Learner，得到 $\hat{\mu}_0$、$\hat{\mu}_1$

**第二阶段**：构造伪处理效应标签（pseudo treatment effect）：

对处理组用户（$T=1$）：

$$
\tilde{\tau}_i^1 = Y_i - \hat{\mu}_0(X_i)
$$

即：实际观测结果 - 如果没有处理的预期结果

对对照组用户（$T=0$）：

$$
\tilde{\tau}_i^0 = \hat{\mu}_1(X_i) - Y_i
$$

即：如果有处理的预期结果 - 实际观测结果

**第三阶段**：分别在处理组和对照组上训练 Uplift 预测模型：

$$
\hat{\tau}_1(x) = \mathbb{E}[\tilde{\tau}^1 | X=x] \quad \text{（在处理组上训练）}
$$

$$
\hat{\tau}_0(x) = \mathbb{E}[\tilde{\tau}^0 | X=x] \quad \text{（在对照组上训练）}
$$

**第四阶段**：加权融合：

$$
\hat{\tau}(x) = g(x) \hat{\tau}_0(x) + (1 - g(x)) \hat{\tau}_1(x)
$$

- $g(x) = P(T=1|X=x)$：倾向得分（propensity score）
- 用处理组比例大的地方更信任 $\hat{\tau}_1$，对照组比例大的地方更信任 $\hat{\tau}_0$

**X-Learner 的优势**：当处理组样本量远大于对照组时（常见广告场景：处理组 90%，对照组 10%），$\hat{\tau}_1$ 能利用更多处理组数据，减少对照组稀少导致的 $\hat{\tau}_0$ 估计偏差。

---

## 3. Uplift 树和随机森林

### 3.1 节点分裂准则

普通决策树的分裂准则：最小化 Gini 不纯度或最大化信息增益。

**Uplift 树的分裂准则**：最大化处理组和对照组之间的结果分布差异（即最大化 Uplift）。

**KL 散度分裂准则**：

$$
\text{Gain}}_{\text{{KL}}(T) = D_{KL}(P^T || P^C)
$$

其中：
- $P^T$：处理组的结果分布 $P(Y|T=1)$
- $P^C$：对照组的结果分布 $P(Y|T=0)$
- $D_{KL}(P||Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)}$

**Euclidean Distance 分裂**（二元结果）：

$$
\text{Gain}}_{\text{{ED}} = 2 \left[p_t \log \frac{p_t}{p_c} + (1-p_t) \log \frac{1-p_t}{1-p_c}\right]
$$

其中 $p_t = P(Y=1|T=1)$，$p_c = P(Y=1|T=0)$。

### 3.2 与普通树的区别

| 组件 | 普通决策树 | Uplift 树 |
|------|-----------|----------|
| 分裂准则 | 最大化 purity（Y 的纯度）| 最大化处理效应异质性 |
| 叶节点值 | Y 的均值/众数 | $\bar{Y}_T - \bar{Y}_C$（Uplift 估计）|
| 训练目标 | 预测 Y | 预测 $\tau(x)$ |

### 3.3 Uplift 随机森林

```python
from causalml.inference.tree import UpliftRandomForestClassifier

uplift_rf = UpliftRandomForestClassifier(
    n_estimators=100,
    evaluationFunction='KL',  # KL 散度分裂
    max_features='sqrt',
    random_state=42
)
uplift_rf.fit(X_train, treatment=T_train, y=Y_train)
uplift_pred = uplift_rf.predict(X_test)
```

---

## 4. 工业应用：广告定向优化

### 4.1 四象限用户分类

这是 Uplift 建模最直观的应用框架：

```
                  有广告时
                 转化  不转化
无广告时  转化  |  ①  |  ②  |
          不转化|  ③  |  ④  |
```

- ①：**Sure Things（必然转化）**：无论有无广告都会转化，广告浪费投入
- ②：**Sleeping Dogs（沉睡狗）**：无广告时转化，有广告时反而不转化（广告打扰或过度推销）
- ③：**Persuadables（说服型）**：无广告不转化，有广告才转化 → **目标用户！**
- ④：**Lost Causes（永久流失）**：无论如何都不转化，广告无效

**广告定向策略**：
- 只对 Persuadables 投放广告
- 避免对 Sleeping Dogs 投放（可能产生负效果）
- Sure Things 和 Lost Causes 可以节省预算

### 4.2 预算分配优化

给定预算 $B$，目标是最大化转化增量：

$$
\max \sum_i \hat{\tau}_i \cdot z_i \quad \text{s.t.} \sum_i c_i z_i \leq B,\ z_i \in \{0, 1\}
$$

**贪心解**：按 $\hat{\tau}_i / c_i$（单位成本 Uplift）排序，从高到低选取直到预算耗尽。

```python
def optimize_targeting(uplift_scores, costs, budget):
    """
    按单位成本 Uplift 排序，贪心分配预算
    """
    roi = uplift_scores / costs  # 每元成本带来的增量转化
    sorted_indices = np.argsort(roi)[::-1]
    
    selected = []
    remaining_budget = budget
    
    for idx in sorted_indices:
        if costs[idx] <= remaining_budget:
            selected.append(idx)
            remaining_budget -= costs[idx]
    
    return selected
```

### 4.3 常见陷阱：选择偏差（Selection Bias）

在非随机数据中（如历史广告投放数据），已有投放策略导致处理组和对照组系统性不同（如高价值用户更可能被投放广告），直接用 T-Learner 会产生有偏估计。

**解决方案**：倾向得分加权（IPW）：

$$
\hat{\tau}_{IPW} = \mathbb{E}\left[\frac{TY}{e(X)} - \frac{(1-T)Y}{1-e(X)}\right]
$$

其中 $e(X) = P(T=1|X)$ 是倾向得分。IPW 通过给少数组样本更高权重来修正选择偏差。

---

## 5. 评估指标

### 5.1 Uplift 曲线和 AUUC

**Uplift 曲线**：将用户按预测 Uplift 从高到低排序，计算累积增量转化（处理组转化 - 对照组转化）。

```python
def uplift_curve(y_true, treatment, uplift_pred):
    """计算 Uplift 曲线"""
    df = pd.DataFrame({
        'y': y_true, 'T': treatment, 'uplift': uplift_pred
    }).sort_values('uplift', ascending=False)
    
    n = len(df)
    cumulative_treat = (df['T'] * df['y']).cumsum() / (df['T'].cumsum() + 1e-10)
    cumulative_ctrl = ((1-df['T']) * df['y']).cumsum() / ((1-df['T']).cumsum() + 1e-10)
    
    return cumulative_treat - cumulative_ctrl  # Uplift 曲线
```

**AUUC（Area Under Uplift Curve）**：Uplift 曲线下面积，类比 ROC 曲线的 AUC。值越高，模型对 Persuadables 的识别能力越强。

### 5.2 Qini 曲线和 Qini 系数

**Qini 曲线（Radcliffe, 2007）**：横轴为按 Uplift 排序后的人群比例（0到1），纵轴为累积增量转化人数：

$$
Q(t) = \left(\frac{n_{t,1}}{n_1} - \frac{n_{t,0}}{n_0}\right) \cdot (n_1 + n_0) \cdot t
$$

- $n_{t,1}$：前 $t$ 比例人群中处理组的转化人数
- $n_{t,0}$：前 $t$ 比例人群中对照组的转化人数

**Qini 系数**：Qini 曲线与随机基线（对角线）之间的面积之比：

$$
\text{Qini} = \frac{\text{Qini 曲线下面积} - \text{随机基线面积}}{\text{理想曲线面积} - \text{随机基线面积}}
$$

### 5.3 与 ROC AUC 的类比

| 概念 | 分类模型 | Uplift 模型 |
|------|---------|-----------|
| 预测目标 | P(Y=1) | τ(x) = E[Y(1)-Y(0)\|X=x] |
| 排序曲线 | ROC（TPR vs FPR）| Uplift 曲线 / Qini 曲线 |
| 面积指标 | AUC | AUUC / Qini 系数 |
| 随机基线 | 对角线（AUC=0.5）| 对角线（AUUC=0）|
| 完美模型 | AUC=1.0 | 完美区分四象限 |

---

## 6. 面试考点

### Q1：为什么 S-Learner 的 Uplift 估计会偏向 0？

S-Learner 将 T 作为普通特征，与 X 中的其他特征（如年龄、性别等）共同决定预测。若 X 维度远大于 1（T 的维度），模型在分裂/训练时可能认为 T 不如其他特征重要，导致 T 的影响被压缩。尤其是正则化较强的树模型（如 Random Forest 的 max_features）可能根本不选择 T 特征分裂，得到的 Uplift 估计几乎为 0。

### Q2：Uplift 建模和 CTR 预估有什么本质区别？

CTR 预估估计的是 P(转化 | 用户特征, 广告特征)，包含了"不投广告也会转化"的部分（即 Sure Things 的贡献）。Uplift 建模估计的是"因广告带来的增量转化"，排除了自然转化。对 Sure Things 用 CTR 选广告会浪费预算；用 Uplift 选广告只对边际用户（Persuadables）投放，ROI 更高。

### Q3：如何在没有随机实验的情况下估计 Uplift？

需要用观测数据（历史数据）进行因果推断，常用方法：(1) 倾向得分匹配（PSM）：为每个处理组用户找倾向得分接近的对照组用户配对；(2) 逆概率加权（IPW）：用倾向得分加权修正选择偏差；(3) 双重机器学习（DML）：同时控制混淆变量和高维特征；(4) 工具变量（IV）：找到只影响处理（广告投放）而不直接影响结果（转化）的变量。关键前提：满足无混淆性假设，即所有影响处理分配的变量都被观测到。

### Q4：倾向得分（Propensity Score）是什么，X-Learner 为什么用它加权？

倾向得分 $e(x) = P(T=1|X=x)$ 是在给定特征 $x$ 下，用户被分配到处理组的概率。X-Learner 中，当处理组样本多（$e(x)$ 大）时，应更信任基于处理组估计的 $\hat{\tau}_1(x)$；当对照组样本多（$e(x)$ 小）时，应更信任 $\hat{\tau}_0(x)$。加权 $g(x) = e(x)$ 实现了这一平衡，等价于根据数据丰富度自动分配模型权重。

### Q5：Qini 系数为负意味着什么？

Qini 系数为负意味着模型的排序比随机排序还差，即将"Sleeping Dogs"和"Lost Causes"排在了高 Uplift 位置，将真正的"Persuadables"排在了低位。这往往是因为：(1) 训练数据中 Sure Things 的正标签占多数，模型混淆了"高转化率"和"高增量转化率"；(2) 特征工程不当，没有包含能区分因果效应的特征。

### Q6：在广告系统中，A/B 测试和 Uplift 建模如何配合？

A/B 测试是 Uplift 建模的数据基础：随机实验保证了无混淆性假设，使 CATE 估计是无偏的。流程：(1) 随机实验收集数据（处理组 / 对照组）；(2) 训练 Uplift 模型，估计每个用户的 $\hat{\tau}(x)$；(3) 按 $\hat{\tau}$ 分层，验证不同 Uplift 分层中的实际增量转化与预测一致；(4) 上线时，对 $\hat{\tau}$ 高的用户投放广告，对低 Uplift 用户节省预算。

### Q7：如何处理 Uplift 建模中的假负样本问题？

假负样本（False Negative）：本应是"Persuadable"的用户在历史数据中没有被投放广告（被错误地放入对照组），观测结果 $Y(0)=0$，但真实的 $Y(1)=1$。这导致 CATE 被低估。解决方案：(1) 尽量使用随机实验数据而非历史数据；(2) 如果必须用历史数据，用倾向得分加权减少系统性偏差；(3) 设置保留样本（holdout set）纯随机分配，定期更新 Uplift 模型。

---

## 参考资料

- Rubin. "Estimating causal effects of treatments in randomized and nonrandomized studies" (Potential Outcomes, 1974)
- Radcliffe & Surry. "Real-World Uplift Modelling with Significance-Based Uplift Trees" (Qini, 2011)
- Künzel et al. "Metalearners for estimating heterogeneous treatment effects using machine learning" (X-Learner, 2019)
- Athey & Imbens. "Recursive partitioning for heterogeneous causal effects" (Causal Trees, 2016)
- Wager & Athey. "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (Causal Forest, 2018)
- Chen et al. "Causalml: Python package for causal machine learning" (2020)
