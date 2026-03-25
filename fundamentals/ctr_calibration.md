# CTR 校准：从排序性到概率准确性

> 标签：#CTR校准 #Platt_Scaling #等渗回归 #温度缩放 #ECE #oCPC #广告出价 #概率校准

---

## 1. 为什么需要校准

### 1.1 排序性 vs 概率准确性的区别

CTR（Click-Through Rate）预估模型通常用 AUC（Area Under ROC Curve）评估**排序性**：AUC=0.75 意味着对任意一个正样本和负样本，模型以 75% 的概率将正样本排在前面。

然而，**排序性好 ≠ 概率准确**：
- 模型 A 预估 CTR：0.01, 0.02, 0.05（真实 CTR：0.01, 0.02, 0.05）→ **已校准**
- 模型 B 预估 CTR：0.1, 0.2, 0.5（真实 CTR：0.01, 0.02, 0.05）→ **未校准（高估 10×）**

两个模型的 AUC 完全相同（排序一致），但绝对概率差异巨大。

### 1.2 oCPC 为什么依赖准确概率

在 oCPA（目标每次转化成本）/oCPC 广告出价中：

$$\text{bid} = \text{CPA\_target} \times \hat{p}(\text{CVR}) \times \hat{p}(\text{CTR})$$

推导：
- 广告主目标：每次有效转化成本 $\leq$ CPA\_target
- 每次点击费用（CPC）：$\text{CPC} = \text{CPA\_target} \times p(\text{CVR})$
- 每次曝光出价（CPM）：$\text{CPM} = \text{CPC} \times p(\text{CTR}) = \text{CPA\_target} \times p(\text{CVR}) \times p(\text{CTR})$

**校准误差的经济影响**：
- 高估 CTR 10%：出价高 10%，可能赢得更多竞价但 ROI 下降
- 低估 CTR 50%：出价低 50%，竞价失败率剧增，曝光量减半，广告主损失严重
- 系统性偏差（如全体高估）：导致平台 eCPM 虚高，广告主满意度下降，长期会导致预算流失

### 1.3 校准问题的来源

1. **训练/预测分布偏差**：训练数据来自特定时间段，而预测在未来（分布漂移）
2. **选择偏差**：训练数据只包含已展示的广告（Exposure Bias），而非所有候选广告
3. **模型本身偏差**：深度模型的输出往往不是良好校准的概率，倾向于输出极端值（过于自信）
4. **样本不均衡**：CTR 预估正负样本比例约 1:100~1:1000，模型倾向于低估 CTR

---

## 2. 校准度量指标

### 2.1 Log Loss（对数损失/交叉熵）

$$\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)\right]$$

- 对错误的高置信度预测施加更重的惩罚（$\log 0 \to -\infty$）
- 同时反映排序性和校准性
- **缺点**：不直接告诉你模型是高估还是低估了多少

### 2.2 期望校准误差（ECE）

**ECE 的计算步骤**：

1. 将预测概率范围 $[0, 1]$ 分成 $M$ 个等宽区间（bucket）
2. 对每个 bucket $B_m$ 计算：
   - `conf(Bm)`：bucket 内预测概率的均值
   - `acc(Bm)`：bucket 内真实点击率
3. 加权求和：

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

```python
def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    y_true: 真实标签 (0/1)
    y_pred: 预测概率
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() == 0:
            continue
        
        bin_size = mask.sum()
        conf = y_pred[mask].mean()  # 预测均值
        acc = y_true[mask].mean()   # 实际点击率
        
        ece += (bin_size / n) * abs(acc - conf)
    
    return ece
```

### 2.3 可靠性图（Reliability Diagram）

X 轴：预测概率（分桶后的均值），Y 轴：实际频率。完美校准的模型应在对角线上（预测 0.1 = 实际 10% 点击率）。

```python
def plot_reliability_diagram(y_true, y_pred, n_bins=10):
    import matplotlib.pyplot as plt
    
    bin_means, bin_accs, bin_sizes = [], [], []
    bins = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(y_pred[mask].mean())
            bin_accs.append(y_true[mask].mean())
            bin_sizes.append(mask.sum())
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.bar(bin_means, bin_accs, width=0.05, alpha=0.7, label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Actual Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.show()
```

---

## 3. Platt Scaling

### 3.1 原理

Platt Scaling（1999）在模型输出分数 $f(x)$ 上拟合一个 logistic 回归：

$$p = \sigma(A \cdot f(x) + B) = \frac{1}{1 + e^{-(Af(x) + B)}}$$

其中 $A$、$B$ 是在**保留验证集**上通过最大似然估计得到的参数：

$$\hat{A}, \hat{B} = \arg\min_{A,B} -\sum_{i} \left[y_i \log \sigma(Af_i + B) + (1-y_i) \log(1-\sigma(Af_i + B))\right]$$

### 3.2 参数含义

- $A < 0$（通常情况）：缩放分数幅度（压缩过于自信的预测）
- $A > 0$：放大分数幅度（通常不常见）
- $B$：平移偏置（调整整体校准中心）

当模型系统性高估时，$|A| < 1$ 压缩分数，使概率更接近 1/2；当系统性低估时，$|A| > 1$ 放大分数。

### 3.3 实现

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import scipy.optimize as optimize

def platt_scaling(val_scores, val_labels):
    """
    在验证集上拟合 Platt Scaling 参数
    val_scores: 模型输出的原始分数（logit 或概率）
    val_labels: 真实标签 (0/1)
    """
    def neg_log_likelihood(params):
        A, B = params
        p = 1 / (1 + np.exp(-(A * val_scores + B)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.sum(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p))
    
    result = optimize.minimize(neg_log_likelihood, x0=[1.0, 0.0], method='L-BFGS-B')
    A, B = result.x
    
    def calibrated_predict(scores):
        return 1 / (1 + np.exp(-(A * scores + B)))
    
    return calibrated_predict

# 使用
calibrator = platt_scaling(val_scores, val_labels)
calibrated_probs = calibrator(test_scores)
```

### 3.4 适用场景和局限

**适用**：
- SVM 等不输出概率的模型（最初为 SVM 设计）
- 当校准关系接近 sigmoid 形状时
- 验证集较小时（只有 2 个参数，不易过拟合）

**局限**：
- 参数少（只有 A、B），无法处理复杂的非线性校准关系
- 假设校准函数是 sigmoid 形状，不一定准确

---

## 4. Isotonic Regression（保序回归）

### 4.1 核心原理

**约束**：仅要求校准函数单调不减（预测分数越高，校准后概率越高）。

**形式**：找到一个分段常数函数 $m(\cdot)$，使得：

$$\hat{p}_i = m(f_i) \quad \text{s.t.} \quad f_i \leq f_j \Rightarrow m(f_i) \leq m(f_j)$$

$$\min_m \sum_i (y_i - m(f_i))^2$$

### 4.2 PAVA 算法（Pool Adjacent Violators Algorithm）

```python
def isotonic_regression(scores, labels):
    """
    PAVA 算法：寻找单调非减的最优拟合
    """
    n = len(scores)
    # 按分数排序
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]
    
    # 初始化：每个点独立
    groups = [[i, sorted_labels[i], sorted_labels[i]] for i in range(n)]
    # [索引范围, 均值, 累积权重]
    
    i = 0
    while i < len(groups) - 1:
        # 检查单调性是否被违反
        if groups[i][1] > groups[i+1][1]:
            # 合并两个组（池化）
            merged_mean = (groups[i][1] * groups[i][2] + 
                          groups[i+1][1] * groups[i+1][2]) / \
                         (groups[i][2] + groups[i+1][2])
            merged_weight = groups[i][2] + groups[i+1][2]
            groups[i] = [groups[i][0], merged_mean, merged_weight]
            groups.pop(i+1)
            if i > 0:
                i -= 1  # 回退检查
        else:
            i += 1
    
    # 将分组映射回原始顺序
    calibrated = np.zeros(n)
    for group in groups:
        start, mean, _ = group[0], group[1], group[2]
        calibrated[start] = mean
    
    return calibrated[np.argsort(sorted_indices)]
```

### 4.3 比较 Platt Scaling vs Isotonic Regression

| 维度 | Platt Scaling | Isotonic Regression |
|------|--------------|---------------------|
| 参数量 | 2（A、B）| 无参数（非参数方法）|
| 灵活性 | 低（sigmoid 约束）| 高（任意单调函数）|
| 数据需求 | 少（几百条够）| 多（推荐 > 1000 条）|
| 过拟合风险 | 低 | 中（数据少时过拟合）|
| 计算开销 | 低 | 低（O(n log n) 排序）|
| 适用场景 | 数据少，关系接近 sigmoid | 数据多，非线性校准关系 |

---

## 5. Temperature Scaling

### 5.1 核心公式

对模型的原始 logit 输出 $z$（softmax 前的值）除以温度 $T$：

$$p = \text{softmax}(z / T)$$

对于二分类（CTR 预估），等价于：

$$\hat{p}(\text{CTR}) = \sigma(z / T)$$

### 5.2 温度参数的效果

**$T > 1$（降低置信度/平滑分布）**：
- 压缩 logit 幅度：高分变低，低分变高
- 概率分布趋于均匀
- 适用于模型过于自信（系统性高估）的情况

**$T < 1$（提高置信度/锐化分布）**：
- 放大 logit 幅度：高分更高，低分更低
- 概率分布更极端
- 适用于模型不够自信（系统性低估）的情况

### 5.3 Temperature Scaling 的特点

```python
def temperature_scaling(val_logits, val_labels):
    """
    在验证集上搜索最优温度 T
    """
    from scipy.optimize import minimize_scalar
    
    def neg_log_likelihood(T):
        scaled_probs = 1 / (1 + np.exp(-val_logits / T))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        return -np.mean(
            val_labels * np.log(scaled_probs) + 
            (1 - val_labels) * np.log(1 - scaled_probs)
        )
    
    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method='bounded')
    T_opt = result.x
    
    print(f"最优温度 T = {T_opt:.3f}")
    if T_opt > 1:
        print("模型过于自信，需要平滑")
    else:
        print("模型置信度不足，需要锐化")
    
    def calibrated_predict(logits):
        return 1 / (1 + np.exp(-logits / T_opt))
    
    return calibrated_predict

# Temperature Scaling 的关键优势：只需调一个参数
# 与 Platt Scaling 的区别：没有偏置项 B，只缩放不平移
```

**优势**：只有 1 个参数，最难过拟合；实测 Deep Learning 分类模型上效果很好（Guo et al. 2017 发现 DNN 普遍过度自信）。

---

## 6. 广告 CTR 校准实践

### 6.1 训练/预测时间的分布偏差

**时间偏移（Temporal Shift）**：
- 训练数据来自 T-7 到 T-1 天
- 预测时间为当天
- 用户行为模式随时间变化（节假日、热点事件等）

**应对策略**：
1. 时间加权训练：近期样本给更高权重 $w_t = \lambda^{T_{current} - T_{sample}}$
2. 在近期数据上重新校准（每天或每小时更新校准参数）
3. 特征工程：加入时间戳特征、节假日标记

### 6.2 在线校准 vs 离线校准

**离线校准**：
- 在离线验证集（时间分割）上训练校准参数
- 定期更新（如每天重训）
- 优点：稳定，可充分优化
- 缺点：有延迟，无法应对突发分布变化

**在线校准（更先进）**：
- 实时观测点击/未点击反馈
- 指数滑动平均（EMA）更新校准参数：$T_t = \alpha T_{t-1} + (1-\alpha) \hat{T}_t$
- 更快响应分布变化，但噪声更大

### 6.3 低流量长尾 ID 的校准挑战

问题：长尾广告主/商品的展示量极少（如每天 100 次曝光），无法在其上可靠估计 CTR。

**方案1：借用全局校准**
- 对低流量 ID，使用全局校准参数（所有广告共享的 Platt Scaling）
- 仅对流量 > 阈值的 ID 做个性化校准

**方案2：分层校准（Hierarchical Calibration）**
```python
def hierarchical_calibration(scores, labels, id_features, min_samples=500):
    """
    根据样本量决定校准粒度：
    - 全局校准：所有广告
    - 品类校准：同品类广告共享参数
    - 个体校准：流量足够大的广告
    """
    global_calibrator = platt_scaling(scores, labels)  # 全局
    
    category_calibrators = {}
    for cat in unique_categories:
        cat_mask = id_features[:, 'category'] == cat
        if cat_mask.sum() >= min_samples:
            category_calibrators[cat] = platt_scaling(
                scores[cat_mask], labels[cat_mask]
            )
    
    def predict(scores, id_feat):
        cat = id_feat['category']
        ad_id = id_feat['ad_id']
        
        # 优先使用最细粒度的可用校准器
        if ad_id in individual_calibrators:
            return individual_calibrators[ad_id](scores)
        elif cat in category_calibrators:
            return category_calibrators[cat](scores)
        else:
            return global_calibrator(scores)
    
    return predict
```

**方案3：贝叶斯校准**：用全局先验 + 个体似然的贝叶斯估计，自然地在数据少时退回全局估计。

### 6.4 校准效果评估流程

```python
def calibration_evaluation(y_true, y_pred, y_calibrated, n_bins=10):
    """完整的校准评估"""
    
    # 1. ECE 对比
    ece_before = expected_calibration_error(y_true, y_pred, n_bins)
    ece_after = expected_calibration_error(y_true, y_calibrated, n_bins)
    print(f"ECE: {ece_before:.4f} -> {ece_after:.4f} (降低 {(1-ece_after/ece_before)*100:.1f}%)")
    
    # 2. Log Loss 对比
    from sklearn.metrics import log_loss
    ll_before = log_loss(y_true, y_pred)
    ll_after = log_loss(y_true, y_calibrated)
    print(f"Log Loss: {ll_before:.4f} -> {ll_after:.4f}")
    
    # 3. AUC（排序性不应下降）
    from sklearn.metrics import roc_auc_score
    auc_before = roc_auc_score(y_true, y_pred)
    auc_after = roc_auc_score(y_true, y_calibrated)
    print(f"AUC: {auc_before:.4f} -> {auc_after:.4f}（期望不变）")
    
    # 4. 高低分桶分析（广告业务关注）
    pred_ctr_high = y_calibrated[y_calibrated > 0.1].mean()
    actual_ctr_high = y_true[y_calibrated > 0.1].mean()
    print(f"高分桶（>0.1）校准比: {pred_ctr_high/actual_ctr_high:.3f}（期望接近 1）")
```

---

## 7. 面试考点

### Q1：为什么 AUC 高的 CTR 模型还需要校准？

AUC 衡量的是模型对正负样本的**相对排序能力**，与绝对概率值无关。例如，模型 A 给出 [0.01, 0.05, 0.1]，模型 B 给出 [0.1, 0.5, 1.0]，两者 AUC 完全相同（排序一致），但只有模型 A 的绝对概率是准确的。在 oCPC 出价等场景，决策依赖准确的绝对概率值，AUC 高的模型如果系统性高估 5 倍，会导致出价虚高 5 倍，产生严重的商业损失。

### Q2：Platt Scaling 在深度学习 CTR 模型上的适用性？

Deep Learning CTR 模型（DLRM、DeepFM 等）通常以 sigmoid 为最后激活函数，理论上输出已是概率。但实际中，由于样本不均衡（正负 1:1000）、训练分布与线上分布不一致等原因，模型输出仍然需要校准。Platt Scaling 在 DL 模型上的局限：DL 模型的校准偏差可能是复杂非线性的（如对高分区间高估、对低分区间低估），而 Platt Scaling 只能做线性修正，此时 Isotonic Regression 或分段 Platt 更合适。

### Q3：温度缩放为什么比 Platt Scaling 少一个参数？有什么意义？

Platt Scaling 有偏置项 B，允许整体平移概率分布（如系统性低估时上移）。Temperature Scaling 只缩放不平移（B=0），假设模型整体无偏，只有方差（置信度）不准。研究发现，现代深度网络的主要校准问题是**过度自信（over-confident）**而非系统偏移，因此只调温度就够用。实践建议：先用 Temperature Scaling（1参数），若效果不够再换 Platt Scaling（2参数）。

### Q4：如何检测 CTR 模型在线上是否发生了校准偏移？

监控指标：(1) 每小时的 Prediction Average / Actual CTR 比值（期望接近 1.0）；(2) 按分数桶的 ECE 趋势（是否在增大）；(3) 高出价广告的 ROI 是否异常（间接反映校准质量）。告警阈值：通常当比值偏离 1.0 超过 ±10% 且持续 1 小时以上，触发模型重新校准或回滚。快速响应：在线校准参数（单个参数 T 或 A/B）可以在几分钟内更新，无需重新训练模型。

### Q5：样本不均衡（1:1000）如何影响 CTR 校准？

训练时为了计算效率，通常会对负样本下采样（如 1:10），这导致模型学到的"基准率"比真实更高，输出概率系统性高估。修正方法：在训练时加入采样权重修正，或在预测时用 Prior Correction：若训练时正负比为 1:r，真实正负比为 1:R，则：$p_{corrected} = \frac{p_{train} \cdot R}{p_{train} \cdot R + (1-p_{train}) \cdot r \cdot (R/r)} = \frac{p_{train}}{p_{train} + (1-p_{train}) \cdot r/R}$。这是解析修正，比重训更高效。

### Q6：Wide & Deep、GBDT 等不同类型模型的校准特点有何不同？

GBDT（XGBoost/LightGBM）：输出通常是叶子节点的比例，已有一定的概率意义，但在极端值区域仍可能过于自信，Isotonic Regression 效果好。Deep Learning（DLRM/DeepFM）：sigmoid 输出在理论上是概率，但实际过度自信（Guo et al. 2017 发现层数越深、正则化越弱的网络校准越差），Temperature Scaling 是首选。Logistic Regression：输出本身就是 Platt 校准，通常无需额外校准。

### Q7：多任务学习中如何对各任务分别进行校准？

多任务模型（MMOE）的不同任务输出头可能有不同的校准偏差。正确做法：对每个任务头独立做校准，不应共享校准参数。步骤：(1) 在任务独立的验证集上评估各任务的 ECE；(2) 对每个任务分别拟合温度或 Platt 参数；(3) 监控各任务校准质量的时间变化，有可能不同任务漂移速度不同。注意：如果任务 A 的校准参数变化很大但任务 B 不变，可能说明任务 A 对应的业务分布发生了变化，需要调查原因。

---

## 参考资料

- Guo et al. "On Calibration of Modern Neural Networks" (Temperature Scaling, ICML 2017)
- Platt. "Probabilistic Outputs for Support Vector Machines" (Platt Scaling, 1999)
- Zadrozny & Elkan. "Transforming Classifier Scores into Accurate Multiclass Probability Estimates" (Isotonic Regression, KDD 2002)
- Niculescu-Mizil & Caruana. "Predicting Good Probabilities With Supervised Learning" (ICML 2005)
- He et al. "Practical Lessons from Predicting Clicks on Ads at Facebook" (广告 CTR 校准实践, KDD 2014)
