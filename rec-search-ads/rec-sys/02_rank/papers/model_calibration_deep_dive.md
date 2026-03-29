# 模型校准（Model Calibration）完整学习笔记
> 发展进程 · 核心概念 · 工业实践
> 日期：2026-03-23 | 作者：MelonEggLearn

---

## TL;DR

1. **校准 = 让模型输出的概率"名副其实"**：预测 CTR=0.1 的样本，真实点击率应该约等于 10%，而不是偏高或偏低。
2. **推荐/广告系统中校准极其重要**：eCPM = bid × pCTR，如果 pCTR 系统性偏高，广告主付更多钱但实际 ROI 不达预期；推荐系统中多目标融合也依赖各目标概率同尺度。
3. **发展路径**：Platt Scaling（1999）→ Isotonic Regression（2002）→ Temperature Scaling（2017）→ 推荐专用校准（负采样修正/位置偏差校正/分布偏移校准，2018-2024）。
4. **不校准的常见原因**：负采样（正负样本比例被人为改变）、位置偏差（展示位置影响点击率）、模型过拟合（极端预测）、训练-服务分布偏移。
5. **现代工业实践**：校准不是"训练完再后处理"，而是融入整个训练-部署-监控 pipeline 的系统工程。

---

## 一、什么是模型校准？为什么重要？

### 1.1 定义

**校准（Calibration）**：模型输出的预测概率 $\hat{p}$ 应与真实事件发生的频率 $p_{true}$ 一致。

数学定义——**完美校准**：

$$
P(\hat{p} = p) = p, \quad \forall p \in [0, 1]
$$

即"模型说概率是0.3的样本，在现实中确实有30%会发生"。

### 1.2 校准的可视化：可靠性图（Reliability Diagram）

```
真实概率
  1.0  |          ....../（完美校准）
  0.8  |      .../  
  0.6  |  .../    
  0.4  |./    
  0.2  |
  0.0  |___________________
       0.0  0.2  0.4  0.6  0.8  1.0  预测概率

过度自信（Overconfident）：曲线在对角线下方（预测高，实际低）
欠置信（Underconfident）：曲线在对角线上方（预测低，实际高）
```

**步骤**：把预测概率分成 10 个等宽 bin（0-0.1，0.1-0.2，...），计算每个 bin 内的真实正例率，画出来就是可靠性图。

### 1.3 ECE（Expected Calibration Error）——校准误差度量

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

- $B_m$：第 m 个 bin 内的样本集合
- $\text{acc}(B_m)$：bin 内真实正例率
- $\text{conf}(B_m)$：bin 内预测概率均值
- $|B_m|/n$：该 bin 的权重（样本占比）

**MCE（Maximum Calibration Error）**：取各 bin 偏差的最大值（关注最坏情况）。

### 1.4 推荐/广告中为什么校准特别重要？

| 场景 | 不校准的后果 |
|------|------------|
| **广告 eCPM 排序** | eCPM = bid × pCTR，pCTR 偏高 → 广告主出价虚高、ROI 不达预期 → 流失广告主 |
| **多目标融合** | rank_score = w₁·pCTR + w₂·pCVR + w₃·pLike，各目标概率尺度不一致 → 权重 w 无法解释 |
| **智能出价** | tCPA 出价系统根据 pCVR 出价，pCVR 偏高 → 实际 CPA 超出预算 |
| **AB 实验** | 实验组 pCTR 校准好、对照组偏高 → 线上 CTR 对比失真 |
| **推荐多样性** | 基于分数的多样性策略依赖概率有意义，不校准则多样性无从优化 |

---

## 二、发展时间线与各阶段核心概念

---

### 📌 阶段0：问题背景（1990s 前）

早期分类器（SVM、决策树、朴素贝叶斯）的输出**不是概率**，或者是概率但严重不准。

- SVM 输出的是距超平面的距离，不是概率
- 朴素贝叶斯输出的是 unnormalized 的概率比，系统性偏极端（趋向0或1）
- 决策树叶节点的训练集比例是有偏估计（叶节点样本少时极不准）

因此"如何把分类器输出变成可信概率"成为机器学习的重要问题。

---

### 📌 阶段1：Platt Scaling（1999）
**论文**：Probabilities for SV Machines（Platt，1999）

#### 核心思想

在 SVM 的 decision function 输出 $f(x)$（原始得分）上，拟合一个 Sigmoid 函数：

$$
\hat{p} = \sigma(Af(x) + B) = \frac{1}{1 + e^{Af(x) + B}}
$$

其中 $A, B$ 是通过在**验证集**（calibration set，与训练集独立）上最大化似然来估计的：

$$
\min_{A, B} \sum_i -y_i \log(\hat{p}_i) - (1-y_i)\log(1-\hat{p}_i)
$$

#### 关键概念

**为什么要用独立验证集（不能用训练集）？**
- SVM 在训练集上过拟合，训练集的 $f(x)$ 分布与测试集不同
- 用训练集拟合 Platt Scaling 会导致校准器学到虚假模式
- 必须使用与 SVM 训练不重叠的数据来拟合 $A, B$

**Platt Scaling 的假设**：
- 假设原始分数 $f(x)$ 与 log odds 是线性关系
- 即 $\log \frac{p}{1-p} = Af(x) + B$（线性关系假设）

**局限性**：
- 线性假设过强：如果模型输出和真实概率是非线性关系，Platt Scaling 效果差
- 样本量需求：需要足够的验证集样本（每个类别至少几百条）
- 只有两个参数（$A, B$），表达能力有限

#### 工业应用

Platt Scaling 至今仍广泛用于：广告 CTR 后处理、医疗风险评分校准。因为简单稳定，容易 debug。

---

### 📌 阶段2：Isotonic Regression（等渗回归，2002）
**论文**：Transforming Classifier Scores into Accurate Multiclass Probability Estimates（Zadrozny & Elkan，2002）

#### 核心思想

用一个**单调递增的阶梯函数**来拟合校准映射，不做任何线性假设：

```
原始概率分布（验证集）：
样本得分:  0.1  0.3  0.5  0.6  0.8
真实标签:   0    1    0    1    1

Isotonic Regression 找到最优单调函数 m(·) 使得：
m(0.1) ≤ m(0.3) ≤ m(0.5) ≤ m(0.6) ≤ m(0.8)
且最小化 Σ(m(p̂ᵢ) - yᵢ)²
```

**Pool Adjacent Violators（PAV）算法**：Isotonic Regression 的经典求解算法，O(n) 复杂度：

```python
def PAV(scores, labels):
    # 按得分排序
    pairs = sorted(zip(scores, labels))
    
    # PAV：合并违反单调性的相邻区间
    blocks = [[y] for _, y in pairs]
    while True:
        # 找到相邻违反单调性的块（前块均值 > 后块均值）
        violation = find_violation(blocks)
        if not violation:
            break
        # 合并：用两块均值替代
        merge(blocks, violation)
    
    # 每个样本的校准后概率 = 所在块的均值
    return [mean(block) for s, block in zip(sorted_scores, blocks)]
```

#### 与 Platt Scaling 的对比

| | Platt Scaling | Isotonic Regression |
|--|---------------|-------------------|
| 函数形式 | 固定（Sigmoid） | 自由（单调阶梯）|
| 参数数量 | 2 个（$A, B$）| 随数据量增长 |
| 数据需求 | 少（几百条）| 多（至少几千条）|
| 过拟合风险 | 低 | 高（小数据集）|
| 表达能力 | 弱（线性假设）| 强（任意单调函数）|
| 适用场景 | 数据少，快速部署 | 数据充足，精度优先 |

#### 局限性

- 在训练数据上过拟合（特别是边界区域）
- 对测试分布偏移鲁棒性差
- 无法外推（训练集未见过的得分范围无法校准）

---

### 📌 阶段3：Histogram Binning（分桶直方图，2001-2005）

#### 核心思想

最直觉的校准方法：把预测概率分成 $M$ 个 bin，每个 bin 的校准值 = 该 bin 内的真实正例率。

```
Bin 划分（等宽 or 等频）：
Bin 1: [0.0, 0.1) → 真实正例率 = 0.05
Bin 2: [0.1, 0.2) → 真实正例率 = 0.13
...
Bin 10:[0.9, 1.0) → 真实正例率 = 0.87

校准：f_cal(p̂) = true_rate(bin containing p̂)
```

**等宽 vs 等频 Binning**：
- **等宽（Equal-Width）**：每个 bin 宽度相同（如 0.1），但高分/低分区间样本可能极少，估计不准
- **等频（Equal-Frequency/Quantile）**：每个 bin 样本数相同，分位数划分，边界在高密度区域更密集

#### BBQ（Bayesian Binning into Quantiles，2015）

引入贝叶斯框架，自动选择最优 bin 数量：

$$
P(\text{bins}|D) \propto P(D|\text{bins}) \cdot P(\text{bins})
$$

用 BIC（Bayesian Information Criterion）作为近似，平衡拟合度（bins 多 → 拟合好）和复杂度（bins 多 → 过拟合）。实验结果比固定 bin 数的 Histogram Binning 更稳定。

---

### 📌 阶段4：Beta Calibration（2017）
**论文**：Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers（Kull et al., 2017）

#### 核心思想

Platt Scaling 用 Sigmoid 校准，但 Sigmoid 假设是对称的。实际中很多分类器（如随机森林、朴素贝叶斯）的输出分布是非对称的（偏向0或1）。Beta Calibration 用 Beta 分布的 CDF 做校准函数，更灵活：

$$
\hat{p}_{cal} = \frac{1}{1 + e^{-a \log \hat{p} - b \log(1-\hat{p}) - c}}
$$

其中 $a, b, c$ 是三个参数（Platt Scaling 是特殊情况，$a=b$，参数退化为 2 个）。

**直觉**：
- $a > b$：对低分区间压缩，对高分区间放大（适合过度自信的分类器）
- $a < b$：反之（适合欠置信的分类器）
- $a = b$：退化为 Platt Scaling（对称情况）

---

### 📌 阶段5：Temperature Scaling（2017）
**论文**：On Calibration of Modern Neural Networks（Guo et al., 2017，ICML 2017）

#### 核心发现

**深度神经网络越来越不校准**：论文系统研究了 ResNet、DenseNet 等现代深度网络，发现它们比传统机器学习方法（SVM、浅层网络）更不准确的校准（更过度自信）。

原因：
1. **深度网络过参数化**，在训练集上完美拟合（loss → 0），预测趋向极端（0或1）
2. **BatchNorm + Weight Decay** 的组合会导致模型更自信（参数范数被压缩，但预测范围扩大）
3. **Depth and Width**：更深更宽的网络更不准确

#### Temperature Scaling

最简单有效的校准方法——在 softmax 之前除以温度 $T$：

$$
\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

**$T$ 的作用**：
- $T > 1$：**软化**概率分布（降低置信度），使 overconfident 模型变校准
- $T < 1$：**锐化**概率分布（提高置信度），使 underconfident 模型变校准
- $T = 1$：无变化（原始 softmax）

**优点极为突出**：
- 只有 **1 个参数**，在验证集上用 NLL 最小化求解
- **不改变模型精度**（Accuracy 不变，因为 argmax 不变）
- **计算极快**，工程部署简单（推理时加一步除法）

#### 关键实验结论

| 方法 | ECE（ResNet-110，CIFAR-100）|
|------|--------------------------|
| 无校准 | 0.1645 |
| Histogram Binning | 0.0273 |
| Isotonic Regression | 0.0199 |
| Platt Scaling | 0.0175 |
| **Temperature Scaling** | **0.0116** |

**Temperature Scaling 以最简单的方式达到最好的效果**。

---

### 📌 阶段6：推荐/广告专用校准（2018-2021）

通用校准方法（Platt/Temperature）在推荐广告场景遇到特有挑战，催生了几类专用技术。

---

#### 6.1 负采样校准修正（2018，Meta/Facebook）
**论文**：Practical Lessons from Predicting Clicks on Ads at Facebook（2014 + 后续）

**问题**：CTR 训练时正负样本比例被人为改变（原始正负约 1:99，训练时下采样负样本到 1:2 或 1:10），导致模型预测的绝对概率偏高（模型以为正例更频繁）。

**校准修正公式**：

设原始正负比例为 $r_{real}$（如 0.01），训练集正负比例为 $r_{train}$（如 0.5），则对模型输出的原始分 $\hat{p}$ 做修正：

$$
\hat{p}_{calibrated} = \frac{\hat{p}}{\ \hat{p} + \frac{1-\hat{p}}{q}}
$$

其中 $q = \frac{r_{real}}{r_{train}}$ 是负采样率（正例保留比例为1，负例保留比例为 q）。

**直觉推导**（贝叶斯视角）：

$$
\text{true odd} = \frac{p}{1-p} \cdot \frac{q}{1} = \frac{\hat{p}_{train}}{1-\hat{p}_{train}} \cdot q
$$

转换回概率即得上述公式。

**工程实现**（Meta 实践）：
```python
def calibrate_negative_sampling(p_hat, sampling_rate):
    """
    p_hat: 模型输出的 sigmoid 概率
    sampling_rate: 负样本保留比例（如 0.1 表示只保留 10% 负样本）
    """
    return p_hat / (p_hat + (1 - p_hat) / sampling_rate)

# 示例
p_hat = 0.15  # 模型预测（在下采样训练集上）
sampling_rate = 0.1  # 负样本保留10%（正例全保留）
p_calibrated = calibrate_negative_sampling(p_hat, 0.1)
# p_calibrated ≈ 0.016（回归到真实空间的低概率）
```

---

#### 6.2 位置偏差校准（Position Bias Calibration，2019）
**论文**：Position Bias Estimation for Unbiased Learning to Rank（等，2019）

**问题**：搜索/推荐中靠前位置的 item 天然点击率更高（不是因为 item 更好，而是因为位置更显眼）。模型如果不处理，会学到"排第1的 item CTR 高"这个虚假规律。

**两种主流方法**：

**方法1：Position 作为特征（训练时）**
```
训练: 输入特征中加入 position_embedding
推理: 固定 position=1（假设都排在第1位），消除位置影响
```
简单有效，但模型需要记忆 position 的影响，容易过拟合。

**方法2：Propensity Score 校正（逆概率加权）**

$$
L_{unbiased} = \sum_i \frac{1}{\theta(pos_i)} \cdot \mathbb{1}[click_i] \cdot \log \hat{p}_i
$$

其中 $\theta(pos_i)$ 是位置 $pos_i$ 被用户实际看到的概率（曝光倾向分），用随机化实验（随机打乱展示位置）来估计。

**Dual Learning 方法（淘宝 2019）**：
- 同时学习 CTR 预估模型 和 位置偏差模型
- 两个模型互相解耦：CTR 模型只预测"真实 CTR"，位置模型只预测"位置对 CTR 的乘数效应"
```
P(click | item, pos) = P(click | item) × θ(pos)
两个目标分别学习，互相不干扰
```

---

#### 6.3 分布偏移校准（Distribution Shift Calibration，2020-2022）

**问题**：推荐系统的训练分布和服务分布往往不同：
- 训练数据是历史分布（旧模型决策产生的曝光）
- 服务时是新模型决策产生的曝光
- 这种 **Feedback Loop** 导致模型在训练时校准，服务时偏差越来越大

**时间分布偏移**：用户兴趣随时间变化，3个月前的模型在当前数据上校准差。

**协变量偏移校准（Covariate Shift）**：

$$
\hat{p}_{calibrated}(x) = \frac{P_{train}(x)}{P_{test}(x)} \cdot \hat{p}_{train}(x)
$$

用密度比估计（Density Ratio Estimation）来纠正分布偏移：
```python
# 用分类器估计密度比
classifier = LogisticRegression()
# 训练集样本标签=0，测试集样本标签=1
classifier.fit(X_combined, y_domain)
# 密度比 w(x) = P_test(x) / P_train(x)
w = classifier.predict_proba(X_test)[:, 1] / classifier.predict_proba(X_test)[:, 0]
```

---

### 📌 阶段7：大规模在线校准（2022-2024）

工业系统中的校准挑战：

**7.1 连续校准（Continuous Calibration）**

传统：训练好模型 → 一次性校准 → 上线。
问题：分布持续变化，校准参数很快失效。

解法：每小时/每天更新校准参数（Temperature / Platt 参数），保持与最新数据对齐：
```
每小时：用过去24h的曝光-点击数据重新拟合 T（Temperature Scaling）
监控：实时 ECE，若 > 0.01 触发重新校准
```

**7.2 分段校准（Segment-wise Calibration）**

不同人群/场景的校准需求不同：
- 新用户 vs 老用户的 pCTR 分布完全不同
- 不同类目（高频/低频）的 CTR 均值差 10x

解法：按 user_group × item_category 分段，每段独立拟合校准参数：
```
calibration_table[user_segment][item_category] = (A, B)
线上：查表做 Platt Scaling
```

**7.3 双校准（Dual Calibration，适用于多目标场景）**

问题：pCTR 和 pCVR 都校准了，但 pCTCVR = pCTR × pCVR 不一定校准（两个误差叠加）。

解法：在 pCTCVR 层再加一次校准，但注意：
- 不能打破 pCTCVR = pCTR × pCVR 的约束（ESMM 的联合约束）
- 方法：对 pCTCVR 用 Platt Scaling，反推修正 pCVR

---

### 📌 阶段8：Label Shift 校准与因果校准（2023-2024 前沿）

**8.1 Label Shift 问题**

不仅特征分布变，标签分布（正负比例）也在变：
- 大促期间 CVR 大幅提升（但历史训练数据是日常 CVR）
- 新冠疫情期间所有消费行为突变

**Label Shift 校准（BBSE 方法）**：

$$
P_{test}(y) = \mathbf{W} \cdot P_{train}(y)
$$

估计标签分布变化矩阵 $\mathbf{W}$，对模型后验做修正。

**8.2 因果视角的校准（Counterfactual Calibration）**

传统校准只处理"观察到的点击"，但有些曝光根本没被用户看到（滑过去了）。

**IPS 加权校准**（逆概率加权，Inverse Propensity Scoring）：

$$
\text{ECE}_{debiased} = \sum_{i} \frac{1}{e(x_i)} \cdot |\hat{p}_i - y_i|
$$

其中 $e(x_i)$ 是样本 $i$ 被真正观察到的概率（曝光倾向分）。

---

## 三、各阶段对比总结

| 阶段 | 方法 | 参数量 | 表达能力 | 数据需求 | 过拟合风险 | 适用场景 |
|------|------|-------|---------|---------|-----------|---------|
| 1 | Platt Scaling | 2 | 弱（线性）| 少 | 低 | 快速部署，通用 |
| 2 | Isotonic Regression | 多（自由）| 强（单调）| 多 | 中 | 数据充足，精度优先 |
| 3 | Histogram Binning | M（bin数）| 中 | 中 | 中 | 分布平滑 |
| 3 | Beta Calibration | 3 | 中（非对称）| 少-中 | 低 | 非对称分类器 |
| 5 | **Temperature Scaling** | **1** | **弱（全局缩放）**| **少** | **极低** | **神经网络，首选** |
| 6 | 负采样修正 | 0（公式）| 精确（理论）| 不需要 | 无 | **推荐广告必须** |
| 6 | 位置偏差校准 | 按位置 | 精确 | 中 | 低 | 搜索/推荐排序 |
| 7 | 分段校准 | 段数×2 | 强 | 多 | 中 | 大规模工业系统 |
| 8 | 因果校准（IPS）| 1（propensity）| 精确 | 中 | 低 | 有曝光偏差的场景 |

---

## 四、工业推荐系统完整校准 Pipeline

```
原始模型训练
    ↓
1. 负采样修正（离线，公式修正）
   → 消除采样偏差，把绝对概率拉回真实空间

2. 位置偏差校准（在线/离线）
   → position feature + propensity 加权训练

3. Temperature Scaling 后处理（离线）
   → 验证集 ECE 最小化，求最优 T
   → 简单稳定，适合作为兜底校准

4. 分段校准（离线，每日更新）
   → 按 user_group × scene 分组，独立 Platt Scaling
   → 处理不同场景 CTR 基率差异

5. 连续校准监控（在线）
   → 实时计算 ECE，自动触发重校准
   → 监控：预测 CTR 均值 vs 实际 CTR 均值的相对偏差

校准质量检验：
    可靠性图（Reliability Diagram）
    ECE < 0.005（严格）/ < 0.01（宽松）
    Brier Score（综合校准+分辨率）
```

---

## 五、面试高频考点（口语化版本）

**Q1. 什么是模型校准，为什么推荐/广告系统必须做校准？**

> 校准就是让模型输出的概率"可信"。如果模型说 CTR=0.1，那真实的点击率就应该大约是 10%，不能虚高或虚低。在广告里，eCPM = bid × pCTR，pCTR 不准的话广告主出价就乱了，ROI 不可预期，广告主会流失。在多目标推荐里，把 pCTR 和 pCVR 加权求和的时候，如果两个值的绝对量纲不一样（一个偏大一个偏小），权重就失去意义。

**Q2. Temperature Scaling 原理是什么？为什么它效果这么好？**

> 就是在 softmax 之前把 logit 除以一个温度 T。T > 1 就是软化概率，把过度自信的预测拉回来。神经网络特别容易过度自信，因为过参数化、训练时强迫 loss 趋向0，导致预测趋向0或1。Temperature Scaling 用一个参数在验证集上最小化 NLL 就能解决这个问题，而且不影响 Accuracy（argmax 不变），实现也极简单，几乎是神经网络的标配后处理。

**Q3. 负采样对 CTR 预估校准的影响，怎么修正？**

> 做 CTR 预估时，正负样本比例可能是1:100，如果原样训练的话负例太多、收敛慢，所以通常会把负例下采样到1:2或1:10。问题是模型以为正例有50%概率发生（按训练集比例），但真实只有1%。所以模型预测的绝对概率是虚高的。修正公式是：p_calibrated = p / (p + (1-p)/q)，其中 q 是负样本的保留比例。这个是数学上严格正确的贝叶斯修正，不需要额外数据，是推荐广告里必做的基础操作。

**Q4. Platt Scaling 和 Isotonic Regression 怎么选？**

> 数据少用 Platt（只有2个参数，不容易过拟合）；数据多用 Isotonic（更灵活，不假设线性）。Platt 假设分类器的原始分和真实概率是 sigmoid 关系，适合 SVM；如果模型输出是非线性的，Platt 效果差。Isotonic Regression 用 PAV 算法拟合单调函数，能处理任意非线性关系，但需要几千以上的校准集，否则过拟合。实际工作中神经网络用 Temperature Scaling 就够了，Platt/Isotonic 更多用在树模型或传统机器学习上。

**Q5. 位置偏差怎么处理？Position 作为特征 vs Propensity Score 有什么区别？**

> 两种方法都是为了去掉"位置越靠前点击率天然越高"这个虚假规律。Position 作为特征：训练时让模型知道位置信息，推理时把 position 固定为1，模拟所有 item 都排第一的情况。简单，但模型参数里会记住 position 的影响，有时会学偏。Propensity Score 方法：估计每个位置被用户看到的概率（曝光倾向），用它对训练样本加权（IPW），相当于模拟一个"所有位置曝光概率相同"的虚拟实验。更理论正确，但需要随机化实验来估计倾向分，工程实现复杂一些。

**Q6. ECE 怎么计算，有哪些局限性？**

> ECE 是把预测概率分成若干 bin，计算每个 bin 里预测均值和真实正例率的加权平均绝对偏差。计算很直观，ECE 越小越好（完美校准=0）。局限性有几个：第一，对 bin 数量 M 很敏感，M 太少（bin 太宽）会掩盖局部不校准；M 太大（每个 bin 样本太少）估计噪声大。第二，ECE 是全局指标，一个局部高度不校准的 bin 可能被其他好的 bin 平均掉。第三，ECE 不区分方向（过度自信 vs 欠置信），有时需要用 signed ECE 或者直接画可靠性图来看方向。

**Q7. 在线推荐系统怎么做持续校准？**

> 分布是持续变化的，一次性校准很快就会失效。做法是：用最近 24-72 小时的数据重新拟合校准参数（Temperature 或 Platt 的 A、B），每天或每小时更新一次。监控侧要实时计算预测 pCTR 的均值和实际 CTR 的比值（校准比），比值偏离 1.0 超过一定阈值（如 ±10%）就触发重校准。另外要分场景、分用户群分别监控，因为不同场景的校准漂移方向可能不一样。

**Q8. 多目标系统里（CTR + CVR），怎么保证乘积 pCTCVR 也是校准的？**

> 就算 pCTR 和 pCVR 都单独校准了，pCTCVR = pCTR × pCVR 的乘积也不一定校准，因为两个有偏差的概率相乘，偏差会放大。最直接的做法是对 pCTCVR 再加一层后处理校准（Platt Scaling 或 Temperature）。但要注意 ESMM 架构里 pCTCVR 是 pCTR × pCVR 的约束，直接修改 pCTCVR 会破坏这个约束。工程上常见的折中：只对 pCTCVR 做校准（不对 pCTR、pCVR 单独校准），或者用双校准（先 pCTR 校准，反推修正 pCVR 使乘积一致）。

---

## 六、快速参考代码

```python
# Temperature Scaling 实现
class TemperatureScaling:
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def calibrate(self, logits):
        return logits / self.temperature
    
    def fit(self, logits_val, labels_val):
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.calibrate(logits_val), labels_val)
            loss.backward()
            return loss
        
        optimizer.step(eval)

# 负采样修正
def correct_negative_sampling(p_hat, neg_sampling_rate):
    """负采样率：负样本保留比例（如0.1 = 10%负样本被保留）"""
    return p_hat / (p_hat + (1 - p_hat) / neg_sampling_rate)

# ECE 计算
def compute_ece(y_pred, y_true, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return ece

# Platt Scaling
from sklearn.linear_model import LogisticRegression
def platt_scaling(scores_val, labels_val, scores_test):
    platt = LogisticRegression()
    platt.fit(scores_val.reshape(-1, 1), labels_val)
    return platt.predict_proba(scores_test.reshape(-1, 1))[:, 1]
```

---

*文档完成：2026-03-23 | MelonEggLearn*
*参考：Platt(1999), Zadrozny&Elkan(2002), Guo et al.(2017 ICML), Niculescu-Mizil(2005), Facebook CTR(2014), 淘宝PAL(2019)*
