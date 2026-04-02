# 广告 CTR/CVR 预估与校准：从模型到上线的全链路

> 📚 参考文献
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../04_bidding/papers/Real_Time_Bidding_Optimization_with_Multi_Agent_Deep_Rein.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Esmm-Cvr](../papers/esmm_cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Est-Ctr-Scaling](../papers/EST_Efficient_Scaling_Laws_in_CTR_Prediction_via_Unified.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Din Deep Interest Network](../papers/DIN_Deep_Interest_Network_for_Click_Through_Rate_Predicti.md) — DIN: Deep Interest Network for Click-Through Rate Prediction
> - [Multi-Objective-Ads-Ranking](../papers/MMoE_PLE_Pareto.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Llm-Enhanced-Ad-Creative-Generation-And-Optimiz...](../../05_creative/papers/LLM_Enhanced_Ad_Creative_Generation_and_Optimization_for.md) — LLM-Enhanced Ad Creative Generation and Optimization for ...
> - [Gnolr Generalized Nested Ordered Logistic Regre...](../papers/GNOLR_Generalized_Nested_Ordered_Logistic_Regression_for.md) — GNOLR: Generalized Nested Ordered Logistic Regression for...
> - [Deepfm-Ctr](../papers/DeepFMDeep_Factorization_Machine.md) — DeepFM：深度因子分解机（Deep Factorization Machine）

> 创建：2026-03-24 | 领域：广告系统 | 类型：综合分析
> 来源：ESMM, GNOLR, Calibration 实践, 延迟转化处理系列

## 📐 核心公式与原理

### 📐 1. ESMM：全空间 CVR 去偏推导

**核心问题**：CVR 模型只在点击样本上训练，但预测时作用于全曝光空间（SSB，Sample Selection Bias）。

ESMM 的概率链式分解：

$$
P(\text{conversion}|\text{impression}) = P(\text{click}|\text{impression}) \times P(\text{conversion}|\text{click, impression})
$$

即：

$$
p_{\text{CTCVR}}(x) = p_{\text{CTR}}(x) \times p_{\text{CVR}}(x)
$$

**推导步骤：**

1. **定义三个空间**：
   - 曝光空间 $\mathcal{S}$：所有曝光样本（$N$ 条）
   - 点击空间 $\mathcal{O}$：发生点击的样本（$n \ll N$）
   - 转化空间：发生转化的样本（更稀疏）

2. **传统 CVR 的偏差来源**：训练时使用 $\mathcal{O}$ 上的样本，但推断时在 $\mathcal{S}$ 上计算。形式上，训练的是 $P(\text{conv}|\text{click, obs})$，而需要的是 $P(\text{conv}|\text{click})$。

3. **ESMM 的解决方案**：共享 Embedding，两个子任务（CTR、CVR）的参数独立，但用 CTCVR 标签（在全空间有标签）约束 CVR 子网络：

$$
\mathcal{L}_{\text{ESMM}} = \underbrace{\mathcal{L}_{\text{CTR}}(\hat{p}_{\text{CTR}},\, y_{\text{click}})}_{\text{全曝光空间}} + \underbrace{\mathcal{L}_{\text{CTCVR}}(\hat{p}_{\text{CTR}} \cdot \hat{p}_{\text{CVR}},\, y_{\text{click}} \cdot y_{\text{conv}})}_{\text{全曝光空间，间接约束 CVR}}
$$

4. **为何有效**：CVR 子网通过链式乘积间接接受全空间标签监督，$\hat{p}_{\text{CVR}}$ 被隐式约束在全曝光分布上训练，消除 SSB。

**符号说明：**

| 符号 | 含义 |
|------|------|
| $p_{\text{CTR}}(x)$ | 曝光 $x$ 的点击概率，在全空间有标签 |
| $p_{\text{CVR}}(x)$ | 点击后转化概率，传统方法只在点击空间有标签 |
| $p_{\text{CTCVR}}(x)$ | 曝光到转化的联合概率，等于两者乘积 |
| $y_{\text{click}} \cdot y_{\text{conv}}$ | CTCVR 标签（仅曝光+点击+转化时为 1）|

**直观理解：** ESMM 用"点击×转化的联合概率"作为全空间可观测的标签，巧妙地把无法在全空间观测的 CVR 藏进了可观测的 CTCVR，通过链式法则反向约束。

---

### 📐 2. Platt Scaling 校准推导

模型输出 logit $f(x)$，经 sigmoid 后得到未校准概率。校准目标：学习 $a, b$ 使得

$$
p_{\text{calib}}(x) = \sigma(a \cdot f(x) + b) = \frac{1}{1 + e^{-(a f(x) + b)}}
$$

**推导步骤：**

1. **为何需要校准**：深度模型输出的 $\hat{p}$ 通常不等于真实后验 $P(y=1|x)$，在极度不平衡数据（如 CTR~1%）上尤为严重

2. **Platt Scaling 等价于 Logistic Regression on logits**：将已训练模型的 logit 作为单一特征，在留出集上训练一个 1D LR：

$$
\min_{a,b} -\sum_{i=1}^M [y_i \log \sigma(a f_i + b) + (1-y_i)\log(1-\sigma(a f_i + b))]
$$

3. **参数含义**：$a < 1$ 表示模型过度自信（概率分布太极端），需要压缩；$b \neq 0$ 表示模型存在系统性偏差（整体偏高/偏低）

4. **Expected Calibration Error（ECE）**：

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|
$$

其中 $B_m$ 是按置信度划分的桶，$|B_m|/n$ 是样本比例权重，理想情况下 ECE = 0（预测概率 = 实际发生率）。

**符号说明：**
- $f(x)$：模型输出 logit（sigmoid 之前的值）
- $a$：logit 缩放系数（压缩/放大模型置信度）
- $b$：偏置项（修正系统性偏差）
- $\text{acc}(B_m)$：桶 $m$ 内样本的实际正例比例
- $\text{conf}(B_m)$：桶 $m$ 内样本的平均预测概率

---

### 📐 3. FTRL 在线学习更新推导

FTRL（Follow The Regularized Leader）在每个样本到来后更新：

$$
w_{t+1} = \arg\min_w \left[\sum_{\tau=1}^t g_\tau^\top w + \frac{1}{2}\sum_{\tau=1}^t \sigma_\tau \|w - w_\tau\|_2^2 + \lambda_1 \|w\|_1 + \frac{1}{2}\lambda_2 \|w\|_2^2\right]
$$

**推导步骤（对第 $i$ 维参数的闭式解）：**

1. 令 $z_{t,i} = \sum_{\tau=1}^t g_{\tau,i} - \sum_{\tau=1}^t \sigma_{\tau,i} w_{\tau,i}$（累积梯度的修正形式）

2. 闭式解：

$$
w_{t+1,i} = \begin{cases} 0 & \text{if } |z_{t,i}| \le \lambda_1 \\ -\frac{z_{t,i} - \text{sgn}(z_{t,i})\lambda_1}{\frac{1}{\alpha}\sum_{\tau=1}^t \sigma_{\tau,i} + \lambda_2} & \text{otherwise} \end{cases}
$$

3. **L1 产生稀疏性**：当 $|z_{t,i}| \le \lambda_1$ 时参数直接置零，自动特征选择——这是广告系统处理亿级特征维度的关键（通常 95%+ 特征维度为零）

4. **学习率自适应**（类 AdaGrad）：$\sigma_{\tau,i} = \frac{1}{\alpha}(\sqrt{\sum_{s=1}^\tau g_{s,i}^2} - \sqrt{\sum_{s=1}^{\tau-1} g_{s,i}^2})$，历史梯度大的维度步长自动缩小

**符号说明：**
- $g_\tau$：第 $\tau$ 个样本的梯度向量
- $\lambda_1$：L1 正则化系数（控制稀疏度）
- $\lambda_2$：L2 正则化系数（控制权重大小）
- $\alpha$：初始学习率
- $z_{t,i}$：第 $i$ 维的累积正则化梯度（FTRL 的内部状态变量）

**直观理解：** FTRL 是"一直追 loss 历史最小值"的贪心策略，L1 正则化在梯度方向上施加 soft-threshold，使绝大多数低频特征的权重精确归零，让模型在亿级特征下仍能高效存储和推断。

---

## 🎯 核心洞察（5条）

1. **广告 CTR 预估的核心不是 AUC 而是校准性**：排序只需要相对顺序正确（AUC），但出价需要绝对概率准确（pCTR=5% 的广告要付的钱和 pCTR=10% 的不同）
2. **CVR 预估的三大难题**：样本选择偏差（只有点击才有转化标签）、延迟转化（点击后 7 天才转化怎么处理？）、正样本极稀疏（转化率通常 <3%）
3. **ESMM 是 CVR 去偏的标准方案**：通过 CTCVR=CTR×CVR 在全曝光空间建模，CVR 分支隐式获得全空间约束
4. **校准（Calibration）是模型上线前的必要步骤**：Platt Scaling（sigmoid 映射）或 Isotonic Regression（保序回归），确保预测概率与实际发生率一致
5. **在线学习是广告模型的工程特色**：广告数据分布变化快（促销/节日），需要小时级/天级模型更新，FTRL（Follow The Regularized Leader）是在线更新的标准算法

---

## 📈 技术演进脉络

```
LR + 手工特征（2010-2014）→ FM/FFM（2014-2016）→ Wide&Deep/DeepFM（2016-2018）
  → DIN/序列建模（2018-2020）→ ESMM 多任务（2018+）→ LLM 增强 CTR（2024+）
校准：无校准 → Platt Scaling → Isotonic Regression → Field-aware Calibration
更新：离线训练 → 天级增量 → FTRL 实时更新 → 增量 + 全量混合
```

---

## 🎓 常见考点（6条）

### Q1: 广告 CTR 为什么需要校准？
**30秒答案**：出价公式 bid = target_CPA × pCVR，如果 pCVR 预估偏高（过校准），实际出价就偏高，广告主亏损；偏低则投不出去。所以广告模型对概率值的绝对准确性要求远高于推荐。

### Q2: Platt Scaling vs Isotonic Regression？
**30秒答案**：Platt Scaling 用 sigmoid 做全局线性校准 `p_calibrated = sigmoid(a×logit + b)`，参数少适合数据量小；Isotonic Regression 用分段常数做非参数校准，更灵活但需要更多数据。

### Q3: 延迟转化怎么处理？
**30秒答案**：①等待窗口法：点击后等 7 天再确认转化标签（简单但浪费数据）；②Elapsed-Time Model：将"点击后经过多久"作为特征，预测最终转化概率；③Fake Negative Calibration：短期内标记为负样本，事后回补正样本。

### Q4: FTRL 在线学习的核心思想？
**30秒答案**：FTRL 在每个样本到达时更新模型参数，带 L1 正则化产生稀疏解（自动特征选择）。核心公式考虑了历史梯度累积（类似 Adagrad）和正则化。适合高维稀疏特征（十亿级特征维度）。

### Q5: 广告和推荐的 CTR 模型有什么关键差异？
**30秒答案**：①校准要求不同（广告严格 vs 推荐宽松）；②更新频率不同（广告天级/小时级 vs 推荐周级）；③样本构建不同（广告有竞价日志 vs 推荐只有曝光日志）；④特征不同（广告有出价/预算/广告主特征）。

### Q6: CVR 正样本极少怎么办？
**30秒答案**：①ESMM 多任务缓解（CTR 任务有大量样本辅助）；②过采样正样本 + 欠采样负样本 + 样本权重修正；③Focal Loss 聚焦难分样本；④数据增广（类似转化事件作为弱正样本）。

---

### Q7: 广告系统的全链路延迟约束是什么？
**30秒答案**：端到端 <100ms：召回 <10ms，粗排 <20ms，精排 <50ms，竞价 <10ms。关键优化：模型蒸馏/量化、特征缓存、异步预计算。

### Q8: 广告和推荐的核心技术差异？
**30秒答案**：①校准要求不同（广告需绝对概率，推荐只需排序）；②约束不同（广告有预算/ROI 约束）；③更新频率不同（广告更高频）；④数据不同（广告有竞价日志）。

### Q9: 广告系统的数据闭环怎么做？
**30秒答案**：展示日志→点击/转化回收→特征构建→模型训练→线上服务。关键：①归因窗口设置（7天/30天）；②延迟转化处理；③样本权重修正；④在线学习增量更新。

### Q10: 广告系统如何处理数据稀疏问题？
**30秒答案**：①多任务学习（用 CTR 辅助 CVR）；②数据增广（LLM 生成/对比学习）；③迁移学习（从相似领域迁移）；④特征工程（高阶交叉特征增加信号密度）。
## 🌐 知识体系连接

- **上游依赖**：CTR/CVR 预估模型、校准理论、在线学习
- **下游应用**：出价策略、eCPM 计算、ROI 优化
- **相关 synthesis**：广告出价体系全景.md, 推荐系统排序范式演进.md
