# 广告效果归因、Uplift 建模与 LTV 预测

> 创建：2026-04-13 | 领域：广告系统 | 类型：综合分析
> 来源文件：
> - `ads/02_rank/synthesis/广告效果归因.md`
> - `ads/uplift/synthesis/Uplift建模技术演进与工业实践.md`
> - `ads/02_rank/synthesis/LTV预测技术演进与工业实践.md`

---

## 一、广告效果归因

### 1.1 归因的核心问题

用户看了搜索广告又看了展示广告最后买了，功劳该归谁？归因模型直接影响各渠道的预算分配。

**核心公式 — eCPM 排序**：

$$
eCPM = pCTR \times pCVR \times bid
$$

### 1.2 归因模型演进

| 模型 | 分配逻辑 | 优缺点 |
|------|---------|--------|
| Last-Click | 全部归最后点击 | 简单但偏向底部漏斗，品牌曝光被低估 |
| First-Click | 全部归首次接触 | 偏向顶部漏斗 |
| Linear | 均匀分配 | 忽略触点差异 |
| Time-Decay | 越近权重越大 | 中庸方案 |
| Position-Based | 首尾各 40%，中间均分 20% | 半启发式 |
| Data-Driven | Shapley Value / Markov Chain | 理论最优，计算复杂 |

### 1.3 Shapley Value 归因

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N|-|S|-1)!}{|N|!} \left[v(S \cup \{i\}) - v(S)\right]
$$

考虑每个触点在所有可能顺序中的边际贡献取平均。唯一满足公平性四条公理（对称性+边际贡献+效率+零贡献者归零）的归因方法，但计算复杂度 $O(2^n)$，工业中用 Monte Carlo 采样近似。

### 1.4 马尔可夫链归因

$$
P(\text{convert}) = \sum_{\text{paths}} \prod_{(i \to j) \in \text{path}} P(j|i)
$$

把用户触点序列建模为马尔可夫链，每个触点的归因份额等于"去掉这个触点后转化概率下降多少"。

### 1.5 增量归因（Incrementality）

$$
\text{Lift} = \frac{\text{CVR}_{exposed} - \text{CVR}_{holdout}}{\text{CVR}_{holdout}}
$$

通过设置 holdout 组（不看广告的对照组），只衡量广告真正带来的额外转化。这是归因的金标准，与 Uplift Modeling 思想一致。

### 1.6 跨渠道归因挑战

1. 用户身份跨设备打通（手机/电脑/平板的行为关联）
2. 不同渠道数据粒度不同（搜索有 query，展示只有曝光）
3. 隐私政策限制（GDPR/CCPA 限制用户追踪）

---

## 二、Uplift 建模：从"谁会买"到"谁因为看了广告才会买"

### 2.1 问题定义

传统 CTR 预测 $P(Y=1|X)$，但广告投放关心**增量效果**：用户是因为看了广告才转化，还是不投也会转化？

**用户四象限**：

| | 看广告会转化 | 看广告不转化 |
|---|---|---|
| **不看也转化** | Sure Things（自然转化）| Lost Causes |
| **不看不转化** | **Persuadables（目标）**| Sleeping Dogs（广告有害）|

### 2.2 因果框架

采用 Rubin Potential Outcomes Framework：

$$
\tau_i = Y_i(1) - Y_i(0)
$$

条件平均处理效应（CATE，Uplift 模型的核心目标）：

$$
\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

### 2.3 Meta-Learner 系列

| 方法 | 核心思路 | 优缺点 |
|------|---------|--------|
| **S-Learner** | 把 $T$ 当普通特征训练单模型，$\hat{\tau}(x) = \hat{\mu}(x,1) - \hat{\mu}(x,0)$ | 简单但 $T$ 信号易被 $X$ 淹没 |
| **T-Learner** | 实验组/对照组各训练独立模型，$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$ | 处理效应显式建模但方差大 |
| **X-Learner** | 三步法交叉残差，$\hat{\tau}(x) = g(x)\hat{d}_0(x) + (1-g(x))\hat{d}_1(x)$ | 适合样本不平衡（广告常见） |

**X-Learner 详解**：
- Step 1：分别训练 $\hat{\mu}_0, \hat{\mu}_1$
- Step 2：计算交叉伪 ITE：$\tilde{\tau}_1^{(i)} = Y_i^{(1)} - \hat{\mu}_0(x_i^{(1)})$
- Step 3：用伪 ITE 训练 uplift 模型，以倾向性得分 $g(x)$ 加权融合

### 2.4 端到端深度学习方法

**TarNet**：共享表示层 $\Phi(x)$ + 两个独立 outcome head $h_0, h_1$：

$$
\hat{Y}(t) = h_t(\Phi(x)), \quad \mathcal{L} = \frac{1}{N}\sum_i [T_i \cdot \ell(h_1(\Phi(x_i)), Y_i) + (1-T_i) \cdot \ell(h_0(\Phi(x_i)), Y_i)]
$$

**CFRNet**：TarNet + IPM 正则（MMD），强制表示空间中实验组/对照组分布对齐：

$$
\mathcal{L}_{\text{CFR}} = \mathcal{L}_{\text{TarNet}} + \alpha \cdot \text{MMD}^2(\{\Phi(x)\}_{T=1}, \{\Phi(x)\}_{T=0})
$$

核心直觉：如果表示空间里两组长得不一样，outcome head 无法公平做反事实预测。与 Domain Adaptation 思想一致。

**DragonNet**：TarNet + propensity score 预测头联合训练：

$$
\mathcal{L}_{\text{Dragon}} = \mathcal{L}_{\text{outcome}} + \beta \cdot \mathcal{L}_{\text{propensity}}
$$

能预测 propensity score 的表示包含了所有混杂因子信息，引导表示学到去混杂能力。

### 2.5 工业落地方法

**双塔 Uplift**：用户塔 + 物品塔 + treatment embedding：

$$
\hat{\tau}(x, item) = \langle u(x) \odot e_{\text{treat}}, v(item) \rangle - \langle u(x) \odot e_{\text{ctrl}}, v(item) \rangle
$$

物品侧可离线预计算存入 ANN 索引，线上延迟与普通双塔召回相当（微秒级）。

**DESCN**（字节跳动）：整体空间 + Cross 结构让两分支互相学习，类比 ESMM 的全空间思想。

**EUEN**（阿里）：多目标 uplift 框架，同时估计 CVR 增量和 Cost 增量，实现 ROI 最优投放：

$$
\text{score} = w_{\text{cvr}} \cdot \hat{\tau}_{\text{cvr}} + w_{\text{cost}} \cdot \hat{\tau}_{\text{cost}}
$$

### 2.6 偏差处理

**IPW（逆概率加权）**：

$$
\hat{\text{ATE}_{\text{IPW}} = \frac{1}{N}\sum_i \left[\frac{T_i Y_i}{\hat{e}(x_i)} - \frac{(1-T_i)Y_i}{1-\hat{e}(x_i)}\right]
$$

**Doubly Robust Estimator**（双重保险）：

$$
\hat{\tau}_{\text{DR}}(x) = \left[\hat{\mu}_1(x) + \frac{T(Y-\hat{\mu}_1(x))}{\hat{e}(x)}\right] - \left[\hat{\mu}_0(x) + \frac{(1-T)(Y-\hat{\mu}_0(x))}{1-\hat{e}(x)}\right]
$$

同时使用 outcome 模型和 propensity 模型，只要其中一个正确，估计就是一致的。

### 2.7 评估指标

**AUUC**：按 $\hat{\tau}$ 降序排列用户，前 $\phi$ 比例用户的累积增量效果曲线下面积。

**Qini Coefficient**：$Q = \text{AUUC}_{model} - \text{AUUC}_{random}$

---

## 三、LTV 预测

### 3.1 LTV 分布三大特征

1. **零膨胀**：70-90% 用户 LTV=0（从未购买）
2. **右偏重尾**：头部 5% 用户贡献 50-80% 总收入
3. **非负约束**：LTV 不可为负

### 3.2 技术演进路线

| 阶段 | 方法 | 特点 |
|------|------|------|
| 统计（2000s） | BG/NBD, Pareto/NBD | 可解释，不需要特征，适合小数据 |
| ML（2015-2018） | RFM + XGBoost | 两阶段：先分类再回归 |
| 深度学习（2019+） | **ZILN** (Google KDD'19) | 端到端联合建模零膨胀+对数正态 |
| 扩展（2021+） | Deep LTV + 序列建模 | 多时间窗口 + Transformer |
| 因果/RL | RL-based LTV 优化 | 长期 reward 建模 |

### 3.3 ZILN 模型（核心）

将 LTV 分布显式建模为两部分混合：

$$
P(Y=y) = \begin{cases} \pi(x) & y = 0 \\ (1-\pi(x)) \cdot \text{LogNormal}(y; \mu(x), \sigma^2) & y > 0 \end{cases}
$$

损失函数：

$$
\mathcal{L} = \underbrace{-\sum_i [y_i^{>0}\log(1-\hat{\pi}_i) + y_i^{=0}\log\hat{\pi}_i]}_{\text{Binary CE}} - \underbrace{\sum_{i:y_i>0} \log\phi\left(\frac{\log y_i - \hat{\mu}_i}{\hat{\sigma}_i}\right)}_{\text{LogNormal NLL}}
$$

预测：$E[\text{LTV}] = (1-\hat{\pi}) \times \exp(\hat{\mu} + \hat{\sigma}^2/2)$

$\sigma$ 的工业用途：不确定性度量 → 高 $\sigma$ 用户谨慎出价（悲观估计 $\exp(\mu - \beta\sigma)$），低 $\sigma$ 用户激进出价。

### 3.4 ZILN vs Tweedie 对比

| 维度 | ZILN | Tweedie |
|------|------|---------|
| 零膨胀处理 | 显式分离零/非零 | 参数 $p$ 自动调节 |
| 不确定性 | 输出 $\sigma$ | 无显式不确定性 |
| 解释性 | 强（两阶段语义清晰） | 统一分布，实现更简 |
| 工具支持 | 深度学习框架 | LightGBM/XGBoost 原生支持 |
| 推荐场景 | 深度学习模型 | 快速 baseline |

### 3.5 LTV 在广告出价中的应用

传统 tCPA → LTV 导向的 tROAS：

$$
\text{bid}_{\text{tCPA}} = \text{CPA}_{target} \times pCVR \quad \longrightarrow \quad \text{bid}_{\text{tROAS}} = \frac{E[\text{LTV}]}{\text{tROAS}}
$$

按 LTV 分层差异化出价：高 LTV 用户 bid × 2.5，中 × 1.2，低 × 0.7，零价值不参竞。

### 3.6 LTV 标签延迟处理

- **代理标签**：7日回访/首购金额等短期指标
- **多任务学习**：同时训练 7d/30d/90d LTV，早期窗口提供频繁监督
- **分段训练策略**：即时训 CTR/CVR → 7天微调 → 30天校准 → 三模型加权融合

### 3.7 冷启动 LTV 预测

| 方案 | 思路 |
|------|------|
| 注册行为特征 | 设备型号/地区/来源/注册时段 |
| 用户相似度 | Top-K 相似历史用户的 LTV 加权平均 |
| Early Signal 快速更新 | 前 24-72h 行为信号每小时刷新预测 |

---

## 四、三者的联系

```
归因（谁的功劳）→ 指导渠道预算分配
Uplift（投了是否有增量）→ 指导用户粒度的投放决策
LTV（用户长期价值）→ 指导出价金额

联合使用：bid = τ_uplift × E[LTV] × 渠道归因权重
```

**Uplift 融入出价**：$\text{bid} = \hat{\tau}_{\text{cvr}} \times \text{CPA}_{target}$，只对高 uplift 用户出高价，避免为"本来就会转化"的用户花钱。

---

## 五、面试高频 Q&A（15 题）

**Q1: 常见的归因模型有哪些？Shapley Value 归因的优势？**
Last-Click/First-Click/Linear/Time-Decay/Position-Based/Data-Driven 六种。Shapley Value 是唯一满足公平性四公理的方法，通过计算每个触点在所有可能组合中的边际贡献取平均。近似用 Monte Carlo 采样。

**Q2: 增量性测试（Lift Test）怎么做？**
随机分两组：实验组正常展示广告，对照组展示公益广告/不展示。两组转化率差异即广告增量效果。是唯一测量因果效应的方法。

**Q3: 为什么不能直接用实验组 CTR 减对照组 CTR 作为 uplift？**
直接做差只给 ATE，无法给个体化 CATE；且观测数据中两组用户分布往往不同（选择偏差），直接做差引入混杂偏差。

**Q4: S-Learner/T-Learner/X-Learner 怎么选？**
S-Learner 最简单但 T 信号易被淹没；T-Learner 方差大；X-Learner 适合样本不平衡场景（广告中对照组流量少），通过交叉残差借用大样本组信息。

**Q5: CFRNet 的 IPM 正则为什么有效？**
强制表示空间中实验/对照组分布对齐，使 outcome head 对反事实预测可信。与 Domain Adaptation 中 source/target domain 对齐思想一致。

**Q6: DragonNet 为什么加 propensity head？**
因果推断理论保证能预测 propensity score 的表示包含所有混杂因子信息。多任务学习引导表示自动去混杂。

**Q7: AUUC 和 AUC 的区别？**
AUC 评估 $P(Y=1)$ 排序质量，但 uplift 目标 $\tau = Y(1)-Y(0)$ 在个体层面不可观测。AUUC 在分组层面检验高 uplift 用户是否真的有更高增量效果。

**Q8: 双塔 uplift 怎么做线上推理？**
物品侧向量离线预计算存 ANN 索引，线上用户向量分别乘 $e_{\text{treat}}$ 和 $e_{\text{ctrl}}$ 做两次内积取差值，延迟微秒级。

**Q9: LTV 为什么用 ZILN 而非直接回归？**
LTV 零膨胀重尾分布：70-90% 为零，剩余右偏。ZILN 分开建模零/正值，Binary CE + LogNormal NLL 联合训练，比独立两阶段模型更好。

**Q10: ZILN 中 σ 参数的工业用途？**
表示预测不确定性。高 σ 用户谨慎出价（悲观估计 $\exp(\mu-\beta\sigma)$），用于 UCB 类探索-利用权衡。

**Q11: LTV 标签延迟怎么处理？**
代理标签（7d 回访）+ 多任务学习（7d/30d/90d 同时训练）+ 异步回填。关键保证 training/serving 时间对齐。

**Q12: ZILN vs Tweedie 怎么选？**
ZILN 显式分离零/非零，输出不确定性，适合深度学习；Tweedie 统一分布参数 $p$ 自动调节，LightGBM 原生支持，适合快速 baseline。

**Q13: Uplift 模型在出价中怎么用？**
$\text{bid} = \hat{\tau}_{\text{cvr}} \times \text{CPA}_{target}$，只对 Persuadables 出高价。比传统 $\text{bid} = \hat{p}_{\text{cvr}} \times \text{CPA}$ 更经济。

**Q14: 如何处理 Sleeping Dogs？**
Uplift 模型天然识别 $\hat{\tau}<0$ 的用户，对其不投广告。金融/保险场景尤为重要。

**Q15: DESCN 的"整体空间"思想和 ESMM 的联系？**
ESMM 在"曝光→点击→转化"全空间建模解决 CVR 样本选择偏差；DESCN 在"实验+对照"全空间建模避免只用实验组的选择偏差。核心一致：不在有偏子集上训练。

---

## 参考文献

1. Attribution Modeling 系列（Shapley Value, Markov Chain）
2. Johansson et al., "Learning Representations for Counterfactual Inference" (CFRNet)
3. Shi et al., "Adapting Neural Networks for the Estimation of Treatment Effects" (DragonNet)
4. Kunzel et al., "Meta-learners for Estimating Heterogeneous Treatment Effects" (X-Learner)
5. Faisal et al., "DESCN: Deep Entire Space Cross Networks" (ByteDance)
6. Google KDD 2019, "A Flexible Framework for Predicting Revenue" (ZILN)
7. Snap 2021, Deep LTV Multi-Task Model

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[synthesis/ads/01_CTR_CVR预估与校准全景|CTR/CVR 预估与校准]]
- [[synthesis/ads/05_竞价与预算优化|竞价与预算优化]]
