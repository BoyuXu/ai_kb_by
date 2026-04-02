# Uplift 建模技术演进与工业实践

> 从"谁会买"到"谁因为看了广告才会买"——因果推断视角下的广告增量效果建模

## 一、问题定义：为什么不能直接用 CTR 差

传统 CTR 模型预测的是 $P(Y=1|X)$——用户会不会转化。但广告投放关心的是**增量效果**：这个用户是**因为看了广告才转化**的，还是**不投广告也会转化**的？

### 用户四象限

| | 看广告会转化 | 看广告不转化 |
|---|---|---|
| **不看广告也转化** | Sure Things（自然转化）| Lost Causes（怎样都不转化）|
| **不看广告不转化** | **Persuadables（可说服）**| Sleeping Dogs（广告反而有害）|

Uplift Modeling 的目标：精准找到 **Persuadables**，避免把预算浪费在 Sure Things 上。

### 核心因果框架

采用 Rubin 因果模型（Potential Outcomes Framework）：

$$
\tau_i = Y_i(1) - Y_i(0)
$$

- $\tau_i$：个体 $i$ 的**个体处理效应**（Individual Treatment Effect, ITE）
- $Y_i(1)$：个体 $i$ 在**接受处理**（看广告）时的潜在结果
- $Y_i(0)$：个体 $i$ 在**未接受处理**（不看广告）时的潜在结果

**直观理解**：ITE 就是同一个人在"平行宇宙 A（看了广告）"和"平行宇宙 B（没看广告）"中行为的差异。现实中我们只能观察到其中一个宇宙，这就是因果推断的根本困难——**反事实不可观测**。

**群体层面的平均处理效应**：

$$
\text{ATE} = \mathbb{E}[\tau_i] = \mathbb{E}[Y(1) - Y(0)]
$$

- $\text{ATE}$：Average Treatment Effect，所有用户的平均增量效果

**条件平均处理效应（Uplift 建模的核心目标）**：

$$
\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

- $\text{CATE}(x)$：给定特征 $x$ 的用户子群的平均增量效果
- $X$：用户特征向量（年龄、历史行为、设备等）

**直观理解**：CATE 回答的问题是"对于长得像 $x$ 的这群用户，投广告平均能多带来多少转化"。Uplift 模型的本质就是学习 $\text{CATE}(x)$ 函数。

---

## 二、Meta-Learner 系列：用现有模型拼接因果估计

### 2.1 S-Learner（Single Model）

**思路**：把处理变量 $T$ 当作普通特征，训练一个模型。

$$
\hat{\tau}(x) = \hat{\mu}(x, T=1) - \hat{\mu}(x, T=0)
$$

- $\hat{\mu}(x, T)$：以 $(X, T)$ 为输入的回归模型预测值
- $T \in \{0, 1\}$：处理指示变量（1=看广告，0=未看）

**直观理解**：训练一个大模型，预测时分别代入"看了广告"和"没看广告"，两次预测的差值就是 uplift 估计。简单粗暴，但模型可能忽略 $T$ 的影响（特别是当 $T$ 的效应相对于 $X$ 的效应很小时）。

**优点**：实现最简单，一个模型搞定
**缺点**：$T$ 的信号容易被 $X$ 淹没；正则化可能抹平 $T$ 的效应

### 2.2 T-Learner（Two Models）

**思路**：对实验组和对照组分别训练独立模型。

$$
\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

- $\hat{\mu}_1(x)$：仅用实验组（$T=1$）数据训练的模型
- $\hat{\mu}_0(x)$：仅用对照组（$T=0$）数据训练的模型

**直观理解**：分别学"看了广告的人怎么转化"和"没看广告的人怎么转化"，然后做差。每个模型只看到一半数据，样本效率低；两个模型的误差会累加。

**优点**：处理效应被显式建模
**缺点**：两个模型独立训练，方差大；对照组通常样本少（广告场景中不投广告的流量是少数）

### 2.3 X-Learner（Cross Model）

**思路**：三步法，利用交叉残差估计减少偏差。

**Step 1**：分别训练 $\hat{\mu}_0, \hat{\mu}_1$（同 T-Learner）

**Step 2**：计算交叉残差（伪 ITE）

$$
\tilde{\tau}_1^{(i)} = Y_i^{(1)} - \hat{\mu}_0(x_i^{(1)}), \quad \tilde{\tau}_0^{(j)} = \hat{\mu}_1(x_j^{(0)}) - Y_j^{(0)}
$$

- $\tilde{\tau}_1^{(i)}$：实验组个体 $i$ 的观测结果减去对照组模型的反事实预测
- $\tilde{\tau}_0^{(j)}$：实验组模型对对照组个体 $j$ 的反事实预测减去实际观测

**Step 3**：用伪 ITE 训练两个 uplift 模型 $\hat{d}_1(x), \hat{d}_0(x)$，加权融合：

$$
\hat{\tau}(x) = g(x) \cdot \hat{d}_0(x) + (1 - g(x)) \cdot \hat{d}_1(x)
$$

- $g(x) \in [0,1]$：倾向性得分（propensity score），$g(x) = P(T=1|X=x)$

**直观理解**：X-Learner 的核心洞察是"用一个组的模型去补全另一个组的反事实"。当实验组和对照组样本量极不平衡时（广告场景常见），它能借助样本多的组的信息来改善样本少的组的估计。

---

## 三、端到端深度学习方法

### 3.1 TarNet（Treatment-Agnostic Representation Network）

**架构**：共享表示层 $\Phi(x)$ + 两个独立 outcome head $h_0, h_1$。

$$
\hat{Y}(t) = h_t(\Phi(x)), \quad t \in \{0, 1\}
$$

$$
\mathcal{L}_{\text{TarNet}} = \frac{1}{N} \sum_{i=1}^{N} \left[ T_i \cdot \ell(h_1(\Phi(x_i)), Y_i) + (1-T_i) \cdot \ell(h_0(\Phi(x_i)), Y_i) \right]
$$

- $\Phi(x)$：共享的特征表示网络，将原始特征映射到低维空间
- $h_t$：第 $t$ 个处理分支的 outcome 预测头
- $\ell(\cdot, \cdot)$：损失函数（分类用 BCE，回归用 MSE）

**直观理解**：底层学一个通用的用户表示，顶层分叉成两个"平行宇宙预测器"。问题在于共享表示可能把实验组和对照组的分布差异编码进去，导致估计偏差。

### 3.2 CFRNet（Counterfactual Regression Network）

**改进**：在 TarNet 基础上加 IPM（Integral Probability Metric）正则项，强制表示空间中实验组和对照组分布对齐。

$$
\mathcal{L}_{\text{CFR}} = \mathcal{L}_{\text{TarNet}} + \alpha \cdot \text{IPM}(\{\Phi(x_i)\}_{T_i=1}, \{\Phi(x_j)\}_{T_j=0})
$$

常用 IPM 选择 —— **MMD（Maximum Mean Discrepancy）**：

$$
\text{MMD}^2 = \left\| \frac{1}{N_1}\sum_{T_i=1} \phi(\Phi(x_i)) - \frac{1}{N_0}\sum_{T_j=0} \phi(\Phi(x_j)) \right\|_{\mathcal{H}}^2
$$

- $\alpha$：正则化强度，控制分布对齐与预测精度的权衡
- $\phi(\cdot)$：再生核希尔伯特空间（RKHS）中的特征映射
- $N_1, N_0$：实验组和对照组的样本数
- $\|\cdot\|_{\mathcal{H}}$：RKHS 范数

**直观理解**：CFRNet 的核心直觉是"如果表示空间里实验组和对照组长得不一样，那 outcome head 就没法公平地做反事实预测"。IPM 正则项强制让两组在表示空间中"混在一起"，从而使得 $h_1$ 对对照组的预测也是可信的。这和 Domain Adaptation 的思想完全类似——把"实验/对照"看作两个 domain。

### 3.3 DragonNet

**改进**：在 TarNet 基础上加入 propensity score 预测头，联合训练。

$$
\mathcal{L}_{\text{Dragon}} = \mathcal{L}_{\text{outcome}} + \beta \cdot \mathcal{L}_{\text{propensity}}
$$

$$
\mathcal{L}_{\text{propensity}} = -\frac{1}{N}\sum_{i=1}^{N}\left[T_i \log \hat{e}(x_i) + (1-T_i)\log(1-\hat{e}(x_i))\right]
$$

- $\hat{e}(x_i) = \sigma(g(\Phi(x_i)))$：倾向性得分预测头的输出
- $\beta$：propensity loss 的权重
- $\sigma$：sigmoid 函数

**直观理解**：DragonNet 的关键洞察来自因果推断理论——如果表示 $\Phi(x)$ 能够准确预测 propensity score，那么它就包含了所有混杂因子的信息，足以做无偏的因果效应估计。通过多任务学习同时预测"谁会被投广告"和"投了会怎样"，让表示自动学到去混杂的能力。

---

## 四、工业实践方法

### 4.1 双塔 Uplift（广告/推荐工业落地）

**动机**：TarNet/CFRNet 推理时需要分别过两个 head 取差值，无法预计算 + 向量检索。

**架构**：用户塔 $u(x)$、物品塔 $v(item)$，加入 treatment embedding $e_t$：

$$
\hat{\tau}(x, item) = \langle u(x), v(item) \rangle_{T=1} - \langle u(x), v(item) \rangle_{T=0}
$$

工程上常简化为：

$$
\hat{\tau}(x, item) = \langle u(x) \odot e_{\text{treat}}, v(item) \rangle - \langle u(x) \odot e_{\text{ctrl}}, v(item) \rangle
$$

- $\odot$：逐元素乘法（Hadamard product）
- $e_{\text{treat}}, e_{\text{ctrl}}$：treatment/control 的可学习 embedding

**直观理解**：把 treatment 信息编码为 embedding 注入双塔，推理时可以预计算物品侧向量，线上只需做两次内积取差值。延迟从毫秒级模型推理降到微秒级向量运算。

### 4.2 DESCN（Deep Entire Space Cross Networks, 字节跳动）

**核心思想**：在整体样本空间上建模，避免样本选择偏差。引入 cross 结构让 treatment 和 control 分支共享信息。

$$
\hat{Y}(t, x) = h_t(\Phi_{\text{shared}}(x) \oplus \Phi_{\text{cross}}^{t \to 1-t}(x))
$$

- $\Phi_{\text{shared}}(x)$：共享底层表示
- $\Phi_{\text{cross}}^{t \to 1-t}(x)$：从另一个 treatment 分支交叉传递的信息
- $\oplus$：拼接操作

**改进点**：
1. 整体空间建模（类比 ESMM 的思想）
2. Cross 结构允许两个分支互相学习，缓解对照组样本不足问题
3. 引入 propensity score 做逆概率加权（IPW）去偏

### 4.3 EUEN（End-to-End User Response Estimation Network, 阿里巴巴）

**核心思想**：将 uplift 建模融入推荐系统的多目标优化框架。

$$
\text{score}(x, item) = w_{\text{cvr}} \cdot \hat{\tau}_{\text{cvr}}(x, item) + w_{\text{cost}} \cdot \hat{\tau}_{\text{cost}}(x, item)
$$

- $\hat{\tau}_{\text{cvr}}$：转化率的增量效果
- $\hat{\tau}_{\text{cost}}$：成本的增量效果
- $w_{\text{cvr}}, w_{\text{cost}}$：业务权重

**直观理解**：不只看广告能多带来多少转化，还要看多花了多少钱。通过多目标 uplift 估计，实现 ROI 最优的投放策略。

---

## 五、因果图方法：ITE 估计的因果推断视角

### 结构因果模型（SCM）

$$
Y = f(X, T, U), \quad T = g(X, V)
$$

- $U, V$：不可观测的噪声变量
- $f, g$：结构方程

**do-calculus 视角下的 ATE**：

$$
\text{ATE} = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]
$$

$$
= \sum_x \left[\mathbb{E}[Y \mid T=1, X=x] - \mathbb{E}[Y \mid T=0, X=x]\right] P(X=x)
$$

- $do(T=t)$：Pearl 的 do 算子，表示"干预设置 $T=t$"（切断所有指向 $T$ 的因果箭头）
- 第二行成立的条件：**无未观测混杂**（后门准则满足）

**直观理解**：$\mathbb{E}[Y|T=1]$ 和 $\mathbb{E}[Y|do(T=1)]$ 的区别在于前者是"观察到看了广告的人的转化率"（混杂了用户本身的特性），后者是"如果强制所有人都看广告的转化率"（因果效应）。后门调整通过条件化 $X$ 消除混杂。

---

## 六、评估指标

### 6.1 AUUC（Area Under Uplift Curve）

由于 ITE 不可直接观测，传统 AUC 无法使用。AUUC 的计算方式：

$$
\text{AUUC} = \int_0^1 f(\phi) \, d\phi
$$

其中 $f(\phi)$ 是 uplift curve：将用户按 $\hat{\tau}$ 降序排列，在前 $\phi$ 比例的用户中计算累积增量效果：

$$
f(\phi) = \frac{1}{N} \left( \frac{\sum_{i \in \text{top-}\phi, T_i=1} Y_i}{|\{i \in \text{top-}\phi, T_i=1\}|} - \frac{\sum_{j \in \text{top-}\phi, T_j=0} Y_j}{|\{j \in \text{top-}\phi, T_j=0\}|} \right) \cdot N \cdot \phi
$$

**直观理解**：类似 ROC 曲线的思想——如果模型完美排序，先触达的用户应该有最大的增量效果。AUUC 越大，模型越能把"可说服"的用户排在前面。

### 6.2 Qini Coefficient

$$
Q = \text{AUUC}_{\text{model}} - \text{AUUC}_{\text{random}}
$$

**直观理解**：Qini 系数衡量模型相对于随机投放的提升。随机投放的 uplift curve 是一条直线（因为按随机顺序触达用户，累积增量效果线性增长）。

---

## 七、实验偏差处理

### 7.1 选择偏差

观测数据中 $P(T=1|X)$ 不均匀（高价值用户更可能被投广告）。

**IPW（逆概率加权）校正**：

$$
\hat{\text{ATE}}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{T_i Y_i}{\hat{e}(x_i)} - \frac{(1-T_i) Y_i}{1 - \hat{e}(x_i)} \right]
$$

- $\hat{e}(x_i)$：估计的倾向性得分

**直观理解**：高价值用户被投广告的概率高，所以实验组中高价值用户过多。IPW 通过"给低概率被选中的样本加大权重"来还原随机分配的效果，类似调查统计中的加权采样。

### 7.2 Doubly Robust Estimator

$$
\hat{\tau}_{\text{DR}}(x) = \left[\hat{\mu}_1(x) + \frac{T(Y - \hat{\mu}_1(x))}{\hat{e}(x)}\right] - \left[\hat{\mu}_0(x) + \frac{(1-T)(Y - \hat{\mu}_0(x))}{1-\hat{e}(x)}\right]
$$

**直观理解**：同时使用 outcome 模型 $\hat{\mu}$ 和 propensity 模型 $\hat{e}$，只要其中一个是正确的，估计就是一致的（consistent）。"双重保险"——两个模型互相兜底。

---

## 八、面试高频题

### Q1: 为什么不能直接用实验组 CTR 减去对照组 CTR 作为 uplift？
**答**：直接做差只给出 ATE（总体平均），无法给出个体化的 CATE。而且观测数据中实验组和对照组的用户分布往往不同（选择偏差），直接做差会引入混杂偏差。Uplift 模型的价值在于估计异质性处理效应——哪些人 uplift 高、哪些人低。

### Q2: S-Learner 和 T-Learner 各有什么问题？什么时候用 X-Learner？
**答**：S-Learner 容易忽略 treatment 效应（$T$ 的信号被 $X$ 淹没）；T-Learner 两个模型独立训练，方差大且无法利用另一组的信息。X-Learner 适合**实验组和对照组样本量极不平衡**的场景（广告中常见——不投广告的流量远少于投的），它通过交叉残差借用大样本组的信息。

### Q3: CFRNet 的 IPM 正则项为什么有效？和 Domain Adaptation 的关系？
**答**：IPM 强制表示空间中实验组和对照组分布对齐，使得 outcome head 对反事实的预测是可信的（类似 Domain Adaptation 中 source 和 target domain 对齐）。本质都是解决**分布偏移**下的泛化问题。

### Q4: AUUC 和 AUC 的区别？为什么 uplift 不能用 AUC？
**答**：AUC 评估的是预测 $P(Y=1)$ 的排序质量，但 uplift 模型预测的是 $\tau = Y(1) - Y(0)$，而 $\tau$ 在个体层面不可观测（只能观测到 $Y(1)$ 或 $Y(0)$ 之一）。因此需要 AUUC：按 $\hat{\tau}$ 排序后，在分组层面检验高 uplift 用户是否真的有更高的增量效果。

### Q5: DragonNet 为什么要加 propensity head？不加行不行？
**答**：根据因果推断理论（sufficiency of propensity score），能预测 propensity score 的表示包含了所有混杂因子信息。加 propensity head 是一种正则化，引导表示学到去混杂的信息，从而使 CATE 估计更准。实验表明比纯 TarNet 显著降低偏差。

### Q6: 工业中对照组流量不足怎么办？
**答**：(1) 用 X-Learner 借助实验组信息；(2) IPW 加权校正；(3) 利用历史 RCT 数据做 pre-training；(4) 迁移学习：从其他场景（如 push 通知实验）迁移 uplift 模型；(5) DESCN 的 cross 结构让两个分支互相学习。

### Q7: 双塔 uplift 怎么做线上推理？延迟如何？
**答**：物品侧向量可离线预计算并存入 ANN 索引。线上只需生成用户向量，分别乘以 $e_{\text{treat}}$ 和 $e_{\text{ctrl}}$ 做两次内积取差值。延迟和普通双塔召回相当（微秒级），远低于 TarNet 需要过两次深度网络的方案。

### Q8: 如何处理 Sleeping Dogs（广告有害的用户）？
**答**：Uplift 模型天然能识别 $\hat{\tau} < 0$ 的用户（Sleeping Dogs）。实践中：(1) 对 $\hat{\tau} < 0$ 的用户不投广告；(2) 设置 uplift 阈值而非只投 top-K；(3) 在金融/保险场景尤为重要——错误营销可能导致用户退订。

### Q9: DESCN 的"整体空间"思想和 ESMM 的联系？
**答**：ESMM 通过在"曝光→点击→转化"整体空间上建模来解决 CVR 的样本选择偏差。DESCN 类似地在"实验+对照"整体空间上建模，避免只用实验组数据导致的选择偏差。核心思想一致：**不要在有偏的子集上训练，而要在整体空间上训练**。

### Q10: Uplift 模型在广告出价中怎么用？
**答**：将 $\hat{\tau}$ 融入出价公式：$\text{bid} = \hat{\tau}_{\text{cvr}} \times \text{CPA}_{\text{target}}$。只对高 uplift 用户出高价，低 uplift 用户降价或不竞标。这比传统的 $\text{bid} = \hat{p}_{\text{cvr}} \times \text{CPA}$ 更经济——避免为"本来就会转化"的用户花钱。

### Q11: 线上 A/B 测试如何验证 uplift 模型？
**答**：(1) 设置随机对照实验（RCT），按 uplift 模型打分分层，检验高分层的增量效果是否显著高于低分层；(2) 比较"全量投放" vs "按 uplift 排序投放 top-K"的 ROI；(3) 用 Qini coefficient 量化模型带来的投放效率提升。

### Q12: 因果森林（Causal Forest）和深度学习方法怎么选？
**答**：因果森林（基于 GRF）适合特征维度中等、需要统计推断（置信区间）的场景；深度学习方法（CFRNet/DragonNet/DESCN）适合高维特征（embedding）、大规模工业数据。实践中常用因果森林做 baseline 和可解释性分析，深度方法做线上部署。
