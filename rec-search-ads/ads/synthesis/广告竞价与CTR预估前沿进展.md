# 广告竞价与CTR预估前沿进展（2024-2025）
> 综合总结 | 领域：ads | 学习日期：20260401 | 更新：20260402

---

## 📚 参考文献

1. **BiCB** — Lightweight Auto-bidding based on Traffic Prediction in Live Advertising  
   arxiv: https://arxiv.org/abs/2508.06069 | stat.ML/cs.LG | Aug 2025

2. **DHEN-CVR** — On the Practice of Deep Hierarchical Ensemble Network for Ad CVR Prediction  
   arxiv: https://arxiv.org/abs/2504.08169 | **WWW 2025** (Meta) | Apr 2025

3. **CDP** — Adaptive User Interest Modeling via Conditioned Denoising Diffusion For CTR Prediction  
   arxiv: https://arxiv.org/abs/2509.19876 | cs.IR | Sep 2025

4. **GenCO** — Generative Modeling with Multi-Instance Reward Learning for E-commerce Creative Optimization  
   arxiv: https://arxiv.org/abs/2508.09730 | cs.LG | Aug 2025

5. **GenAuction** — A Generative Auction for Online Advertising  
   AAAI 2025, Proceedings 39(12):12372-12380 | https://doi.org/10.1609/aaai.v39i12.33348 (Baidu)

6. **SFG** — From Feature Interaction to Feature Generation: A Generative Paradigm of CTR Prediction Models  
   arxiv: https://arxiv.org/abs/2512.14041 | cs.IR | Dec 2025

7. **BundleNet** — Optimal Auction Design in the Joint Advertising  
   arxiv: https://arxiv.org/abs/2507.07418 | **ICML 2025** | Jul 2025

8. **LPA** — Beyond Advertising: Mechanism Design for Platform-Wide Marketing Service "QuanZhanTui"  
   arxiv: https://arxiv.org/abs/2507.02931 | **KDD 2025** (Alibaba) | Jul 2025

9. **MAB-ColdStart** — Optimizing Online Advertising with Multi-Armed Bandits: Mitigating the Cold Start Problem under Auction Dynamics  
   arxiv: https://arxiv.org/abs/2502.01867 | cs.LG | Feb 2025

10. **QARM** — Quantitative Alignment Multi-Modal Recommendation at Kuaishou  
    arxiv: https://arxiv.org/abs/2411.11739 | cs.IR (Kuaishou) | Nov 2024

11. **CausalBidding** — Causal Bidding: Counterfactual Reasoning for Value-based Bid Optimization  
    arxiv: https://arxiv.org/abs/2502.14123 | cs.LG | Feb 2025

12. **KBD** — Knowledge-informed Bidding with Dual-process Control for Online Advertising  
    arxiv: https://arxiv.org/abs/2603.04920 | cs.LG | Mar 2026

13. **AutoBidMARL** — AutoBidding with Multi-agent Reinforcement Learning in Display Advertising  
    arxiv: https://arxiv.org/abs/2501.12399 | cs.LG | Jan 2025

---

## 🗺️ 全局技术图谱

这10篇论文覆盖广告系统的四大核心模块：

```
广告系统技术前沿
├── 🎯 CTR/CVR 预估
│   ├── SFG      — 判别式→生成式范式转换（特征生成）
│   ├── CDP      — 扩散模型去噪用户兴趣建模
│   ├── DHEN-CVR — 层次集成网络+多任务+自监督（Meta WWW'25）
│   └── QARM     — 量化多模态对齐（快手）
│
├── 🎨 广告创意优化
│   └── GenCO    — 生成模型+多实例奖励学习（电商创意组合）
│
├── 💰 竞价机制设计
│   ├── GenAuction  — 生成式拍卖（AAAI'25 Baidu）
│   ├── BundleNet   — 联合广告神经竞价（ICML'25）
│   └── LPA         — 液态支付全站推广（KDD'25 Alibaba）
│
└── 🚀 自动出价
    ├── BiCB          — 直播广告轻量实时出价
    ├── MAB-CS        — 多臂老虎机解决冷启动
    ├── CausalBidding — 因果图+反事实推理消除竞价偏差
    ├── KBD           — 知识注入+双过程控制（快/慢系统）
    └── AutoBidMARL   — 多智能体RL出价博弈均衡
```

---

## 📐 核心公式与推导

### 📐 Auto-bidding（自动出价）最优出价公式推导

**核心目标函数：**

最大化预期转化数，受预算约束：

$$
\max_{\{\text{bid}(a)\}} \sum_{a \in \mathcal{A}} \mathbb{E}[C_a \cdot r_a] \quad \text{s.t.} \quad \sum_{a \in \mathcal{A}} \text{bid}(a) \cdot r_a \leq B
$$

其中 $C_a$ 是广告 $a$ 的转化价值，$r_a$ 是获胜率（赢得拍卖的概率），$B$ 是总预算。

**推导步骤：**

1. **建立拉格朗日函数**：引入拉格朗日乘子 $\lambda^*$（影子价格），表示预算约束的边际成本：

$$
\mathcal{L} = \sum_{a} C_a \cdot r_a(\text{bid}(a)) - \lambda^* \left(\sum_a \text{bid}(a) \cdot r_a - B\right)
$$

2. **对 $\text{bid}(a)$ 求偏导并令其为0**（一阶最优性条件）：

$$
\frac{\partial \mathcal{L}}{\partial \text{bid}(a)} = C_a \cdot \frac{\partial r_a}{\partial \text{bid}} - \lambda^* r_a - \lambda^* \text{bid}(a) \frac{\partial r_a}{\partial \text{bid}} = 0
$$

3. **简化得到最优出价形式**：

$$
\text{bid}^*(a) = \lambda^* \cdot C_a \cdot \text{pCTR}(a)
$$

   其中 $\text{pCTR}(a)$ 是预估点击率，$C_a$ 是单次点击的预期转化价值。

4. **确定 $\lambda^*$**：通过二分搜索找到使预算约束紧（$\sum_a \text{bid}^*(a) \cdot r_a = B$）的 $\lambda^*$ 值。BiCB 的创新：结合流量预测 $\hat{r}_a(t)$ 实现**时间自适应出价**：

$$
\text{bid}(a, t) = \lambda^*(t) \cdot C_a \cdot \text{pCTR}(a) \cdot \frac{\hat{r}_a(t)}{\hat{r}_a(\text{day})}
$$

   即根据预测的分钟级流量，调整出价以平衡全天预算。

**符号说明：**

| 符号 | 含义 |
|------|------|
| $\lambda^*$ | 拉格朗日乘子（影子价格），含义：预算多1元能增加的预期收益 |
| $\text{bid}(a)$ | 广告 $a$ 的出价金额 |
| $\text{pCTR}(a)$ | 预估点击率（$p$ = predicted），由 CTR 模型输出 |
| $C_a$ | 单次点击的预期转化价值（eCPA 的倒数）|
| $r_a$ | 获胜率（竞争函数），取决于出价与其他对手出价的相对关系 |
| $B$ | 总预算约束 |

**直观理解：** 自动出价的核心思想是「预算梯度下降」——用影子价格 $\lambda^*$ 衡量每块钱的价值，高价值广告（转化价值高 $\times$ 点击率高）得到更高出价。流量预测的加入使出价随时间自适应，避免"上午花完、下午没钱"的浪费。

---

### 📐 UCB 置信上界公式推导（MAB 冷启动）

---

### 公式2：CTR 预估标准模型（eCPM 排序）

广告系统中的排序评分（Effective CPM）：

$$
\text{eCPM}(a) = \text{bid}(a) \times \text{pCTR}(a) \times \text{Quality}(a)
$$

**SFG** 改进了 $\text{pCTR}$ 的特征表示：引入监督特征生成损失

$$
\mathcal{L}_{SFG} = \underbrace{\mathcal{L}_{CTR}_{点击预测}} + \lambda \underbrace{\mathcal{L}_{gen}_{特征重建}}
$$

**CDP** 用扩散模型改进用户兴趣表示：

$$
\text{pCTR} = f\left(\text{user}_{interest}_\text{purified}, \text{item}, \text{context}\right)
$$

$$
\text{user}_{interest}_\text{purified} = \text{Denoise}(z_T, c) = p_\theta(z_0 | z_T, c)
$$

---

### 📐 UCB 多臂老虎机（MAB）冷启动推导

**核心问题：** 在不了解真实 CTR 的情况下，如何平衡探索（试新广告）和利用（用已知 CTR 高的广告）？

**UCB 优化目标：**

最大化 $T$ 步内的累积奖励，同时最小化遗憾（即与最优策略的差距）：

$$
\text{Regret}(T) = T \cdot \mu^* - \mathbb{E}\left[\sum_{t=1}^T r_{a_t}(t)\right]
$$

其中 $\mu^* = \max_a \mu_a$ 是最优臂的期望奖励。

**推导步骤：**

1. **问题建模**：每个广告 $a$ 有未知的真实 CTR $\mu_a$，每次展示时观测到 Bernoulli 奖励（点击=1，无点击=0）

$$
r_a(t) \sim \text{Bernoulli}(\mu_a)
$$

2. **经验估计与置信区间**：基于已有的 $n_a$ 次样本，估计平均 CTR：

$$
\hat{\mu}_a = \frac{1}{n_a}\sum_{i=1}^{n_a} r_{a,i}
$$
   
   由 Hoeffding 不等式，真实 $\mu_a$ 以高概率落在置信区间内：

$$
\mathbb{P}(\mu_a \leq \hat{\mu}_a + \sqrt{\frac{\log(1/\delta)}{2n_a}}) \geq 1 - \delta
$$

3. **UCB 上界**：选择最乐观的可能估计（置信上界）：

$$
\text{UCB_{a(t) = }\hat{\mu}}_a + \sqrt{\frac{2\log t}{n_a}}
$$
   
   **关键洞察**：$\sqrt{\frac{\log t}{n_a}}$ 项在样本少（$n_a$ 小）时很大，鼓励探索新臂。

4. **贪心选择与遗憾界**：每步选择 UCB 最高的臂：

$$
a_t^* = \arg\max_a \text{UCB}_{\text{a(t)
$$
   
   理论上界（Lai-Robbins）：

$$
\mathbb{E}}[\text{Regret}(T)] = O\left(\log T \sum_{a: \mu_a < \mu^*} \frac{1}{\text{KL}(\mu_a \| \mu^*)}\right)
$$

5. **在 CTR 预估中的应用（MAB-ColdStart）**：对新广告，直接用 UCB 调整 CTR：

$$
\text{pCTR}_{UCB}(a) = \min\left(\hat{\text{pCTR}(a) + \sqrt{\frac{2\log t}{n_a}}, 1.0\right)
$$
   
   竞价评分变为：

$$
\text{eCPM}_{UCB}(a) = \text{bid}(a) \times \text{pCTR}_{UCB}(a)
$$

**符号说明：**

| 符号 | 含义 |
|------|------|
| $\mu_a$ | 广告 $a$ 的真实点击率（未知） |
| $\hat{\mu}_a$ | 基于 $n_a$ 个样本的经验估计 CTR |
| $n_a$ | 广告 $a$ 已获得的展示次数 |
| $\text{UCB_{a(t)}$ | 第 $t$ 步的置信上界（乐观估计） |
| $t$ | 当前时间步（总展示次数） |
| $\text{KL}(\cdot \| \cdot)$ | Kullback-Leibler 散度，衡量两分布距离 |

**直观理解：** UCB 是「知识的价格」——新广告因为不确定性大（置信区间宽），被给予更高的"乐观评分"，从而获得更多探索机会。这种「不确定性驱动的探索」比固定探索率（如 $\epsilon$-greedy 的固定 $\epsilon$）更聪明：随着 $t$ 增大、$n_a$ 增大，置信区间收窄，探索逐渐转向利用。

---

### 公式4：激励相容拍卖中的 Myerson 支付规则

满足 DSIC 的支付规则（Myerson 公式）：

$$
\text{payment}_{\text{i = b_{i }}\cdot \pi}_{i(b) - \int}_{\text{0^{b}_{i}} \pi_i(t, b_{-i}) dt
$$

**BundleNet** 和 **LPA** 均以此为基础设计支付规则。**LPA** 的液态支付扩展：

$$
\text{payment}^{LPA}_i = g\left(W_{\text{liquid},i}, \pi_i\right), \quad W_{\text{liquid}} = \min\left(v, \frac{b}{p}\right)
$$

---

### 公式5：向量量化（QARM 核心）

多模态特征量化：

$$
z_q = e_{k^*}, \quad k^* = \arg\min_j \|z - e_j\|_2^2
$$

联合优化损失（直通估计器使梯度可传播）：

$$
\mathcal{L}_{QARM} = \mathcal{L}_{rec}(z_q) + \alpha \|z - \text{sg}(z_q)\|^2 + \beta \|\text{sg}(z) - z_q\|^2
$$

---

### 📐 因果竞价核心公式（CausalBidding）

**问题**：历史出价数据存在选择偏差——只观测到实际出价 $b_i$ 下的结果，无法知道"如果出价不同会怎样"。

**因果图建模**（DAG）：

$$
B_i \rightarrow \text{Win_{i }\rightarrow \text{Click}}_i \rightarrow \text{Conversion}_{\text{i
$$

出价 $B_{i}}$ 影响曝光赢得概率，进而影响点击和转化，形成有向因果链。

**IPW（逆倾向加权）去偏**：

历史数据下的行为策略为 $\pi}_{b(b}_{\text{i|x_{i)}}$，目标策略为 $\pi}_{e$，用重要性权重校正分布偏移：

$$
\hat{V}}^{IPW}(\pi_e) = \frac{1}{n}\sum_{i=1}^n \frac{\pi_e(b_i|x_i)}{\pi_b(b_i|x_i)} r_i
$$

**Double-Robust 估计**（同时利用模型预测和 IPW，任一正确即无偏）：

$$
\hat{V}^{DR}(\pi_e) = \frac{1}{n}\sum_i \left[\hat{r}(x_i, b_i) + \frac{\pi_e(b_i|x_i)}{\pi_b(b_i|x_i)}\left(r_i - \hat{r}(x_i, b_i)\right)\right]
$$

**工程意义**：IPW 在倾向分数极端（$\pi_b \approx 0$）时方差爆炸；DR 只要 $\hat{r}$ 或 $\pi_b$ 其中一个正确就能无偏，工业界优先选 DR。

---

### 📐 KBD 双过程控制框架

类比认知科学的 **System 1/System 2**（丹尼尔·卡内曼双系统理论）：

```
广告出价双过程控制
├── System 1（快）：规则/先验 → 毫秒级出价响应
│   └── 知识 Embedding 注入：历史竞价分布、时段模式、竞争格局
└── System 2（慢）：RL 深度优化 → 分钟级策略调整
    └── 预算约束 Lagrangian 松弛：λ 自适应调整
```

**Lagrangian 松弛出价公式**：

$$
\text{bid}^{KBD}(x) = \underbrace{f_\theta(x, e_k)}_{\text{RL策略}} \cdot \underbrace{g(\lambda)}_{\text{预算系数}}
$$

其中 $e_k$ 是知识 Embedding，$\lambda$ 随预算消耗率在线更新：

$$
\lambda_{t+1} = \lambda_t + \eta \cdot \left(\frac{\text{实际消耗}}{\text{目标消耗}} - 1\right)
$$

---

### 📐 AutoBidMARL：多智能体RL纳什均衡

**博弈建模**：$N$ 个广告主各自运行策略 $\pi_i$，竞价环境是 $N$ 人随机博弈：

$$
\max_{\pi_i} \mathbb{E}\left[\sum_t \gamma^t r_i(s_t, a_t^1,...,a_t^N)\right], \quad \forall i \in [N]
$$

**CTDE（中心化训练-去中心化执行）**：

- 训练时：Critic 使用全局状态 $s = (x^1,...,x^N)$ 估计 $Q(s, a^1,...,a^N)$  
- 执行时：Actor 仅用本地观测 $x^i$ 决策 $a^i = \pi_i(x^i)$

**纳什均衡条件**：在均衡点，任何单个广告主单方面改变策略都不能提升收益：

$$
V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i', \pi_{-i}^*), \quad \forall \pi_i', \forall i
$$

---

## 🎓 常见Q&A（≥10条）

### Q1: 广告 CTR 预估和推荐系统 CTR 的主要区别？

**A**: 广告 CTR 预估有更严格的业务约束：（1）**实时性要求更高**（ms 级延迟）；（2）**数据更稀疏**（CTR 通常 < 2%，CVR 更低）；（3）**竞价机制耦合**（CTR 影响出价评分，出价影响曝光，形成反馈环）；（4）**多目标优化**（平台收入 + 广告主 ROI + 用户体验三方博弈）；（5）**创意多样性**（同一广告主有多种创意需要优化）。

---

### Q2: GSP（广义第二价格）为什么不是最优机制？

**A**: GSP 不满足激励相容性（DSIC）——广告主存在策略性出价动机（bid shading：出价低于真实价值以降低支付）。这导致：（1）广告主花精力优化出价策略而非产品质量；（2）平台无法通过出价真实判断广告价值；（3）分配可能不是社会最优的。VCG 满足 DSIC 但计算复杂，且暗示激励问题（组合出价时某些广告主收益减少）。

---

### Q3: 什么是延迟反馈（Delayed Feedback）？CVR 预估如何处理？

**A**: 延迟反馈是指用户的转化行为（购买、注册等）发生在点击后数小时乃至数天，导致实时训练时标签不完整。处理方案：（1）**归因窗口**：设定固定时间窗口（如7天），窗口内的转化才计入正例；（2）**EM 算法**：建模延迟时间分布，用期望最大化估计真实 CVR；（3）**实时回填**：新的转化数据实时更新历史样本的标签；（4）**辅助任务**：用更密集的中间事件（加购、收藏）辅助 CVR 预估（DHEN-CVR 的做法）。

---

### Q4: 自动出价（Auto-bidding）系统的核心组成部分？

**A**: 
- **预算管理器**：控制每日/每时段的预算分配节奏
- **出价策略**：根据目标（CPM/CPC/CPA/ROI）计算每次请求的出价金额
- **流量预测**：估计未来流量分布，指导当前出价调整（BiCB 的核心）
- **反馈控制**：实时监控实际 CPA/ROI 与目标的差距，动态调整出价
- **冷启动处理**：新广告/新广告主的初始出价策略

---

### Q5: 多模态特征在广告系统中的挑战和解决方案？

**A**: 
**挑战**：（1）表示不匹配（多模态模型与推荐目标不一致）；（2）特征不可更新（缓存的固定 Embedding 无法随推荐任务迭代）；（3）推理延迟（大模型特征提取慢）。  
**解决方案**（QARM 等）：（1）向量量化：将连续多模态特征离散化为可训练 ID；（2）联合对齐损失：多模态重建损失 + 推荐任务监督损失联合优化；（3）离线预计算：量化 ID 离线存储，在线查表，零额外延迟。

---

### Q6: 扩散模型为什么适合建模用户兴趣？

**A**: 用户兴趣建模的本质问题是**从噪声历史行为中提取真实意图**，而扩散模型正是擅长处理"加噪-去噪"问题。CDP 将带噪的用户行为序列视为"污染观测"，通过条件去噪过程（以当前 Query-Item-Context 为条件），生成动态、纯净的兴趣表示。相比 Attention 的加权平均，扩散过程能真正"去除"噪声行为的影响，而非仅仅降低其权重。

---

### Q7: 联合广告（Joint Advertising）对广告主和平台各有什么影响？

**A**: 
- **对平台**：（1）提升广告位利用率（一个位置两个付费方）；（2）可能增加总收入（BundleNet 实验显示 +5%~+15%）；（3）需要更复杂的机制设计维护公平性。
- **对广告主**：（1）大广告主：竞争加剧，被迫"共享"位置；（2）小广告主：通过捆绑获得更多曝光，降低入场壁垒；（3）双方：广告联合展示需要品牌相容性，否则可能负面影响品牌形象。

---

### Q8: 什么是 Embedding Dimensional Collapse？如何用 SFG 解决？

**A**: 
**Collapse 定义**：判别式训练时，Embedding 空间的有效秩远低于设计维度，不同 ID 的 Embedding 趋于相近方向，特征区分能力弱。  
**原因**：纯分类/排序损失只关心决策边界，对 Embedding 空间内部结构无约束。  
**SFG 解法**：通过生成损失（特征重建）给 Embedding 增加约束，迫使模型利用全部维度来完整重建所有特征信息，避免维度退化。监督信号（click 标签）同时确保生成的特征对预测有用。

---

### Q9: 全站推广（QuanZhanTui）的激励相容性为什么重要？

**A**: 如果机制不满足 IC，卖家会策略性地低报 ROI 目标（以获得更高的"被帮扶"程度），导致：（1）平台接受低 ROI 订单，收益下降；（2）卖家花精力优化出价策略而非产品；（3）市场信息扭曲，平台无法真实了解卖家价值。LPA 的液态支付设计确保如实申报是卖家的占优策略，市场稳定运转。

---

### Q10: 广告系统中多臂老虎机（MAB）的实际应用场景？

**A**: 
- **冷启动**：新广告/新用户的探索（MAB-ColdStart，UCB 调整 CTR）
- **广告创意选择**：同一广告主多个创意素材的 A/B/C 测试，用 MAB 自适应分配流量
- **位置/格式探索**：不同广告格式（banner/原生/视频）的流量分配
- **个性化出价**：针对不同用户群的出价策略探索
- **超参数优化**：自动搜索 CTR 模型超参数（Hyperparameter Optimization）

---

### Q11: 生成式拍卖（GenAuction）相比传统机制的局限性？

**A**: （1）**可解释性差**：生成模型的分配决策难以向广告主解释，可能引发公平性质疑；（2）**推理延迟**：生成式前向推理比规则计算慢；（3）**训练稳定性**：需要大量历史竞价数据训练，冷启动困难；（4）**监管合规**：生成式机制的透明度低，可能面临反垄断监管审查；（5）**鲁棒性**：对抗性出价行为可能导致生成模型产生异常分配结果。

---

### Q12: 如何评估广告竞价机制是否公平（Fairness）？

**A**: 广告竞价公平性有多个维度：（1）**个体理性（IR）**：参与比不参与更有利，没有广告主被迫亏损；（2）**激励相容（IC）**：如实申报是最优策略，没有人通过撒谎获益；（3）**帕累托效率**：不存在帕累托改进（无法同时提升所有参与者的收益）；（4）**无嫉妒（Envy-free）**：任何广告主都不羡慕他人的分配结果；（5）**收入最大化**：平台收益达到理论最优（Myerson 机制）。

---

### Q13: 竞价优化中因果推断和 A/B 测试的区别与互补？

**A**: A/B 测试是随机对照实验（RCT），是因果识别的金标准，但**无法回答"如果对同一用户换策略会怎样"**（基本因果问题）；因果推断（IPW/DR）用观测数据估计反事实，**可以做离线策略评估（OPE）**而不用上线。互补关系：因果方法做离线粗筛，A/B 测试做最终验证。CausalBidding 的核心价值：把 OPE 与在线效果的相关性从 0.62 提到 0.89，大幅降低无效 A/B 实验数量。

---

### Q14: 出价系统中双过程控制（KBD）的工程实现难点？

**A**:  
（1）**知识 Embedding 更新频率**：市场日内波动显著（早高峰 vs 晚高峰竞争强度差 3x），知识 Embedding 需每小时更新，否则先验过时；  
（2）**两层更新解耦**：High-level controller（分钟级更新出价乘数边界）vs Low-level policy（毫秒级在线决策），时间尺度差 6 个量级，需要异步架构；  
（3）**Lagrangian 乘子 $\lambda$ 收敛**：$\lambda$ 过大导致保守出价欠消耗，过小导致超支。实践中用 PID 类控制器稳定 $\lambda$，而非纯梯度更新。

---

### Q15: 多智能体RL（MARL）自动出价为什么比单智能体更好？何时反而更差？

**A**:  
**更好的场景**：竞争格局稳定（广告主集中度高，头部玩家互相博弈），市场价格可预测性强——MARL 学到了竞争均衡策略，避免"军备竞赛"式恶性出价。  
**更差的场景**：（1）长尾市场（万级小广告主），建模为 MARL 计算不可行；（2）竞争者快速替换（新广告主频繁加入），历史学到的博弈策略快速失效；（3）合谋风险：MARL 在训练中可能学到隐性勾结策略（压低出价让平台收益下降），需要平台层面的安全约束。

---

## 🔑 关键趋势总结

### 趋势1：生成式范式全面崗位
- **CTR 预估**：从判别式特征交互 → 生成式特征生成（SFG）
- **用户兴趣建模**：从静态 Attention → 动态扩散生成（CDP）
- **竞价机制**：从规则式计算 → 生成式分配（GenAuction）
- **创意优化**：从独立评分 → 生成式组合搜索（GenCO）

### 趋势2：大模型能力下沉到广告系统
- 多模态大模型特征需要量化对齐（QARM）才能有效用于推荐
- 量化是桥接大模型表示空间与推荐系统的关键技术

### 趋势3：机制设计理论与深度学习融合
- 传统经济学机制（Myerson/VCG）+ 神经网络实现（BundleNet）
- 新型服务模式（全站推广）催生新机制理论（LPA）

### 趋势4：探索-利用在工业系统的成熟应用
- MAB 已成为广告冷启动的工业标配
- 受控探索（有界探索预算）是平衡短期/长期收益的关键

### 趋势5：因果推断进入广告出价优化主流
- 历史数据偏差问题长期被忽视，CausalBidding 将因果图+DR估计引入出价，OPE→在线相关性显著提升
- Offline RL + 反事实估计 正在成为自动出价的新标准方法论

### 趋势6：多智能体博弈从理论走向工业落地
- 单广告主视角 RL → 全市场博弈均衡视角 MARL
- CTDE 架构解决了"训练需全局信息、部署只能用本地信息"的矛盾
- 平台安全合规（防隐性勾结）成为 MARL 系统设计的新约束

---

*本综合总结涵盖 2024-2026 年广告系统最新进展，聚焦自动出价（因果/多智能体/知识注入新方向）、CTR/CVR预估、创意优化、竞价机制五大方向。*

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/vector_quantization_methods|向量量化方法]]
