# Auto Bidding 技术演进：从规则出价到强化学习

> 📚 参考文献
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../ads/papers/Real_Time_Bidding_Optimization_with_Multi_Agent_Deep_Rein.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/Multi_Objective_Optimization_for_Online_Advertising_Balan.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Action Is All You Need Dual-Flow Generative Ran...](../../ads/papers/Action_is_All_You_Need_Dual_Flow_Generative_Ranking_Netwo.md) — Action is All You Need: Dual-Flow Generative Ranking Netw...
> - [Autobid-Reinforcement-Learning-For-Automated-Ad...](../../ads/papers/AutoBid_Reinforcement_Learning_for_Automated_Ad_Bidding_w.md) — AutoBid: Reinforcement Learning for Automated Ad Bidding ...
> - [Plum Adapting Pre-Trained Language Models For Indu](../../ads/papers/PLUM_Adapting_Pre_trained_Language_Models_for_Industrial.md) — PLUM: Adapting Pre-trained Language Models for Industrial...
> - [Onerec-Think In-Text Reasoning For Generative R...](../../ads/papers/OneRec_Think_In_Text_Reasoning_for_Generative_Recommendat.md) — OneRec-Think: In-Text Reasoning for Generative Recommenda...
> - [Llm-Enhanced-Ad-Creative-Generation-And-Optimiz...](../../ads/papers/LLM_Enhanced_Ad_Creative_Generation_and_Optimization_for.md) — LLM-Enhanced Ad Creative Generation and Optimization for ...
> - [Generative Click-Through Rate Prediction With Appl](../../ads/papers/Generative_Click_through_Rate_Prediction_with_Application.md) — Generative Click-through Rate Prediction with Application...

> 整理时间：2026-03-16  
> 作者：MelonEggLearn  
> 参考资料：Bidding Machine (TKDE 2018), wzhe06/Ad-papers, RTB Survey, 阿里/字节/美团公开工作

## 📐 核心公式与原理

### 1. 最优出价

$$
bid^* = v \cdot pCTR \cdot pCVR
$$

- 出价 = 价值 × 点击率 × 转化率

### 2. 预算约束

$$
\sum_{t=1}^T c_t \leq B
$$

- 总花费不超过预算 B

### 3. Lagrangian 松弛

$$
L = \sum_t v_t x_t - \lambda(\sum_t c_t x_t - B)
$$

- λ 控制预算约束的松紧

---

## 演进时间线

```
2010-2015  规则出价 & 手动调价
    ↓
2015-2019  基于模型预测的自动出价（oCPC/oCPA + PID Pacing）
    ↓
2018-2021  基于优化的智能出价（LP + 对偶方法 + Bid Shading）
    ↓
2020-至今  强化学习自动出价（DQN/PPO/DDPG）
    ↓
2022-至今  多智能体 & LLM 出价（MARL / 生成式出价）
```

| 时期 | 核心方法 | 代表工作 |
|------|----------|----------|
| 2010-2015 | CPM固定出价、人工调bid | 早期RTB系统 |
| 2015-2019 | oCPC/oCPA + PID | 百度凤巢、阿里妈妈P4P |
| 2018-2021 | LP/对偶优化、Bid Shading | Bidding Machine (TKDE 2018) |
| 2020-至今 | RL自动出价 | 阿里RL Bidding, 字节Auto Bidding |
| 2022-至今 | MARL / LLM | Multi-Agent RTB, GPT-based策略 |

---

## 第一阶段：规则出价 & 手动调价（2010-2015）

### 核心机制

**CPM（千次展示费用）出价**是最早的广告出价方式：
- 广告主按固定 CPM 价格竞买展示机会
- 媒体侧使用**二价拍卖（Second Price Auction）**：赢家支付第二高价 + 0.01 元
- 出价策略完全依赖人工经验：运营人员根据投放效果定期手动调整出价

**CPC（点击付费）出价**：
- 出价公式：`bid = CPC目标价`
- 平台根据 eCPM = bid × pCTR 进行排序
- 广告主仅在被点击时付费

**人工调价流程**：
```
数据报表（昨日CPA/ROI）→ 人工分析 → 手动调价 → 等待效果
（周期：天级，响应慢）
```

### 局限性

1. **响应延迟**：人工调价周期为天/周级，无法应对流量实时波动
2. **颗粒度粗**：一个广告计划一个出价，无法对不同流量机会差异化出价
3. **目标不对齐**：CPM/CPC 出价不能直接优化广告主真正关心的 CPA（转化成本）
4. **规模不可扩展**：大型广告主有成千上万个计划，人工管理成本极高
5. **缺乏预算控制**：无法保证预算在时间上均匀消耗（凌晨流量便宜但质量差）

---

## 第二阶段：基于模型预测的自动出价（2015-2019）

### oCPC / oCPA 出价公式

**oCPC（Optimized Cost Per Click）**：以CPC形式出价，但系统内部换算为 eCPM，目标是优化 CPA。

核心思路：用模型预测转化率（CVR），将 CPA 目标转化为每次展示的出价。

**oCPA 出价公式推导**：

设：
- `CPA_target`：广告主设定的目标转化成本
- `pCTR`：系统预测点击率
- `pCVR`：系统预测点击后转化率
- `pConv = pCTR × pCVR`：展示层转化率

则每次展示的期望价值：

```
V(impression) = CPA_target × pConv
              = CPA_target × pCTR × pCVR
```

最优出价（假设二价拍卖）：

```
bid* = CPA_target × pCTR × pCVR
```

**oCPC 变体（带权重）**：

```
bid = α × CPC_target × pCTR / pCTR_avg
```

其中 α 是系数，由系统根据预算消耗情况动态调整。

### CVR 预测模型的作用

CVR 预测是 oCPC/oCPA 的核心模块，典型架构：

```
特征工程
├── 用户特征：历史行为、人口属性、设备信息
├── 广告特征：创意类型、广告主行业、历史CTR
├── 上下文特征：时间、位置、App类型
└── 交叉特征：用户×广告 交叉

模型选择（演进）：
LR → FM → GBDT+LR → DNN → DIN（用户兴趣建模）
```

**CVR 预测难点**：
1. **样本稀疏**：转化行为发生率极低（CVR通常 < 1%）
2. **延迟反馈**：转化可能在点击后数小时/天才发生（Delayed Feedback 问题）
3. **样本选择偏差**：CVR 模型只在点击样本上训练，但出价在展示层
   - ESMM（阿里 2018）通过整空间多任务学习解决此问题

### PID Pacing 算法（详细）

PID（比例-积分-微分）控制器被广泛用于预算 Pacing 和出价调整。

**问题定义**：
- 目标：让实际消耗曲线跟随目标消耗曲线（预算均匀消耗）
- 控制变量：出价乘数（bid multiplier）`λ`
- 反馈信号：实际消耗 vs 目标消耗的偏差

**PID 控制公式**：

```
e(t) = target_spend(t) - actual_spend(t)    # 误差

λ(t) = K_p × e(t)                          # 比例项（P）
     + K_i × Σe(τ)Δτ                       # 积分项（I）
     + K_d × (e(t) - e(t-1))/Δt           # 微分项（D）

bid(t) = base_bid × (1 + λ(t))
```

**各项作用**：
| 项 | 符号 | 作用 | 过大的后果 |
|----|------|------|-----------|
| 比例 | K_p | 快速响应当前误差 | 震荡 |
| 积分 | K_i | 消除稳态误差，处理历史累积 | 超调 |
| 微分 | K_d | 预测趋势，抑制快速变化 | 噪声放大 |

**在广告 Pacing 中的应用**：

```python
# 伪代码
def pid_pacing(t, actual_spend, target_spend, prev_error, integral):
    error = target_spend - actual_spend  # 消耗不足则 error > 0
    integral += error * dt
    derivative = (error - prev_error) / dt
    
    adjustment = Kp * error + Ki * integral + Kd * derivative
    new_bid_multiplier = clip(1.0 + adjustment, 0.1, 3.0)
    return new_bid_multiplier
```

**实际应用细节**：
- 通常以 5-15 分钟为控制周期
- 积分项需要 **Anti-windup** 处理（防止预算已耗尽但积分项继续累积）
- LinkedIn/Facebook 均有 PID Pacing 系统的公开论文（Smart Pacing）

### 预算约束下的 KKT 最优出价推导（有公式！）

**问题建模**：

在固定预算 B 下，最大化总转化量（或总价值）。

设：
- $n$ 个广告展示机会
- 对第 $i$ 个机会出价 $b_i$，赢得概率 $w_i(b_i)$（单调递增）
- 每次展示的转化价值 $v_i = \text{pCTR}_i \times \text{pCVR}_i \times \text{CPA}_{\text{target}}$
- 赢得时的支付（二价拍卖中约为对手最高价 $m_i$）

**优化问题**：

$$
\max_{b_1, \ldots, b_n} \sum_{i=1}^{n} v_i \cdot w_i(b_i)
$$

$$
\text{s.t.} \quad \sum_{i=1}^{n} m_i \cdot w_i(b_i) \leq B
$$

$$
b_i \geq 0, \quad \forall i
$$

**拉格朗日函数**：

$$
\mathcal{L}(b, \lambda) = \sum_{i=1}^{n} v_i \cdot w_i(b_i) - \lambda \left( \sum_{i=1}^{n} m_i \cdot w_i(b_i) - B \right)
$$

**KKT 条件**：

对每个 $b_i$ 求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial b_i} = v_i \cdot w_i'(b_i) - \lambda \cdot m_i \cdot w_i'(b_i) = 0
$$

$$
w_i'(b_i)(v_i - \lambda \cdot m_i) = 0
$$

由于 $w_i'(b_i) > 0$（赢得概率关于出价单调递增），得：

$$
v_i = \lambda \cdot m_i
$$

等价于：

$$
b_i^* = v_i / \lambda = \frac{\text{pCTR}_i \times \text{pCVR}_i \times \text{CPA}_{\text{target}}}{\lambda}
$$

**关键结论**：
- **最优出价 = 期望价值 / 拉格朗日乘子 λ**
- λ 是全局预算约束的对偶变量，可理解为**预算的边际价值**
- 当预算充足时 λ → 0，出价 → 无穷（不受约束）；预算紧张时 λ 增大，出价降低
- **互补松弛条件**：要么预算耗尽（$\sum m_i w_i = B$），要么 λ = 0

**λ 的求解（二分搜索）**：

```python
def find_optimal_lambda(v_list, m_list, B, tol=1e-6):
    lo, hi = 0, max(v_list) / min(m_list)
    while hi - lo > tol:
        lam = (lo + hi) / 2
        bids = [v / lam for v, m in zip(v_list, m_list)]
        spend = sum(win_prob(b) * m for b, m in zip(bids, m_list))
        if spend > B:
            lo = lam  # λ 太小 → 出价太高 → 超预算 → 增大λ
        else:
            hi = lam
    return (lo + hi) / 2
```

### 局限性：为什么模型预测出价不够？

1. **模型误差累积**：pCTR × pCVR 的预测误差直接影响出价精度，且难以解耦
2. **独立性假设违反**：每次出价假设相互独立，忽略了**序列决策**和跨时段的预算影响
3. **静态λ问题**：λ（预算拉格朗日乘子）通常离线计算，无法实时响应市场价格波动
4. **对手策略盲区**：不建模竞争对手的出价行为，容易导致过高或过低出价
5. **非平稳市场**：实际出价环境时刻变化（节假日、大促等），静态模型无法自适应
6. **延迟反馈**：转化数据延迟到达，导致 CVR 模型更新滞后

---

## 深度理论分析：出价系统的核心框架

### 广告出价的三大问题

广告出价系统实质上在解决三个层次的问题，这三个问题需要分别优化但又互相关联：

#### 问题一：单次出价优化（AutoBid/GRPO 类）
- **目标**：最大化转化量，满足 CPA ≤ 目标 + 日预算不超支
- **传统方法**：PID 控制（比例-积分-微分），根据 CPA 偏差反馈调整出价系数
- **RL 方法（AutoBid）**：把全天竞拍建模为 MDP，状态=剩余预算+当前时段+竞争强度，动作=出价系数 k，RL 学习跨时段最优策略
- **核心技术**：Lagrangian Relaxation 处理约束（CPA/Budget → 软约束 → 对偶乘子自适应调整）

#### 问题二：多目标平衡
- **目标**：eCPM（收入）vs 相关性（用户体验）vs 广告主 ROI
- **工程解法**：加权求和（Score = w₁×eCPM + w₂×质量分）+ 动态权重（实时感知广告填充率调档）
- **关键洞察**：纯收入最大化是短视的，用户体验差 → 留存下降 → 未来收入损失；LTV 模型估算长期价值纳入优化

#### 问题三：预算分配与 Pacing
- **目标**：在给定预算下，均匀分散消耗（避免前期花完）
- **传统方法**：PID 控制或对偶梯度下降
- **非平稳优化**：在流量分布实时变化（早晚高峰差异）的情况下自适应调整

**技术演进脉络**：
```
手动出价（2010前）
  → 规则出价（公式）
    → oCPM/oCPA（预估pCTR/pCVR优化）
      → PID 控制 Pacing（工业主流 2015-2020）
        → AutoBid RL（2021+，头部平台）
          → 多目标 RL + LTV 建模（当前前沿）
```

### 5 条核心洞察（跨文献共性规律）

1. **出价本质是约束优化问题**：在预算、ROI、曝光量等多约束下最大化广告主目标（GMV/转化数），数学上是 KKT 条件求解

2. **oCPC/oCPA 是工业标配**：通过预估 CTR/CVR 自动调节出价，广告主只需设定目标 CPA，系统自动执行 `bid = target_CPA × pCVR × 系数`

3. **RL 出价是终极形态但落地困难**：状态空间（剩余预算、时段、竞争强度）+ 动作空间（出价系数）+ 奖励（转化-成本），但 off-policy 评估和 sim2real gap 是主要障碍

4. **Pacing 是出价的"节奏控制器"**：PID 控制器确保预算均匀消耗，避免前期花完后期无量，是出价系统的必备组件

5. **一价拍卖（FPA）正在取代二价拍卖（SPA）**：Google/Meta 已迁移到 FPA，出价策略从"truthful bidding"变为"strategic bidding"，Bid Shading 成为关键技术

### 跨文献共性规律表

| 规律 | 体现论文/系统 | 说明 |
|------|-------------|------|
| **约束优化统一框架** | AutoBid, Budget Pacing Guide | 所有出价问题最终归结为带约束的优化问题 |
| **离线仿真是 RL 落地前提** | AIGB, VirtualBidder | 没有高质量的离线仿真器，RL 出价无法安全上线 |
| **Pacing + Bidding 解耦** | PID Pacing, LP 对偶 | 工业系统将"花多少"（Pacing）和"每次出多少"（Bidding）分开管理 |
| **冷启动依赖探索机制** | UCB, Thompson Sampling | 新广告/广告主需要 Explore-Exploit 平衡获取初始数据 |

### PID Pacing 的生活类比

你有 100 块零花钱过一整天：
- **没有 Pacing（贪心）**：早上看到好吃的全花了，下午饿肚子
- **简单 Throttling**：随机跳过一半机会（抛硬币），省钱但可能错过最好的食物
- **PID 控制（最常用）**：每小时检查剩余预算，花多了就少参与，花少了就多参与——像自动调节油门的司机
- **对偶梯度下降**：维护一个"花钱欲望系数"，预算消耗快时自动降低出价，消耗慢时提高——像理性的收藏家，货多时出价低，货紧时出价高
- **非平稳自适应（Wasserstein）**：连流量分布都会变（早高峰→午休→晚高峰），用 Wasserstein 距离追踪变化，自适应调整

### 对偶梯度下降 Pacing 详解

#### 数学框架

**原始问题**：

$$
\max_{bid_1, \ldots, bid_T} \sum_{t=1}^{T} value(bid_t) \quad \text{s.t.} \quad \sum_{t=1}^{T} cost(bid_t) \leq Budget
$$

**引入拉格朗日乘子 λ**（预算的边际价值）：

$$
L(bid, \lambda) = \sum_{t=1}^{T} value(bid_t) - \lambda \left( \sum_{t=1}^{T} cost(bid_t) - Budget \right)
$$

**关键结论**：最优出价满足

$$
bid_t^* = \arg\max_{bid} [value(bid) - \lambda \times cost(bid)]
$$

**λ 的直觉**：
- λ 代表"1元预算能带来的边际价值"，即预算的影子价格
- λ 越大（预算越紧张），出价越低（更保守）
- λ 越小（预算充裕），出价越接近真实价值

#### 在线更新规则

$$
\lambda_{t+1} = \lambda_t + \eta \cdot (cost(bid_t) - \frac{Budget}{T})
$$

- 若实际消耗 > 目标消耗 → 增大 λ → 出价降低 → 减少竞争
- 若实际消耗 < 目标消耗 → 减小 λ → 出价提高 → 赢更多

**与 PID 控制对比**：

| 项 | PID 控制 | 对偶梯度下降 |
|-----|---------|-----------|
| **原理** | 反馈控制（偏差驱动） | 优化方法（边际分析） |
| **实现** | 需要参数调优（Kp, Ki, Kd） | 参数较少，主要是学习率 η |
| **非平稳适应** | 需要手动调参 | 自适应能力强 |
| **理论保证** | 启发式，无最优性保证 | 有收敛性证明 |
| **工程采用** | 主流（Facebook, LinkedIn） | 学术界多，工程应用少 |

### 第一价格拍卖（FPA）中的 Bid Shading

#### 问题设置

**二价拍卖（SPA，目前逐渐淘汰）**：
- 赢家支付第二高价
- 最优策略：`bid = true_value`（Vickrey定理）
- 不需要估计对手，truthful bidding 是占优策略

**一价拍卖（FPA，Google/Meta 新标准）**：
- 赢家支付自己出价
- 若按真实估值 v 出价赢了，收益 = v - v = 0（没有利润！）
- 因此需要压低出价，这就是 **Bid Shading**

#### 最优 Shading 公式

设对手出价分布为 F(·)，概率密度为 f(·)，我方真实估值为 v，则最优出价为：

$$
bid^* = v - \frac{1 - F(bid^*)}{f(bid^*)}
$$

**直觉**：
- 分子 $(1 - F(bid^*))$ 是以该出价赢的概率
- 分母 $f(bid^*)$ 是对手在该点的密度
- 比值 $\frac{1 - F(bid^*)}{f(bid^*)}$ 是"为了多赢一个单位概率，愿意让渡多少利润"

#### 在实践中的估计

**关键技术**：需要在线学习对手出价分布 F(b)

```python
# 伪代码：估计 Bid Landscape
def update_bid_landscape(market_prices_history):
    """用历史对手出价更新分布估计"""
    F_estimate = empirical_cdf(market_prices_history)
    
    # 计算 Shading 因子
    for bid in candidate_bids:
        win_prob = 1 - F_estimate(bid)
        density = estimate_density(F_estimate, bid)
        shading_factor = (win_prob / density)
        optimal_bid = true_value - shading_factor
```

### 非平稳环境下的 Wasserstein Pacing

#### Wasserstein 距离的优势

当流量分布实时变化（早高峰 → 午休 → 晚高峰），我们需要度量分布变化的幅度。

**TV 距离问题**：对任何不重叠的分布都给出最大值 1，对小幅位置偏移过于激进。

**Wasserstein 距离优势**：

$$
W_1(F_t, F_{t-1}) = \int_0^\infty |F_t(b) - F_{t-1}(b)| db
$$

- 考虑分布的几何结构，小幅位移只产生小的 W 值
- 直觉：将一个分布变换为另一个分布的最小"地球搬运成本"

#### 遗憾界

在非平稳 FPA 环境中，遗憾界为：

$$
\text{Regret}(T) = O\left(\sqrt{T} + \sum_{t=1}^{T} W_1(F_t, F_{t-1})\right)
$$

- 第一项 $\sqrt{T}$ 是静态情况下的标准 regret
- 第二项 $\sum W_1$ 是分布变化带来的额外遗憾
- 当流量分布变化平缓时（如周期性），总 regret 约束紧

---

## 第三阶段：基于优化的智能出价（2018-2021）

### 线性规划（LP）出价：Bid Landscape + 期望收益最大化

**Bid Landscape 建模**：

Bid Landscape 是对市场竞价环境的建模——给定一个广告机会，不同出价 b 对应的赢得概率 $P(\text{win}|b)$。

常用模型：
- **Log-normal 分布**：$P(\text{win}|b) = \Phi\left(\frac{\ln b - \mu}{\sigma}\right)$
- **Survival 分析**：将竞价市场价格建模为生存时间

**LP 出价框架**：

在 Bid Landscape 已知的前提下，对每个广告机会建立 LP：

$$
\max \sum_{i} v_i \cdot P_i(\text{win}|b_i)
$$

$$
\text{s.t.} \quad \sum_{i} \mathbb{E}[\text{cost}_i | b_i] \leq B
$$

$$
\text{其中 } \mathbb{E}[\text{cost}_i | b_i] = \int_0^{b_i} x \cdot f_i(x) dx
$$

**Bidding Machine 框架**（TKDE 2018, arXiv:1803.02194）：

联合优化三个子问题：
1. **utility 预测**：预测每个 impression 的点击/转化价值
2. **market price 预测**：预测对手出价分布（Bid Landscape）
3. **bid optimization**：基于 1+2 的结果求解最优出价

核心公式：
```
optimal_bid_i = argmax_b [v_i × P(win|b) - E[cost|b]]
```

### Bid Shading（程序化广告）

**背景**：程序化广告从**二价拍卖（Vickrey）**向**一价拍卖（First Price）**迁移。

- **二价拍卖**：赢家支付第二高价，最优策略是真实出价
- **一价拍卖**：赢家支付自己出的价，应该"压价"（Bid Shading）

**Bid Shading 原理**：

在一价拍卖中，最优出价满足：

$$
b^* = v - \frac{1 - F(b^*)}{f(b^*)}
$$

其中 $F(\cdot)$ 是对手出价的 CDF，$f(\cdot)$ 是 PDF。

**实践方法**：
1. 训练 Bid Landscape 模型，估计市场价格分布 $F(m)$
2. 计算 Shading 因子 $\rho \in (0,1)$，使得 $b_{\text{shaded}} = \rho \times v$
3. $\rho$ 的估计：$\rho^* = \arg\max_\rho [v - \rho v] \cdot F(\rho v)$

**工程实现**：
- 微软/谷歌广告交易所迁移到一价拍卖后，各 DSP 都需要实现 Bid Shading
- 通常维护每个（流量源×广告类型）的独立 Bid Landscape 模型

### DMBGD / 对偶方法：KKT + 拉格朗日乘子

**对偶梯度下降（Dual-based Method）**：

回到第二阶段的 KKT 推导，对偶变量 λ 的更新：

$$
\lambda^{(t+1)} = \lambda^{(t)} - \alpha \cdot \nabla_\lambda \mathcal{L}
$$

$$
\nabla_\lambda \mathcal{L} = B - \sum_i m_i \cdot w_i(b_i^*)
$$

即：
```
λ更新规则：
- 若实际消耗 < 目标预算（预算有余）→ 降低λ → 出价提高 → 赢更多
- 若实际消耗 > 目标预算（超预算）→ 提高λ → 出价降低 → 减少竞争
```

**DMBGD（Dual-based Mini-Batch Gradient Descent）**：

阿里/字节等公司的对偶方法实现：
1. 每个 mini-batch 内部用当前 λ 计算最优 bid
2. 用实际消耗与预算的差异更新 λ
3. 支持多约束（预算 + ROI + 转化量下限）

**多约束对偶问题**：

$$
\max_{b_i} \sum_i v_i w_i(b_i)
$$

$$
\text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B \quad (\text{预算})
$$

$$
\quad \quad \sum_i v_i w_i(b_i) / \sum_i c_i w_i(b_i) \geq \text{ROI}_{\text{target}} \quad (\text{ROI约束})
$$

拉格朗日：$\mathcal{L} = \sum_i (v_i - \lambda_1 c_i - \lambda_2(c_i \cdot \text{ROI} - v_i)) w_i(b_i) + \lambda_1 B$

最优出价：$b_i^* = \frac{(1+\lambda_2)v_i - (\lambda_1 + \lambda_2 \cdot \text{ROI})c_i}{\text{适当系数}}$

### 约束优化框架：max ROI s.t. budget constraint

**通用框架**：

$$
\max_{b} \text{ROI}(b) = \frac{\sum_i v_i w_i(b_i)}{\sum_i c_i w_i(b_i)}
$$

$$
\text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B
$$

**分形式转化**（Dinkelbach 方法）：

迭代求解，每轮固定 ROI 值 $\rho^{(k)}$，求解：

$$
\max_b \sum_i (v_i - \rho^{(k)} c_i) w_i(b_i) \quad \text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B
$$

该子问题变为标准的线性约束凸优化，用对偶方法求解，然后更新 $\rho$。

---

## 第四阶段：强化学习自动出价（2020-至今）

### 为什么需要 RL？

| 问题 | 传统方法的局限 | RL 的优势 |
|------|--------------|-----------|
| **序列决策** | 每次出价独立决策，不考虑全局 | MDP 建模全局序列 |
| **非平稳环境** | 静态 λ 无法实时适应市场变化 | 在线策略更新 |
| **长期收益** | 只优化当前时刻价值 | 优化全局 episode 累积奖励 |
| **复杂约束** | 预算约束难以精确满足 | 将约束融入奖励或价值函数 |
| **对手建模** | 不感知竞争对手策略 | MARL 中显式建模竞争 |

**核心理念**：将广告出价问题建模为 **马尔可夫决策过程（MDP）**，用 RL Agent 学习最优策略。

### 状态空间设计（state）

完整的 state $s_t$ 包含以下分量：

#### 1. 当前机会特征（Opportunity Features）
```
s_opportunity = {
    pCTR_t,          # 预估点击率
    pCVR_t,          # 预估转化率
    pValue_t,        # 预估价值 = pCTR × pCVR × CPA_target
    impression_quality_score,  # 流量质量分
    slot_position,   # 广告位置（头部/中部/底部）
    user_features,   # 用户特征向量（可选，维度高时用embedding）
    context_features # 时间、场景特征
}
```

#### 2. 当前时刻的预算状态（Budget State）
```
s_budget = {
    remaining_budget_t,  # 剩余预算
    total_budget,         # 总预算
    budget_ratio_t,       # 剩余预算比例 = remaining/total
    time_ratio_t,         # 剩余时间比例 = remaining_time/total_time
    spend_rate,           # 当前消耗速率（过去N分钟）
    pacing_ratio          # 实际消耗/目标消耗
}
```

#### 3. 历史竞价统计（Market State）
```
s_market = {
    win_rate_window,      # 过去T个时间窗口的赢率
    avg_cost_window,      # 过去T个窗口的平均成本
    market_price_percentile, # 市场价格分位数（25th/50th/75th）
    competition_intensity # 竞争激烈程度指标
}
```

#### 4. 投放效果累积（Performance State）
```
s_performance = {
    cumulative_clicks_t,    # 累计点击数
    cumulative_conversions_t, # 累计转化数
    cumulative_cost_t,      # 累计花费
    current_CPA_t,          # 当前实际CPA
    current_ROI_t,          # 当前实际ROI
    cpa_vs_target,          # CPA相对目标的偏差
}
```

**典型 state 维度**：合并后约 20-50 维，高维特征可压缩后再输入。

### 动作空间设计（action）

#### 方案一：直接出价（连续动作）
```
a_t ∈ [bid_min, bid_max]  # 连续值，使用 DDPG/SAC 等算法
```

优点：灵活；缺点：训练不稳定，探索困难

#### 方案二：出价乘数（连续）
```
a_t = λ_t ∈ [0.1, 3.0]
bid_t = base_bid × λ_t    # base_bid 来自 oCPA 公式
```

优点：基准出价合理，调整幅度有限，训练更稳定

#### 方案三：离散出价档位（离散动作）
```
a_t ∈ {0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0} × base_bid
# 使用 DQN 或 PPO（离散版）
```

优点：训练稳定，便于工程实现；缺点：精度有限

#### 方案四：分级决策
```
高层 Agent（秒级/分级）：决定 Pacing 策略（激进/保守）
低层 Agent（每次竞价）：决定具体出价乘数
```

用于**分层强化学习（HRL）**，减少高层决策频率。

### 奖励函数设计（reward）

奖励是 RL 出价中最关键也最难设计的部分。

#### 基础奖励
```
r_t = value_gained - cost_paid
    = (click_t × pCVR × CPA_target - actual_cost_t) × win_t
```

#### 带预算惩罚的奖励
```
r_t = value_t × win_t
    - α × max(0, actual_spend_t - budget_target_t)   # 超预算惩罚
    - β × max(0, budget_target_t - actual_spend_t)   # 欠消耗惩罚（可选）
```

#### 约束满足奖励（Constrained MDP）
```
# 将约束转化为拉格朗日乘子形式
r_t = value_t - λ_budget × cost_t - λ_CPA × (cost_t - CPA_target)
```

#### 奖励延迟问题（Delayed Reward）处理
- 转化可能在点击后 1-72 小时才发生
- **归因方案**：点击时给予部分即时奖励（pCVR × CPA_target），转化时给差值
- **序列奖励**：用 GAE（广义优势估计）处理长时序奖励

#### Episode 设计
```
一个 Episode = 一天的广告投放（00:00 - 24:00）
每步 = 一次竞价机会（毫秒级），但通常聚合为分钟级
Terminal = 预算耗尽 or 时间窗口结束
```

### 主流算法：DQN / PPO / DDPG 在出价中的应用

#### DQN（Deep Q-Network）
- **适用**：离散动作空间（出价档位选择）
- **核心**：Q(s,a) 函数，选 Q 值最大的动作
- **广告中的改进**：
  - Dueling DQN：分离状态价值和动作优势，更稳定
  - Double DQN：解耦选择和评估，减少高估
- **局限**：动作空间不能太大（通常 < 20 档位）

```python
# DQN 出价核心逻辑
def bid(state):
    q_values = dqn_network(state)  # [0.5x, 0.7x, 1.0x, 1.5x, 2.0x]
    action_idx = argmax(q_values)
    return bid_levels[action_idx] * base_bid
```

#### PPO（Proximal Policy Optimization）
- **适用**：连续和离散动作均可
- **核心**：限制策略更新幅度（clip 机制），训练稳定
- **广告中优势**：
  - 在线训练：收集新数据 → 更新策略（on-policy 更接近真实分布）
  - Actor-Critic 结构：同时学习策略和价值函数

```
PPO Loss = min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)
其中 r_t = π(a|s) / π_old(a|s) 为重要性比率
    A_t 为优势函数估计
    ε = 0.1 ~ 0.2（clip 范围）
```

#### DDPG（Deep Deterministic Policy Gradient）
- **适用**：连续动作空间（出价乘数为连续值）
- **核心**：Actor 直接输出出价，Critic 评估 Q 值
- **广告中改进**：
  - TD3：Twin Delayed DDPG，解决 Q 值高估
  - SAC：Soft Actor-Critic，加入熵正则，增强探索

```python
# DDPG Actor
class BiddingActor(nn.Module):
    def forward(self, state):
        x = self.encoder(state)
        # 输出出价乘数 λ ∈ (0.1, 3.0)
        lambda_ = 0.1 + 2.9 * torch.sigmoid(self.fc_out(x))
        return lambda_
```

### 代表工作

#### 阿里 RL Bidding (2020)
- **论文**：《Reinforcement Learning based Whole-Chain Recommendations》及相关工作
- **框架**：将整个广告投放周期建模为 MDP
- **创新**：
  1. **全局预算感知**：state 包含剩余预算/剩余时间，Agent 学会在不同预算阶段采取不同策略
  2. **Bid Landscape 辅助**：用历史竞价数据估计赢率函数，辅助 reward 计算
  3. **离线预训练 + 在线微调**：避免冷启动问题，离线仿真器中预训练，在线 A/B 测试微调
- **效果**：相比 PID Pacing + oCPA，GMV 提升 ~5%，ROI 改善 ~8%

#### 字节 Auto Bidding (2021)
- **系统**：字节跳动广告系统的全自动化出价
- **关键设计**：
  1. **分层架构**：粗粒度（campaign 级）+ 细粒度（impression 级）双层出价
  2. **对偶方法 + RL 融合**：用对偶方法提供出价基准，RL 学习调整乘数
  3. **多目标优化**：同时优化 GMV、ROI、CPA 多个指标
  4. **实时特征**：毫秒级特征抽取，保证出价及时性
- **训练**：离线模拟器（log replay）+ 在线 shadow mode 验证

#### 美团 Auto Bidding with RL (2022)
- **场景**：美团广告（外卖/酒旅等本地生活）
- **特点**：
  1. **稀疏奖励处理**：本地生活转化周期长，用预期价值 shaping reward
  2. **上下文感知**：加入餐厅/商家特征作为 state，个性化出价策略
  3. **安全约束**：加入出价上下界硬约束，防止异常出价
  4. **多约束学习**：同时满足预算 + CPA + ROI 三重约束

### 核心挑战：非平稳环境、竞争对手建模、奖励延迟

#### 1. 非平稳环境（Non-stationary Environment）
**问题**：
- 流量分布随时间变化（节假日、大促、一天内的流量波峰）
- 竞争对手策略不断调整
- 模型在历史数据上训练，部署后分布偏移

**解决方案**：
- **在线学习（Online RL）**：持续收集最新数据更新策略
- **元学习（Meta-RL）**：学习"如何快速适应新环境"的元策略
- **上下文感知**：将时间特征（工作日/周末/节假日）加入 state
- **域随机化（Domain Randomization）**：训练时随机扰动环境参数，增强泛化

```python
# 在线更新示意
for each_auction_batch:
    experience = collect_rollout(policy)
    update_policy(experience)  # PPO 更新
    update_lambda(budget_constraint)  # 对偶变量更新
```

#### 2. 竞争对手建模（Opponent Modeling）
**问题**：
- 实际出价环境是多 Agent 博弈，竞争对手策略影响赢率
- 单 Agent RL 假设环境固定，无法处理对手策略变化

**解决方案**：
- **Mean Field RL**：将对手群体行为建模为均场（mean field），降低复杂度
- **历史行为建模**：用 LSTM 预测对手下一步出价分布
- **Bid Landscape 更新**：实时更新市场价格模型，隐式感知对手变化

#### 3. 奖励延迟（Reward Delay）
**问题**：
- 广告点击可立即观测，转化（购买/注册）可能延迟数小时到数天
- RL 训练需要及时的奖励信号，否则信用分配（Credit Assignment）困难

**解决方案**：
- **即时奖励代理**：用预测转化概率 $\hat{p}_{CVR}$ 作为即时 reward，真实转化到来后修正
- **归因与修正**：转化到来时，回溯修正对应 impression 的 reward
- **延迟感知模型**：将历史窗口内的延迟转化统计加入 state

---

## 第五阶段：多智能体 & LLM 出价（2022-至今）

### MARL（多智能体RL）：考虑竞争对手反应

**动机**：现实中的广告竞价是多个广告主同时参与的博弈，单 Agent RL 将其他 Agent 视为环境，忽略了竞争关系的动态性。

**MARL 出价框架**：

```
Agent_1（广告主A） ←→ 共享拍卖环境 ←→ Agent_2（广告主B）
        ↑                                      ↑
   出价策略π_1                            出价策略π_2
```

**主要范式**：

| 范式 | 思路 | 代表算法 |
|------|------|----------|
| **合作 MARL** | 多个代理合作最大化整体收益 | QMIX, MADDPG |
| **竞争 MARL** | 零和博弈，纳什均衡求解 | NFSP, CFR |
| **混合 MARL** | 联合体内合作，联合体间竞争 | MAPPO |

**Real-Time Bidding with MARL**：
- 每个 DSP 对应一个 Agent
- 奖励：各自 Agent 的 campaign 表现（而非共享奖励）
- 训练：Centralized Training with Decentralized Execution（CTDE）
  - 训练时 Critic 可以观测所有 Agent 的动作（全局信息）
  - 执行时 Actor 只用本地状态出价

**均场博弈（Mean Field Game）**：
- 当竞争 Agent 数量很多时，用均场近似处理对手群体
- 每个 Agent 只需关注"平均对手行为"，大幅降低复杂度
- 适合大规模广告市场（数百个 DSP 同时竞争）

### 生成式出价 / LLM-based 出价策略

**新范式**：利用 LLM 的语言理解和推理能力，从广告主需求生成出价策略。

#### 方案一：LLM 作为策略生成器
```
广告主输入："我要投放运动鞋广告，目标人群是18-25岁男性，
            月预算10万，ROI要求 > 3"
           ↓
LLM（GPT-4/Claude）解析意图
           ↓
输出结构化出价策略参数：
{
    "base_bid": 2.5,
    "target_CPA": 50,
    "max_bid": 8.0,
    "audience_multipliers": {"age_18_25_male": 1.3, ...},
    "time_multipliers": {"evening": 1.2, "noon": 0.8}
}
```

#### 方案二：LLM + RL 混合
- LLM 提供策略初始化（warm start），避免 RL 冷启动
- RL 在 LLM 策略基础上进行细粒度优化
- LLM 解析复杂约束（自然语言 → 约束函数）

#### 方案三：Prompt-based 出价调整
- 将当前投放效果数据格式化为 prompt
- LLM 分析并给出出价调整建议
- 适合辅助人工决策，而非全自动

**挑战**：
- LLM 推理延迟（秒级）与实时出价（毫秒级）的矛盾
- LLM 输出不稳定性（相同 prompt 可能给出不同建议）
- 端到端的效果难以保证（无反馈闭环）

---

## 各方法横向对比表

| 维度 | 规则出价 | oCPC/oCPA | LP/对偶方法 | RL 出价 | MARL |
|------|---------|-----------|------------|---------|------|
| **建模复杂度** | 低 | 中 | 高 | 高 | 极高 |
| **响应速度** | 天级 | 分钟级 | 秒级 | 毫秒级 | 毫秒级 |
| **全局最优** | ❌ | ❌ | 近似 | ✅（长期） | ✅（博弈均衡） |
| **序列决策** | ❌ | ❌ | 部分 | ✅ | ✅ |
| **多约束支持** | ❌ | 有限 | ✅ | ✅ | ✅ |
| **对手感知** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **冷启动** | 容易 | 容易 | 中等 | 难 | 最难 |
| **可解释性** | 高 | 中 | 高 | 低 | 低 |
| **工程复杂度** | 低 | 中 | 高 | 极高 | 极高 |
| **样本效率** | — | — | 高 | 低 | 最低 |
| **代表系统** | 早期RTB | 凤巢oCPC | Bidding Machine | 阿里/字节 | 学术 |

---

## 工程实现要点

### 离线训练 + 在线部署

**标准流程**：

```
┌─────────────────────────────────────────────┐
│                   离线阶段                    │
│                                               │
│  历史Log  →  Bid Simulator  →  RL训练         │
│  (竞价日志)   (回放/仿真)     (1000+ epochs)   │
│                     ↓                         │
│              策略模型 checkpoint               │
└─────────────────────────────────────────────┘
                       ↓ 部署
┌─────────────────────────────────────────────┐
│                   在线阶段                    │
│                                               │
│  实时特征 → 策略推理 → 出价决策 → 拍卖结果      │
│  (ms级)    (TensorFlow Serving / TorchServe) │
│                     ↓                         │
│              在线反馈 → 增量更新（小批量）      │
└─────────────────────────────────────────────┘
```

**关键挑战**：
1. **在线-离线一致性**：确保线上特征与训练特征一致（Feature Store）
2. **模型版本管理**：多个 Campaign 可能使用不同版本的策略
3. **延迟要求**：出价决策必须在 50-100ms 内完成（含特征获取）

### 环境模拟器（Bid Simulator）

**为什么需要模拟器**：
- 真实环境中 RL 探索成本极高（每次探索都要花真钱）
- 需要大量轨迹才能训练好策略（通常 10^6 次以上模拟竞价）

**模拟器构建方案**：

#### 方案一：Log Replay（日志回放）
```python
class LogReplaySimulator:
    def __init__(self, historical_logs):
        self.logs = historical_logs  # 历史竞价日志
    
    def step(self, bid, impression_idx):
        market_price = self.logs[impression_idx]['market_price']
        win = bid > market_price
        cost = market_price if win else 0  # 二价拍卖
        reward = self.logs[impression_idx]['value'] if win else 0
        return win, cost, reward
```

优点：忠实还原历史；缺点：无法模拟不同出价策略下的反事实结果

#### 方案二：生成式模拟器
```python
class GenerativeSimulator:
    def __init__(self):
        self.market_price_model = ...  # 学到的 Bid Landscape 模型
        self.ctr_model = ...           # CTR 预测模型
        self.cvr_model = ...           # CVR 预测模型
    
    def step(self, context, bid):
        market_price = self.market_price_model.sample(context)
        win = bid > market_price
        if win:
            click = bernoulli(self.ctr_model(context))
            if click:
                convert = bernoulli(self.cvr_model(context))
        cost = market_price if win else 0
        return win, cost, click, convert
```

#### 方案三：混合模拟器（生产首选）
- 市场价格用真实分布，转化概率用模型预测
- 定期用新数据更新模拟器，保持与真实环境同步

### 安全约束（出价上下限、预算保护）

**出价安全约束**：

```python
def safe_bid(raw_bid, context):
    # 1. 绝对上下限
    bid = clip(raw_bid, bid_floor=0.1, bid_cap=100.0)
    
    # 2. 相对基准限制（相对 oCPA 出价的倍数）
    base = ocpa_base_bid(context)
    bid = clip(bid, base * 0.3, base * 5.0)
    
    # 3. 预算保护（剩余预算 < 阈值时强制降价）
    if remaining_budget < emergency_threshold:
        bid = bid * 0.5
    
    # 4. 历史价格过滤（防止出价异常高）
    p99_market = get_market_price_percentile(context, 0.99)
    bid = min(bid, p99_market * 2)
    
    return bid
```

**预算保护机制**：

```
三级预算保护：
  Level 1（预算剩余 < 20%）：降低出价乘数至 0.8
  Level 2（预算剩余 < 5%）：强制进入保守模式，只投高质量流量
  Level 3（预算剩余 < 1%）：暂停出价
```

**监控与熔断**：
- 实时监控：CPA 偏离目标 > 50% 触发告警
- 自动熔断：异常出价连续 3 次触发 → 回退到 oCPA 保守策略
- 人工干预接口：运营可随时覆盖 RL 策略

---

## 工业常见做法

### 出价核心公式

广告系统中的出价通常采用如下统一公式：

$$
\text{bid}_{\text{final}} = k \times pCVR \times \text{CPA}_{\text{target}}
$$

其中：
- $k$ 是 Pacing 系数（由 PID 或对偶方法计算，通常 0.1 ~ 3.0）
- $pCVR$ 是转化率预估
- $CPA_{\text{target}}$ 是广告主设定的目标转化成本

**分层设计**：
```
bid_final = value × shading_factor × pacing_multiplier
          = (pCTR × pCVR × CPA_target) × shading_factor × pacing_multiplier
```

### 预算 Pacing 最佳实践

**实现层次**：
1. **全局预算分配**：campaign 级别每天设定预算，PID 控制器负责小时级调整
2. **实时监控**：5-15 分钟一个控制周期，对比实际消耗 vs 目标消耗
3. **多约束处理**：Min-pacing 策略，同时满足预算、CPA、频控等约束
   ```python
   # 多约束的 Min-Pacing
   multiplier = min(
       budget_multiplier,      # 预算约束的乘数
       cpa_multiplier,         # CPA 约束的乘数
       freq_control_mult       # 频控约束的乘数
   )
   ```

4. **反防冲**（Anti-windup）：预算耗尽后停止积分项累积，防止积分饱和

### RL 部署安全防护

**三层安全防护**：

1. **约束 MDP（Constrained MDP）**：在奖励函数中加入约束惩罚项
   ```
   r_t = value_t - λ_budget × cost_t
   ```

2. **硬约束后处理**：在 Actor 输出上做 clip
   ```
   bid = min(actor_output, remaining_budget / expected_win_rate)
   ```

3. **监控熔断**：
   - CPA 偏离目标 > 50% 触发告警
   - 异常出价连续 3 次 → 回退到 oCPA 保守策略
   - 设置出价上界：min(actor_output, p99_historical_price × 2)

---

## 参考文献

1. **Bidding Machine**: Kan Ren et al. "Learning to Bid for Directly Optimizing Profits in Display Advertising." *IEEE TKDE*, 2018. [arXiv:1803.02194]

2. **Real-Time Bidding with RL**: Han Cai et al. "Real-Time Bidding by Reinforcement Learning in Display Advertising." *WSDM*, 2017.

3. **Budget-Constrained Bidding via RL**: Di Wu et al. "Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising." *CIKM*, 2018.

4. **Deep RL for Sponsored Search**: Jun Zhao et al. "Deep Reinforcement Learning for Sponsored Search Real-time Bidding." *KDD*, 2018.

5. **MARL RTB**: Junqi Jin et al. "Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising." *CIKM*, 2018.

6. **Smart Pacing**: Jian Xu et al. "Smart Pacing for Effective Online Ad Campaign Optimization." *KDD*, 2015.

7. **Bid Landscape**: Ying Cui et al. "Bid Landscape Forecasting in Online Advertising Auction: A Regression Approach." *KDD*, 2011.

8. **oCPC in Taobao**: Han Zhu et al. "Optimized Cost per Click in Taobao Display Advertising." *KDD*, 2017.

9. **ESMM**: Xiao Ma et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." *SIGIR*, 2018.

10. **Alimama Auto Bidding**: 阿里妈妈技术团队. 《智能出价技术演进》内部分享, 2021.

11. **wzhe06/Ad-papers**: https://github.com/wzhe06/Ad-papers — 计算广告论文合集

12. **Display Advertising with RTB**: Jun Wang et al. "Display Advertising with Real-Time Bidding and Behavioural Targeting." *Foundations & Trends in IR*, 2017.

13. **Lagrangian Relaxation for Bidding**: Claudio Fischetti et al. KKT Optimality in Budget-Constrained Bidding, *Operations Research*, 2019.

14. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms." *arXiv*, 2017.

15. **TD3**: Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*, 2018.

> 📝 面试考点见：[ads_qa_extracted.md](../../interview/ads_qa_extracted.md)
