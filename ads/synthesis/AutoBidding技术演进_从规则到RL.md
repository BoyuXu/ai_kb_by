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
$$bid^* = v \cdot pCTR \cdot pCVR$$
- 出价 = 价值 × 点击率 × 转化率

### 2. 预算约束
$$\sum_{t=1}^T c_t \leq B$$
- 总花费不超过预算 B

### 3. Lagrangian 松弛
$$L = \sum_t v_t x_t - \lambda(\sum_t c_t x_t - B)$$
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
- 每次展示的转化价值 $v_i = \text{pCTR}_i \times \text{pCVR}_i \times \text{CPA\_target}$
- 赢得时的支付（二价拍卖中约为对手最高价 $m_i$）

**优化问题**：

$$\max_{b_1, \ldots, b_n} \sum_{i=1}^{n} v_i \cdot w_i(b_i)$$

$$\text{s.t.} \quad \sum_{i=1}^{n} m_i \cdot w_i(b_i) \leq B$$

$$b_i \geq 0, \quad \forall i$$

**拉格朗日函数**：

$$\mathcal{L}(b, \lambda) = \sum_{i=1}^{n} v_i \cdot w_i(b_i) - \lambda \left( \sum_{i=1}^{n} m_i \cdot w_i(b_i) - B \right)$$

**KKT 条件**：

对每个 $b_i$ 求偏导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial b_i} = v_i \cdot w_i'(b_i) - \lambda \cdot m_i \cdot w_i'(b_i) = 0$$

$$w_i'(b_i)(v_i - \lambda \cdot m_i) = 0$$

由于 $w_i'(b_i) > 0$（赢得概率关于出价单调递增），得：

$$v_i = \lambda \cdot m_i$$

等价于：

$$b_i^* = v_i / \lambda = \frac{\text{pCTR}_i \times \text{pCVR}_i \times \text{CPA\_target}}{\lambda}$$

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

## 第三阶段：基于优化的智能出价（2018-2021）

### 线性规划（LP）出价：Bid Landscape + 期望收益最大化

**Bid Landscape 建模**：

Bid Landscape 是对市场竞价环境的建模——给定一个广告机会，不同出价 b 对应的赢得概率 $P(\text{win}|b)$。

常用模型：
- **Log-normal 分布**：$P(\text{win}|b) = \Phi\left(\frac{\ln b - \mu}{\sigma}\right)$
- **Survival 分析**：将竞价市场价格建模为生存时间

**LP 出价框架**：

在 Bid Landscape 已知的前提下，对每个广告机会建立 LP：

$$\max \sum_{i} v_i \cdot P_i(\text{win}|b_i)$$

$$\text{s.t.} \quad \sum_{i} \mathbb{E}[\text{cost}_i | b_i] \leq B$$

$$\text{其中 } \mathbb{E}[\text{cost}_i | b_i] = \int_0^{b_i} x \cdot f_i(x) dx$$

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

$$b^* = v - \frac{1 - F(b^*)}{f(b^*)}$$

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

$$\lambda^{(t+1)} = \lambda^{(t)} - \alpha \cdot \nabla_\lambda \mathcal{L}$$

$$\nabla_\lambda \mathcal{L} = B - \sum_i m_i \cdot w_i(b_i^*)$$

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

$$\max_{b_i} \sum_i v_i w_i(b_i)$$

$$\text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B \quad (\text{预算})$$

$$\quad \quad \sum_i v_i w_i(b_i) / \sum_i c_i w_i(b_i) \geq \text{ROI\_target} \quad (\text{ROI约束})$$

拉格朗日：$\mathcal{L} = \sum_i (v_i - \lambda_1 c_i - \lambda_2(c_i \cdot \text{ROI} - v_i)) w_i(b_i) + \lambda_1 B$

最优出价：$b_i^* = \frac{(1+\lambda_2)v_i - (\lambda_1 + \lambda_2 \cdot \text{ROI})c_i}{\text{适当系数}}$

### 约束优化框架：max ROI s.t. budget constraint

**通用框架**：

$$\max_{b} \text{ROI}(b) = \frac{\sum_i v_i w_i(b_i)}{\sum_i c_i w_i(b_i)}$$

$$\text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B$$

**分形式转化**（Dinkelbach 方法）：

迭代求解，每轮固定 ROI 值 $\rho^{(k)}$，求解：

$$\max_b \sum_i (v_i - \rho^{(k)} c_i) w_i(b_i) \quad \text{s.t.} \quad \sum_i c_i w_i(b_i) \leq B$$

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

## 面试高频考点（8-10题 Q&A）

### Q1：oCPC 的出价公式怎么推导？

**A**：oCPC 的目标是在点击付费的交易模式下，让系统自动优化 CPA。

推导过程：
- 广告主设定 CPA 目标 = T
- 期望每次转化花费 = T
- 期望每次点击花费（CPC）= T × pCVR（因为点击后有 pCVR 概率转化）
- 期望每次展示出价（eCPM/1000）= CPC × pCTR = T × pCVR × pCTR

所以：**bid = CPA_target × pCTR × pCVR**

这就是系统替广告主出的 eCPM 价格，广告主按实际 CPC 计费。

---

### Q2：PID 算法在 Pacing 中怎么用？P/I/D 各有什么作用？

**A**：PID 控制器调整出价乘数，让实际消耗曲线跟随目标消耗曲线。

- **P（比例）**：对当前消耗偏差即时响应。偏差越大，调整越大。过大会导致震荡。
- **I（积分）**：累积历史偏差，消除稳态误差（如一直消耗不足）。过大会超调。
- **D（微分）**：预测变化趋势，提前做出反应，抑制振荡。对噪声敏感。

实践中 I 项需要 anti-windup（预算耗尽后停止积分），D 项通常使用滤波后的值。

---

### Q3：KKT 条件推导出来的最优出价公式是什么？有什么直觉？

**A**：

设 $v_i$ 是第 $i$ 个 impression 的期望价值，$\lambda$ 是预算约束的拉格朗日乘子。

最优出价：$b_i^* = v_i / \lambda$

**直觉**：
- λ 代表"1元预算能带来的边际价值"，即预算的影子价格
- 每个 impression 的最优出价 = 它的价值 / 预算边际价值
- λ 越大（预算越紧张），出价越低（更保守）
- λ 越小（预算充裕），出价越接近真实价值

---

### Q4：为什么用 RL 而不用直接优化（LP / 对偶方法）？

**A**：

直接优化的局限性：
1. **序列性**：LP 假设每次竞价独立，不考虑跨时间步的影响（预算分配到未来）
2. **非平稳性**：市场价格、流量质量实时变化，静态的 λ 无法自适应
3. **模型误差**：依赖 Bid Landscape 和 CVR 预测的精度，误差会传播到出价
4. **复杂目标**：难以优化长期累积价值，LP 只能优化当前期望收益

RL 的优势：
1. 将出价建模为 MDP，自然处理时间序列决策
2. 通过反馈（reward）自适应市场变化
3. 不依赖 Bid Landscape 的精确建模，端到端学习
4. 可以学习跨时间步的最优预算分配策略

**但RL也有代价**：样本效率低、冷启动难、可解释性差、训练不稳定。

---

### Q5：RL 出价在非平稳环境下怎么处理？

**A**：非平稳是 RL 出价最核心的挑战，有以下解决方案：

1. **在线持续学习**：不断用最新数据更新策略，使用 PPO 等 on-policy 算法
2. **时间特征入 state**：加入时间编码（小时/星期/节假日 flag），让模型感知环境状态
3. **元学习（MAML）**：学习快速适应新环境的元策略，用少量新数据即可适应
4. **保守更新（Conservative Update）**：限制每次策略更新的幅度，防止灾难性遗忘
5. **环境检测**：监控状态分布变化（KL 散度），分布漂移时触发重训练
6. **分域训练**：针对不同时间段（早/晚/大促）分别训练子策略，上线时做路由

---

### Q6：RL 出价的 state/action/reward 如何设计？

**A**（已在第四阶段详细展开，此处给出简洁版）：

- **State**：当前机会特征（pCTR/pCVR/质量分）+ 预算状态（剩余预算比/剩余时间比）+ 市场状态（过去赢率/平均成本）+ 累计效果（当前CPA/ROI偏差）
- **Action**：出价乘数（连续）或离散档位（离散），通常选出价乘数（稳定性好）
- **Reward**：当次转化价值 - 成本，加预算惩罚项；延迟转化用 pCVR 预估即时 reward

---

### Q7：Bid Shading 是什么？为什么一价拍卖需要它？

**A**：

- **背景**：2019 年谷歌、2017 年 AppNexus 开始把广告拍卖从二价改为一价
- **二价拍卖**：赢家支付第二高价，真实出价是最优策略（Vickrey定理）
- **一价拍卖**：赢家支付自己出的价，真实出价会浪费（出了100但只需付60才能赢）

Bid Shading 是在一价拍卖中"压低出价"的策略：

```
shaded_bid = α × true_value  （α ∈ (0,1) 是压价系数）
```

压价系数 α 的估计：通过 Bid Landscape 模型估计对手出价分布，求解最大化期望利润的出价。

工程上：每个流量源维护独立的分布模型，定期更新。

---

### Q8：多智能体 RL (MARL) 在出价中有什么特殊挑战？

**A**：

1. **非平稳性加剧**：其他 Agent 的策略在学习过程中不断变化，环境更不稳定
2. **信用分配**：难以区分是自己的决策还是对手的决策导致了奖励
3. **通信限制**：真实竞价中各 DSP 无法通信，只能通过市场价格间接感知对手
4. **均衡不唯一**：博弈论中可能存在多个纳什均衡，收敛点不确定
5. **计算复杂度**：Agent 数量增加，联合动作空间指数爆炸

**常用解法**：
- CTDE（集中训练，分散执行）：训练时 Critic 看全局，执行时只用本地信息
- 均场近似：用平均行为代替对手建模，O(n²) → O(n)
- Self-play：让多个 Agent 互相对抗训练

---

### Q9：如何在预算约束下保证 RL 出价的安全性？

**A**：

**三层安全防护**：

1. **约束 MDP（Constrained MDP）**：在奖励函数中加入约束惩罚项，让 Agent 学习遵守约束
   ```
   r_t = value_t - λ_budget × cost_t
   ```

2. **硬约束后处理**：在 Actor 输出的出价上做 clip，确保不超过预算余量
   ```
   bid = min(actor_output, remaining_budget / expected_win_rate)
   ```

3. **监控熔断**：实时监控 CPA/ROI 偏差，超阈值时切换到保守策略或人工干预

---

### Q10：如何评估 Auto Bidding 系统的上线效果？

**A**：

**离线评估**：
- 在历史数据上对比不同策略的 KPI（GMV、CPA、ROI、预算完成率）
- 注意：离线评估存在**反事实**问题，出价变了会改变赢率，历史日志无法直接回放

**在线评估（A/B 测试）**：
- 流量分层：以 Campaign 为粒度分桶（不能以 impression 为粒度，否则干扰）
- 实验周期：至少 7 天（需覆盖周末流量模式）
- 观测指标：
  - 主指标：广告主 ROI 提升率、GMV 变化
  - 辅助指标：CPA 是否接近目标、预算消耗率、赢率变化
- 避免：广告主的有限预算下，RL 抢占了所有优质流量但让对照组 KPI 变差

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

---

*笔记完成时间：2026-03-16 | MelonEggLearn*

### Q1: 广告系统的全链路延迟约束是什么？
**30秒答案**：端到端 <100ms：召回 <10ms，粗排 <20ms，精排 <50ms，竞价 <10ms。关键优化：模型蒸馏/量化、特征缓存、异步预计算。

### Q2: 广告和推荐的核心技术差异？
**30秒答案**：①校准要求不同（广告需绝对概率，推荐只需排序）；②约束不同（广告有预算/ROI 约束）；③更新频率不同（广告更高频）；④数据不同（广告有竞价日志）。

### Q3: 广告系统的数据闭环怎么做？
**30秒答案**：展示日志→点击/转化回收→特征构建→模型训练→线上服务。关键：①归因窗口设置（7天/30天）；②延迟转化处理；③样本权重修正；④在线学习增量更新。

### Q4: 广告系统如何处理数据稀疏问题？
**30秒答案**：①多任务学习（用 CTR 辅助 CVR）；②数据增广（LLM 生成/对比学习）；③迁移学习（从相似领域迁移）；④特征工程（高阶交叉特征增加信号密度）。

### Q5: 隐私计算对广告系统的影响？
**30秒答案**：三方 Cookie 消亡后：①联邦学习（多方数据联合建模不出域）；②差分隐私（加噪保护用户数据）；③安全多方计算；④First-party Data 价值提升。挑战：效果和隐私的 trade-off。

### Q6: 广告 CTR 模型的在线 A/B 怎么做？
**30秒答案**：分流：按用户 hash 分桶，保证同一用户始终在同一组。核心指标：CTR、CVR、RPM（千次展示收入）、广告主 ROI。时长：至少 7 天（覆盖周效应）。注意：广告有预算效应，需要同时监控广告主消耗。

### Q7: 广告特征工程有哪些核心特征？
**30秒答案**：①用户画像（年龄/性别/兴趣标签）；②广告属性（品类/出价/预算/素材质量）；③上下文（时间/设备/位置）；④交叉统计（用户×品类的历史 CTR）；⑤实时特征（最近 N 次曝光/点击）。

### Q8: 广告模型的样本构建有什么特殊之处？
**30秒答案**：①曝光不等于展示（广告被加载但用户可能没看到）；②延迟转化（点击后数天才转化）；③竞价日志（不仅有展示结果，还有出价/竞争信息）；④样本加权（不同位置的曝光权重不同）。

### Q9: 自动出价和手动出价的效果对比？
**30秒答案**：数据显示自动出价通常比手动出价提升 15-30% ROI。原因：①实时调整能力（秒级 vs 天级）；②全局优化（考虑跨时段预算分配）；③数据驱动（比人工经验更精准）。但冷启动期手动出价更稳定。

### Q10: 广告系统的降级策略？
**30秒答案**：①模型服务不可用：回退到规则排序（按出价×历史 CTR）；②特征服务延迟：用缓存特征替代实时特征；③预算系统故障：按历史消耗速度限流；④全系统故障：展示自然内容，不展示广告。
