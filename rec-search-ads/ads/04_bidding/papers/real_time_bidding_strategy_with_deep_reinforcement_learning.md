# Real-time Bidding Strategy with Deep Reinforcement Learning
> 来源：arxiv/1906.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
实时竞价（RTB）中的出价策略优化：
- 传统出价：固定出价或简单公式（bid = base_ctr × base_cvr × 1000/target_cpa）
- 无法捕获预算动态：预算快消完时应降价，预算充足时可激进
- 忽略时间因素：黄金时段（晚8-10点）流量质量高，应提高出价
- 跨时间步依赖：当前出价影响后续预算，需序列决策

## 核心方法与创新点
**DRL-based RTB（Deep Reinforcement Learning for Real-Time Bidding）**

**MDP 建模：**
```
状态 s_t = [remaining_budget, remaining_time, historical_win_rate, 
            predicted_ctr, predicted_cvr, traffic_features, hour_of_day, ...]
动作 a_t = bid_price ∈ [0, max_bid]  # 连续动作空间
奖励 r_t = win × cvr_reward - bid_cost
目标: maximize Σ_t r_t  s.t.  Σ_t cost_t ≤ Budget
```

**Actor-Critic 架构（DDPG）：**
```python
# Actor：输出出价
bid = Actor(state; θ_actor)  # 输出连续出价值

# Critic：评估 Q 值
Q(s, a) = Critic(state, bid; θ_critic)

# 更新
L_critic = MSE(Q(s,a), r + γ·Q(s',Actor(s')))
L_actor  = -E[Q(s, Actor(s))]
```

**预算约束处理：**
```python
# Lagrangian 约束
L_constrained = L_actor + λ·(Σ cost - Budget)
# λ 由对偶梯度上升更新
```

**探索噪声（Ornstein-Uhlenbeck）：**
```
bid_explore = bid + OU_noise(θ, σ)  # 时序相关探索噪声
```

## 实验结论
- iPinYou 竞价数据集：
  - 总转化数 +8.2%（vs 线性出价策略）
  - 相同预算下获得转化 +12.1%（vs 固定 CPA 出价）
  - 预算利用率（消耗/预算）：96.3%（vs 规则策略 78%）
- 不同预算规模：低预算广告主受益更大（+15%）

## 工程落地要点
1. **状态特征实时性**：remaining_budget 实时更新（Redis），历史 win_rate 滑动窗口
2. **动作空间离散化**：将连续出价离散化为 N 档（如 10-1000 分 100 档），转化为 DQN
3. **离线仿真训练**：用历史竞价日志构建模拟器，安全探索
4. **安全出价上限**：actor 输出 × base_bid，不超过设定上限
5. **延迟归因**：转化延迟（T+1/T+7），用即时代理奖励（预测 CVR）补偿

## 常见考点
**Q1: RTB 为什么适合用 RL 建模？**
A: RTB 天然是序列决策问题：每次竞价都影响预算消耗，进而影响后续竞价能力；有明确的奖励（转化）和约束（预算）；状态（剩余预算/时间/流量质量）与动作（出价）的关系复杂，难以用规则刻画。

**Q2: DDPG vs DQN，哪个更适合出价问题？**
A: 出价是连续动作空间，DQN 需要离散化（信息损失）；DDPG 直接输出连续出价，更精确。但 DDPG 训练不稳定，实践中常用 SAC（Soft Actor-Critic）作为替代，探索更好，稳定性更高。

**Q3: 预算约束如何在 RL 中处理？**
A: ①约束 MDP：Lagrangian Relaxation，将预算约束转化为惩罚项 ②安全层：Actor 输出后加预算感知过滤层，根据剩余预算缩放出价 ③状态编码：将 remaining_budget/total_budget 作为状态特征，让 Agent 自适应。

**Q4: 离线历史日志训练 RTB 模型有什么偏差？**
A: Counterfactual 问题：历史日志记录的是旧策略下的竞价结果，新策略的出价可能不在历史分布中（未竞得的曝光没有奖励信号）。解决：Offline RL（批量离线 RL）+ IPS 校正 + Conservative Q-Learning。

**Q5: 多广告主共享预算的 RTB 如何协调？**
A: ①独立 Agent：每个广告主独立 RL Agent，互不干扰 ②共享价值网络：共享 Q 函数，私有 Actor ③集中式 RL：平台层面全局优化（复杂但效果好）。实践中通常用独立 Agent + 平台层预算分配规则。

## 模型架构详解

### 出价策略框架
- **状态空间**：预算剩余比例、当前时段、历史消耗速率、市场竞争强度
- **动作空间**：出价倍数（bid multiplier）或绝对出价值
- **奖励函数**：转化价值 × 约束满足度（预算/ROI约束的拉格朗日松弛）

### 预算控制模块
- PID 控制器：基于消耗偏差的反馈控制
- 在线学习：根据实时竞价结果动态调整出价
- 约束优化：拉格朗日乘子法处理预算和ROI约束

### 离线训练 → 在线部署
- 离线：历史竞价日志训练策略网络
- 在线：A/B Test 验证 + 安全兜底（出价上下限）
- 探索与利用：ε-greedy 或 Thompson Sampling


## 与相关工作对比

| 维度 | 本文方法 | 传统方法 | 优势 |
|------|---------|---------|------|
| 出价策略 | 数据驱动/RL | 规则/PID | 自适应市场变化 |
| 约束处理 | 软约束+安全网 | 硬阈值 | 更好的预算利用率 |
| 探索策略 | 结构化探索 | ε-greedy | 降低探索代价 |
| 多目标 | 帕累托优化 | 加权求和 | 平衡多方利益 |


## 面试深度追问

- **Q: 如何在出价中平衡探索与利用？**
  A: 1) ε-greedy + 退火：初期高探索逐步收敛；2) Thompson Sampling：对不确定性高的出价空间增加探索；3) 安全探索：设置出价上下限避免极端损失。

- **Q: 预算消耗不均匀（流量波动）怎么处理？**
  A: 1) 分时段预算分配（基于历史流量模式）；2) PID 控制器实时调整出价；3) Pacing 算法平滑消耗曲线。关键指标：消耗平滑度、转化效率。

- **Q: 离线评估出价策略的挑战？**
  A: 竞价环境的反事实问题——改变出价会影响赢标概率和最终价格。解决方案：1) IPW（逆倾向加权）；2) 市场模拟器（基于历史竞价分布）；3) 离线 Replay + 保守估计。

- **Q: 多约束优化（预算+ROI+成本）如何实现？**
  A: 拉格朗日对偶方法，将约束转化为惩罚项。在线更新拉格朗日乘子（primal-dual 迭代），需要平衡约束满足度和目标优化。
