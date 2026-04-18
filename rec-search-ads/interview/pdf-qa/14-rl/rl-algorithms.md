# 推荐系统中的强化学习算法

## 1. MDP 建模基础

### 1.1 推荐系统的 MDP 定义
```
五元组 (S, A, P, R, gamma)：

State（状态）：
  - 用户画像（年龄、性别、偏好标签）
  - 历史行为序列（最近 N 次交互的物品 Embedding 序列）
  - 上下文信息（时间、设备、位置）
  - 编码方式：RNN/GRU/Transformer 将变长序列编码为定长状态向量

Action（动作）：
  - 推荐的物品或物品列表
  - 挑战：动作空间 = 全量物品库（百万~亿级）

Transition（状态转移）：
  - 用户状态随交互演变（看了 A 后兴趣变化）
  - 部分可观测：只能看到用户的显式反馈

Reward（奖励）：
  - 即时奖励：点击(+1)、购买(+5)、停留时长(+0.01*秒)
  - 负反馈：快速划过(-0.5)、不感兴趣(-1)

Gamma（折扣因子）：
  - 通常 0.9-0.99，平衡短期收益与长期留存
```

### 1.2 与传统推荐的核心区别
```
传统推荐（监督学习）：
  - 目标：max P(click | user, item)
  - 假设：样本独立同分布 (i.i.d.)
  - 局限：贪心选即时最优，不考虑对用户状态的影响

RL 推荐：
  - 目标：max E[sum(gamma^t * r_t)]（累积奖励最大化）
  - 核心：考虑当前推荐对用户未来状态的影响
  - 优势一：探索-利用平衡，主动发现用户潜在兴趣
  - 优势二：优化长期指标（留存率、LTV），非短期点击

何时需要 RL？
  - 用户兴趣会因推荐内容而变化（非稳态用户）
  - 短期最优 ≠ 长期最优（如推荐低质内容短期 CTR 高但伤留存）
  - 需要主动探索用户兴趣边界
```

---

## 2. Q-Learning / DQN

### 2.1 基本原理
```
Q 函数：Q(s, a) = E[累积奖励 | 在状态 s 执行动作 a，之后按最优策略行动]

Bellman 方程：
  Q(s, a) = r + gamma * max_a' Q(s', a')

DQN：用深度神经网络逼近 Q 函数
  输入：状态 s（用户表示向量）
  输出：所有动作的 Q 值（或给定 (s,a) 输出标量）

训练流程：
  1. 收集交互数据存入 Replay Buffer
  2. 随机采样 mini-batch
  3. 计算 TD 目标：y = r + gamma * max_a' Q_target(s', a')
  4. 最小化 MSE(Q(s,a), y)
  5. 定期更新目标网络（soft update 或 hard copy）
```

### 2.2 推荐系统中的挑战与解决
```
挑战一：动作空间巨大
  问题：百万级物品，无法枚举 Q(s, a) 对所有 a

  解决方案 A：候选集预筛选 + DQN 重排
    - 召回层先缩小到千级候选集
    - DQN 在候选集上选最优动作
    - 优点：简单实用
    - 缺点：最优动作可能不在候选集中

  解决方案 B：连续动作空间
    - 策略网络输出一个连续的"用户兴趣向量"
    - 用 ANN 检索最近的物品作为推荐
    - 优点：不受候选集限制
    - 缺点：连续向量到离散物品的映射有偏

  解决方案 C：分层动作空间
    - 第一层：选品类/话题（数十个）
    - 第二层：在品类内选具体物品（数百个）
    - 分层 DQN 分别训练
    - 优点：有效缩小搜索空间

  解决方案 D：Wolpertinger 策略
    - 在连续空间输出 proto-action
    - 找到最近的 K 个离散物品
    - 在 K 个物品中选 Q 值最高的
    - 结合连续和离散的优势

挑战二：状态表示
  问题：用户状态包含变长的历史行为序列

  解决方案：
    - GRU 编码：将行为序列压缩为固定维状态向量
    - Transformer：自注意力捕捉长程依赖
    - 实践中 128-256 维已足够
```

### 2.3 DQN 改进技术
```
Double DQN：
  问题：标准 DQN 过估计 Q 值（max 操作导致正偏差）
  解决：选动作用主网络，评估用目标网络
  y = r + gamma * Q_target(s', argmax_a Q_main(s', a))

Dueling DQN：
  将 Q(s,a) 分解为 V(s) + A(s,a)
  V(s)：状态价值（与动作无关）
  A(s,a)：优势函数（选这个动作相对平均好多少）
  在推荐中：V(s) 捕捉用户当前活跃度，A(s,a) 捕捉物品匹配度

Prioritized Experience Replay：
  重要样本（TD 误差大）被更频繁采样
  在推荐中：稀有行为（购买、差评）优先学习
```

---

## 3. Policy Gradient / REINFORCE

### 3.1 基本原理
```
直接参数化策略 pi_theta(a|s)，无需学 Q 函数

策略梯度定理：
  grad(J) = E[sum_t gamma^t * G_t * grad(log pi(a_t|s_t; theta))]

REINFORCE 算法：
  for each episode:
      收集轨迹 tau = (s_0, a_0, r_0, ..., s_T, a_T, r_T)
      for t = 0 to T:
          G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}    # 未来累积回报
          theta += alpha * G_t * grad(log pi(a_t|s_t))

直觉：
  G_t > 0 的动作 → 增大其概率
  G_t < 0 的动作 → 减小其概率
```

### 3.2 在推荐系统中的优势
```
1. 天然适合大规模离散动作空间
   - 输出概率分布 pi(a|s)，不需要枚举所有 Q(s,a)
   - 可以用 softmax 策略对候选集打分

2. 支持随机策略
   - 按概率采样推荐 → 天然实现探索
   - 无需额外的 epsilon-greedy 等探索机制

3. 可直接优化列表级指标
   - 将推荐列表的生成建模为序列决策
   - 每步选一个物品加入列表
   - 奖励设计可融入多样性、新颖性等列表级指标
```

### 3.3 降方差技术
```
问题：REINFORCE 方差极高，收敛慢

方法一：Baseline 减法
  使用 baseline b(s) 减小方差：
  grad(J) = E[sum_t (G_t - b(s_t)) * grad(log pi(a_t|s_t))]
  常用 baseline：V(s) 的估计值（需额外训练价值网络）

方法二：GAE（Generalized Advantage Estimation）
  A_t = sum_{l=0}^{infinity} (gamma*lambda)^l * delta_{t+l}
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  lambda 控制偏差-方差权衡：lambda=0 低方差高偏差，lambda=1 高方差低偏差

方法三：因果性裁剪
  过去的奖励不影响未来的策略梯度
  G_t 只用 t 之后的奖励，不用之前的
  减少不相关信号的方差贡献
```

---

## 4. Actor-Critic

### 4.1 架构设计
```
Actor（策略网络）：
  输入：状态 s（用户表示）
  输出：动作概率分布 pi(a|s)
  作用：决定推荐什么物品

Critic（价值网络）：
  输入：状态 s（或 s,a）
  输出：V(s) 或 Q(s,a)
  作用：评估推荐质量

更新规则：
  Critic 更新：最小化 TD 误差 (r + gamma*V(s') - V(s))^2
  Actor 更新：沿 Advantage 方向更新策略
    A(s,a) = Q(s,a) - V(s) 或 用 TD 误差近似
    theta_actor += alpha * A(s,a) * grad(log pi(a|s))
```

### 4.2 A2C / A3C
```
A2C（Advantage Actor-Critic）：
  同步多环境并行收集数据
  用 Advantage 替代 G_t 更新 Actor
  更稳定的训练

A3C（Asynchronous Advantage Actor-Critic）：
  多个 worker 异步并行与环境交互
  各自计算梯度更新全局参数
  优势：充分利用多核 CPU
  在推荐中：各 worker 对应不同用户群或场景
```

### 4.3 PPO（Proximal Policy Optimization）
```
核心思想：限制策略更新幅度，避免大步更新导致崩溃

目标函数：
  L = E[min(
    r_t * A_t,                          # 标准目标
    clip(r_t, 1-epsilon, 1+epsilon) * A_t  # 截断目标
  )]

  r_t = pi_new(a|s) / pi_old(a|s)  # 策略比率
  epsilon 通常取 0.1-0.2

在推荐系统中的优势：
  1. 训练稳定性好 → 适合在线学习
  2. 样本效率较高 → 适合数据成本高的场景
  3. 超参数少 → 调参成本低
```

### 4.4 SAC（Soft Actor-Critic）
```
在标准 Actor-Critic 基础上加入最大熵正则化：
  J = E[sum(r_t + alpha * H(pi(·|s_t)))]

H 是策略的熵，鼓励探索多样的动作
alpha 自动调节（target entropy）

在推荐中的意义：
  - 最大熵 → 推荐多样性
  - 自动探索-利用平衡
  - 避免策略过早收敛到狭窄的物品子集
```

---

## 5. 在线 vs 离线 RL

### 5.1 在线 RL
```
特点：与真实用户实时交互收集数据

优势：
  - 数据分布与当前策略一致（on-policy）
  - 可以直接观测策略效果

风险：
  - 探索阶段可能损害用户体验
  - 推荐低质量物品导致用户流失

实践方案：
  1. 小比例流量在线探索（1-5%）
  2. 安全约束：限制与当前最优策略的偏差范围
  3. 混合策略：90% 确定性推荐 + 10% RL 探索
```

### 5.2 离线 RL（Batch RL / Offline RL）
```
特点：从历史日志数据中学习，不与用户交互

核心挑战：分布偏移（Distribution Shift）
  - 日志数据由旧策略生成
  - 新策略可能选择旧策略从未选过的动作
  - 对这些动作的 Q 值估计不可靠 → 过估计 → 策略崩溃

解决方案：

BCQ（Batch-Constrained Q-Learning）：
  - 限制新策略只选择日志中出现过的动作
  - 用生成模型估计行为策略的动作分布
  - 只在行为策略概率高的动作中选最优

CQL（Conservative Q-Learning）：
  - 对日志中未出现的动作惩罚 Q 值
  - L = L_TD + alpha * E_a~uniform[Q(s,a)] - alpha * E_a~data[Q(s,a)]
  - 效果：学到的 Q 值是真实 Q 值的下界 → 保守但安全

IQL（Implicit Q-Learning）：
  - 避免显式评估日志外动作的 Q 值
  - 用分位数回归估计 V(s)（期望 vs 最优值的权衡）
  - 适合推荐场景：不需要穷举动作空间
```

---

## 6. 多臂赌博机（MAB）在推荐中的应用

### 6.1 经典方法对比
```
epsilon-Greedy：
  以概率 1-epsilon 选当前估计最优，epsilon 概率随机探索
  epsilon 通常随时间衰减：epsilon_t = epsilon_0 / sqrt(t)
  简单但探索效率低

UCB（Upper Confidence Bound）：
  选择 argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]
  乐观面对不确定性：探索不足的臂有更大的置信上界
  确定性策略，regret bound: O(sqrt(K*T*ln(T)))

Thompson Sampling：
  为每个臂维护后验分布 Beta(alpha, beta)
  每轮从后验采样 → 选最大采样值的臂
  实践中 regret 优于 UCB
  自然适应非平稳环境（如用户偏好变化）
```

### 6.2 上下文赌博机
```
LinUCB：
  假设奖励与上下文特征线性相关：r = theta^T * x + noise
  为每个臂维护参数的后验：
    A_a += x * x^T
    b_a += r * x
    theta_a = A_a^{-1} * b_a
  选择：argmax_a [theta_a^T * x + alpha * sqrt(x^T * A_a^{-1} * x)]

  适用场景：
  - 冷启动推荐
  - 新闻/广告推荐（物品快速更新）
  - 特征维度不太高的场景

Neural Contextual Bandit：
  用深度网络替代线性模型建模 r = f(x; theta)
  不确定性估计：MC Dropout / 深度集成 / 深度核学习
  适合特征空间复杂的推荐场景
```

### 6.3 MAB vs 完整 RL
```
                    MAB             完整 RL
状态转移            无               有
时间跨度            单步             多步
建模复杂度          低               高
适用场景            独立决策         序列决策
推荐应用            广告竞价         会话推荐

何时用 MAB？ → 决策之间无依赖（如首页 Banner 位推荐）
何时用 RL？  → 当前推荐影响用户未来状态（如视频流推荐）
```

---

## 7. 面试高频问题

```
Q: DQN 在推荐中的主要瓶颈是什么？
A: 动作空间巨大。需要枚举所有物品的 Q 值才能选最优，百万级物品不可行。
   解决：候选集预筛选、连续动作空间+ANN、分层动作空间。

Q: Policy Gradient vs DQN 在推荐场景中如何选择？
A: PG 更适合大动作空间（输出概率分布不需枚举）和需要随机策略的场景。
   DQN 更适合动作空间可控的重排阶段（候选集已缩小到百级）。
   实践中 Actor-Critic 类方法最常用（兼顾两者优点）。

Q: 离线 RL 在推荐中的最大风险？
A: 分布偏移导致 Q 值过估计。模型在日志中未见过的动作上给出过高 Q 值，
   实际执行效果远差于预期。用 CQL/BCQ 等保守方法约束。

Q: Thompson Sampling 为什么比 UCB 在推荐中更实用？
A: 1) 天然支持随机探索（UCB 是确定性策略）
   2) 更好适应非平稳环境（用户偏好变化）
   3) 实现简单且工程友好（只需维护 alpha/beta 参数）
   4) 实证 regret 通常优于 UCB

Q: 在线 RL 在推荐系统中如何保证安全？
A: 1) 小比例流量做探索（1-5%）
   2) 安全约束：新策略与基线策略的 KL 散度 < 阈值
   3) 推荐质量下限：RL 推荐的物品必须通过质量门槛
   4) 熔断机制：探索组 CTR 降幅超阈值则自动回退
```
