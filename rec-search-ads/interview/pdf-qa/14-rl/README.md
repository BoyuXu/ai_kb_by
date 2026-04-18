# 强化学习与序列决策推荐（RL in RecSys）

## 本章文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [README.md](README.md) | 总览与核心概念速查 | ~190 |
| [rl-algorithms.md](rl-algorithms.md) | MDP建模/DQN/Policy Gradient/Actor-Critic/在线vs离线RL/MAB | ~290 |
| [reward-design.md](reward-design.md) | 奖励函数设计/稀疏奖励/延迟奖励/多目标/探索-利用/OPE | ~270 |

---

## 1. 推荐系统的 MDP 建模

### 1.1 基本框架
将推荐过程建模为马尔可夫决策过程（MDP）：
- 状态（State）：用户画像 + 历史行为序列 + 上下文（时间、设备等）
- 动作（Action）：推荐的物品或物品列表
- 奖励（Reward）：用户反馈（点击、购买、停留时长等）
- 转移（Transition）：用户状态随交互演变
- 折扣因子（gamma）：平衡即时收益与长期价值

### 1.2 与传统推荐的本质区别
传统推荐 -> 预测 P(click|user, item)，贪心选最高分
RL 推荐 -> 最大化长期累积奖励 sum(gamma^t * r_t)

核心优势：
- 考虑推荐行为对用户状态的影响（非独立同分布假设）
- 能学习探索策略，主动收集信息
- 优化长期指标（留存率、LTV）而非短期点击

## 2. 核心算法

### 2.1 Q-Learning / DQN
- 学习动作价值函数 Q(s, a) = E[累积奖励 | 状态s, 动作a]
- DQN 用深度神经网络逼近 Q 函数
- 选择动作：a = argmax_a Q(s, a)

DQN 在推荐中的挑战：
- 动作空间巨大（百万级物品）：无法枚举所有 Q(s, a)
- 解决方案：
  - 候选集预筛选 + DQN 重排
  - 连续动作空间：输出用户兴趣向量，最近邻检索物品
  - 分层动作空间：先选类别再选物品

### 2.2 Policy Gradient / REINFORCE
- 直接学习策略 pi(a|s)
- 参数化策略：用神经网络输出动作概率分布
- 梯度更新：theta += alpha * G_t * grad(log pi(a_t|s_t))

```
# REINFORCE 伪代码
for each episode:
    采集轨迹 (s_0, a_0, r_0, ..., s_T, a_T, r_T)
    for t = 0 to T:
        G_t = sum(gamma^k * r_{t+k})  # 累积回报
        theta += alpha * G_t * grad(log pi(a_t|s_t; theta))
```

优势：适合大规模离散动作空间，输出概率分布天然支持随机探索
劣势：高方差，需要 baseline 降方差

### 2.3 Actor-Critic
- Actor：策略网络 pi(a|s)，决定推荐什么
- Critic：价值网络 V(s) 或 Q(s,a)，评估推荐质量
- Actor 按 Critic 的评价方向更新，Critic 按 TD 误差更新

优势：比纯 Policy Gradient 方差更低，比纯 Q-Learning 更适合连续/大规模动作空间

### 2.4 对比总结

```
方法            适用场景              优点                    缺点
DQN            小动作空间            稳定、off-policy        大动作空间不适用
REINFORCE      大离散空间            直接优化策略            高方差
Actor-Critic   大/连续空间           低方差+灵活            需同时训练两个网络
```

## 3. 在线 vs 离线强化学习

### 3.1 在线 RL
- 与真实用户实时交互收集数据
- 优点：数据分布与当前策略一致
- 风险：探索阶段可能严重损害用户体验
- 实践：通常只在小比例流量上做在线探索

### 3.2 离线 RL（Batch RL / Offline RL）
- 从历史日志数据中学习策略，不需要在线交互
- 核心挑战：分布偏移（distribution shift）
  - 日志数据由旧策略生成，新策略的动作分布可能与旧策略差异很大
  - 对未在日志中出现的动作，Q 值估计不可靠
- 解决方案：
  - 保守 Q-Learning（CQL）：对未见过的动作惩罚 Q 值
  - Batch Constrained DQN：限制新策略只选择日志中出现过的动作
  - 重要性采样（IS）：用倾向得分加权校正分布偏移

### 3.3 离策略评估（OPE, Off-Policy Evaluation）
在不部署新策略的情况下估计其效果：
- IPS 估计器：w = pi_new(a|s) / pi_old(a|s)，高方差
- 双重稳健估计器（DR）：结合 IPS 和直接拟合
- MAGIC 估计器：多步 IS 与模型估计的加权组合

## 4. 多臂赌博机（MAB）

### 4.1 经典方法

epsilon-greedy：
- 以 1-epsilon 概率选当前最优臂，epsilon 概率随机探索
- 简单易实现，但探索效率低
- epsilon 通常随时间衰减

UCB（Upper Confidence Bound）：
- 选择 argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]
- 乐观面对不确定性：未被充分探索的臂有更大置信上界
- 确定性策略，无需调 epsilon

Thompson Sampling：
- 为每个臂维护奖励的后验分布（如 Beta 分布）
- 从后验采样决定选择，概率匹配方式平衡探索-利用
- 收到反馈后贝叶斯更新后验参数

```
# Thompson Sampling for Bernoulli Bandits
alpha[a], beta[a] = 1, 1  # 先验
for each round:
    theta[a] = Beta(alpha[a], beta[a]).sample()  # 采样
    a* = argmax_a theta[a]                        # 选臂
    observe reward r
    if r == 1: alpha[a*] += 1
    else:      beta[a*] += 1
```

### 4.2 上下文赌博机（Contextual Bandit）

LinUCB：
- 假设奖励与上下文特征线性相关：r = theta^T * x + noise
- 为每个臂维护参数的后验（均值+协方差矩阵）
- 选择 argmax_a [theta_a^T * x + alpha * sqrt(x^T * A_a^{-1} * x)]
- 适合冷启动场景，利用用户/物品上下文特征

上下文 Thompson Sampling：
- 贝叶斯线性回归建模奖励与上下文关系
- 从后验采样参数，计算期望奖励选择动作
- 更新：B_a += x * x^T, mu_a = B_a^{-1} * f_a

## 5. 奖励函数设计

### 5.1 即时奖励
- 点击：r = 1（简单但短视）
- 加权组合：r = w1*click + w2*purchase + w3*dwell_time
- 负反馈惩罚：快速划过、举报

### 5.2 长期奖励
- 用户留存作为延迟奖励
- 用户生命周期价值（LTV）
- 挑战：延迟反馈和信用分配问题

### 5.3 复合奖励
- 结合短期和长期指标
- 加入多样性/新颖性/公平性奖励项
- 帕累托多目标优化

## 6. 大规模状态/动作空间的解决方案

### 6.1 状态表示
- 用 RNN/GRU/Transformer 编码用户历史行为序列为固定维向量
- 注意力机制动态聚合历史行为
- 状态压缩：用 Autoencoder 降维

### 6.2 动作空间
- 分层动作：先选类别/话题，再选具体物品
- 向量化动作：策略网络输出连续向量，ANN 检索最近物品
- 候选集预筛选：先用轻量模型选千级候选集，RL 在候选集上精排
- Wolpertinger 策略：在连续动作空间产生proto-action，再映射到最近的离散动作

## 7. 系统架构

### 7.1 整体流程
```
离线训练（历史日志 -> 模型训练 -> 策略更新）
    |
    v
在线服务（请求 -> 状态编码 -> 策略推理 -> 推荐）
    |
    v
反馈闭环（用户行为 -> 日志收集 -> Kafka -> 实时特征更新）
```

### 7.2 关键工程考量
- 模型实时性：在线学习 vs 定期全量更新
- 探索安全性：限制探索比例，设置推荐质量下限
- A/B 测试：对比 RL 策略 vs 贪心策略的长期指标
- 特征实时性：Flink 流处理实时更新用户状态

## 8. 好奇心驱动探索（Curiosity-Driven Exploration）
- 思想：对模型预测不准确的状态给予内在奖励，鼓励探索未知
- 内在奖励 = 状态转移的预测误差
- 适合推荐：鼓励系统推荐预测不确定的物品以收集更多信息
- 需平衡好奇心奖励与外在奖励（用户反馈）的权重
