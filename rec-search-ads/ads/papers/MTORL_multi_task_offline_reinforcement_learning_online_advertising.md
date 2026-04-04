# Multi-task Offline Reinforcement Learning for Online Advertising (MTORL)

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

在线广告竞价与出价（Bidding）优化是序列决策问题：
- 单次 RL 无法在不同广告目标（CPM/CPC/CPA）间泛化
- 在线探索（Online RL）风险高（影响真实收入）
- 历史日志数据包含分布偏移（Offline 数据 ≠ 当前策略分布）

$$\max_\pi \mathbb{E}_{\tau \sim \pi}[\sum_t r_t] \quad \text{s.t. online safe (no explore)}$$

## 核心方法与创新点

**MTORL** 结合多任务学习与离线强化学习：

1. **统一状态表示**：
   - 广告特征 + 用户特征 + 竞价环境特征 + 任务类型（CPM/CPC/CPA）
   - Task token 嵌入使单一模型服务多种出价目标

2. **保守离线 RL（Conservative Q-Learning, CQL）适配**：
   
$$\mathcal{L}_{\text{CQL}} = \alpha \cdot \mathbb{E}_{s}[\log \sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a) \sim D}[Q(s,a)] + \mathcal{L}_{\text{Bellman}}$$

通过惩罚 OOD 动作（离线数据未出现的出价），避免外推误差。

3. **多任务共享底座 + 任务特定头**：
   - 共享 Encoder 学习通用广告环境表示
   - 任务头分别优化 CPM ROI、CPC 点击率、CPA 转化率

4. **行为约束正则化（BCO）**：
   - 防止策略偏离历史出价分布过多
   
$$\mathcal{L}_{\text{BCO}} = \mathbb{E}[D_{KL}(\pi_\theta(\cdot|s) || \pi_\beta(\cdot|s))]$$

## 实验结论

- ROI 提升：CPM +4.1%，CPC +6.3%，CPA +8.7%
- 多任务 vs 单任务：额外提升 +2-3%（知识迁移）
- 离线训练安全，上线零在线探索风险

## 工程落地要点

- 离线数据需包含完整的 Reward 信号（成交/点击/展示）
- CQL 超参 α 控制保守程度：过大→保守过度（错过高价值出价），过小→外推误差
- 多任务切换：Task token 在线推理时动态注入
- 周期离线重训（每周），避免数据分布漂移

## 面试考点

1. **Q**: 为什么广告出价优化适合 Offline RL？  
   **A**: 在线探索代价极高（影响真实收入），而历史日志数据海量（数十亿条），Offline RL 利用历史数据安全优化。

2. **Q**: Offline RL 的主要挑战（外推误差）是什么？  
   **A**: Q 网络在 OOD 动作（历史日志未见过的出价）上过于乐观，导致策略选择错误动作。CQL/BCO 通过惩罚 OOD 动作解决。

3. **Q**: 多任务 RL 如何共享知识？  
   **A**: 共享底座学习通用广告环境动态（竞价景气、用户意图），任务特定头针对各优化目标微调，底座知识跨任务迁移。
