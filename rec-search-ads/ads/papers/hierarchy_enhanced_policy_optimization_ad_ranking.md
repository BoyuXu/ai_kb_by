# Hierarchy Enhanced Policy Optimization for Ad Ranking

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

广告排序（Ad Ranking）的核心挑战：
1. **长期回报 vs 即时收益**：CTR 最高的广告不一定带来最好的用户长期体验和广告主 ROI
2. **多目标权衡**：需要同时优化平台收入（RPM）、用户体验（留存率、满意度）、广告主效果（转化率）
3. **探索-利用困境**：强化学习在广告排序中探索成本高（展示不优广告会损失收入）

Hierarchy Enhanced Policy Optimization（HEPO）提出层次化策略优化框架，在不同时间粒度上同时优化即时和长期目标。

## 核心方法与创新点

### 层次化策略框架

**High-Level Policy（高层策略）**：长期规划
- 输入：用户 session 级别的状态（历史行为模式、疲劳度）
- 输出：本次 session 的广告展示策略（如：本 session 预算、类别偏好）
- 更新频率：每个 session 或每天

**Low-Level Policy（低层策略）**：即时排序
- 输入：单次请求的特征（query、上下文、候选广告）
- 输出：当前页面的广告排序
- 更新频率：每次请求实时

$$\pi_{low}(a | s, g) = \pi_{low}^{RL}(a | s) \cdot \text{guided by } g$$

其中 $g$ 是高层策略输出的"指导信号"（goals）。

### 奖励设计

**即时奖励**（Low-level）：
$$r_t = \alpha \cdot CTR_t + \beta \cdot Revenue_t - \gamma \cdot \text{UX\_penalty}_t$$

**长期奖励**（High-level）：
$$R_{session} = \sum_{t} \gamma^t r_t + \lambda \cdot \text{RetentionBonus}$$

### 策略优化

两层策略交替优化：
- Low-level 用 Actor-Critic（PPO）优化即时排序
- High-level 用 Q-learning 优化 session 级目标

## 实验结论

- 在某头部广告平台仿真环境和线上 A/B 测试中验证
- 相比纯 CTR 排序，长期用户留存率提升约 1.2%
- RPM（每千次展示收入）提升约 0.8%，同时用户满意度指标也有提升
- 层次化结构优于单一 RL 策略，训练更稳定

## 工程落地要点

1. **仿真环境构建**：RL 广告排序必须先在离线仿真中验证，直接线上探索风险极高
2. **奖励延迟处理**：长期奖励（留存率）需要数天才能观测，需要设计奖励代理模型
3. **安全约束**：低层策略必须满足约束（收入不低于某阈值），可用 Constrained RL（CPO）
4. **分层部署**：高层策略可以离线计算（每天/每 session 刷新），低层策略实时推理
5. **冷启动**：RL 探索初期用传统 CTR 模型作为 baseline 策略，逐步提升 RL 比例
6. **监控告警**：设置收入下降阈值，一旦 RL 策略导致收入异常立即回滚

## 面试考点

**Q：广告排序中用强化学习和传统监督学习（CTR 预估）的最大区别是什么？**
A：监督学习优化即时 CTR，把每次展示独立看待，无法考虑序列决策和长期影响；强化学习将广告展示视为序列决策过程，可以建模用户疲劳（过多广告降低未来点击率）、长期留存（好的广告体验提升次日留存）等长期效应。代价是训练复杂、探索成本高。

**Q：层次化 RL（Hierarchical RL）在广告场景的优势是什么？**
A：广告决策天然存在多时间尺度：session 级的预算分配（每天决策一次）和请求级的实时排序（毫秒级）。HRL 允许高层策略关注长期目标（留存、ROI），低层策略快速响应即时请求，两层分工明确，比单层 RL 更容易优化和调试。

**Q：如何处理强化学习中广告探索（exploration）的成本问题？**
A：1) 仿真环境预训练：先在历史数据构建的 user simulator 中充分探索；2) 约束探索：用 Constrained RL 限制收入下降幅度；3) 渐进式上线：从小流量（1%）开始，逐步扩大 RL 策略覆盖率；4) Safe RL：如果 episode reward 低于阈值，自动回退到保守策略。
