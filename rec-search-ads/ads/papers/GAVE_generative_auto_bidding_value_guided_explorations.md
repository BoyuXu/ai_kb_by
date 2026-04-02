# GAVE: Generative Auto-Bidding with Value-Guided Explorations
> 来源：arXiv:2504.14587 | 领域：ads | 学习日期：20260330

## 问题定义
自动出价中的探索问题：为了优化长期 ROI，需要探索未知的流量、物料组合；但探索成本高（浪费预算）且广告主不接受短期效果波动。GAVE 提出生成式自动出价框架，用价值引导（Value-Guided）的探索策略，在保证短期 KPI 的同时系统性探索提升长期效益。

## 核心方法与创新点
1. **Generative Bidding Policy**：用 Diffusion Model 在出价空间生成候选出价分布 $p(b | s, g)$，$s$ 是状态（用户/流量特征），$g$ 是 KPI 目标（ROI/CPA）。
2. **Value-Guided Sampling**：在 Diffusion 采样过程中注入 value function $V(s)$ 作为 guidance：高价值状态探索更激进，低价值状态保守。
3. **两级时间尺度**：高层（分钟级）用 value function 决定探索预算分配；低层（毫秒级）用 diffusion policy 决定单次出价。
4. **Counterfactual Exploration**：用因果推断估计"如果出价不同"的反事实结果，降低真实探索成本。
5. **安全约束**：Lagrangian 约束确保探索期间预算超支率 <5%，ROI 短期下降 <3%。

## 实验结论
- 某大型广告平台：长期（7天）GMV +5.6%，短期（1天）GMV +1.2%（探索-利用权衡）
- 探索发现的高价值流量段，持续贡献 ROI 提升 +2.8%
- 预算超支率控制在 2.1%（满足安全约束）

## 工程落地要点
- Diffusion 采样步数控制（DDIM 10步），满足毫秒级出价延迟
- Value function 需要频繁更新（5分钟级），用流式数据在线学习
- 探索强度需要根据广告主风险偏好动态调整（ROI 敏感广告主降低探索）
- 因果反事实估计需要 IPW（Inverse Propensity Weighting）校正历史数据偏差

## 常见考点
- Q: 广告出价中 Exploration vs Exploitation 为什么难？
  - A: 探索成本实际（真金白银），不像 RL 仿真环境；广告主 KPI 短期要求严格；市场竞争环境非平稳，历史数据 stale
- Q: Value-Guided Diffusion 和标准 Diffusion 的区别？
  - A: 标准 Diffusion 从先验 $p(b)$ 采样；Value-Guided 在每步去噪时加入 $\nabla_b V(s, b)$ 梯度，引导采样向高价值区域聚焦
- Q: Counterfactual 出价估计的挑战？
  - A: 反事实出价 $b'$ 下的竞价结果不可直接观测，需要拍卖理论（VCG/GSP）建模，或用历史竞价价格分布模拟

## 数学公式

$$
b^* = \arg\max_b \mathbb{E}[v(b)] \quad \text{s.t.} \quad \text{short-term ROI} \geq \text{floor}
$$

$$
\text{Value-Guided Score: } \tilde{x}_{t-1} = x_{t-1} + \alpha \nabla_{x_t} V(x_t)
$$
