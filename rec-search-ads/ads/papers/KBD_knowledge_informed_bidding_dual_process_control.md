# KBD: Knowledge-informed Bidding with Dual-process Control for Online Advertising
> 来源：arXiv:2603.04920 | 领域：ads | 学习日期：20260330

## 问题定义
在线广告自动出价（Auto-bidding）需要在实时竞价（RTB）中为每次曝光机会决定出价。传统方法分为规则出价（稳定但次优）和 RL 出价（最优但不稳定），且缺乏对市场动态的先验知识利用。KBD 提出双过程控制框架，融合领域知识（Knowledge）和实时自适应（Adaptive RL），实现稳定高效出价。

## 核心方法与创新点
1. **Dual-Process 架构**：类比人类认知双系统——System 1（快速/直觉）用规则/先验快速响应；System 2（慢速/理性）用 RL 深度优化。
2. **知识注入**：将市场先验（历史出价分布、竞争格局、时段模式）编码为 knowledge embedding，注入 RL policy 网络。
3. **预算约束建模**：用 Lagrangian Relaxation 将预算约束化为 soft penalty，将约束出价问题转化为无约束 RL。
4. **双层控制**：高层 controller 设定出价乘数（bid multiplier）边界；低层 policy 在边界内精细调整，保证稳定性。
5. **知识更新**：定期用真实竞价数据更新 knowledge embedding，适应市场变化。

## 实验结论
- 某电商平台广告：同等预算下 GMV +3.8%，ROI +2.1%（对比单纯 RL 出价）
- 预算超支率从 RL 的 12% 降至 KBD 的 3%（双过程控制效果）
- 冷启动广告主（新账户）收益提升最显著 +8.2%（知识迁移优势）

## 工程落地要点
- Knowledge embedding 更新频率建议每小时一次（日内市场波动明显）
- Lagrangian 乘子 $\lambda$ 需要在线自适应调整（用预算消耗率作为信号）
- 双层控制中，high-level controller 需要比 policy 更低频更新（分钟级 vs 毫秒级）
- RL 训练需要模拟器（Auction Simulator）避免线上探索成本

## 面试考点
- Q: 自动出价（Auto-bidding）的核心优化目标是什么？
  - A: 在预算约束 $\sum b_i \leq B$ 下最大化目标（GMV/转化数/曝光）。形式化为约束 RL 问题
- Q: Lagrangian Relaxation 如何处理预算约束？
  - A: $\mathcal{L}(b, \lambda) = \mathbb{E}[v] - \lambda(\sum b_i - B)$，对偶变量 $\lambda$ 作为"预算价格"，梯度上升更新 $\lambda$
- Q: 为什么纯 RL 出价不稳定？
  - A: RL 探索需要 variance（不稳定）；reward 稀疏（转化延迟）；非平稳市场使 policy 频繁失效

## 数学公式

$$
\max_\pi \mathbb{E}_\pi[\sum_t v_t] \quad \text{s.t.} \quad \mathbb{E}[\sum_t b_t] \leq B
$$

$$
\text{Dual: } \min_{\lambda \geq 0} \max_\pi \mathbb{E}[\sum_t (v_t - \lambda b_t)] + \lambda B
$$
