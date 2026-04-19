# Improved Online Learning Algorithms for CTR Prediction in Ad Auctions

> 来源：https://arxiv.org/abs/2403.00845 | 领域：ads | 学习日期：20260420

## 问题定义

广告拍卖中，平台作为卖方需要在线学习各广告候选的 CTR，同时通过 pay-per-click 方式向赢家收费。核心挑战：

1. **CTR 未知**：平台需要同时学习 CTR 和做分配决策（exploration-exploitation）
2. **广告主行为建模**：广告主可能是近视的（只优化当前轮）或非近视的（优化长期效用）
3. **拍卖机制与学习的耦合**：出价排序取决于 CTR 估计质量，但 CTR 估计依赖曝光分配

## 核心方法与创新点

### 1. Myopic Advertisers（近视广告主）

提出基于 UCB 的在线拍卖机制：

$$
\text{Score}_a(t) = b_a \cdot \text{UCB}_a(t) = b_a \cdot \left(\hat{\mu}_a(t) + \sqrt{\frac{2\log t}{n_a(t)}}\right)
$$

**理论保证**：
- 最坏情况下：$O(\sqrt{T})$ Regret（tight，不可改进）
- 静态最优有间隔时：**负 Regret**（即超越最优固定策略）—— 这是因为 UCB 的乐观估计恰好导致更优的探索分配

$$
\text{Regret}(T) = O\left(\sum_{a: \Delta_a > 0} \frac{\log T}{\Delta_a}\right)
$$

其中 $\Delta_a = b_1 \mu_1 - b_a \mu_a$ 是次优臂与最优臂的价值差距。

### 2. Non-Myopic Advertisers（非近视广告主）

当广告主考虑长期效用时，会策略性地调整出价来影响平台的学习过程：
- 广告主可能故意提高出价以获得更多曝光（加速学习自己的 CTR）
- 或降低出价让平台认为自己价值低（长期降低竞争压力）

论文分析了这种博弈均衡下的 Regret 性质。

## 核心 Insight

1. **CTR 学习和拍卖设计不可分离** —— 传统方法分别优化 CTR 模型和拍卖机制，但实际上 CTR 估计的准确性直接影响拍卖收入，而拍卖的分配决策又影响 CTR 数据收集
2. **UCB 在拍卖中有"超额收益"** —— 乐观估计导致更多探索，如果最优广告的 CTR 有优势间隔，UCB 的探索反而比知道真实 CTR 还赚钱（负 Regret）
3. **非近视广告主是真实世界的常态** —— 工业界的大广告主会策略性出价，简单的 UCB 机制可能被操纵

## 面试考点

- Q: 为什么 UCB 在拍卖中能实现负 Regret？
  > 当最优臂有 gap 时，UCB 的乐观估计让次优臂获得一些"本不该有"的曝光，但这些探索的成本（次优分配）小于从中获得的信息价值（更快确认最优臂），净效果为正。
- Q: 非近视广告主如何影响平台策略？
  > 广告主会策略性出价来操纵平台的学习过程。平台需要设计"防操纵"的学习机制，如随机化探索（非确定性 UCB）或 Bayesian 拍卖。

---

## 相关链接

- [[mab_cold_start_auction_dynamics]] — MAB 冷启动
- [[genauction_generative_auction_online_advertising]] — 生成式拍卖
