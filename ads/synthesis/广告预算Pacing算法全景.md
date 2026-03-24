# 知识卡片 #010：广告预算 Pacing 算法全景

> 📚 参考文献
> - [Joint-Value-Estimation-Bidding-Repeated-Fpa](../../ads/papers/Joint_Value_Estimation_and_Bidding_in_Repeated_First_Pric.md) — Joint Value Estimation and Bidding in Repeated First-Pric...
> - [Adaptive-Bidding-Budget-Nonstationary](../../ads/papers/Adaptive_Bidding_Policies_for_First_Price_Auctions_with_B.md) — Adaptive Bidding Policies for First-Price Auctions with B...
> - [Practical-Guide-Budget-Pacing](../../ads/papers/A_Practical_Guide_to_Budget_Pacing_Algorithms_in_Digital.md) — A Practical Guide to Budget Pacing Algorithms in Digital ...
> - [Est-Ctr-Scaling](../../ads/papers/EST_Efficient_Scaling_Laws_in_CTR_Prediction_via_Unified.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Budget-Pacing-Strategies-For-Large-Scale-Ad-Cam...](../../ads/papers/Budget_Pacing_Strategies_for_Large_Scale_Ad_Campaigns.md) — Budget Pacing Strategies for Large-Scale Ad Campaigns
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../ads/papers/Real_Time_Bidding_Optimization_with_Multi_Agent_Deep_Rein.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Learning-Bid-Nonstationary-Fpa](../../ads/papers/Learning_to_Bid_in_Non_Stationary_Repeated_First_Price_Au.md) — Learning to Bid in Non-Stationary Repeated First-Price Au...
> - [Budget Pacing](../../ads/papers/budget_pacing.md) — 预算控制与Pacing机制详解


> 创建：2026-03-20 | 领域：在线广告·出价 | 难度：⭐⭐⭐⭐
> 来源：Budget Pacing Guide (2503.06942)、Adaptive Bidding FPA (2505.02796)、Joint Value Estimation FPA


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

## 🌟 一句话解释

广告主给了 $1000/天的预算，Pacing 的任务是**控制花钱的节奏**：既不能 9 点就花完（下午没曝光），也不能晚上 11 点才开始突击（错过黄金流量），同时要在第一价格拍卖中智能出价而非简单报真实估值。

---

## 🎭 生活类比

你有 100 块零花钱过一整天：
- **没有 Pacing（贪心）**：早上看到好吃的全花了，下午饿肚子
- **简单 Throttling**：随机跳过一半机会（抛硬币），省钱但可能错过最好的食物
- **PID 控制**：每小时检查剩余预算，花多了就少参与，花少了就多参与——像自动调节油门的司机
- **对偶梯度下降**：维护一个"花钱欲望系数"，预算消耗快时自动降低出价，消耗慢时提高——像理性的收藏家，货多时出价低，货紧时出价高
- **非平稳自适应（Wasserstein）**：连流量分布都会变（早高峰→午休→晚高峰），用 Wasserstein 距离追踪变化，自适应调整

---

## ⚙️ 技术演进脉络

```
【时代一：简单 Throttling（随机跳过）】
  按参与率 p 随机决定是否参与拍卖
  ✅ 简单易实现 ❌ 不区分流量质量，可能错过黄金流量

【时代二：PID 控制 Pacing（工业主流）】
  P（当前偏差）+ I（累积偏差）+ D（变化趋势）→ 出价乘数
  ✅ 工程成熟，直觉强 ❌ 参数调优依赖经验，非平稳环境需频繁调参

【时代三：对偶梯度下降 Pacing（理论最优）】
  维护对偶变量 λ（预算影子价格），在对偶空间梯度下降
  ✅ 理论保证，自适应 ✅ 天然处理约束
  ❌ 实现复杂，需要在线学习对手出价分布

【时代四：第一价格拍卖自适应出价（非平稳）】
  FPA 中不再有 "truthful bidding is optimal"
  需要学习 shading 策略 + Wasserstein 距离追踪分布偏移
  ✅ 非平稳性遗憾界 O(√T + W_T) ✅ Wasserstein 度量紧致
```

---

## 🔬 三大算法核心机制

### PID 控制 Pacing

```
目标预算消耗曲线: spend_target(t) = budget × (t/T)
实际消耗: spend_actual(t)
误差 e(t) = spend_target(t) - spend_actual(t)

bid_multiplier(t) = 1 + Kp×e(t) + Ki×∫e + Kd×de/dt

出价 = pCTR × pCVR × targetCPA × bid_multiplier
```

### 对偶梯度下降

```
原问题: max Σ value(bid_t) s.t. Σ cost(bid_t) ≤ Budget

引入对偶变量 λ（预算的边际价值）:
L(bid, λ) = Σ value(bid_t) - λ(Σ cost(bid_t) - Budget)

最优出价: bid*_t = argmax [value - λ × cost]
         对 FPA: bid*_t = v_t - 1/(f(b_t)) 其中 f 是对手出价密度

λ 更新: λ_{t+1} = λ_t + η(cost(bid_t) - budget/T)
```

### 第一价格拍卖 Shading

```
第二价格（Vickrey）: bid = v（truthful，因为支付第二高价）
第一价格: 若出价 b，赢则支付 b，因此需要 shading

最优 shading: bid* = v - (1 - F(v))/f(v)
  F(v): 对手出价分布的 CDF
  f(v): 对手出价分布的 PDF

非平稳下 Wasserstein 追踪: W_1(F_t, F_{t-1}) 度量分布偏移
遗憾界: O(√T + Σ W_1(F_t, F_{t-1}))
```

---

## 🏭 工业落地关键点

| 挑战 | 工程解法 |
|------|---------|
| 多约束（预算 + ROAS + 频控） | Min-pacing：各约束分别算乘数，取最小值 |
| 流量非平稳（早晚高峰差异大） | 滑动窗口 + 周期性重置对偶变量 |
| A/B 测试设计 | 广告主级别分桶（防预算竞争），覆盖完整日/周周期 |
| 延迟要求（<10ms） | 对偶变量预计算，出价时直接查表 |
| 冷启动（新广告主） | 使用同类广告主历史数据初始化参数 |

---

## 🆚 和已有知识的对比

**Pacing vs 竞价策略（Bidding）**：
- Pacing：控制花钱节奏（时间维度调节出价乘数）
- Bidding/Shading：决定单次出价多少（价值估计 × 策略 shading）
- 工业中：bid = value × shading_factor × pacing_multiplier，三者叠加

**FPA Shading vs 传统 SPAtruthful bidding**：
- SPA（第二价格）：按真实估值出价是占优策略，不需要估计对手
- FPA（第一价格）：需要估计对手出价分布 F(b)，计算最优 shading
- 谷歌、腾讯、百度广告系统均已切换到 FPA

---

## 🎯 面试考点

**Q1：为什么第一价格拍卖中不能按真实估值出价？**
A：SPA 中赢者支付第二高价，truthful bidding 是占优策略（无论对手出价如何，按真值出价都不差）。FPA 中赢者支付自己出价，若按真值 v 出价赢了收益为 v-v=0，因此必须 shading：出价低于 v，通过牺牲部分赢面换取正收益。最优 shading 量取决于对手出价分布的形状。

**Q2：预算 Pacing 和广告出价是什么关系？**
A：分层控制。出价（bidding）决定"这个流量值多少"，Pacing 决定"现在该用多少力气去争"。公式：bid_final = bid_base × pacing_multiplier。两者解耦：出价模型优化单次机会价值，Pacing 控制整体预算消耗节奏。

**Q3：Wasserstein 距离为什么比 TV 距离更适合度量拍卖环境的非平稳性？**
A：TV 距离对分布的任何不重叠都给出最大值 1，对小的位置偏移过于激进。Wasserstein 距离（Earth Mover's Distance）考虑分布的几何结构，小幅度的分布平移只会产生小的 W 值。在广告出价中，流量质量的渐进变化（早高峰→午休）更适合用 Wasserstein 描述，遗憾界更紧。

**Q4：多约束情况下（预算 + ROAS 约束）如何设计 Pacing？**
A：有三种方案：① Min-pacing：分别算预算约束乘数 m1 和 ROAS 约束乘数 m2，bid × min(m1, m2)，工程简单但次优；② 耦合对偶：两个约束各引入一个对偶变量，联合梯度下降，理论最优但实现复杂；③ 序贯：先满足 ROAS 约束再处理预算，直觉强但不保证最优。实践中 min-pacing 接近最优（~5% gap），是工业主流。

---

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
