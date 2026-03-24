# 广告出价体系：从手动规则到 RL 自动出价

> 📚 参考文献
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/Multi_Objective_Optimization_for_Online_Advertising_Balan.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Multi Objective Online Advertising](../../ads/papers/Multi_Objective_Optimization_for_Online_Advertising_Balan.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Budget-Pacing-Strategies-For-Large-Scale-Ad-Cam...](../../ads/papers/Budget_Pacing_Strategies_for_Large_Scale_Ad_Campaigns.md) — Budget Pacing Strategies for Large-Scale Ad Campaigns
> - [Practical-Guide-Budget-Pacing](../../ads/papers/A_Practical_Guide_to_Budget_Pacing_Algorithms_in_Digital.md) — A Practical Guide to Budget Pacing Algorithms in Digital ...
> - [Multi-Objective-Ads-Ranking](../../ads/papers/MMoE_PLE_Pareto.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Budget Pacing](../../ads/papers/budget_pacing.md) — 预算控制与Pacing机制详解


**一句话**：广告出价就是"在拍卖场里替广告主抢位置"——怎么抢、抢多少、分几天抢完，这一套系统就是出价体系。

**类比**：你去拍卖会买古董，预算 10 万，目标是买到 3 件。你可以：(A) 每件都出 3.3 万（规则）；(B) 请一个懂行的经纪人（AutoBid AI），他知道什么时候东西被低估、什么时候竞争激烈要忍一忍，帮你把 10 万花得更值。

**广告出价的三大问题**（今日两篇各解决一个方向）：

### 问题一：单次出价优化（AutoBid/GRPO 类）
- **目标**：最大化转化量，满足 CPA ≤ 目标 + 日预算不超支
- **传统方法**：PID 控制（比例-积分-微分），根据 CPA 偏差反馈调整出价系数
- **RL 方法（AutoBid）**：把全天竞拍建模为 MDP，状态=剩余预算+当前时段+竞争强度，动作=出价系数 k，RL 学习跨时段最优策略
- **核心技术**：Lagrangian Relaxation 处理约束（CPA/Budget → 软约束 → 对偶乘子自适应调整）

### 问题二：多目标平衡（今日 Multi-Objective 论文）
- **目标**：eCPM（收入）vs 相关性（用户体验）vs 广告主 ROI
- **工程解法**：加权求和（Score = w₁×eCPM + w₂×质量分）+ 动态权重（实时感知广告填充率调档）
- **关键洞察**：纯收入最大化是短视的，用户体验差 → 留存下降 → 未来收入损失；LTV 模型估算长期价值纳入优化

**技术演进脉络**：
```
手动出价（2010前）→ 规则出价（公式）→ oCPM/oCPA（预估pCTR/pCVR优化）
→ PID 控制 Pacing（工业主流 2015-2020）→ AutoBid RL（2021+，头部平台）
→ 多目标 RL + LTV 建模（当前前沿）
```

**和推荐多目标的横向对比**：
| 维度 | 广告多目标出价 | 推荐多目标排序 |
|------|-------------|-------------|
| 目标冲突 | 收入 vs 用户体验 | CTR vs 时长 vs 多样性 |
| 约束方式 | 预算/CPA 为硬约束 | 广告密度/内容比例 |
| 优化视角 | 跨时段（全天预算分配）| 当次（单次请求排序）|
| 长期建模 | LTV 必须建模 | 留存/下播率 |
| 数学框架 | 拉格朗日对偶 | Pareto 优化/线性加权 |

**工业常见做法**：
- **出价核心公式**：实际出价 = k × pCVR × 期望价值（k 由 AutoBid/RL 决定）
- **预算 Pacing**：不能让广告主早上 10 点预算花完，PID 控制消耗曲线平滑
- **RL 部署**：离线模拟器（历史竞拍 replay）训练，在线设出价上界防异常
- **乘子更新频率**：Lagrange 乘子每天更新（episode=1天），策略每小时微调

**面试考点**：
- Q: eCPM 是什么？为什么是广告竞价的核心中间量？ → 统一 CPM/CPC/CPA 不同计费模式，使不同广告主公平竞价
- Q: Pacing 为什么重要？ → 时段分布不均会导致预算集中消耗 + 无法触达低峰期受众
- Q: 带约束 RL (CMDP) 最常用哪种解法？ → Lagrangian Relaxation，将约束转软约束，对偶迭代优化
- Q: 多目标广告排序最大的工程难点？ → 动态权重需要实时感知系统状态（填充率 API），且 LTV 模型误差会误导决策


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
