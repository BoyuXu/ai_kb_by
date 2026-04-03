# A/B 测试 & 实验设计完全指南

> 更新时间：2026-03-13 | 面向算法工程师面试（推荐/搜索/广告方向）

---

## 核心概念

### 1. A/B 测试的统计基础

A/B 测试本质是**假设检验**，判断实验组与对照组的指标差异是否为真实效果，还是随机波动。

**核心参数**：
- **显著性水平 α（Type I Error）**：错误拒绝原假设的概率（通常设 0.05）
- **统计功效 1-β（Power，Type II Error β）**：正确发现真实效果的概率（通常设 0.8 或 0.9）
- **最小可检测效果 MDE**：实验需要能检测到的最小提升幅度
- **样本量 N**：与 α、β、MDE、指标方差共同决定

**样本量公式**（双尾检验）：
```
N = 2 × (z_{α/2} + z_β)² × σ² / δ²

其中：
- z_{α/2} = 1.96（α=0.05 时）
- z_β = 0.84（β=0.2，Power=80% 时）
- σ²：指标方差
- δ：MDE（期望检测到的最小差值）
```

**常见误区**：
- ❌ 实验跑几天发现显著就停止（Peeking Problem / Optional Stopping）
- ❌ 多重检验不纠正（同时测试10个指标，随机显著率 ~40%）
- ❌ 样本量不足就下结论

### 2. 推荐系统中的指标体系

**北极星指标**（关键指标，不能同时优化太多）：
- 短期：CTR（点击率）、CVR（转化率）、GMV
- 长期：DAU、留存率、用户生命周期价值（LTV）

**护栏指标（Guardrail Metrics）**：
- 实验必须满足的底线，如页面加载时长不超过 N ms、崩溃率不上升
- 即使目标指标提升，如果护栏指标恶化，实验应被拒绝

**反直觉情况**：
- CTR 提升但 CVR 下降 → 可能"标题党"吸引错误用户
- 人均点击上升但用户留存下降 → 可能过度推送导致用户疲劳

### 3. 实验偏差与常见问题

**Sample Ratio Mismatch（SRM）**：
- 分配比例与实际流量比例不符（如预期 50:50，实际 48:52）
- 原因：流量分割 bug、Bot 流量、缓存差异
- **必须先检查 SRM，否则实验结论无效**
- 检验方法：卡方检验（Chi-square test）

**网络效应（Network Effect）**：
- 实验组和对照组用户之间存在交互，违反 SUTVA 假设
- 典型场景：社交推荐（A 关注 B，A 在实验组，B 在对照组）
- 解决方案：以图社区（community）为单位分桶，而非用户

**新奇效应（Novelty Effect）**：
- 用户因为新功能带来的新鲜感而短期提升参与度，长期会恢复
- 识别方法：观察指标随时间的衰减趋势，延长实验时长

**混淆因素**：
- 节假日、大促、外部事件导致实验期间数据异常
- 解决：Holdout Set（长期对照组）、时间校正

### 4. 方差缩减技术（Variance Reduction）

目的：在不增加样本量的情况下，通过减少指标方差来提升检验功效。

**CUPED（Controlled-experiment Using Pre-Experiment Data）**：
- 利用实验前的协变量（如用户历史 CTR）对指标进行调整
- 调整后指标：Y̅ = Y - θ × (X - X̅)，θ 通过最小化方差求得
- 实际效果：方差减少 30%-70%，相当于增加几倍样本量
- 广泛应用：Microsoft ExP、Airbnb、Netflix

**Stratified Sampling（分层采样）**：
- 按用户特征（如活跃度、地域）分层，在每层内独立随机分配
- 减少组间不平衡导致的方差

### 5. 多变量测试（MVT）与 AA 测试

**AA 测试**：实验前验证分桶系统是否正确，两组都应用相同策略，期望 p > 0.05（无显著差异）。

**多变量测试（MVT）**：
- 同时测试多个变量（如推荐 UI、排序策略、候选数量）
- 需要用 Bonferroni 校正或 FDR 控制多重检验问题
- Bonferroni：每个检验的显著性水平 = α / 检验数量

---

## 工程实践

### 实验平台架构

```
用户请求
    ↓
[流量分配层] → 哈希(user_id + experiment_id) % 10000 → 桶号
    ↓
[实验配置服务] → 根据桶号返回实验参数（缓存在本地，低延迟）
    ↓
[业务逻辑层] → 按实验参数执行不同策略
    ↓
[埋点系统] → 记录 (user_id, experiment_id, bucket, event, timestamp)
    ↓
[数据仓库] → 实验日志 T+1 聚合
    ↓
[分析平台] → 统计检验、可视化、报告
```

**关键设计原则**：
1. **哈希一致性**：相同用户在同一实验中始终落入相同桶（重复访问体验一致）
2. **实验隔离**：多个实验同时运行时，不同实验之间相互独立（正交分层）
3. **低延迟**：实验配置本地缓存，不能增加推理路径延迟
4. **可回滚**：实验参数动态下发，快速关闭异常实验

### 正交分层实验（Overlapping Experiments）

工业界通常同时运行数百个实验，需要正交分层：

```
总流量 100%
├── Layer A（召回策略层）
│   ├── Exp A1（新召回模型）：50%
│   └── Exp A2（对照）：50%
└── Layer B（排序策略层）—— 与 Layer A 正交
    ├── Exp B1（新排序特征）：50%
    └── Exp B2（对照）：50%
```

每个 Layer 独立哈希，用户在 A 和 B 中的分配相互独立，4 个组合同时被测试。

### 推荐系统常见实验场景

**召回实验**：
- 目标指标：召回率@K、覆盖度、新颖度
- 特殊考虑：多路召回中替换一路，注意候选总量的变化会影响排序

**排序实验**：
- 目标指标：NDCG、CTR、CVR、时长（视频）
- 实验单元：通常以用户为单位（user-level experiment）

**重排/干预实验**：
- 多样性干预（如强制插入长尾内容），可能短期 CTR 下降，需关注留存
- 时长 vs CTR 的 tradeoff：视频推荐中用户时长比 CTR 更有价值

### 数据分析流程

```python
# 实验分析示例（Python + scipy）
from scipy import stats
import numpy as np

# 实验数据
control_ctr = np.array([0.082, 0.079, ...])  # 对照组日 CTR
treatment_ctr = np.array([0.089, 0.091, ...])  # 实验组日 CTR

# 1. 检查 SRM
observed = [len(control_ctr), len(treatment_ctr)]
expected = [sum(observed)/2, sum(observed)/2]
chi2, p_srm = stats.chisquare(observed, expected)
print(f"SRM check: chi2={chi2:.4f}, p={p_srm:.4f}")  # 期望 p > 0.05

# 2. 双尾 t 检验
t_stat, p_value = stats.ttest_ind(treatment_ctr, control_ctr)
print(f"t-test: t={t_stat:.4f}, p={p_value:.4f}")

# 3. 计算 lift
lift = (treatment_ctr.mean() - control_ctr.mean()) / control_ctr.mean()
print(f"CTR Lift: {lift:.2%}")

# 4. 置信区间
ci = stats.t.interval(0.95, df=len(control_ctr)-1,
                       loc=treatment_ctr.mean() - control_ctr.mean(),
                       scale=stats.sem(treatment_ctr - control_ctr))
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### 工具与平台

| 平台 | 特点 |
|------|------|
| **Microsoft ExP** | 业界标杆，已运行16年+，每月数千个实验 |
| **Airbnb Experimentation** | ERF（Experiment Reporting Framework） |
| **Netflix** | 统计严谨，Spark 大规模计算 |
| **字节跳动火山引擎** | A/B Test 平台，国内互联网通用 |
| **腾讯实验平台** | 支持微信/QQ 产品线 |
| **开源方案** | GrowthBook、Flagsmith、Optimizely |

---

## 面试高频考点

### Q1：A/B 测试的完整流程是什么？

**A**：
1. **实验设计**：明确假设（H0/H1）、确定主要指标和护栏指标、计算所需样本量（基于 MDE、α、Power）
2. **AA 测试**：验证分桶系统正确性，确保两组在实验前无统计差异
3. **实验运行**：不要 Peek，遵守预先设定的实验时长（通常至少覆盖一个完整周期）
4. **SRM 检查**：验证实验组/对照组的流量比例符合预期
5. **统计分析**：t 检验（连续指标）或 Z 检验（比例指标），计算 p 值和置信区间
6. **业务解读**：结合业务背景解释统计显著性，评估是否部署
7. **全量或关闭**：根据结论决策，记录实验结果

### Q2：如何确定 A/B 测试的样本量？

**A**：
样本量由四个因素共同决定：
- **期望 MDE δ**：想检测到的最小提升（如 CTR 相对提升 2%），越小需要越多样本
- **指标方差 σ²**：可用历史数据估算，方差越大需要越多样本
- **显著性水平 α**：通常 0.05，越严格需要越多样本
- **功效 1-β**：通常 0.8，要求越高需要越多样本

计算公式：`N ≈ 2 × (z_{α/2} + z_β)² × σ² / δ²`

**实际技巧**：
- 使用历史数据估算均值和方差
- 在线样本量计算器（如 Evan's A/B Tools）
- MDE 不要定太小，否则实验时间极长

### Q3：什么是 Peeking Problem？如何解决？

**A**：
**Peeking Problem**：实验期间频繁查看结果，一旦 p < 0.05 就停止实验。这违背了假设检验的前提——样本量需在实验前固定。

**问题**：如果每天都"偷看"并在显著时停止，虚假阳性率远超 5%（可能达到 30%+）。

**解决方案**：
1. **提前固定实验时长**：在设计阶段就确定，坚持到底
2. **Sequential Testing（序贯检验）**：使用 Always Valid Inference（如 mSPRT），允许在任意时间点检验，但控制 Type I Error
3. **Bayesian A/B Testing**：用贝叶斯方法，设定停止规则（如后验概率 > 95%），理论上可以随时停止

### Q4：推荐系统中如何处理 A/B 测试的网络效应问题？

**A**：
推荐系统中的网络效应指：用户之间的交互使得实验组和对照组不独立，违反 SUTVA（稳定单元处理假设）。

**场景**：社交推荐中，A 用户在实验组（看到新推荐），A 的内容被 B（对照组）消费，导致 B 的行为也被改变。

**解决方案**：
1. **图划分（Graph Partitioning）**：以强连接社区为单位分配实验，同一社区内用户在相同组
2. **Ego Network Clustering**：以用户的局部社交网络为单位分桶
3. **延迟测量（Holdback）**：保留一个长期对照组（如 5% holdback），避免网络效应污染

对于大多数内容推荐（无社交关系），网络效应通常较小，可以忽略。

### Q5：什么情况下 A/B 测试结论是可信的？有哪些必须检查的前提条件？

**A**：
必须检查的前提：
1. **SRM（Sample Ratio Mismatch）**：实际流量比例必须符合设计，否则实验有 bug
2. **AA 测试通过**：证明分桶系统在实验前无偏
3. **实验时长足够**：覆盖业务周期（如至少一周，避免工作日/周末偏差）
4. **独立性**：实验单元之间相互独立（无网络效应）
5. **稳定性**：实验期间无外部干扰（大促、灰度上线等）
6. **多重检验控制**：测试多个指标时，用 Bonferroni 或 FDR 方法控制

### Q6：解释置信区间的含义，以及如何向产品经理解释 A/B 测试结果？

**A**：
**置信区间**：不是"95% 概率真实值在此区间内"（贝叶斯解释），而是"如果重复实验100次，95次的置信区间会覆盖真实值"（频率派解释）。

**向 PM 解释的简洁方式**：
"我们有 95% 的把握，新推荐策略带来的 CTR 提升在 1.5% 到 3.2% 之间。这个提升在统计上是可靠的，不太可能是随机波动。预计若全量上线，每天可以多带来约 X 次点击。"

关键：区分**统计显著性**（p < 0.05）和**业务显著性**（提升是否值得上线的成本）。

### Q7：CUPED 方差缩减技术的原理？

**A**：
CUPED（Controlled-experiment Using Pre-Experiment Data）通过引入协变量来减少实验指标的方差，从而以更少的样本达到相同的检验功效。

**核心公式**：
```
Y_cuped = Y - θ × (X - E[X])
其中 X 是实验前的协变量（如历史 CTR），θ = Cov(Y,X) / Var(X)
```

**为什么有效**：
- 用户之间的基础差异（有些用户本来 CTR 高）是指标方差的主要来源
- 用实验前的历史行为"控制"掉这部分差异后，剩余方差大幅减少
- 通常能减少 30%-70% 的方差，等效于增加 1.5-3x 的样本量

**实践注意**：协变量 X 必须是实验开始前的数据，否则会引入偏差。

---

## 参考资料

1. **Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing** (Ron Kohavi, Diane Tang, Ya Xu)
   - https://www.amazon.com/Trustworthy-Online-Controlled-Experiments-Practical/dp/1108724264

2. **Microsoft ExP - Variance Reduction (CUPED)**
   - https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/

---

## Cookie Day 分析算法

> 补充于 2026-03-13 | 解决新老用户混合导致的实验偏差问题

### 1. 什么是 Cookie Day？

**Cookie Day（曝光日/入组日）** 指用户第一次被分入实验桶的日期，即 **用户的实验第 0 天**。

核心问题：A/B 实验通常持续 7-14 天，但不同用户的入组时间不同：
- 实验开始就入组的用户，实验期间已有 7+ 天的曝光
- 实验后期才入组的用户，只有 1-2 天曝光

如果不考虑这个差异，直接汇总所有用户数据，**新老用户的行为差异会混入实验结果**，导致效果被低估或高估。

---

### 2. Cookie Day 效应的两种偏差

#### 2.1 新奇效应（Novelty Effect）
- 用户刚接触新功能时，好奇心驱动行为（CTR 虚高）
- 随着时间推移，兴趣衰减，指标回落
- 如果实验期短，捕捉到的是"新鲜感"而非真实长期效果

#### 2.2 学习效应（Learning / Primacy Effect）
- 与新奇效应相反：用户需要时间适应新功能
- 初期指标可能低于对照组，但长期来看更好
- 典型场景：新 UI、新推荐策略

```
           新奇效应                学习效应
CTR ↑                       CTR
    \                              /
     \                            /
      \_____ 稳定值              /_____ 稳定值
      
 0  1  2  3  4  5  6  7d     0  1  2  3  4  5  6  7d
         (cookie day)                 (cookie day)
```

---

### 3. Cookie Day 分析方法

#### 3.1 按入组天数分层分析

将所有用户按 **在实验中的第几天（day_in_exp）** 分组，分别计算指标：

```python
# 伪代码
df['day_in_exp'] = df['event_date'] - df['cookie_day']  # 距离入组天数

# 按 day_in_exp 分组，计算每日指标差异
result = df.groupby(['exp_group', 'day_in_exp']).agg(
    ctr=('click', 'mean'),
    user_cnt=('user_id', 'nunique')
).reset_index()

# 画出曲线，观察是否收敛
plot(result, x='day_in_exp', y='ctr', hue='exp_group')
```

**解读：**
- 若实验组曲线从高到低收敛 → 新奇效应，需等待稳定
- 若实验组曲线从低到高收敛 → 学习效应，需延长实验
- 若两组曲线平行且稳定 → 实验结果可信

#### 3.2 仅保留"满周期用户"

只统计 **入组时间距实验结束满 N 天** 的用户，确保每个用户有相同的观察窗口：

```python
# 只保留在实验第 1 天就入组的用户（最老的用户）
valid_users = df[df['cookie_day'] == experiment_start_date]['user_id']
result = df[df['user_id'].isin(valid_users)]
```

**优缺点：**
- ✅ 消除时间窗口不一致的偏差
- ❌ 丢弃了大量后续入组用户，样本量减少，功效下降

#### 3.3 按 Cookie Day 分组的 Delta 图

横轴为 Cookie Day（用户入组后第几天），纵轴为实验组 vs 对照组的指标 **差值（Delta）**：

```
Delta(CTR) = CTR_treatment - CTR_control

Day 0: +8%  ← 新奇效应高峰
Day 1: +5%
Day 2: +3%
Day 3: +2%
Day 4: +2%  ← 趋于稳定
Day 5: +2%
Day 6: +2%
Day 7: +2%  ← 真实长期效果约 +2%
```

若 Delta 趋于平稳，取平稳段的均值作为真实效果。

---

### 4. 工业界实践

| 公司 | Cookie Day 实践 |
|------|----------------|
| **Microsoft (Exp Platform)** | 标准化 "Days Since Treatment" 分析，检测新奇/学习效应 |
| **Netflix** | 对长期推荐效果（留存率）强制观察 7+ cookie days |
| **Airbnb** | 实验报告中内置 cookie day 趋势图，自动标注是否稳定 |
| **字节跳动** | 抖音等产品的推荐实验通常等待 3-5 天指标稳定后再决策 |
| **美团/快手** | 对新推荐策略设置"预热期"（不计入正式统计的前 1-2 天）|

---

### 5. Cookie Day 与 Novelty Effect 的处理策略

**策略一：延长实验周期**
- 至少等待 1-2 个完整的周期（7 天），确保 cookie day 效应消退
- 工业界经验：推荐系统实验通常至少跑 7 天，视指标稳定性而定

**策略二：新用户 vs 老用户分层实验**
- 分别对"首次曝光用户"和"重复曝光用户"统计结果
- 若两组结论一致，结果更可信

**策略三：时间序列检验（CUPED + cookie day）**
- 对每个 cookie day 分别做 CUPED 方差缩减
- 取平稳期的 CUPED 估计量作为最终效果

**策略四：设置 Burn-in 期**
- 实验开始后前 1-3 天数据不纳入统计（"预热期"）
- 只用第 3-14 天数据做决策，规避新奇效应高峰

---

### 6. 常见考点：Cookie Day 相关问题

**Q：你们的 A/B 实验如何处理新奇效应？**
> 我们通过 Cookie Day 分析来检测。具体做法是：将用户按入组天数分层，画出每天的指标 Delta 曲线。如果曲线在第 3-4 天后趋于平稳，说明新奇效应已消退，取稳定段的均值作为真实效果。同时我们会设置 3 天的"预热期"，不将前 3 天的数据纳入正式统计。

**Q：实验跑了 3 天，实验组比对照组高 15%，能上线吗？**
> 不能直接上线。需要先做 Cookie Day 分析，判断这 15% 是真实效果还是新奇效应。如果用户入组第 1 天高 20%、第 2 天高 15%、第 3 天还没收敛，说明可能是新奇效应，需要继续跑。建议至少观察 7 天，等 Delta 曲线平稳后再决策。

**Q：实验结果为什么会随时间衰减？**
> 可能原因：① 新奇效应消退；② 用户习惯改变；③ 实验样本构成变化（早期入组用户 vs 后期入组用户行为差异）。Cookie Day 分析可以区分这三种情况。

**Q：Cookie Day 和 Holdout 实验有什么区别？**
> Cookie Day 分析是在同一个实验内按时间维度分层分析，用于检测效应随时间的变化；Holdout 是在实验结束后保留一批用户不上线新功能，用于测量长期效果（数周到数月）。两者互补，Cookie Day 关注短期收敛，Holdout 关注长期价值。

---

3. **Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED Paper)**
   - https://dl.acm.org/doi/10.1145/2433396.2433413

4. **Evan Miller - How Not To Run an A/B Test (Peeking Problem)**
   - https://www.evanmiller.org/how-not-to-run-an-ab-test.html

5. **Airbnb - Experiment Reporting Framework**
   - https://medium.com/airbnb-engineering/experiment-reporting-framework-4e3fcd29e6c0

6. **Causal Inference in Statistics: A Primer** (Pearl, Glymour, Jewell)
   - 因果推断基础理论

7. **网易互娱 A/B 实验平台技术实践**
   - https://zhuanlan.zhihu.com/p/68252036


---
## AB 实验实战 Checklist（避免踩坑）

### 实验前 Checklist

- [ ] 确定主指标（Primary Metric）和护栏指标（Guardrail Metrics）
  - 主指标：你要优化的（CTR、GMV、留存率）
  - 护栏指标：不能变差的（P99延迟、崩溃率、差评率）
- [ ] 计算最小样本量（Power Analysis）
  - MDE（最小可检测效应）设置为多大？通常 1-5%
  - 显著性水平 α = 0.05，统计效力 β = 0.8
  - 样本量 $n \approx \frac{16\sigma^2}{\delta^2}$（$\sigma$ 是指标方差，$\delta$ 是 MDE）
- [ ] 确定实验单元（是"用户"还是"请求"？）
  - 用户级别：更干净，但样本量要求更高
  - 请求级别：样本量大，但同一用户可能既在实验组又在对照组（SUTVA 违反）
- [ ] AA 测试：实验开始前，两组随机分流，验证基线无显著差异

### 实验中 Checklist

- [ ] 检查流量分配是否均匀（每天监控实验/对照比例）
- [ ] 监控护栏指标（延迟、错误率）是否异常
- [ ] 不要提前停止实验（即使已经显著）—— "偷看效应"（Peeking Problem）使 α 虚高
- [ ] 记录实验期间的外部事件（节假日、竞品发布、系统事故）

### 实验后 Checklist

- [ ] 等待预定时间（不要因为"已经显著"就提前结束）
- [ ] 检查 Novelty Effect：实验初期效果好但后来衰减（用户对新功能有短期好奇心）
- [ ] 分人群分析：整体无显著，特定人群（新用户、高活用户）可能显著
- [ ] 长尾效应：上线后 14 天的效果 vs 实验期间的效果是否一致

### 常见统计陷阱

**陷阱1：多重检验（Multiple Testing）**
- 同时检验10个指标，其中1个显著的概率 = $1-(1-0.05)^{10} \approx 40\%$
- 修正：Bonferroni 修正（$\alpha' = \alpha/10 = 0.005$）

**陷阱2：辛普森悖论**
- 整体 CTR 实验组 > 对照组，但按人群分层后每个人群都是对照组 > 实验组
- 原因：实验组和对照组的人群构成不同（流量分配不均）
- 解法：分层分析，检查各层的方向是否一致

**陷阱3：网络效应（Network Effect）**
- 社交/团购类产品：实验组用户和对照组用户互相影响，SUTVA 假设不成立
- 解法：按"社交群体"而非"用户"分组（Cluster Randomization）
