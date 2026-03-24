# 预算控制与Pacing机制详解

> 日期：2026-03-18 | 领域：广告系统 | 作者：MelonEggLearn

```
┌─────────────────────────────────────────────────────────────────────┐
│                    预算控制与Pacing 全景图                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   广告主设置预算B_total                                               │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│   │ 理想消耗曲线  │────▶│  PID控制器        │────▶│  bid_multiplier│  │
│   │ budget_target│     │  P + I + D       │     │  调整出价      │  │
│   └─────────────┘     └──────────────────┘     └────────────────┘  │
│         ▲                      │                       │           │
│         │                      ▼                       ▼           │
│   F(t)流量分布函数        Anti-windup            实际竞价系统        │
│                           积分限幅                                   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  多级预算层级：Account ▶ Campaign ▶ AdGroup                  │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   Under-Delivery 检测 ──▶ 原因分析 ──▶ 平台自动修复                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 一、为什么需要Pacing

### 1.1 无Pacing时的问题

广告系统天然存在流量不均匀分布的特征。用户活跃度在一天中呈现明显的波峰波谷：

- **早9点**：上班通勤刷手机，流量峰值
- **中午12点**：午休时间，流量次峰值
- **晚8点**：黄金时段，一天中最大峰值

如果广告系统不进行预算控制，将会发生以下问题：

```
时间线示意：
00:00      09:00      12:00      18:00      20:00      24:00
  │          │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼          ▼
[低流量]  [高峰→预算耗尽]        [中流量]  [黄金时段→无预算！]
  $0         $$$0         $0         $0         $0         $0

结果：预算在09:30左右全部消耗完毕，其余时间零曝光
```

**具体危害：**

1. **ROI下降**：早9点未必是转化率最高的时段，错过黄金晚高峰
2. **竞争成本虚高**：流量峰值时竞争激烈，eCPM偏高，性价比低
3. **用户覆盖不均**：同一用户可能早上看到广告，下午完全看不到
4. **广告主体验差**：投放一天但实际只跑了几小时

### 1.2 Pacing的核心目标

Pacing（预算平滑）的本质是一个**预算分配的时间序列调度问题**：

> **目标**：让广告的实际消耗曲线 `budget_spent(t)` 尽可能贴合"理想消耗曲线" `budget_target(t)`，从而在预算耗尽之前最大化广告效果。

```
理想状态：
  budget_spent(t) ≈ budget_target(t) for all t ∈ [0, T]
  budget_spent(T) = B_total   # 刚好花完预算

避免：
  - Over-delivery：超预算（平台需赔付差额）
  - Under-delivery：预算花不完（广告主ROI未最大化）
```

**Pacing的本质不是"省钱"，而是"让钱花在刀刃上"。**

---

## 二、理想消耗曲线的设计

### 2.1 等速分配（Uniform Pacing）

最简单的策略：将全天预算按时间线性分配。

```python
# 等速分配公式
budget_target(t) = B_total × (t / T)

# 例如：日预算1000元，全天24小时
# 每小时目标消耗 = 1000 / 24 ≈ 41.67元
# 前6小时目标 = 250元
# 前12小时目标 = 500元
```

**优点：**
- 实现简单，易于理解
- 不依赖历史数据

**缺点：**
- 完全忽略流量质量波动
- 流量低谷期强行消耗 → 出价虚高
- 流量高峰期反而被限速 → 错过优质流量

### 2.2 流量感知分配（Traffic-Aware Pacing）

更智能的策略：根据历史流量分布，提前预分配预算。

**核心思想：** 如果历史数据显示晚8点的流量占全天20%，那么就为晚8点预留20%的预算。

```
预算曲线公式：
  budget_target(t) = B_total × F(t)

其中：
  F(t) = ∫₀ᵗ f(τ) dτ  （累积流量分布函数）
  f(τ) = 时刻τ的流量密度（由历史数据统计得到）
  F(T) = 1.0            （全天流量归一化）
```

**流量分布估计方法：**

```python
# 离散化为15分钟槽位
time_slots = 96  # 24小时 × 4槽/小时

# 历史流量统计（过去7天同类型广告）
historical_traffic = [...]  # shape: (7, 96)

# 计算平均流量分布
avg_traffic = np.mean(historical_traffic, axis=0)
f_normalized = avg_traffic / avg_traffic.sum()  # 归一化

# 累积分布
F = np.cumsum(f_normalized)

# 任意时刻t的目标消耗
def budget_target(t, B_total):
    slot_idx = int(t / (24*60) * 96)
    return B_total * F[slot_idx]
```

### 2.3 两种方式的对比

```
┌──────────────────┬──────────────────┬──────────────────┐
│    维度           │  Uniform Pacing  │ Traffic-Aware    │
├──────────────────┼──────────────────┼──────────────────┤
│  实现复杂度       │  低               │  中              │
│  数据依赖        │  无               │  历史流量数据     │
│  流量波动适应    │  差               │  好              │
│  冷启动          │  支持            │  需要历史数据     │
│  新广告计划      │  可用            │  参考同类广告     │
│  实时性          │  强（直接计算）  │  依赖预计算结果   │
└──────────────────┴──────────────────┴──────────────────┘
```

**工业实践**：两者结合使用。新广告使用Uniform Pacing兜底，积累数据后切换到Traffic-Aware。

---

## 三、PID控制器详解

### 3.1 为什么用PID

PID（Proportional-Integral-Derivative）控制器是一种经典的反馈控制算法，在工业控制领域已有百年历史。广告Pacing系统借用了PID的核心思想：

```
控制系统类比：
  工业场景：维持水箱水位恒定
    ├── 测量：当前水位（budget_spent）
    ├── 目标：期望水位（budget_target）
    └── 执行：调节进水阀门（bid_multiplier）

  广告场景：维持预算消耗在理想曲线上
    ├── 测量：当前已消耗预算（budget_spent(t)）
    ├── 目标：理想消耗曲线值（budget_target(t)）
    └── 执行：调整出价倍率（bid_multiplier）
```

### 3.2 PID三个分量详解

**误差定义：**

```
e(t) = budget_target(t) - budget_spent(t)

e(t) > 0：消耗慢了，需要提高出价（加速消耗）
e(t) < 0：消耗快了，需要降低出价（减慢消耗）
e(t) = 0：完美跟踪
```

#### P分量（比例控制）

```
P(t) = Kp × e(t)

物理含义：
  - 对当前偏差的即时响应
  - e越大 → 调整幅度越大
  - 典型取值：Kp ∈ [0.01, 0.5]

直觉解释：
  就像开车时，前方有障碍物，越近踩刹车越用力
  当前差距越大，调整越激进
```

**P控制的局限：** 存在稳态误差。比例控制只能缩小误差但无法完全消除，因为当e(t)很小时，P分量也很小，不足以驱动系统到达目标。

#### I分量（积分控制）

```
I(t) = Ki × ∫₀ᵗ e(τ) dτ

离散化：
I(t) = Ki × Σ e(τ_k) × Δt

物理含义：
  - 对历史累积误差的响应
  - 消除稳态偏差
  - 典型取值：Ki ∈ [0.001, 0.05]

直觉解释：
  就像一个"记仇"的控制器
  如果持续偏低消耗，I分量会累积增大，持续推高出价
  直到历史欠账全部补回来
```

**I控制的问题：Integral Windup（积分饱和）**

当系统长时间处于饱和状态（如预算耗尽后e(t)持续为负），积分项会无限累积，导致系统恢复时出现严重超调。

#### D分量（微分控制）

```
D(t) = Kd × de/dt

离散化：
D(t) = Kd × (e(t) - e(t-Δt)) / Δt

物理含义：
  - 对误差变化趋势的预测性响应
  - 抑制超调和振荡
  - 典型取值：Kd ∈ [0.001, 0.1]

直觉解释：
  就像有预见性的驾驶员
  不仅看当前位置（P），也看速度（D）
  快要超过目标时提前减速，防止一头冲过去
```

**D控制的问题：** 对噪声敏感。如果e(t)有随机波动，D项会放大噪声，导致出价剧烈抖动。工业实践中常对e(t)做低通滤波后再计算D项。

### 3.3 完整PID出价公式

```python
class PacingPIDController:
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        # Anti-windup参数
        self.integral_max = 0.5
        self.integral_min = -0.3
    
    def update(self, budget_target, budget_spent, dt=60):
        """
        每分钟调用一次，更新bid_multiplier
        
        Args:
            budget_target: 当前时刻的目标消耗金额
            budget_spent:  当前实际消耗金额
            dt:            更新间隔（秒），默认60秒
        
        Returns:
            bid_multiplier: 出价倍率
        """
        # 计算误差
        error = budget_target - budget_spent
        
        # P分量
        P = self.Kp * error
        
        # I分量（带Anti-windup）
        self.integral += error * dt
        # 积分截断（Anti-windup核心）
        self.integral = np.clip(self.integral, 
                                self.integral_min / self.Ki,
                                self.integral_max / self.Ki)
        I = self.Ki * self.integral
        
        # D分量（低通滤波后计算）
        d_error = (error - self.prev_error) / dt
        D = self.Kd * d_error
        self.prev_error = error
        
        # 合并三个分量
        adjustment = P + I + D
        
        # bid_multiplier限制在合理范围内
        bid_multiplier = np.clip(1.0 + adjustment, 0.1, 3.0)
        
        return bid_multiplier
    
    def get_actual_bid(self, base_bid):
        """实际出价 = 基础出价 × 倍率"""
        return base_bid * self.bid_multiplier
```

### 3.4 Anti-windup机制

**问题场景：**

```
场景：广告主预算耗尽（budget_spent = B_total）
     此时系统停止竞价，但误差 e(t) = budget_target(t) - B_total < 0（持续为负）
     I分量：Ki × Σ负数 → 积分变成非常大的负值

恢复场景（第二天重置预算）：
     正常情况：bid_multiplier 应该接近1.0
     Windup情况：I分量拖出极大负值 → bid_multiplier 极低
     结果：广告完全停止出价，恢复异常缓慢！
```

**Anti-windup解决方案：**

```python
# 方案1：积分截断（Clamping）
# 将积分项限制在[-I_max, I_max]范围内
self.integral = np.clip(self.integral, -I_max, I_max)

# 方案2：条件积分（Conditional Integration）
# 只在输出未饱和时更新积分
if not output_saturated:
    self.integral += error * dt

# 方案3：反向追踪（Back-Calculation）
# 使用实际输出与期望输出的差值来修正积分
anti_windup_correction = (u_actual - u_desired) / Tt
self.integral -= anti_windup_correction * dt
```

### 3.5 参数调优方法

```
Kp（比例增益）调优：
  - 先将Ki=0, Kd=0
  - 逐步增大Kp直到系统出现振荡
  - 取振荡临界值的50%作为Kp

Ki（积分增益）调优：
  - 固定Kp，逐步增大Ki
  - 观察稳态误差消除速度
  - 太大会导致超调，太小收敛慢

Kd（微分增益）调优：
  - 最后调，且通常设置较小值
  - 主要作用：减少超调
  - 注意：D项对噪声敏感，不宜过大

典型取值参考（广告系统，以日预算为单位）：
  Kp: 0.05 ~ 0.2    # 每元误差调整5%-20%的倍率
  Ki: 0.001 ~ 0.02  # 积分增益较小，防止过激反应
  Kd: 0.01 ~ 0.1    # 微分增益适中
```

---

## 四、Under-Delivery 问题

### 4.1 定义与业务影响

**Under-Delivery**（欠交付）是指广告计划在投放周期内，**实际消耗金额显著低于广告主设定的预算上限**。

```
严重程度分级：
  轻度：实际消耗 = 80%~95% 预算    → 轻微调整
  中度：实际消耗 = 50%~80% 预算    → 需要介入排查
  重度：实际消耗 < 50% 预算        → 立即告警，广告主投诉风险

业务影响：
  广告主视角：花不出去钱，曝光/转化目标达不成
  平台视角：收入损失，广告主信任度下降，可能流失到竞品
```

### 4.2 常见原因分析

#### 原因1：定向过窄

```
现象：Campaign日预算1万，但每天只花500元
原因：
  - 地域：仅定向北京朝阳区CBD，潜在用户极少
  - 兴趣：同时要求"汽车+健身+高端消费+25-30岁男性"
  - 设备：仅定向iPhone14 Pro Max以上机型
  
排查：检查目标受众规模估算（Audience Size）
  - Meta广告：受众选择时会显示"预估受众规模"
  - Google Ads：关键词规划师显示搜索量
  
解决：
  - 放宽地域定向（城市级 → 省级）
  - 减少兴趣标签（2-3个核心标签）
  - 使用Lookalike Audience扩展相似用户
```

#### 原因2：出价过低

```
现象：竞价频繁参与但极少赢得展示
排查：Win Rate = 赢得竞价次数 / 参与竞价次数
  - Win Rate < 10%：出价极度偏低
  - Win Rate 10%~30%：出价偏低
  - Win Rate > 50%：出价合理

诊断：
  查看 eCPM 建议值 vs 当前出价
  if 当前bid < 建议eCPM × 0.8：
      → 出价不足，需要提价

解决：
  - 提高CPC/CPM出价上限
  - 启用自动出价（让平台帮助优化）
  - 分时段出价：高峰期适当提价
```

#### 原因3：创意质量差

```
现象：赢得竞价但CTR极低，eCPM计算值偏低
机制：
  eCPM = CTR × CPC × 1000（对于CPC广告）
  或
  eCPM = pCTR × pCVR × 出价 × 1000（对于CPA广告）
  
  即使出价相同，CTR低的广告eCPM也低，在竞价中处于劣势

排查：
  - 检查CTR是否低于行业基准（各行业不同，一般1%-5%）
  - 检查Quality Score（Google Ads质量得分）
  - A/B测试不同创意

解决：
  - 优化广告图片/视频（更吸引人的视觉内容）
  - 优化文案（更清晰的价值主张）
  - 优化落地页（降低跳出率，提升CVR）
```

#### 原因4：预算设置过高

```
现象：市场上根本没有足够的匹配流量
计算：
  市场可用流量总价值 = 每日可竞价次数 × 平均eCPM / 1000
  
  如果 B_total >> 市场可用流量总价值：
      → Under-Delivery不可避免

例子：
  某细分市场每天只有1000次有效竞价机会
  每次平均花费5元
  市场总流量价值 = 5000元/天
  
  但广告主设置日预算50000元
  → 最多只能花5000元（90% under-delivery）
```

### 4.3 系统化排查漏斗

```
广告效果漏斗排查法：

层级1：曝光（Impression）
  检查：predicted_impressions vs actual_impressions
  若：actual << predicted
  可能原因：预算、出价、受众太窄
      ↓
层级2：竞价参与（Auction Entry）
  检查：参与竞价次数 vs 目标受众请求次数
  若：参与率低
  可能原因：定向不匹配、广告计划被暂停/限制
      ↓
层级3：竞价胜出（Win）
  检查：Win Rate
  若：Win Rate < 15%
  可能原因：出价过低
      ↓
层级4：点击（Click）
  检查：CTR
  若：CTR < 行业基准50%
  可能原因：创意质量差
      ↓
层级5：转化（Conversion）
  检查：CVR
  若：CVR异常低
  可能原因：落地页问题、受众不精准
```

### 4.4 平台自动处理机制

```python
# 平台的自动Under-Delivery修复逻辑（伪代码）
def auto_fix_under_delivery(campaign):
    current_delivery_rate = campaign.spent / campaign.budget
    hours_elapsed = campaign.hours_elapsed
    expected_rate = hours_elapsed / 24.0
    
    if current_delivery_rate < expected_rate * 0.7:
        # 欠交付超过30%，触发自动修复
        
        if campaign.win_rate < 0.15:
            # 出价问题：自动提价
            campaign.bid *= 1.2
            log("auto-increase bid by 20%")
        
        elif campaign.audience_size < MIN_AUDIENCE_SIZE:
            # 受众问题：放宽定向
            campaign.expand_audience()
            log("auto-expand audience")
        
        elif campaign.ctr < INDUSTRY_BENCHMARK_CTR * 0.5:
            # 创意问题：推送更多A/B测试
            campaign.trigger_creative_ab_test()
            log("trigger creative optimization")
```

---

## 五、多级预算控制

### 5.1 预算层级结构

广告平台通常支持三级预算控制：

```
广告账户 (Account)
├── 账户总预算上限（月度/年度）
│
├── 广告计划1 (Campaign 1) ── 日预算: ¥500
│   ├── 广告组1 (AdGroup 1) ── 日预算: ¥200
│   │   ├── 广告创意A
│   │   └── 广告创意B
│   └── 广告组2 (AdGroup 2) ── 日预算: ¥300
│       ├── 广告创意C
│       └── 广告创意D
│
└── 广告计划2 (Campaign 2) ── 日预算: ¥800
    ├── 广告组3 (AdGroup 3) ── 日预算: ¥500
    └── 广告组4 (AdGroup 4) ── 日预算: ¥300
```

**关键约束：**
- AdGroup实际消耗 ≤ AdGroup日预算
- Campaign实际消耗 ≤ Campaign日预算
- Account实际消耗 ≤ Account总预算
- 三个层级同时有效，取最小约束

### 5.2 日预算 vs 总预算（Flight Budget）

```
日预算（Daily Budget）：
  - 每天重置
  - 适合持续性投放
  - 平台保证每天不超出，但可能±20%波动
  - 例：每天¥1000，连续投放30天

总预算（Flight Budget / Lifetime Budget）：
  - 投放周期内总消耗上限
  - 适合活动促销投放
  - 平台可跨日灵活分配（某天多花某天少花）
  - 例：双11活动7天，总预算¥50000
```

```python
# Flight Budget跨日分配示例
class FlightBudgetAllocator:
    def __init__(self, total_budget, start_date, end_date):
        self.total_budget = total_budget
        self.remaining_days = (end_date - start_date).days + 1
        self.remaining_budget = total_budget
    
    def daily_allocation(self, predicted_traffic_ratio):
        """
        每天分配预算：考虑剩余预算和剩余天数
        predicted_traffic_ratio: 今天流量占剩余周期流量的比例
        """
        if self.remaining_days == 0:
            return 0
        
        # 等速兜底：剩余预算 / 剩余天数
        uniform_allocation = self.remaining_budget / self.remaining_days
        
        # 流量感知调整
        traffic_aware = self.remaining_budget * predicted_traffic_ratio
        
        # 混合策略：70%流量感知 + 30%等速
        daily_budget = 0.7 * traffic_aware + 0.3 * uniform_allocation
        
        return daily_budget
```

### 5.3 Shared Budget（共享预算）

**场景：** 广告主有10个Campaign，但总预算有限，希望平台自动在各Campaign间最优分配。

```
传统方式：
  Campaign1: ¥100
  Campaign2: ¥100
  ...
  Campaign10: ¥100
  
  问题：Campaign3已花完预算但机会很多，Campaign7预算很多但没流量
  
共享预算方式：
  Shared Budget: ¥1000 (所有Campaign共享)
  
  平台根据实时竞价效果动态分配：
  Campaign3（高ROI）: 今天分配¥400
  Campaign7（低ROI）: 今天分配¥50
  其他：按效果分配剩余¥550
```

**平台分配算法（简化版）：**

```python
def shared_budget_allocation(campaigns, shared_budget):
    """
    基于预期ROI的共享预算分配
    """
    # 计算每个Campaign的效率指标
    efficiency_scores = []
    for campaign in campaigns:
        # 效率 = 历史ROAS（广告花费回报率）× 受众饱和度权重
        roas = campaign.historical_revenue / campaign.historical_spend
        saturation_factor = 1.0 - campaign.audience_saturation  # 受众饱和度越低，机会越多
        score = roas * saturation_factor
        efficiency_scores.append(score)
    
    # Softmax归一化（保证每个Campaign都能分到预算）
    scores_array = np.array(efficiency_scores)
    allocation_weights = softmax(scores_array)
    
    # 按权重分配预算
    allocations = shared_budget * allocation_weights
    
    return allocations
```

### 5.4 加速消耗 vs 标准消耗

```
加速消耗（Accelerated Delivery）：
  - 不限速，来了流量就竞价
  - 优点：触达更多用户，不错过任何机会
  - 缺点：预算可能提前耗尽，失去后续流量
  - 适用：预算充足、追求最大覆盖的品牌广告

标准消耗（Standard Delivery）：
  - 启用Pacing，平滑分配预算
  - 优点：全天均衡投放，不错过黄金时段
  - 缺点：某些流量高峰期会主动让步
  - 适用：预算有限、追求ROI的效果广告

注意：Google Ads在2019年已废弃"加速消耗"选项
     认为标准消耗几乎在所有情况下效果更好
```

---

## 六、Pacing的工程实现

### 6.1 高频更新架构

```
更新频率权衡：
  
  每秒更新：精度最高，但：
    - 需要实时预算统计（高延迟）
    - 频繁数据库操作（高负载）
    - 实际无必要（出价决策不需要秒级精度）
  
  每分钟更新（推荐）：
    - 平衡精度与开销
    - 1分钟内的误差可忽略
    - 系统负载可控
  
  每5分钟更新：
    - 节省系统开销
    - 但可能对突发流量响应不足

系统架构：
  
  竞价服务器（Ad Servers）
       │ 每次竞价消耗记录
       ▼
  预算统计服务（Budget Tracker）
       │ 聚合统计，每分钟推送
       ▼
  Pacing控制器（Pacing Controller）
       │ 更新bid_multiplier，每分钟
       ▼
  配置下发（Config Push）
       │ 广播到所有竞价服务器
       ▼
  竞价服务器读取新的bid_multiplier
```

### 6.2 分布式预算统计

大型广告平台的竞价请求由数百台服务器处理，需要分布式统计：

```python
# 方案1：Redis集中计数（适合中小型平台）
import redis

class RedisbudgetTracker:
    def __init__(self):
        self.redis = redis.Redis(host='budget-redis', port=6379)
    
    def record_spend(self, campaign_id, amount):
        """记录消耗（原子操作）"""
        key = f"campaign:{campaign_id}:daily_spend"
        self.redis.incrbyfloat(key, amount)
        # 设置过期时间（次日自动重置）
        self.redis.expireat(key, tomorrow_midnight_timestamp())
    
    def get_spend(self, campaign_id):
        """查询当前消耗"""
        key = f"campaign:{campaign_id}:daily_spend"
        value = self.redis.get(key)
        return float(value) if value else 0.0

# 问题：高并发下Redis成为瓶颈
# 解决：使用Redis Cluster横向扩展

# 方案2：本地缓存 + 批量同步（适合大型平台）
class LocalCacheBudgetTracker:
    def __init__(self, sync_interval=5):
        self.local_spend = defaultdict(float)  # 本地累计
        self.sync_interval = sync_interval     # 5秒同步一次
    
    def record_spend(self, campaign_id, amount):
        """本地记录，不立即同步"""
        self.local_spend[campaign_id] += amount
    
    def sync_to_central(self):
        """定期批量同步到中央存储"""
        for campaign_id, spend in self.local_spend.items():
            central_db.increment(campaign_id, spend)
        self.local_spend.clear()
```

### 6.3 预算统计的误差容忍

```
预算超出的两类来源：

1. 竞价成功但尚未结算（In-Flight）
   - 广告已展示但消耗记录还在传输中
   - 延迟：几秒到几分钟
   - 影响：实际消耗可能比统计多5%-10%

2. 分布式竞价的统计延迟
   - 多台服务器各自记录，需要汇总
   - 中央统计比实际消耗有滞后
   - 影响：Pacing控制器看到的数据有延迟

工程解决方案：
  预算保护线（Budget Guard）= 0.95 × B_total
  即在到达预算保护线时就开始降速，而非等到100%
  
  预留5%作为"误差缓冲"：
  - 覆盖In-Flight消耗
  - 覆盖统计延迟导致的超出
  - 保护广告主利益（不超预算）

实际效果：
  正常情况：实际消耗 ≈ 97%~100% × B_total（合理范围）
  Over-delivery率：< 3%（行业可接受水平）
```

---

## 七、面试高频考点

### 考点1：PID参数如何调优？如何判断Kp/Ki/Kd是否合适？

**标准答案思路：**

```
判断指标：
  - 稳态误差（Steady-State Error）：I分量不够时偏高
  - 超调量（Overshoot）：P/I偏大或D不足时发生
  - 收敛速度：P不足时收敛慢
  - 振荡频率：P/I过大时出现持续振荡

调优步骤：
  1. Ziegler-Nichols法：先找临界增益Ku，再按经验公式设定
  2. 手动调优：Kp→Ki→Kd依次调，二分搜索
  3. 在线学习：用强化学习自动调参（现代系统）

广告场景特殊性：
  - 预算是单调递增的（只增不减），误差函数非对称
  - 超出预算（over-delivery）比欠交付更严重，需要不对称调参
  - Kp应对"超支"更敏感（更大的负向修正）
```

### 考点2：如果广告预算在中午突然用完，Pacing系统应该如何响应？

**标准答案思路：**

```
短期：立即停止竞价（bid_multiplier → 0）
     不是设为0.1（还会消耗），是直接停止参与竞价

Anti-windup处理：
  - 积分项的负值累积（大量负误差）需要被截断
  - 否则次日恢复时需要很长时间才能正常出价

根因分析：
  中午耗尽预算的可能原因：
  1. Pacing参数问题（Kp太大，响应过激）
  2. 流量预测不准（误判中午流量少）
  3. 某次异常流量（竞价量突然暴增）

预防机制：
  - 每小时设置子预算上限（避免单小时消耗过多）
  - 流量异常检测（突增时主动降速）
  - 分钟级监控告警
```

### 考点3：Under-Delivery的根本原因是什么？如何用数据排查？

**标准答案思路：**

```
排查漏斗（从顶到底）：

Step1: 受众规模检查
  metric: estimated_reach（预估受众规模）
  threshold: < 1000人/天 → 定向过窄
  action: 放宽定向条件

Step2: 竞价参与率
  metric: auction_entry_rate = entered_auctions / eligible_requests
  threshold: < 50% → 广告计划有问题
  action: 检查广告计划状态、定向匹配

Step3: 竞价胜率
  metric: win_rate = won_auctions / entered_auctions
  threshold: < 15% → 出价不足
  action: 提高出价或启用自动出价

Step4: 点击率
  metric: CTR = clicks / impressions
  threshold: < 行业基准50% → 创意问题
  action: 优化创意素材

Step5: 转化率（若有）
  metric: CVR = conversions / clicks
  threshold: 极低 → 落地页问题
  action: 优化落地页
```

### 考点4：Shared Budget如何在多个Campaign间分配？平台用什么算法？

**标准答案思路：**

```
核心问题：多背包问题（Multiple Knapsack Problem）的在线变体

贪心近似算法：
  1. 计算每个Campaign的边际效用（增加1元预算能带来多少ROI提升）
  2. 按边际效用排序，依次分配
  3. 动态重新评估（每小时或每次有Campaign耗尽预算时）

实际平台做法：
  Google Ads：基于预期转化量最大化分配
  Facebook Ads：CBO（Campaign Budget Optimization）
    - 实时计算各AdSet的机会集
    - 按照预期ROAS动态分配
    - 考虑受众疲劳度（避免单一AdSet过度投放）

关键约束：
  - 每个Campaign有最低保障预算（避免某些Campaign被完全抢占）
  - 平台通常不会让单Campaign拿走超过60%的共享预算
```

### 考点5：Pacing控制中，如何处理"白天流量特别好但预算在深夜就耗尽"的情况？

**标准答案思路：**

```
问题本质：凌晨有流量垃圾/刷量，消耗了真实预算

解决方案层次：
Layer1: 流量质量过滤（预防）
  - 异常IP/设备过滤
  - 点击欺诈检测（Click Fraud Detection）
  - 凌晨0-6点流量质量评分 → 低质量请求直接拒绝竞价

Layer2: 分时段出价（主动控制）
  - 设置凌晨时段出价系数极低（0.1-0.3x）
  - 保留预算给白天优质流量

Layer3: 子预算限制（硬约束）
  - 每小时消耗不超过日预算的8%（24小时均分+20%弹性）
  - 凌晨0-6点合计不超过日预算的5%

Layer4: 事后补偿（被动恢复）
  - 检测到异常消耗后，次日自动提升预算（有上限）
  - 或向广告主退款并重新投放
```

---

## 参考资料

1. **Agarwal et al. (2014)**《Budget Pacing for Targeted Online Advertisements》- Yahoo Research
2. **Lee et al. (2013)**《Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising》- LinkedIn
3. **Google Ads Help**《About budget types and pacing》
4. **Meta Business**《Campaign Budget Optimization Best Practices》
5. **Ziegler-Nichols方法**《Optimum Settings for Automatic Controllers》(1942)
6. 《分布式系统中的预算计数问题》- 字节跳动技术博客

---

*MelonEggLearn | 广告系统学习笔记 | 2026-03-18*
