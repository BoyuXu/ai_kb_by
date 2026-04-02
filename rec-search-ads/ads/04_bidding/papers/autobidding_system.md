# 智能出价Autobidding系统架构

> 日期：2026-03-18 | 领域：广告系统 | 标签：Autobidding, Smart Bidding, 对偶优化, PID, RL, KKT

---

```
┌────────────────────────────────────────────────────────────────────┐
│                     Autobidding 系统全景                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  广告主                                                              │
│  ┌─────────────────────────────────────────────┐                   │
│  │ 只设目标：target_CPA / target_ROAS / 最大化  │                   │
│  └─────────────────────────────────────┬───────┘                   │
│                                         │                           │
│                                         ▼                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                   Autobidding 引擎                        │      │
│  │                                                           │      │
│  │  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │      │
│  │  │ 信号层(40+)   │  │ 预估模型层   │  │ 出价优化层   │  │      │
│  │  │ 设备/位置/时间 │→│ pCTR/pCVR   │→│ 对偶优化/PID │  │      │
│  │  │ 受众/竞争环境 │  │ 价值预估     │  │ RL调节       │  │      │
│  │  └───────────────┘  └──────────────┘  └──────┬───────┘  │      │
│  │                                               │           │      │
│  └───────────────────────────────────────────────┼──────────┘      │
│                                                  ▼                  │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │         RTB竞价市场（eCPM统一竞价）                       │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  三大平台对比：                                                       │
│  Google Smart Bidding │ 字节Autobidding（oCPM）│ 腾讯ADQ             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 一、Autobidding的定义与价值主张

### 1.1 什么是Autobidding

**Autobidding（自动出价/智能出价）**是一种广告出价范式：

> **广告主只声明业务目标，平台全权代理每次竞价的出价决策。**

传统手动出价的痛点：
```
手动出价时代：
  广告主需要：
  - 对每个关键词/受众/版位分别设置出价
  - 每天监控数据，手动调整
  - 大促时紧急调价
  - 平衡多个Campaign的预算分配
  
  结果：
  ❌ 响应延迟（人工调整 vs 实时竞价）
  ❌ 规模瓶颈（Campaign数量多，人力不够）
  ❌ 信息劣势（平台有实时竞争数据，广告主没有）
  ❌ 次优决策（人脑无法处理40+维度信号）
```

Autobidding的价值主张：
```
Autobidding时代：
  广告主只需要：
  - 设定业务目标（CPA=50元 或 ROAS=3 或 最大化转化）
  - 设定预算上限
  - 上传素材和受众定向
  
  平台全自动：
  ✅ 实时评估每次竞价机会的价值
  ✅ 利用40+信号优化出价
  ✅ 跨时段、跨设备动态调价
  ✅ 在预算约束内最大化转化量/ROAS
```

### 1.2 三种核心模式

```
┌─────────────────────────────────────────────────────────────┐
│  模式1：目标CPA（tCPA）                                       │
│  ─────────────────────────────────────────────────────────  │
│  目标：每次转化成本 ≤ CPA_target                              │
│  适合：以转化量为KPI的效果广告主                              │
│  优化目标：在tCPA约束下最大化转化量                           │
│                                                               │
│  bid* = value_per_conversion × P(conversion)                 │
│       = CPA_target × pCVR × pCTR                             │
├─────────────────────────────────────────────────────────────┤
│  模式2：目标ROAS（tROAS）                                     │
│  ─────────────────────────────────────────────────────────  │
│  目标：广告收入 / 广告支出 ≥ ROAS_target                     │
│  适合：以营收为KPI的电商广告主                                │
│  优化目标：在tROAS约束下最大化总收入                          │
│                                                               │
│  bid* = (expected_order_value / ROAS_target) × pCVR × pCTR  │
├─────────────────────────────────────────────────────────────┤
│  模式3：最大化转化量（Maximize Conversions）                  │
│  ─────────────────────────────────────────────────────────  │
│  目标：在预算耗尽前获得最多转化                               │
│  适合：有固定预算、不设CPA限制的广告主                        │
│  优化目标：max Σ conversions, s.t. Σ cost ≤ Budget           │
│                                                               │
│  bid* = value / λ    （λ由预算约束决定，见第四节）            │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 三大平台对比

| 维度 | Google Smart Bidding | 字节跳动Autobidding | 腾讯ADQ |
|------|---------------------|---------------------|---------|
| 竞价货币 | CPC/CPA/ROAS | eCPM（统一） | eCPM |
| 信号数量 | 40+（官方公布） | 未公开，但包含兴趣图谱 | 未公开 |
| 模型更新频率 | 实时（每次竞价） | 实时 | 实时 |
| Portfolio支持 | ✅ 跨Campaign共享预算 | 部分支持 | 部分支持 |
| 学习期 | 约2-4周（50次转化） | 约1-2周 | 约2周 |
| 强项场景 | 搜索广告、购物广告 | 短视频、信息流 | 社交、小程序 |
| 透明度 | 较高（有解释工具） | 较低（黑盒） | 较低 |

---

## 二、Google Smart Bidding架构（深入）

### 2.1 实时竞价信号：40+维度

Google Smart Bidding在每次竞价时融合40+实时信号：

```
信号分类：

┌──────────────────────────────────────────────────────┐
│  用户意图信号                                         │
│  - 搜索词（查询词本身）                               │
│  - 搜索历史（过去14天）                               │
│  - 页面内容（展示网络）                               │
│  - 搜索合作伙伴网站                                   │
├──────────────────────────────────────────────────────┤
│  用户属性信号                                         │
│  - 设备类型（手机/PC/平板）                           │
│  - 操作系统（iOS/Android）                            │
│  - 浏览器                                            │
│  - 用户位置（实时位置 vs 目标位置）                   │
│  - 受众列表（再营销/相似受众）                        │
│  - 人口统计（年龄/性别/收入段）                       │
├──────────────────────────────────────────────────────┤
│  时间上下文信号                                       │
│  - 日期（工作日/周末）                                │
│  - 时段（小时）                                       │
│  - 与用户上次互动的时间间隔                           │
│  - 季节性（节假日）                                   │
├──────────────────────────────────────────────────────┤
│  竞争环境信号                                         │
│  - 竞争对手出价估计（拍卖洞察）                       │
│  - 该广告位的历史赢率                                 │
│  - 当前时段竞争强度                                   │
├──────────────────────────────────────────────────────┤
│  广告主历史信号                                       │
│  - 账户层级转化历史                                   │
│  - Campaign/AdGroup历史                               │
│  - 创意质量得分                                       │
└──────────────────────────────────────────────────────┘
```

### 2.2 机器学习模型架构

Google Smart Bidding的核心是**竞价时实时预测转化概率**：

```python
# Google Smart Bidding 模型简化架构（推测）

class SmartBiddingModel:
    """
    在每次竞价时（<100ms），实时推理：
    P(conversion | user_context, ad, query, ...)
    """
    
    def __init__(self):
        # 稀疏特征：搜索词、用户ID、广告ID等
        self.sparse_embeddings = SparseEmbeddingLayer(vocab_size=1e9)
        
        # 稠密特征：时间、位置、设备等
        self.dense_features = DenseFeatureEncoder()
        
        # 跨特征交叉
        self.cross_layer = CrossNetwork(num_layers=6)
        
        # 深层MLP
        self.mlp = MLP(layers=[1024, 512, 256, 128, 1])
    
    def predict_conversion_prob(self, context):
        sparse_feat = self.sparse_embeddings(context.sparse)
        dense_feat = self.dense_features(context.dense)
        
        combined = concat(sparse_feat, dense_feat)
        crossed = self.cross_layer(combined)
        
        p_convert = sigmoid(self.mlp(crossed))
        return p_convert
    
    def compute_bid(self, context, target_cpa):
        p_convert = self.predict_conversion_prob(context)
        
        # 最优出价 = 价值 × 转化概率
        bid = target_cpa * p_convert
        
        # 预算约束调整（通过λ）
        bid = bid / self.lambda_budget  # 见第四节
        
        return bid
```

### 2.3 Portfolio Bidding：跨Campaign共享策略

```
Portfolio Bidding 概念：
  传统：每个Campaign独立管理出价和预算
  Portfolio：多个Campaign共享一个智能出价策略 + 共享预算池

优点：
  1. 更大的数据量 → 模型学习更快，更准确
  2. 预算跨Campaign自动分配 → 哪个Campaign ROI高，钱往那边流
  3. 避免Campaign间的内部竞争（同一广告主不同Campaign互相抬价）

数学表达：
  多Campaign联合优化问题：
  max  Σ_{c∈campaigns} Σ_{i∈auctions_c} conv_i × bid_i
  s.t. Σ_{c} cost_c ≤ Total_Budget
       CPA_c ≤ target_CPA  ∀c

  通过共享λ（拉格朗日乘子）统一协调各Campaign出价
```

### 2.4 Smart Bidding演进路线

```
历史演进：

2016: Enhanced CPC (eCPC)
  └── 半自动：广告主手动出价，系统微调±30%
  └── 信号：设备、位置、受众
  └── 局限：调整空间有限，仍需人工监控

2017: Target CPA (tCPA)
  └── 全自动：广告主设CPA目标，系统全权出价
  └── 信号：扩展到20+维度
  └── 局限：只能优化转化量，不能优化价值

2018: Target ROAS (tROAS)
  └── 价值维度：不只是转化量，而是转化价值（订单金额）
  └── 需要：转化价值数据（订单金额回传）
  └── 适合：电商、旅游等有明确订单价值的场景

2019-2021: Maximize Conversions / Maximize Conversion Value
  └── 无约束优化：不设CPA/ROAS目标，只设预算
  └── 平台完全自主：自由度最高，效果最依赖数据量

2022+: Performance Max (PMax)
  └── 跨渠道：搜索+展示+YouTube+购物统一优化
  └── AI生成内容：自动组合创意
  └── 完全自动化：广告主只设目标和素材
```

---

## 三、字节跳动Autobidding系统

### 3.1 oCPM生态：统一框架

字节的核心是以**eCPM为竞价货币**的统一竞价框架：

```
字节广告竞价统一化：

目标      出价方式         eCPM计算
─────────────────────────────────────────────────────
CPM       直接出CPM       eCPM = CPM_bid
CPC       出每次点击价    eCPM = CPC × pCTR × 1000
oCPM      出每个转化价    eCPM = CPA × pCTR × pCVR × 1000
oCPC      出每次点击价    eCPM = CPC_eff × pCTR × 1000
ROAS      出ROAS目标      eCPM = (value/ROAS) × pCTR × pCVR × 1000

所有广告在同一个eCPM维度公平竞争
平台收益 = Σ winning_eCPM / 1000
```

### 3.2 粗排-精排-竞价三阶段出价

字节广告系统的多阶段出价架构：

```
候选广告池（数百万）
       ↓
┌──────────────────────────────────────────────────────────┐
│  阶段1：粗排（Rough Ranking）                             │
│  目标：从数百万快速筛选数千                              │
│  模型：轻量级线性模型或小MLP                             │
│  延迟：<5ms                                              │
│  出价：简化版eCPM（少量特征）                            │
└──────────────────────────────────────────────────────────┘
       ↓ 数千候选
┌──────────────────────────────────────────────────────────┐
│  阶段2：精排（Fine Ranking）                              │
│  目标：从数千精选数十                                     │
│  模型：大规模深度模型（DIN/Transformer）                  │
│  延迟：<20ms                                             │
│  出价：精准eCPM（全量特征）                              │
│  多任务：同时预测CTR/CVR/交互率/完播率                   │
└──────────────────────────────────────────────────────────┘
       ↓ 数十候选
┌──────────────────────────────────────────────────────────┐
│  阶段3：竞价（Auction）                                   │
│  目标：最终决定展示哪个广告                               │
│  模型：精确出价优化 + Autobidding调节                     │
│  延迟：<5ms（竞价决策）                                  │
│  出价：final_eCPM = base_eCPM × bid_multiplier(λ, PID)  │
└──────────────────────────────────────────────────────────┘
       ↓ 1个广告展示
```

### 3.3 OCPX + RL的结合

字节在Autobidding中引入强化学习，参考论文：**"Auto Bidding using Reinforcement Learning"**

```python
# 字节AutoBidding RL框架（基于论文推测的简化版）

# MDP建模
class AutoBiddingMDP:
    """
    State:  当前广告主的状态（剩余预算、已花费、时段、CPA目标达成情况）
    Action: 出价乘数调整 Δk（在基础出价上乘以调节系数）
    Reward: 转化价值 - λ × 成本  （平衡转化量和CPA约束）
    """
    
    def get_state(self, advertiser_id, timestamp):
        return {
            "remaining_budget": get_remaining_budget(advertiser_id),
            "budget_ratio": remaining / total_budget,
            "time_ratio": timestamp / day_end,
            "current_cpa": get_current_cpa(advertiser_id),
            "cpa_ratio": current_cpa / target_cpa,
            "recent_win_rate": get_win_rate(advertiser_id, last_1h=True),
            "market_competition": get_competition_index(timestamp)
        }
    
    def compute_reward(self, conversions, cost, target_cpa, lambda_budget):
        # 转化价值
        value = sum(conversion.value for c in conversions)
        
        # 约束惩罚（CPA超标时惩罚）
        cpa_penalty = max(0, cost/max(conversions,1) - target_cpa) * lambda_penalty
        
        # 预算惩罚（浪费或超支时惩罚）
        budget_penalty = budget_waste_penalty(cost)
        
        return value - cpa_penalty - budget_penalty

# RL训练方式：离线 + 在线
# 1. 离线：用历史竞价日志训练初始策略（行为克隆）
# 2. 在线：在真实环境中探索-利用，持续更新
# 注意：在线RL需要小心exploration（避免大幅偏离baseline导致广告主CPA飙升）
```

### 3.4 短视频广告的特殊性

字节（抖音/TikTok）广告有独特的优化目标：

```
短视频广告特有指标：
┌─────────────────────────────────────────────────────────┐
│  互动类指标                                              │
│  - 完播率（播完整个视频的比例）                          │
│  - 有效播放率（≥3秒/≥5秒）                              │
│  - 点赞率、评论率、分享率                                │
│  - 关注率（看完广告后关注账号）                          │
├─────────────────────────────────────────────────────────┤
│  转化类指标                                              │
│  - 商品点击率（视频中的商品链接）                        │
│  - 小程序转化率                                          │
│  - App下载率                                             │
│  - 直播间进入率（广告引流到直播）                        │
├─────────────────────────────────────────────────────────┤
│  品牌类指标                                              │
│  - 品牌记忆度（品牌搜索量提升）                          │
│  - 情感倾向（评论情感分析）                              │
└─────────────────────────────────────────────────────────┘

多目标出价：
  eCPM = w1 × pCTR × CPC_value
       + w2 × pComplete × CPComplete_value
       + w3 × pCVR × CPA_value
       + w4 × pShare × CPShare_value

  权重w由广告主目标决定，系统自动平衡
```

---

## 四、Autobidding的核心算法

### 4.1 对偶优化方法

**问题建模**：在预算约束下最大化广告主价值

```
原始问题（Primal Problem）：

max  Σ_{i} x_i × v_i          # 最大化总价值（转化价值之和）
s.t. Σ_{i} x_i × c_i ≤ B     # 预算约束
     x_i ∈ {0,1}               # 是否赢得第i次竞价

其中：
  i：单次竞价机会
  x_i：是否赢得该次竞价（1=赢，0=输）
  v_i：赢得该次竞价的期望价值
  c_i：赢得该次竞价的成本
  B：预算上限
```

**拉格朗日对偶化**：

```python
# 将约束加入目标函数
# 拉格朗日函数：
# L(x, λ) = Σ x_i × v_i - λ × (Σ x_i × c_i - B)
#          = Σ x_i × (v_i - λ × c_i) + λ × B

# 对每次竞价独立优化：
# 当 v_i - λ × c_i > 0 时，参与竞价（x_i=1）
# 当 v_i - λ × c_i ≤ 0 时，放弃竞价（x_i=0）

# 最优出价（在二价拍卖中）：
# bid_i* = v_i / λ

# 物理含义：
# λ = 预算的边际价值（每多花1元能多获得多少价值）
# λ大 → 预算紧张 → 出价保守（bid低）
# λ小 → 预算充裕 → 出价激进（bid高）
```

```python
# 出价公式核心
def compute_optimal_bid(value_per_conversion, p_convert, lambda_budget):
    """
    value_per_conversion: 广告主认为一次转化值多少
    p_convert: 这次曝光的转化概率（模型预测）
    lambda_budget: 预算约束的拉格朗日乘子
    
    最优出价 = 期望价值 / λ
    """
    expected_value = value_per_conversion * p_convert
    bid = expected_value / lambda_budget
    return bid

# 等价地，对于tCPA模式：
def compute_tCPA_bid(target_cpa, p_ctr, p_cvr, lambda_budget):
    # 期望价值 = CPA_target × pCTR × pCVR
    expected_value = target_cpa * p_ctr * p_cvr
    # 出价（eCPM）
    bid_ecpm = expected_value * 1000 / lambda_budget
    return bid_ecpm
```

### 4.2 λ的物理含义与二分搜索

```
λ（拉格朗日乘子）的直觉理解：

λ = dRevenue/dBudget
  = "多花1元预算能多获得多少转化价值"
  = 预算的"性价比"

不同λ值的含义：
  λ = 1.0：花1元能获得1元价值（盈亏平衡）
  λ = 2.0：花1元能获得2元价值（高回报，预算紧张）
  λ = 0.5：花1元只获得0.5元价值（低回报，预算宽松）

对出价的影响：
  bid* = value / λ
  λ越大 → bid越小 → 赢得竞价少 → 少花钱
  λ越小 → bid越大 → 赢得竞价多 → 多花钱
```

```python
def find_lambda_by_binary_search(
    auctions,           # 竞价机会列表
    budget,             # 总预算
    value_per_conv,     # 每次转化价值
    tolerance=0.01,
    max_iter=100
):
    """
    二分搜索找到最优λ，使得在λ对应的出价策略下，
    实际花费 ≈ 预算
    
    核心：λ越大→出价越低→花费越少；λ越小→出价越高→花费越多
    """
    lambda_low = 0.01    # λ下界（出价极高，花费极多）
    lambda_high = 100.0  # λ上界（出价极低，花费极少）
    
    for _ in range(max_iter):
        lambda_mid = (lambda_low + lambda_high) / 2
        
        # 在lambda_mid下模拟竞价，计算总花费
        total_spend = simulate_spend(auctions, lambda_mid, value_per_conv)
        
        if abs(total_spend - budget) / budget < tolerance:
            return lambda_mid  # 找到目标λ
        
        if total_spend > budget:
            lambda_low = lambda_mid   # 出价太高，提高λ（降低出价）
        else:
            lambda_high = lambda_mid  # 出价太低，降低λ（提高出价）
    
    return (lambda_low + lambda_high) / 2

def simulate_spend(auctions, lambda_val, value_per_conv):
    total_cost = 0
    for auction in auctions:
        bid = auction.expected_value / lambda_val
        if bid >= auction.market_clearing_price:
            # 二价拍卖：花费为第二高出价
            total_cost += auction.market_clearing_price
    return total_cost
```

**实际系统中λ的更新**：

```python
# 离线求解λ（日级别）
lambda_daily = find_lambda_by_binary_search(historical_auctions, daily_budget, ...)

# 在线实时调整λ（时段级别）
# 用PID控制器微调λ，处理实际花费与计划的偏差
class LambdaController:
    def __init__(self, target_lambda):
        self.current_lambda = target_lambda
        self.pid = PIDController(Kp=0.05, Ki=0.001, Kd=0.01)
    
    def update(self, actual_spend_rate, target_spend_rate):
        # 花得太快 → 提高λ（降低出价）
        # 花得太慢 → 降低λ（提高出价）
        spend_error = actual_spend_rate - target_spend_rate
        adjustment = self.pid.step(spend_error)
        
        self.current_lambda *= (1 + adjustment)
        self.current_lambda = max(0.1, self.current_lambda)
        return self.current_lambda
```

### 4.3 PID控制器：实时调节出价乘数

实际系统中通常用**出价乘数k**（而不是直接改λ）做实时调节：

```python
class BidMultiplierPID:
    """
    用PID控制出价乘数k，让实际指标趋近目标
    
    指标可以是：
    - 预算消耗率（消耗速度）
    - CPA（每次转化成本）
    - ROAS（广告回报率）
    """
    
    def __init__(self, target_metric, Kp=0.1, Ki=0.005, Kd=0.02):
        self.target = target_metric
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.k = 1.0          # 出价乘数，初始为1
        self.integral = 0
        self.prev_error = 0
    
    def update(self, actual_metric, dt=1.0):
        error = actual_metric - self.target
        
        self.integral += error * dt
        # 积分限幅，防止积分饱和
        self.integral = max(-10, min(10, self.integral))
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # 计算调整量
        delta_k = -(self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
        
        # 更新出价乘数
        self.k *= (1 + delta_k)
        self.k = max(0.1, min(5.0, self.k))  # 限幅防止极端情况
        
        return self.k
    
    def get_adjusted_bid(self, base_bid):
        return base_bid * self.k

# 使用示例（每分钟调整一次）
pid_cpa = BidMultiplierPID(target_metric=50.0)  # 目标CPA=50元

for minute in range(24*60):
    current_cpa = fetch_realtime_cpa()
    k = pid_cpa.update(current_cpa)
    
    # 所有广告的出价都乘以k
    apply_bid_multiplier(k)
```

### 4.4 RL方法简述

**MDP建模**（详细见 AutoBidding技术演进_从规则到RL.md）：

```
State:  s_t = [剩余预算比例, 时间比例, 当前CPA/目标CPA, 市场竞争强度]
Action: a_t = 出价乘数 k ∈ [0.1, 3.0]
Reward: r_t = 转化价值 - λ × 成本（单步奖励）
Goal:   max E[Σ γ^t r_t]（长期累计奖励最大化）

常用算法：
- PPO（Proximal Policy Optimization）：稳定性好，实践常用
- A3C：异步并行，适合大规模竞价场景
- DDPG：连续动作空间，适合出价乘数连续调节

关键挑战：
- 奖励延迟（转化延迟，不是即时反馈）
- 非平稳环境（竞争对手策略会变化）
- 探索代价高（在线探索可能导致广告主CPA飙升）
```

---

## 五、Autobidding的工程挑战

### 5.1 稀疏转化信号：冷启动

```
问题：
  小广告主：每天转化 < 10次 → 模型无法学习
  新广告：没有历史数据 → 无法初始化λ

解决方案体系：
┌──────────────────────────────────────────────────────────┐
│  方案1：相似广告借鉴                                      │
│  - 基于广告主行业/垂类/历史CVR找相似广告                 │
│  - 迁移其λ值作为初始值                                   │
│  - 风险：相似度不高时初始化误差大                        │
├──────────────────────────────────────────────────────────┤
│  方案2：分层聚合                                          │
│  - 账号级别（更多数据）→ Campaign级别 → AdGroup级别       │
│  - 数据稀少时用更高层级的统计值平滑                      │
├──────────────────────────────────────────────────────────┤
│  方案3：贝叶斯更新                                        │
│  - 以类目先验初始化Beta分布                               │
│  - 每次转化事件更新后验                                   │
│  - 数据越多，先验影响越小                                 │
├──────────────────────────────────────────────────────────┤
│  方案4：多任务学习                                        │
│  - 多个广告主共享底层特征表示                             │
│  - 稀疏广告主从密集广告主的梯度中受益                    │
└──────────────────────────────────────────────────────────┘
```

### 5.2 非平稳环境：大促与竞争变化

```
非平稳性来源：
  1. 大促（双十一/618）：流量激增，竞争加剧，CVR波动
  2. 竞争对手变化：新竞争者入场，出价策略切换
  3. 用户行为漂移：季节性、突发事件（疫情等）
  4. 广告素材生命周期：新素材CTR高，老素材疲劳

应对策略：
┌─────────────────────────────────────────────────────────────┐
│  短期适应（小时级别）                                        │
│  - 加大PID的Ki（加快响应速度）                              │
│  - 缩短λ更新窗口（从每日改为每4小时）                       │
│  - 实时监控市场竞争指数，动态调整出价                       │
├─────────────────────────────────────────────────────────────┤
│  大促特殊处理                                                │
│  - 预先检测大促日期，切换专用大促模型                       │
│  - 预设更保守的λ初始值（大促竞争激烈，预算消耗快）          │
│  - 大促期间放宽CPA目标SLA（告知广告主学习期延长）           │
├─────────────────────────────────────────────────────────────┤
│  竞争对手适应                                                │
│  - 实时监控市场清算价格（第二高出价分布）                   │
│  - 竞争强度指标 = 实际花费 / 历史同期花费                   │
│  - 竞争加剧时提高λ预算系数，防止超支                        │
└─────────────────────────────────────────────────────────────┘
```

```python
# 大促检测 + 自动切换
class PromotionAwareAutoBidding:
    def __init__(self, base_lambda, promo_lambda_multiplier=1.5):
        self.base_lambda = base_lambda
        self.promo_lambda_multiplier = promo_lambda_multiplier
    
    def is_promotion_period(self, timestamp):
        promo_dates = ["11-11", "06-18", "12-12", "01-01"]  # 大促日期
        date_str = datetime.fromtimestamp(timestamp).strftime("%m-%d")
        
        # 大促前3天开始切换
        for promo in promo_dates:
            promo_date = datetime.strptime(promo, "%m-%d")
            if -3 <= (promo_date - current_date).days <= 1:
                return True
        return False
    
    def get_lambda(self, timestamp):
        if self.is_promotion_period(timestamp):
            # 大促期间λ更大 → 出价更保守 → 避免预算提前耗尽
            return self.base_lambda * self.promo_lambda_multiplier
        return self.base_lambda
```

### 5.3 多目标权衡：CPA达标 vs GMV最大化

```
矛盾的两个目标：

从广告主角度：
  目标A：CPA ≤ 50元（每次转化不超过50元）
  目标B：总GMV最大化（在预算内拿到尽量多的转化价值）

矛盾场景：
  流量A：pCVR=10%，订单价值=200元，成本=30元 → CPA=30，ROI=6.67
  流量B：pCVR=5%，订单价值=1000元，成本=45元 → CPA=45，ROI=22.2
  
  纯CPA优化：选A（CPA更低）
  纯GMV优化：选B（价值更高）
  
  如果系统只优化CPA，会错过高价值流量B！

解决框架：
┌─────────────────────────────────────────────────────────────┐
│  方案1：加权多目标                                           │
│  objective = GMV - λ_cpa × max(0, CPA - CPA_target)         │
│  λ_cpa是CPA约束的惩罚权重，需要调参                         │
├─────────────────────────────────────────────────────────────┤
│  方案2：约束满足 + 次级优化                                  │
│  主约束：CPA ≤ target_CPA（硬约束）                         │
│  次级目标：在满足CPA前提下，最大化GMV                        │
│  实现：先用KKT找满足CPA的最优λ，再用λ出价                   │
├─────────────────────────────────────────────────────────────┤
│  方案3：Pareto前沿探索                                       │
│  离线计算CPA-GMV的帕累托前沿                                 │
│  让广告主在前沿上选择偏好点                                   │
│  系统执行对应的出价策略                                       │
└─────────────────────────────────────────────────────────────┘
```

```python
# 多目标出价（约束满足方式）
def multi_objective_bid(
    expected_value,      # 期望转化价值（如订单金额）
    p_convert,           # 转化概率
    target_cpa,          # CPA约束
    budget,              # 预算约束
    lambda_budget,       # 预算约束的λ
    lambda_cpa           # CPA约束的λ（满足时为0，不满足时>0）
):
    """
    KKT条件下的最优出价：
    bid* = (value × p_convert) / (lambda_budget + lambda_cpa / p_convert)
    """
    numerator = expected_value * p_convert
    denominator = lambda_budget + lambda_cpa / max(p_convert, 1e-6)
    
    bid = numerator / denominator
    return max(0, bid)
```

---

## 六、常见考点

### 考点1：Autobidding的对偶优化推导

> **题目**：在预算约束下最大化转化量，请推导最优出价公式，并解释λ的含义。

```
标准解答：

原始问题：
  max  Σ v_i × x_i
  s.t. Σ c_i × x_i ≤ B
       x_i ∈ {0,1}

拉格朗日松弛：
  L = Σ v_i × x_i - λ(Σ c_i × x_i - B)
    = Σ (v_i - λ × c_i) × x_i + λB

最优决策：
  当 v_i - λ × c_i > 0 时，参与竞价（x_i=1）
  否则放弃（x_i=0）

在二价拍卖中，最优出价：
  bid* = v_i / λ

λ的含义：预算的边际价值，即多花1元能多获得多少转化价值
  λ大 → 预算紧张 → 出价低
  λ小 → 预算充裕 → 出价高

求解λ：二分搜索，找使实际花费=预算的λ值
```

### 考点2：Google Smart Bidding的学习期

> **题目**：为什么Smart Bidding有学习期？需要多少转化才能结束学习期？如何帮助广告主度过学习期？

```
学习期原因：
1. 模型冷启动：新Campaign没有CVR历史，需要探索
2. λ校准：对偶变量需要数据才能收敛到真实的最优值
3. 转化样本积累：CVR模型需要足够样本才准确

数据要求（Google官方建议）：
  - tCPA：过去30天至少50次转化
  - tROAS：过去30天至少15次（高价值）转化
  - 低于阈值时，系统进入"学习中"状态

帮助度过学习期的实践：
1. 充足预算：至少是tCPA目标的10-20倍/天
2. 不要频繁修改：每次修改会重启学习
3. 先用宽松目标：初始tCPA设高些，有数据后收紧
4. 扩大受众：更多曝光 → 更快积累转化数据
5. Portfolio Bidding：合并Campaign共享数据池

面试加分：
  提到Google的"Protected Learning Period"——
  学习期内CPA超标，Google会自动补偿
```

### 考点3：PID控制器在Autobidding中的应用

> **题目**：Autobidding中PID控制器控制的是什么？三个参数如何影响出价？

```
PID控制的对象：
  控制变量：出价乘数k（或λ）
  被控量：实际CPA / 实际预算消耗率
  目标值：target_CPA / 计划消耗率

三参数作用：
  P（比例）：
  - 当前CPA偏差的即时响应
  - CPA高了10元 → 立即降价k% 
  - 问题：如果只有P，会产生持续震荡

  I（积分）：
  - 历史偏差的累积响应
  - 如果CPA持续高于目标，I会持续增大
  - 逐渐把k压低，消除稳态误差
  - 问题：积分饱和 → 需要限幅

  D（微分）：
  - 预测CPA变化趋势
  - CPA正在快速下降时，D阻止k继续下降
  - 减少过调（overshoot）
  - 问题：对噪声敏感，实践中常省略D

工程实践：
  实际系统多用PI控制（省略D）
  更新频率：5-15分钟一次（太频繁会震荡，太慢响应不及时）
  限幅：k ∈ [0.1, 3.0]，防止极端情况
```

### 考点4：RL Autobidding的MDP建模

> **题目**：如何用强化学习建模Autobidding？State/Action/Reward如何设计？

```
MDP建模方案：

State设计：
  - 时间维度：剩余预算比例、时间进度（今天过了多少）
  - 性能维度：当前CPA/目标CPA比值、近期赢率
  - 市场维度：当前市场竞争强度、历史同期花费

Action设计：
  - 连续空间：出价乘数k ∈ [0.1, 3.0]（DDPG/SAC）
  - 离散空间：{降价20%, 降价10%, 不变, 涨价10%, 涨价20%}（DQN）

Reward设计（关键！）：
  - 转化价值 - λ_budget × 成本 - λ_cpa × CPA超标惩罚
  - 稀疏奖励问题：转化延迟 → 用即时代理奖励（点击/加购）

训练挑战：
  1. Exploration代价：在线探索会导致广告主CPA飙升
     解决：离线训练（用历史日志的反事实估计）
  2. 非平稳问题：竞争环境每天都在变
     解决：频繁重新训练，或用Meta-RL快速适应

面试加分点：
  提到字节论文中用"虚拟环境"模拟竞价市场
  用离线日志构建simulator，在simulator上训练RL
  避免直接在线探索的风险
```

### 考点5：多目标Autobidding的工程取舍

> **题目**：广告主既想控制CPA ≤ 50元，又想最大化总GMV，这两个目标有时会冲突，系统如何处理？

```
矛盾的本质：
  CPA约束：希望每次转化便宜（倾向数量多但价值低的流量）
  GMV最大化：希望高价值转化（倾向价值高但可能CPA也高的流量）

两种处理框架：

框架1：约束优化（常见于Google tROAS）
  min CPA                # 主目标
  s.t. GMV/Cost ≥ ROAS  # ROAS下界约束
  实现：先保CPA，再在满足CPA的流量中选高价值的

框架2：加权目标（字节等平台）
  objective = α × GMV + β × (-CPA_violation)
  通过α/β权重平衡两个目标
  λ由平台内部根据广告主历史表现调整

实践中的折中：
  设置"弹性CPA"：允许高价值流量的CPA超出目标20%
  引入价值分层：低价值转化严格CPA，高价值转化放宽CPA
  
  示例：
  订单金额 < 100元：CPA严格 ≤ 50元
  订单金额 100-500元：CPA ≤ 80元
  订单金额 > 500元：CPA ≤ 150元（高价值不惜成本）
```

---

## 参考资料

1. **Google Smart Bidding**：Google Ads Help Center，Smart Bidding白皮书
2. **字节AutoBidding RL**：Zhao H et al. "Jointly Optimizing Query Generation and Auto-Bidding Using Reinforcement Learning." CIKM 2022.
3. **对偶优化**：Zhang W et al. "Optimal Real-Time Bidding for Display Advertising." KDD 2014.
4. **Budget Pacing**：Xu J et al. "Smart Pacing for Effective Online Ad Campaign Optimization." KDD 2015.
5. **Portfolio Bidding**：Google Research Blog，"How Smart Bidding Works."
6. **OCPX框架**：字节跳动巨量引擎技术文章

---

> 📌 **核心记忆点**：Autobidding = 广告主设目标 + 平台用 `bid* = value/λ` 出价 + 二分搜索求λ + PID实时调节 + RL长期优化。三大挑战：冷启动/非平稳/多目标。Google靠40+信号+Portfolio，字节靠oCPM统一框架+RL。
