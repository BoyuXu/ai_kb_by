# 前沿出价方向：多目标/Constrained/Bid Landscape/隐私计算/LLM

> 日期：2026-03-18 | 领域：广告系统前沿 | 作者：MelonEggLearn

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        前沿出价系统全景图                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    广告主目标（多目标）                            │   │
│   │  max GMV  &  CPA ≤ target  &  ROI ≥ target  &  曝光 ≥ min       │   │
│   └──────────────────┬─────────────────────────────────────────────┘   │
│                       │                                                  │
│          ┌────────────┼────────────────────────────┐                    │
│          ▼            ▼                            ▼                    │
│   ┌─────────────┐ ┌──────────────────┐ ┌───────────────────────────┐   │
│   │  多目标联合   │ │ Constrained      │ │   Bid Landscape 预测      │   │
│   │  出价        │ │ Bidding          │ │   w(b) = P(win|bid=b)     │   │
│   │  帕累托前沿  │ │ 拉格朗日/ADMM    │ │   最优出价计算            │   │
│   └─────────────┘ └──────────────────┘ └───────────────────────────┘   │
│                                                                          │
│          ┌──────────────────────┬───────────────────────┐               │
│          ▼                      ▼                        ▼               │
│   ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│   │ Privacy-Safe│    │  LLM辅助出价     │    │   统一出价框架       │   │
│   │ Bidding     │    │  创意/受众/策略  │    │   实时竞价引擎       │   │
│   │ 联邦/差分隐私│    │  <10ms约束       │    │   延迟<10ms          │   │
│   └─────────────┘    └──────────────────┘    └─────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 一、多目标联合出价

### 1.1 业务背景与挑战

现代广告主的诉求已经不再是单一目标，而是同时追求多个维度：

```
典型多目标场景：电商平台广告主
  目标1：最大化GMV（成交金额）      → 越多越好
  目标2：控制CPA（获客成本）        → CPA ≤ ¥50/单
  目标3：维持ROI                   → 每1元广告费带来5元收入
  目标4：保证品牌曝光量             → 日曝光 ≥ 100万次
  
  问题：这四个目标可能相互冲突！
  例如：
    追求最大GMV → 可能需要高出价 → CPA上升 → 违反CPA约束
    严格控制CPA → 可能错过高价值用户 → GMV下降
```

**多目标优化的核心挑战：**
- 目标间存在帕累托权衡（Pareto Trade-off）
- 各目标的量纲不同（元/次/比率）
- 约束和目标需要同时满足

### 1.2 帕累托前沿：理论基础

```
帕累托最优定义：
  解x*是帕累托最优，当且仅当不存在另一个解x使得：
  - 所有目标在x中不劣于在x*中
  - 至少一个目标在x中严格优于在x*中
  
直觉：无法在不损害任何其他目标的前提下，进一步改善某个目标

帕累托前沿（Pareto Front）：所有帕累托最优解的集合

GMV
  │
  │    ×  ×
  │       ×  ×         （帕累托前沿上的点）
  │           ×
  │             ×
  │               ×
  └──────────────────── CPA（越低越好，取负值）
  
广告主的任务：在帕累托前沿上选择符合业务优先级的点
```

### 1.3 实现方式1：加权求和法

```
出价公式：
  bid = α × CTR_value + β × CVR_value
       = α × pCTR × CPC_cap + β × pCVR × Revenue_per_conversion

其中：
  α：点击价值权重
  β：转化价值权重
  pCTR：预测点击率
  pCVR：预测转化率
```

**Python实现：**

```python
def weighted_sum_bid(pctr, pcvr, cpc_cap, revenue_per_conv, alpha=0.3, beta=0.7):
    """
    加权求和出价
    
    Args:
        pctr: 预测点击率 (0~1)
        pcvr: 预测转化率 (0~1)  
        cpc_cap: CPC出价上限
        revenue_per_conv: 每次转化价值
        alpha: 点击权重
        beta: 转化权重（通常alpha+beta=1）
    
    Returns:
        bid: 出价金额
    """
    click_value = pctr * cpc_cap
    conversion_value = pcvr * revenue_per_conv
    
    bid = alpha * click_value + beta * conversion_value
    return bid
```

**局限性：**

```
问题1：权重难以设定
  α和β的最优值因广告主、时段、受众而异
  需要大量人工调参，且缺乏理论依据

问题2：约束无法硬保证
  加权求和只是软约束
  无法保证 CPA ≤ target_CPA（可能时刻违反）

问题3：尺度不统一
  CTR_value和CVR_value量级可能差100倍
  直接加权结果由量级大的目标主导

适用场景：
  - 约束较宽松
  - 权重可以通过历史数据学习（AutoML）
  - 实时性要求极高（计算量最小）
```

### 1.4 实现方式2：约束优化（拉格朗日方法）

**问题形式化：**

```
原始优化问题（Primal Problem）：

  max  E[GMV] = Σᵢ pCVRᵢ × revenueᵢ × xᵢ
  
  s.t. E[CPA] = Σᵢ costᵢ × xᵢ / Σᵢ pCVRᵢ × xᵢ ≤ target_CPA
       Σᵢ costᵢ × xᵢ ≤ Budget
       xᵢ ∈ {0, 1}  （是否竞价赢得第i次机会）

其中 costᵢ 依赖出价策略（竞价机制决定）
```

**拉格朗日松弛：**

```
拉格朗日函数：
L(x, λ) = E[GMV] - λ × (E[CPA] - target_CPA)
         = Σᵢ [pCVRᵢ × revenueᵢ - λ × (costᵢ - target_CPA × pCVRᵢ)] × xᵢ

其中 λ ≥ 0 是拉格朗日乘子（对偶变量）

对偶问题：
  min  g(λ) = max_{x} L(x, λ)
   λ≥0

最优性条件（KKT条件）：
  1. 原可行性：CPA_actual ≤ target_CPA
  2. 对偶可行性：λ ≥ 0
  3. 互补松弛：λ × (CPA_actual - target_CPA) = 0
     → 要么约束未激活（λ=0），要么CPA恰好等于target（λ>0）
```

**最优出价公式推导：**

```
在GSP/VCG拍卖中，对于第i次竞价机会：
  最优出价 b* = pCVRᵢ × (revenueᵢ - λ × target_CPA + λ)
              = pCVRᵢ × revenueᵢ + λ × pCVRᵢ × (1 - target_CPA)

简化为：
  b*(i) = value(i) / (1 + λ)

其中：
  value(i) = pCVRᵢ × revenueᵢ  （不考虑约束时的出价）
  λ：CPA约束对应的拉格朗日乘子

对偶变量λ的物理含义：
  λ = 0：CPA约束不紧，按全额价值出价
  λ > 0：CPA约束激活，需要折扣出价
  λ越大 → 出价折扣越大 → CPA越低（代价：GMV下降）
```

**λ的在线更新（对偶梯度下降）：**

```python
class LagrangianBidder:
    def __init__(self, target_cpa, learning_rate=0.01):
        self.target_cpa = target_cpa
        self.lambda_cpa = 0.0  # 对偶变量初始化为0
        self.lr = learning_rate
    
    def compute_bid(self, pcvr, revenue_per_conv):
        """计算最优出价"""
        value = pcvr * revenue_per_conv
        bid = value / (1 + self.lambda_cpa)
        return max(bid, 0.01)  # 最低出价0.01元
    
    def update_lambda(self, actual_cpa, batch_size=1000):
        """
        每批次结束后更新λ（梯度下降）
        
        梯度：∂g/∂λ = actual_CPA - target_CPA
          actual_CPA > target_CPA → 违反约束 → 增大λ → 降低出价
          actual_CPA < target_CPA → 约束有余量 → 减小λ → 提高出价
        """
        gradient = actual_cpa - self.target_cpa
        self.lambda_cpa = max(0, self.lambda_cpa + self.lr * gradient)
        return self.lambda_cpa
```

### 1.5 实现方式3：多目标进化算法（MOO）

**NSGA-II（Non-dominated Sorting Genetic Algorithm II）：**

```
算法流程：
  1. 初始化种群（多组出价策略参数）
  2. 计算每个策略在所有目标上的表现
  3. 非支配排序：找到帕累托前沿（第一层），再找次前沿（第二层），...
  4. 拥挤度计算：同层中保持多样性（避免聚集）
  5. 选择+交叉+变异 → 下一代
  6. 重复2-5，直到帕累托前沿收敛

输出：一组帕累托最优策略，供广告主按偏好选择

在广告系统中的应用：
  - 离线优化出价策略参数空间
  - 不适合实时出价（计算量太大）
  - 适合定期（每天/每周）更新出价策略
```

---

## 二、Constrained Bidding（约束出价）

### 2.1 定义与核心思想

**Constrained Bidding** 是在多个约束条件的硬性限制下，寻找最优出价策略的问题。

```
与Autobidding的对比：

Autobidding（自动出价）：
  - 约束简单：仅预算上限 + 单一目标（max clicks或max conversions）
  - 约束层次少
  - 实现相对简单

Constrained Bidding（约束出价）：
  - 约束复杂：预算 + CPA + ROI + 频次 + 时段 等多维约束
  - 约束层次嵌套（Account级约束 > Campaign级约束）
  - 约束可能相互冲突（需要优先级处理）
  - 是Autobidding的超集和推广
```

### 2.2 典型约束体系

```
约束类型1：预算约束
  Σᵢ costᵢ ≤ Budget
  - 总预算硬上限
  - 可以拆分为日预算/小时预算

约束类型2：CPA约束
  Σᵢ costᵢ / Σᵢ convᵢ ≤ target_CPA
  - 平均获客成本上限
  - 注意：这是全局平均约束，不是每次转化的约束

约束类型3：ROI约束
  Σᵢ revenueᵢ / Σᵢ costᵢ ≥ target_ROI
  - 广告投入产出比下限
  - 等价于：每元广告费至少带来target_ROI元收入

约束类型4：频次约束
  ∀user u: Σᵢ [userᵢ = u] × showᵢ ≤ freq_cap
  - 每个用户的最大展示次数
  - 防止广告骚扰，维护用户体验

约束类型5：竞争保护约束
  ∀competitor c: share_of_voice(self) ≥ min_sov_vs(c)
  - 相对竞品的曝光份额下限（品牌广告常见）
```

**约束冲突处理：**

```python
class ConstraintPriorityResolver:
    """
    当多个约束相互冲突时，按优先级处理
    """
    
    PRIORITY = {
        'budget': 1,      # 最高优先级（绝对不能超预算）
        'legal': 1,       # 法律合规（等同预算）
        'roi': 2,         # 第二优先级
        'cpa': 3,         # 第三优先级
        'frequency': 4,   # 第四优先级
        'impression': 5   # 最低优先级（尽力而为）
    }
    
    def resolve(self, bid, active_constraints):
        """
        多约束冲突时的出价决策
        
        策略：按优先级逐一检查，使用最保守（最低）的出价
        """
        final_bid = bid
        
        # 按优先级排序约束
        sorted_constraints = sorted(
            active_constraints, 
            key=lambda c: self.PRIORITY[c.type]
        )
        
        for constraint in sorted_constraints:
            constraint_bid = constraint.compute_max_bid(bid)
            final_bid = min(final_bid, constraint_bid)
        
        return max(final_bid, MIN_BID)  # 最低出价保障
```

### 2.3 对偶分解求解方法

对于大规模约束优化，对偶分解（Dual Decomposition）是标准工业方法：

```
原始问题（N次竞价机会）：

  max  Σᵢ vᵢ × xᵢ
  s.t. Σᵢ cᵢ × xᵢ ≤ Budget        (λ₁: 预算约束)
       Σᵢ cᵢ × xᵢ / Σᵢ pᵢ × xᵢ ≤ CPA_target  (λ₂: CPA约束)
       Σᵢ rᵢ × xᵢ / Σᵢ cᵢ × xᵢ ≥ ROI_target  (λ₃: ROI约束)
       xᵢ ∈ {0,1}

对偶分解后，每次竞价独立决策：
  bid(i) = vᵢ / (λ₁ + λ₂ × (cᵢ/pᵢ - CPA_target) + λ₃)

对偶变量更新（次梯度法）：
  λ₁ ← max(0, λ₁ + α₁ × (Σcᵢxᵢ - Budget))
  λ₂ ← max(0, λ₂ + α₂ × (actual_CPA - CPA_target))
  λ₃ ← max(0, λ₃ + α₃ × (ROI_target - actual_ROI))
```

### 2.4 ADMM算法

**ADMM（Alternating Direction Method of Multipliers）** 是处理大规模分布式约束优化的现代方法：

```
ADMM将问题分解为可并行处理的子问题：

增广拉格朗日：
  L_ρ(x, z, y) = f(x) + g(z) + y^T(Ax + Bz - c) + (ρ/2)||Ax + Bz - c||²

交替更新步骤：
  x更新：每个广告计划独立优化（可并行）
  z更新：全局约束协调（串行，但计算量小）
  y更新：对偶变量更新（类似梯度下降）

广告系统应用：
  x对应各Campaign的出价策略（各自独立优化）
  z对应跨Campaign的共享约束（共享预算、总ROI）
  ADMM允许Campaign并行优化，只通过z协调

收敛性质：
  - 凸问题保证收敛
  - 非凸问题（如整数规划）可能局部收敛
  - 通常100-500次迭代即可收敛
  - 工业实践：每5-10分钟运行一次ADMM更新λ
```

---

## 三、Bid Landscape预测

### 3.1 定义与重要性

**Bid Landscape**（竞价景观）回答了一个核心问题：

> **对于给定的竞价机会，如果我出价b，赢得该次展示的概率是多少？**

```
数学定义：
  w(b) = P(win | bid = b, auction_context)
  
  满足：
  - w(0) = 0        （出价0必输）
  - w(∞) = 1        （出价无穷大必赢，理论上）
  - w(b) 单调递增   （出价越高，赢的概率越大）
  
竞价景观的完整形式：
  还需要知道 E[cost | win, bid = b]（赢了要付多少钱）
  
  二价拍卖（GSP）：E[cost|win,b] = E[second_price | second_price < b]
  一价拍卖：E[cost|win,b] = b（直接付出价）
```

**为什么Bid Landscape很重要：**

```
最优出价计算：
  期望收益 = P(win|b) × [value - E(cost|win,b)]
           = w(b) × [v - c(b)]

  argmax_b [w(b) × (v - c(b))]
  
  如果不知道w(b)：只能盲目出价，无法做最优决策
  如果知道w(b)：可以精确计算每个出价水平的期望收益
```

### 3.2 参数化建模方法

**方法1：Log-Logistic分布**

```
假设竞争对手出价服从Log-Logistic分布：
  
  w(b) = P(max_competitor_bid < b) = 1 / (1 + (α/b)^β)

其中：
  α：尺度参数（中位数竞争对手出价）
  β：形状参数（竞价集中度）

特点：
  - 重尾分布，适合竞价金额
  - 参数可以从历史数据MLE估计
  - 计算高效（一个简单公式）

Python实现：
  from scipy.stats import fisk  # Log-Logistic分布

  def fit_log_logistic(historical_winning_prices):
      c, loc, scale = fisk.fit(historical_winning_prices, floc=0)
      return c, scale  # β, α
  
  def win_probability(bid, alpha, beta):
      return 1 / (1 + (alpha / bid) ** beta)
```

**方法2：Gamma分布**

```
w(b) = Γ(k, θ; b) = ∫₀ᵇ f_Gamma(x; k, θ) dx

其中：
  k：形状参数
  θ：尺度参数（均值 = k×θ）

适用场景：
  竞价价格非对称分布、右偏明显时效果好
  常见于搜索广告（长尾词价格差异大）
```

**方法3：非参数化（经验CDF）**

```python
class NonParametricBidLandscape:
    """
    基于历史数据的非参数竞价景观估计
    """
    
    def __init__(self, smoothing_window=10):
        self.historical_clearing_prices = []
        self.smoothing_window = smoothing_window
    
    def update(self, clearing_prices):
        """收集历史成交价格"""
        self.historical_clearing_prices.extend(clearing_prices)
        # 保留最近N天数据（滑动窗口）
        if len(self.historical_clearing_prices) > 100000:
            self.historical_clearing_prices = \
                self.historical_clearing_prices[-100000:]
    
    def win_probability(self, bid):
        """
        P(win | bid = b) = P(clearing_price < b)
        = 历史成交价格中低于b的比例
        """
        prices = np.array(self.historical_clearing_prices)
        return np.mean(prices < bid)
    
    def win_probability_smooth(self, bid):
        """平滑版本（使用核密度估计）"""
        from scipy.stats import gaussian_kde
        if not hasattr(self, '_kde'):
            self._kde = gaussian_kde(self.historical_clearing_prices)
        return float(self._kde.integrate_box_1d(0, bid))
    
    def optimal_bid(self, value):
        """
        计算最优出价（二价拍卖）
        argmax_b [w(b) × (value - E[clearing_price | clearing_price < b])]
        """
        bids = np.linspace(0, value * 2, 1000)
        expected_profits = []
        
        prices = np.array(self.historical_clearing_prices)
        
        for b in bids:
            win_prob = np.mean(prices < b)
            if win_prob > 0:
                expected_cost = np.mean(prices[prices < b])
            else:
                expected_cost = 0
            expected_profit = win_prob * (value - expected_cost)
            expected_profits.append(expected_profit)
        
        optimal_idx = np.argmax(expected_profits)
        return bids[optimal_idx]
```

### 3.3 Bid Landscape的三大应用

**应用1：最优出价计算**

```
给定价值v和竞价景观w(b)，计算最优出价：

二价拍卖（GSP）：
  期望收益(b) = w(b) × (v - E[cost|b])
  
  其中 E[cost|b] = E[second_highest_bid | all < b]
  
  对 b 求导并令导数=0：
  w'(b) × (v - c(b)) - w(b) × c'(b) = 0
  
  解方程得到最优 b*

一价拍卖（First-Price）：
  期望收益(b) = w(b) × (v - b)
  
  最优条件：
  w'(b) × (v - b) = w(b)
  
  对于对数正态w(b)，有解析解：b* = v × β/(β+1)
```

**应用2：Off-Policy RL评估**

```
问题：我想评估"如果使用出价策略π₂替代π₁，效果如何？"
     但不能直接线上A/B测试（风险太高）

方法：重要性采样（IS）+ Bid Landscape

  历史数据：在策略π₁下收集的 {(context_i, bid₁ᵢ, win_i, reward_i)}
  
  评估π₂的期望奖励：
  E_{π₂}[R] ≈ (1/n) Σᵢ [w₂(bid₂ᵢ) / w₁(bid₁ᵢ)] × reward_i × win_i
  
  其中：
  bid₂ᵢ = π₂(context_i)        新策略的出价
  w₂(bid₂ᵢ) / w₁(bid₁ᵢ)     重要性权重
  
  实际应用：字节跳动/快手的离线评估系统大量使用此技术
```

**应用3：Bid Shading（一价市场价格压低）**

```
背景：
  2019年前：数字广告普遍使用二价拍卖（GSP）
  2019年后：Google/OpenX等切换为一价拍卖（First-Price）
  
  一价拍卖的问题：
  - 赢者诅咒（Winner's Curse）：赢了反而亏了
  - 出价应低于真实价值
  
Bid Shading算法：
  
  目标：找到最优出价 b* < v，使期望收益最大
  
  Step1：用Bid Landscape估计 w(b)
  Step2：计算期望收益 E[profit(b)] = w(b) × (v - b)
  Step3：梯度下降或二分搜索找到最优 b*
  
  典型结果：b* ≈ 0.6 ~ 0.9 × v（具体取决于竞争激烈程度）
  
  工业实现：The Trade Desk、AppNexus等DSP的核心竞价逻辑
```

---

## 四、Privacy-Safe Bidding（隐私计算出价）

### 4.1 背景：广告信号的系统性消失

```
时间线：
  2018年：GDPR生效（欧洲），数据使用须明确同意
  2020年：CCPA生效（加州），类似GDPR
  2021年：Apple ATT（App Tracking Transparency）
          → iOS14.5后，用户需主动同意追踪
          → Meta广告信号损失约30-40%（公司公告）
  2022年：Firefox、Safari默认禁用第三方Cookie
  2024年：Chrome计划弃用第三方Cookie（多次延期）
  2025年：Privacy Sandbox进入正式推广阶段
  
影响量化：
  信号损失前：可追踪用户跨站行为，精准定向
  信号损失后：
  - 受众匹配率下降20-40%
  - 归因窗口收窄（延迟 + 聚合）
  - pCTR/pCVR预测精度下降
  → 出价参考信号减少 → 出价更保守 → 平台收入下降
```

### 4.2 差分隐私在出价系统的应用

**差分隐私（Differential Privacy）基础：**

```
定义：
  算法M满足ε-差分隐私，若对任意相邻数据集D, D'（仅差一条记录）：
  P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]  对任意输出集合S
  
  ε越小 → 隐私保护越强（但精度损失越大）
  ε越大 → 精度越好（但隐私保护越弱）
```

**在广告出价中的应用：**

```python
# 场景：用户行为统计（用于更新出价模型）
# 不加差分隐私：直接上报用户的真实行为数据
# 加差分隐私：在数据聚合时添加拉普拉斯噪声

import numpy as np

def dp_aggregate_conversions(user_conversions: dict, epsilon: float = 1.0):
    """
    带差分隐私的转化数据聚合
    
    Args:
        user_conversions: {user_id: conversion_value}
        epsilon: 隐私预算
    
    Returns:
        noisy_total: 加噪后的总转化值
    """
    true_total = sum(user_conversions.values())
    
    # 敏感度：单个用户最大贡献（需要clip）
    max_contribution = 100.0  # 单用户最大转化金额
    
    # 拉普拉斯机制
    sensitivity = max_contribution
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(0, noise_scale)
    
    noisy_total = true_total + noise
    return max(0, noisy_total)  # 保证非负

# Google的FLOC/Topics API实现了差分隐私：
# 用户的兴趣标签（Topic）是k-匿名的（同类用户≥k个）
```

### 4.3 联邦学习CTR预估

```
传统CTR预估的隐私问题：
  广告平台需要用户的跨App行为数据来训练模型
  但跨App数据共享涉及隐私法规
  
联邦学习解决方案：
  
  广告平台（中央服务器）          用户设备（本地）
       │                               │
       │ ① 下发全局模型参数           │
       ├──────────────────────────────▶│
       │                               │ ② 本地训练
       │                               │    使用本地用户数据
       │                               │    计算梯度更新
       │ ③ 上传加密梯度（非原始数据）  │
       │◀──────────────────────────────│
       │ ④ 聚合所有设备的梯度         │
       │    更新全局模型               │
       └───────────────────────────────┘
  
  关键：
  - 原始用户数据永不离开设备
  - 只传输梯度（可进一步加同态加密）
  - 满足差分隐私（添加噪声后再上传）
  
Apple的PPML（Privacy-Preserving Machine Learning）：
  - 用于Siri、触控ID等本地模型训练
  - Safari Intelligent Tracking Prevention
  
Google的FLOC → Topics API：
  - 浏览器本地计算用户兴趣标签（Topics）
  - 只暴露粗粒度兴趣（~350个分类）
  - 不暴露具体URL/搜索词
```

### 4.4 Privacy Sandbox与出价系统适配

```
Google Privacy Sandbox核心API：

1. Topics API
   用途：替代第三方Cookie的兴趣定向
   机制：浏览器本地分析浏览历史 → 分配Top3兴趣标签
   对出价影响：
   - 只有粗粒度兴趣信号（vs 之前的精细行为）
   - pCTR预测精度下降5-15%（估计）
   - 需要重新训练适配Topics的特征工程

2. Protected Audience API（FLEDGE）
   用途：再营销广告（Retargeting）
   机制：
   - 用户行为在浏览器本地记录（兴趣组）
   - 竞价在浏览器本地执行（Worklet）
   - 广告主出价逻辑下载到浏览器本地运行
   
   出价代码示例（在浏览器Worklet中运行）：
   function generateBid(interestGroup, auctionSignals, perBuyerSignals, 
                        trustedBiddingSignals, browserSignals) {
     const base_bid = interestGroup.userBiddingSignals.base_bid;
     const pctr = trustedBiddingSignals.pctr;  // 从出价信任服务器获取
     return {bid: base_bid * pctr, render: interestGroup.ads[0].renderUrl};
   }

3. Attribution Reporting API
   用途：归因（替代像素追踪）
   机制：聚合报告（延迟，添加噪声）
   对出价影响：
   - 归因数据延迟增加（实时 → 延迟数小时）
   - 转化数据有噪声 → pCVR模型训练质量下降
   - 需要适配不精确的转化信号
```

### 4.5 对出价系统的综合影响与应对

```
影响评估：
  信号类型      损失程度    出价精度影响
  用户跨站ID    80-100%    ████████░░
  跨App行为     60-80%     ██████░░░░
  精细兴趣标签  40-60%     ████░░░░░░
  上下文信号    0-10%      █░░░░░░░░░
  
  综合：pCTR/pCVR精度下降10-25%（保守估计）
  出价后果：出价保守化（避免亏损），覆盖率下降

应对策略：
  1. 第一方数据强化（First-Party Data）
     - 建设广告主自有DMP（Data Management Platform）
     - 通过CRM数据做受众匹配（Hashed Email/Phone）
     - 建设会员体系，鼓励用户授权

  2. 上下文广告复兴（Contextual Advertising）
     - 不依赖用户ID，只看页面内容
     - LLM理解页面语义 → 精准内容匹配
     - Google的Keyword Contextual是典型代表

  3. 隐私增强技术（PETs）
     - 安全多方计算（MPC）：多方数据联合分析不泄露
     - 同态加密：在加密数据上计算
     - 可信执行环境（TEE）：如Intel SGX

  4. 模型不确定性建模
     - 信号不足时，使用不确定性感知的出价
     - 高不确定性 → 保守出价（降低风险）
     - Thompson Sampling等贝叶斯方法
```

---

## 五、LLM在出价系统的应用

### 5.1 创意生成：LLM提升CTR

```
传统痛点：
  - 广告主需要手工编写数十个文案版本
  - A/B测试耗时（需要足够的流量和时间）
  - 创意疲劳：同一文案多次展示后CTR下降

LLM解决方案：
  1. 自动文案生成
     Input:  产品信息 + 目标受众 + 平台风格要求
     Output: 50个不同风格的文案变体
     
     Prompt示例：
     "你是一个电商广告文案专家。为以下产品生成10个不同风格的广告标题。
      产品：耳机 | 价格：299元 | 目标用户：18-25岁大学生
      要求：每个标题15字以内，突出性价比，适合抖音平台风格"
     
     LLM输出：
     - "大学生必备！299元封神耳机"
     - "不到300块，音质媲美千元机"
     - "打工人耳机选这个，省钱听好歌"
     ...

  2. 动态创意优化（DCO）
     - 根据用户特征实时生成个性化文案
     - LLM离线预生成用户群组级别的文案
     - 实时从预生成库中检索最匹配的创意
     
     效果：Meta报告LLM生成创意平均CTR提升12-18%

  3. 多模态创意
     - DALL-E/Stable Diffusion生成广告图片
     - 文案+图片联合优化
     - A/B测试自动化（Bandits算法选最优创意）
```

### 5.2 受众理解：自然语言定向

```
传统定向的局限：
  广告主：我想针对"喜欢精酿啤酒、关注手工艺品、有品味的城市白领"投放
  平台：只能选择"年龄25-35、收入水平高、兴趣标签：啤酒、手工"
  → 语义理解差距，定向不准

LLM辅助定向：
  Step1: 广告主用自然语言描述目标受众
  Step2: LLM理解语义，映射到平台标签体系
  Step3: 进一步推断"相似用户"（Lookalike扩展）
  
  示例：
  Input:  "精酿啤酒爱好者，关注生活品质，一线城市，月收入1.5万以上"
  
  LLM Processing:
  - "精酿啤酒" → 兴趣标签：精酿啤酒、手工艺品、高端消费品
  - "生活品质" → 消费水平：高消费，喜欢有机食品、设计师品牌
  - "一线城市" → 地域：北京/上海/广州/深圳
  - "月收入1.5万" → 职业：白领/金领，行业：互联网/金融/咨询
  
  LLM还能推断隐式受众：
  - "可能还喜欢：威士忌、精品咖啡、文艺电影"
  - "可能职业：产品经理、设计师、咨询顾问"
  
  效果：字节跳动ABA（AI-Based Audience）受众扩量，转化率提升15-25%
```

### 5.3 策略生成：LLM作为出价规则生成器

```
场景：帮助广告主自动配置复杂出价策略

传统方式：
  广告主需要手工配置：
  - 分时段出价（8个时段 × N个调整系数）
  - 分地域出价（全国300+城市的系数）
  - 分设备出价（iOS/Android/PC的系数）
  → 参数爆炸，人工无法最优配置

LLM辅助：
  Input：广告主的自然语言需求 + 历史数据分析结果
  
  示例对话：
  广告主："最近我的广告周一到周四效果很好，但周末转化率很低，
           而且北京用户比上海用户贵很多，我不知道怎么调整"
  
  LLM分析后生成出价策略：
  {
    "schedule": {
      "weekday": 1.0,
      "weekend": 0.75  // 周末降价25%
    },
    "geo": {
      "beijing": 0.85,  // 北京成本高，适当降价
      "shanghai": 1.0,
      "default": 1.1   // 其他城市成本低，适当提价
    },
    "rationale": "历史数据显示北京CPA比上海高40%，
                  且周末CVR比工作日低32%，建议差异化出价"
  }

LLM的角色定位：
  不是实时出价引擎（延迟不允许）
  而是"策略顾问"：
  - 离线分析 → 生成出价规则
  - 生成的规则由传统引擎实时执行
  - LLM每天/每周更新一次规则
```

### 5.4 LLM的核心限制

```
最大瓶颈：实时延迟不兼容

广告出价系统的延迟要求：
  RTB（实时竞价）: 全流程 < 100ms
  出价决策本身:   < 10ms
  
LLM推理延迟：
  GPT-4（API调用）: 500ms ~ 3000ms   → 完全不可用
  本地小模型（7B）: 50ms ~ 200ms     → 仍然过慢
  蒸馏模型（BERT-like）: 1-5ms       → 勉强可用
  
解决路径：
  方案A：LLM离线 + 规则在线
    LLM离线生成出价规则表/查找表
    在线查表（<1ms）执行
    适合：分时段、分地域等结构化规则
  
  方案B：LLM蒸馏
    用LLM生成大量训练数据
    训练小模型（BERT/MLP）模拟LLM决策
    小模型在线推理（<5ms）
    适合：受众理解、CTR预测等需要泛化的任务
  
  方案C：异步辅助
    LLM处理非实时任务（创意生成、受众分析）
    传统系统处理实时出价
    两者通过特征工程协作
    适合：大多数实际部署场景
```

---

## 六、面试高频考点

### 考点1：多目标出价中，拉格朗日乘子λ的物理含义是什么？如何在线更新？

**标准答案思路：**

```
λ的物理含义：
  λ是约束条件的"影子价格"（Shadow Price）
  
  具体到CPA约束：
  λ表示"CPA约束每放松1元，目标函数（GMV）能增加多少"
  
  直觉：
  λ = 0：CPA约束非绑定（CPA << target），出价可以更激进
  λ = 5：每放宽1元CPA，可以多赚5元GMV（约束很紧）
  λ = ∞：约束非常紧，任何超出都是重大损失

在线更新方法（对偶梯度下降）：
  λ ← max(0, λ + α × (actual_CPA - target_CPA))
  
  学习率α的选择：
  - 太大：λ振荡不稳定
  - 太小：收敛慢，响应不及时
  - 实践：α ≈ 0.01，每1000次竞价更新一次

收敛证明：
  在凸优化问题中，对偶梯度下降保证收敛到最优对偶解
  强对偶性（Slater条件）保证原始最优 = 对偶最优
```

### 考点2：Bid Landscape如何处理数据稀疏问题？

**标准答案思路：**

```
数据稀疏场景：
  - 新广告主：历史出价数据少
  - 长尾关键词：每天竞价次数极少
  - 特殊时段：节假日竞价环境异常
  
解决方案：
  1. 分层估计（Hierarchical Model）
     广告组级 → 广告计划级 → 账户级 → 行业级（从细到粗）
     数据充足时用细粒度，稀疏时退化到粗粒度
  
  2. 贝叶斯方法
     Prior：行业平均的Bid Landscape分布
     Likelihood：当前广告的历史数据
     Posterior = 两者融合（自动平衡稀疏性）
  
  3. 迁移学习
     用相似广告主/关键词的Landscape初始化
     随着数据积累逐步调整
  
  4. 时间衰减
     最近7天数据权重高，更早数据权重低
     平衡数据量和时效性
```

### 考点3：一价拍卖下如何实现Bid Shading？为什么二价拍卖不需要？

**标准答案思路：**

```
二价拍卖不需要Bid Shading的原因：
  在二价拍卖（GSP/VCG）中，出真实价值是弱占优策略
  无论对手怎么出价，诚实出价的期望收益 ≥ 任何其他策略
  这是Myerson的经典结论（1981年诺贝尔奖相关工作）

一价拍卖需要Bid Shading的原因：
  在一价拍卖中，出真实价值期望利润=0（赢了也没赚）
  最优出价一定低于真实价值
  
  数学：
  期望利润(b) = P(win|b) × (v - b) = w(b) × (v - b)
  最优条件：w'(b) × (v - b) = w(b)
  
  对于均匀分布竞争对手：b* = (v + E[second])/2

工业实现（DSP端）：
  Step1：预先估计w(b)（基于历史数据）
  Step2：对每次竞价，计算b* = argmax [w(b) × (v-b)]
  Step3：出价b*（而非v）
  
  效果：
  平均节省15-25%的广告费（相同赢率下）
  是DSP的核心竞争力之一
```

### 考点4：联邦学习CTR预估有哪些挑战？如何保证模型质量？

**标准答案思路：**

```
核心挑战：
  1. 通信效率
     梯度上传带宽 = 模型大小 × 更新频率
     大模型（BERT-based）梯度GB级别，难以在移动端上传
     解决：梯度压缩（TopK稀疏化、量化）、异步更新
  
  2. 数据异质性（Non-IID）
     不同设备的用户行为分布差异巨大
     全局模型可能对某些用户群体效果差
     解决：FedProx（加正则约束本地更新幅度）、个性化FL
  
  3. 拜占庭容错
     恶意设备可能上传污染梯度
     解决：Byzantine-robust聚合（Krum、Bulyan算法）
  
  4. 隐私保证强度
     梯度本身可能泄露训练数据（梯度逆推攻击）
     解决：添加本地差分隐私（Local DP），梯度加噪
  
  5. 统计不稳定
     每次参与训练的设备不同，梯度估计有偏差
     解决：FedAvg的收敛分析，控制参与设备数量
```

### 考点5：LLM在实时广告系统中最实际可行的应用方式是什么？

**标准答案思路：**

```
核心约束：实时出价 < 10ms，LLM无法直接参与

最可行的三种方式：

① 离线规则生成（最成熟）
   LLM每天运行一次：
   分析前7天数据 → 生成出价调整规则 → 写入规则引擎
   在线：规则查表（<1ms）
   
   已被字节/腾讯/百度采用（2023-2024）

② 创意生成（直接落地）
   LLM离线批量生成广告文案/图片
   存入创意素材库
   在线：根据用户特征检索最匹配的创意（近似最近邻）
   
   Meta、Google已大规模部署

③ 知识蒸馏（中期方向）
   用LLM生成标注数据（用户意图、受众分析）
   训练轻量模型替代LLM
   轻量模型在线推理（1-5ms）
   
   华为、阿里等在研阶段

不可行的方式：
  ❌ LLM实时调用（API）做出价决策（延迟>500ms）
  ❌ 在线LLM做创意选择（延迟不满足RTB要求）
  ❌ 每次竞价都让LLM分析用户意图（算力成本爆炸）
```

---

## 参考资料

1. **Mehta et al. (2007)**《AdWords and Generalized Online Matching》- Google Research（拉格朗日对偶在广告中的应用奠基论文）
2. **Deb et al. (2002)**《A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II》
3. **Zhang et al. (2016)**《Bid Landscape Forecasting in Online Ad Exchange Marketplace》- KDD
4. **McMahan et al. (2017)**《Communication-Efficient Learning of Deep Networks from Decentralized Data》- 联邦学习奠基论文
5. **Google Privacy Sandbox Documentation**《Protected Audience API》
6. **The Trade Desk Research**《Bid Shading in First-Price Auctions》(2019)
7. **Choi et al. (2020)**《A Constrained Optimization Framework for Bidding in Display Advertising》
8. 《ADMM算法在大规模广告出价系统的应用》- 阿里妈妈技术博客
9. **Abadi et al. (2016)**《Deep Learning with Differential Privacy》- Google Brain

---

*MelonEggLearn | 广告系统前沿学习笔记 | 2026-03-18*
