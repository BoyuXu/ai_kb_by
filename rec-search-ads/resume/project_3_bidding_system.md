# 项目3：搜索广告自动竞价系统

## 项目概览

这是我在某搜索广告平台主导的核心商业系统。通过将手工竞价（广告主手动调整出价）转变为自动竞价（设定目标 CPA，平台自动优化），我为广告主创造了显著的 ROI 提升，为平台增加了广告收入。项目周期 16 个月，涉及算法设计、对偶优化、PID 控制、在线学习等多个模块。

---

## 一、问题背景

### 1.1 手工竞价的痛点

搜索广告平台日均 10 亿 + 次竞价请求。之前，广告主需要手工管理关键词出价：

```
广告主的工作流（旧）：
1. 登陆平台，查看关键词列表（100-10000 个）
2. 对每个关键词，设定出价：
   高竞争词（如 "手机"）：出价 5 元 / 次点击
   低竞争词（如 "山寨手机"）：出价 0.5 元 / 次点击
3. 每天/ 每周调整一次
4. 观察转化数据，手工优化

结果：
- 广告主工作繁重，大部分人调不好，ROI 不稳定
- 高热度词容易过度付费（出价过高，CPC 爆炸）
- 冷门词容易错失机会（出价过低，无展示）
- 新广告冷启动难（没有历史数据参考）
```

### 1.2 痛点的数据化表现

```
广告主满意度调查：
├─ "出价很难调" : 62%
├─ "不知道什么是最优出价" : 81%
├─ "浪费钱在过度付费上" : 54%
└─ "错过了低竞争的好机会" : 47%

平台的问题：
├─ CPA 达标率（广告主目标达成） : 65% ← 太低！
├─ 广告主续费率 : 72% ← 因为不赚钱，续费意愿低
└─ 平均账户生命周期 : 4.2 个月 ← 短，需要不断获新
```

### 1.3 商业机遇

如果能提升 **CPA 达标率**，就能：
- 提升广告主满意度和续费率
- 降低广告主流失，提升终身价值
- 增加新广告主愿意尝试平台

目标：设计自动竞价系统，让广告主只需设定"目标 CPA"（目标单位转化成本，如 50 元），平台自动优化出价，达成目标。

---

## 二、核心算法

### 2.1 转化率预估（CVR Modeling）

自动竞价的基础是准确的 CVR（Conversion Rate）预估。但有个难题：

```
观察数据：
  展示 → 点击 → 浏览 → 加购 → 支付（转化）
           ↓
       我们有这个标签（CTR）
           ↓
                   ↓
                我们有这个标签（CVR）
           
问题：CVR 标签只在"被点击"的样本上才存在！
```

这是**样本偏差问题**，与项目1的 ESMM 思路一致。采用 ESMM 多任务学习：

```python
# ESMM 网络结构（简化）
# 输入：广告特征、用户特征、上下文特征
# 输出：CTR 和 pCVR（post-click CVR）

def esmm_loss(y_click, y_conversion, pred_ctr, pred_pcvr):
    """
    ESMM 损失函数
    y_click: 是否点击（展示级别标签）
    y_conversion: 是否转化（仅在点击样本上有）
    """
    # 任务 1：CTR 预估（展示级别）
    ctr_loss = binary_crossentropy(y_click, pred_ctr)
    
    # 任务 2：pCVR 预估（点击级别）
    # 在点击样本上，pCVR ≈ conversion / click
    pcvr_loss = binary_crossentropy(y_conversion, pred_pcvr)
    
    total_loss = 0.5 * ctr_loss + 0.5 * pcvr_loss
    return total_loss
```

### 2.2 延迟转化的处理

还有一个 tricky 的问题：用户不是点击后立即转化的。

```
时间线：
Day 0, 10:00  : 用户点击广告
Day 0, 14:00  : 用户浏览网页
Day 2, 08:00  : 用户再次返回，完成转化

问题：
- 我们的训练数据是"Day 0 晚上"拉取的
- 但"Day 2 的转化"还没有发生
- 所以 Day 0 的模型训练时，看不到这个转化
- 导致 CVR 系统性低估
```

解决：**观测窗口调整**

```python
def prepare_training_data(events_log, observation_window_days=7):
    """
    准备训练数据，考虑延迟转化
    
    observation_window_days: 观察窗口，如果用户在点击后 7 天内转化，算转化
    """
    training_samples = []
    
    for click_event in events_log:
        click_time = click_event['time']
        user_id = click_event['user_id']
        ad_id = click_event['ad_id']
        
        # 查询这个用户在 [click_time, click_time + 7 days] 内是否转化
        conversion = check_conversion_in_window(
            user_id, ad_id,
            start=click_time,
            end=click_time + observation_window_days
        )
        
        sample = {
            'features': extract_features(click_event),
            'label_click': 1,
            'label_conversion': conversion  # 如果发生转化就标 1
        }
        training_samples.append(sample)
    
    return training_samples
```

### 2.3 自动竞价策略：对偶优化

这是项目的**核心创新**。

#### 问题的数学定义

广告主的目标：

$$
\max \text{Conversions} \quad \text{s.t.} \quad CPA \leq \text{target}_{	ext{CPA}}
$$

其中：

$$
CPA = \frac{\text{Total Cost}}{\text{Total Conversions}} = \frac{\sum \text{bid}_{\text{i}}{\sum \text{conversion}_{\text{i}}
$$

这是个非线性的约束优化问题。直接求解很难。

#### 对偶优化（Lagrangian Relaxation）

使用拉格朗日乘数法，转化为无约束问题：

$$
L(\mathbf{bid}, \lambda) = \sum_i \text{conversion}_{\text{i(\text{bid}_i) - \lambda \left( \sum_i \text{bid}_{i} - \text{target}_{\text{CPA}} \sum_i \text{conversion}_{\text{i(\text{bid}_i) \right)
$$

展开：

$$
L = \sum_i [\text{conversion}_{\text{i(\text{bid}_i) \cdot (1 + \lambda \cdot \text{target}_{	ext{CPA}}) - \lambda \cdot \text{bid}_{\text{i]
$$

对 $\text{bid}_i$ 求偏导，在最优点：

$$
\frac{\partial \text{conversion}_{\text{i}}{\partial \text{bid}_{\text{i}} \cdot (1 + \lambda \cdot \text{target}_{	ext{CPA}}) = \lambda
$$

假设转化与出价成对数关系（常见假设）：

$$
\text{conversion}_{i} = \alpha}_{\text{i \cdot \text{log}}(\text{bid}_{\text{i)
$$

则：

$$
\frac{\alpha}_{\text{i}}{\text{bid}_{\text{i}} \cdot (1 + \lambda \cdot \text{target}_{	ext{CPA}}) = \lambda
$$

解得最优出价：

$$
\text{bid}_{\text{i^* = \frac{\alpha}_{\text{i}}{\lambda} \cdot (1 + \lambda \cdot \text{target}_{	ext{CPA}})
$$

进一步简化，定义 $\beta = \frac{1 + \lambda \cdot \text{target}_{	ext{CPA}}}{\lambda}$，得：

$$
\text{bid}_{\text{i^* = \beta \cdot \text{target}_{\text{CPA}} \cdot \text{pCVR}_{\text{i
$$

**直观理解**：最优出价 = 目标 CPA × 预估转化率 × 调整系数 $\beta$

- 如果某关键词的 pCVR 高（容易转化），出价就高
- 如果某关键词的 pCVR 低（难转化），出价就低
- 系数 $\beta$ 通过 PID 控制实时调整，以满足预算约束

#### 伪代码实现

```python
class AutoBiddingSystem:
    def }}_{\text{}}_{\text{init}}_{\text{}}_{\text{(self, target}}_{\text{cpa, pCVR}}_{\text{model):
        self.target}}_{\text{cpa = target}}_{\text{cpa  # 广告主设定的目标
        self.pcvr}}_{\text{model = pCVR}}_{\text{model
        self.beta = 1.0  # 初始调整系数
    
    def get}}_{\text{optimal}}_{\text{bid(self, keyword):
        """
        根据对偶优化计算最优出价
        """
        # 预估这个关键词的转化率
        pCVR = self.pcvr}}_{\text{model.predict(keyword.features)
        
        # 对偶优化公式
        bid = self.beta * self.target}}_{\text{cpa * pCVR
        
        # 限制出价范围（避免异常高/低）
        bid = np.clip(bid, min}}_{\text{bid=0.1, max}}_{\text{bid=100)
        
        return bid
    
    def update}}_{\text{beta}}_{\text{with}}_{\text{pid(self, actual}}_{\text{cpa, target}}_{\text{cpa):
        """
        PID 控制器：根据实际 CPA 动态调整 β
        使实际 CPA 逐渐逼近目标
        """
        error = actual}}_{\text{cpa - target}}_{\text{cpa
        
        # PID 控制参数
        Kp, Ki, Kd = 0.1, 0.05, 0.02
        
        # P（比例）：误差越大，调整越大
        self.beta += Kp * error
        
        # I（积分）：累积误差
        self.integral}}_{\text{error += error
        self.beta += Ki * self.integral}}_{\text{error
        
        # D（微分）：误差变化率
        self.beta += Kd * (error - self.last}}_{\text{error)
        
        # 保证 β 在合理范围
        self.beta = np.clip(self.beta, 0.5, 2.0)
        
        self.last}}_{\text{error = error
```

### 2.4 冷启动：Thomson Sampling

新广告或新关键词没有转化数据，如何竞价？

使用 **Thompson Sampling**（汤普森采样），在探索和利用之间找平衡：

```python
class ColdStartBidding:
    def }}_{\text{}}_{\text{init}}_{\text{}}_{\text{(self):
        self.prior}}_{\text{alpha = 1  # Beta 分布的先验参数
        self.prior}}_{\text{beta = 9   # 表示先验相信转化率是 1/10
    
    def sample}}_{\text{conversion}}_{\text{rate(self, keyword):
        """
        从 Beta 分布采样转化率
        
        随着观察到更多数据，后验分布越来越尖锐
        """
        # 后验 = 先验 + 观察数据
        posterior}}_{\text{alpha = self.prior}}_{\text{alpha + keyword.num}}_{\text{conversions
        posterior}}_{\text{beta = self.prior}}_{\text{beta + keyword.num}}_{\text{clicks - keyword.num}}_{\text{conversions
        
        # 从 Beta 分布采样
        sampled}}_{\text{rate = np.random.beta(posterior}}_{\text{alpha, posterior}}_{\text{beta)
        
        return sampled}}_{\text{rate
    
    def get}}_{\text{cold}}_{\text{start}}_{\text{bid(self, keyword, target}}_{\text{cpa, beta=1.0):
        """
        冷启动竞价
        """
        # 采样转化率（包含不确定性）
        sampled}}_{\text{crate = self.sample}}_{\text{conversion}}_{\text{rate(keyword)
        
        # 用采样的转化率计算出价
        bid = beta * target}}_{\text{cpa * sampled}}_{\text{crate
        
        # 如果没有数据，用一个保守的默认出价
        if keyword.num}}_{\text{clicks == 0:
            bid = default}}_{\text{bid  # 如 target}}_{\text{cpa / 10
        
        return bid
```

Thompson Sampling 的妙处在于：
- **初期**：不确定性大，采样的转化率波动大，探索多个出价范围
- **后期**：观察到足够数据，采样的转化率集中，自动聚焦到最优出价

---

## 三、多目标平衡

### 3.1 广告主目标 vs 平台目标

```
广告主目标：
  ├─ CPA ≤ 50 元（成本控制）
  ├─ Conversions 最大（转化量）
  └─ Budget 不超支

平台目标：
  ├─ 广告收入最大化
  ├─ 用户体验（不被骚扰）
  └─ 公平性（大小广告主均衡）
```

这些目标有时矛盾：
- 对广告主最优的出价 != 对平台收入最优的出价
- 如果所有广告主都把 CPA 设得很紧，平台流量竞争激烈，eCPM 反而下降

### 3.2 KKT 条件求解帕累托前沿

定义综合目标函数：

$$
F = w}_{\text{1 \cdot \text{Conversions}} + w_2 \cdot \text{Platform Revenue} - w_3 \cdot \text{Unfairness}
$$

其中权重 $w_1, w_2, w_3$ 由产品和商业团队协商。

使用 KKT 条件（Karush-Kuhn-Tucker）求解最优解：

```python
def solve_optimal_allocation():
    """
    求解帕累托前沿上的最优分配
    """
    from scipy.optimize import minimize
    
    def objective(bids):
        # 计算目标函数
        conversions = sum(predict_conversions(bid) for bid in bids)
        revenue = sum(bid * predict_conversions(bid) for bid in bids)
        fairness = compute_fairness_loss(bids)
        
        return -(0.4 * conversions + 0.4 * revenue - 0.2 * fairness)
    
    def constraint_cpa(bids, i):
        # 约束：CPA_i ≤ target_CPA_i
        cost_i = bids[i] * expected_clicks[i]
        conversions_i = predict_conversions(bids[i])
        cpa_i = cost_i / (conversions_i + 1e-6)
        return target_cpa[i] - cpa_i
    
    # 求解
    result = minimize(
        objective,
        x0=initial_bids,
        constraints=[{'type': 'ineq', 'fun': constraint_cpa} for i in range(n)]
    )
    
    return result.x
```

---

## 四、线上实现与运维

### 4.1 延迟要求

RTB 竞价需要在 **<100ms** 内返回出价。

```
竞价流程：
  请求 (0ms)
    ↓ 0-5ms
  特征获取（从 Redis 缓存）
    ↓ 5-20ms
  pCVR 预估（模型推理）
    ↓ 20-40ms
  对偶优化（计算出价）
    ↓ 40-60ms
  PID 更新（调整 β）
    ↓ 60-100ms
  返回出价 (100ms)
```

### 4.2 模型蒸馏

pCVR 预估模型（DeepFM）原本推理延迟 30ms，太慢。

使用知识蒸馏压到 15ms：

```python
# Teacher: DeepFM（精度高，但慢）
teacher_model = load_deepfm_model()

# Student: Shallow MLP（快）
student_model = ShallowMLP(hidden_dims=[128, 64, 1])

# 蒸馏
for batch in training_data:
    teacher_pred = teacher_model(batch.features)
    student_pred = student_model(batch.features)
    
    # Soft label loss（学 Teacher 的概率分布）
    soft_loss = KL_divergence(teacher_pred / T, student_pred / T)
    
    # Hard label loss（学真实标签）
    hard_loss = cross_entropy(batch.labels, student_pred)
    
    loss = 0.3 * hard_loss + 0.7 * soft_loss
    optimizer.step(loss)
```

### 4.3 降级策略

如果 pCVR 模型服务故障，系统需要有备选方案：

```python
def get_bid_with_fallback(keyword, target_cpa):
    try:
        # 尝试调用 pCVR 模型
        pcvr = pcvr_service.predict(keyword.features, timeout=10)
        bid = compute_optimal_bid(pcvr, target_cpa)
    except Exception as e:
        # 服务故障或超时，使用降级策略
        if e.status == 'timeout':
            # 用简单的启发式方法
            bid = heuristic_fallback_bid(keyword, target_cpa)
        elif e.status == 'service_error':
            # 用昨天的竞价（保持稳定）
            bid = yesterday_bid[keyword.id]
        else:
            # 其他错误，使用保守出价
            bid = 0.5 * target_cpa
    
    return bid
```

---

## 五、AB 测试与效果

### 5.1 测试设计

```
对照组（手工竞价）：40% 广告主
实验组（自动竞价）：40% 广告主
保留（手工竞价+对照）：20%（防止样本污染）

持续时间：8 周
样本量：10000 个广告主，500 万个关键词
```

### 5.2 主要指标

| 指标 | 对照 | 实验 | 差异 | 显著性 |
|------|------|------|------|--------|
| CPA 达标率 | 65% | 92% | **+27 pp** | <0.001 |
| 广告主转化量 | 100 | 118 | **+18%** | <0.001 |
| 广告主成本 | 100 | 88 | **-12%** | <0.001 |
| 平台 eCPM | 2.5 元 | 2.52 元 | +0.8% | <0.05 |
| **平台收入** | - | - | **+7%** | <0.001 |
| 广告主满意度 | 6.2/10 | 8.1/10 | **+1.9** | <0.001 |

最关键的是：
- **CPA 达标率 +27 pp**（从 65% → 92%），广告主目标达成率大幅提升
- **广告主成本降低 12%**（避免过度付费），但转化量还增加 18%（因为竞价更精准）
- **平台收入 +7%**（虽然 eCPM 微增，但流量增加更多）

### 5.3 分层分析

不同规模的广告主效果差异：

```python
# 按消费金额分层
small_spenders (< 1000 元/月):
  CPA 达标率提升：65% → 88%（+23 pp）
  成本：-8%（空间有限）
  
medium_spenders (1000-10000 元/月):
  CPA 达标率提升：62% → 94%（+32 pp）⭐
  成本：-13%（竞价优化空间大）
  
large_spenders (> 10000 元/月):
  CPA 达标率提升：72% → 92%（+20 pp）
  成本：-14%（量级大，优化收益高）
```

发现：**中等规模广告主的收益最大**。这是因为：
- 小广告主原本就是目标不清（随意出价），自动竞价帮助不大
- 大广告主原本就有 BI 团队手工优化，自动竞价的收益有限
- 中等广告主最有需求（既有预算约束，也无团队优化），收益最大

---

## 六、关键技术洞察

### 6.1 自动竞价的难点不在算法

对偶优化公式很简单：$bid = \beta \times CPA \times pCVR$

真正的难点在于：

1. **pCVR 预估的准确性**：如果 pCVR 错了，竞价就完全偏离
2. **PID 控制的调参**：$K_p, K_i, K_d$ 怎么设置，直接决定收敛速度和稳定性
3. **冷启动的数据稀缺**：新关键词没有转化数据，怎么快速找到最优出价？
4. **工程化的复杂性**：特征延迟、模型更新、故障降级等，都是工程细节

### 6.2 CVR 延迟转化的危害

如果不处理延迟转化，你的模型会：

```
真实场景：用户点击后 3 天转化
你的训练数据：Day 0 18:00 拉取的数据
结果：这个转化在训练集中被标为 "未转化"

危害：
  ├─ CVR 模型系统性低估
  ├─ 竞价出价过低（因为认为转化率低）
  └─ 用户体验差，广告主不满
```

这是我初版系统犯的错误。后来通过 7 天观察窗口改善了，CVR AUC 直接从 0.62 提升到 0.75。

### 6.3 PID 控制的威力

很多工程师一开始想用更复杂的控制理论（MPC、卡尔曼滤波等）。但经验表明，**简单的 PID 控制** 在这个问题上效果最好：

```
为什么 PID 这么有效：

1. P（比例）：误差大 → 快速调整
2. I（积分）：累积误差 → 不会持续偏离
3. D（微分）：误差变化率 → 防止振荡

例子：
  目标 CPA = 50 元
  当前实际 = 52 元（超过目标 2%）
  
  P：52 - 50 = 2，调整 0.1 × 2 = 0.2
  
  如果超过持续 10 小时：
  I：累积误差 = 2 × 10 = 20，调整 0.05 × 20 = 1.0（增大调整）
  
  如果误差在缩小（从 2 → 1）：
  D：变化率 = -1，调整 0.02 × (-1) = -0.02（减小调整，防止过度修正）
```

### 6.4 什么时候要 Online Learning

自动竞价面临的一个问题：竞争对手也在出价，市场价格（eCPC）在变化。静态模型两周后就开始陈旧。

解决：**在线学习**

```python
def update_model_online(new_event):
    """
    每次竞价后，用实际结果更新模型（不等待批量训练）
    """
    # 新的转化数据到来
    # 立即用梯度下降更新模型参数
    loss = compute_loss(new_event, current_model)
    gradient = compute_gradient(loss)
    
    # 小步长更新（防止过度拟合）
    new_model = current_model - learning_rate * gradient
    
    # 用新模型处理下一个请求
    deploy(new_model)
```

在线学习的好处：
- 模型自动适应市场价格变化
- 不需要等待夜间批量训练，实时反应
- 新关键词的数据立即被利用

缺点：
- 需要注意样本偏差（在线数据来自当前模型，不是随机采样）
- 需要更强的监控（防止模型快速衰退）

---

## 七、讲故事要点

### 7.1 30 秒电梯演讲

> "我在搜索广告平台设计了一个自动竞价系统。广告主之前需要手工调整出价，效率低，CPA 达标率只有 65%。我通过 ESMM 多任务学习来准确预估转化率，然后用对偶优化（Lagrangian）计算最优出价，再用 PID 控制器实时调整，让广告主只需设定'目标 CPA'，平台自动优化。结果 CPA 达标率从 65% 提升到 92%，广告主转化量 +18%，平台收入 +7%。"

### 7.2 完整讲述

**问题**：广告主手工竞价很困难。他们需要管理 100-10000 个关键词，对每个设定出价。难点是：高热度词容易过度付费，冷门词容易错过机会，新广告没有参考。结果，CPA（单位转化成本）达标率只有 65%，广告主很不满。

**解决方案**：我设计了自动竞价系统。三个关键部分：
1. **准确的 CVR 预估**：用 ESMM 多任务学习，同时在展示和点击两个层级建模，纠正样本偏差
2. **最优竞价计算**：用对偶优化（Lagrangian），推导出公式 $bid = \beta \times target\_CPA \times pCVR$
3. **动态调整**：用 PID 控制器实时调整系数 $\beta$，使实际 CPA 逐渐逼近目标

**结果**：线上 AB 测试，CPA 达标率从 65% 提升到 92%。虽然平均出价降低（因为竞价更精准），但转化量反而增加 18%。平台收入也增加 7%。

**学到的**：最大的收获是理解了**数学优化在工程中的价值**。对偶优化的公式只有一行，但通过它，我们把一个复杂的人工优化问题转化为一个系统性的算法。后来，我也用这个思路解决了其他多约束的优化问题。

---

## 八、总结

这个项目让我学到：

1. **解决广告主的实际痛点，比讨好平台指标更重要**：虽然平台收入只增 7%，但广告主的 CPA 达标率从 65% → 92% 是个质的飞跃。满意的广告主才会续费和推荐。

2. **数学优化可以替代人工调优**：以前，优化出价靠广告主和数据分析师手工调整。现在，一个对偶优化公式 + PID 控制器，自动完成了这个工作。

3. **冷启动是个系统性的问题**：不是单靠更复杂的模型就能解决。Thompson Sampling 用贝叶斯思想，在有限信息下做出合理决策。

4. **工程化比算法创新更关键**：ESMM、对偶优化都不是我发明的（都来自论文）。但如何在线上环境中正确地实现它们（处理延迟转化、PID 调参、在线学习、故障降级），才是真正的价值。
