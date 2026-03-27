# 多目标优化与Pareto框架：在冲突目标间寻找平衡

> 📚 参考文献与源文件
> - [广告系统多目标优化](../../ads/synthesis/广告系统多目标优化.md) — 广告收入 vs 用户体验的权衡
> - [推荐系统排序范式演进](../../rec-sys/synthesis/推荐系统排序范式演进.md) — 多任务模型与目标平衡
> - [推荐系统重排与多样性](../../rec-sys/synthesis/推荐系统重排与多样性.md) — 多目标约束优化
> - [生成式排序范式](../../ads/synthesis/生成式排序范式.md) — 端到端多目标优化

> 💡 **本文统一讲解多目标优化的数学框架、算法方案和工业落地**，覆盖广告、推荐、搜索的多目标平衡问题

---

## 第 1 部分：多目标优化的数学框架

### 1.1 问题定义

实际系统中存在**多个相互冲突的目标**：

- **广告系统**：最大化 eCPM ↔ 保证用户体验 ↔ 公平性
- **推荐系统**：最大化 CTR ↔ 保证多样性 ↔ 新用户体验
- **搜索系统**：最大化相关性 ↔ 保证新鲜性 ↔ 多样性

#### 多目标优化问题

$$\begin{align}
\text{maximize} \quad & \{f_1(x), f_2(x), ..., f_K(x)\} \\
\text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, ..., m \\
& h_j(x) = 0, \quad j = 1, ..., n
\end{align}$$

**例**：推荐系统
$$\begin{align}
\text{maximize} \quad & \{\text{CTR}(x), \text{Diversity}(x), \text{Novelty}(x)\} \\
\text{subject to} \quad & \text{Latency}(x) \leq 100ms \\
& \text{ColdStart Ratio}(x) \geq 0.1
\end{align}$$

### 1.2 Pareto 优化的核心概念

#### Pareto 支配

设两个解 $x^a$ 和 $x^b$，若对所有目标 $i$：

$$
f_i(x^a) \geq f_i(x^b)
$$

且至少有一个目标严格不等，则说 $x^a$ **Pareto 支配** $x^b$。

**含义**：$x^a$ 在所有方面都不比 $x^b$ 差，至少在某方面更优。

#### Pareto 前沿

所有 Pareto 非支配解的集合称为 **Pareto 前沿** 或 **Pareto 最优集**。

```
CTR
↑
│     · Pareto Front
│    /│\
│   / │ \
│  /  │  \
│ /   │   · A (dominated)
│/    │
└─────────→ Diversity

关键：Pareto 前沿上的任何解都不能同时改进两个目标
```

### 1.3 多目标优化的三种方法

#### 方法 1：加权求和（最简单、最常用）

$$
\text{maximize} \quad \sum_{i=1}^K w_i \times f_i(x), \quad \sum w_i = 1
$$

**优点**：
- 实现简单（转化为单目标优化）
- 线性、易于理解

**缺点**：
- 无法发现非凸 Pareto 前沿
- 权重选择困难（如何设定 CTR:CVR:Diversity 的比例？）

**应用场景**：
- 权重由业务KPI驱动（如平台方想多赚钱，调高 eCPM 权重）
- 线上系统的标配

#### 方法 2：约束方法（Epsilon-Constraint）

$$\begin{align}
\text{maximize} \quad & f_1(x) \\
\text{subject to} \quad & f_i(x) \geq \epsilon_i, \quad i = 2, ..., K \\
& g_j(x) \leq 0
\end{align}$$

**思想**：一个目标为主，其他目标作为约束。

**例**：
```
最大化 CTR，但要保证：
  - Diversity ≥ 0.5
  - Latency ≤ 100ms
  - ColdStart Ratio ≥ 0.1
```

**优点**：
- 可以发现整个 Pareto 前沿
- 清晰的约束定义

**缺点**：
- 约束参数难以设定
- 有些约束可能不可行

#### 方法 3：Pareto 前沿参数化（高级）

对每组不同的权重 $w$，求解 $\sum w_i f_i(x)$，得到 Pareto 前沿上的一个点。

通过改变 $w$ 的值，扫描整个前沿。

```
权重 w = [0.8, 0.2, 0]  → 点 P1（重视 CTR）
权重 w = [0.5, 0.3, 0.2] → 点 P2（平衡）
权重 w = [0.2, 0.2, 0.6] → 点 P3（重视多样性）
```

**应用**：
- 不同用户群体选择不同权重
  - 高消费用户：更多商业内容（高 eCPM）
  - 普通用户：更多内容多样性

---

## 第 2 部分：工业级实现方案

### 2.1 两层架构：排序 + 重排

实际系统采用**两层管道**处理多目标：

```
┌─────────────────────────────────────┐
│         精排层（Ranking）             │
│  优化：CTR、CVR、品牌安全等           │
│  模型：MMOE、PLE、深度网络            │
│  输出：每个候选的综合得分             │
└────────────┬────────────────────────┘
             ↓ 得分最高的 K 个候选

┌─────────────────────────────────────┐
│         重排层（Reranking）           │
│  优化：多样性、公平性、约束条件        │
│  方法：贪心启发式、线性规划、RL       │
│  输出：最终展示列表                   │
└─────────────────────────────────────┘
```

**分离的好处**：
- 精排聚焦于预测精度（CTR/CVR）
- 重排聚焦于列表级别的优化（多样性/公平性）
- 易于独立优化和 A/B 测试

### 2.2 重排算法对比

#### 2.2.1 贪心算法（快速、常用）

```python
result = []
for position in [1, 2, ..., K]:
    # 从剩余候选中选择满足多样性约束的最高分者
    candidates = {c for c in pool if c not in result}
    # 过滤不满足约束的候选
    valid = [c for c in candidates if check_diversity(c, result)]
    # 选择得分最高的
    best = max(valid, key=lambda c: score[c])
    result.append(best)
```

**复杂度**：O(K × n)，非常快

#### 2.2.2 线性规划（最优、计算贵）

将排序问题建模为整数规划：

$$\begin{align}
\text{maximize} \quad & \sum_{i=1}^n \text{score}_i \times x_i \\
\text{subject to} \quad & \sum_{i=1}^n x_i = K \\
& x_i \in \{0, 1\}
\end{align}$$

加入约束：

$$
\text{diversity}_{\text{constraint}}(x) \geq \text{threshold}
$$

**求解**：通常用贪心近似，精确解 NP-hard

**优点**：理论最优

**缺点**：求解时间长，不适合线上

#### 2.2.3 强化学习（灵活、学习成本高）

```
状态：已生成的列表前 t 个位置
行动：选择下一个候选物品
奖励：r = α × score - β × penalty(diversity)
       - penalty 确保多样性，score 确保精度

目标：学习策略 π，最大化累计奖励
```

**优点**：
- 可以直接优化列表级目标
- 自动平衡多个目标
- 无需手工调参数

**缺点**：
- 需要大量交互数据
- 训练复杂，容易不稳定

### 2.3 多目标加权的动态调整

#### 静态权重的问题

```
w_CTR = 0.5, w_CVR = 0.3, w_Diversity = 0.2

问题：
- 若平台想增加收入，需要提高 w_CTR
- 若用户反馈体验差，需要提高 w_Diversity
- 手工调整，缺乏灵活性
```

#### 动态权重方案

**方案 1：基于 KPI 驱动**

```
if platform_revenue < target:
    w_eCPM = 0.6  (提高）
    w_Diversity = 0.2 (降低)
else:
    w_eCPM = 0.4
    w_Diversity = 0.4
```

**方案 2：不确定性加权（Uncertainty Weighting）**

$$
w_i \propto \frac{1}{\sigma_i^2}
$$

其中 $\sigma_i$ 是任务 i 的不确定性（模型预测方差）。

**直觉**：预测不确定的任务权重降低，预测确定的任务权重提高。

**优点**：自动调整，无需人工干预

#### 方案 3：强化学习权重优化

将权重向量作为 RL 的状态，根据长期平台 KPI 优化：

```
agent_state = {
  platform_revenue: 浮点数
  user_satisfaction: 浮点数
  advertiser_roi: 浮点数
}

agent_action = {
  w_eCPM: 0.3-0.7
  w_Diversity: 0.2-0.5
  w_Fairness: 0.1-0.3
}

reward = f(platform_revenue, user_satisfaction, advertiser_roi)
```

---

## 第 3 部分：Pareto 前沿的应用案例

### 3.1 分用户群体的 Pareto 前沿

```
场景：同一个系统，不同用户偏好不同

高消费用户：
  权重：w_eCPM = 0.6, w_Diversity = 0.2, w_Novelty = 0.2
  → 看更多商业内容，更少新奇内容
  
普通用户：
  权重：w_eCPM = 0.3, w_Diversity = 0.4, w_Novelty = 0.3
  → 平衡商业和自然内容，重视多样性
  
内容消费者：
  权重：w_eCPM = 0.1, w_Diversity = 0.5, w_Novelty = 0.4
  → 很少商业内容，重视多样性和新奇度
```

**实现**：
```python
user_segment = classify_user(user_profile)
weights = pareto_front[user_segment]  # 查表获取权重
score = sum(w * f(x) for w, f in zip(weights, objectives))
```

### 3.2 时间序列的 Pareto 前沿变化

```
早晚高峰（用户使用频繁）：
  w_CTR = 0.6（追求高参与）
  w_Diversity = 0.2（可容忍重复）

低谷时段（用户稀少）：
  w_CTR = 0.3（收入有限）
  w_Diversity = 0.5（重视体验，留住用户）
```

**自动调整**：
```python
hour = get_current_hour()
traffic_density = get_traffic_density(hour)
if traffic_density > threshold:
    weights = high_engagement_weights
else:
    weights = experience_weights
```

---

## 第 4 部分：多目标指标的设计

### 4.1 目标函数的量化

**目标 1：收入相关**

$$
f_{\text{revenue}} = \text{eCPM} = \text{pCTR} \times \text{pCVR} \times \text{bid}
$$

**目标 2：用户体验**

$$
f_{\text{satisfaction}} = \text{CTR} - \text{penalty}(\text{广告density})
$$

**目标 3：多样性**

$$
f_{\text{diversity}} = \frac{1}{K} \sum_{i=1}^K \text{novelty}(\text{item}_i)
$$

其中 novelty 可以是：
- Category 多样性（不能全是同一类）
- Semantic 多样性（embedding 相似度不能太高）

**目标 4：公平性（广告主侧）**

$$
f_{\text{fairness}} = \text{min}(\text{impression}(\text{advertiser}_i)) \quad \forall i
$$

最大化最小的广告主曝光数，保证没有广告主被忽视。

### 4.2 目标间的权衡分析

```
三目标平衡的经典权衡：

        CTR
       /  \
      /    \
     /      \
    ────────── Diversity
    \        /
     \      /
      \    /
     Novelty

任何两个目标改进，第三个会恶化
```

**例**：
- 提高 CTR → 推荐热门内容 → 多样性下降 ❌
- 提高多样性 → 推荐冷门内容 → CTR 下降 ❌
- 同时优化需要找到平衡点

---

## 第 5 部分：领域对比

### 5.1 目标优先级不同

| 领域 | 1st 优先 | 2nd 优先 | 3rd 优先 |
|------|---------|---------|---------|
| **广告** | 平台收入 | 用户体验 | 公平性 |
| **推荐** | 用户满意度 | 多样性 | 新用户体验 |
| **搜索** | 相关性 | 多样性 | 新鲜性 |

### 5.2 约束的强度

- **广告**：约束最严格（必须满足多样性、预算等）
- **推荐**：约束中等（多样性推荐，但可放宽）
- **搜索**：约束最少（相关性为王）

---

## 第 6 部分：常见陷阱与最佳实践

### 6.1 陷阱 1：权重不当导致的冲突

❌ **错误**：
```
w_CTR = 0.8, w_Diversity = 0.2
线上运行后，用户投诉内容单调，虽然 CTR 高
```

✅ **正确**：
```
定期 A/B 测试不同权重组合
监控用户反馈、留存率等长期指标
```

### 6.2 陷阱 2：约束不可行

❌ **错误**：
```
约束：Diversity ≥ 0.9 AND CTR ≥ 历史最高值
结果：无可行解，系统崩溃
```

✅ **正确**：
```
约束：Diversity ≥ 0.5（松一些）
或：Diversity ≥ 0.9 OR CTR ≥ 0.8（灵活组合）
```

### 6.3 最佳实践

1. **两层架构**：分离排序和重排，各自聚焦不同目标
2. **参数化前沿**：预先计算不同权重对应的前沿点
3. **定期审视**：定期检查权重设置是否符合业务目标
4. **A/B 验证**：任何权重调整都要通过 A/B 测试验证
5. **可解释性**：记录权重设置决策，便于审计和调整

---

## 总结表

| 优化方法 | 复杂度 | 效果 | 适用 |
|---------|--------|------|------|
| 加权求和 | 低 | 中 | 线上系统（标配） |
| 约束方法 | 中 | 高 | 实验阶段 |
| Pareto 参数化 | 高 | 高 | 多用户群体 |
| 强化学习 | 很高 | 很高 | 高级优化 |

---

## 知识体系连接

**上游依赖**：
- CTR/CVR 预估
- 排序算法（MMOE、LTR）
- 特征系统

**下游应用**：
- 混排策略（广告+内容）
- 重排与约束优化
- 在线 A/B 测试

**相关 synthesis**：
- [01_CTR_CVR预估统一框架](./01_CTR_CVR预估统一框架.md)
- [02_排序系统演进_多域统一](./02_排序系统演进_多域统一.md)

---

*MelonEggLearn - 多目标优化 Pareto 统一框架*
