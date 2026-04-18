# 多任务损失设计与权重策略

## 1. 多任务损失的基本框架

总损失：
```
L_total = sum_i (w_i * L_i)
```

每个任务的 loss 类型取决于任务性质：
- CTR/CVR（二分类）：Binary Cross-Entropy
- 停留时长（回归）：MSE / Huber Loss / 分桶后做有序回归
- 完播率（0-1 连续值）：BCE 或 MSE
- 多分类任务：Cross-Entropy

核心问题：w_i 怎么设？不同任务的 loss 量级、梯度方向、学习速度都不同。

## 2. 固定权重法

最简单的方案：手动设定各任务权重。

```
L = 1.0 * L_ctr + 0.5 * L_cvr + 0.3 * L_duration
```

缺陷：
- 不同任务 loss 的量级不同，权重设置依赖经验
- 训练过程中各任务的难度动态变化，固定权重无法适应
- 多次实验调参成本高

适用场景：任务少（2-3个）、已有成熟经验、快速上线。

## 3. 不确定性加权 (Uncertainty Weighting / Homoscedastic Uncertainty)

### 3.1 核心思想

基于贝叶斯理论，用同方差不确定性（homoscedastic uncertainty）自动学习各任务权重。不确定性大的任务（更"难"或更"嘈杂"）自动获得较低权重。

### 3.2 数学推导

假设任务 i 的输出服从高斯分布（回归）或拉普拉斯分布（分类），其方差 sigma_i^2 为可学习参数。

最大化似然等价于最小化：
```
L_total = sum_i ( 1/(2 * sigma_i^2) * L_i + log(sigma_i) )
```

实际实现中学习 log(sigma_i^2) 以保证数值稳定：
```
s_i = log(sigma_i^2)    # 可学习参数
L_total = sum_i ( exp(-s_i) * L_i + s_i )
```

### 3.3 直觉理解

- sigma_i 大 → exp(-s_i) 小 → 任务 i 的 loss 权重低 → 不确定性大的任务被降权
- 但 log(sigma_i) 项防止 sigma_i 无限大（否则所有任务权重趋近 0）
- 两项的平衡自动找到最优权重

### 3.4 实现要点

```python
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        # 初始化 log_vars 为 0，对应 sigma^2=1，权重=1
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
```

优点：自动、简单、无需额外前向/反向传播
缺点：假设任务 loss 是高斯/拉普拉斯的，实际可能不满足

## 4. GradNorm

### 4.1 核心思想

动态调整权重使各任务在共享层的梯度范数保持均衡。训练快的任务权重降低，训练慢的任务权重升高。

### 4.2 算法步骤

1. 计算各任务 loss 对共享参数最后一层的梯度范数：
```
G_i = ||grad(w_i * L_i, theta_shared_last)||_2
```

2. 计算平均梯度范数和各任务的逆训练速率：
```
G_avg = mean(G_i)
r_i = L_i(t) / L_i(0)           # 当前 loss / 初始 loss
r_avg = mean(r_i)
target_i = G_avg * (r_i / r_avg)^alpha  # 目标梯度范数
```

3. 最小化梯度范数与目标的差异来更新权重：
```
L_grad = sum_i |G_i - target_i|
w_i ← w_i - lr_w * grad(L_grad, w_i)
```

4. 归一化权重使其均值为 1（可选）

### 4.3 关键参数

alpha（不对称参数）：
- alpha = 0：所有任务梯度范数趋同（不考虑训练速率差异）
- alpha = 1：完全按训练速率比例调整（训练慢的任务被大幅加权）
- 推荐 alpha = 1.5（适度偏向训练慢的任务）

### 4.4 优缺点

优点：动态适应训练过程，不依赖 loss 分布假设
缺点：需要额外反向传播计算梯度范数，增加约 30-50% 训练时间

## 5. PCGrad (Projecting Conflicting Gradients)

### 5.1 核心思想

不调权重，直接在梯度层面解决冲突。当两个任务的梯度方向冲突（余弦相似度 < 0）时，将冲突梯度投影到对方梯度的法平面上。

### 5.2 算法

```
对每对任务 (i, j):
  if cos(g_i, g_j) < 0:  # 梯度冲突
    g_i = g_i - (g_i . g_j / ||g_j||^2) * g_j   # 投影，去除冲突分量

最终梯度 = sum(g_i)  # 无冲突分量的梯度求和
```

### 5.3 直觉

- 冲突时：去掉 g_i 在 g_j 方向上的负分量，保留与 g_j 正交的分量
- 不冲突时：保持原样
- 效果：消除梯度对抗，保留一致的更新方向

### 5.4 实现注意

- 需要分别对每个任务做反向传播，计算各自的梯度
- 任务数 k 时，需要 k 次反向传播
- 投影顺序会影响结果，通常随机打乱顺序

## 6. 其他梯度操纵方法

### 6.1 GradVac (Gradient Vaccine)

- 目标：不仅消除冲突，还要让梯度方向趋于一致
- 方法：维护一个指数移动平均的梯度余弦相似度，当当前 batch 的相似度低于历史均值时做投影
- 比 PCGrad 更保守，减少过度投影

### 6.2 MGDA (Multiple Gradient Descent Algorithm)

- 求解各任务梯度的最小范数凸组合
- 等价于找 Pareto 下降方向
- 数学上保证每步都是 Pareto 改进
- 计算成本高，实际工业中较少直接使用

### 6.3 CAGrad (Conflict-Averse Gradient)

- 找一个与所有任务梯度角度最小的更新方向
- 在 Pareto 最优的约束下最大化平均任务改进

## 7. Pareto 优化框架

### 7.1 基本概念

Pareto 最优（Pareto Optimal）：
- 定义：不存在任何其他解能在不损害某个任务的前提下改善另一个任务
- Pareto 前沿：所有 Pareto 最优解的集合，形成一条权衡曲线

### 7.2 在多任务学习中的应用

- 不同的 loss 权重组合对应 Pareto 前沿上的不同点
- 目标：找到 Pareto 前沿上满足业务需求的那个点
- 实际做法：训练多组权重配置，画出各任务指标的 trade-off 曲线

### 7.3 与业务决策的关系

在 Pareto 前沿上选哪个点取决于业务优先级：
- 电商：可能更重 CVR（收入直接相关）
- 短视频：可能更重完播率 + 互动率（用户留存）
- 广告：需要平衡 CTR（收入）和用户体验（长期留存）

## 8. 任务 Loss 设计实践

### 8.1 CTR + CVR

```
L_ctr = BCE(pred_ctr, label_click)
L_cvr = BCE(pred_cvr, label_convert)
```

注意：CVR 的 label 只在点击样本上有定义，需要 ESMM 等方法处理样本偏差。

### 8.2 停留时长

直接回归问题：
```
L_dur = Huber(pred_duration, actual_duration)
```

分桶有序回归（更稳健）：
- 将停留时长分为 [0-5s, 5-15s, 15-30s, 30-60s, 60s+]
- 用有序回归（Ordinal Regression）预测
- 避免异常值（如用户挂机）对回归 loss 的影响

### 8.3 完播率

```
L_finish = BCE(pred_finish, label_finish)  # 二值化：是否完播
```

或连续值：
```
L_watch = MSE(pred_ratio, actual_watch_ratio)  # 观看比例回归
```

### 8.4 组合 Loss 示例（短视频推荐）

```
L = w1 * L_ctr + w2 * L_finish + w3 * L_like + w4 * L_duration
```

典型初始权重：1.0, 1.0, 0.5, 0.1（根据量级调整后再用 GradNorm 动态调节）

## 9. 面试高频问题

Q: 不确定性加权和 GradNorm 分别在什么层面解决问题？
A: 不确定性加权在 loss 层面——通过学习任务方差自动调整 loss 的加权系数。GradNorm 在梯度层面——直接操纵不同任务在共享参数上的梯度范数使其均衡。

Q: PCGrad 能完全消除负迁移吗？
A: 不能。PCGrad 只处理梯度方向冲突，不解决特征空间竞争的根本问题。需要配合结构设计（如 PLE 的专属 Expert）一起使用。

Q: GradNorm 的 alpha 参数怎么调？
A: alpha 控制对训练速度差异的敏感度。alpha=0 忽略训练速度差异；alpha 越大，越倾向于给训练慢的任务更高权重。推荐从 1.5 开始，根据任务 loss 曲线的收敛情况调整。

Q: 实际工业中最常用哪种权重策略？
A: 不确定性加权最常用（实现简单、效果稳定）。GradNorm 在任务差异大的场景下效果更好但计算成本高。PCGrad 在学术上常用，工业中因多次反向传播开销较少直接使用。

Q: 如何判断多任务 loss 设计是否合理？
A: 三个信号——(1) 各任务 loss 收敛速度是否大致一致；(2) 各任务指标是否出现 seesaw（此消彼长）；(3) 梯度余弦相似度是否持续为负。出现这些问题就需要调整权重策略或模型结构。
