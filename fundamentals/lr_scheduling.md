# 学习率策略 — 面试向深度总结

> 标签：#learning-rate #warmup #cosine-annealing #OneCycleLR #搜广推 #面试

> 关联：[[attention_transformer]]、[[lora_peft]]

---

## 🆚 总对比表：所有学习率策略一览

| 策略 | 公式 / 形状 | 关键超参 | 最适场景 | 优势 | 劣势 |
|------|------------|---------|---------|------|------|
| Constant | $\eta_t = \eta_0$ | $\eta_0$ | 小模型快速验证 | 最简单 | 无法适应训练阶段 |
| StepLR | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$ | $\gamma$, step size $T$ | CV 经典训练 | 直觉简单、可控 | 突变点可能导致 loss 抖动 |
| ExponentialLR | $\eta_t = \eta_0 \cdot \gamma^t$ | $\gamma$ (per epoch) | 需要持续衰减 | 平滑连续 | 后期 LR 过小，训练停滞 |
| CosineAnnealingLR | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos\frac{t\pi}{T_{max}})$ | $T_{max}$, $\eta_{min}$ | GPT/BERT 预训练 | 平滑、尾部慢降 | 需准确估计总步数 |
| CosineAnnealingWarmRestarts | 周期性 cosine + 重启 | $T_0$, $T_{mult}$, $\eta_{min}$ | 长训练、逃离局部最优 | 周期重启探索新区域 | 超参多、调参难 |
| OneCycleLR | warmup → peak → cosine decay below $\eta_0$ | $\eta_{max}$, div_factor, pct_start | 超收敛、快速训练 | 训练快 30-50% | 对 $\eta_{max}$ 敏感 |
| ReduceLROnPlateau | metric 停滞时降 LR | patience, factor, threshold | 不确定训练长度 | 自适应、不需预设总步数 | 被动反应、可能降太晚 |
| Linear Decay | $\eta_t = \eta_0 \cdot (1 - t/T)$ | $T$ (总步数) | BERT/GPT fine-tuning | 简单有效 | 尾部 LR 趋近 0 |
| Inverse Sqrt | $\eta_t = \eta_0 / \sqrt{t}$ | $\eta_0$ | 原始 Transformer | 理论优雅 | 衰减速度固定 |
| Polynomial Decay | $\eta_t = (\eta_0 - \eta_{end}) \cdot (1 - t/T)^p + \eta_{end}$ | $p$, $\eta_{end}$ | 灵活控制衰减曲线 | $p$ 可调衰减速度 | 多一个超参 |

---

## 1. 为什么学习率重要

> *"The learning rate is the single most important hyperparameter."* — Andrej Karpathy

### 1.1 学习率过大的后果

- **发散（divergence）**：梯度更新步长超过 loss 曲面的曲率半径，参数在最优点周围来回震荡甚至飞出
- **Loss 震荡**：训练 loss 忽高忽低，无法收敛
- **数值溢出**：FP16 训练中，大 LR 容易导致 gradient overflow → NaN

直觉理解：想象在山谷中下山，步子太大直接迈到对面山坡上。

### 1.2 学习率过小的后果

- **收敛极慢**：需要数倍训练时间才能达到相同 loss
- **陷入局部最优 / 鞍点**：步长太小无法翻越 loss landscape 中的 barrier
- **sharp minima**：小 LR 倾向收敛到 sharp minima，泛化性能差

### 1.3 学习率与其他超参的耦合

| 超参 | 与 LR 的关系 |
|------|-------------|
| Batch Size | 线性缩放规则：$\eta \propto B$（Goyal et al., 2017） |
| Weight Decay | AdamW 中 WD 与 LR 解耦，但实际影响仍然耦合 |
| Warmup Steps | LR 越大，需要越多 warmup steps |
| Optimizer | Adam 对 LR 更鲁棒（自适应），SGD 对 LR 极敏感 |
| Model Size | 更大模型通常需要更小的 LR（GPT-3: 6e-5 vs GPT-2: 6e-4） |

---

## 2. Warmup 详解

### 2.1 为什么需要 Warmup

三个核心原因：

**原因 1：Adam 二阶矩估计初期不准**

Adam 的更新公式：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

训练初期 $t$ 很小时，$v_t$ 的估计基于极少样本，bias correction 虽然数学上无偏，但方差极大。此时若 LR 很大，更新方向不稳定 → 参数被带偏。

**原因 2：Fine-tuning 中保护预训练权重**

预训练模型（如 BERT、GPT）的权重已经在一个好的 loss basin 中。训练初期如果 LR 过大，会破坏（catastrophic forgetting）预训练学到的表示。Warmup 让模型先「小心探索」再「大步优化」。

这也是为什么 [[lora_peft]] 中 LoRA fine-tuning 需要的 warmup 相对较少 —— LoRA 只更新低秩增量，对预训练权重的干扰天然更小。

**原因 3：BatchNorm / LayerNorm 统计量初期不稳定**

BN 的 running mean/var 在前几个 batch 波动很大。大 LR + 不准的归一化 = 训练崩溃。

### 2.2 Linear Warmup

$$\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}$$

- 最常用，实现简单
- 从 0（或极小值）线性增长到 $\eta_{max}$

### 2.3 Cosine Warmup

$$\eta_t = \eta_{max} \cdot \frac{1}{2}\left(1 - \cos\left(\frac{t}{T_{warmup}} \cdot \pi\right)\right)$$

- 比 linear 更平滑，初期增长更慢、末期增长更快
- 适合对 LR 变化敏感的大模型

### 2.4 Warmup 步数经验值

| 场景 | Warmup 占比 | 典型值 |
|------|------------|--------|
| 大模型预训练（GPT/LLaMA） | 0.1% - 1% | 2000 steps |
| BERT 预训练 | 1% - 5% | 10k steps / 1M total |
| Fine-tuning（BERT/RoBERTa） | 6% - 10% | 500-1000 steps |
| LoRA fine-tuning | 3% - 5% | 更少即可 |
| 搜广推 CTR 模型 | 0% - 2% | 常不用 warmup |

---

## 3. 衰减策略详解

### 3.1 StepLR

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{step} \rfloor}$$

- 每 $T_{step}$ 个 epoch 将 LR 乘以 $\gamma$（通常 $\gamma = 0.1$）
- 经典用法：ResNet 训练在 epoch 30, 60, 90 分别降 10x
- **优点**：简单直觉，手动控制降 LR 时机
- **缺点**：LR 突变可能导致 loss spike；需要人工选择 milestone

```python
# PyTorch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 或 MultiStepLR
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
```

### 3.2 ExponentialLR

$$\eta_t = \eta_0 \cdot \gamma^t$$

- 每个 epoch 乘以 $\gamma$（通常 0.95-0.99）
- 比 StepLR 更平滑，但后期 LR 指数衰减到极小值
- 实际使用较少，多被 CosineAnnealing 取代

### 3.3 CosineAnnealingLR

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}} \cdot \pi\right)\right)$$

- 从 $\eta_{max}$ 平滑降到 $\eta_{min}$，形状像半个 cosine
- **核心优势**：尾部衰减慢 → 模型在低 LR 阶段有足够时间精细调整
- **GPT 系列标配**：GPT-2/3/4、LLaMA 均使用 cosine decay
- $\eta_{min}$ 通常设为 $0.1 \times \eta_{max}$，不完全降到 0

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=max_lr * 0.1
)
```

### 3.4 CosineAnnealingWarmRestarts（SGDR）

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i} \cdot \pi\right)\right)$$

- 每个周期 $T_i$ 结束后 LR 重启到 $\eta_{max}$
- $T_{mult} > 1$ 时每个周期逐渐变长（如 10, 20, 40 epochs）
- **核心思想**：warm restart 帮助逃离局部最优，探索 loss landscape 的不同 basin
- 适合长训练、数据分布复杂的场景

### 3.5 OneCycleLR — 超收敛

核心思想（Leslie Smith, 2018）：

$$
\text{Phase 1 (warmup):} \quad \eta_0 / \text{divFactor} \to \eta_{max} \\
\text{Phase 2 (decay):} \quad \eta_{max} \to \eta_0 / (\text{divFactor} \times \text{finalDivFactor})
$$

- Phase 1 占 `pct_start`（通常 30%）的训练步数
- Phase 2 用 cosine decay 降到比初始 LR 还低的值
- **关键发现**：高 LR 阶段起正则化作用（类似大 noise SGD），final phase 的极低 LR 帮助收敛到 flat minima

**超收敛（Super-Convergence）**：在某些场景下训练速度提升 5-10x。

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=total_steps,
    pct_start=0.3, div_factor=25, final_div_factor=1e4
)
```

### 3.6 ReduceLROnPlateau

- 监控 validation loss/metric，当连续 `patience` 个 epoch 无改善时降 LR
- $\eta_{new} = \eta_{old} \times \text{factor}$（通常 factor=0.1-0.5）
- **优点**：自适应，不需要预设总训练步数
- **缺点**：被动响应，可能在 plateau 上浪费了 patience 个 epoch

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

### 3.7 Inverse Square Root Decay

$$\eta_t = \eta_0 \cdot \frac{1}{\sqrt{\max(t, T_{warmup})}}$$

- 原始 [[attention_transformer]] 论文（Vaswani et al., 2017）使用的策略
- 完整公式：$\eta_t = d_{model}^{-0.5} \cdot \min(t^{-0.5}, t \cdot T_{warmup}^{-1.5})$
- warmup 阶段线性增长，之后按 $1/\sqrt{t}$ 衰减
- 现已被 cosine decay 取代，但面试中经常考

---

## 4. 策略对比矩阵

| 策略 | 训练曲线形状 | 需要预设总步数 | 自适应性 | 实现复杂度 | 推荐场景 |
|------|-------------|--------------|---------|-----------|---------|
| StepLR | 阶梯下降 | 否（手动设 milestone） | 无 | ⭐ | CV 经典模型 |
| CosineAnnealing | 平滑半余弦 | 是 | 无 | ⭐⭐ | 预训练、通用首选 |
| OneCycleLR | 先升后降（超越初始） | 是 | 无 | ⭐⭐ | 追求训练速度 |
| WarmRestarts | 锯齿 cosine | 否（周期自动） | 部分 | ⭐⭐⭐ | 长训练 |
| ReduceOnPlateau | 不规则阶梯 | 否 | 强 | ⭐ | 不确定收敛点 |
| Linear Warmup+Decay | 三角形 | 是 | 无 | ⭐ | BERT fine-tuning |
| Inverse Sqrt | warmup + 幂律衰减 | 否 | 无 | ⭐ | 原始 Transformer |

---

## 5. 实战选择指南

### 5.1 BERT / RoBERTa Fine-tuning

```
策略：Linear Warmup (10% steps) + Linear Decay
LR: 2e-5 ~ 5e-5
Warmup: 前 6-10% 步数
Optimizer: AdamW (weight_decay=0.01)
Batch: 16-32
Epochs: 3-5
```

要点：fine-tuning 的 LR 比预训练小 10-100x，因为只需微调不需大幅改变参数。

### 5.2 GPT / LLaMA 预训练

```
策略：Linear Warmup + Cosine Decay
LR: 6e-4 (GPT-2) / 3e-4 (GPT-3) / 1.5e-4 (LLaMA-65B)
min_lr: 0.1 × max_lr
Warmup: 2000 steps
Optimizer: AdamW (β1=0.9, β2=0.95, wd=0.1)
```

模型越大 LR 越小：这是因为大模型的梯度方差更大，需要更小的步长保持稳定。

### 5.3 搜广推 CTR 模型

```
策略：Constant LR 或 StepLR（偏保守）
LR: 1e-3 ~ 1e-4
Warmup: 通常不用或极短
Optimizer: Adam / Adagrad（sparse 特征用 Adagrad）
特点：在线学习场景常用固定 LR
```

搜广推模型训练的特殊性：
- 数据量极大（数十亿样本），不需要多 epoch
- 特征稀疏（embedding 为主），Adagrad 天然自适应
- 在线学习要求 LR 不能降太低，否则无法追踪数据分布漂移

### 5.4 LoRA Fine-tuning

```
策略：Linear/Cosine Warmup + Cosine Decay
LR: 1e-4 ~ 2e-5（比全参数 fine-tuning 可稍大）
Warmup: 3-5% 步数（比全参数少）
原因：LoRA 只更新低秩增量 ΔW=BA，对原始权重干扰小
```

详见 [[lora_peft]] 中关于学习率与 rank 的关系：rank 越大，LR 应越小。

### 5.5 大 Batch 训练

**线性缩放规则（Linear Scaling Rule）**：

$$\eta_{new} = \eta_{base} \times \frac{B_{new}}{B_{base}}$$

- Goyal et al. (2017)：当 batch size 从 256 扩大到 8192 时，LR 也线性放大 32x
- **但需要更长的 warmup**：大 batch + 大 LR 初期更不稳定
- LAMB optimizer（Layer-wise Adaptive Moments）：专为大 batch 设计，per-layer 自适应 LR

| Batch Size | LR | Warmup |
|-----------|-----|--------|
| 256 | 0.1 | 无 |
| 8192 | 3.2 | 5 epochs |
| 32768 | 12.8 | 5 epochs + LAMB |

---

## 6. Warmup + Decay 组合策略

实际训练中几乎不会单独使用 warmup 或 decay，而是组合：

| 组合 | 应用 | 代表工作 |
|------|------|---------|
| Linear Warmup + Linear Decay | BERT fine-tuning | Devlin et al. (2019) |
| Linear Warmup + Cosine Decay | GPT 预训练 | Radford et al. (2019), LLaMA |
| Linear Warmup + Inverse Sqrt | 原始 Transformer | Vaswani et al. (2017) |
| Cosine Warmup + Cosine Decay | 大模型预训练 | PaLM, Chinchilla |
| Linear Warmup + Step Decay | CV 训练 | ResNet + warmup |

**PyTorch 实现组合策略**：

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup = LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
```

---

## 7. 面试高频问答

### Q1: 为什么 Transformer 需要 Warmup？

**一句话**：Adam 二阶矩初期估计不准 + 大模型初始梯度方差大，warmup 防止训练初期参数被「带偏」。

**展开**：Adam 的 $v_t$ 在前几步只基于 1-2 个 mini-batch 的梯度平方，估计极不可靠。虽然有 bias correction $\hat{v}_t = v_t / (1-\beta_2^t)$，但方差仍然很大。此时大 LR 会导致参数剧烈震荡。Warmup 从小 LR 开始，等 $v_t$ 稳定后再放大。

### Q2: Cosine Decay 比 Linear Decay 好在哪？

**一句话**：Cosine 尾部衰减更慢，给模型更多时间在低 LR 下精细调整到 flat minima。

**展开**：Linear Decay 在训练后半段 LR 急剧下降，可能导致模型还没充分收敛 LR 就已经很小了。Cosine 在中段 LR 较高（维持探索能力），尾部缓慢下降（精细收敛），实证效果更好。

### Q3: OneCycleLR 为什么能实现超收敛？

**一句话**：高 LR 阶段提供隐式正则化（类似大噪声 SGD），帮模型跳出 sharp minima 进入 flat minima。

**展开**：Leslie Smith 发现，训练过程中 LR 先升后降相比单调衰减能更快收敛。高 LR 阶段的大梯度噪声起到正则化效果，类似于增大 batch noise。最终阶段的极低 LR 帮助模型精确收敛。

### Q4: 搜广推模型为什么常用固定 LR？

**一句话**：数据量极大且分布持续变化，固定 LR 保持模型对新数据的适应能力。

**展开**：CTR 模型通常训练一个 epoch 甚至用 online learning。数据分布随时间漂移（用户行为变化），如果 LR 衰减太低，模型无法追踪分布变化。此外，稀疏特征用 Adagrad 天然提供 per-feature 自适应 LR。

### Q5: LR 与 Batch Size 的关系？

**一句话**：线性缩放规则 $\eta \propto B$，但需要配合更多 warmup。

**展开**：直觉上大 batch 的梯度估计更准（方差更小 $\propto 1/B$），所以可以用更大步长。但 Hoffer et al. (2017) 指出线性缩放有上限，batch 过大时 LR 不能无限放大，需要 LAMB/LARS 等 layer-wise 自适应方法。

### Q6: 常见坑点有哪些？

| 坑 | 后果 | 解决方案 |
|----|------|---------|
| Warmup 太短 | 训练初期 loss spike | 增加到 5-10% |
| Cosine 的 $T_{max}$ 设错 | LR 提前到底或结束时还在高位 | $T_{max}$ = total_training_steps |
| ReduceOnPlateau + warmup 冲突 | Warmup 阶段 metric 上升触发降 LR | Warmup 期间禁用 plateau 检测 |
| 大 batch 未调 LR | 欠拟合 | 线性缩放 + 延长 warmup |
| Fine-tuning 用预训练的 LR | 灾难性遗忘 | Fine-tuning LR 应为预训练的 1/10 到 1/100 |
| 忘记 scheduler.step() | LR 恒定不变 | 检查 LR 曲线是否符合预期 |

### Q7: 如何 debug 学习率问题？

```python
# 打印每个 epoch 的 LR
for epoch in range(num_epochs):
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
    train(...)
    scheduler.step()

# 或用 TensorBoard / WandB 记录 LR 曲线
writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
```

经验法则：**永远画出 LR 曲线**。如果 loss 在某个点突然 spike，先检查那个时刻的 LR。

---

## 8. 面试一句话速记

| 策略 | 面试一句话 |
|------|-----------|
| Warmup | Adam 二阶矩初期不准，小 LR 先稳定再放大 |
| StepLR | 简单粗暴每 N epoch 降 10x，ResNet 标配 |
| CosineAnnealing | 平滑余弦衰减，尾部慢降利于 flat minima，LLM 标配 |
| WarmRestarts | 周期重启 LR 逃离局部最优，适合长训练 |
| OneCycleLR | 先升后降，高 LR 起正则化作用，超收敛 |
| ReduceOnPlateau | 自适应降 LR，不需预设总步数 |
| Inverse Sqrt | 原始 Transformer 用，现已被 cosine 取代 |
| Linear Scaling | 大 batch 成比例放大 LR，配合 LAMB/LARS |

---

*最后更新：2026-04-14*
