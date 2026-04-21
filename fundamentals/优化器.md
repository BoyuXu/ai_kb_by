# 优化器演进全景 — 面试向深度总结

> 标签：#optimizer #SGD #Adam #AdamW #LAMB #Lion #搜广推 #面试

> 关联：[[attention_transformer]] | [[lora_peft]] | [[mmoe_multitask]]

---

## 1. 总对比表

| 优化器 | 更新规则概要 | 核心创新 | 最佳场景 | 关键超参 |
|--------|------------|---------|---------|---------|
| SGD | $\theta \leftarrow \theta - \eta \nabla L$ | 最朴素的梯度下降 | 小模型、凸优化 | $\eta$ |
| SGD + Momentum | $v_t = \mu v_{t-1} + \nabla L$ | 动量加速 + 越过局部极小 | CV (ResNet, VGG) | $\eta, \mu=0.9$ |
| Nesterov (NAG) | lookahead gradient at $\theta - \eta \mu v$ | 先走一步再修正方向 | CV 微调 | $\eta, \mu=0.9$ |
| AdaGrad | $\theta \leftarrow \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L$ | 参数级自适应学习率 | 稀疏特征（NLP/CTR） | $\eta=0.01, \epsilon$ |
| RMSProp | 指数移动均值替代累积平方和 | 修复 AdaGrad 学习率单调递减 | RNN、在线学习 | $\eta, \rho=0.9, \epsilon$ |
| Adam | 一阶动量 $m_t$ + 二阶动量 $v_t$ + bias correction | 集大成者：Momentum + RMSProp | Transformer、NLP、搜广推 | $\eta, \beta_1=0.9, \beta_2=0.999, \epsilon$ |
| AdamW | 解耦 weight decay 直接作用于参数 | 修复 Adam + L2 的耦合问题 | LLM 预训练、BERT fine-tune | $\eta, \beta_1, \beta_2, \lambda$ |
| LAMB | 逐层自适应缩放 trust ratio | 大 batch 训练不崩 | BERT 大规模预训练 | $\eta, \beta_1, \beta_2, \lambda$ |
| Lion | $\text{sign}(m_t)$ 更新 | 内存省一半、更新全是 ±1 | LLM、ViT | $\eta, \beta_1=0.9, \beta_2=0.99$ |

---

## 2. 演进链详解

### 2.1 SGD（Stochastic Gradient Descent）

**更新公式：**

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

**核心思想：** 每次用一个 mini-batch 的梯度近似全局梯度，沿负梯度方向更新参数。

**优点：**
- 实现极简，计算开销最小
- 引入的随机噪声有天然正则化效果，倾向收敛到**平坦极小值（flat minima）**
- 泛化性能好（这是 SGD 在 CV 中至今不死的原因）

**缺点：**
- 收敛慢，尤其在高曲率 / 条件数差的 loss surface 上震荡严重
- 所有参数共享同一个学习率，对稀疏特征不友好
- 对学习率极其敏感

**面试一句话：** SGD 简单但泛化好，CV 模型训练的黄金标准。

---

### 2.2 SGD + Momentum

**更新公式：**

$$v_t = \mu v_{t-1} + \nabla L(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

**物理直觉（重要！）：** 想象一个小球在 loss 曲面上滚动。$v_t$ 是速度，$\mu$ 是摩擦系数的倒数。小球会积累历史方向的"惯性"：
- 梯度方向一致 → 速度越来越快（加速）
- 梯度方向震荡 → 正负抵消（阻尼）

**核心改进：**
- 加速了梯度方向一致的维度
- 抑制了梯度方向震荡的维度（典型：窄长山谷中的横向震荡）

**典型配置：** $\mu = 0.9$，相当于最近 10 步梯度的指数加权平均。

**面试一句话：** Momentum 用指数移动平均积累梯度历史，加速一致方向、抑制震荡。

---

### 2.3 Nesterov Accelerated Gradient (NAG)

**更新公式：**

$$v_t = \mu v_{t-1} + \nabla L(\theta_t - \eta \mu v_{t-1})$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

**核心改进：** "先跳一步，再看梯度"。标准 Momentum 是在当前位置算梯度再加速；NAG 是先按动量走到预估位置，在那里算梯度。

**为什么收敛更快？**
- 标准 Momentum：冲过头了才发现该减速 → 来回震荡
- NAG：先看看冲过去之后的梯度，提前"刹车" → 震荡更小
- 理论上凸优化收敛率从 $O(1/t)$ 提升到 $O(1/t^2)$

**面试一句话：** NAG 在动量方向上做 lookahead，提前修正过冲，收敛更稳。

---

### 2.4 AdaGrad

**更新公式：**

$$G_t = G_{t-1} + (\nabla L)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)$$

其中 $G_t$ 是历史梯度平方的**累积和**（对角矩阵，按参数独立）。

**核心创新：** **参数级自适应学习率**。梯度历史大的参数（高频特征）→ 学习率小；梯度历史小的参数（低频/稀疏特征）→ 学习率大。

**优点：** 天然适合稀疏数据，CTR 预估中效果好。

**致命问题：** $G_t$ 只增不减 → 学习率单调递减 → 训练后期学习率趋近于零，提前停止学习。

**面试一句话：** AdaGrad 按参数累积梯度平方做自适应 LR，但后期学习率会衰减到零。

---

### 2.5 RMSProp

**更新公式：**

$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot (\nabla L)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla L(\theta_t)$$

**核心改进：** 用**指数移动平均 (EMA)** 替代 AdaGrad 的累积求和。窗口内的梯度信息有自然衰减，不会单调递减。

**直觉：** $\rho = 0.9$ 相当于只看最近约 10 步的梯度平方均值。

**注意：** RMSProp 从未正式发表论文，是 Hinton 在 Coursera 课程中提出的。

**面试一句话：** RMSProp 用 EMA 修复 AdaGrad 的学习率单调递减问题。

---

### 2.6 Adam（Adaptive Moment Estimation）

**完整更新公式：**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L \quad \text{(一阶动量：梯度的 EMA)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L)^2 \quad \text{(二阶动量：梯度平方的 EMA)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**核心创新：** Momentum（$m_t$）+ RMSProp（$v_t$）+ **bias correction**。

**偏差修正为什么需要？** 初始化 $m_0 = v_0 = 0$，前几步 EMA 严重偏向零。除以 $(1-\beta^t)$ 在训练初期放大估计值，后期 $\beta^t \to 0$ 修正消失。

**$\epsilon$ 的作用（面试常问）：** 防止分母为零，通常 $\epsilon = 10^{-8}$。但在混合精度训练中可能需要调大（如 $10^{-6}$），否则 fp16 下溢。

**优点：**
- 收敛快，对超参不敏感
- 自适应学习率 + 动量，适用范围广

**缺点：**
- 泛化性能可能不如 SGD（sharp minima 问题）
- weight decay 与 L2 正则化耦合（→ 引出 AdamW）

**面试一句话：** Adam = Momentum + RMSProp + 偏差修正，是 Transformer 时代的默认优化器。

---

### 2.7 AdamW（解耦权重衰减）⭐ 面试重点

**这是面试最高频的优化器问题之一。** 必须理解 L2 正则化和 weight decay 的区别。

#### L2 正则化（Adam + L2）

损失函数加 L2 惩罚：$L_{reg} = L + \frac{\lambda}{2}||\theta||^2$

梯度变为：$\nabla L_{reg} = \nabla L + \lambda \theta$

带入 Adam 更新：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(\nabla L + \lambda\theta_t)$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L + \lambda\theta_t)^2$$

**问题：** $\lambda\theta$ 被 Adam 的自适应学习率 $\frac{1}{\sqrt{\hat{v}_t}+\epsilon}$ 缩放了！对于梯度历史大的参数，weight decay 的实际效果被削弱；梯度历史小的参数，weight decay 被放大。**正则化强度与自适应学习率耦合**。

#### AdamW（解耦 weight decay）

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2$$

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

**关键区别：** weight decay 项 $\lambda\theta_t$ 不经过 $m_t, v_t$ 的自适应缩放，**直接从参数中减去**。

#### 数学上为什么不同？

当自适应学习率 $\alpha_t = \frac{1}{\sqrt{\hat{v}_t}+\epsilon} \neq 1$ 时：

- **Adam + L2：** 有效衰减 = $\eta \cdot \alpha_t \cdot \lambda\theta_t$（被 $\alpha_t$ 缩放）
- **AdamW：** 有效衰减 = $\eta \cdot \lambda\theta_t$（固定）

在标准 SGD 中 $\alpha_t = 1$，两者等价。但在 Adam 中 $\alpha_t$ 对每个参数不同，**L2 和 weight decay 不再等价**。

#### 实践影响

| 对比 | Adam + L2 | AdamW |
|------|----------|-------|
| 正则化强度 | 被自适应 LR 耦合，不均匀 | 所有参数同等衰减 |
| 超参调优 | $\lambda$ 和 LR 相互影响 | 解耦，更容易调 |
| 泛化效果 | 较差 | 更好，接近 SGD 的泛化 |
| 使用现状 | 已被淘汰 | BERT、GPT 全部使用 |

**面试一句话：** AdamW 把 weight decay 从梯度中解耦出来，避免被自适应学习率缩放，正则化效果更均匀。

---

### 2.8 LAMB（Layer-wise Adaptive Moments optimizer for Batch training）

**更新公式：**

$$r_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t$$

$$\text{trustRatio} = \frac{||\theta_t||}{||r_t||}$$

$$\theta_{t+1} = \theta_t - \eta \cdot \text{trustRatio} \cdot r_t$$

**核心创新：** 在 AdamW 基础上加了**逐层 trust ratio**。每层的更新幅度按该层参数范数和更新范数的比值缩放。

**为什么需要？** 大 batch 训练（如 batch=65536 训练 BERT）时，不同层的梯度尺度差异巨大。trust ratio 让每层更新的相对幅度保持一致，防止某些层更新过大导致训练崩溃。

**标志性成果：** 将 BERT 预训练从 3 天缩短到 76 分钟（batch 65536，1024 TPU）。

**面试一句话：** LAMB 在 AdamW 上加逐层 trust ratio，解决大 batch 训练中层间梯度尺度不一致问题。

---

### 2.9 Lion（EvoLved Sign Momentum）

**更新公式：**

$$\text{update}_t = \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) \nabla L)$$

$$m_t = \beta_2 m_{t-1} + (1-\beta_2) \nabla L$$

$$\theta_{t+1} = \theta_t - \eta \cdot (\text{update}_t + \lambda \theta_t)$$

**核心创新：**
- 更新量只取 sign（±1），**每个参数更新幅度相同**
- 由 Google Brain 用 AutoML（程序搜索）发现，不是人手工设计
- 不需要存 $v_t$（二阶动量），**内存省约 50%**

**与 Adam 的关键区别：**
- Adam 存 $m_t + v_t$（2 份额外参数） → Lion 只存 $m_t$（1 份）
- Adam 的更新幅度因参数而异 → Lion 全部是 ±$\eta$（更像 SGD 的均匀性）
- LLM 预训练中 Lion 效果与 AdamW 持平，内存优势显著

**面试一句话：** Lion 只用梯度符号更新，内存省一半，是 AutoML 搜出来的优化器。

---

## 3. 关键对比深入

### 3.1 Adam vs SGD：速度 vs 泛化

| 维度 | SGD + Momentum | Adam |
|------|---------------|------|
| 收敛速度 | 慢，需要精调 LR schedule | 快，对 LR 不敏感 |
| 泛化性能 | **更好**（flat minima） | 稍差（sharp minima） |
| 超参敏感度 | 高（LR、momentum、schedule 都要调） | 低（默认参数常够用） |
| loss surface | 倾向平坦极小值 | 可能收敛到尖锐极小值 |

**为什么 Adam 泛化差？** Adam 的自适应学习率让模型更容易收敛到 loss 值低但曲率大的 sharp minima。这类极小值对训练集过拟合，测试集表现差。SGD 的全局学习率+噪声让它更容易跳出 sharp minima 落入 flat minima。

**工业实践：**
- CV 模型（ResNet 等）：SGD + Momentum + cosine LR schedule 仍是主流
- NLP / Transformer：Adam 或 AdamW 是唯一选择（SGD 在 Transformer 上基本训不动）
- 搜广推：Adam / AdamW（模型结构多样，需要鲁棒性）

### 3.2 Adam vs AdamW：什么时候差异明显？

差异在以下情况变大：
1. **模型参数量大**（更多参数需要正则化）
2. **训练时间长**（耦合效应累积）
3. **学习率较大**（自适应缩放系数 $\alpha_t$ 偏离 1 更远）
4. **使用 learning rate warmup**（warmup 期间 $\alpha_t$ 变化剧烈）

实验结论：在 BERT/GPT 级别模型上，AdamW 比 Adam+L2 在验证集上稳定好 0.5-2%。

### 3.3 场景选择总结

```
模型类型 → 推荐优化器
─────────────────────────
CNN / ResNet         → SGD + Momentum + cosine schedule
Transformer / BERT   → AdamW + linear warmup + cosine/linear decay
LLM 预训练 (>1B)    → AdamW (or Lion for memory saving)
LLM 大 batch 训练   → LAMB
搜广推 CTR 模型      → Adam / AdamW（稀疏特征多时考虑 AdaGrad embedding + Adam dense）
GAN                  → Adam（β1=0, β2=0.9 常见配置）
LoRA / PEFT 微调     → AdamW（与 [[lora_peft]] 配合标准做法）
多任务模型           → AdamW（[[mmoe_multitask]] 等多塔结构统一优化）
```

---

## 4. 实战选择指南

| 场景 | 优化器 | 学习率 | Weight Decay | 备注 |
|------|--------|--------|-------------|------|
| ResNet-50 ImageNet | SGD+Momentum | 0.1, cosine decay | 1e-4 | 标准 recipe |
| BERT-base fine-tune | AdamW | 2e-5, linear warmup 10% | 0.01 | HuggingFace 默认 |
| GPT-3 预训练 | AdamW | 6e-4, warmup 375M tokens | 0.1 | OpenAI 论文配置 |
| LLaMA-7B 预训练 | AdamW | 3e-4, cosine to 3e-5 | 0.1 | $\beta_1=0.9, \beta_2=0.95$ |
| DeepFM CTR | Adam | 1e-3 | - | embedding 用 lazy Adam |
| DIN 序列模型 | Adam | 1e-3, exponential decay | 1e-6 | 与 [[attention_transformer]] 中的 target attention 配合 |
| ViT-L/16 | Lion | 3e-4 | 0.1 | 内存友好，效果持平 AdamW |

---

## 5. 面试高频问答

### Q1: Adam 中 $\epsilon$ 的作用？
**A:** 防止分母 $\sqrt{\hat{v}_t}$ 为零导致除零错误。默认 $10^{-8}$。混合精度训练（fp16）中需要调大到 $10^{-6}$ 甚至 $10^{-4}$，否则 fp16 精度不足会导致 $\sqrt{\hat{v}_t} + \epsilon$ 被截断为 $\sqrt{\hat{v}_t}$。

### Q2: 为什么 Adam 要配 warmup？
**A:** Adam 初期 $v_t$（二阶动量）估计不准确，虽然有 bias correction，但方差仍大。如果初始学习率就给满，$\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ 这个比值可能非常大，导致参数更新过猛。warmup 让学习率从小到大渐变，配合 $v_t$ 逐渐稳定，避免训练初期的不稳定。

### Q3: 为什么 Transformer 不用 SGD？
**A:** Transformer 的 loss surface 比 CNN 复杂得多（多头注意力 + 残差 + LayerNorm），条件数差，SGD 在这种 surface 上震荡严重、收敛极慢。Adam 的自适应学习率能自动处理不同参数的尺度差异。参考 [[attention_transformer]] 中的训练细节。

### Q4: AdamW 和 Adam+L2 到底有什么区别？
**A:** 见 §2.7 详细推导。**一句话：** L2 把 $\lambda\theta$ 加进梯度，被 Adam 的自适应 LR 缩放了；AdamW 直接从参数减 $\lambda\theta$，不受自适应 LR 影响。SGD 下两者等价，Adam 下不等价。

### Q5: 搜广推中优化器怎么选？
**A:** CTR 模型通常 embedding 层参数量占 90%+，dense 层占少量。常见做法：
- Embedding 层：AdaGrad（对稀疏特征友好）或 lazy Adam（只更新命中的 embedding）
- Dense 层：Adam / AdamW
- 整体：如果不分层就用 Adam，够用。大模型（如 [[mmoe_multitask]] 多任务架构）建议 AdamW。

### Q6: LAMB 为什么能支撑大 batch？
**A:** 大 batch 下梯度估计更准但步长可能过大。LAMB 的 trust ratio $\frac{||\theta||}{||r||}$ 确保每层参数更新的相对幅度（相对于参数本身大小）是可控的，不会因为某层梯度特别大就把该层参数"带飞"。

### Q7: Lion 省内存的原理？
**A:** Adam 需存 $m_t$（一阶）+ $v_t$（二阶）= 2x 模型参数量的额外内存。Lion 只存 $m_t$，更新时只用 sign，不需要 $v_t$。对于 LLaMA-65B 级别的模型，节省数十 GB 显存。

---

## 6. 学习率与优化器的关系

学习率调度（lr scheduling）与优化器选择紧密相关：

| 优化器 | 典型 LR Schedule | 原因 |
|--------|-----------------|------|
| SGD | step decay / cosine | SGD 需要积极降 LR 才能收敛到好的极小值 |
| Adam | warmup + constant / linear decay | warmup 稳定 $v_t$ 估计，后期适当衰减 |
| AdamW | warmup + cosine decay | LLM 标准配置，warmup 10% steps |
| LAMB | warmup + polynomial decay | 大 batch 需要更长 warmup |

**核心原则：**
- 自适应优化器（Adam 系）自带"隐式 LR 衰减"（$v_t$ 增大 → 有效 LR 下降），因此外部 schedule 可以相对温和
- SGD 没有自适应机制，必须靠外部 schedule 手动降 LR

详细 schedule 策略参见 → `lr_scheduling.md`

---

## 7. 演进路线图

```
SGD
 │
 ├─ + Momentum ─── + Nesterov（动量方向演进）
 │
 └─ AdaGrad ─── RMSProp ─── Adam ─── AdamW ─── LAMB（自适应方向演进）
                                        │
                                        └─── Lion（AutoML 新方向）
```

两条主线：
1. **动量线（SGD → Momentum → NAG）：** 解决收敛速度问题
2. **自适应线（AdaGrad → RMSProp → Adam → AdamW → LAMB）：** 解决参数级学习率适配问题
3. **新范式（Lion）：** 用程序搜索而非人工设计发现优化器

最终在实践中胜出的是 **AdamW**——它集成了两条线的优点（动量 + 自适应 + 正确的正则化），成为 Transformer 时代的事实标准。

---

> 最后更新：2026-04-14 | 关联知识：[[attention_transformer]] [[lora_peft]] [[mmoe_multitask]]
