# 深度学习基础面试考点

> 来源：AIGC-Interview-Book 深度学习基础章节
> 更新：2026-03-12

---

## 一、激活函数

### 1.1 常见激活函数对比

| 函数 | 公式 | 范围 | 优点 | 缺点 |
|------|------|------|------|------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | (0,1) | 可解释为概率 | 梯度消失、非零均值 |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | (-1,1) | 零均值 | 梯度消失 |
| ReLU | $\max(0,x)$ | [0,+∞) | 计算快、不饱和 | Dead Neuron（负值梯度为0） |
| Leaky ReLU | $\max(\alpha x, x)$ | (-∞,+∞) | 解决Dead Neuron | 需要选择α |
| ELU | $x$（x>0）; $\alpha(e^x-1)$（x≤0）| - | 负值均值接近0 | 指数计算慢 |
| GELU | $x\cdot\Phi(x)$ | - | 平滑、性能好 | 计算复杂 |
| SiLU/Swish | $x \cdot \sigma(x)$ | - | 自门控、平滑 | 计算开销略高 |
| SwiGLU | $(xW_1)\odot\sigma(xW_2)$ | - | LLM首选 | 参数多（3个矩阵） |

### 1.2 关键面试点

**Q：为什么需要非线性激活函数？**
→ 无激活函数的多层网络等价于单层线性变换，无法拟合复杂非线性关系

**Q：ReLU的Dead Neuron问题？**
→ 当x<0时梯度永远为0，神经元永久失活；解决：Leaky ReLU/ELU/换学习率

**Q：Sigmoid梯度消失原因？**
→ 导数最大值0.25，多层反向传播后梯度趋近于0

**Q：LLM常用哪种激活函数？**
→ SwiGLU（LLaMA/Qwen/DeepSeek）、GELU（BERT/GPT）

**Q：GELU公式是什么？**
→ $\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1+\tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

---

## 二、归一化方法

### 2.1 BN/LN/GN/RMSNorm 对比

| 方法 | 归一化维度 | 适用场景 | 优点 |
|------|-----------|---------|------|
| Batch Norm（BN）| 沿Batch维度 | CNN | 训练稳定，加速收敛 |
| Layer Norm（LN）| 沿Feature维度 | NLP/Transformer | 不依赖Batch大小 |
| Group Norm（GN）| 按Channel分组 | 小Batch CNN | BN退化时的替代 |
| Instance Norm（IN）| 单样本每通道 | 风格迁移 | 实例级归一化 |
| RMS Norm | 均方根归一化 | LLM | 更快（无均值） |

**BN公式：**

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma\hat{x} + \beta
$$

**LN公式：** 对每个样本的所有特征做归一化

**面试核心：**
- BN在训练时用batch统计，推理用运行均值/方差
- LN在NLP中更适合：序列长度不固定，batch小
- RMSNorm去掉均值计算：$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2}$

---

## 三、梯度问题

### 3.1 梯度消失与梯度爆炸

**梯度消失：**
- 原因：Sigmoid/Tanh导数<1，多层连乘后趋近0
- 后果：深层网络参数无法有效更新
- 解决：ReLU系激活函数、残差连接、LayerNorm、梯度裁剪

**梯度爆炸：**
- 原因：权重过大，反向传播时梯度指数增长
- 解决：梯度裁剪（Gradient Clipping）、权重正则化、BatchNorm

**梯度裁剪：**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3.2 Dropout

- 训练时随机将神经元置零（p概率）
- 推理时乘以 (1-p) 缩放（或训练时除以(1-p)保持期望不变）
- 本质：集成多个子网络，防止过拟合
- Transformer中常用：Attention Dropout、FFN Dropout

---

## 四、CNN 核心原理

### 4.1 卷积操作

**输出尺寸计算：**

$$
H_{out} = \lfloor\frac{H_{in} + 2P - K}{S}\rfloor + 1
$$

- H: 高度，P: padding，K: kernel size，S: stride

**参数量计算：**

$$
\text{params} = K \times K \times C_{in} \times C_{out} + C_{out}
$$

（含bias）

**特点：**
- 局部连接：每个神经元只连接局部感受野
- 权值共享：同一卷积核在全图滑动
- 平移不变性

### 4.2 深度可分离卷积

$$
\text{params}}_{\text{{DW}} = K \times K \times C_{in} + C_{in} \times C_{out}
$$

相比普通卷积节省约 $\frac{1}{C_{out}} + \frac{1}{K^2}$ 的参数

---

## 五、Attention 机制

### 5.1 Self-Attention vs Cross-Attention

- **Self-Attention：** Q/K/V来自同一序列
- **Cross-Attention：** Q来自Decoder，K/V来自Encoder输出

### 5.2 时间复杂度

- Self-Attention: $O(n^2 d)$
- LSTM: $O(n d^2)$
- n较小时Attention更好，n很大时是瓶颈

---

## 六、RNN 与 LSTM

### 6.1 RNN 问题

- 长程依赖消失：梯度消失/爆炸
- 无法并行训练

### 6.2 LSTM 门控

| 门 | 公式 | 作用 |
|----|------|------|
| 遗忘门 f | $\sigma(W_f[h_{t-1},x_t])$ | 控制遗忘多少历史 |
| 输入门 i | $\sigma(W_i[h_{t-1},x_t])$ | 控制新信息写入量 |
| 候选 C̃ | $\tanh(W_c[h_{t-1},x_t])$ | 候选记忆 |
| 输出门 o | $\sigma(W_o[h_{t-1},x_t])$ | 控制输出 |

---

## 七、训练技巧

### 7.1 混合精度训练（AMP）

**FP16/BF16 + FP32：**
- 权重存储：FP32（保精度）
- 前向计算：FP16（快）
- 梯度缩放：Loss Scale避免FP16下溢

**BF16 vs FP16：**
- BF16：指数位多（8位），数值范围大（不易溢出），适合LLM
- FP16：尾数位多（10位），精度高，需要Loss Scale

### 7.2 梯度累积

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

效果等同于更大batch，但不增加显存

### 7.3 学习率调度

**Warmup策略：**
- 线性warmup：前N步从0线性增加到目标lr
- 原因：初始参数随机，大lr不稳定；warmup让模型先"热身"

**常见Schedule：**
- Cosine Decay：余弦衰减，平滑
- Linear Decay：线性衰减
- OneCycleLR：warmup + cosine decay组合

### 7.4 优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|---------|
| SGD | 简单，需调lr和momentum | 图像CNN |
| Adam | 自适应lr，收敛快 | NLP/Transformer |
| AdamW | Adam + Weight Decay解耦 | LLM训练主流 |
| LAMB | 大batch训练，自适应lr | 大规模分布式 |

---

## 八、高频面试题

1. **BN和LN的区别，为什么Transformer用LN？**
   → BN依赖batch维度统计，NLP中batch小且序列长度不固定；LN对每个样本独立归一化

2. **梯度消失/爆炸如何解决？**
   → ReLU激活、残差连接、梯度裁剪、BN/LN、合理权重初始化

3. **Dropout在训练和推理时的区别？**
   → 训练时随机置零，推理时关闭（或等比缩放）

4. **为什么卷积共享权重？**
   → 减少参数量（平移不变性假设：不同位置特征检测器相同）

5. **混合精度训练为什么要FP32主权重？**
   → FP16/BF16精度不足以精确表示小的梯度更新，FP32累积更新

6. **学习率warmup的作用？**
   → 初始参数随机，大lr可能导致剧烈震荡；warmup从小lr开始稳定优化方向

7. **Adam相比SGD的优势？**
   → 自适应学习率，收敛更快；对超参不敏感；适合稀疏梯度

8. **梯度累积有什么限制？**
   → BatchNorm统计不准确（每个mini-batch BN统计独立）；LN无此问题
