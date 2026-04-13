# 激活函数全景 — 面试向深度总结

> Tags: #activation #ReLU #GELU #SwiGLU #sigmoid #搜广推 #面试
>
> Related: [[attention_transformer]] · [[ctr_calibration]] · [[mmoe_multitask]]

---

## 1. 总对比表

| Name | Formula | Derivative | Range | Pros | Cons | Used In |
|------|---------|-----------|-------|------|------|---------|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | $\sigma(1-\sigma)$ | $(0,1)$ | 概率解释，输出有界 | 梯度饱和，非零中心 | 二分类输出层，CTR |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2(x)$ | $(-1,1)$ | 零中心，收敛快于 sigmoid | 仍然饱和 | RNN，早期 MLP |
| ReLU | $\max(0,x)$ | $\begin{cases}1&x>0\\0&x\leq0\end{cases}$ | $[0,+\infty)$ | 计算快，缓解梯度消失 | 死神经元，非零中心 | CNN，MLP 隐藏层 |
| LeakyReLU | $\max(\alpha x,x)$ | $\begin{cases}1&x>0\\\alpha&x\leq0\end{cases}$ | $(-\infty,+\infty)$ | 缓解死神经元 | α 需调参 | GAN，检测网络 |
| PReLU | $\max(\alpha x,x)$, α learnable | 同 LeakyReLU | $(-\infty,+\infty)$ | α 自适应 | 多一个参数 | ResNet (He 2015) |
| ELU | $\begin{cases}x&x>0\\\alpha(e^x-1)&x\leq0\end{cases}$ | $\begin{cases}1&x>0\\\alpha e^x&x\leq0\end{cases}$ | $(-\alpha,+\infty)$ | 平滑，负值推均值趋零 | exp 计算开销 | 较少主流使用 |
| SELU | $\lambda\begin{cases}x&x>0\\\alpha(e^x-1)&x\leq0\end{cases}$ | $\lambda\begin{cases}1&x>0\\\alpha e^x&x\leq0\end{cases}$ | $(-\lambda\alpha,+\infty)$ | 自归一化 | 需 lecun_normal 初始化 | SNN |
| GELU | $x\cdot\Phi(x)$ | $\Phi(x)+x\cdot\phi(x)$ | $\approx(-0.17,+\infty)$ | 平滑，概率解释 | 计算略贵 | BERT, GPT, ViT |
| SiLU/Swish | $x\cdot\sigma(x)$ | $\sigma(x)+x\cdot\sigma(x)(1-\sigma(x))$ | $\approx(-0.28,+\infty)$ | 平滑，非单调 | 计算略贵 | EfficientNet |
| Mish | $x\cdot\tanh(\text{softplus}(x))$ | 复杂（见正文） | $\approx(-0.31,+\infty)$ | 更平滑 | 计算最贵 | YOLOv4 |
| GLU | $(xW)\otimes\sigma(xV)$ | — | 取决于门控 | 信息门控 | 参数量翻倍 | 早期 NLP |
| SwiGLU | $(xW)\otimes\text{SiLU}(xV)$ | — | 取决于门控 | 门控+平滑，效果最佳 | 参数量翻倍 | LLaMA, PaLM |
| GeGLU | $(xW)\otimes\text{GELU}(xV)$ | — | 取决于门控 | 门控+概率解释 | 参数量翻倍 | 部分 LLM |

---

## 2. 经典激活函数

### 2.1 Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

**导数：**

$$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

导数最大值仅为 0.25（在 $x=0$ 处取到），这意味着每经过一层，梯度至少缩小为 $\frac{1}{4}$。

**核心问题：**
1. **梯度饱和（Gradient Saturation）**：当 $|x|>5$ 时，$\sigma'(x)\approx 0$，梯度几乎消失，深层网络训练困难
2. **输出非零中心（Non-zero-centered）**：$\sigma(x)\in(0,1)$，导致后续层的梯度更新出现 zig-zag 现象，收敛变慢
3. **exp 运算开销**：相比 ReLU 的 max 操作，指数运算更耗时

**适用场景：** 二分类输出层（输出概率）、门控机制（LSTM/GRU 的 gate）、CTR 预估输出层（参见 [[ctr_calibration]]）

> **面试一句话：** Sigmoid 做输出层天然输出概率，但做隐藏层会梯度消失——因为导数最大才 0.25，层一多就指数级衰减。

### 2.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**导数：**

$$\tanh'(x) = 1 - \tanh^2(x)$$

**相比 Sigmoid 的优势：**
- 零中心输出 $(-1,1)$，梯度更新不会出现全正/全负的问题
- 导数最大值为 1（在 $x=0$），比 sigmoid 的 0.25 大，梯度传播更强

**仍然存在的问题：** 当 $|x|>3$ 时依然饱和，深层网络梯度消失问题没有根本解决。

**适用场景：** RNN/LSTM 隐藏层（历史原因 + 零中心有助于时序建模），归一化层后的激活。

> **面试一句话：** Tanh 是 sigmoid 的平移缩放版，零中心是核心优势，但仍然逃不开饱和问题。

### 2.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

**导数：**

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**为什么 ReLU 是里程碑：**
1. **计算效率极高**：只需一次比较操作，比 exp/tanh 快一个数量级
2. **缓解梯度消失**：正区间导数恒为 1，无论多深梯度都不会衰减
3. **稀疏激活**：约 50% 的神经元输出为 0，带来正则化效果和表示高效性

**核心问题——死神经元（Dying ReLU）：** 见第 6 节详述。

**适用场景：** CNN 首选（AlexNet 以来的标配），MLP 隐藏层，任何对速度敏感的场景。

> **面试一句话：** ReLU 用一个 max 操作解决了梯度消失 + 计算效率两大问题，但代价是死神经元——负区间梯度永远为 0，一旦偏置推到负区间就回不来了。

---

## 3. ReLU 变种

### 3.1 LeakyReLU

$$\text{LeakyReLU}(x) = \max(\alpha x, x), \quad \alpha = 0.01 \text{ (typical)}$$

负区间给一个小斜率 $\alpha$，让梯度不至于完全为零。

**关键点：**
- $\alpha$ 通常取 0.01 或 0.1
- 解决死神经元问题，但 $\alpha$ 是超参数
- 实践中性能提升不一定显著——很多场景和 ReLU 差不多

### 3.2 PReLU (Parametric ReLU)

$$\text{PReLU}(x) = \max(\alpha x, x), \quad \alpha \text{ 为可学习参数}$$

He et al. (2015) 在 ResNet 论文中提出。每个 channel（或每个神经元）学习自己的 $\alpha$。

**要点：**
- 比 LeakyReLU 多了自适应能力
- 参数量增加很少（一个 channel 一个 $\alpha$）
- 在 ImageNet 上曾超越 human-level performance

### 3.3 ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

**优势：**
- 负值区域平滑过渡，均值更接近零（类似 batch norm 的效果）
- $x=0$ 处连续且导数连续

**劣势：** 负区间需要 exp 运算，计算比 ReLU 贵。

### 3.4 SELU (Scaled ELU)

$$\text{SELU}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

其中 $\alpha \approx 1.6733$，$\lambda \approx 1.0507$，这两个值是推导出来的，不是调的。

**自归一化（Self-Normalizing）性质：** 在满足以下条件时，网络激活值自动收敛到均值 0、方差 1：
1. 使用 `lecun_normal` 初始化
2. 全连接层（不是 CNN/RNN）
3. 不使用 BatchNorm（因为 SELU 自带归一化）

**限制太多，实际使用场景有限。**

> **面试一句话：** SELU 理论上很美——自归一化不需要 BN，但条件苛刻（全连接 + lecun_normal），实际工程几乎不用。

---

## 4. 现代激活函数 (Transformer 时代)

### 4.1 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的 CDF：$\Phi(x) = P(X \leq x), X \sim \mathcal{N}(0,1)$。

**近似公式（工程实现常用）：**

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

**为什么 GELU 适合 Transformer：**
1. **概率解释**：$x$ 乘以「$x$ 在高斯分布中的累积概率」——直觉上是根据输入值的「相对大小」来加权。较大的输入几乎不变（$\Phi(x)\to 1$），较小的被压制（$\Phi(x)\to 0$）
2. **平滑过渡**：在 $x=0$ 附近是平滑的，不像 ReLU 有硬拐点。对 [[attention_transformer]] 中的梯度流非常友好
3. **非单调**：在 $x \approx -0.17$ 处有一个小的负值区域，提供更丰富的表达能力

**使用：** BERT、GPT-2、GPT-3、ViT 全部使用 GELU。

> **面试一句话：** GELU 是用高斯 CDF 做门控的软 ReLU——输入大则保留，输入小则抑制，平滑性让 Transformer 梯度更稳。

### 4.2 SiLU / Swish

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

Swish 是 Google Brain 用 NAS 搜索出来的，后来发现和 SiLU（Sigmoid Linear Unit）是同一个函数。

**可学习 β 版本：**

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

- $\beta \to 0$：退化为线性函数 $x/2$
- $\beta \to \infty$：退化为 ReLU
- $\beta = 1$：标准 SiLU

**性质：**
- 平滑、非单调
- 在 $x \approx -1.28$ 处有全局最小值 $\approx -0.28$
- 无上界、有下界

**使用：** EfficientNet, 部分 LLM 的 FFN 层。

> **面试一句话：** SiLU/Swish 是 $x \cdot \sigma(x)$，NAS 搜出来的。β 可调时能在线性和 ReLU 之间连续插值。

### 4.3 Mish

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1+e^x))$$

**特点：**
- 比 Swish 更平滑（二阶导数更平滑）
- 全局最小值 $\approx -0.31$
- 计算成本最高（tanh + softplus + 乘法）

**使用：** YOLOv4 默认激活函数，CV 领域有一定影响力，但 NLP 中很少用。

### 4.4 GLU 家族（面试重点）

GLU（Gated Linear Unit）及其变种是 LLM 时代最重要的激活函数，**必须掌握**。

#### 基础 GLU

$$\text{GLU}(x, W, V, b, c) = (xW + b) \otimes \sigma(xV + c)$$

核心思想：**将输入分成两路，一路做信息变换，一路做门控**。$\otimes$ 是逐元素乘法。

#### SwiGLU

$$\text{SwiGLU}(x, W, V) = (xW) \otimes \text{SiLU}(xV)$$

**LLaMA / PaLM / Chinchilla 的标准 FFN 激活。**

#### GeGLU

$$\text{GeGLU}(x, W, V) = (xW) \otimes \text{GELU}(xV)$$

#### 为什么 GLU 变种如此有效

1. **信息门控机制**：门控路径可以学习「哪些特征该通过、哪些该抑制」——类似 LSTM 的 gate，但用在 FFN 里
2. **梯度流改善**：门控乘法为梯度提供了两条路径（类似 residual connection 的精神）
3. **表达能力更强**：两个投影矩阵 $W, V$ 使得 FFN 能学到更复杂的特征交互

#### FFN 维度调整（面试常问）

标准 Transformer FFN：
$$\text{FFN}(x) = \text{GELU}(xW_1)W_2, \quad W_1 \in \mathbb{R}^{d \times 4d}$$

SwiGLU FFN（LLaMA 做法）：
$$\text{FFN}(x) = [\text{SiLU}(xW_\text{gate}) \otimes (xW_\text{up})]W_\text{down}$$

其中 $W_\text{gate}, W_\text{up} \in \mathbb{R}^{d \times \frac{8d}{3}}$，$W_\text{down} \in \mathbb{R}^{\frac{8d}{3} \times d}$。

**为什么是 $\frac{8d}{3}$ 而不是 $4d$？** 因为 GLU 有两个投影（gate + up），参数量翻倍。为了保持总参数量和标准 FFN 一致，将 hidden dim 从 $4d$ 调整为 $\frac{2}{3} \times 4d = \frac{8d}{3}$。

> **面试一句话：** SwiGLU 把 FFN 变成门控结构——一路做变换、一路做门控、逐元素相乘。为保持参数量不变，hidden dim 要乘 $\frac{2}{3}$。

---

## 5. 重点对比: ReLU vs GELU vs SiLU vs SwiGLU

### 5.1 详细对比表

| 维度 | ReLU | GELU | SiLU/Swish | SwiGLU |
|------|------|------|-----------|--------|
| 公式复杂度 | 最低 | 中等 | 中等 | 高（两路投影） |
| 平滑性 | 不平滑（$x=0$ 处不可导） | 平滑 | 平滑 | 平滑 |
| 单调性 | 单调 | 非单调 | 非单调 | 非单调 |
| 死神经元 | 严重 | 无 | 无 | 无 |
| 稀疏性 | 高（50% 为 0） | 低 | 低 | 中（门控产生的） |
| 计算开销 | 最低 | 中等 | 中等 | 最高 |
| 参数量影响 | 无 | 无 | 无 | FFN 增加 50% 参数（或调维度） |
| 代表模型 | ResNet, VGG | BERT, GPT, ViT | EfficientNet | LLaMA, PaLM |

### 5.2 为什么 BERT/GPT 用 GELU

1. **Transformer 中 attention 输出在 0 附近密集分布**，ReLU 的硬拐点会造成不必要的信息丢失
2. **GELU 的概率门控**让小但有意义的负值信号得以保留——这在 NLU 任务中很重要
3. **平滑的梯度**使得 Adam 等自适应优化器的二阶矩估计更稳定

与 [[attention_transformer]] 的协同：attention 输出经过 LayerNorm 后分布接近正态，GELU 的高斯 CDF 门控恰好匹配这个分布。

### 5.3 为什么 LLaMA 用 SwiGLU

1. **门控机制引入信息选择**：不只是逐元素非线性变换，还能学会「选择性遗忘」
2. **实验优势明显**：PaLM 论文实验显示 SwiGLU 在同等参数量下 loss 更低
3. **FFN 维度调整的 trick**：$\frac{2}{3} \times 4d$ 保持参数量一致，但表达能力更强

### 5.4 为什么 CNN 仍然用 ReLU

1. **速度**：CNN 层数多、feature map 大，每个像素都要过激活。ReLU 的 max 操作比 sigmoid/tanh/GELU 快 5-10 倍
2. **稀疏激活对 CV 有益**：50% 为零的特征图天然起到 dropout-like 的正则化效果
3. **收益不明显**：在 ImageNet 上 GELU vs ReLU 差异 < 0.5%，不值得增加计算开销

---

## 6. 死神经元问题 & 解法

### 6.1 机制

当一个 ReLU 神经元的输入（加权和 + 偏置）始终 $\leq 0$ 时：
- 输出恒为 0
- 梯度恒为 0
- 权重无法更新
- **永久死亡，不可恢复**

### 6.2 触发条件

1. **学习率过大**：权重更新幅度过大，偏置被推到深负区间
2. **不良初始化**：初始权重使大部分输入为负
3. **数据分布偏移**：训练到后期，输入分布变化导致神经元输出全负

### 6.3 检测

```python
# 检测死神经元比例
def dead_neuron_ratio(model, dataloader):
    activations = {}
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(
                lambda m, i, o, name=name: activations.setdefault(name, []).append((o > 0).float().mean().item())
            ))
    # forward pass over a few batches
    for batch in itertools.islice(dataloader, 10):
        model(batch)
    for h in hooks:
        h.remove()
    for name, acts in activations.items():
        ratio = 1.0 - sum(acts) / len(acts)
        print(f"{name}: {ratio:.2%} dead")
```

### 6.4 解决方案

| 方案 | 原理 | 代价 |
|------|------|------|
| LeakyReLU / PReLU | 负区间给小梯度 | 几乎无 |
| 合理初始化（He init） | 保持方差 $\text{Var}(w) = \frac{2}{n_\text{in}}$ | 无 |
| BatchNorm | 归一化后输入分布居中，减少全负概率 | BN 的计算开销 |
| 降低学习率 | 减小权重更新幅度 | 收敛变慢 |
| 用 GELU/SiLU 替换 | 从根本上消除死区 | 计算略贵 |

> **面试一句话：** 死神经元 = ReLU 负区间梯度为零 → 权重不再更新 → 永久死亡。治标用 LeakyReLU，治本用 GELU 或正确初始化 + BN。

---

## 7. 搜广推中的激活函数选择

### 7.1 CTR 模型

| 层 | 推荐激活函数 | 原因 |
|----|------------|------|
| 输出层 | Sigmoid | 输出概率，接 cross-entropy loss |
| MLP 隐藏层 | ReLU 或 PReLU | 计算快，大规模 serving 友好 |
| 特征交互层 | 取决于架构 | DIN 用 PReLU (Dice)，DCN 无显式激活 |

**DIN 的 Dice 激活：** 阿里 DIN 模型提出的 data-aware 激活函数，本质是基于 batch 统计量自适应调整 PReLU 的拐点：

$$\text{Dice}(x) = p(x) \cdot x + (1-p(x)) \cdot \alpha x$$

其中 $p(x) = \frac{1}{1+e^{-\frac{x-\mathbb{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}}}}$，即对 BN 后的值做 sigmoid。

CTR 模型输出层的 sigmoid 与 calibration 密切相关，参见 [[ctr_calibration]]。

### 7.2 Transformer-based 推荐

随着推荐系统引入 Transformer 架构（如 SASRec、BERT4Rec、P5），激活函数选择与 NLP 趋同：

- **Self-attention 后的 FFN**：GELU（BERT4Rec）或 SwiGLU（新模型）
- **预测头**：Sigmoid（CTR）或 Softmax（top-K 推荐）

### 7.3 多任务学习

在 [[mmoe_multitask]] 等多任务架构中：
- 共享底座的 MLP：ReLU/PReLU（计算效率优先）
- Expert 网络：ReLU
- Gate 网络：**Softmax**（注意不是激活函数，而是归一化）
- 各 Tower 输出层：根据任务类型选择（CTR 用 sigmoid，回归用 linear）

### 7.4 工程考量

搜广推系统 serving 延迟敏感，激活函数的选择还要考虑：
1. **推理速度**：ReLU >> GELU > SwiGLU，高 QPS 场景 ReLU 仍是首选
2. **量化友好性**：ReLU 输出非负，INT8 量化更简单；GELU 负值区域量化损失大
3. **ONNX/TensorRT 支持**：ReLU 原生支持，GELU 需要自定义算子或近似实现

---

## 8. 面试高频问答

### Q1: 为什么不用 Sigmoid 做隐藏层激活？
**答：** 两个致命问题——(1) 梯度饱和：导数最大 0.25，深层网络梯度指数衰减；(2) 非零中心：输出全正，导致梯度 zig-zag。用 ReLU 或 GELU 替代。
**常见坑：** 别说 sigmoid 完全不用——输出层和门控还是要用的。

### Q2: ReLU 在 $x=0$ 处不可导，怎么做反向传播？
**答：** 实践中定义 $x=0$ 时导数为 0（或 1，都可以），因为 $x$ 精确等于 0 的概率为零，对训练无影响。这是 sub-gradient 的思想。
**常见坑：** 不要纠结数学严谨性——工程上无影响。

### Q3: GELU 和 ReLU 的本质区别？
**答：** ReLU 是硬门控（$x>0$ 全过，$x\leq0$ 全杀），GELU 是软门控（根据 $x$ 在高斯分布中的位置概率决定保留多少）。GELU 在 $x=0$ 附近平滑过渡，允许小负值部分通过，梯度更稳定。

### Q4: 解释 SwiGLU 为什么比 GELU 好？
**答：** SwiGLU 引入了 **门控机制**（两路投影 + 逐元素乘），FFN 能学到更复杂的特征交互。GELU 只是逐元素非线性，SwiGLU 相当于给 FFN 加了一个注意力 gate。代价是参数量增加，需要调整 hidden dim 为 $\frac{2}{3} \times 4d$ 来保持总参数量不变。

### Q5: 搜广推模型该用什么激活函数？
**答：** 看场景——(1) 传统 CTR（DeepFM/DCN）：隐藏层 ReLU/PReLU + 输出 Sigmoid；(2) Transformer-based 推荐：GELU；(3) 如果是自研 LLM-based 推荐：SwiGLU。**serving 延迟约束下优先 ReLU。**
参见 [[ctr_calibration]] 中输出层 sigmoid 与 calibration 的关系。

### Q6: 死神经元比例多少算正常？
**答：** 训练初期 5-15% 是正常的（稀疏激活的好处），但如果超过 50% 或持续上升就有问题。检测方法：forward hook 统计每个 ReLU 层的零值比例。

### Q7: Transformer 的 FFN 为什么要扩展到 4d 再压回 d？
**答：** FFN 的作用是做 key-value memory（Geva et al., 2021），扩展到 4d 提供更多「记忆槽位」。高维空间中非线性变换的表达能力更强，最后压回 d 做信息筛选。SwiGLU 版本调到 $\frac{8d}{3}$ 是参数量对齐。

### Q8: BatchNorm 和激活函数的先后顺序？
**答：** 争议话题。原始 BN 论文说 BN → Activation，但后来很多实践（如 ResNet v2）证明 Activation → BN 也可以。经验上差别不大，但 BN 放在 ReLU 前面能让输入分布居中、减少死神经元。

---

## 参考文献

1. Nair & Hinton (2010). Rectified Linear Units Improve Restricted Boltzmann Machines
2. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance (PReLU)
3. Hendrycks & Gimpel (2016). Gaussian Error Linear Units (GELUs)
4. Ramachandran et al. (2017). Searching for Activation Functions (Swish)
5. Shazeer (2020). GLU Variants Improve Transformer
6. Dauphin et al. (2017). Language Modeling with Gated Convolutional Networks (GLU)
7. Zhou et al. (2018). Deep Interest Network (Dice activation)
