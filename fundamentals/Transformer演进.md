# Transformer 架构演进全景：从 Attention Is All You Need 到现代 LLM

> 标签：#transformer #architecture #LLM #attention #RoPE #FlashAttention #MoE #DeepSeek #面试
> 关联：[[attention_transformer]] | [[kv_cache_inference]] | [[activation_functions]] | [[08_LLM预训练与架构演进]] | [[07_MoE架构与稀疏激活]] | [[attention_in_recsys]]

---

## 1. 演进总览时间线

| 年份 | 模型 | 参数量 | 训练数据 | 关键架构改动 |
|------|------|--------|----------|-------------|
| 2017 | Transformer | 65M | WMT (4.5M句对) | Self-Attention + Encoder-Decoder + Sinusoidal PE |
| 2018 | GPT-1 | 117M | BookCorpus (5GB) | **Decoder-Only** + 可学习位置编码 |
| 2018 | BERT | 340M | 16GB 文本 | **Encoder-Only** + MLM + NSP |
| 2019 | GPT-2 | 1.5B | WebText (40GB) | Pre-Norm (LayerNorm 前移) |
| 2020 | GPT-3 | 175B | 300B tokens | Dense Scaling + In-Context Learning |
| 2022 | Chinchilla | 70B | 1.4T tokens | **Scaling Law 修正**：数据量 ≈ 20× 参数量 |
| 2023 | LLaMA | 7-65B | 1.4T tokens | **RoPE + SwiGLU + Pre-RMSNorm** |
| 2023 | Mistral-7B | 7B | 未公开 | **Sliding Window Attention + GQA** |
| 2024 | LLaMA-3 | 8-405B | 15T tokens | **GQA** + 128K 上下文 |
| 2025 | DeepSeek-V3 | 671B (37B活) | 14.8T tokens | **MLA + MoE + MTP + FP8训练** |
| 2025 | Qwen3 | 235B (22B活) | 36T tokens | MoE + GQA + 四阶段训练 |

> **趋势总结**：参数 Scaling → 数据 Scaling → 稀疏 Scaling (MoE)；架构趋同为 Decoder-Only + RoPE + SwiGLU + Pre-RMSNorm + GQA/MLA。

### 1.1 三代架构范式

**第一代：Encoder-Decoder (2017-2019)**
- 原始 Transformer、T5、BART
- Encoder 双向 attention 理解输入，Decoder 单向 attention 生成输出
- 适合翻译、摘要等 seq2seq 任务，但每个任务需要 fine-tune

**第二代：Encoder-Only + Decoder-Only 分化 (2018-2020)**
- BERT (Encoder-Only)：MLM 预训练 → 下游 fine-tune，理解型任务主导
- GPT 系列 (Decoder-Only)：自回归预训练 → few-shot prompt，生成型任务主导
- 关键转折点：GPT-3 证明 Decoder-Only 通过 scale up 也能做好理解任务

**第三代：Decoder-Only 统一 (2023-至今)**
- LLaMA 系列确立 Decoder-Only + 开源模式
- 架构改进从"选择范式"变为"优化组件"（attention/位置编码/FFN/norm）
- MoE 成为 scale up 的新范式，突破 dense 模型的算力瓶颈

### 1.2 Scaling Law 驱动的架构选择

Kaplan et al. (2020) 的 Scaling Law：

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

其中 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$。关键推论：
- 数据量的边际收益 > 参数量的边际收益
- 固定 FLOPs 预算下，Chinchilla 最优：$N \propto C^{0.5}$, $D \propto C^{0.5}$
- 这解释了为什么 LLaMA (7B, 1.4T) 优于 GPT-3 (175B, 300B)——后者严重欠训练

---

## 2. Attention 机制演进

> 详细基础见 [[attention_transformer]]，本节聚焦变体对比。

### 2.1 从 MHA 到 MLA

| 变体 | 年份 | KV Head 数 | KV Cache/层 | 核心思想 |
|------|------|-----------|------------|---------|
| MHA | 2017 | $h$ | $2 \times h \times d_k \times n$ | 每个 head 独立 KV |
| MQA | 2019 | 1 | $2 \times d_k \times n$ | **所有 head 共享 1 组 KV** |
| GQA | 2023 | $g$ ($1 < g < h$) | $2 \times g \times d_k \times n$ | **分 $g$ 组，组内共享 KV** |
| MLA | 2024 | - | $d_c \times n$ ($d_c \ll h \times d_k$) | **低秩联合压缩 KV** |

**MHA (Multi-Head Attention)**

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

每个 head 有独立的 $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d \times d_k}$。KV Cache 大小 = $2 \times L \times h \times d_k \times n$（L 层，n 序列长度）。

**MQA (Multi-Query Attention)**

- **Why 改**：推理时 KV Cache 是内存瓶颈，MHA 的 KV 随 head 数线性增长。
- **改了什么**：所有 Query head 共享同一组 K 和 V。$W^K, W^V$ 全局只有一份。
- **效果**：KV Cache 缩小 $h$ 倍（如 $h=32$ 则 32× 压缩），推理速度提升，但质量略有下降。

**GQA (Grouped-Query Attention)**

- **Why 改**：MQA 压缩太激进，质量损失明显。需要 MHA 和 MQA 之间的平衡点。
- **改了什么**：将 $h$ 个 Query head 分为 $g$ 组，每组共享一组 KV。$g=1$ 退化为 MQA，$g=h$ 退化为 MHA。
- **效果**：LLaMA-3 用 $g=8$（8 KV heads, 64 Q heads），质量接近 MHA，KV Cache 缩小 8×。

$$
\text{GQA}: \text{head}_{i} = \text{Softmax}\left(\frac{Q_i K_{\lfloor i/(h/g) \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor i/(h/g) \rfloor}
$$

**MLA (Multi-head Latent Attention) — DeepSeek-V2/V3**

- **Why 改**：GQA 仍按 head 粒度共享，没有利用 KV 之间的冗余。能否用更少维度表示 KV？
- **改了什么**：将 KV 联合压缩到低秩隐向量 $c_t \in \mathbb{R}^{d_c}$，推理时只缓存 $c_t$。

$$
c_t = W^{DKV} h_t, \quad d_c \ll d \quad (\text{如 } d_c = 512, d = 7168)
$$
$$
K_t = W^{UK} c_t, \quad V_t = W^{UV} c_t
$$

- **效果**：KV Cache 只有 $d_c \times n$。DeepSeek-V3 中 $d_c = 512$，相比 GQA ($g=8, d_k=128$) 的 $2 \times 8 \times 128 = 2048$ 维，压缩约 4×；相比 MHA 压缩约 28×。且质量不降反升，因为低秩投影自带正则化。

> **面试重点**：GQA 成为主流是因为工程简单且效果够好。MLA 更优但需要特殊的吸收技巧（absorption trick）将 $W^{UK}$ 吸收进 $W^Q$ 避免推理时还原 K 的计算开销。

---

## 3. 位置编码演进

| 方案 | 年份 | 可学习 | 外推性 | 长度泛化 | 代表模型 |
|------|------|--------|--------|---------|---------|
| Sinusoidal | 2017 | 否 | 差 | 差 | Transformer |
| Learned Absolute | 2018 | 是 | 无 | 差 | GPT-1/2, BERT |
| ALiBi | 2022 | 否 | 好 | 中等 | BLOOM, MPT |
| RoPE | 2022 | 否 | 好 | 好（配合扩展） | LLaMA 全系, DeepSeek, Qwen |
| YaRN | 2023 | 否 | 极好 | 极好 | LLaMA 长上下文版 |

### 3.1 Sinusoidal 位置编码（原始 Transformer）

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

- **原理**：不同频率的三角函数编码位置，相对位置可通过线性变换推导。
- **问题**：与内容无关的加法编码，外推到训练长度之外效果骤降。

### 3.2 ALiBi (Attention with Linear Biases)

- **Why 改**：位置编码加在输入上太间接，能否直接在 attention score 上施加位置约束？
- **改了什么**：不加位置编码，直接在 attention score 上减去线性偏置：

$$
\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} - m \cdot |i - j|\right)
$$

其中 $m$ 是每个 head 预设的斜率（几何级数），远距离 token 的 attention 自然衰减。

- **效果**：零参数，外推性好，训练 1K 可推理 2K+。但衰减是线性的，对需要精确远距离定位的任务（如 coding）不够灵活。

### 3.3 RoPE (Rotary Position Embedding)

- **Why 改**：需要同时满足：(1) 编码相对位置信息，(2) 与内容交互，(3) 良好外推性。
- **改了什么**：将位置编码建模为向量空间中的旋转：

$$
f(q, m) = R_{\Theta, m} \cdot q = \begin{pmatrix} q_1 \\ q_2 \\ q_3 \\ q_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \end{pmatrix} + \begin{pmatrix} -q_2 \\ q_1 \\ -q_4 \\ q_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \end{pmatrix}
$$

关键性质：$\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)$，内积只依赖相对位置 $m-n$。

- **效果**：统一了所有现代 LLM。配合 NTK-aware 插值或 YaRN 可扩展到 128K+ 上下文。

### 3.4 YaRN / NTK-aware 扩展

- **Why 改**：RoPE 在训练长度内优秀，超出后高频分量出现周期混叠。
- **改了什么**：NTK-aware 调整 RoPE 的基频 $\theta$，区分高低频分量分别处理；YaRN 进一步加温度缩放和注意力缩放。
- **效果**：LLaMA-3 用 RoPE + 渐进式上下文扩展，从 8K 训练扩展到 128K。

> **为什么 RoPE 统一了所有现代 LLM？** (1) 自然编码相对位置，(2) 与 attention 计算深度耦合（旋转作用于 Q/K），(3) 外推性配合插值方案可扩展，(4) 实现简单，与 FlashAttention 兼容。

### 3.5 位置编码对比总结

| 对比维度 | Sinusoidal | Learned | ALiBi | RoPE |
|----------|-----------|---------|-------|------|
| 参数量 | 0 | $L_{max} \times d$ | 0 | 0 |
| 编码类型 | 绝对 | 绝对 | 相对(隐式) | 相对(显式) |
| 注入方式 | 加到输入 | 加到输入 | 加到 attention score | 旋转 Q/K |
| 训练长度外推 | 差 | 无 | 好（线性衰减） | 好（配合 NTK） |
| 对远距离依赖 | 中性 | 中性 | 负面（强制衰减） | 中性（模型自学） |
| FlashAttention兼容 | 是 | 是 | 需特殊处理 | 是 |
| 长度泛化方案 | 无 | 无 | 内置 | NTK/YaRN/PI |

**Position Interpolation (PI)**：将超出训练长度的位置按比例缩放回训练范围内。如训练长度 4K，推理 8K 时将所有位置除以 2。简单但需要少量 fine-tune。

**NTK-aware Interpolation**：不均匀缩放——低频维度（编码长程位置）压缩更多，高频维度（编码局部位置）压缩更少。相比 PI 效果更好，且可以 training-free。

---

## 4. FFN 选型演进

> 详细激活函数对比见 [[activation_functions]]。

FFN（Feed-Forward Network）占 Transformer 总参数的约 2/3，是架构演进中变化最大的组件之一。从 2017 年的 ReLU FFN 到 2025 年的 MoE SwiGLU，经历了四代演进。

### 4.0 演进总览

| 代际 | 变体 | 年份 | 公式核心 | FFN 参数量 | 代表模型 |
|------|------|------|---------|-----------|---------|
| 第一代 | ReLU FFN | 2017 | $\max(0, xW_1)W_2$ | $2 \times d \times d_{ff}$ | Transformer |
| 第一代 | GELU FFN | 2018 | $\text{GELU}(xW_1)W_2$ | $2 \times d \times d_{ff}$ | GPT-2/3, BERT |
| 第二代 | GLU | 2016/2020 | $(xW_1 \otimes \sigma(xW_g))W_2$ | $3 \times d \times d_{ff}'$ | - |
| 第二代 | GeGLU | 2020 | $(xW_1 \otimes \text{GELU}(xW_g))W_2$ | $3 \times d \times d_{ff}'$ | Gemma |
| 第二代 | SwiGLU | 2020 | $(xW_1 \otimes \text{Swish}(xW_g))W_2$ | $3 \times d \times d_{ff}'$ | **LLaMA 全系, DeepSeek, Qwen** |
| 第三代 | MoE FFN | 2022+ | Top-K 路由到 N 个专家 FFN | $N \times \text{单专家参数}$ | Mixtral, DeepSeek-V3, Qwen3 |
| 前沿 | MoE + 共享专家 | 2024 | 1 共享专家 + N 路由专家 | $(N+1) \times \text{单专家}$ | DeepSeek-V2/V3 |

### 4.1 第一代：标准 FFN (ReLU/GELU)

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

原始 Transformer 用 $d_{ff} = 4d$（如 $d=512$, $d_{ff}=2048$），参数量 = $2 \times d \times d_{ff} = 8d^2$。

**ReLU → GELU 的演进**：

$$
\text{ReLU}(x) = \max(0, x) \qquad \text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])
$$

| 对比 | ReLU | GELU |
|------|------|------|
| 平滑性 | 非平滑（0点不可导） | 平滑，处处可导 |
| 零点行为 | 硬截断，负值全为0 | 软截断，小负值保留少量信息 |
| 死神经元 | 严重（~10-30% 神经元永久失活） | 极少 |
| 梯度特性 | 正区恒为1，负区恒为0 | 连续变化，梯度信号更丰富 |
| 代表模型 | 原始 Transformer, T5 | GPT-2/3, BERT, 早期 LLM |

**Why GELU 取代 ReLU**：语言模型需要处理大量接近零的激活值（如"不太相关但有微弱信号"的特征），ReLU 的硬截断丢失这些弱信号。GELU 的概率加权（$x \cdot \Phi(x)$）让模型自适应地保留或抑制，实验中 pre-training loss 稳定低于 ReLU。

### 4.2 第二代：Gated Linear Unit (GLU) 系列

**GLU 的核心创新**：将 FFN 拆成两个分支——**内容分支**产出信息，**门控分支**决定放行多少。

$$
\text{GLU}(x) = (xW_1) \otimes \sigma(xW_g)
$$

其中 $\otimes$ 是逐元素乘法，$\sigma$ 是 Sigmoid。Dauphin et al. (2016) 提出，Shazeer (2020) 系统测试了所有激活函数变体。

**GLU 变体全家谱**：

| 变体 | 门控激活函数 $\sigma_g$ | 公式 | 效果排名 |
|------|----------------------|------|---------|
| GLU | Sigmoid | $(xW_1) \otimes \sigma(xW_g)$ | 第四 |
| ReGLU | ReLU | $(xW_1) \otimes \text{ReLU}(xW_g)$ | 第三 |
| GeGLU | GELU | $(xW_1) \otimes \text{GELU}(xW_g)$ | 第二 |
| **SwiGLU** | **Swish/SiLU** | $(xW_1) \otimes \text{Swish}(xW_g)$ | **第一** |

$$
\text{Swish}(x) = x \cdot \sigma(\beta x), \quad \beta=1 \text{ 时称 SiLU}
$$

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_g) \otimes xW_1) W_2
$$

**Shazeer (2020) 实验结论**（PaLM 论文验证）：
- 在相同参数量和 FLOPs 下，SwiGLU > GeGLU > ReGLU > GELU > ReLU
- SwiGLU 比标准 GELU FFN 在 C4 数据集上 loss 低约 0.05-0.1（大模型中极显著）

**为什么门控有效？直觉解释**：

```
标准 FFN：x → 激活(x * W1) → output
  所有信息经过同一个激活函数，无选择性

GLU 系列：x → [内容分支: x * W1] ⊗ [门控分支: σ(x * Wg)] → output
  门控分支学会"这个维度的信息是否有用"
  等价于一个逐维度的 soft attention
```

门控机制让 FFN 变成了一个**可学习的特征选择器**——内容分支提取候选特征，门控分支筛选哪些通过。这比逐元素激活（如 GELU）的表达力更强。

**参数量设计**：GLU 引入第三个矩阵 $W_g$，总参数 = $3 \times d \times d_{ff}'$。为保持与标准 FFN 参数量相当：

$$
3 \times d \times d_{ff}' = 2 \times d \times 4d \implies d_{ff}' = \frac{8d}{3} \approx 2.67d
$$

LLaMA 实际用 $d_{ff}' = \frac{8}{3}d$ 再向上取到 256 的整数倍（如 $d=4096$ 时 $d_{ff}'=11008$）。DeepSeek-V3 ($d=7168$) 用 $d_{ff}'=18432$。

**SwiGLU 现已成为绝对主流**——LLaMA 全系列、DeepSeek、Qwen、Gemma、Mistral、Yi、Baichuan 均采用。无一例外。

### 4.3 第三代：MoE FFN（稀疏 FFN）

> MoE 详解见 [[07_MoE架构与稀疏激活]]。

将单个 FFN 替换为 N 个「专家 FFN」+ 1 个路由器。每个 token 只激活 Top-K 个专家。

$$
\text{MoE-FFN}(x) = \sum_{i \in \text{TopK}} g_i(x) \cdot \text{FFN}_i(x)
$$

$$
g(x) = \text{TopK}(\text{Softmax}(x W_r), K)
$$

| 模型 | 总专家数 N | 激活 K | 单专家结构 | 总参数 | 激活参数 | 稀疏率 |
|------|---------|-------|----------|--------|---------|--------|
| Switch Transformer | 128 | 1 | ReLU FFN | 1.6T | ~5B | 99.7% |
| Mixtral-8x7B | 8 | 2 | SwiGLU | 47B | 13B | 72% |
| DeepSeek-V3 | 256+1共享 | 8+1 | SwiGLU | 671B | 37B | 94.5% |
| Qwen3 | 128 | 8 | SwiGLU | 235B | 22B | 90.6% |

**MoE 中每个专家的 FFN 结构**：本质上就是一个缩小版的 SwiGLU FFN。如 Mixtral 的每个专家 $d_{ff}'=14336$（与 Mistral-7B 相同），但只激活 2 个。

**MoE FFN 的工程挑战**：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 负载不均衡 | 路由器偏好某些专家 | 辅助 loss / 偏置动态调整 (DeepSeek) |
| 专家冗余 | 多个专家学到相似功能 | 共享专家 (DeepSeek-V2) |
| 通信开销 | 分布式训练中 token 需发送到不同设备的专家 | Expert Parallelism + All-to-All 通信 |
| 推理不确定性 | 不同 token 走不同专家，batch 化困难 | Token Dropping / 预分配缓冲区 |

### 4.4 共享专家 + 路由专家（DeepSeek-V2/V3）

**Why 改**：纯路由 MoE 中，所有专家都通过竞争分配 token。但有些知识是"通用的"（如语法、基础语义），不需要专门化。

**改了什么**：引入 1 个始终激活的共享专家，处理通用知识；其余 256 个路由专家处理专门化知识。

$$
\text{MoE-FFN}(x) = \text{FFN}_{\text{shared}}(x) + \sum_{i \in \text{TopK}} g_i(x) \cdot \text{FFN}_i(x)
$$

**效果**：减少专家间冗余，提升参数利用率。共享专家相当于一个"公共底座"，路由专家在其基础上做差异化。

### 4.5 FFN 宽度比例演进

| 模型 | FFN 类型 | $d_{ff}/d$ 比例 | 实际值 ($d$→$d_{ff}$) |
|------|---------|----------------|---------------------|
| Transformer (2017) | ReLU | 4.0× | 512→2048 |
| GPT-3 (2020) | GELU | 4.0× | 12288→49152 |
| LLaMA (2023) | SwiGLU | 2.69× (≈8/3) | 4096→11008 |
| LLaMA-3 (2024) | SwiGLU | 3.25× | 8192→26624 (405B 的 d 和 d_ff) |
| DeepSeek-V3 (2025) | SwiGLU | 2.57× | 7168→18432 |
| Qwen3 (2025) | SwiGLU | 2.57× | 5120→13184 (32B dense) |

**趋势**：SwiGLU 的三矩阵设计天然需要缩小 $d_{ff}$，2.5-2.7× 是最常见的比例。部分模型（如 LLaMA-3 405B）用更大比例以增加 FFN 容量。

### 4.6 FFN 选型决策树

```
需要设计 FFN？
├─ 模型 < 1B 参数？
│   └─ 用 SwiGLU (d_ff = 8d/3 取到 256 整数倍)
│       简单、高效、验证充分
│
├─ 模型 1-100B 参数 (Dense)?
│   └─ 用 SwiGLU，考虑 d_ff/d 比例：
│       - 参数预算紧：2.67× (标准 LLaMA)
│       - 参数预算充裕：3.0-3.5× (更大 FFN 容量)
│
└─ 模型 100B+ 参数？
    └─ 用 MoE + SwiGLU：
        - 专家数 64-256，Top-K = 2-8
        - 加共享专家 (DeepSeek 方案)
        - 辅助无损负载均衡
```

### 4.7 面试高频追问

**Q: 为什么所有 GLU 变体中 SwiGLU 最好？**

Swish ($x \cdot \sigma(x)$) 本身是一个自门控激活函数——它用自身值做门控。在 GLU 框架中再加一层外部门控，形成"双重门控"。Shazeer 的消融实验表明，Swish 作为门控激活函数时梯度流最稳定（Sigmoid 饱和问题轻微），同时非单调性保留了 GELU 的优势（允许小负值通过）。

**Q: SwiGLU 的 $d_{ff}$ 为什么是 $8d/3$ 而不是其他值？**

纯粹为了参数量对齐。标准 FFN 有 2 个矩阵，参数 = $2 \times d \times 4d = 8d^2$。SwiGLU 有 3 个矩阵，要保持总参数相同：$3 \times d \times d_{ff}' = 8d^2 \implies d_{ff}' = 8d/3$。然后向上取到硬件友好的整数倍（64/128/256），以最大化 GPU Tensor Core 利用率。

**Q: MoE FFN 中每个专家的 capacity 怎么设？**

Capacity factor $C$：每个专家在一个 batch 中最多处理 $C \times B/N$ 个 token（$B$ = batch token 总数，$N$ = 专家数）。$C=1.0$ 表示完全均匀分配，实践中 $C=1.05-1.25$ 留少量余量。超出 capacity 的 token 被 drop 或溢出到共享专家。DeepSeek-V3 的辅助无损均衡使实际负载接近均匀，$C$ 的敏感度大幅降低。

**Q: 为什么不用 ReLU²（Squared ReLU）？**

Primer (So et al., 2021) 和部分实验表明 $\text{ReLU}^2(x) = (\max(0,x))^2$ 在某些设置下优于 GELU。但 (1) ReLU² 输出值域无上界且增长更快，可能导致 FP16/BF16 溢出；(2) SwiGLU 的门控机制在大模型上更稳健；(3) ReLU² 没有在 >10B 模型上被充分验证。工业界倾向选择验证最充分的方案（SwiGLU），而非边际实验最优的方案。

---

## 5. 归一化演进

| 方案 | 位置 | 公式 | 参数量 | 训练稳定性 | 代表模型 |
|------|------|------|--------|-----------|---------|
| Post-LN | Attention/FFN 之后 | $\text{LN}(x + \text{SubLayer}(x))$ | $2d$ (γ, β) | 深层梯度不稳 | Transformer, BERT |
| Pre-LN | Attention/FFN 之前 | $x + \text{SubLayer}(\text{LN}(x))$ | $2d$ (γ, β) | 稳定 | GPT-2/3 |
| Pre-RMSNorm | Attention/FFN 之前 | $x + \text{SubLayer}(\text{RMS}(x))$ | $d$ (仅 γ) | 极稳定 | LLaMA, DeepSeek, Qwen |
| DeepNorm | 残差缩放 + Post-LN | $\text{LN}(\alpha x + \text{SubLayer}(x))$ | $2d$ + $\alpha$ | 深层极稳 | GLM-130B |

### 5.1 Post-Norm vs Pre-Norm

**Post-Norm（原始）**：$y = \text{LayerNorm}(x + \text{SubLayer}(x))$

- 梯度需要穿过 LayerNorm，深层时梯度方差爆炸。训练需要 warmup。

**Pre-Norm**：$y = x + \text{SubLayer}(\text{LayerNorm}(x))$

- **Why 改**：残差连接直通梯度，LayerNorm 只规范化子层输入。
- **效果**：训练更稳定，无需 warmup。GPT-2 开始采用，此后成为标准。

### 5.2 RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma
$$

- **Why 改**：LayerNorm 需要计算均值和方差（两次 reduction），RMSNorm 只计算 RMS（一次 reduction）。
- **改了什么**：去掉减均值和 bias $\beta$，只保留缩放 $\gamma$。
- **效果**：计算量减少约 7-10%，参数量减半（$d$ vs $2d$），质量无损。Zhang & Sennrich (2019) 证明 LayerNorm 的效果主要来自缩放不变性（RMS 部分），减均值贡献很小。

### 5.3 DeepNorm

$$
\text{DeepNorm}(x) = \text{LayerNorm}(\alpha \cdot x + \text{SubLayer}(x))
$$

- **Why 改**：Post-Norm 表达能力更强，能否通过残差缩放 $\alpha$ 恢复其训练稳定性？
- **改了什么**：残差分支乘以 $\alpha > 1$（如 $\alpha = (2L)^{1/4}$），子层权重初始化缩小为 $\beta = (8L)^{-1/4}$。
- **效果**：GLM-130B 成功用 DeepNorm + Post-LN 训练。但工程复杂度高，需要精心计算 $\alpha, \beta$，未被广泛采用。

### 5.4 QK-Norm 和附加归一化

现代模型还在特定位置添加额外归一化：
- **QK-Norm**：对 Q 和 K 在 head dimension 上做 RMSNorm，防止 attention logits 过大。Gemma-2、Cohere Command-R 使用。
- **为什么需要**：随着模型变深，Q/K 的幅度可能增长，导致 softmax 进入饱和区（所有权重集中在一个 token 上）。QK-Norm 是一种简单的修复。

> **为什么现代 LLM 全用 Pre-RMSNorm？** Pre 保证训练稳定性，RMS 减少计算开销。两者结合是 Pareto 最优，无理由用更复杂的方案。

---

## 6. 注意力效率优化

> KV Cache 优化详见 [[kv_cache_inference]]。

| 方案 | 复杂度 | 精确/近似 | 内存 | 适用场景 |
|------|--------|----------|------|---------|
| 标准 Attention | $O(n^2 d)$ | 精确 | $O(n^2)$ | 短序列 |
| Sparse Attention | $O(n \sqrt{n} \cdot d)$ | 近似 | $O(n\sqrt{n})$ | 长文本（GPT-3 Sparse） |
| Linear Attention | $O(nd^2)$ | 近似 | $O(nd)$ | $d \ll n$ 时有效 |
| FlashAttention | $O(n^2 d)$ | **精确** | $O(n)$ | **通用，当前标准** |
| FlashAttention-2 | $O(n^2 d)$ | **精确** | $O(n)$ | 并行度优化，2× faster |
| FlashAttention-3 | $O(n^2 d)$ | **精确** | $O(n)$ | H100 异步+FP8，1.5× faster |
| Ring Attention | $O(n^2 d / P)$ | 精确 | 分布在 $P$ 设备 | 超长序列跨设备 |
| Sliding Window | $O(n \cdot w \cdot d)$ | 近似 | $O(nw)$ | Mistral, 局部+稀疏全局 |

### 6.1 FlashAttention：IO-Aware Tiling

- **Why 改**：标准 attention 中 $n^2$ 的 attention matrix 必须写入 HBM（显存），内存和 IO 是瓶颈，不是计算。
- **改了什么**：将 Q/K/V 分 tile 加载到 SRAM（片上缓存），在 SRAM 中完成 softmax（online softmax 算法），**从不显式存储 $n^2$ attention matrix**。

```
标准 Attention 内存访问：
Q, K → HBM → 计算 QK^T → 写回 HBM (n² 大小) → 读 HBM → softmax → 写 HBM → 读 HBM + V → output

FlashAttention 内存访问：
Q block, K block, V block → SRAM → 计算 + online softmax + output block → 写回 HBM
```

- **效果**：计算量不变（仍是 $O(n^2 d)$），但内存从 $O(n^2)$ 降到 $O(n)$，实际速度提升 2-4×。

> **面试关键**：FlashAttention 不是近似方法，不改变计算结果。它改变的是**计算顺序**（tiling）和**内存访问模式**（IO-aware），利用 GPU 内存层级结构（SRAM >> HBM 带宽）。

### 6.2 FlashAttention-2 和 FlashAttention-3

**FlashAttention-2 改进**：
- 减少非 matmul FLOPs（softmax 中的 rescaling 操作）
- 优化并行策略：沿序列长度维度并行（FA1 沿 batch/head 并行），利用率更高
- 速度：A100 上达到理论 FLOPS 的 72%（FA1 约 50%）

**FlashAttention-3 改进**：
- 针对 H100 的 Tensor Core 异步流水线（WGMMA + TMA）
- 支持 FP8 attention（适配 DeepSeek-V3 的 FP8 训练）
- 低精度与高精度的混合策略：FP8 GEMM + FP32 accumulation

### 6.3 Ring Attention

- **Why 改**：超长序列（100K+）的 KV 放不进单张 GPU 的 HBM。
- **改了什么**：将序列分到多个设备，每个设备持有一段 Q 和对应的 KV。通过环形通信（ring topology），每个设备依次接收其他设备的 KV 块，完成分布式 attention。
- **效果**：序列长度可以线性 scale with 设备数。Gemini 1.5 Pro 的 1M 上下文使用类似技术。

### 6.4 Sliding Window Attention (Mistral)

- **Why 改**：长序列中远距离 attention 权重通常很小，是否可以限制每个 token 只看局部窗口？
- **改了什么**：每层只关注最近 $w$ 个 token（如 $w=4096$），通过多层堆叠实现 $L \times w$ 的有效感受野。
- **效果**：Mistral-7B 用 $w=4096$，32 层堆叠后有效感受野 131K。底层局部+高层全局的混合策略。

### 6.5 Sparse Attention 和 Linear Attention

**Sparse Attention (GPT-3 Sparse, Longformer, BigBird)**：
- 固定稀疏模式：局部窗口 + 全局 token + 随机连接
- 复杂度从 $O(n^2)$ 降到 $O(n\sqrt{n})$ 或 $O(n \log n)$
- 问题：固定模式可能错过关键的远距离依赖

**Linear Attention (Katharopoulos et al., 2020)**：
- 用核函数近似 softmax：$\text{Attn}(Q, K, V) \approx \phi(Q)(\phi(K)^T V)$
- 利用矩阵乘法结合律：先算 $\phi(K)^T V$（$d \times d$ 大小），复杂度 $O(nd^2)$
- 问题：近似质量差，softmax 的非线性特性难以完美替代

> **为什么 FlashAttention 赢了？** Sparse 和 Linear Attention 都是近似方法，有精度损失。FlashAttention 证明了：标准 attention 的瓶颈不在计算而在 IO，正确的优化方向是硬件感知的实现，而非数学上的近似。

---

## 7. 训练效率优化

### 7.1 混合精度训练

| 精度 | 位数 | 动态范围 | 代表 |
|------|------|---------|------|
| FP32 | 32 | $10^{±38}$ | 传统训练 |
| FP16 | 16 | $10^{±5}$ | 混合精度（需 loss scaling） |
| BF16 | 16 | $10^{±38}$ | **当前主流**（范围同 FP32） |
| FP8 (E4M3) | 8 | $10^{±9}$ | DeepSeek-V3 训练 |

- **Why BF16 而非 FP16**：FP16 动态范围小，梯度容易溢出需要 loss scaling。BF16 与 FP32 同范围，无需额外处理。
- **FP8 训练 (DeepSeek-V3)**：在 forward pass 用 FP8 计算 GEMM，显著减少计算量和内存。但需要细粒度量化（per-tile scaling）保持精度。

### 7.2 3D 并行

| 并行策略 | 切分维度 | 通信开销 | 适用场景 |
|----------|---------|---------|---------|
| DP (Data Parallel) | Batch | AllReduce 梯度 | 模型能放进单卡 |
| TP (Tensor Parallel) | 层内矩阵 | AllReduce 激活值 | 模型太大，单层需多卡 |
| PP (Pipeline Parallel) | 层间 | P2P 发送激活值 | 模型太深，减少通信 |
| ZeRO | 优化器状态/梯度/参数 | AllGather 参数 | 大内存需求 |

现代训练通常组合：TP(8 卡) × PP(8 节点) × DP(N)。DeepSeek-V3 用 TP=1（MLA 的隐向量维度小，单卡可计算）+ EP(Expert Parallel) + PP + DP。

### 7.3 MoE 稀疏化

> 详见 [[07_MoE架构与稀疏激活]]。

| 模型 | 年份 | 总参数 | 激活参数 | 专家数 | Top-K | 关键创新 |
|------|------|--------|---------|--------|-------|---------|
| Switch Transformer | 2022 | 1.6T | ~5B | 128 | 1 | **Top-1 路由 + 负载均衡 loss** |
| Mixtral-8x7B | 2024 | 47B | 13B | 8 | 2 | **GQA + 滑动窗口 + MoE** |
| DeepSeek-V3 | 2025 | 671B | 37B | 256+1 | 8 | **辅助无损负载均衡 + 共享专家** |
| Qwen3 | 2025 | 235B | 22B | 128 | 8 | MoE + 四阶段训练 |

**负载均衡的核心问题**：路由不均匀 → 部分专家过载（成为瓶颈）+ 部分专家饿死（浪费参数）。

**DeepSeek 的辅助无损负载均衡**：不用额外 loss 项（传统方法会损害模型质量），而是在路由分数上加可调偏置项，动态调整使各专家负载均匀。

### 7.4 Multi-Token Prediction (MTP)

- **Why 改**：标准 NTP 每个 forward 只预测下一个 token，训练信号利用率低。
- **改了什么 (DeepSeek-V3)**：每个位置同时预测未来 $k$ 个 token（$k=2$），多个预测头共享主干。

$$
\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{K} \lambda_k \cdot \mathcal{L}_{\text{NTP}}^{(k)}
$$

其中 $\mathcal{L}_{\text{NTP}}^{(k)}$ 是预测第 $k$ 个未来 token 的交叉熵。

- **效果**：训练时提升样本效率（每个 forward 获得 $K$ 倍训练信号）；推理时 MTP 头可作为推测解码的 draft model，加速 1.8×。
- **实现细节**：每个 MTP 头是一个浅层 Transformer block（1-2 层），接收主干的 hidden state 作为输入。头之间可以 causal chain（head $k$ 的预测作为 head $k+1$ 的输入），也可以独立预测。

### 7.5 Tokenizer 演进

| 方法 | 代表模型 | 词汇量 | 特点 |
|------|---------|--------|------|
| BPE | GPT-2/3 | 50K | 字节级 BPE，通用性强 |
| SentencePiece BPE | LLaMA | 32K | 多语言友好 |
| Tiktoken (BPE) | GPT-4 | 100K | 更大词汇表，中文效率↑ |
| BPE + 扩展 | LLaMA-3 | 128K | 大幅提升非英语效率 |

- **趋势**：词汇表越来越大（32K → 128K），因为更大词汇表 = 更短序列 = 更少推理步骤。
- **中文效率**：GPT-2 tokenizer 中一个汉字可能需要 2-3 tokens，LLaMA-3 的 128K 词汇表将中文效率提升 ~2×。

---

## 8. KV Cache 优化

> 基本原理和详细对比见 [[kv_cache_inference]]。

### 8.1 为什么 KV Cache 是推理瓶颈

自回归生成中，每个新 token 需要与所有之前 token 的 K/V 做 attention。KV Cache 避免重复计算，但内存 = $2 \times L \times h \times d_k \times n \times \text{sizeof(dtype)}$。

以 LLaMA-70B 为例：$L=80, h=64, d_k=128, n=4096, \text{FP16}$ → KV Cache = $2 \times 80 \times 64 \times 128 \times 4096 \times 2$ bytes = **10.7 GB**。单个请求就占满半张 A100。

### 8.2 优化方案全景

| 方案 | 层级 | 压缩比 | 精度损失 | 工程复杂度 |
|------|------|--------|---------|-----------|
| MQA/GQA | 架构级 | $h/g$ × | 微小（GQA）| 低（训练时决定） |
| MLA | 架构级 | $hd_k/d_c$ × (约 4-28×) | 无/微正 | 中（需 absorption trick）|
| PagedAttention | 系统级 | ~1×（消除碎片） | 无 | 中（vLLM 实现） |
| KV Cache 量化 | 压缩级 | 2-4× | 可控 | 低 |
| Prefix Caching | 复用级 | 共享前缀 | 无 | 低 |

**PagedAttention (vLLM)**：将 KV Cache 按页管理（类似 OS 虚拟内存），消除预分配浪费。不改变计算，只改变内存管理。

**KV Cache 量化**：将 FP16 KV 压缩到 INT4/INT8。per-channel 或 per-token 量化，配合校准数据。

**Prefix Caching**：多请求共享相同前缀（如 system prompt）的 KV Cache。vLLM 的 automatic prefix caching 自动检测共享前缀。

### 8.3 KV Cache 量化实战数据

| 量化方案 | 位宽 | 压缩比 | 质量影响 (PPL变化) | 适用场景 |
|----------|------|--------|-------------------|---------|
| FP16 (基线) | 16-bit | 1× | 0 | 精度优先 |
| INT8 per-channel | 8-bit | 2× | <0.1% PPL增加 | 通用推荐 |
| INT4 per-group | 4-bit | 4× | 0.3-1% PPL增加 | 高吞吐场景 |
| INT2 (KIVI) | 2-bit | 8× | 1-3% PPL增加 | 极端内存受限 |

实际部署中，INT8 KV Cache 量化已成为标配（几乎无损），INT4 在 batch size 较大时是合理选择。

### 8.4 Absorption Trick 详解（MLA 推理关键）

MLA 的挑战：推理时需要从 $c_t$ 还原 $K_t = W^{UK} c_t$，这个矩阵乘法抵消了缓存压缩的收益。

**Absorption 思路**：将上投影矩阵"吸收"进 Query 的计算中：

$$
Q_t^T K_t = Q_t^T (W^{UK} c_t) = (W^{UK^T} Q_t)^T c_t = \tilde{Q}_t^T c_t
$$

预计算 $\tilde{Q}_t = W^{UK^T} Q_t$（对 Q 做额外线性变换），然后直接用 $\tilde{Q}$ 和压缩的 $c_t$ 做 attention。

**限制**：RoPE 在 absorption 后需要特殊处理（因为 RoPE 作用在 Q/K 上，吸收后位置信息混入了 $W^{UK}$）。DeepSeek-V2 的解决方案是为 RoPE 分配额外的 decoupled dimensions。

> **MLA vs GQA 推理对比**：GQA 存 $2g d_k$ 维/层，MLA 存 $d_c$ 维/层。DeepSeek-V3 的 $d_c=512$ vs GQA-8 的 $2048$，MLA 省 4×。但 MLA 推理时需要上投影还原 K/V 或用 absorption trick 避免还原。

---

## 9. 现代 LLM 架构拆解对比

| 维度 | GPT-3 (2020) | LLaMA (2023) | LLaMA-3 (2024) | Mistral-7B (2023) | DeepSeek-V3 (2025) | Qwen3 (2025) |
|------|-------------|-------------|----------------|-------------------|-------------------|-------------|
| **参数量** | 175B | 7-65B | 8-405B | 7B | 671B (37B活) | 235B (22B活) |
| **Attention** | MHA | MHA | **GQA** | **GQA** | **MLA** | **GQA** |
| **位置编码** | 绝对可学习 | **RoPE** | **RoPE** | **RoPE** | **RoPE** | **RoPE** |
| **FFN** | GELU | **SwiGLU** | **SwiGLU** | **SwiGLU** | **SwiGLU** | **SwiGLU** |
| **Norm** | Pre-LN | **Pre-RMSNorm** | **Pre-RMSNorm** | **Pre-RMSNorm** | **Pre-RMSNorm** | **Pre-RMSNorm** |
| **MoE** | 否 | 否 | 否 | 否 | **是 (256+1专家)** | **是 (128专家)** |
| **上下文** | 2K | 2K→4K | **128K** | 32K (SW) | **128K** | **128K** |
| **训练数据** | 300B tok | 1.4T tok | **15T tok** | 未公开 | **14.8T tok** | **36T tok** |
| **训练精度** | FP16 | FP16 | BF16 | BF16 | **FP8** | BF16 |
| **关键创新** | In-Context | 开源+高效 | 数据Scaling | 滑动窗口 | MLA+MTP+辅助均衡 | 四阶段训练 |

> **趋势提炼**：
> 1. **架构趋同**：RoPE + SwiGLU + Pre-RMSNorm 成为标配，差异在 attention (GQA vs MLA) 和 density (Dense vs MoE)
> 2. **Scaling 策略分化**：Dense 模型靠数据 Scaling (LLaMA-3: 15T)，MoE 靠参数效率 (DeepSeek: 671B 总参但只激活 37B)
> 3. **训练效率前沿**：FP8 训练 (DeepSeek)、MTP (DeepSeek)、四阶段训练 (Qwen3)

### 9.1 各模型的独特设计选择

**LLaMA (2023) — 奠基者**：
- 将 Chinchilla Scaling Law 付诸实践：7B 模型用 1.4T tokens（Chinchilla 建议 ~140B）
- 第一个将 RoPE + SwiGLU + Pre-RMSNorm 组合推向主流的开源模型
- 证明了开源小模型可以匹敌闭源大模型

**Mistral-7B (2023) — 工程创新**：
- Sliding Window Attention 首次在主流模型中使用，兼顾效率和长上下文
- GQA with 8 KV heads（7B 模型中首次）
- Rolling buffer：KV Cache 固定大小 = 窗口大小，超出部分循环覆盖

**DeepSeek-V3 (2025) — 架构前沿**：
- **MLA**：打破 KV Cache 瓶颈，让 671B 参数的模型推理可承受
- **辅助无损负载均衡**：解决 MoE 训练中质量 vs 均衡的矛盾
- **MTP**：训练和推理双重收益
- **FP8 训练**：整体训练成本仅 \\\\\$5.57M（同级别模型的 1/10）
- **共享专家**：1 个始终激活的共享专家 + 256 个路由专家（Top-8），共享专家捕获通用知识

**Qwen3 (2025) — 训练策略创新**：
- 四阶段训练：Stage 1 预训练 → Stage 2 长上下文扩展 → Stage 3 思维模式训练 → Stage 4 通用 RL
- 支持"思维模式切换"：同一模型可以在 thinking/non-thinking 模式间切换
- 36T tokens 训练数据（截至 2025 最大）

---

## 10. 面试高频问题

### Q1: 为什么 Decoder-Only 成为主流而不是 Encoder-Decoder？

**答**：三个原因——(1) **统一性**：一个架构处理所有任务（生成、理解、翻译），无需为不同任务设计不同架构。(2) **Scaling 效率**：Encoder-Decoder 的参数分散在 encoder 和 decoder，Decoder-Only 集中参数更高效。(3) **In-Context Learning**：Decoder-Only 的 causal attention 自然支持 few-shot prompt，Encoder 的双向 attention 反而不利于序列生成。Google 的 T5 和 UL2 实验表明，在足够大的模型规模下 Decoder-Only 占优。

### Q2: RoPE vs ALiBi 核心区别？

**答**：ALiBi 在 attention score 上加线性衰减偏置，是一种"惩罚远距离"的机制；RoPE 通过旋转将相对位置信息编码进 Q/K 向量，让 attention 自己学习远近关系。RoPE 更灵活——不强制衰减，模型可以学到"远距离强相关"的模式（如 code 中的括号匹配）。实验中 RoPE 在 coding 和长上下文任务上显著优于 ALiBi。

### Q3: FlashAttention 为什么不改变计算结果？

**答**：FlashAttention 用 online softmax 算法（Milakov & Gimelshein, 2018），通过维护 running max 和 running sum，将 softmax 分块计算的结果与全局计算完全一致。数学上等价，改变的只是计算顺序和内存访问模式。核心是 tiling（分块加载到 SRAM）+ IO-awareness（最小化 HBM 读写）。

### Q4: MoE 的 load balancing 怎么做？

**答**：传统方法（Switch Transformer）在 loss 中加辅助负载均衡项 $\alpha \cdot \sum_i f_i \cdot P_i$，鼓励 router 均匀分配。缺点是辅助 loss 会损害主任务质量。DeepSeek-V3 的创新是**辅助无损负载均衡**：不加 loss 项，而是给每个专家维护一个偏置 $b_i$，周期性根据实际负载调整。负载高的专家降低 $b_i$，低的增加，实现动态平衡且不影响模型训练。另外引入**共享专家**（始终激活），处理通用知识避免专家间冗余。

### Q5: DeepSeek MLA 为什么比 GQA 更省内存？

**答**：GQA 按 head 粒度共享 KV，KV Cache 维度 = $2 \times g \times d_k$（如 $2 \times 8 \times 128 = 2048$）。MLA 将所有 head 的 KV 联合压缩到一个低秩隐向量 $c_t \in \mathbb{R}^{d_c}$（如 $d_c = 512$），Cache 维度 = $d_c = 512$。MLA 利用的是 KV 在 head 间的低秩结构，压缩更彻底。推理时需要 absorption trick 将上投影矩阵吸收进 Q 的投影，避免还原 K/V 的计算开销。

### Q6: Pre-Norm vs Post-Norm 对训练的影响？

**答**：Post-Norm 中梯度路径 = $\frac{\partial \text{LN}(x + f(x))}{\partial x}$，LayerNorm 的非线性导致深层梯度方差膨胀，需要 warmup 和精心调参。Pre-Norm 中梯度路径 = $1 + \frac{\partial f(\text{LN}(x))}{\partial x}$，残差连接提供恒等捷径，梯度直通。代价是 Pre-Norm 的理论表达能力略弱（Xiong et al., 2020），但在大模型中这个差距可忽略。DeepNorm 通过缩放残差 $\alpha$ 在 Post-Norm 中恢复稳定性，但工程复杂度高，未被广泛采用。

### Q7: 为什么用 RMSNorm 不用 LayerNorm？

**答**：Zhang & Sennrich (2019) 实验表明 LayerNorm 的效果主要来自 re-scaling（除以 RMS），re-centering（减均值）贡献极小。RMSNorm 去掉减均值和 bias $\beta$，计算量减少 ~10%，参数量减半（$d$ vs $2d$），但效果无损。在大规模训练中 10% 计算量 = 数百万美元成本差异。

### Q8: SwiGLU 相比 GELU 的优势？FFN 选型的演进逻辑？

**答**：FFN 经历了四代演进——ReLU FFN → GELU FFN → GLU 门控系列 (SwiGLU) → MoE 稀疏化。SwiGLU 将 FFN 拆为内容分支和门控分支，门控分支 $\text{Swish}(xW_g)$ 相当于逐维度的 soft attention，选择性放行信息，比 GELU 的逐元素激活表达力更强。Shazeer (2020) 系统测试了所有 GLU 变体（GLU/ReGLU/GeGLU/SwiGLU），SwiGLU 在同参数量下 loss 最低。代价是引入第三个矩阵 $W_g$，通过缩小 $d_{ff}$（从 $4d$ 到 $\frac{8}{3}d$）保持总参数量不变。第三代 MoE 进一步将单个 SwiGLU FFN 替换为多个专家 FFN + 路由器，用稀疏激活实现参数效率的量级提升。详见 Section 4。

### Q9: Scaling Law 对架构选择的指导？

**答**：Kaplan et al. (2020) 和 Hoffmann et al. (2022, Chinchilla) 的 Scaling Law 表明：(1) 性能主要取决于**计算量 C、数据量 D、参数量 N** 的 power law 关系，而非具体架构细节。(2) 在固定计算预算下，存在最优的 N-D 比。这意味着架构改进（如 SwiGLU、RoPE）的收益是常数级的"偏移量"，不改变 scaling 曲线的斜率。但这些常数在大规模训练中价值巨大——同样的 loss 可以少用 10-30% 的计算量。MoE 改变了 scaling 的性质：总参数 N 很大但激活参数小，用更少 FLOPs 达到同等能力。

### Q10: 如果让你设计一个新的 LLM 架构，你会做什么改进？

**答（开放题，展示思考深度）**：
1. **Attention 层**：采用 MLA 而非 GQA，进一步压缩 KV Cache，但研究如何消除 absorption trick 的限制（如不支持 RoPE 直接作用于压缩空间）。
2. **长上下文**：混合 Sliding Window（底层）+ 全局 Attention（高层）+ Mamba/SSM 层（超长依赖），形成层级注意力。
3. **训练效率**：MoE + MTP + FP8 已是前沿，下一步可能是 dynamic sparse training（训练中动态调整哪些参数活跃）。
4. **推理效率**：原生支持推测解码的架构设计——MTP 头不是附加的，而是架构的核心部分。
5. **模态统一**：一个 tokenizer + 一个 backbone 处理文本/图像/音频/视频，共享 attention 但模态间用 cross-attention 或 MoE 路由。

### Q11: 如何理解 Transformer 中的残差连接？

**答**：残差连接 $y = x + f(x)$ 有两层理解。(1) **梯度层面**：$\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$，恒等项 $I$ 保证梯度至少为 1，解决深层网络梯度消失。(2) **功能层面**：每一层只需要学习"残差函数" $f(x)$（即对输入的修正量），比直接学习目标映射更容易。Pre-Norm 将 LayerNorm 放在 $f$ 内部，进一步保证了残差分支的梯度通路不被 Norm 阻断。

### Q12: Decoder-Only 模型如何做理解任务（如分类）？

**答**：两种方式。(1) **Prompt + 生成**：将分类任务转化为生成任务，如 "Is this positive or negative? Answer: " → 模型生成 "positive"。(2) **取最后一个 token 的 hidden state** 作为序列表示，接分类头。方式 1 更灵活（zero-shot/few-shot），方式 2 更高效（fine-tune 场景）。关键洞察：Decoder 的 causal mask 虽然限制了每个 token 只能看到左边，但最后一个 token 能看到所有之前的 token，其 hidden state 包含完整序列信息。

---

## 附：关键论文索引

| 论文 | 年份 | 贡献 |
|------|------|------|
| Attention Is All You Need | 2017 | Transformer 架构 |
| Improving Language Understanding by GPT | 2018 | Decoder-Only 预训练 |
| BERT | 2018 | Encoder-Only + MLM |
| Language Models are Unsupervised Multitask Learners (GPT-2) | 2019 | Pre-Norm + 更大规模 |
| Language Models are Few-Shot Learners (GPT-3) | 2020 | In-Context Learning |
| RoFormer (Su et al.) | 2021 | RoPE |
| FlashAttention (Dao et al.) | 2022 | IO-aware tiling |
| Training Compute-Optimal LLMs (Chinchilla) | 2022 | Scaling Law 修正 |
| LLaMA (Touvron et al.) | 2023 | 开源高效 LLM |
| GQA (Ainslie et al.) | 2023 | Grouped-Query Attention |
| Mistral 7B | 2023 | Sliding Window + GQA |
| GLU Variants (Shazeer) | 2020 | SwiGLU |
| DeepSeek-V2/V3 | 2024/2025 | MLA + MoE + MTP |
| Qwen3 Technical Report | 2025 | MoE + 四阶段训练 |

---

> **本文定位**：架构改进全景对比，聚焦「每个改进的 Why/What/Effect」。
> - 原始 Attention 机制+搜广推应用 → [[attention_transformer]]
> - KV Cache 推理优化详解 → [[kv_cache_inference]]
> - 预训练里程碑+Scaling 策略 → [[08_LLM预训练与架构演进]]
> - MoE 架构详解 → [[07_MoE架构与稀疏激活]]
> - Attention 在搜广推的应用 → [[attention_in_recsys]]
> - 激活函数基础 → [[activation_functions]]