# LLM 预训练与架构演进：从 GPT 到 LLaMA 到 DeepSeek

> 综合自 2 篇 synthesis | 更新：2026-04-13 | 领域：LLM 预训练/架构
> 关联：[[concepts/attention_in_recsys]] | [[07_MoE架构与稀疏激活]]

---

## 预训练里程碑

| 维度 | GPT-3 (2020) | LLaMA (2023) | LLaMA-3 (2024) | DeepSeek-V3 (2025) | Qwen3 (2025) |
|------|-------------|-------------|----------------|-------------------|-------------|
| 参数量 | 175B | 7-65B | 8-405B | 671B (37B 激活) | 235B (22B 激活) |
| 架构 | Dense | Dense + RoPE + SwiGLU | Dense + GQA | MoE + MLA | MoE |
| 训练数据 | 300B tokens | 1.4T tokens | **15T tokens** | 14.8T tokens | - |
| 位置编码 | 绝对位置 | RoPE | RoPE | RoPE | RoPE |
| 注意力 | MHA | MHA | GQA | MLA（低秩 KV） | GQA |
| 开源 | 否 | 是 | 是 | 是 | 是 |
| Scaling 策略 | 参数优先 | Chinchilla 最优 | 数据+参数 | MoE 稀疏 Scaling | 四阶段训练 |

---

## 一、核心公式

### Next Token Prediction（预训练目标）

$$
\mathcal{L}_{\text{NTP}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

**直觉**：预测下一个词需要理解语法、语义、世界知识、推理能力——你必须"理解"一段话才能续写。Causal mask 确保每个位置只能看到之前的 token。

### Chinchilla Scaling Law

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$

- $N$：模型参数量
- $D$：训练数据 token 数
- $L_\infty$：不可约损失（自然文本熵）

**核心发现**：$N$ 和 $D$ 应等比增长。GPT-3 "参数过大、数据不足"（175B 参数只用 300B tokens），Chinchilla 70B + 1.4T tokens 效果更好。

### Cosine 学习率调度

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

开始大步探索（快速降 loss），后期小步精调。Warmup 线性增大学习率让 Adam 动量估计稳定。

---

## 二、架构演进

### Decoder-Only 成为主流

| 架构 | 代表 | 适用 |
|------|------|------|
| Encoder-Only | BERT | 判别任务（分类/NER） |
| Encoder-Decoder | T5 | 翻译/摘要 |
| **Decoder-Only** | **GPT/LLaMA** | **通用生成（最通用）** |

### 关键架构组件演进

```
GPT-3 (2020): 标准 Transformer + 绝对位置编码 + MHA
  → LLaMA (2023): + RoPE + SwiGLU + RMSNorm + Pre-Norm
    → LLaMA-3 (2024): + GQA + 15T tokens + 多语言
      → DeepSeek-V2 (2024): + MLA + MoE
        → DeepSeek-V3 (2025): + 256 精细 Expert + 共享 Expert
```

### RoPE（旋转位置编码）

相对位置编码，通过旋转矩阵编码 token 间相对距离，支持外推到更长序列。LLaMA 后成为标配。

### GQA（Grouped Query Attention）

$$
\text{KV}_{GQA} = \frac{G}{H} \times \text{KV}_{MHA}
$$

$G$ 个 KV head 被 $H$ 个 Q head 共享。$G = H/4$ 时精度损 <0.5%，工业最优解。

### SwiGLU（门控激活函数）

替代 ReLU/GELU，在 FFN 中引入门控机制，效果更好（LLaMA 标配）。

---

## 三、3D 并行训练

| 维度 | 分工 | 说明 |
|------|------|------|
| Data Parallelism | 相同模型副本处理不同数据 | 梯度 AllReduce 同步 |
| Tensor Parallelism | 单层矩阵乘法分到多 GPU | 节点内通信 |
| Pipeline Parallelism | 不同层放不同 GPU | micro-batch 流水线 |

### 混合精度训练

前向/反向用 FP16/BF16（减少显存+加速），梯度累积和参数更新用 FP32（保持精度）。BF16 比 FP16 好（指数位更多，不易溢出）。Loss Scaling 放大 loss 防 FP16 下溢。

---

## 四、预训练数据工程

1. **去重**：MinHash/SimHash 去近似重复文档
2. **质量过滤**：基于 perplexity 或分类器打分
3. **敏感信息去除**：PII 脱敏
4. **比例调整**：代码/数学/英语/中文按最优比例混合

**数据质量决定模型上限**：LLaMA-3 用 15T tokens 高质量数据（代码+数学+多语言），DeepSeek 强调数据去重和质量过滤。

---

## 五、核心洞察

1. **Scaling Law 指导资源分配**：模型参数 N 和训练 token 数 D 按 1:20 比例增长
2. **Decoder-Only 是最通用架构**：自回归预训练 + 指令微调
3. **数据质量 >> 数据数量**：高质量去重数据是关键
4. **训练效率工程不亚于模型创新**：3D 并行、混合精度、梯度检查点使万亿参数训练可能
5. **MoE 稀疏 Scaling 成为新趋势**：DeepSeek-V3 证明稀疏激活可将训练成本降至 Dense 的 1/3

---

## 面试高频 Q&A

### Q1: Decoder-Only 的预训练目标？
**30秒**：Causal Language Modeling——给定前 n 个 token 预测第 n+1 个。Loss = $-\sum \log P(x_t | x_{<t})$。Causal mask 保证只看过去不看未来。

### Q2: Chinchilla Scaling Law 的核心发现？
**30秒**：给定固定计算预算，$N$ 和 $D$ 应等比增长。GPT-3 "参数过大、数据不足"。Chinchilla 70B + 1.4T tokens 效果更好。最优比例约 1:20（参数:数据）。

### Q3: 3D 并行怎么分工？
**30秒**：Data Parallelism 处理不同数据（梯度 AllReduce）；Tensor Parallelism 切矩阵乘法到多 GPU；Pipeline Parallelism 不同层放不同 GPU（micro-batch 流水线）。

### Q4: 混合精度训练原理？
**30秒**：前向/反向 FP16/BF16（省显存+加速），梯度累积/更新 FP32（保精度）。BF16 优于 FP16（指数位多不溢出）。Loss Scaling 防下溢。

### Q5: 预训练数据怎么处理？
**30秒**：去重（MinHash）→ 质量过滤（perplexity/分类器）→ PII 脱敏 → 比例混合（代码/数学/多语言按最优配比）。

### Q6: RoPE 相比绝对位置编码的优势？
**30秒**：编码相对距离而非绝对位置，自然支持外推到更长序列。通过旋转矩阵实现，计算高效。LLaMA 后成为标配。

### Q7: GQA 相比 MHA 和 MQA 的权衡？
**30秒**：MHA KV Cache 大；MQA（$G=1$）压缩最大但精度损 ~2%；GQA（$G=H/4$）精度损 <0.5% 且 KV Cache 降 4x，工业最优解。LLaMA-3/Mistral 标配。

### Q8: 为什么 Decoder-Only 成为主流？
**30秒**：自回归预训练目标最简单通用（只需预测下一个词），天然支持生成，In-context Learning 能力强。Encoder-Decoder 在翻译/摘要仍有优势，但通用性不如 Decoder-Only。

---

## 记忆助手

- **预训练 = 大量阅读**：模型读了万亿 token 后理解了语言规律，预测下一个词需要理解一切
- **Scaling Law = 节食法则**：模型和数据要均衡增长，"大胃口小饭量"或"小胃口大饭量"都浪费
- **3D 并行 = 流水线工厂**：DP 复制工人，TP 拆任务，PP 排流水线
- **RoPE = 旋转编码**：不记绝对位置，记相对距离，天然支持外推
- **数据质量口诀**：去重→过滤→脱敏→配比

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[07_MoE架构与稀疏激活|MoE 稀疏 Scaling]]
- [[01_LLM推理优化全景|推理优化与预训练架构]]
