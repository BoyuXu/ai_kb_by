# LLM 预训练：从 GPT 到 Llama 到 DeepSeek 的技术演进

> 📚 参考文献
> - [Moe-Llama-Mixture-Of-Experts-For-Efficient-Larg...](../papers/daily/20260321_moe-llama-mixture-of-experts-for-efficient-large-language-model-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Moe-Llama Mixture-Of-Experts For Efficient Larg...](../papers/daily/20260323_moe-llama_mixture-of-experts_for_efficient_large_la.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../papers/daily/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Moe-Llama-Mixture-Of-Experts-Efficient-Llm-Serving](../papers/daily/20260321_moe-llama-mixture-of-experts-efficient-llm-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Llama 3 The Llama 3 Herd Of Models](../papers/daily/20260323_llama_3_the_llama_3_herd_of_models.md) — LLaMA 3: The Llama 3 Herd of Models
> - [Llmorbit-A-Circular-Taxonomy-Of-Large-Language-...](../papers/daily/20260321_llmorbit-a-circular-taxonomy-of-large-language-models-from-scaling-walls-to-agentic-ai-systems.md) — LLMOrbit: A Circular Taxonomy of Large Language Models fr...
> - [Moe Llama Mixture Of Experts](../papers/daily/20260322_moe_llama_mixture_of_experts.md) — MoE-LLaMA: Mixture-of-Experts for Efficient LLM Serving
> - [Kvcache Compression For Long-Context Llm Infere...](../papers/daily/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...

> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：GPT 系列, Llama 系列, DeepSeek, Chinchilla Scaling Law

## 📐 核心公式与原理

### 1. Self-Attention

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- Transformer 核心计算

### 2. KV Cache

$$
\text{Memory} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times dtype\_size
$$

- KV Cache 内存占用公式

### 3. LoRA

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

- 低秩适配，r << d 大幅减少可训练参数

---

## 🎯 核心洞察（4条）

1. **Scaling Law 指导模型大小和数据量的分配**：Chinchilla 证明最优配比是模型参数 N 和训练 token 数 D 按 1:20 的比例增长
2. **Decoder-Only 成为主流架构**：GPT 证明自回归预训练 + 指令微调是最通用的范式，Encoder-Decoder（T5）和 Encoder-Only（BERT）退居特定场景
3. **数据质量决定模型上限**：Llama 3 用了 15T tokens 的高质量数据（代码、数学、多语言），DeepSeek 强调了数据去重和质量过滤的重要性
4. **训练效率的工程创新不亚于模型创新**：3D 并行（Data + Tensor + Pipeline）、混合精度训练、梯度检查点等技术使万亿参数模型训练成为可能

---

## 🎓 常见考点（5条）

### Q1: Transformer Decoder-Only 的预训练目标？
**30秒答案**：Causal Language Modeling（CLM）——给定前 n 个 token，预测第 n+1 个 token。Loss = -Σ log P(x_t | x_{<t})。Causal mask 确保每个位置只能看到之前的 token，不能看到未来。

### Q2: Chinchilla Scaling Law 的核心发现？
**30秒答案**：给定固定计算预算，模型参数 N 和训练 token 数 D 应该等比增长（N ∝ D）。之前的 GPT-3 等模型"参数过大、数据不足"——175B 参数只用了 300B tokens。Chinchilla 70B + 1.4T tokens 效果更好。

### Q3: 3D 并行怎么分工？
**30秒答案**：①Data Parallelism（数据并行）：相同模型副本处理不同数据，梯度 AllReduce 同步；②Tensor Parallelism（张量并行）：单层的矩阵乘法分到多张 GPU；③Pipeline Parallelism（流水线并行）：不同层放在不同 GPU，micro-batch 流水线执行。

### Q4: 混合精度训练的原理？
**30秒答案**：前向和反向用 FP16/BF16（减少显存+加速计算），梯度累积和参数更新用 FP32（保持精度）。关键技术：Loss Scaling 放大 loss 防止 FP16 下溢。BF16 比 FP16 更好（指数位更多，不容易溢出）。

### Q5: LLM 预训练数据怎么处理？
**30秒答案**：①去重（MinHash/SimHash 去近似重复文档）；②质量过滤（基于 perplexity 或分类器打分）；③敏感信息去除（PII 脱敏）；④比例调整（代码/数学/英语/中文按最优比例混合）。

---

### Q6: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q7: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q8: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q9: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。

### Q10: MoE 的优势和挑战？
**30秒答案**：优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。
## 🌐 知识体系连接

- **上游依赖**：Transformer 架构、分布式训练、数据工程
- **下游应用**：SFT/RLHF 对齐、推理部署、下游微调
- **相关 synthesis**：LLM对齐方法演进.md, MoE架构设计.md


## 📐 核心公式直观理解

### 公式 1：Chinchilla Scaling Law

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$

- $N$：模型参数量
- $D$：训练数据 token 数
- $A, B, \alpha, \beta$：拟合常数
- $L_\infty$：不可约损失（entropy of natural text）

**直观理解**：预训练 loss 由两项决定——模型太小（第一项大）或数据太少（第二项大）。Chinchilla 的核心发现是 $N$ 和 $D$ 应该等比例增长：参数翻倍时数据也必须翻倍，否则就是浪费算力训一个"吃不饱"或"消化不了"的模型。

### 公式 2：Next Token Prediction 损失

$$
\mathcal{L}_{\text{NTP}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

- $T$：序列长度
- $x_t$：第 $t$ 个 token
- $\theta$：模型参数

**直观理解**：预训练的唯一目标——给定前面所有 token，预测下一个 token。这个看似简单的目标之所以强大，是因为预测下一个词需要理解语法、语义、世界知识、推理能力——你必须"理解"一段话才能续写。

### 公式 3：学习率调度（Cosine Schedule）

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

**直观理解**：学习率从大到小余弦衰减——开始大步探索（快速降 loss），后期小步精调（避免震荡）。Warmup 阶段线性增大学习率是为了让 Adam 的动量估计稳定，避免初始梯度不准时走偏。

