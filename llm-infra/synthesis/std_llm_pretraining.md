# LLM 预训练：从 GPT 到 Llama 到 DeepSeek 的技术演进

> 📚 参考文献
> - [Moe-Llama-Mixture-Of-Experts-For-Efficient-Larg...](../../llm-infra/20260321_moe-llama-mixture-of-experts-for-efficient-large-language-model-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Moe-Llama Mixture-Of-Experts For Efficient Larg...](../../llm-infra/20260323_moe-llama_mixture-of-experts_for_efficient_large_la.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Moe-Llama-Mixture-Of-Experts-Efficient-Llm-Serving](../../llm-infra/20260321_moe-llama-mixture-of-experts-efficient-llm-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Llama 3 The Llama 3 Herd Of Models](../../llm-infra/20260323_llama_3_the_llama_3_herd_of_models.md) — LLaMA 3: The Llama 3 Herd of Models
> - [Llmorbit-A-Circular-Taxonomy-Of-Large-Language-...](../../llm-infra/20260321_llmorbit-a-circular-taxonomy-of-large-language-models-from-scaling-walls-to-agentic-ai-systems.md) — LLMOrbit: A Circular Taxonomy of Large Language Models fr...
> - [Moe Llama Mixture Of Experts](../../llm-infra/20260322_moe_llama_mixture_of_experts.md) — MoE-LLaMA: Mixture-of-Experts for Efficient LLM Serving
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...


> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：GPT 系列, Llama 系列, DeepSeek, Chinchilla Scaling Law

---

## 🎯 核心洞察（4条）

1. **Scaling Law 指导模型大小和数据量的分配**：Chinchilla 证明最优配比是模型参数 N 和训练 token 数 D 按 1:20 的比例增长
2. **Decoder-Only 成为主流架构**：GPT 证明自回归预训练 + 指令微调是最通用的范式，Encoder-Decoder（T5）和 Encoder-Only（BERT）退居特定场景
3. **数据质量决定模型上限**：Llama 3 用了 15T tokens 的高质量数据（代码、数学、多语言），DeepSeek 强调了数据去重和质量过滤的重要性
4. **训练效率的工程创新不亚于模型创新**：3D 并行（Data + Tensor + Pipeline）、混合精度训练、梯度检查点等技术使万亿参数模型训练成为可能

---

## 🎓 面试考点（5条）

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

## 🌐 知识体系连接

- **上游依赖**：Transformer 架构、分布式训练、数据工程
- **下游应用**：SFT/RLHF 对齐、推理部署、下游微调
- **相关 synthesis**：std_llm_alignment_evolution.md, std_llm_moe_architecture.md
