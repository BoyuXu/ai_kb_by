# MoE-LLaMA: Mixture-of-Experts for Efficient Large Language Model Serving
> 来源：https://arxiv.org/search/?query=MoE+LLaMA+mixture+experts+efficient+serving&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
密集型LLM（Dense LLM）推理时激活全部参数，计算效率低。MoE（Mixture-of-Experts）允许每次只激活部分参数，在保持模型容量的同时减少推理计算量。本文研究MoE在LLaMA架构上的高效serving优化。

## 核心方法与创新点
- MoE FFN层：将前馈网络替换为多个专家（Expert）+ Router
- Top-K路由：每个token选择K个最相关专家（通常K=2）
- 专家并行：不同专家可以在不同GPU上并行计算
- 负载均衡：router加auxiliary loss，避免专家使用不均衡

## 实验结论
MoE-LLaMA相比同参数Dense-LLaMA，推理FLOPs减少约60%（只激活2/8个专家）；在相同推理预算下可以使用4x大的MoE模型；混合精度量化后，8x7B MoE可在单A100上serving。

## 工程落地要点
- MoE推理的主要瓶颈是专家路由的all-to-all通信，需要高带宽互联（NVLink）
- 专家负载不均衡会导致长尾延迟，需要动态负载均衡
- 在批量推理时，不同token路由到不同专家，batch不规则，需要特殊处理

## 常见考点
1. **Q: MoE的基本原理？** A: N个专家FFN + Router；Router根据token生成gate权重；选Top-K专家计算加权和
2. **Q: MoE相比Dense的优劣？** A: 优：同FLOPs下参数量更大（容量更大）；劣：all-to-all通信开销、负载均衡复杂
3. **Q: 负载均衡辅助损失（auxiliary loss）如何设计？** A: 最小化专家使用分布的方差，让所有专家均等使用；通常权重约0.01
4. **Q: 专家并行（Expert Parallelism）的原理？** A: 不同专家分布在不同GPU，token的专家路由变成跨GPU的all-to-all通信
5. **Q: DeepSeek-V2/V3中MoE的创新？** A: Fine-grained expert（专家更多更小）、Shared expert（所有token共享的常规专家）
