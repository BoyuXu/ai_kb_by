# MoE-LLaMA: Mixture-of-Experts for Efficient LLM Serving

> 来源：arxiv | 日期：20260322 | 领域：LLM工程

## 问题定义

Dense LLM（如 LLaMA-70B）推理时所有参数都参与计算，单请求计算量大，多请求 batching 受显存限制。MoE（Mixture of Experts）架构通过稀疏激活（每次只激活部分专家）显著提升参数效率，但工程部署复杂（专家路由、负载均衡、通信开销）。

## 核心方法与创新点

- **MoE 替换 FFN**：将 LLaMA 的 FFN 层替换为 MoE 层（N 个专家，每个 token 激活 top-K 个专家）
- **专家数量配置**：
  - Total Experts E = 8/16/64（参数量等比增长）
  - 激活专家 K = 2（每个 token 激活 2 个专家，计算量接近 1 个 Dense FFN）
  - 等效参数量：E 个专家 × 1/K 激活 ≈ E/K 倍参数量，计算量不变
- **路由机制**：
  - Top-K Gating：softmax(W_gate × x) 选 top-K 专家
  - Auxiliary Loss（负载均衡）：防止专家崩塌（所有 token 路由到同一个专家）
- **工程优化**：
  - Expert Parallelism：不同专家分布在不同 GPU，All-to-All 通信
  - Expert Offloading：冷门专家卸载到 CPU，热门专家常驻 GPU
  - FP8 专家权重：专家参数量化，显存减少 4×

## 实验结论

- MoE-LLaMA（8 experts, K=2）vs Dense-LLaMA：等计算量下，MMLU 提升 3.2%，代码 HumanEval 提升 4.8%
- 推理吞吐：相同参数量下，MoE 吞吐是 Dense 的 2.3×（稀疏激活减少 FLOPS）
- Expert Offloading：在单 4090 GPU（24GB）服务 70B MoE 模型，延迟 <200ms（可接受）
- 负载均衡损失有效：专家利用率方差从 0.15 降至 0.04

## 工程落地要点

- **专家数量选择**：E=8 是工程和效果的平衡点，E>16 All-to-All 通信开销显著增加
- **Expert Parallelism 配置**：建议 E 整除 GPU 数，每个 GPU 承载相同数量的专家
- **路由策略**：Token Choice（token 选专家）vs Expert Choice（专家选 token）——后者负载均衡更好但实现复杂
- **监控**：生产环境需监控每个专家的激活频率分布，防止专家崩塌影响效果

## 常见考点

1. **Q：MoE 的"参数量大但计算量小"是如何实现的？**
   A：MoE 有 E 个专家但每个 token 只激活 top-K 个（通常 K=2），实际参与计算的 FFN 权重只有 K/E 的比例。总参数量 = E × expert_size，计算量 = K × expert_size（与单专家接近）

2. **Q：专家负载不均衡为什么是问题？如何解决？**
   A：不均衡导致部分专家过载（成为瓶颈）、部分专家空闲（参数浪费）；工程上热点专家 GPU 成为计算瓶颈。解决：辅助损失（鼓励均匀分配）；Expert Choice 路由；动态专家容量

3. **Q：MoE 和 Dense LLM 的 serving 架构有什么核心区别？**
   A：Dense LLM：Tensor Parallelism（切参数矩阵）；MoE：Expert Parallelism（切专家，需 All-to-All token dispatch/gather）。MoE 的通信模式更复杂，需要专门的调度系统（如 DeepSpeed-MoE）
