# GRPO：让大模型自己学会推理的 RL 算法

> 📚 参考文献
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Grpo-Group-Relative-Policy-Optimization-Llm-Rea...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-llm-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Flashattention-3-Fast-And-Accurate-Attention-Fo...](../../llm-infra/20260321_flashattention-3-fast-and-accurate-attention-for-llms-on-next-gen-accelerators.md) — FlashAttention-3: Fast and Accurate Attention for LLMs on...
> - [Vllm Efficient Memory Management For Large Lang...](../../llm-infra/20260323_vllm_efficient_memory_management_for_large_language.md) — vLLM: Efficient Memory Management for Large Language Mode...


**一句话**：GRPO 让模型对同一道题做多次尝试，通过"对答案们打相对分"来学习，不再需要单独养一个打分模型。

**类比**：你学数学时，老师不是单独判断"这次比昨天进步了多少"（需要记住你历史表现的 Critic），而是把你班同学的答案都摊开比一比——你这次比班里平均水平好多少，就给多少奖励。组内排名就是 advantage。

**核心机制**（5步）：
1. 对同一道题（如数学推理），让模型生成 G=8 个不同回答
2. 用规则打分（答对=1, 格式对=0.1, 否则=0）得到 G 个 reward
3. 计算组内均值和标准差，归一化得到每个回答的 advantage
4. 用 PPO 的 clip 目标更新策略，保留 KL 惩罚防止模型"走偏"
5. 无需 Critic 网络 → 节省 50% 显存，收敛更稳定

**和 PPO 的区别**：
| 维度 | PPO | GRPO |
|------|-----|------|
| 需要 Critic | ✅ 需要（同量级模型）| ❌ 不需要 |
| Advantage 来源 | 时序差分 V 函数估计 | 组内相对 reward |
| 显存需求 | 2x | 1x |
| 适用任务 | 通用对齐 | 可验证任务（数学/代码）|
| 方差 | 较高（单次 V 估计不准） | 较低（G 次统计更稳定）|

**工业常见做法**：
- 先 SFT 冷启动（让模型至少能生成合理格式），再 GRPO fine-tune
- G 通常取 8-16；太小方差高，太大计算贵
- 温度设 0.7-1.0：需要多样性，否则 G 个答案全对/全错，方差为 0，梯度消失
- 用 vLLM 并行采样 G 个回答，加速采样阶段
- 监控 KL divergence（>0.1 需降 LR）和组内 reward 方差（接近 0 = 题太简单/难）

**面试考点**：
- Q: GRPO 为何特别适合数学/代码任务？ → 有可验证 reward（规则判对错），无需人工标注偏好
- Q: DeepSeek-R1-Zero 为何自发产生 CoT？ → GRPO 优化压力下发现"先想再答"答对率更高，reward 更高，行为被强化稳定涌现
- Q: GRPO 的 advantage 归一化为什么有效？ → 消除 reward scale 影响，只保留相对优劣信号，类似 batch normalization 的稳定效果

**演进脉络**：`REINFORCE (1992) → PPO (2017, 带 Critic + clip) → GRPO (2024, 去 Critic + 组内对比)`，核心驱动：LLM 训练的显存成本越来越贵，GRPO 用数学技巧绕开了 Critic。


## 📐 核心公式与原理

### 1. Self-Attention
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Transformer 核心计算

### 2. KV Cache
$$\text{Memory} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times dtype\_size$$
- KV Cache 内存占用公式

### 3. LoRA
$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$
- 低秩适配，r << d 大幅减少可训练参数

### Q1: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q2: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q3: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q4: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。

### Q5: MoE 的优势和挑战？
**30秒答案**：优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。

### Q6: PagedAttention（vLLM）的核心思想？
**30秒答案**：借鉴操作系统虚拟内存分页，将 KV Cache 切分为固定大小的「页」，按需分配。解决传统方式预分配最大序列长度导致的内存浪费（平均浪费 60-80%）。

### Q7: Continuous Batching 是什么？
**30秒答案**：传统 Static Batching 等最长序列完成才处理下一批。Continuous Batching 每个 token step 都可以加入新请求，序列完成立即释放。将 GPU 利用率从 ~30% 提升到 ~80%。

### Q8: GRPO 和 PPO 的核心区别？
**30秒答案**：PPO 需要 value network 估计 advantage；GRPO 用 group 内的相对奖励替代 value network：采样 G 个输出，用组内排名作为 baseline。更简单、更稳定、不需要额外模型。

### Q9: RAG vs Fine-tuning 怎么选？
**30秒答案**：RAG：知识频繁更新、需要引用来源、不想改模型。Fine-tuning：任务固定、需要特定风格/格式、追求最低延迟。两者可结合：fine-tune 后的模型 + RAG 检索。

### Q10: LLM 推理的三大瓶颈？
**30秒答案**：①Prefill 阶段：计算密集（大量矩阵乘）；②Decode 阶段：内存密集（KV Cache 读写）；③通信：多卡推理时的 AllReduce。优化方向：FlashAttention（①）、PagedAttention（②）、TP/PP 并行（③）。
