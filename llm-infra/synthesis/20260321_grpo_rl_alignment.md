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
