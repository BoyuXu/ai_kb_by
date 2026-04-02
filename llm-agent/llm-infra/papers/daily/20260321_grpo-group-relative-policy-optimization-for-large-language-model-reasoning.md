# GRPO: Group Relative Policy Optimization for Large Language Model Reasoning

> 来源：arxiv (DeepSeek) | 日期：20260321 | 领域：llm-infra

## 问题定义

训练 LLM 的推理能力（Reasoning）需要强化学习（RL）来优化可验证的奖励（数学答案正确与否）。传统 PPO（Proximal Policy Optimization）用于 LLM 的问题：

1. **需要独立 Critic 网络**：PPO 需要与 Policy 同等大小的 Critic/Value 网络估计状态价值函数，内存开销翻倍
2. **训练不稳定**：Critic 的估计误差会影响 Policy 更新的方向
3. **长序列奖励稀疏**：数学推理中奖励只在最终答案处给出，中间推理步骤的奖励估计困难

GRPO（Group Relative Policy Optimization）是 DeepSeek-Math 和 DeepSeek-R1 使用的核心 RL 算法，通过组相对优势消除了 Critic 网络。

## 核心方法与创新点

1. **组内相对优势（Group Relative Advantage）**：
   - 对同一问题采样 G 个回答（a group）
   - 用组内各回答的奖励均值作为基线（Baseline）
   - 优势函数：`A_i = (r_i - mean(r_1,...,r_G)) / std(r_1,...,r_G)`
   - 不需要独立 Critic，直接用组内相对排名作为价值估计

2. **GRPO 目标函数**：
   ```
   L_GRPO = E[Σᵢ min(ratio_i × A_i, clip(ratio_i, 1-ε, 1+ε) × A_i)]
             - β × KL(π_θ || π_ref)
   ```
   其中 `ratio_i = π_θ(a_i|q) / π_old(a_i|q)`，clip 防止策略更新过大，KL 防止偏离参考模型（SFT 模型）

3. **奖励设计**：
   - **准确性奖励**：答案格式正确 +0.1，数学验证正确 +1.0
   - **格式奖励**：要求使用 `<think>...</think>` 标签分离推理过程
   - **长度惩罚**（可选）：防止模型生成过长的无意义"推理"

4. **与 PPO 对比**：
   - GRPO：无需 Critic，每组 G 个样本的均值作为 baseline，内存节省 ~50%
   - PPO：Critic 网络需要单独训练，估计精度更高但代价更大
   - GRPO 在数学推理场景几乎与 PPO 等效，但训练更稳定

## 实验结论

- DeepSeek-R1 使用 GRPO：在 AIME 2024 数学竞赛准确率 **79.8%**（超越 GPT-4o 的 74.6%）
- 相比 SFT 基线：GRPO 训练后数学推理能力 **+25-35%**（取决于任务难度）
- 训练效率：无 Critic 模式下 GPU 内存使用降低 **~40%**，可在相同资源下训练更大 batch
- 涌现行为：GRPO 训练自然产生"自我修正"（self-verification）行为，无需显式训练

## 工程落地要点

1. **组大小 G 的选择**：G 越大基线估计越稳定，但计算量线性增加。实践中 G=8-16 是常见配置，平衡稳定性和效率
2. **奖励 Shaping**：稀疏奖励（只有最终答案）训练难，需要引入过程奖励（PRM）或格式奖励帮助训练信号传播
3. **参考模型更新**：KL 正则中的参考模型通常固定（不随策略更新），定期更新（类似 Target Network）有时能提升效果
4. **混合精度**：GRPO 的 forward（生成 G 个样本）部分用 BF16，梯度计算用 FP32，防止数值不稳定

## 常见考点

- Q: GRPO 和 PPO 的主要区别是什么？
  A: PPO 使用独立的 Critic/Value 网络估计优势函数；GRPO 对同一问题采样多个回答，用组内奖励的相对排名（均值和方差标准化）作为优势估计，无需 Critic。GRPO 更简单、内存更省，在奖励可直接计算（如数学验证）的场景效果几乎等同 PPO。

- Q: 为什么 RL 能提升 LLM 的推理能力，而 SFT 不能？
  A: SFT 是模仿学习，只能学习训练数据中的推理模式；RL 通过自我探索（采样不同推理路径）+ 奖励信号发现 SFT 数据中没有的正确推理策略。关键：RL 允许模型"犯错-修正"，学习到更鲁棒的推理过程，而非死记硬背特定答案格式。

- Q: LLM 的 RLHF 和用于数学推理的 RL 有什么不同？
  A: RLHF 的奖励来自人工反馈（主观，难以规模化，奖励模型可能被"破解"）；数学/代码推理的 RL 用可验证奖励（答案对错 = 客观奖励，无法被破解）。后者训练更稳定，奖励更可靠，是 DeepSeek R1 等工作成功的关键原因之一。
