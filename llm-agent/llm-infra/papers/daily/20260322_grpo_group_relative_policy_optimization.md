# GRPO: Group Relative Policy Optimization for Large Language Model Reasoning

> 来源：arxiv (DeepSeek) | 日期：20260322 | 领域：LLM工程

## 问题定义

PPO（Proximal Policy Optimization）是 RLHF 的主流算法，但需要额外的 Critic 模型（与 LLM 同规模），计算开销大（实际显存/计算 ~2× policy model）。GRPO 提出无 Critic 的组相对策略优化，显著降低 RLHF 训练成本，同时提升数学推理性能。

## 核心方法与创新点

- **组相对奖励（Group Relative Reward）**：
  - 对同一 prompt，采样 G 个输出（组）
  - 每个输出的相对优势 = (reward_i - mean_group) / std_group
  - 用组内相对比较代替 Critic 的绝对价值估计
- **消除 Critic 模型**：无需训练和维护与 policy 同等规模的 value network，显存减少约 40%
- **优化目标**：
  ```
  L = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)] - β × KL(π_θ || π_ref)
  ```
  与 PPO 形式相同，但 A_t（优势）来自组内相对比较而非 Critic 估计
- **规则奖励函数**：对数学推理任务，奖励 = 答案是否正确（0/1）+ 格式奖励（是否包含 CoT）
- **应用于 DeepSeek-Math**：GRPO 训练使 DeepSeek-Math-7B 在 MATH 基准达到 51.7%

## 实验结论

- MATH 基准：GRPO 训练的 DeepSeek-Math-7B（51.7%）> SFT + PPO（46.3%）> 纯 SFT（42.1%）
- 计算效率：GRPO vs PPO 显存减少 38%，训练吞吐提升 1.5×
- 组大小 G 的影响：G=8 最优，G<4 奖励估计方差过大，G>16 额外收益小
- KL 散度约束：β=0.04 最优，过大抑制探索，过小导致分布漂移

## 工程落地要点

- **采样策略**：同一 prompt 采样 G=8 个输出，Temperature=0.9（保证多样性）
- **Reward 函数设计**：
  - 数学推理：答案正确性（主）+ 步骤格式（辅）
  - 代码生成：单元测试通过率
  - 通用对齐：需要 reward model（此时 GRPO 主要优势是省 Critic）
- **训练稳定性**：奖励 normalization（组内归一化）对稳定训练至关重要
- **参考模型更新**：建议每隔 N 步更新 π_ref，防止 KL 约束失效

## 常见考点

1. **Q：GRPO 和 PPO 的核心区别是什么？**
   A：PPO 用 Critic 模型估计绝对状态价值（A_t = R_t - V(s_t)）；GRPO 用组内多个采样结果的相对比较估计优势（A_t = (R_t - mean_G) / std_G），无需 Critic

2. **Q：为什么 GRPO 对数学推理特别有效？**
   A：数学推理有明确的 0/1 奖励信号（答对/错），规则奖励函数高质量；组内多样采样能探索不同推理路径，通过相对比较筛选出正确的思维链

3. **Q：GRPO 的局限性？**
   A：需要同一 prompt 多次采样（G×推理成本）；对奖励函数质量敏感（需要能明确判断对错的任务）；通用对话场景仍需 reward model，优势不如数学推理场景明显
