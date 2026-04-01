# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

LLM 的强化学习（RLHF/RLAIF）训练面临以下挑战：
1. **训练不稳定**：PPO 在大模型上容易因 KL 散度过大崩溃
2. **奖励稀疏**：数学推理等任务奖励信号稀疏，探索效率低
3. **工程复杂性**：Actor、Critic、Reward Model 多组件协调困难
4. **规模化**：如何高效支持 70B+ 规模的 RL 训练

DAPO（Direct Advantage Policy Optimization）提出一套改进 PPO 的 RL 训练算法和系统，在 ByteDance 广告 LLM 团队规模化部署。

## 核心方法与创新点

### 算法改进：DAPO 核心技术

**1. Clip-Higher 技巧**

标准 PPO 对策略更新的 ratio 做对称 clip：$[1-\epsilon, 1+\epsilon]$

DAPO 提出非对称 clip：

$$
clip(r_t, 1-\epsilon_{low}, 1+\epsilon_{high}), \quad \epsilon_{high} > \epsilon_{low}
$$

对于 advantage > 0 的动作（应该增强的行为），允许更大的更新幅度；对于 advantage < 0 的动作，更保守地抑制，防止过度惩罚。

**2. Token-Level Policy Gradient Loss**

标准方法用 sequence-level 平均，DAPO 改为 token-level loss：

$$
\mathcal{L} = -\frac{1}{\sum_{t} |tokens_t|} \sum_{t} \sum_{k} A_t \log \pi_\theta(a_{t,k} | s_{t,k})
$$

避免长序列的 advantage 被平均稀释，对长推理链更友好。

**3. 去除 KL 惩罚项**

传统 PPO 有 KL 散度约束：$r_t - \beta KL[\pi_\theta || \pi_{ref}]$，DAPO 通过动态调整 clip 范围替代 KL 约束，简化训练。

**4. 过滤低奖励样本**

对 reward 极低的 group 不做梯度更新，防止错误信号污染策略。

### 系统架构

- **vLLM + Ray** 分布式推理生成
- **Megatron-LM** 分布式训练更新
- 异步 rollout 与参数更新解耦，提升 GPU 利用率

## 实验结论

- 在 AIME（数学竞赛）和代码生成任务上显著优于标准 PPO 和 GRPO
- Clip-Higher 技巧单独贡献约 3% pass@1 提升
- Token-level loss 对长链推理提升约 2%
- 在 30B/70B 模型规模验证有效性
- 系统吞吐量比朴素 RL 训练提升约 2.5x

## 工程落地要点

1. **$\epsilon$ 参数选择**：$\epsilon_{low}=0.2, \epsilon_{high}=0.4$ 是实践起点，需根据任务调整
2. **Rollout batch size**：每个 prompt 生成 G=8~16 个回答用于 advantage 估计
3. **奖励函数设计**：广告 LLM 场景的奖励需结合 CTR 预估信号和内容质量
4. **Reference model 冻结**：DAPO 去掉 KL 约束后，reference model 可以定期更新而非完全冻结
5. **内存管理**：70B 模型 RL 训练需要 ZeRO-3 + 梯度 checkpoint，仔细管理内存
6. **监控**：关键指标：entropy（防止策略崩溃）、reward distribution、KL divergence from reference

## 面试考点

**Q：DAPO 的 Clip-Higher 为什么要非对称地设置 clip 范围？**
A：正向 advantage 的动作应该被增强，给更大的 $\epsilon_{high}$ 允许策略更快学习好的行为；负向 advantage 的动作应该被抑制，但过大的惩罚容易导致策略过度收缩失去多样性，小 $\epsilon_{low}$ 更保守。非对称设计在探索效率和稳定性之间取得更好平衡。

**Q：Token-level policy gradient loss 相比 sequence-level 有什么优势？**
A：Sequence-level 对所有 token 取平均，对短序列和长序列给予相同权重；但长链推理（500+ tokens）的 advantage 被平均后每个 token 的信号非常弱，学习困难。Token-level loss 保留了每个 token 的独立贡献，对长推理链的学习效果更好。

**Q：广告系统中 LLM 的 RL 训练和通用 LLM（如 ChatGPT）的 RLHF 有什么区别？**
A：1) 奖励信号：广告 LLM 用 CTR/转化率等业务指标作为奖励，而非人类偏好；2) 约束更严格：广告内容有合规要求，奖励函数要结合内容安全；3) 分布特性：广告文案分布比通用对话窄，RL 优化更快收敛；4) 规模考量：广告系统每天处理海量请求，对推理延迟要求更高，模型不能过大。
