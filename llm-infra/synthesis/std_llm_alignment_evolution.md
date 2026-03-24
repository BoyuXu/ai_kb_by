# LLM 对齐方法演进：从 RLHF 到 DPO 到 GRPO

> 📚 参考文献
> - [Rlvr Reinforcement Learning With Verifiable Rew...](../../llm-infra/20260323_rlvr_reinforcement_learning_with_verifiable_rewards.md) — RLVR: Reinforcement Learning with Verifiable Rewards for ...
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Grpo-Group-Relative-Policy-Optimization-Llm-Rea...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-llm-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Moe-Llama-Mixture-Of-Experts-For-Efficient-Larg...](../../llm-infra/20260321_moe-llama-mixture-of-experts-for-efficient-large-language-model-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Kimi K1.5 Scaling Reinforcement Learning With Llms](../../llm-infra/20260323_kimi_k1.5_scaling_reinforcement_learning_with_llms.md) — KIMI k1.5: Scaling Reinforcement Learning with LLMs
> - [Efficient-Long-Context-Llms-Survey-Benchmark-20...](../../llm-infra/20260321_efficient-long-context-llms-survey-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Grpo Group Relative Policy Optimization](../../llm-infra/20260322_grpo_group_relative_policy_optimization.md) — GRPO: Group Relative Policy Optimization for Large Langua...


> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：InstructGPT, RLHF, DPO, GRPO, DeepSeek-R1, RLVR 系列

---

## 🎯 核心洞察（5条）

1. **对齐的本质是"让模型按人类偏好行为"**：预训练模型会说流利的废话，对齐让它有用、安全、诚实
2. **RLHF → DPO → GRPO 的演进核心是"去掉 Critic"**：RLHF 需要训练 Reward Model + PPO 优化（4 个模型），DPO 直接用偏好数据优化（2 个模型），GRPO 用组内排名替代 Critic（1 个模型 + 采样）
3. **RLVR 是推理对齐的新范式**：Reinforcement Learning with Verifiable Rewards——用可验证的奖励（数学题对错、代码是否通过测试）训练推理能力，不需要人工标注
4. **DPO 简单但有天花板**：DPO 假设偏好可以被一个隐式 Reward Model 完美建模，对复杂偏好（多维度评判）能力有限
5. **对齐税（Alignment Tax）不可忽视**：过度对齐会降低模型的通用能力（"太安全以至于什么都不敢说"），需要在安全性和有用性之间平衡

---

## 📈 技术演进脉络

```
SFT 监督微调（2022, InstructGPT 第一步）
  → RLHF = SFT + Reward Model + PPO（2022, ChatGPT）
    → DPO 直接偏好优化（2023, 去掉 RM + PPO）
      → KTO 基于单点反馈的对齐（2024）
        → GRPO 组内排名替代 Critic（2024, DeepSeek）
          → RLVR 可验证奖励 + RL（2025, DeepSeek-R1）
            → Constitutional AI 自我对齐（Anthropic 路线）
```

**关键转折点**：
- **RLHF（2022）**：ChatGPT 验证对齐的巨大价值，从"能力强但不可控"到"能力强且有用"
- **DPO（2023）**：将 RLHF 的复杂训练流水线简化为一个 loss function，大幅降低对齐门槛
- **GRPO/RLVR（2024-2025）**：DeepSeek 证明推理能力可以通过 RL 训练获得，不需要人工标注推理过程

---

## 🔗 跨文献共性规律

| 规律 | 体现 | 说明 |
|------|------|------|
| 训练复杂度持续降低 | RLHF→DPO→GRPO | 每一代都在减少训练所需的模型数量和工程复杂度 |
| 自动化奖励取代人工标注 | RLVR, Constitutional AI | 可验证奖励（数学/代码）和 AI 自评替代昂贵的人工标注 |
| 对齐和能力可以协同提升 | DeepSeek-R1 | RLVR 训练推理能力的同时也在做对齐 |
| 过对齐风险真实存在 | Alignment Tax 研究 | Safety 过滤太严格导致模型拒绝合理请求 |

---

## 🎓 面试考点（6条）

### Q1: RLHF 的完整训练流程？
**30秒答案**：三步——①SFT：在人工标注的高质量对话上微调；②Reward Model：用人类偏好对比数据训练打分模型（A 比 B 好）；③PPO：用 RM 的分数作为奖励信号，用 PPO 算法优化 SFT 模型。
**追问方向**：PPO 训练中的 KL 散度约束有什么用？答：防止模型偏离 SFT 分布太远（reward hacking），保持输出质量。

### Q2: DPO 相比 RLHF 的核心简化？
**30秒答案**：DPO 数学上证明 RLHF 的最优策略可以直接用偏好数据训练，无需先训练 RM 再用 PPO 优化。Loss = -log σ(β(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))。
**追问方向**：DPO 的假设和限制？答：假设偏好可以被 Bradley-Terry 模型描述（pair-wise 比较），对多维度偏好建模能力有限。

### Q3: GRPO 的创新点？
**30秒答案**：GRPO 不需要 Critic/RM——对同一 prompt 采样 G 个回答，用奖励函数（如数学题对错）计算各回答的分数，组内标准化后作为 advantage 信号。本质是"班级排名法"而非"单独评分法"。
**追问方向**：奖励函数怎么设计？答：可验证任务用规则（对/错），开放任务用 LLM-as-judge 或简单启发式。

### Q4: RLVR 为什么能训练推理能力？
**30秒答案**：RLVR 用数学题/代码题等有标准答案的任务，奖励 = "答案对不对"。模型通过 RL 探索不同的推理路径（Chain-of-Thought），自然涌现出"一步步推理"的能力。DeepSeek-R1 就是用这种方式训练的。
**追问方向**：RLVR vs RLHF 的本质区别？答：RLVR 的奖励是客观可验证的，RLHF 的奖励是主观人类偏好。

### Q5: SFT、RLHF、DPO 什么时候该用哪个？
**30秒答案**：①SFT：基础对齐，有高质量标注数据时首选，简单有效；②RLHF：追求最优效果但有工程资源（需要 4 个模型并行训练）；③DPO：有偏好数据但工程资源有限，性价比最高。
**追问方向**：能否跳过 SFT 直接 DPO？答：不推荐，SFT 提供了良好的初始策略，直接 DPO 收敛困难。

### Q6: 对齐评估怎么做？
**30秒答案**：①自动评估：MT-Bench/AlpacaEval 用 LLM 打分；②人类评估：盲评对比（A/B 测试）；③安全评估：Red Teaming（对抗测试），看模型是否会被诱导输出有害内容。
**追问方向**：LLM-as-judge 的问题？答：偏好自己的风格（verbosity bias）、位置偏差（倾向选第一个）、可能被 prompt 操纵。

---

## 🌐 知识体系连接

- **上游依赖**：PPO 强化学习、语言模型预训练、人类偏好标注
- **下游应用**：ChatBot、Agent 系统、安全审核
- **相关 synthesis**：std_llm_inference_optimization.md, std_llm_moe_architecture.md
- **相关论文笔记**：synthesis/20260321_grpo_rl_alignment.md, synthesis/20260323_rlvr_vs_rlhf_posttraining.md
