# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
> 来源：https://arxiv.org/abs/2501.12948 | 领域：llm-infra | 日期：20260323

## 问题定义
如何通过强化学习（RL）激发LLM的推理能力，无需大量人工标注的CoT数据。DeepSeek-R1展示了通过GRPO和可验证奖励（verifiable rewards），LLM可以自发涌现长链推理能力。

## 核心方法与创新点
- GRPO算法：Group Relative Policy Optimization，无需critic model的轻量RL
- 可验证奖励：数学/代码任务有客观答案，无需人工打分
- 推理涌现：RL训练中，模型自发学会"思考"（长推理链）
- 蒸馏策略：将R1的推理能力蒸馏到小模型（1.5B-70B）

## 实验结论
DeepSeek-R1在AIME 2024达到79.8%（接近OpenAI o1），MATH达到97.3%；完全通过RL（无SFT cold start效果不佳），实际需要先SFT冷启动再RL；蒸馏到7B模型保留约90%的推理能力。

## 工程落地要点
- GRPO比PPO更简单（无value network），训练成本约减少40%
- 可验证奖励是关键：数学/代码/逻辑题有明确答案，无需奖励模型
- 推理模型的serving需要处理长输出（数千token），需要optimized KV cache

## 常见考点
1. **Q: GRPO（Group Relative Policy Optimization）的原理？** A: 对同一问题采样G个回答，用组内相对排名计算优势函数，无需critic网络
2. **Q: 为什么可验证奖励（RLVR）比RLHF更稳定？** A: 奖励明确客观（答案对/错），无奖励模型的过拟合风险（reward hacking）
3. **Q: DeepSeek-R1训练的"aha moment"是什么？** A: RL训练中，模型自发出现"回溯重思考"行为，类似人类解题时的顿悟
4. **Q: 推理能力为何可以蒸馏到小模型？** A: 大模型的推理链作为训练数据，小模型学习模仿推理过程
5. **Q: Long CoT的工程挑战？** A: 生成长推理链latency高（数秒）、需要流式输出、用户体验设计（thinking动画）
