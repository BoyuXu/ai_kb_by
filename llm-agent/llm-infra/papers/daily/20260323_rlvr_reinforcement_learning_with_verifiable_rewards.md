# RLVR: Reinforcement Learning with Verifiable Rewards for LLM Post-Training
> 来源：https://arxiv.org/abs/2501.12599 | 领域：llm-infra | 日期：20260323

## 问题定义
RLHF（基于人类反馈的RL）需要奖励模型，存在reward hacking风险。RLVR提出使用可验证奖励（Verifiable Rewards）的强化学习，直接用答案正确性作为奖励，更稳定，无需奖励模型。

## 核心方法与创新点
- 可验证奖励（Verifiable Rewards）：数学答案/代码执行结果/逻辑验证等客观奖励
- 无奖励模型：直接判断生成结果是否正确，不需要额外的打分模型
- GRPO/PPO/REINFORCE++：多种RL算法在可验证奖励下的比较
- 奖励塑形（Reward Shaping）：格式奖励、过程奖励等补充信号

## 实验结论
RLVR在数学任务（MATH/AIME）上相比纯SFT提升显著（>10%）；可验证奖励比奖励模型更稳定（无reward hacking）；GRPO相比PPO在计算效率上优约40%；过程奖励（PRM）比结果奖励（ORM）更优。

## 工程落地要点
- 代码执行沙箱是RLVR用于代码任务的核心基础设施
- 数学任务需要鲁棒的答案解析器（处理不同格式的正确答案）
- 混合奖励（可验证+人类偏好）可以覆盖更多任务类型

## 常见考点
1. **Q: RLVR与RLHF的核心差异？** A: RLHF用奖励模型（可能hacking）；RLVR用客观验证（稳定但适用任务有限）
2. **Q: 什么任务适合RLVR？** A: 数学（答案对错）、代码（测试通过）、逻辑（形式验证）；不适合开放问答
3. **Q: Reward Hacking（奖励黑客）是什么？** A: 模型学会在不真正完成任务的情况下获得高奖励分数（欺骗奖励模型）
4. **Q: GRPO（Group Relative Policy Optimization）的优势？** A: 无需critic（value）网络，减少40%计算；用组内相对奖励代替绝对值
5. **Q: 过程奖励（PRM）相比结果奖励（ORM）的优势？** A: 对每步推理打分，提供更密集的监督信号，减少稀疏奖励的训练困难
