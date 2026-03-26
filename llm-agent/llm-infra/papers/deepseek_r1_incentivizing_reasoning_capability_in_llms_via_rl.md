# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
> 来源：arxiv/2501.12948 | 领域：llm-infra | 学习日期：20260326

## 问题定义
LLM 的数学/代码/逻辑推理能力提升面临：
- 传统 SFT 需要大量高质量 CoT 标注数据，成本高
- RLHF 通常改善对话质量，对推理能力提升有限
- 如何让模型自主发现长链推理策略（self-play）
- 推理模型与通用对话能力的平衡

## 核心方法与创新点
**DeepSeek-R1**：纯 RL 激励推理能力（无需推理过程监督数据）。

**三阶段训练流程：**
```
Phase 1 - Cold Start（SFT 热启动）：
  少量推理格式数据，训练模型学会 <think>...</think> 格式
  
Phase 2 - RL（GRPO）：
  纯 RL 强化推理，不需要推理过程标注
  奖励：答案正确性（规则验证）+ 格式奖励
  
Phase 3 - Rejection Sampling + SFT（蒸馏回 SFT）：
  用 R1 生成高质量 CoT 数据，蒸馏到小模型（R1-Distill）
```

**GRPO（Group Relative Policy Optimization）：**
```python
# 对同一 query 采样 G 个输出
outputs = [policy.generate(query) for _ in range(G)]

# 组内相对奖励（不需要 Critic 模型）
rewards = [compute_reward(o) for o in outputs]
baseline = mean(rewards)

# 策略梯度（相对优势）
advantages = [(r - baseline) / std(rewards) for r in rewards]
L_grpo = -Σ_i advantages_i · log π(o_i | query)
```

**奖励函数设计：**
```python
def reward(output, query):
    answer = extract_answer(output)
    correct = verify_answer(answer, gold_answer)
    
    format_ok = "<think>" in output and "</think>" in output
    
    return 1.0 × correct + 0.1 × format_ok - 0.5 × (not format_ok)
    # 语言一致性奖励：中文 query → 中文回答（防止语言混乱）
```

**关键涌现行为（Emergent Behaviors）：**
- 自我反思（Self-reflection）：模型自发在推理中验证中间结论
- 长思维链（Long CoT）：复杂问题自发生成数千 token 的推理过程
- 回溯搜索（Backtracking）：发现错误后自动重新尝试

## 实验结论
- AIME 2024（竞赛数学）：79.8%（vs GPT-4o 9.3%，vs Claude-3.5 16%）
- MATH-500：97.3%
- Codeforces（编程竞赛）：2029 Elo 分（超过 96.3% 人类选手）
- MMLU：90.8%（通用能力基本保持）
- 思考 token 数：平均 1000-8000 token（复杂问题更长）

## 工程落地要点
1. **GRPO vs PPO**：GRPO 不需要 Value Network（Critic），训练资源减半；PPO 理论更完善但显存消耗大
2. **奖励 Hacking 防控**：格式奖励权重不能太高，否则模型学会用冗余 `<think>` 标签刷奖励
3. **语言混合问题**：多语言数据训练时加语言一致性奖励（query 语言 = 输出语言）
4. **蒸馏策略**：R1-7B/14B 用 R1-671B 生成的 CoT 蒸馏，效果接近大模型
5. **推理长度控制**：生成时设置 max_thinking_tokens 防止无限推理

## 面试考点
**Q1: DeepSeek-R1 为什么可以通过纯 RL 激励推理而不需要 CoT 标注？**
A: RL 的奖励信号是答案正确性（可以规则验证）。模型通过大量试错，自发发现"写推理过程"有助于答对问题（因为推理过程帮助生成正确答案），从而强化了 CoT 行为。类比：不需要教孩子每一步，只要有对错反馈，孩子会自己找到解题策略。

**Q2: GRPO 相比 PPO 的优劣？**
A: GRPO 优势：无需 Critic（Value Network），显存节省 ~50%，训练更简单稳定。劣势：基线估计用 batch 内平均（而非精确 V(s)），方差更大，收敛可能较慢。实践中 GRPO 在推理任务上效果与 PPO 相当，但工程代价更低。

**Q3: R1 的涌现行为（自我反思/回溯）是真正的推理吗？**
A: 是行为上的推理：模型通过 RL 学会了在 token 序列中表达反思和回溯，并且这些行为确实提高了答案正确率。但这是否等同于人类认知的"推理"存在争议。关键是：这些涌现行为在 CoT 标注数据中是人为设计的，而 R1 是从奖励信号中自发习得。

**Q4: 为什么 R1 在 Codeforces 上表现出色，但在某些 Chat 任务上退化？**
A: RL 对代码/数学的奖励信号精确（执行通过=正确），模型在这些任务上优化充分。Chat 任务的奖励模糊（人类偏好），RL 训练不稳定，且过度"推理模式"导致对话冗长、格式不符合聊天习惯。R1 系列中通用对话通过 SFT Mix 保持。

**Q5: 如何将 R1 的推理能力迁移到领域特定任务（如广告/推荐）？**
A: ①蒸馏：用 R1 生成领域 CoT 数据，SFT 领域模型 ②Domain RL：在领域任务上继续 RL，定义领域奖励（如推荐准确率）③Prompt Engineering：用 System Prompt 激活 R1 的推理模式 ④混合微调：通用 CoT + 领域 SFT 数据联合训练。
