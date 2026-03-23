# RLVR vs RLHF：LLM 后训练的两条路
> 知识卡片 | 创建：2026-03-23 | 领域：llm-infra

---

**一句话**：RLHF 用「人工打分员」（奖励模型）来判断好坏，RLVR 直接用「考试答案对错」来判断——后者更稳定，但只适用于有客观标准的任务（数学/代码/逻辑）。

**类比**：RLHF 像「请老师评作文」（主观，老师可能被文采骗），RLVR 像「对数学答案」（客观，1+1 就是 2）。DeepSeek-R1 和 Kimi K1.5 都用了 RLVR 路线，结果 LLM 自发「学会思考」。

---

## 核心机制对比

```
RLHF（Reinforcement Learning from Human Feedback）
├── 流程：SFT → 奖励模型训练 → PPO 优化 → 模型
├── 奖励来源：奖励模型（RM）打分，RM 由人类偏好数据训练
├── 风险：Reward Hacking（模型欺骗奖励模型，获高分但不完成任务）
├── 适用：开放生成（写作、对话、创意内容）
└── 代表：InstructGPT, ChatGPT

RLVR（Reinforcement Learning with Verifiable Rewards）
├── 流程：SFT（可选）→ RLVR 直接用客观奖励 → 模型
├── 奖励来源：程序验证（答案对错、测试通过与否）
├── 风险：覆盖任务有限（不适合开放问答）
├── 适用：数学、代码、形式逻辑等有客观标准的任务
└── 代表：DeepSeek-R1-Zero, Kimi K1.5
```

---

## RL 算法进化线

```
PPO（2017，Schulman）
├── 需要 Actor + Critic 两个网络（内存 2x）
├── 复杂超参（clip ratio, GAE λ, entropy coef）
└── 工业标准，但工程实现复杂

GRPO（Group Relative Policy Optimization，DeepSeek）
├── 去掉 Critic 网络 → 显存节省 40-50%
├── 核心：同一 prompt 采样 G 个回答，用组内相对奖励归一化 advantage
├── KL 惩罚防止策略漂移（token 级别）
└── 效果：DeepSeek-Math 7B MATH +5.2pp；计算效率 ~1.4x PPO

REINFORCE++（简化版 PPO）
├── 无 Critic，无 GAE，更简单
└── 适合 verifiable reward 场景（奖励稀疏但精确）

今日关键结论（RLVR 论文）：
- 过程奖励（PRM）> 结果奖励（ORM）：每步推理打分 > 只看最终答案
- 格式奖励（format reward）有效：强制<think>结构，质量稳定提升
- GRPO 在 RLVR 场景实际效果最佳
```

---

## DeepSeek-R1 的惊人现象：自发 CoT 涌现

```
实验（R1-Zero）：
  输入：纯数学问题，零 SFT，只有 RLVR
  输出：模型自发学会 <think>...</think> 格式，产生 CoT 推理

为什么会这样？
  强化学习压力 → 模型发现「先想清楚再回答」得到更高验证奖励
  → 推理是一种被奖励强化出来的「工具行为」

类比：让 AI 做数学题得分高 → AI 自己发明了演算纸
```

---

## 工业落地实践

| 能力需求 | 推荐方案 | 原因 |
|---------|---------|------|
| 数学/代码推理 | RLVR + GRPO | 客观奖励，稳定，无 reward hacking |
| 写作/对话助手 | RLHF + PPO/DPO | 无客观标准，需人类偏好 |
| 多任务混合 | RLHF（软奖励）+ RLVR（硬奖励）组合 | 互补覆盖 |
| 低成本微调 | DPO（无需在线 RL）| 效果 ~PPO 的 80%，实现简单 |

**广告推荐中的 RLVR 应用**：
- CTR/CVR 预测 → 客观奖励（用户行为）→ 可直接用 RLVR 框架
- 创意文案生成 → 主观好坏 → 仍需 RLHF 或 LLM-as-judge
- 出价策略优化（AutoBid）→ 可验证收益 → RLVR 天然适配

---

## 面试考点

1. **Q: RLVR 和 RLHF 最核心的差异？**
   A: 奖励来源不同：RLHF 用学出来的奖励模型（可被欺骗），RLVR 用程序验证（客观不可欺骗）

2. **Q: Reward Hacking 是什么？如何防止？**
   A: 模型找到奖励模型的「漏洞」，表面得分高但实际质量差；防止：奖励模型多样化、KL 约束防漂移、用 RLVR 替代

3. **Q: GRPO 为什么可以不需要 Critic？**
   A: PPO 的 Critic 估计 baseline（减少方差），GRPO 用同 prompt 的 G 个回答的平均奖励作 baseline，组内相对比较代替绝对值估计

4. **Q: 过程奖励（PRM）vs 结果奖励（ORM）？**
   A: ORM 只看最终答案（稀疏，第一步错也得负 reward）；PRM 对每步推理打分（密集，更精确的学习信号）

5. **Q: DeepSeek-R1-Zero 的 CoT 涌现说明了什么？**
   A: 推理能力可以通过纯 RL 自发获得，不一定需要 CoT 数据 SFT；意味着 scaling RL 可能比 scaling data 更有效
