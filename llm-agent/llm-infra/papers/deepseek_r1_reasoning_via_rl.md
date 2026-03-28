# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> 来源：arxiv | 领域：llm-infra | 学习日期：20260328

## 问题定义

大型语言模型（LLM）在复杂推理任务（数学、代码、逻辑推理）上表现不足。现有方法依赖大量人工标注的 Chain-of-Thought（CoT）数据进行监督微调，成本高且难以扩展。如何通过强化学习（RL）直接激励模型产生推理能力，而不依赖人工标注的推理链？

## 核心方法与创新点

### 1. Group Relative Policy Optimization（GRPO）
DeepSeek-R1 的核心训练算法，去掉了 PPO 中需要的 critic 模型，改为对同一 prompt 采样多个输出，计算组内相对奖励来估计基线：

$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{(q,a)\sim\mathcal{D}} \frac{1}{G} \sum_{i=1}^{G} \left[ \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\cdot, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

其中 $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ 是归一化优势。

### 2. 四阶段训练流程
- **Stage 1: Cold Start SFT**：用少量精心筛选的 CoT 数据初始化，避免 RL 早期不稳定
- **Stage 2: RL 训练（DeepSeek-R1-Zero）**：仅用基于规则的奖励（格式奖励 + 准确性奖励），自主涌现推理能力
- **Stage 3: Rejection Sampling + SFT**：将 RL 模型生成的高质量推理链收集为 SFT 数据
- **Stage 4: RL 精调**：在多任务上联合 RL 对齐

### 3. 关键涌现现象
- **Aha Moment**：模型在训练中自发出现"重新思考"行为，无需人工设计
- **长思维链**：随 RL 训练进行，CoT 长度自发增长，推理更深入
- **自我验证**：模型开始自主检查答案正确性

### 4. 蒸馏小模型
将 DeepSeek-R1 的推理能力蒸馏到 1.5B/7B/8B/14B/32B 模型，小模型获得超越同规模 SOTA 的推理能力。

## 实验结论

| 基准 | DeepSeek-R1 | OpenAI o1 |
|------|-------------|-----------|
| AIME 2024 | 79.8% | 79.2% |
| MATH-500 | 97.3% | 96.4% |
| Codeforces | 2029 Elo | 1891 Elo |
| MMLU | 90.8% | 91.8% |

- **DeepSeek-R1-Zero**（纯 RL，无 SFT）在 AIME 上从 15.6% 提升到 71.0%
- 7B 蒸馏模型在 AIME 上达到 55.5%，超过 QwQ-32B-Preview

## 工程落地要点

1. **奖励函数设计**：使用规则验证的准确性奖励（数学可验证答案）+ 格式奖励（`<think>`标签合规性），避免奖励 hacking
2. **训练稳定性**：Cold Start SFT 是 RL 稳定训练的关键，纯 RL 从头训练容易产生"语言混合"等退化
3. **GRPO vs PPO**：GRPO 省去 value model，减少约 50% 显存消耗
4. **采样策略**：每个 prompt 采样 G=8 个输出，增加组内多样性
5. **拒绝采样数据飞轮**：RL 模型 → 采样高质量数据 → SFT → RL，形成正向循环

## 面试考点

**Q1: DeepSeek-R1-Zero 是如何在没有任何 CoT 监督数据的情况下学会推理的？**

A: 通过 GRPO 强化学习，仅使用基于规则的奖励信号（答案正确与否）。模型通过试错自主发现有效推理策略，逐渐涌现 CoT 行为。

**Q2: GRPO 相比 PPO 的核心优势是什么？**

A: GRPO 通过同一 prompt 的多个采样输出估计基线，无需训练独立的 value/critic 模型，节省约 50% 显存，训练更高效。

**Q3: 为什么 Cold Start SFT 对 RL 训练很重要？**

A: 纯 RL 从随机初始化开始容易出现语言混合、格式混乱等退化问题。Cold Start 提供稳定的初始策略，让 RL 在合理分布上优化。

**Q4: DeepSeek-R1 的蒸馏策略是什么？为何有效？**

A: 用 R1 生成的推理链（包含长思维链）对小模型做 SFT 蒸馏，而非让小模型自己做 RL。小模型通过模仿大模型的推理过程获得能力，跳过了昂贵的 RL 探索阶段。

**Q5: 什么是"Aha Moment"？其工程意义是什么？**

A: 指模型在 RL 训练中自发出现"等等，让我重新想想"的反思行为，无需人工设计。这说明复杂推理行为可以从简单的环境反馈中涌现，为设计自主学习系统提供了理论依据。
