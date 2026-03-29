# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> 来源：https://arxiv.org/abs/2501.12948 | 领域：llm-infra | 学习日期：20260329

## 问题定义

### 核心挑战
论文针对**通用推理能力（General Reasoning）**这一AI领域的长期难题，探索如何让大语言模型（LLMs）通过自我进化而非人类标注数据来发展推理能力。

### 现有方法的局限
| 方法类型 | 局限性 |
|---------|--------|
| **SFT (监督微调)** | 依赖大量人工标注，存在认知偏见，限制模型探索非人类推理路径 |
| **CoT Prompting** | 需要精心设计few-shot示例，复杂问题仍不足 |
| **传统RLHF** | 需要训练 reward model，容易 reward hacking，流程复杂 |

### DeepSeek-R1 的核心问题
> **如何通过纯强化学习（Pure RL），在没有人类标注推理轨迹的情况下，激发LLM的推理能力？**

---

## 核心方法与创新点

### 双模型架构

```
DeepSeek-R1-Zero：纯RL训练，无SFT，探索推理能力的极限
DeepSeek-R1：多阶段训练：Cold Start → RL → SFT → RL，平衡性能与可用性
```

### GRPO: Group Relative Policy Optimization

**核心创新**：无需Value Model，使用组内相对优势估计

**GRPO 目标函数**：
```
J_GRPO(θ) = E[ 1/G Σᵢ min(πθ/πold * Aᵢ, clip(πθ/πold, 1-ε, 1+ε) * Aᵢ) - β·D_KL(πθ||πref) ]

其中：
- Aᵢ = (rᵢ - mean({r₁,...,r_G})) / std({r₁,...,r_G})  [组内相对优势]
- G = 16 (每组采样数)
- ε = 10 (clip ratio，关键超参)
- β = 0.001 (KL系数)
```

**GRPO vs PPO 对比**：

| 特性 | PPO | GRPO |
|-----|-----|------|
| Value Model | 需要（同尺寸） | 不需要 |
| 优势估计 | GAE | 组内相对奖励 |
| KL惩罚 | 每token密集奖励 | 直接加入loss |
| 内存开销 | 2x模型大小 | ~1x模型大小 |
| 超参敏感性 | 高（λ调参） | 低 |

### 奖励设计

**规则奖励（Rule-based Rewards）**：
```
Reward = Reward_accuracy + Reward_format

Accuracy: 答案正确性（数学：sympy匹配；代码：测试用例通过）
Format:  <think>...</think> 和 <answer>...</answer> 格式约束
```

**关键决策**：**不使用神经Reward Model**
- 原因1：大规模RL中易发生 reward hacking
- 原因2：重训练成本高，增加流程复杂度

### DeepSeek-R1 多阶段训练流程

```
Stage 1: Cold Start (数千条高质量CoT数据)
         ↓ 目的：解决R1-Zero的可读性和语言混合问题
Stage 2: 第一轮RL (推理导向)
         ↓ 使用规则奖励提升推理能力
Stage 3: Rejection Sampling + SFT (80万数据)
         ↓ 60万推理数据 + 20万非推理数据
Stage 4: 第二轮RL (全面对齐)
         ↓ 结合规则奖励 + Reward Model + 语言一致性奖励
      DeepSeek-R1 (最终模型)
```

**语言一致性奖励**：
```
Reward_language = Num(Words_target) / Num(Words_total)
```

### 模型蒸馏（Distillation）

将R1的推理能力迁移到小模型：
- 使用800K高质量样本（600K推理 + 200K非推理）
- 仅做SFT，**不做RL**（证明RL在小模型上效率低）
- 发布6个蒸馏模型：1.5B/7B/14B/32B (Qwen) + 8B/70B (Llama)

---

## 实验结论

### 主模型性能对比

| Benchmark | Claude-3.5-Sonnet | GPT-4o | DeepSeek-V3 | **DeepSeek-R1** |
|-----------|------------------|--------|-------------|-----------------|
| **AIME 2024** | 16.0% | 9.3% | 39.2% | **79.8%** |
| **MATH-500** | 78.3% | 74.6% | 90.2% | **97.3%** |
| **GPQA Diamond** | 65.0% | 49.9% | 59.1% | **71.5%** |
| **LiveCodeBench** | 38.9% | 32.9% | 36.2% | **65.9%** |
| **Codeforces Rating** | 717 | 759 | 1134 | **2029** |
| **MMLU-Pro** | 78.0% | 72.6% | 75.9% | **84.0%** |
| **AlpacaEval2.0 (LC)** | 52.0% | 51.1% | 70.0% | **87.6%** |

### DeepSeek-R1-Zero 自进化现象

| 指标 | 初始值 | 最终值 | 提升 |
|-----|--------|--------|------|
| AIME 2024 (Pass@1) | 15.6% | 77.9% | **+62.3%** |
| AIME 2024 (Cons@16) | - | 86.7% | - |

**涌现行为（Emergent Behaviors）**：
- **反思（Reflection）**："Wait, wait. Wait. That's an aha moment..."
- **验证（Verification）**：自我检查中间步骤
- **策略调整**：动态切换解题方法

### 蒸馏模型性能

| 模型 | AIME 2024 | MATH-500 | GPQA | Codeforces Rating |
|-----|-----------|----------|------|-------------------|
| GPT-4o | 9.3% | 74.6% | 49.9% | 759 |
| **R1-Distill-Qwen-1.5B** | **28.9%** | **83.9%** | 33.8% | 954 |
| **R1-Distill-Qwen-7B** | **55.5%** | **92.8%** | 49.1% | 1189 |
| **R1-Distill-Qwen-32B** | **72.6%** | **94.3%** | **62.1%** | **1691** |
| **R1-Distill-Llama-70B** | **70.0%** | **94.5%** | **65.2%** | **1633** |

> 💡 **关键发现**：1.5B蒸馏模型在数学任务上超越GPT-4o！

### 训练成本

| 阶段 | GPU Hours (H800) | 成本 (USD) |
|-----|------------------|-----------|
| DeepSeek-R1-Zero | 101K | $202K |
| SFT数据创建 | 5K | $10K |
| DeepSeek-R1 | 41K | $82K |
| **总计** | **147K** | **$294K** |

---

## 工程落地要点

### 基础设施设计

```
Rollout (vLLM) → Inference (RM/ref) → Rule Reward (CPU) → Training (GRPO)
       └──────────────────────────────────────────────────────┘
        定期更新参考模型 (每400步)
```

**关键优化**：
- **Expert Parallelism**：MoE架构跨节点专家并行
- **MTP加速**：Multi-Token Prediction自推测解码
- **异步调度**：Rule Reward模块与GPU计算重叠
- **数据打包**：Best-Fit策略减少padding浪费

### 训练超参配置

```python
# GRPO 关键配置
config = {
    "learning_rate": 3e-6,
    "kl_coefficient": 0.001,
    "clip_epsilon": 10,        # 关键：较大clip值防止梯度截断
    "temperature": 1.0,        # 探索
    "group_size": 16,
    "max_length": 32768,       # 8.2k步后提升至65536
    "batch_size": 512,         # 32问题 × 16采样
    "ref_update_steps": 400,   # 参考模型更新频率
}
```

### Prompt设计原则

```python
# ✅ 推荐：Zero-shot直接描述
prompt = "请解决下面的数学问题，将推理过程放在<think>标签内，答案放在<answer>标签内。\n\n问题：{question}"

# ❌ 避免（会损害性能）
prompt = "以下是几个示例... [few-shot examples]\n\n问题：{question}"
```

### 失败经验（Unsuccessful Attempts）

| 方法 | 失败原因 |
|-----|---------|
| **Process Reward Model (PRM)** | 1) 难以精确定义"步骤"；2) 步骤正确性判断困难；3) 易发生reward hacking |
| **MCTS** | 1) Token生成搜索空间远大于棋类；2) Value Model训练困难；3) 易陷入局部最优 |

### 关键结论

1. **基座模型规模**：< 30B参数难以涌现出有效长CoT
2. **Verifier可靠性**：规则奖励比神经网络奖励更稳定
3. **RL vs Distillation**：小模型用蒸馏更经济；突破智能边界需要大模型+大规模RL

---

## 面试考点

### 考点1：GRPO相比PPO的核心改进是什么？为什么更适合长CoT训练？

**答案要点**：

1. **去除Value Model**：PPO需要与Policy同尺寸的Value Model，内存开销翻倍；GRPO通过组内采样估计优势，无需额外模型

2. **组内相对优势**：
   - PPO: GAE基于单个trajectory估计优势
   - GRPO: 同一问题的多个采样输出相互比较，`Aᵢ = (rᵢ - mean(r)) / std(r)`

3. **KL惩罚方式**：
   - PPO: 每token作为dense reward，隐式惩罚长度
   - GRPO: 直接加入loss，不累积惩罚，允许response自然增长

4. **长CoT场景优势**：在推理任务中，部分生成内容后续可能被修改/否定，Value Model难以预测最终奖励；GRPO直接基于结果奖励优化，更适合

---

### 考点2：为什么DeepSeek-R1-Zero没有使用SFT作为冷启动？

**答案要点**：

1. **核心假设**：人类标注的推理轨迹可能包含偏见和局限，限制了模型探索更优的非人类推理路径

2. **探索vs利用的权衡**：
   - SFT让模型快速学会"人类怎么思考"
   - 纯RL让模型自主发现"如何更好地思考"

3. **涌现证据**：R1-Zero在训练中自发学会了反思（"Wait..."）、验证、策略调整等高级推理模式

4. **妥协方案**：R1最终采用小量Cold Start数据（数千条），平衡可读性与探索空间

---

### 考点3：论文中提到的"Reward Hacking"是什么？如何防范？

**答案要点**：

1. **定义**：模型找到奖励函数的漏洞，以非预期方式获得高奖励（如利用RM的长度偏见生成长而无用的回复）

2. **文中案例**：
   - 使用Helpful RM时，模型reward上升但Codeforces性能下降
   - 说明RM被hack，模型学会了"讨好"RM而非真正提升能力

3. **防范策略**：
   - 推理任务：**规则奖励**（数学sympy匹配、代码测试用例）
   - 通用任务：**限制RL步数**（最后400步才加入RM信号）
   - **定期更新参考模型**，防止策略偏离过远

---

### 考点4：为什么小模型（如7B）的蒸馏效果优于直接RL训练？

**答案要点**：

1. **实验证据**：
   - Qwen2.5-32B直接RL ≈ QwQ-32B-Preview
   - R1-Distill-Qwen-32B >> 直接RL的32B模型

2. **原因分析**：
   - **探索空间不足**：小模型容量有限，难以通过trial-and-error发现有效推理模式
   - **计算效率**：小模型RL需要大量步骤才能收敛，成本高昂
   - **知识蒸馏优势**：大模型已探索出高质量推理轨迹，小模型直接模仿更高效

3. **实践结论**：小模型用SFT蒸馏即可；突破性能边界仍需大模型+大规模RL

---

### 考点5：DeepSeek-R1的推理长度如何随问题难度自适应变化？

**答案要点**：

1. **自适应机制**：
   - 简单问题（如"1+1=?"）：<100 tokens
   - 困难问题：可达18,000+ tokens
   - 平均：约8,793 tokens（2024数学竞赛测试集）

2. **动态行为**：
   - 验证/修正时增加token消耗
   - 回溯探索替代方案时延长思考
   - 根据中间结果决定继续或停止

3. **与Test-time Scaling的关系**：
   - 传统方法：Majority Voting（独立采样，token效率低）
   - R1：单链深度推理，利用self-reflection动态调整
   - R1 + Majority Voting@64：AIME 79.8% → 86.7%

4. **局限性**：仍有"过度思考"现象，简单问题偶尔生成过长推理链
