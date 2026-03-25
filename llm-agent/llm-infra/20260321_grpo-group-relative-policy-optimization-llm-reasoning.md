# GRPO: Group Relative Policy Optimization for Large Language Model Reasoning

> 来源：https://arxiv.org/abs/2402.03300 | 日期：20260321 | 领域：llm-infra

## 问题定义

大语言模型的推理能力提升长期依赖 RLHF（PPO 算法），但 PPO 在 LLM 对齐场景中存在显著工程问题：

1. **需要独立的 Critic 模型**：与 Policy 同量级，训练时需要 2x 显存
2. **Variance 高**：单样本的 reward baseline 估计不准确，导致梯度方差大、收敛慢
3. **超参敏感**：PPO 的 clip 参数、KL penalty 系数需要精心调整
4. **推理效率低**：每步 RL 需要额外的 Critic 前向传播

**DeepSeek-Math 团队**提出 GRPO（Group Relative Policy Optimization），通过对同一问题采样多个回答并在组内进行相对比较，消除 Critic 模型的需求，同时降低 variance。这也是 DeepSeek-R1 系列的核心训练算法。

## 核心方法与创新点

### GRPO vs PPO 的本质区别

**PPO Advantage 估计：**
```
A_t = r_t + γV(s_{t+1}) - V(s_t)  # 需要 Critic V 函数
```

**GRPO Advantage 估计（组内相对）：**
```python
# 对同一问题 q 采样 G 个回答 {o_1, o_2, ..., o_G}
# 每个回答得到 reward {r_1, r_2, ..., r_G}
# 用组内均值和标准差归一化 advantage

mean_r = mean(r_1, ..., r_G)
std_r = std(r_1, ..., r_G)
A_i = (r_i - mean_r) / std_r  # 组内相对 advantage
```

**GRPO 优化目标：**
```
J_GRPO(θ) = E_{q~P(Q), {o_i}~π_θ_old(·|q)} [
  (1/G) Σᵢ min(
    (π_θ(oᵢ|q) / π_θ_old(oᵢ|q)) · Aᵢ,
    clip(ratio, 1-ε, 1+ε) · Aᵢ
  ) - β · KL(π_θ || π_ref)
]
```

### 关键创新点

1. **无需 Critic 模型**：组内相对打分取代价值函数，节省 50% 显存
2. **低方差 Advantage 估计**：G 个样本的组内统计比单样本 baseline 更稳定
3. **Token-level KL 惩罚**：直接约束每个 token 与参考模型的 KL 散度
4. **Outcome-based Reward**：结合规则奖励（数学：答案正确性）+ 格式奖励（CoT 格式）

### Reward 设计（针对数学推理）
```python
def compute_reward(response, ground_truth):
    # 规则奖励：答案是否正确
    if extract_answer(response) == ground_truth:
        return 1.0
    # 格式奖励：是否包含 <think>...</think> 标签
    elif has_think_tags(response):
        return 0.1
    else:
        return 0.0
```

## 实验结论

**DeepSeek-Math 7B GRPO 训练结果：**

| 模型 | MATH | GSM8K | AMC23 |
|------|------|-------|-------|
| DeepSeek-Math-7B (SFT only) | 46.8% | 82.9% | - |
| DeepSeek-Math-7B-RL (GRPO) | **52.0%** | **88.2%** | **40.4%** |
| GPT-4 (reference) | 42.5% | 92.0% | - |

**GRPO vs PPO 工程对比：**
- 显存：GRPO 节省约 45%（无 Critic 模型）
- 训练速度：GRPO 快约 30%（无 Critic 前向传播）
- 收敛稳定性：GRPO 方差更低，超参数更鲁棒

**DeepSeek-R1 系列扩展结果：**
- R1-Zero（纯 GRPO，无 SFT）自发展现出链式思维推理
- R1（SFT 冷启动 + GRPO）达到 OpenAI o1 水平

## 工程落地要点

**1. GRPO 超参数设置参考**
```yaml
# 常用配置（7B-70B 模型）
G: 8              # 每个问题采样数，越大越稳定，计算越贵
ε: 0.2            # clip 系数（同 PPO）
β: 0.01           # KL 惩罚系数（小值鼓励探索）
lr: 1e-6          # 学习率（比 SFT 低10x）
batch_size: 256   # 有效 batch = 256 × G = 2048 responses
```

**2. 采样策略优化**
- 每个问题采样 G=8 个回答，需要 G 次 policy forward pass
- 可用 vLLM 批量生成（高吞吐），显著加速采样阶段
- 温度建议 0.7-1.0（需要足够多样性，否则组内 reward variance 为 0）

**3. 数据质量**
- GRPO 对数据质量敏感：需要**可验证**的 reward（数学、代码）
- 避免开放性生成任务（reward 难以定义，组内比较无意义）
- 训练集需要覆盖难度梯度（太简单 → 全对，方差为 0；太难 → 全错）

**4. 训练稳定性**
- 监控 KL divergence（超过 0.1 需降低 learning rate）
- 监控组内 reward variance（接近 0 说明问题太简单或太难）
- 建议先 SFT 冷启动再 GRPO（避免早期探索崩溃）

## 面试考点

- Q: GRPO 和 PPO 的核心区别是什么？
  A: PPO 需要独立的 Critic 模型估计值函数（V function），以 V 值作为 baseline 计算 advantage。GRPO 对同一问题采样 G 个回答，用组内 reward 的均值和标准差归一化来估计 advantage，无需 Critic 模型，节省 50% 显存，同时组统计比单次 V 函数估计方差更低。

- Q: GRPO 为什么特别适合数学和代码任务？
  A: 这类任务有明确的 reward 信号（答案对/错、代码通过/不通过测试），可以设计规则化 reward 函数，无需人工标注偏好数据。组内 G 个回答的对错分布自然形成对比，advantage 估计准确。开放性任务（写作、对话）reward 难以量化，组内比较缺乏明确依据，GRPO 效果会下降。

- Q: DeepSeek-R1 为什么能自发产生链式思维（CoT）？
  A: 纯 GRPO 训练（R1-Zero）中，模型发现展开推理步骤（<think>...详细计算...</think>）能显著提升最终答案正确率，因此 reward 更高。在 RL 优化压力下，模型自发学会了"先想后答"策略，这种行为被 reward 强化后稳定涌现，无需人工设计 CoT 格式或提示。
