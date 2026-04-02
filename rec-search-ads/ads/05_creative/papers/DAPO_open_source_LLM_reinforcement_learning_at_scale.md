# DAPO: An Open-Source LLM Reinforcement Learning System at Scale
> 来源：https://arxiv.org/abs/2503.14476 | 领域：ads | 学习日期：20260329

## 问题定义

大规模 LLM 强化学习（RL）训练面临的核心挑战：

1. **技术细节封闭**：OpenAI o1、DeepSeek R1 等顶尖推理模型的 RL 训练细节未公开，社区难以复现
2. **训练不稳定**：大规模 RL 训练容易出现策略崩溃、奖励爆炸、梯度爆炸等问题
3. **可扩展性差**：现有 RL 算法（PPO、GRPO）在长序列推理任务上效率低下
4. **推理能力上限**：仅靠 SFT 无法充分激活模型的复杂推理潜力

## 核心方法与创新点

### DAPO 算法：四大关键技术

**① 解耦裁剪（Decoupled Clip）**

传统 PPO 使用统一裁剪系数 ε，DAPO 解耦正负样本的裁剪策略：

$$
\mathcal{L}_{DAPO} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high})\hat{A}_t\right)\right]
$$

- 正优势样本（$\hat{A}_t > 0$）：使用较大 $\varepsilon_{high}$，允许更大幅度的策略改进
- 负优势样本（$\hat{A}_t < 0$）：使用较小 $\varepsilon_{low}$，防止策略过度退化

**② 动态采样（Dynamic sAmpling Policy Optimization）**

- 监控训练批次中的样本有效性（有效样本 = 策略真正学到知识的样本）
- 动态丢弃"太难"或"太容易"的样本（避免梯度信号退化）
- 维持合适的探索-利用平衡

**③ Token 级策略梯度（Token-Level Policy Gradient）**

替代传统的序列级策略梯度：

$$
\mathcal{L}_{token} = -\frac{1}{\sum_i |y_i|} \sum_{i,t} \hat{A}_i \log \pi_\theta(y_{i,t}|x_i, y_{i,<t})
$$

**④ 过长惩罚（Overlong Penalty）**

推理模型倾向于生成超长推理链（reward hacking），引入长度惩罚避免无效冗余推理。

### 开源系统组件

基于 **verl** 框架实现，完整开源：
- 训练代码
- 精心策划和处理的训练数据集
- 模型权重（Qwen2.5-32B base model）

## 实验结论

| 基准测试 | 结果 |
|---------|------|
| AIME 2024 | **50分**（使用 Qwen2.5-32B base model） |
| 对比 DeepSeek-R1-Zero | 性能相当，但训练稳定性更好 |

AIME 2024 50分是数学推理能力的重要里程碑，证明开源 RL 系统可以复现顶尖推理能力。

## 工程落地要点（广告系统应用视角）

1. **广告推荐中的 RL 训练稳定性**：DAPO 的解耦裁剪和动态采样对广告 RL 同样有效（GR2 等重排RL就是使用 DAPO）
2. **Reward Hacking 防范**：过长惩罚的思路可迁移到广告 RL（防止模型生成冗长无用的推理来刷高奖励）
3. **Token 级梯度**：适合推荐场景中的序列生成任务（slate recommendation、list generation）
4. **开源基础设施**：verl 框架直接可用于广告重排 RL 训练，降低工程门槛
5. **动态采样**：广告 RL 中也存在样本质量不均匀问题，动态采样可提升训练效率

## 常见考点

**Q1: DAPO 的"解耦裁剪"相比 PPO 的标准裁剪解决了什么问题？**
A: PPO 用统一 ε 裁剪正负样本，导致：①正样本（有利动作）更新幅度受限，学习慢 ②负样本（不利动作）可能被过度惩罚导致策略崩溃。DAPO 解耦后：大 ε_high 让正样本充分学习，小 ε_low 稳健地惩罚负样本，同时保证训练稳定。

**Q2: 动态采样（Dynamic Sampling）在 LLM RL 训练中如何工作？为什么重要？**
A: 监控每个训练批次的有效样本比例（通常定义为策略概率比 r_t 在 clip 范围外的比例）。太多样本已经"学会了"（全部在 clip 内）→ 增大难度；太多样本"太难"（全部 clip 住）→ 降低难度。重要性：避免无效计算，保持有效的梯度信号。

**Q3: Token 级策略梯度相比序列级有什么优势？**
A: 序列级只用最终奖励反馈整个序列，长序列中早期 token 获得的梯度信号稀疏且延迟大（信用分配问题）。Token 级在每个 token 上计算优势函数，梯度信号更密集，训练更高效，特别适合推理链长的场景。

**Q4: 为什么 LLM 推理模型需要防范"过长推理链"（Overlong Reasoning）？**
A: 模型发现生成长推理链即使错误也可能获得部分奖励，或通过冗长重复来"显示自信"。Overlong Penalty：超过阈值长度的 token 给予负奖励惩罚，强制模型高效推理。

**Q5: DAPO 在广告系统的重排 RL（如 GR2）中有哪些直接应用价值？**
A: ①解耦裁剪提升广告重排 RL 训练稳定性（广告场景奖励更嘈杂）②动态采样适应广告流量分布变化 ③Overlong Penalty 防止重排 LLM 生成冗余推理 ④开源 verl 框架直接降低广告 RL 系统搭建成本 ⑤Token 级梯度适合广告 slate generation 任务
