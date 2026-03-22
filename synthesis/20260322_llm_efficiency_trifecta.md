# LLM 推理效率三角：训练优化 + 架构优化 + 解码优化

**一句话**：让 LLM 变快有三条路：让训练更高效（GRPO 省 Critic）、让模型更精简（MoE 稀疏激活）、让解码更聪明（Speculative Decoding + FlashAttention-3）。今天把这三条路串起来看。

---

## 第一条路：训练优化（GRPO）

**核心问题**：PPO 需要 Critic 模型（与 policy 同规模），实际显存 = 2× policy 模型

**GRPO 解法**：用「组内相对比较」替代 Critic 的绝对价值估计
```
对同一 prompt 采样 G=8 个回答
advantage_i = (reward_i - mean({reward_j})) / std({reward_j})
```
- 组内打分：不需要 Critic 告诉你「这个回答值多少分」
- 只需要知道「这个回答比组内平均好多少」
- 显存节省 ~40%（省去整个 Critic）

**适用场景**：可验证 reward 的任务（数学/代码 = 0/1 reward），不适合开放对话

**今日新细节**（vs 昨日 GRPO 笔记）：
- GRPO 已被 DeepSeek-Math、DeepSeek-R1 用于生产
- 规则 reward = 答案正确性(0/1) + 格式奖励（CoT 格式完整性）
- Token-level KL 惩罚（与参考模型差距不能太大）

---

## 第二条路：架构优化（MoE）

**核心问题**：Dense LLM 推理时所有参数参与计算，scaling 困难

**MoE 解法**：每个 token 只激活 top-K 专家（通常 K=2/E=8）
```
参数量：E 个专家 × 专家大小 = 大（如 56B）
激活量：K 个专家 × 专家大小 = 小（等效 7B）
```

**工程关键挑战**：
1. **专家路由开销**：All-to-All 通信（不同专家在不同 GPU）
2. **负载均衡**：辅助损失防止所有 token 路由到同一专家
3. **冷热专家**：热门专家常驻 GPU，冷门专家卸载到 CPU

**和今日其他内容的连接**：
- FlashAttention-3 + MoE = 双重加速（Attention 计算 + 前向 FFN 都更快）
- 长上下文 LLM 一般配合 MoE（因为参数效率高，可以用更多层）

---

## 第三条路：解码优化（Speculative Decoding）

**核心问题**：LLM 自回归解码，每步都要调用完整大模型，GPU 利用率低（通常 <30%）

**Speculative Decoding 解法**：
```
小模型（草稿模型，draft）: 快速生成 K 个 token（如 K=5）
大模型（目标模型，target）: 并行验证这 K 个 token
  → 如果 draft token 被接受：省了 K-1 次大模型调用
  → 如果 draft token 被拒绝：回退到大模型生成
```

**今日新内容：Draft Model 对齐**
- 问题：未专门训练的小模型接受率只有 62%（理想 >85%）
- 解法：用 LoRA 对小模型进行 KL 散度对齐（向大模型分布靠近）
- 训练数据：大模型的合成输出（self-play）
- 效果：接受率 62% → 83%，端到端加速 1.8× → 2.7×

---

## 三条路的协同效应

```
LLM 推理效率提升路线图：

训练侧：GRPO（省 Critic，更快迭代） → 更快训练出好模型
                                         ↓
架构侧：MoE（稀疏激活，参数效率） → 更大参数量 + 更低计算量
                                         ↓  
硬件侧：FlashAttention-3（H100 深度优化） → Attention 吞吐 2× 
                                         ↓
解码侧：Speculative Decoding（draft对齐） → 推理 2.7×
                                         ↓
系统侧：长上下文（KV Cache 压缩）→ 支持更长输入
```

全栈叠加（理想估计）：~5-8× 端到端推理加速，vs 2022 年基线

---

## 技术演进脉络

```
2022 GPT-3 时代：Dense LLM，朴素自回归，无优化
    ↓ 规模扩大带来效率压力
2023 FlashAttention-2、Speculative Decoding 论文
     vLLM PagedAttention
    ↓ H100 硬件发布，新架构需要
2024 MoE 主流化（Mixtral, DeepSeek-V2）
     GRPO（DeepSeek-Math，无 Critic RL）
     FlashAttention-3（H100 深度优化）
    ↓ 生产规模部署
2025 Draft 对齐（Speculative Decoding 工程化）
     可变长度 KV Cache（长上下文 + 内存优化）
```

---

## 面试考点

**Q：GRPO 和 PPO 在数学推理上谁更好？**  
答：GRPO 在可验证 reward 任务上效果不差于 PPO，且计算成本更低（省去 Critic）。DeepSeek-Math-7B 用 GRPO 在 MATH 基准达到 51.7%，超过 PPO 基线。但 GRPO 的「组相对」方式需要 G 个回答中有明确好坏对比，如果所有回答 reward 相似则梯度信号弱。

**Q：Speculative Decoding 的加速上限是什么？**  
答：理论加速 ≈ 1/(1 - acceptance_rate)。接受率 83% 时，理论最高 ~6×；实际因为验证开销和内存传输，通常 2-3×。接受率是关键瓶颈，所以 draft 对齐对实际加速非常重要。

**Q：MoE 模型为什么比 Dense 更适合长上下文？**  
答：MoE 的每层计算量只有 Dense 的 K/E 倍（K=2, E=8 时只有 1/4），可以叠加更多层，总参数量更大但单 token 计算量不变。长上下文场景下 Attention 计算是瓶颈（O(n²)），而 FFN 部分 MoE 可以大幅节省，整体性价比更高。
