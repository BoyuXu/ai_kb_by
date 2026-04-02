# LLM 推理效率三角：训练优化 + 架构优化 + 解码优化

> 📚 参考文献
> - [Speculative Decoding Draft Alignment](../papers/daily/20260322_speculative_decoding_draft_alignment.md) — Efficiently Aligning Draft Models for Speculative Decoding
> - [Grpo-Group-Relative-Policy-Optimization-Llm-Rea...](../papers/daily/20260321_grpo-group-relative-policy-optimization-llm-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Efficiently-Aligning-Draft-Models-Via-Parameter...](../papers/daily/20260321_efficiently-aligning-draft-models-via-parameter-and-data-efficient-adaptation-for-speculative-decoding.md) — Efficiently Aligning Draft Models via Parameter- and Data...
> - [Vllm-Paged-Attention](../papers/daily/20260317_vllm-paged-attention.md) — vLLM PagedAttention：LLM 推理内存管理革命
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../papers/daily/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Efficiently-Aligning-Draft-Models-Speculative-D...](../papers/daily/20260321_efficiently-aligning-draft-models-speculative-decoding-peft.md) — Efficiently Aligning Draft Models via Parameter- and Data...
> - [Recurrent-Drafter-Speculative-Decoding](../papers/daily/20260319_recurrent-drafter-speculative-decoding.md) — Recurrent Drafter for Fast Speculative Decoding
> - [Grpo Group Relative Policy Optimization](../papers/daily/20260322_grpo_group_relative_policy_optimization.md) — GRPO: Group Relative Policy Optimization for Large Langua...

**一句话**：让 LLM 变快有三条路：让训练更高效（GRPO 省 Critic）、让模型更精简（MoE 稀疏激活）、让解码更聪明（Speculative Decoding + FlashAttention-3）。今天把这三条路串起来看。

## 📐 核心公式与原理

### 1. Self-Attention

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- Transformer 核心计算

### 2. KV Cache

$$
\text{Memory} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times dtype\_size
$$

- KV Cache 内存占用公式

### 3. LoRA

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

- 低秩适配，r << d 大幅减少可训练参数

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

## 常见考点

**Q：GRPO 和 PPO 在数学推理上谁更好？**  
答：GRPO 在可验证 reward 任务上效果不差于 PPO，且计算成本更低（省去 Critic）。DeepSeek-Math-7B 用 GRPO 在 MATH 基准达到 51.7%，超过 PPO 基线。但 GRPO 的「组相对」方式需要 G 个回答中有明确好坏对比，如果所有回答 reward 相似则梯度信号弱。

**Q：Speculative Decoding 的加速上限是什么？**  
答：理论加速 ≈ 1/(1 - acceptance_rate)。接受率 83% 时，理论最高 ~6×；实际因为验证开销和内存传输，通常 2-3×。接受率是关键瓶颈，所以 draft 对齐对实际加速非常重要。

**Q：MoE 模型为什么比 Dense 更适合长上下文？**  
答：MoE 的每层计算量只有 Dense 的 K/E 倍（K=2, E=8 时只有 1/4），可以叠加更多层，总参数量更大但单 token 计算量不变。长上下文场景下 Attention 计算是瓶颈（O(n²)），而 FFN 部分 MoE 可以大幅节省，整体性价比更高。

### Q1: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q2: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q3: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q4: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。

### Q5: MoE 的优势和挑战？
**30秒答案**：优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。

### Q6: PagedAttention（vLLM）的核心思想？
**30秒答案**：借鉴操作系统虚拟内存分页，将 KV Cache 切分为固定大小的「页」，按需分配。解决传统方式预分配最大序列长度导致的内存浪费（平均浪费 60-80%）。

### Q7: Continuous Batching 是什么？
**30秒答案**：传统 Static Batching 等最长序列完成才处理下一批。Continuous Batching 每个 token step 都可以加入新请求，序列完成立即释放。将 GPU 利用率从 ~30% 提升到 ~80%。

### Q8: GRPO 和 PPO 的核心区别？
**30秒答案**：PPO 需要 value network 估计 advantage；GRPO 用 group 内的相对奖励替代 value network：采样 G 个输出，用组内排名作为 baseline。更简单、更稳定、不需要额外模型。

### Q9: RAG vs Fine-tuning 怎么选？
**30秒答案**：RAG：知识频繁更新、需要引用来源、不想改模型。Fine-tuning：任务固定、需要特定风格/格式、追求最低延迟。两者可结合：fine-tune 后的模型 + RAG 检索。

### Q10: LLM 推理的三大瓶颈？
**30秒答案**：①Prefill 阶段：计算密集（大量矩阵乘）；②Decode 阶段：内存密集（KV Cache 读写）；③通信：多卡推理时的 AllReduce。优化方向：FlashAttention（①）、PagedAttention（②）、TP/PP 并行（③）。


## 📐 核心公式直观理解

### 公式 1：推理效率三角约束

$$
\text{Quality} \times \text{Throughput} \times \text{Latency}^{-1} \leq C_{\text{hardware}}
$$

**直观理解**：推理部署永远面临三选二——要低延迟就牺牲吞吐（小 batch），要高吞吐就容忍高延迟（大 batch），要好质量就上大模型（慢且贵）。所有优化技术都是在这个三角上找更好的帕累托前沿。

### 公式 2：Arithmetic Intensity 与硬件瓶颈判断

$$
I = \frac{\text{FLOPs}}{\text{Bytes\_accessed}} \quad \Rightarrow \quad \begin{cases} I > I_{\text{roofline}}: & \text{compute-bound} \\ I < I_{\text{roofline}}: & \text{memory-bound} \end{cases}
$$

- $I_{\text{roofline}} = \frac{\text{Peak FLOPS}}{\text{Peak BW}}$：硬件的计算/带宽比

**直观理解**：就像搬砖——如果砖多人少（compute-bound），加人（更多计算单元）有效；如果路太窄搬不快（memory-bound），拓宽路（更高带宽）有效。LLM decode 阶段几乎都是 memory-bound（每个 token 只做一次矩阵-向量乘，计算量小但要读整个 KV Cache）。

### 公式 3：Token/s 和 TTFT 的关系

$$
\text{TTFT} = \frac{N_{\text{prompt}} \times \text{FLOPs\_per\_token}}{\text{GPU\_FLOPS}} + T_{\text{overhead}}
$$

**直观理解**：TTFT 主要由 prefill 阶段决定——prompt 越长首 token 越慢。这就是为什么 RAG 系统（prompt 动辄上万 token）特别关注 TTFT 优化：FlashAttention 减少 IO、Chunked Prefill 平滑计算、Prefix Caching 避免重复计算。

