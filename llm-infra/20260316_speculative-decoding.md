# 投机解码（Speculative Decoding）：加速 LLM 推理的黑科技

> 来源：arxiv (Google 2023) | 日期：20260316 | 领域：llm-infra

## 问题定义

LLM 自回归生成的根本瓶颈：每次生成一个 token，需要执行一次完整的前向传播。对于长序列生成（如摘要、翻译、代码生成），推理延迟由 **token 数量**决定，而非计算量本身：

```
生成时间 = num_tokens × 单token生成时间
         ≈ 100 tokens × 50ms/token = 5000ms（太慢！）
```

传统方案（KV cache、量化）优化单 token 生成时间，但改进有限（约 2-3x）。投机解码提供新思路：**预测多个候选 token，批量验证**。

## 核心方法与创新点

### 投机解码流程

1. **草稿阶段（Draft）**：用轻量级模型（如 Llama-7B）快速生成 K 个候选 token。
2. **验证阶段（Verify）**：大模型（Llama-70B）一次前向传播验证所有 K 个 token 的概率。
3. **接受或回滚**：
   - 若所有 token 概率都满足条件，全部接受，推进 K 步。
   - 若某个 token 被拒绝，回滚到该位置，大模型自己生成该 token。

### 数学形式

设 p_small(token_i | prefix) 和 p_large(token_i | prefix) 分别为小模型和大模型的概率。

接受条件（Rejection Sampling）：
```
if p_large(token_i) > p_small(token_i):
    accept token_i
else:
    accept with probability = p_large(token_i) / p_small(token_i)
    否则 reject，重新采样
```

这个机制保证 **验证后的分布完全等同于大模型的分布**（无偏），不产生额外误差。

### 实际加速分析

```
时间成本 = K×(小模型推理时间) + 1×(大模型验证时间)

例子：
- 小模型 7B：单 token 5ms，生成 K=8 个需要 40ms
- 大模型 70B：单 token 50ms，验证 8 个 token 只需 50ms（批量验证共享）
- 总计：40 + 50 = 90ms，生成 8 个 token
- 对比原方案：8 × 50 = 400ms
- 加速比：4.4x！
```

关键洞察：大模型可以**批量验证多个位置的 token**，是现有 KV cache 机制支持的。

## 实验结论

- LLaMA-70B 生成 100 token，用 LLaMA-7B 草稿（K=8）：端到端加速 **2-4x**。
- 生成任务越长、大小模型性能差距越大，加速效果越好（可达 5-6x）。
- 生成结果 **完全一致**，无质量损失。

## 工程落地要点

- **成本分析**：需要同时加载大小两个模型，显存占用 = 2x 单模型。在 80GB A100 上可装 70B+13B；如果显存不足，考虑量化小模型。
- **小模型选择**：不一定要同家族的小版本，可以用其他高效模型（如 Phi、MistralLite）作为 draft model，只要推理速度快即可。
- **K 值调优**：K 越大加速比越高，但 draft 错误累积增加；通常 K=4-16 是最优。
- **框架支持**：TensorRT-LLM、vLLM 都已支持 speculative decoding；Hugging Face transformers 暂不支持。

## 面试考点

- Q: 投机解码为什么不产生额外误差（biasing）？
  A: 采样 rejection sampling 的接受概率 = p_large / p_small，保证了事后概率分布严格来自大模型。数学上这是重要性采样（Importance Sampling）的应用，理论上无偏。

- Q: 为什么大模型能"批量验证"多个位置？不是应该 seq-to-seq 吗？
  A: 自回归生成时，token_i 的计算依赖 token_1...token_{i-1}，但如果 token_i 来自预测（草稿模型），大模型可以同时计算所有候选位置 1..K 的下一 token 概率（通过 causal mask）。一次前向传播就能验证 K 个位置，时间开销接近单个 token。

- Q: 小模型和大模型的"适配"程度会影响加速吗？
  A: 会的。如果小模型和大模型的概率分布相差很大，大模型会频繁拒绝小模型的预测，需要频繁进行额外采样（重新生成），加速效果降低。最优情况是小模型和大模型很接近（如 7B vs 70B，同家族），加速比达到 4x+；跨家族时可能只有 2-3x。
