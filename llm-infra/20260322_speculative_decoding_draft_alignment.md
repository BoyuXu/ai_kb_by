# Efficiently Aligning Draft Models for Speculative Decoding

> 来源：arxiv | 日期：20260322 | 领域：LLM工程

## 问题定义

投机解码（Speculative Decoding）通过小模型（draft model）预生成 token 序列，大模型（target model）并行验证，可显著加速推理。但 draft model 和 target model 的输出分布不对齐会导致接受率（acceptance rate）低，加速效果差。如何高效对齐 draft model 使其分布尽量接近 target model？

## 核心方法与创新点

- **问题分析**：Draft model 未专门为 target model 训练，分布差异导致接受率 40-70%（理想应 >85%）
- **参数高效对齐（PEFT-based Alignment）**：
  - 用 LoRA 对现有小模型进行轻量级微调，使其分布向 target model 靠近
  - 训练目标：最小化 draft model 和 target model 输出分布的 KL 散度
  - 训练数据：用 target model 生成的合成数据（self-play）
- **数据高效对齐**：
  - 用 DPO（Direct Preference Optimization）代替 KL 蒸馏，只需相对偏好数据（无需 logprob）
  - 选取 draft model 接受率低的样本重点训练（难例挖掘）
- **层级对齐**：除最终输出分布，还对中间隐藏层表示做对齐（类似层级 KD）

## 实验结论

- 接受率提升：从基础小模型的 62% 提升到对齐后的 83%（LLaMA-7B draft + LLaMA-70B target）
- 端到端推理加速：从 1.8× 提升到 2.7×（vs 无 speculative decoding 的 1×）
- PEFT 训练成本：只需 target model 推理 1000 个样本，LoRA 微调 1 小时（A100）
- 无损加速：验证机制保证输出分布与 target model 完全一致

## 工程落地要点

- **Draft Model 选择**：建议与 target model 同系列（如 LLaMA-7B draft + LLaMA-70B target），共享 tokenizer 和词汇表
- **投机长度（γ）**：建议 4-8 个 token，过长导致接受率下降，过短加速收益小
- **批量推理**：Speculative Decoding 在 batch size >1 时效率下降（因为不同请求的接受 token 数不一致），适合单请求低延迟场景
- **硬件优化**：Draft model 和 target model 共驻 GPU，需规划显存分配（draft 通常占 target 的 10-20%）

## 面试考点

1. **Q：投机解码的核心原理是什么？为什么能加速？**
   A：Draft model 并行生成 K 个 token（速度快），target model 并行验证（一次 forward pass 验证 K 个 token，计算量不变但吞吐提升）。当接受率高时，相当于每次 forward 产出多个 token

2. **Q：为什么需要对齐 draft model？不对齐会怎样？**
   A：不对齐时接受率低（<60%），大量 draft token 被拒绝，退化为接近自回归的效率；极端情况下比直接用 target model 还慢（因为多了 draft model 的开销）

3. **Q：投机解码在哪些场景下加速效果最好？**
   A：单请求低延迟场景（interactive chatbot）；draft 和 target 分布接近的任务（代码补全、固定格式输出）；target model 参数量远大于 draft model（10x+）
