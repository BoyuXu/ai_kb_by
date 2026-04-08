# KVQuant: Towards 10 Million Context Length LLM Inference

> 来源：NeurIPS 2024 | 领域：llm-infra | 学习日期：20260408

## 问题定义

长上下文 LLM 推理中，KV Cache 是内存瓶颈：
- 上下文长度 N 增加时，KV Cache 线性增长：$\text{Memory}_{KV} = 2 \times L \times H \times d \times N \times \text{precision}$
- 100万 token 时，KV Cache 可达数百 GB

**核心问题**：如何通过量化大幅压缩 KV Cache，支持超长上下文推理？

## 核心方法与创新点

**KVQuant 三阶段方法**：

1. **Per-Channel Key Quantization**：
   - Key 矩阵在通道维度上分布差异大
   - 逐通道量化替代逐 token 量化，减少量化误差

2. **Pre-RoPE Key Quantization**：
   - RoPE 位置编码会扭曲 Key 分布
   - 在 RoPE 之前量化，推理时按需加 RoPE
   - 牺牲少量计算换取更好的量化质量

3. **Non-Uniform Quantization**：
   - 基于敏感度的非均匀量化码本
   - 对 outlier channel 采用更高精度

## 关键结果

- 3-bit 量化下困惑度降低 < 0.1
- LLaMA-7B 单 A100 支持 100 万 token
- 8 × A100 支持 1000 万 token 上下文

## 工程启示

- KV Cache 量化是长上下文推理的关键优化方向
- Pre-RoPE 量化思路可推广到其他位置编码方案
- 与 FlashAttention 等方法正交，可组合使用

## 面试考点

- KV Cache 量化的三个维度：精度、粒度、均匀性
- RoPE 对量化的影响及解决方案
- 长上下文推理的内存分析公式
