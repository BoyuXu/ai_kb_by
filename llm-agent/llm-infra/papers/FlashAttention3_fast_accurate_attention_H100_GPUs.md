# FlashAttention-3: Fast and Accurate Attention on H100 GPUs
> 来源：arXiv:2407.08608 | 领域：llm-infra | 学习日期：20260330

## 问题定义
FlashAttention-2 在 A100 GPU 上达到理论峰值 70% FLOP 利用率，但在 H100 上因架构差异（Tensor Core 新指令、异步计算）未能充分利用。FlashAttention-3 针对 H100 的 Hopper 架构特性（WGMMA 指令、TMA、FP8）重新设计，达到 H100 峰值 FLOP 的 75%+。

## 核心方法与创新点
1. **Producer-Consumer Asynchrony**：H100 引入 TMA（Tensor Memory Accelerator）异步数据搬移，FlashAttention-3 将 IO（从 HBM 到 SRAM）和计算（WGMMA）重叠，消除等待气泡。
2. **WGMMA 指令**：利用 H100 的 Warpgroup-level Matrix Multiply-Accumulate 指令，允许整个 warpgroup（4 warp）协作完成矩阵乘，提升 SM 利用率。
3. **Intra-warpgroup Pipelining**：在单个 warpgroup 内流水线化数学运算和内存访问，双缓冲消除数据等待。
4. **FP8 支持**：实现精确 FP8 attention（8-bit 浮点），在 H100 上 FP8 Tensor Core 速度是 BF16 的 2x，FlashAttention-3 FP8 达到 1.5 PetaFLOP/s。
5. **Softmax Accuracy**：FP8 精度下 softmax 数值稳定性是挑战，通过 block-wise rescaling 保证精度损失 <0.1%。

## 实验结论
- BF16 attention（H100）：1.0 PetaFLOP/s（约峰值 75%），FlashAttention-2 为 0.67 PetaFLOP/s
- FP8 attention（H100）：1.5 PetaFLOP/s（约峰值 75% FP8 峰值）
- 序列长度 8K：推理吞吐相比 FlashAttention-2 提升 1.5x-1.8x

## 工程落地要点
- WGMMA 指令需要 CUDA 12.3+、H100/H800 GPU，不向下兼容 A100
- FP8 attention 精度损失可接受（<0.1%），但需要模型权重也是 FP8 才能完全发挥
- 与 vLLM/TensorRT-LLM 集成：需要更新 attention kernel 调用接口
- 自定义 attention（如 Sliding Window、Cross-attention）需手动移植 FlashAttention-3 模板

## 面试考点
- Q: FlashAttention 的核心思想？
  - A: 传统 attention 需要将 Q×K^T 的完整 attention 矩阵存到 HBM（IO 密集）；FlashAttention 将 Q/K/V 分块（tiling）在 SRAM 内完成计算，大幅减少 HBM 读写，attention 从 IO-bound 变为 compute-bound
- Q: FlashAttention 如何处理 softmax 的归一化（需要全局 max/sum）？
  - A: 在线 softmax（online softmax algorithm）：逐块计算时维护 running max $m$ 和 running sum $l$，每处理新块时用公式修正前面的结果，最终等价于全局 softmax
- Q: H100 相比 A100 在 attention 加速的主要新特性？
  - A: ① TMA 异步 IO；② WGMMA（warpgroup 级矩阵乘，比 A100 的 wmma 更大且更快）；③ FP8 Tensor Core（速度 2x）；④ 更大 L2 Cache

## 数学公式

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

$$
\text{Online Softmax: } m_i = \max(m_{i-1}, \max_j s_{ij}), \quad l_i = e^{m_{i-1}-m_i} l_{i-1} + \sum_j e^{s_{ij}-m_i}
$$
