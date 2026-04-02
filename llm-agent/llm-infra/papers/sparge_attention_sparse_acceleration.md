# SpargeAttention: Accurate Sparse Attention for Plug-and-Play Acceleration

> 来源：GitHub thu-ml/SpargeAttn (清华) | 领域：llm-infra | 学习日期：20260402

## 问题定义

LLM 推理中注意力计算是主要瓶颈，时间复杂度为 O(n²)。现有稀疏注意力方法（如 sliding window、local attention）通过预定义模式减少计算，但可能丢失重要的长距离依赖。SpargeAttention 实现精确的动态稀疏注意力加速。

## 核心方法与创新点

1. **动态稀疏性**：运行时动态判断哪些注意力分数接近零并跳过计算，而非使用固定稀疏模式
2. **即插即用**：直接替换 FlashAttention，无需修改模型架构或重新训练
3. **精度保证**：通过块级重要性评估预判稀疏位置，对重要区域仍执行精确计算
4. **GPU 优化**：利用 GPU 的 warp-level 并行性实现高效的稀疏计算内核

## 实验结论

- 在 LLaMA-3-70B 上推理速度提升 1.8-2.5 倍
- 输出质量与 FlashAttention 几乎无差异（perplexity 差异 < 0.01）
- 在长序列（>8K token）上加速效果更显著

## 工程落地要点

- **硬件适配**：当前实现主要优化 NVIDIA Ampere/Hopper 架构
- **序列长度**：短序列（<2K）加速不明显，长序列收益大
- **兼容性**：支持 HuggingFace Transformers 和 vLLM 集成

## 面试考点

1. **Q：动态稀疏 vs 静态稀疏注意力的区别？**
   A：静态稀疏（如 sliding window）预定义注意力模式，简单但可能丢信息；动态稀疏根据实际注意力分数运行时决定稀疏位置，更精确。
2. **Q：为什么能做到即插即用不影响精度？**
   A：通过块级重要性评估预判哪些区域注意力分数接近零，只跳过这些区域。重要区域仍用精确计算。
3. **Q：FlashAttention 和 SpargeAttention 的关系？**
   A：FlashAttention 优化内存访问但计算量不变（仍是 O(n²)）；SpargeAttention 在此基础上跳过不重要的计算块，实际减少计算量。
