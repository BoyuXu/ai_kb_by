# PerSRec: Efficient Sequential Recommendation via Personalization

- **Type**: Research Paper (Facebook Research)
- **URL**: https://arxiv.org/abs/2601.03479

## 核心贡献

通过段式压缩将长用户序列编码为可学习token，解决Transformer二次复杂度问题。

## 关键技术

- **段式压缩**: 长序列分段，每段压缩为segment embedding
- **可学习token**: 减少计算成本的同时保持精度
- **兼容性**: 适用于HSTU/HLLM等现有Transformer模型

## 面试考点

长序列建模效率，注意力机制优化，序列推荐模型架构。
