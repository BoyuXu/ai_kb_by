# Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs
> 来源：arXiv:2604.04722 | 领域：llm-infra | 学习日期：20260419

## 问题定义
端侧LLM推理受KV Cache内存和带宽制约，KV Cache随上下文长度线性增长，常主导解码开销。现有固定精度量化"一视同仁"，浪费bits在低重要性token上，同时过度压缩关键token。

## 核心方法与创新点
1. **Token-wise自适应量化**：按token重要性动态分配精度 {2-bit, 4-bit, 8-bit, FP16}
2. **轻量特征提取**：token频率 + quality score + attention variance + entropy-based uncertainty
3. **数据驱动控制器**：小型MLP学习最优精度分配策略
4. **在线解码集成**：量化决策在autoregressive解码过程中实时进行

## 实验结论
- SmolLM-360M on HellaSwag：解码延迟降低 **17.75%**，准确率提升 **7.60点**
- 与FP16仅差 **0.30点**

## 面试考点
- Q: KV Cache量化的主流方法？
  - A: ①KIVI（per-channel/per-token量化）；②KVQuant（非均匀量化+离群值保护）；③MiniKV（2-bit极限压缩）；④自适应量化（本文）
- Q: 为什么Key和Value需要不同的量化策略？
  - A: Key参与注意力分数计算（对误差敏感），Value参与加权求和（对误差更鲁棒）→ Key需要更高精度
