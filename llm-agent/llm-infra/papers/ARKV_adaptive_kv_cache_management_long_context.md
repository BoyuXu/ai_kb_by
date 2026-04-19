# ARKV: Adaptive Resource-Efficient KV Cache Management for Long-Context LLM Inference
> 来源：arXiv:2603.08727 | 领域：llm-infra | 学习日期：20260419

## 核心方法
1. **有限内存预算下的自适应管理**：动态分配精度级别给cached tokens
2. **Per-layer Attention Dynamics**：根据每层注意力模式动态调整
3. **Token-level Importance**：基于attention score的token重要性评估
4. **长上下文推理优化**：专为长文本场景设计

## 面试考点
- Q: 长上下文推理的核心瓶颈？
  - A: ①KV Cache内存线性增长；②注意力计算O(n²)；③缓存淘汰策略影响质量
