# Kunlun: A Scalable Recommendation Architecture with Predictable Scaling Laws for Meta Ads

> 来源：arXiv 2603.04448 (Meta) | 领域：rec-sys | 学习日期：20260402

## 问题定义

大规模推荐系统的模型扩展缺乏可预测的 Scaling Laws——不像 LLM 领域有明确的参数-性能幂律关系。Meta 提出 Kunlun 架构，首次在工业级广告推荐系统中建立可预测的 Scaling Laws，使模型扩展的收益可预估。

## 核心方法与创新点

1. **可预测 Scaling Laws**：系统性研究模型参数量、训练数据量和计算量三者与推荐效果的幂律关系，建立预测公式
2. **Kunlun 架构**：基于 HSTU 改进的推荐 Transformer 架构，引入更高效的注意力机制和 Embedding 层设计
3. **MFU 优化**：Model FLOPs Utilization 从 17% 提升到 37%，使得 2 倍 Scaling 效率成为可能
4. **Embedding Scaling**：特别研究了 Embedding 表大小对推荐效果的 Scaling 关系，发现 Embedding 维度比表大小更关键

## 实验结论

- Scaling Laws 在 Meta 广告推荐系统上得到验证，预测误差 < 5%
- 模型参数扩大 10 倍，广告收入提升 1.2%（在 Meta 规模下是巨大收益）
- MFU 37% 使得大规模训练的成本可控

## 工程落地要点

- **Scaling 规划**：可以用小规模实验预测大规模模型的预期收益，指导资源分配
- **计算预算**：MFU 优化需要针对硬件（GPU/TPU）定制化的算子实现
- **增量训练**：大规模模型的全量训练成本高，需要高效的增量训练策略

## 常见考点

1. **Q：推荐系统的 Scaling Laws 和 LLM 的有什么不同？**
   A：推荐系统有大量稀疏特征（Embedding 表），Scaling 不仅涉及模型参数，还涉及 Embedding 维度和表大小。此外推荐的评估指标（CTR/CVR）比语言模型的 perplexity 更复杂。
2. **Q：MFU 为什么在推荐系统中特别低？**
   A：推荐模型有大量 Embedding 查找和稀疏操作，这些操作的计算密度低，硬件利用率差。Kunlun 通过算子融合和内存优化提升 MFU。
3. **Q：Embedding 维度 vs 表大小的 trade-off？**
   A：增大维度提升表征能力，增大表大小提升覆盖率。实验发现维度的边际收益更高，建议优先扩大维度。
