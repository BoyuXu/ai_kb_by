# Towards Large-scale Generative Ranking
> 来源：https://arxiv.org/abs/2505.04180 | 领域：rec-sys | 日期：20260323

## 问题定义
探索大规模工业推荐系统中生成式排序的可行性与挑战。重点研究如何将生成式模型应用于数百亿用户、数十亿物品的工业级排序任务，解决效率、质量和可扩展性的trilemma。

## 核心方法与创新点
- 分层生成架构：粗生成（beam search）→精细排序（pointwise rerank）
- 推理加速：speculative decoding、grouped query attention、flash attention
- 高效serving：prefix sharing、continuous batching、KV cache管理
- 混合精度：INT8/INT4量化降低显存和延迟

## 实验结论
在大规模推荐系统中，生成式排序相比经典DNN排序在NDCG上提升约3%；通过工程优化，P99 latency控制在50ms以内（符合工业要求）；量化损失约0.5% AUC。

## 工程落地要点
- P99 latency是工业排序的硬约束（通常≤50ms），生成式模型需要严格优化
- 候选集大小限制（通常50-500），超过此规模生成复杂度不可控
- 灰度实验时需要控制生成式和传统排序的流量分配

## 面试考点
1. **Q: 大规模生成式排序面临的最大挑战？** A: Latency（生成比MLP慢10-100x）、内存（KV cache大）、一致性（生成结果不确定性）
2. **Q: Speculative decoding如何降低推理延迟？** A: 小模型生成多个draft token，大模型并行验证，接受率高时大幅降低延迟
3. **Q: 工业排序的latency预算如何分配？** A: 特征查询（5ms）、特征处理（5ms）、模型推理（30ms）、后处理（10ms），总≤50ms
4. **Q: 量化对生成式推荐质量的影响？** A: INT8通常损失<0.5%，INT4损失1-2%，需要针对推荐任务做量化感知训练
5. **Q: 生成式排序如何与现有推荐系统集成？** A: 作为额外的重排层，或替换现有精排，需要影子模式（shadow mode）验证
