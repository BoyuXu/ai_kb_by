# 理解与优化多阶段 AI 推理管道

> 来源：Understanding and Optimizing Multi-Stage AI Inference Pipelines | 日期：20260318 | 领域：llm-infra

## 问题定义

生产 AI 系统通常由多个推理阶段组成：检索 → 重排序 → LLM 生成 → 数据提取，各阶段使用不同的模型（BM25、BERT、LLaMA、Claude）、硬件（CPU、GPU、TPU）、框架（Elasticsearch、PyTorch、vLLM）。如何理解这些异构系统的**端到端延迟瓶颈**和**整体吞吐量限制**？如何优化使得总延迟最小？

## 核心方法与创新点

1. **多阶段管道建模**：将 n 个串联的推理阶段建模为队列网络（Queueing Network）：
   - 每个阶段有自己的吞吐量 (λ_i，items/sec) 和延迟 (L_i)。
   - 整体吞吐量 λ_system = min(λ_1, λ_2, ..., λ_n)（最慢阶段决定）。
   - 整体延迟 L_system ≠ Σ L_i（需考虑排队延迟 queuing delay）。

2. **关键路径分析**：识别系统中的"瓶颈"（bottleneck stage），即 λ_i 最小的阶段。优化应优先优化瓶颈，优化非瓶颈收益有限。

3. **批处理的双刃剑**：增加 batch size 可提升吞吐量，但增加延迟（排队等待）。需要在目标延迟 SLA 约束下，找最大的 batch size。

4. **异构系统中的资源分配**：给定固定的计算预算（GPU 显存、CPU 核数），如何分配到各阶段以最小化端到端延迟？例如，是否应该用更大的 retrieval 模型（更精准）或更大的 reranker？

5. **动态调度**：不同查询的复杂度不同（短查询 vs 长查询），不同阶段的处理时间差异大，设计智能队列调度（SJF、优先级队列）而非 FIFO，降低平均延迟。

## 实验结论

- 在某搜索系统的 4 阶段管道（retrieval → reranking → LLM reading comprehension → summarization），LLM 阶段是瓶颈（吞吐量 10 req/sec），优化此阶段收益最大。
- Retrieval 延迟 (100ms) < LLM 延迟 (2s) 但吞吐量足够（100 req/sec），优化 retrieval 对端到端延迟贡献度低（< 5%）。
- 在 SLA 延迟 3s 的约束下，最优 batch size = 6（LLM），相应 reranker batch size = 12（4 个 rerank per retrieval result）。
- 优先级队列调度（按查询长度分级）相比 FIFO 降低 P95 延迟约 25%。

## 工程落地要点

- **延迟分解**：用分布式追踪（Jaeger、Datadog）分解每个阶段的延迟贡献，识别真实瓶颈而非推测。
- **批大小自适应**：根据 GPU 利用率和排队延迟动态调整 batch size，避免固定大小导致低利用率或高延迟。
- **阶段间数据格式转换**：格式转换（embedding 格式、序列化）往往是被忽视的开销，需要优化。
- **缓存跨阶段共享**：某些中间结果（如 embedding）可被多个下游阶段复用，缓存管理策略需要全局考虑。

## 面试考点

**Q: 多阶段系统的整体延迟为什么不等于各阶段延迟之和？**
- 存在排队延迟（Request 在队列中等待）和并行度不匹配导致的空闲（某阶段处理慢，下游无工作可做）。

**Q: 如何识别系统的瓶颈阶段？**
- 计算每阶段的吞吐量瓶颈 (λ_i)，最小的就是系统瓶颈；也可观察队列深度，瓶颈阶段的队列通常最长。

**Q: Batch Processing 和 Streaming 的 trade-off？**
- Batch：吞吐高，延迟高；Streaming：延迟低，吞吐低；RAG 通常混合（large batch for retrieval，small batch or streaming for LLM）。
