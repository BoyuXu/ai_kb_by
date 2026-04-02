# Continuous Batching and Dynamic Memory Management for High-Throughput LLM Serving
> 来源：https://arxiv.org/search/?query=continuous+batching+dynamic+memory+LLM+serving&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
传统LLM serving使用静态batching，等待整个batch完成才接受新请求，导致GPU利用率低（<30%）。Continuous Batching和PagedAttention（vLLM的核心）解决这个问题，将GPU利用率提升至>80%。

## 核心方法与创新点
- Continuous Batching（Orca）：请求完成后立即加入新请求，无需等整个batch
- PagedAttention（vLLM）：KV cache用页表管理，消除内存碎片
- 动态内存分配：按需分配KV cache页，避免过度分配导致OOM
- Chunked Prefill：将长prompt分块处理，平衡prefill和decode阶段

## 实验结论
Continuous Batching相比静态batching，吞吐量提升约3x；vLLM相比FasterTransformer，吞吐量提升约24x；PagedAttention消除了约60%的KV cache碎片；可以支持更大batch size。

## 工程落地要点
- vLLM是目前工业界最广泛使用的LLM serving框架
- Chunked prefill和decode的资源争抢需要精细调优
- 多模型serving（多LoRA）需要特殊的KV cache隔离策略

## 常见考点
1. **Q: Continuous Batching（Orca）解决什么问题？** A: 传统静态batching需要等最长序列完成，新请求必须等待；continuous batching允许即时加入
2. **Q: PagedAttention的核心思想？** A: 借鉴OS虚拟内存，KV cache用固定大小的块（page）管理，逻辑连续物理分散
3. **Q: LLM推理的两个阶段prefill和decode各自的特点？** A: Prefill：处理prompt，计算密集（compute-bound）；Decode：逐token生成，内存密集（memory-bound）
4. **Q: 如何提升LLM推理的GPU利用率？** A: Continuous batching、Tensor并行、Flash Attention、Speculative Decoding
5. **Q: vLLM与SGLang的主要差异？** A: vLLM通用性好；SGLang针对结构化输出（JSON/grammar）有基于constraint的优化
