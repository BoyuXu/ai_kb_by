# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
> 来源：https://arxiv.org/abs/2501.01005 | 日期：20260319

## 问题定义
LLM推理服务中，注意力计算是性能瓶颈，且不同场景（prefill/decode、不同batch大小、不同KV Cache布局）需要不同优化策略。FlashAttention等固定实现难以覆盖推理服务的多样化需求。FlashInfer提供灵活、高性能的注意力计算引擎，支持多种KV Cache布局和推理场景。

## 核心方法与创新点
1. **JIT内核生成**：根据运行时参数（序列长度、batch大小、KV布局）即时编译最优CUDA内核，而非预编译固定内核
2. **统一KV Cache抽象**：支持Paged KV Cache（vLLM）、RadixTree KV Cache（SGLang）、连续KV Cache等多种布局，统一接口
3. **稀疏注意力支持**：原生支持Block-Sparse Attention、Sliding Window Attention、树形注意力（投机解码用）
4. **Cascade推理优化**：共享前缀（System Prompt）的KV Cache只计算一次，多个请求复用，显著减少重复计算
5. **CUDA Graph兼容**：与CUDA Graph集成，进一步减少内核启动开销，适合高吞吐批量推理

## 实验结论
- 在NVIDIA A100上，decode阶段吞吐量比标准FlashAttention高1.5x-2x
- Cascade推理对共享长前缀场景（如相同system prompt）吞吐量提升3x+
- JIT编译首次开销约10-50ms，后续缓存命中几乎无开销

## 工程落地要点
- SGLang、MLC-LLM等框架已集成FlashInfer
- Cascade推理特别适合企业AI助手（统一system prompt+多用户）场景
- 首次请求的JIT编译延迟对P99延迟有影响，建议预热
- 支持FP16/BF16/FP8量化，可配合AWQ/GPTQ量化模型使用

## 面试考点
**Q: FlashAttention的核心创新是什么？**
A: 通过IO感知的分块计算（Tiling），将注意力的中间结果保留在SRAM中（而非写回HBM），避免HBM读写瓶颈。内存占用从O(n²)降为O(n)，且无近似误差。

**Q: Paged Attention（vLLM）解决了什么问题？**
A: 传统KV Cache连续内存分配导致大量内存碎片（平均浪费60%-80%）。Paged Attention将KV Cache分块（Page）管理，按需分配，内存利用率>95%，显著提升批量推理吞吐量。

**Q: LLM推理的Prefill和Decode阶段区别？**
A: Prefill：并行处理输入prompt，计算所有token的KV Cache，计算密集型（Compute-Bound）；Decode：自回归逐token生成，每步只计算一个token，内存密集型（Memory-Bound）。两阶段特性不同，需要不同优化策略。
