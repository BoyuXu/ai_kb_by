# 从前缀缓存到 Fusion RAG 缓存：加速 RAG 中的 LLM 推理

> 来源：From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented Generation | 日期：20260318 | 领域：llm-infra

## 问题定义

RAG 系统每次请求流程：检索文档 → 拼接 prompt（system + retrieved docs + query） → LLM 生成。其中**检索文档占 prompt 的大部分 token**（通常 60-80%），而每次请求检索的文档集合不同，导致传统 KV Cache 命中率极低。标准 Prefix Cache 要求 prefix 完全一致才能命中，但 RAG 的 prompt = system_prefix（固定） + docs（变化） + query（变化），结构不适合 prefix cache。

如何高效复用已计算的文档 KV Cache，避免重复计算？

## 核心方法与创新点

1. **文档级 KV 缓存**：将每个文档的 KV Cache 独立存储（以文档 hash 为 key），与查询无关。当同一文档在不同请求中被检索到，直接复用其 KV Cache，跳过 prefill 计算。

2. **Fusion 机制**：RAG 的注意力计算中，文档之间、文档与查询之间需要交叉注意力。简单拼接 KV Cache 忽略了文档间的依赖关系。Fusion RAG Cache 设计：
   - 文档内 KV Cache 独立计算并缓存（self-attention within doc）。
   - 文档间 cross-attention 在检索时按需计算（开销较小，因文档数量 K 通常 < 10）。
   - Query-over-docs attention 使用 paged attention 高效计算。

3. **缓存驱逐策略**：按文档访问频率 + 新鲜度做 LRU-K 驱逐，常用文档（FAQ 类）常驻缓存。

4. **与 vLLM 集成**：在 vLLM 的 PagedAttention 基础上实现，无需修改模型权重。

## 实验结论

- 在 HotpotQA 和 NaturalQuestions 的 RAG 场景，相比无缓存基线，TTFT（Time To First Token）降低约 45%，吞吐量提升约 2.3x。
- 文档重复使用率在典型 QA 场景约 60-70%（常见文档被多次检索），缓存命中率高。
- 内存开销：缓存全量文档 KV 约需 2-4x 显存增量，需要合理设置缓存容量上限。

## 工程落地要点

- **文档 hash 一致性**：文档内容修改（更新）时需要 cache invalidation，建议用内容 hash 而非文档 ID（内容不变则 hash 不变，换 ID 无需重新计算）。
- **缓存预热**：系统启动时对高频文档（知识库 top-1000）做预计算，避免冷启动期间延迟高。
- **Chunking 策略**：文档分 chunk 越小，命中率越高（粒度细，复用性强），但管理开销越大；建议 chunk size = 512-1024 tokens。
- **与流式生成结合**：TTFT 优化对用户体验影响最直接（首字延迟），需在流式场景优先验证。

## 常见考点

**Q: LLM 推理的 Prefill 和 Decode 阶段的区别？**
- Prefill：将输入 prompt 所有 token 并行计算，生成 KV Cache，计算密集（矩阵乘）；Decode：逐 token 自回归生成，每步用前缀 KV Cache，内存带宽密集。TTFT 受 Prefill 决定，吞吐受 Decode 决定。

**Q: KV Cache 是什么？为什么重要？**
- KV Cache：存储注意力机制中已计算的 Key 和 Value 矩阵，避免重复计算前缀；对于长 prompt（RAG 场景），KV Cache 决定了推理效率，不用 KV Cache 则每个生成 token 都需要重算全部前缀。

**Q: PagedAttention 的原理？**
- 类比操作系统的虚拟内存分页，将 KV Cache 分成固定大小的块（pages），按需分配，避免显存碎片化，支持更大 batch size 和更长序列。
