# CacheClip：有效 KV 缓存复用加速 RAG

> 来源：CacheClip: Accelerating RAG with Effective KV Cache Reuse | 日期：20260318 | 领域：llm-infra

## 问题定义

在 RAG 中，每个请求的 prompt = system_prompt + retrieved_documents + user_query。System prompt 通常固定，可以用标准 Prefix Cache 命中。但 retrieved documents 每次不同，即使两次请求检索到了相同的文档集合，文档的**排列顺序**不同（由重排序分数决定），导致拼接后的 prefix 不同，Prefix Cache 无法命中。

**CacheClip** 解决：在文档顺序可变的情况下，如何高效复用文档的 KV Cache？

## 核心方法与创新点

1. **文档顺序不变性分析**：发现 transformer 的自注意力中，文档 i 对文档 j 的注意力（当 j > i 时，文档 j 看不到 i 的位置，因果掩码）具有特殊结构：相对顺序不变时，文档内的 KV Cache 可以复用，但位置编码（RoPE）会随位置改变。

2. **位置无关 KV Cache**：将文档的 KV Cache 计算与绝对位置解耦：
   - 缓存文档的"相对 KV"（不含位置编码偏移）。
   - 检索时根据文档实际插入位置，动态注入 RoPE 位置偏移（只需 O(1) 的向量加法，代价极小）。

3. **Clip 操作**：当两次请求的文档集合部分重叠（例如共享 3/5 的文档），对重叠文档直接 clip（截取）缓存的 KV，对新文档重新计算，整合后形成完整 KV Cache。

4. **缓存组织**：按文档 hash → 位置无关 KV Cache 的映射建立索引，支持快速查找和局部更新。

## 实验结论

- 在 5 文档 RAG 场景，文档复用率约 40-70%（根据查询多样性），TTFT 降低约 35-50%。
- 相比 Fusion RAG Cache，CacheClip 更轻量（无需修改注意力机制），实现复杂度更低。
- 多次请求同一文档集合（不同顺序）时，TTFT 接近全命中 Prefix Cache（提升约 4x）。
- 内存开销：位置无关 KV 体积与标准 KV 相同，无额外开销。

## 工程落地要点

- **RoPE 兼容性**：CacheClip 专为 RoPE 位置编码设计（LLaMA、Mistral、Qwen 等），ALiBi 或绝对位置编码需要修改适配方案。
- **文档 hash 稳定性**：文档预处理（去标点、截断、格式化）需保证相同内容产生相同 hash，避免缓存失效。
- **与 vLLM 集成**：需要在 vLLM 的 attention kernel 层注入 RoPE 位置偏移，需要一定的工程改造。
- **缓存大小限制**：位置无关 KV 的存储量与知识库大小线性相关，需设置文档数上限（如 top-10K 频繁文档）。

## 面试考点

**Q: RoPE（旋转位置编码）的核心思想？**
- 将位置信息编码为对 Q/K 向量的旋转变换：`q_m = q * e^(im*θ)`，位置 m 的 query 和位置 n 的 key 的内积只依赖相对位置 (m-n)，具有外推性（支持超出训练长度的序列）。

**Q: 为什么文档顺序变化会破坏 Prefix Cache？**
- Prefix Cache 要求 token 序列从头完全一致才能命中；文档顺序改变导致拼接 prompt 不同，即使 token 集合相同，缓存也无法复用。

**Q: CacheClip 和 Prompt Cache 有什么联系和区别？**
- 都是避免重复计算；Prompt Cache（vLLM prefix caching）只支持完全相同的前缀；CacheClip 支持局部文档级复用，更灵活但实现复杂度更高。
