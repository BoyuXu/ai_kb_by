# From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in RAG

> 来源：arXiv | 日期：20260317

## 问题定义

RAG 场景中，LLM 需要处理 `[System Prompt] + [Retrieved Documents] + [User Query]` 的长输入。每次新查询时，Retrieve Documents 往往是不同的，但 System Prompt 是固定的。传统 KV Cache（Prefix Cache）只能缓存前缀（System Prompt），对动态变化的 Retrieved Documents 无缓存效果，导致 LLM Prefill 阶段的大量重复计算。

**核心问题**：不同查询检索到的文档有**部分重叠**（热门文档被多个查询共同检索到），如何利用这种重叠来缓存和复用 KV Cache？

## 核心方法与创新点

1. **文档级 KV Cache**
   - 对每个文档独立预计算并缓存其 KV（Key-Value）表示
   - 缓存粒度：document chunk（512 tokens）→ 对应 KV tensor
   - RAG 请求中，已缓存的文档直接复用 KV，只计算未缓存文档的 KV

2. **Fusion Mechanism**
   - 挑战：独立缓存的文档 KV 是在没有其他文档上下文时计算的，拼接后 attention 的上下文不一致
   - 解决：Fusion Module 对缓存的文档 KV 做上下文对齐（轻量 cross-attention 调整）
   - Approximate Fusion：用低秩矩阵近似 KV 调整，保持推理速度

3. **缓存管理策略**
   - LRU（Least Recently Used）+ 文档访问频率结合的缓存淘汰策略
   - 热门文档（高频被检索）常驻缓存，长尾文档按需计算

4. **与 PagedAttention 集成**
   - 文档 KV Cache 存储在 vLLM 的 KV Block 中
   - 跨请求共享物理 KV Block（Copy-on-Write 机制）

## 实验结论

- RAG Prefill 延迟降低约 45%（文档重用率高时可达 60%）
- 热门文档重用率约 35~50%（工业 RAG 场景）
- Fusion 误差（vs 精确 attention）<0.5%，生成质量无显著损失

## 工程落地要点

1. **文档粒度选择**：chunk 太小缓存命中率低，太大内存占用高；512~1024 tokens 是 sweet spot
2. **缓存容量规划**：每个 chunk 的 KV 大小 = 2 × num_layers × num_heads × head_dim × seq_len × sizeof(float16)
3. **缓存预热**：热门文档列表可离线统计，系统启动时预先填充缓存
4. **Fusion 开关**：精度要求高的场景可关闭 approximate fusion，使用精确计算

## 常见考点

- **Q: KV Cache 在 LLM 推理中的作用？**
  A: Transformer 自回归生成时，每个新 token 需要与所有历史 token 计算注意力。KV Cache 存储历史 token 的 Key 和 Value 矩阵，避免重复计算，使生成时间从 $O(n^2)$ 降至 $O(n)$。

- **Q: RAG 场景 Prefix Cache 的局限？**
  A: Prefix Cache 只能缓存序列的前缀（连续前缀）。RAG 中文档在 System Prompt 之后，不同请求的文档集合不同，传统 Prefix Cache 无法利用文档级别的重用。Fusion RAG Cache 突破了"连续前缀"的限制。

- **Q: 独立文档 KV 拼接时的 attention 一致性问题怎么解决？**
  A: 每个文档的 KV 是在"无其他文档"的上下文中计算的，拼接后 self-attention 的上下文不完整（缺少文档间的交叉注意力）。Fusion Module 通过低秩近似补偿这种不一致，而不需要重新计算所有文档的 KV。
