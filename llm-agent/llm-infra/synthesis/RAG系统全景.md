# RAG 系统全景：检索增强生成的工程实践

> 📚 参考文献
> - [Multi-Agent Llm Systems Coordination Protocols ...](../../llm-infra/20260323_multi-agent_llm_systems_coordination_protocols_and_.md) — Multi-Agent LLM Systems: Coordination Protocols and Emerg...
> - [Creativity-Llm-Multi-Agent-Survey](../../llm-infra/20260319_creativity-llm-multi-agent-survey.md) — Creativity in LLM-based Multi-Agent Systems: A Survey
> - [Efficient-Long-Context-Llms-Survey-And-Benchmar...](../../llm-infra/20260321_efficient-long-context-llms-survey-and-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Creativity-Llm-Multiagent-Survey](../../llm-infra/20260319_creativity-llm-multiagent-survey.md) — Creativity in LLM-based Multi-Agent Systems: A Survey
> - [Efficient-Long-Context-Llms-Survey-Benchmark-20...](../../llm-infra/20260321_efficient-long-context-llms-survey-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Beyond-Rag-Agent-Memory](../../llm-infra/20260316_beyond-rag-agent-memory.md) — Beyond RAG for Agent Memory: Retrieval by Decoupling and ...
> - [Grpo-Group-Relative-Policy-Optimization-Llm-Rea...](../../llm-infra/20260321_grpo-group-relative-policy-optimization-llm-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...

> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：RAG Survey, HyDE, Self-RAG, CRAG, Chunk 策略系列

## 📐 核心公式与原理

### 1. Self-Attention

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- Transformer 核心计算

### 2. KV Cache

$$
\text{Memory} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times dtype\_size
$$

- KV Cache 内存占用公式

### 3. LoRA

$$
W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

- 低秩适配，r << d 大幅减少可训练参数

---

## 🎯 核心洞察（5条）

1. **RAG 是 LLM 落地的关键技术**：解决 LLM 的知识时效性、幻觉和领域专业性问题，让 LLM 基于检索到的事实回答
2. **Chunk 策略决定 RAG 质量的下限**：文档切分太大失去精度，太小丢失上下文。最佳实践：512-1024 tokens + 20% overlap + 按语义段落切分
3. **检索 → 重排 → 生成 的三阶段架构**：向量检索（粗召回 top-50）→ Cross-Encoder 重排（精选 top-5）→ LLM 基于 top-5 生成回答
4. **Query 改写是提升 RAG 效果的最高杠杆**：用户原始 query 直接检索效果差，HyDE（先生成假设答案再检索）、Multi-Query（多角度改写）可提升 Recall 10-20%
5. **Self-RAG 和 CRAG 代表"自适应 RAG"方向**：模型自己判断"需不需要检索"、"检索结果是否有用"，避免所有问题都检索的浪费

---

## 🎓 面试考点（6条）

### Q1: RAG 的基本架构？
**30秒答案**：①离线：文档切分（Chunking）→ 编码为向量 → 存入向量数据库；②在线：用户 Query → （可选）Query 改写 → 向量检索 top-K → （可选）Reranker 精选 → 将检索结果 + Query 拼接为 Prompt → LLM 生成回答。

### Q2: Chunk 策略有哪些？
**30秒答案**：①固定长度切分（512 tokens + 128 overlap）——最简单；②语义段落切分（按标题/段落边界）——保持语义完整性；③递归切分（先按段落、段落太长再按句子）——兼顾两者；④Agentic Chunking（让 LLM 判断切分边界）——最灵活但最慢。

### Q3: RAG 中检索效果差怎么优化？
**30秒答案**：①改善 Chunk：调整大小、增加 overlap、使用语义切分；②改善 Query：HyDE/Multi-Query 改写；③改善 Embedding：用 domain-specific 模型（如 BGE-M3）；④加 Reranker：Cross-Encoder 精排 top-K；⑤加 metadata filter：按时间/类别预过滤。

### Q4: 如何减少 RAG 中的幻觉？
**30秒答案**：①引用标注（让 LLM 明确标注哪些信息来自哪个 chunk）；②Self-consistency（多次生成取一致的回答）；③Faithfulness 检查（用另一个 LLM 验证回答是否忠于检索内容）；④Grounding Score（计算回答与检索内容的语义重叠度）。

### Q5: Self-RAG 的工作原理？
**30秒答案**：训练 LLM 输出特殊 token 来决定：①是否需要检索 [Retrieve]；②检索结果是否相关 [IsRel]；③生成的回答是否被检索内容支持 [IsSup]。这样模型自主决定何时检索、如何使用检索结果。

### Q6: RAG vs Fine-tuning vs 长上下文，什么时候用哪个？
**30秒答案**：RAG——知识频繁更新、需要引用来源、知识量大（>100K tokens）；Fine-tuning——任务特定（特定格式/风格）、知识相对固定；长上下文——知识量适中（<128K）、需要全局理解。三者可组合使用。

---

### Q7: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q8: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q9: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q10: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。
## 🌐 知识体系连接

- **上游依赖**：向量检索（Dense Retrieval）、LLM、Cross-Encoder
- **下游应用**：企业知识库问答、客服系统、搜索增强
- **相关 synthesis**：混合检索融合_多路召回实践.md, LLM推理优化完整版.md, 搜索Query理解.md
