# 知识卡片 #006：KV Cache 压缩技术全景

> 📚 参考文献
> - [Kvsharer-Layer-Wise-Kv-Cache-Sharing](../../llm-infra/20260320_KVSharer-Layer-Wise-KV-Cache-Sharing.md) — KVSharer: Efficient Inference via Layer-Wise Dissimilar K...
> - [Vllm-Paged-Attention](../../llm-infra/20260317_vllm-paged-attention.md) — vLLM PagedAttention：LLM 推理内存管理革命
> - [Efficient-Long-Context-Llms-Survey-Benchmark-20...](../../llm-infra/20260321_efficient-long-context-llms-survey-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Zsmerge-Zero-Shot-Kv-Cache-Compression](../../llm-infra/20260320_ZSMerge-Zero-Shot-KV-Cache-Compression.md) — ZSMerge: Zero-Shot KV Cache Compression for Memory-Effici...
> - [Efficient-Long-Context-Llms-Survey-And-Benchmar...](../../llm-infra/20260321_efficient-long-context-llms-survey-and-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Flashinfer-Attention-Engine-Llm](../../llm-infra/20260319_flashinfer-attention-engine-llm.md) — FlashInfer: Efficient and Customizable Attention Engine f...
> - [Flashinfer-Attention-Engine-Llm-Inference](../../llm-infra/20260319_flashinfer-attention-engine-llm-inference.md) — FlashInfer: Efficient and Customizable Attention Engine f...


> 创建：2026-03-20 | 领域：LLM推理·内存优化 | 难度：⭐⭐⭐⭐
> 来源：ZSMerge (2503.10714)、KVSharer (2410.18517)


## 📐 核心公式与原理

### 1. Self-Attention
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Transformer 核心计算

### 2. KV Cache
$$\text{Memory} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len \times dtype\_size$$
- KV Cache 内存占用公式

### 3. LoRA
$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$
- 低秩适配，r << d 大幅减少可训练参数

---

## 🌟 一句话解释

LLM 推理时 KV Cache 可占 GPU 内存 **80%+**，压缩它有两个思路：**层内压缩**（去掉不重要 token）和**层间共享**（不同层复用同一份 KV），前者靠 ZSMerge，后者靠 KVSharer。

---

## 🎭 生活类比

想象你在阅读一本书并做笔记（KV Cache = 你的笔记本）：

- **层内压缩（ZSMerge）**：重要句子完整抄，不重要的只记摘要（残差合并），而不是直接划掉——这样笔记更精简但信息不丢
- **层间共享（KVSharer）**：第 3 章和第 7 章的笔记有些可以复用，不必重写——关键是复用的笔记要**互补**而非重复，才有价值

---

## ⚙️ 技术演进脉络

```
【时代一：朴素 KV Cache】
  全量保存所有 token 的 K/V → 内存线性增长，超长上下文 OOM

【时代二：层内 Token 裁剪（H2O / StreamingLLM）】
  保留重要 token，丢弃不重要的 → 信息永久丢失

【时代三：PagedAttention (vLLM)】
  分页管理 KV Cache 内存 → 解决碎片化问题，但总量未减少

【时代四：ZSMerge / KVSharer（2024-2025）】
  ZSMerge：残差合并（信息保留）+ head 级细粒度 + 零样本
  KVSharer：层间共享（正交方向）+ 反直觉的"不相似"共享
  → 两者可叠加，实现更大压缩率
```

---

## 🔬 核心机制对比

| 维度 | ZSMerge | KVSharer |
|------|---------|----------|
| **压缩方向** | 层内（token 维度） | 层间（层维度） |
| **核心操作** | 不重要 token 残差合并到邻近 token | 不同层 KV 直接共享 |
| **关键洞察** | 残差 > 丢弃，保留上下文 | 不相似共享 > 相似共享，保互补性 |
| **是否需训练** | 无需，零样本 | 无需，即插即用 |
| **压缩比** | 最高 20:1（内存降至 5%） | KV 计算量减少 30% |
| **加速效果** | 长上下文吞吐 +3x | 生成速度 +1.3x |
| **可叠加** | ✅ 可与 KVSharer 叠加 | ✅ 可与 ZSMerge 叠加 |

---

## 🏭 工业常见做法（论文 vs 落地）

| 论文假设 | 工业实际 |
|---------|---------|
| 单一压缩方法 | 多方法组合（ZSMerge + KVSharer + 量化）|
| 离线评估质量 | 在线 A/B 测试 + 质量监控报警 |
| 均匀压缩率 | 动态调整（内存充裕时低压缩，OOM 预警时高压缩）|
| 无需微调 | 极端压缩下需要少量 SFT 校准 |
| 单机测试 | 需要适配分布式推理框架（vLLM / TensorRT-LLM）|

---

## 🆚 和已有知识的对比

**vs PagedAttention (vLLM)**：
- PagedAttention 解决**内存碎片**问题（分配效率），不减少 KV 总量
- ZSMerge/KVSharer 减少**总内存占用**，可叠加在 PagedAttention 上

**vs MQA / GQA（多查询/分组注意力）**：
- MQA/GQA 在**模型设计阶段**减少 KV head 数，需要重新训练
- ZSMerge/KVSharer 在**推理阶段**对已有模型即插即用，无需修改模型

---

## 🎯 面试考点

**Q1：KV Cache 为什么会占 GPU 内存 80%+？**
A：对于 70B 模型，FP16 权重约 140GB，但 KV Cache 大小 = `2 × num_layers × num_heads × head_dim × seq_len × batch_size × sizeof(dtype)`。长上下文（32K+）× 大 batch 时，KV 轻松超过权重大小。

**Q2：为什么"不相似"的 KV Cache 共享效果更好？**
A：相似 KV 共享会加剧信息冗余，相当于多层读同一本书的同一页。不相似 KV 捕获互补信息，共享后每层仍能获取差异化内容，类似多样化集成的效果。

**Q3：ZSMerge 的残差合并如何避免信息丢失？**
A：被压缩的 token 信息通过加权平均合并到保留 token 的 KV 中（残差连接），而不是直接丢弃。保留 token 的 attention 计算时隐式包含了被合并 token 的信息。

**Q4：如何为长文本服务选择 KV 压缩方案？**
A：128K token 以下优先 PagedAttention；超过 128K 考虑叠加 ZSMerge（20:1 压缩）；同时用 KVSharer 减少 30% 计算；对质量要求极高时只用轻度压缩 + 质量监控回退。

---

### Q1: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q2: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q3: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q4: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。

### Q5: MoE 的优势和挑战？
**30秒答案**：优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。

### Q6: PagedAttention（vLLM）的核心思想？
**30秒答案**：借鉴操作系统虚拟内存分页，将 KV Cache 切分为固定大小的「页」，按需分配。解决传统方式预分配最大序列长度导致的内存浪费（平均浪费 60-80%）。

### Q7: Continuous Batching 是什么？
**30秒答案**：传统 Static Batching 等最长序列完成才处理下一批。Continuous Batching 每个 token step 都可以加入新请求，序列完成立即释放。将 GPU 利用率从 ~30% 提升到 ~80%。

### Q8: GRPO 和 PPO 的核心区别？
**30秒答案**：PPO 需要 value network 估计 advantage；GRPO 用 group 内的相对奖励替代 value network：采样 G 个输出，用组内排名作为 baseline。更简单、更稳定、不需要额外模型。

### Q9: RAG vs Fine-tuning 怎么选？
**30秒答案**：RAG：知识频繁更新、需要引用来源、不想改模型。Fine-tuning：任务固定、需要特定风格/格式、追求最低延迟。两者可结合：fine-tune 后的模型 + RAG 检索。

### Q10: LLM 推理的三大瓶颈？
**30秒答案**：①Prefill 阶段：计算密集（大量矩阵乘）；②Decode 阶段：内存密集（KV Cache 读写）；③通信：多卡推理时的 AllReduce。优化方向：FlashAttention（①）、PagedAttention（②）、TP/PP 并行（③）。
