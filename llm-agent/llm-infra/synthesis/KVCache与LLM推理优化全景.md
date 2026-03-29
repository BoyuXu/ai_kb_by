# KV Cache 压缩与 LLM 推理优化全景
> 📚 参考文献
> - [Flashattention-3-Fast-And-Accurate-Attention-Fo...](../../llm-infra/20260321_flashattention-3-fast-and-accurate-attention-for-llms-on-next-gen-accelerators.md) — FlashAttention-3: Fast and Accurate Attention for LLMs on...
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Efficient-Long-Context-Llms-Survey-Benchmark-20...](../../llm-infra/20260321_efficient-long-context-llms-survey-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Vllm-Paged-Attention](../../llm-infra/20260317_vllm-paged-attention.md) — vLLM PagedAttention：LLM 推理内存管理革命
> - [Efficient-Long-Context-Llms-Survey-And-Benchmar...](../../llm-infra/20260321_efficient-long-context-llms-survey-and-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Flashattention-3-Fast-Accurate-Attention-Next-G...](../../llm-infra/20260321_flashattention-3-fast-accurate-attention-next-gen-accelerators.md) — FlashAttention-3: Fast and Accurate Attention for LLMs on...
> - [Longer-Long-Sequence-Industrial-Rec](../../llm-infra/20260319_longer-long-sequence-industrial-rec.md) — LONGER: Scaling Up Long Sequence Modeling in Industrial R...
> - [Continuous Batching And Dynamic Memory Manageme...](../../llm-infra/20260323_continuous_batching_and_dynamic_memory_management_f.md) — Continuous Batching and Dynamic Memory Management for Hig...

> 知识卡片 | 创建：2026-03-23 | 领域：llm-infra

## 架构总览

```mermaid
graph TB
    subgraph "标准 KV Cache（内存碎片问题）"
        S1[Request 1: 预分配max_len内存]
        S2[Request 2: 预分配max_len内存]
        S3[大量内存浪费]
        S1 --> S3
        S2 --> S3
    end
    subgraph "PagedAttention（vLLM）"
        P1[Block Table<br/>逻辑→物理映射]
        P2[物理Block 1]
        P3[物理Block 2]
        P4[物理Block 3]
        P1 --> P2
        P1 --> P3
        P1 --> P4
        P5[按需分配，无碎片]
        P2 --> P5
    end
```

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

**一句话**：KV Cache 是长上下文推理的「内存黑洞」，今天的内容把三条压缩路线（量化、驱逐、稀疏）和三个系统（vLLM、SGLang、Continuous Batching）串成了一张完整的推理优化地图。

**类比**：KV Cache 像「会议室的黑板」——每个 token 写下自己的「键值笔记」，上下文越长黑板越满。KV 量化是用铅笔代替马克笔（精度低但省空间），KV 驱逐是擦掉不重要的内容（有损），PagedAttention 是把黑板换成活页笔记本（分页管理）。

---

## 核心机制：KV Cache 的三条压缩路线

### 路线 A：量化压缩（最安全）
```
FP16 KV → INT8 KV：显存 -50%，质量几乎无损（PPL 变化 <0.5）
INT8 → INT4：显存 -75%，质量轻微下降（约 1-2%）

⚠️ 挑战：KV 激活值 outlier 多，不能用普通对称量化
✅ 方案：per-token 动态量化 + outlier 分离存储（FP16 + INT8 混合）
```

### 路线 B：Token 驱逐（有损但效果好）
```
H2O（Heavy Hitter Oracle）：
    保留累积注意力分数最高的 20% token
    → 显存 -80%，短任务质量 ~100%，长文档质量 -3%

StreamingLLM：
    必须保留「注意力 sink」（开头 4 个 token） + 最近窗口
    → 实现无限长上下文，代价是远距离信息丢失

⚠️ 红线：Attention Sink token（BOS/system prompt）绝对不能驱逐
    → 实验证明：驱逐 sink 后质量崩溃（PPL 从 3 → 100+）
```

### 路线 C：稀疏注意力（结构性优化）
```
Longformer / BigBird：局部窗口 + 全局 token
FlashAttention：IO-aware 分块，显存 O(n²) → O(n)（计算仍 O(n²)）
MLA（DeepSeek-V3）：低秩 KV 压缩，单头 KV 512维→64维，推理显存 -5x

今日新：组合方案（量化 + 驱逐）可减少显存 80%，质量损失约 5%
```

---

## 系统层：vLLM / SGLang / Continuous Batching

### vLLM（PagedAttention）
```
问题：KV Cache 提前分配 → 碎片 + 无法共享
方案：虚拟连续页 → 实际物理分散（类 OS 虚拟内存）

核心收益：
  ├── 内存利用率从 ~60% → ~95%
  ├── 多 request 共享 Prefix KV（相同 system prompt 只存一份）
  └── 支持 beam search 的 copy-on-write 语义
```

### SGLang（RadixAttention）
```
进化：PagedAttention + KV Cache 基数树索引
方案：前缀树（Radix Tree）管理所有 session 的 KV 共享

收益：
  ├── 多用户共享 system prompt：KV 命中率 ~85%
  ├── 支持 constrained decoding（JSON / 正则），推理加速 6x
  └── 特别适合：ReAct Agent（重复结构 prompt），批量推理
```

### Continuous Batching（连续批处理）
```
旧方式：Static Batching → 等最长请求结束才释放 GPU
新方式：Iteration-level Batching → 每个 decoding step 重新组 batch

收益：GPU 利用率从 ~30% → ~80%，吞吐量提升 2-3x
代价：需要高频调度（每 token 一次），增加调度开销约 5%
```

---

## 工业落地的实际选择

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 通用生产 API | PagedAttention + INT8 KV | 安全、高利用率 |
| 长对话助手（>32K） | H2O 驱逐 + Sliding Window | 显存可控 |
| Agent/ReAct 应用 | SGLang + RadixAttention | 重复 prefix，KV 命中率高 |
| 批量离线推理 | Continuous Batching + FP8 | 最高吞吐量 |
| 实时广告（<10ms） | KV Cache 不压缩 + 小模型 | SLA 严格，不能引入量化波动 |

---

## 技术演进脉络

```
静态 KV Cache (2020-2022) → 固定长度，内存浪费
    ↓
PagedAttention / vLLM (2023) → 解决内存碎片，行业标准
    ↓
H2O / StreamingLLM (2023) → token 驱逐，无限上下文
    ↓
SGLang / RadixAttention (2024) → KV 跨请求共享
    ↓
KV INT4 + 稀疏化组合 (2024-2025) → 压缩 80% 显存
    ↓（预测）
学习型 KV 驱逐策略 → 根据任务类型动态调整保留策略
```

---

## 面试考点

1. **Q: KV Cache 的显存占用公式？**
   A: `2 × num_layers × num_heads × head_dim × seq_len × precision_bytes`
   GPT-3 (96层,96头,128dim,FP16): 128K tokens ≈ 288GB

2. **Q: Attention Sink 是什么？为什么驱逐它会崩溃？**
   A: 开头几个 token（BOS/system prompt）被所有后续 token 高度关注，承担「信息排水渠」角色；驱逐后信息流断裂

3. **Q: PagedAttention 和 OS 虚拟内存的类比？**
   A: KV Block ↔ 内存页，Block Manager ↔ 页表，Copy-on-Write ↔ Beam Search 分叉，Physical Block Pool ↔ 物理内存

4. **Q: SGLang 相比 vLLM 的核心优势？**
   A: Radix Tree 支持跨 session 的精细前缀共享，vLLM 只支持静态 prefix；SGLang 对 ReAct Agent 场景吞吐量 ~3x

5. **Q: KV INT4 量化的特殊挑战是什么？**
   A: KV 激活值分布不规则（大 outlier），需要 per-token 动态缩放；通道间差异大，不能用 per-tensor 量化

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
