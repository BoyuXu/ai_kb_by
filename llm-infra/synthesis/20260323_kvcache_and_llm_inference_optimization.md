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
