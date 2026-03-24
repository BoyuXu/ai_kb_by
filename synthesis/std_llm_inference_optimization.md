# LLM 推理优化完整版：从 KV Cache 到 Speculative Decoding 到分布式推理

> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：FlashAttention-3, KV Cache 压缩系列, Speculative Decoding, MegaScale-Infer, vLLM

---

## 🎯 核心洞察（5条）

1. **推理成本三大来源**：Memory（KV Cache 随序列长度线性增长）、Compute（Attention O(n²) 复杂度）、IO（GPU 内存带宽是瓶颈而非计算能力），不同优化瞄准不同瓶颈
2. **KV Cache 是内存杀手**：70B 模型生成 4096 token 的 KV Cache 约 40GB，比模型参数本身还大；压缩 KV Cache 是长文本推理的关键
3. **FlashAttention 的核心不是"更快的注意力算法"而是"更好的内存管理"**：通过 tiling（分块计算）避免写出完整的 attention 矩阵到 HBM，减少 IO 时间 2-4x
4. **Speculative Decoding 用"小模型猜 + 大模型验"加速生成**：小模型连续猜 K 个 token，大模型一次并行验证，接受率 70-90% 时速度提升 2-3x
5. **Continuous Batching 是 serving 的核心创新**：vLLM 的 PagedAttention 将 KV Cache 分页管理，不同请求可以动态加入/退出 batch，GPU 利用率从 30% 提升到 80%+

---

## 📈 技术演进脉络

```
朴素 Attention O(n²)（~2020）
  → FlashAttention-1 分块 IO 优化（2022）
    → FlashAttention-2 并行度优化（2023）
      → FlashAttention-3 异步流水线（2024）
→ KV Cache 基础实现（~2020）
  → Grouped Query Attention GQA（2023）
    → KV Cache 量化/稀疏化（2024）
      → 跨层 KV 共享 KVSharer（2025）
→ vLLM PagedAttention（2023）
  → Continuous Batching 普及（2024）
    → 分离式推理 Prefill/Decode 解耦（2025）
→ Speculative Decoding（2023）
  → 自适应投机解码 Nightjar（2024-2025）
```

**关键转折点**：
- **FlashAttention（2022）**：证明 Attention 优化的关键不在算法而在 IO，改变了整个推理优化思路
- **vLLM PagedAttention（2023）**：KV Cache 分页管理使大规模 serving 成为现实
- **GQA（2023）**：Llama-2 采用，KV Cache 减少 4-8x，几乎无精度损失

---

## 🔗 跨文献共性规律

| 规律 | 体现 | 说明 |
|------|------|------|
| IO 才是真正瓶颈 | FlashAttention, PagedAttention | 现代 GPU 的 TFLOPS 远超 HBM 带宽，推理受限于搬数据 |
| 压缩比和精度的 Pareto 前沿 | KV Cache 量化, 模型量化 | INT8 几乎无损，INT4 需要校准，INT2 严重降质 |
| 批处理提升 GPU 利用率 | Continuous Batching | 请求之间共享 GPU 时间片，避免空闲等待 |
| 用空间换时间的反向趋势 | Speculative Decoding | 多跑一个小模型来减少大模型的推理次数 |

---

## 🎓 面试考点（8条）

### Q1: FlashAttention 的核心原理？
**30秒答案**：传统 Attention 需要将 QK^T（n×n 矩阵）写到 GPU HBM 再读回来做 softmax，IO 成本 O(n²)。FlashAttention 将 Q/K/V 分块（tiling），在 SRAM 中完成分块 attention 计算（online softmax），避免写出完整 n×n 矩阵。
**追问方向**：FlashAttention-3 比 2 快在哪？答：异步流水线——数据搬运和 Tensor Core 计算重叠执行，H100 利用率从 35% 到 75%。

### Q2: KV Cache 为什么重要？怎么压缩？
**30秒答案**：自回归生成中，每个新 token 需要和之前所有 token 做 attention，KV Cache 缓存历史 K/V 避免重复计算。压缩方式：①GQA（多 Query 头共享 KV 头）；②量化（FP16→INT8）；③稀疏化（只保留重要 token 的 KV）；④跨层共享（KVSharer）。
**追问方向**：GQA 的 group 数怎么选？答：Llama-2 用 8 组（32 头分 8 组），KV Cache 减 4x。

### Q3: Continuous Batching 怎么工作？
**30秒答案**：传统 static batching 要等一个 batch 全部完成才能处理下一个；Continuous Batching 允许请求随时加入/退出——一个请求完成生成后立即让出位置，新请求立即加入。vLLM 的 PagedAttention 将 KV Cache 分页，动态分配/回收。
**追问方向**：PagedAttention 的 page 大小怎么选？答：通常 16-256 tokens/page，太小碎片多，太大浪费。

### Q4: Speculative Decoding 的原理和适用条件？
**30秒答案**：小模型（draft model）快速生成 K 个候选 token，大模型（target model）并行验证这 K 个 token 的概率分布，数学保证与大模型直接生成的分布一致。适用条件：draft model 和 target model 的分布足够接近（接受率 >70%）。
**追问方向**：draft model 怎么选？答：同架构的小模型（如 7B→70B）、或剪枝/蒸馏的版本。

### Q5: 模型量化的精度-效率权衡？
**30秒答案**：FP32→FP16：几乎无损，2x 加速；FP16→INT8：轻微损失（<0.5% 性能下降），2x 加速+内存减半；INT8→INT4：需要 GPTQ/AWQ 校准，1-2% 损失，适合部署；INT4→INT2：严重降质，仅实验用。
**追问方向**：GPTQ 和 AWQ 的区别？答：GPTQ 逐层量化+重建，AWQ 根据 activation 分布保护重要权重。

### Q6: Prefill 和 Decode 阶段有什么区别？
**30秒答案**：Prefill 处理整个 prompt（compute-bound，大量矩阵乘法），Decode 逐 token 生成（memory-bound，主要是 KV Cache 读取）。解耦推理（Disaggregated Inference）将两阶段分到不同 GPU 集群。
**追问方向**：为什么要解耦？答：Prefill 需要大算力（A100），Decode 需要大内存带宽（H100），混合使用效率低。

### Q7: MoE 推理的特殊挑战？
**30秒答案**：MoE 模型总参数量大（如 Mixtral 47B/141B）但每次只激活一部分专家（如 2/8），挑战是：①专家分布在不同 GPU 上需要 AllToAll 通信；②专家负载不均衡导致 GPU 空闲。
**追问方向**：怎么优化 MoE 推理？答：MegaScale-Infer 将 attention 和 expert 分别部署在不同节点，异步流水线。

### Q8: 长文本推理（100K+ tokens）的瓶颈？
**30秒答案**：KV Cache 内存爆炸（70B 模型 100K tokens 需要 ~1TB KV Cache），O(n²) 注意力计算时间增长。解决：①稀疏注意力（只看最近窗口 + 全局 token）；②KV Cache 分页到 CPU/SSD；③Ring Attention 分布式计算。

---

## 🌐 知识体系连接

- **上游依赖**：Transformer 架构、GPU 硬件原理（HBM/SRAM/Tensor Core）
- **下游应用**：LLM Serving 系统（vLLM/TGI/TensorRT-LLM）、RAG 推理、Agent 推理
- **相关 synthesis**：std_llm_alignment_evolution.md, std_llm_moe_architecture.md
- **相关论文笔记**：synthesis/20260323_kvcache_and_llm_inference_optimization.md, llm-infra/01_llm_fundamentals.md
