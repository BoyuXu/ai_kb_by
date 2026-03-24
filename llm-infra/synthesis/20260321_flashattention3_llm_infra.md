# FlashAttention-3 + LLM 推理基础设施：榨干 H100 的每一个 TFLOP

> 📚 参考文献
> - [Efficient Long Context Llm Survey](../../llm-infra/20260322_efficient_long_context_llm_survey.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Moe-Llama-Mixture-Of-Experts-Efficient-Llm-Serving](../../llm-infra/20260321_moe-llama-mixture-of-experts-efficient-llm-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Kvcache Compression For Long-Context Llm Infere...](../../llm-infra/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Efficient-Long-Context-Llms-Survey-Benchmark-20...](../../llm-infra/20260321_efficient-long-context-llms-survey-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Efficient-Long-Context-Llms-Survey-And-Benchmar...](../../llm-infra/20260321_efficient-long-context-llms-survey-and-benchmark-2025-2026.md) — Efficient Long-Context LLMs: Survey and Benchmark 2025-2026
> - [Moe-Llama-Mixture-Of-Experts-For-Efficient-Larg...](../../llm-infra/20260321_moe-llama-mixture-of-experts-for-efficient-large-language-model-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Speculative Decoding Draft Alignment](../../llm-infra/20260322_speculative_decoding_draft_alignment.md) — Efficiently Aligning Draft Models for Speculative Decoding
> - [Recurrent-Drafter-Speculative-Decoding](../../llm-infra/20260319_recurrent-drafter-speculative-decoding.md) — Recurrent Drafter for Fast Speculative Decoding


**一句话**：FlashAttention-3 通过让 GPU 的"搬运工"和"计算员"同时干活（不互相等待），把 H100 的注意力计算效率从 35% 提升到 75%。

**类比**：老工厂（FA2）：搬运工把材料搬过来，计算员才开始干活，干完了搬运工再搬下一批——互相等待浪费时间。新工厂（FA3）：搬运工搬第二批时，计算员同时处理第一批，流水线作业，吞吐翻倍。

**核心机制（3项创新）**：
1. **Warp 专业化**：把 GPU 的并行线程（warp）分工——Producer warp 专门加载数据（TMA 指令），Consumer warp 专门做矩阵乘法（WGMMA 指令），通过共享显存异步协作，真正重叠 compute/memory
2. **Softmax 延迟隐藏**：Softmax 的 max-reduction 操作穿插在 WGMMA 等待期执行，不占用额外时间
3. **FP8 精度支持**：Q/K 用 E4M3（精度高），V/Output 用 E5M2（范围大），Softmax 在 FP32 做，per-tile scale 校正 → 精度损失极小但吞吐接近 FP8 峰值

**性能数字（H100, BF16, seq_len=8K）**：
- FA2 → FA3：**1.5-2x 加速**；峰值利用率 35% → 75%
- FA3 (FP8)：~120 TFLOP/s（H100 峰值 FP8 ~200 TFLOP/s，61% 利用率）
- 端到端 LLaMA-3-8B 训练：step time 降低 **~25%**

**和今日其他 LLM Infra 论文的连接**：
- **FlashAttention-3**：解决 Attention 计算瓶颈（硬件榨汁）
- **MoE-LLaMA**：解决 MoE 模型的 Expert Dispatch 开销（架构优化）
- **Speculative Decoding（草稿模型对齐）**：解决自回归生成延迟（算法优化）
- **Long-Context LLM Survey**：这三种优化在长上下文下效果更显著（长序列 O(n²) 的压力让 FA3 收益更大）

**三种提速方向的互补性**：
```
计算效率（FlashAttention-3）× 架构效率（MoE 解耦）× 解码效率（Speculative Decoding）
                    ↓
工业 LLM serving 的三叉优化
```

**工业常见做法**：
- H100 推理服务：FA3（BF16）是标配；FP8 用于吞吐要求极高、精度要求略低的场景（如广告 CTR 粗排 LLM）
- 集成入 vLLM/TRT-LLM：无需手写 CUDA，直接 pip install flash-attn==3.x
- 显存节省：FA3 的 IO-aware 算法使 Attention 显存 O(n) 而非 O(n²)，8K 序列下节省约 10GB
- 与 GQA/MQA 协同：FA3 对 GQA 原生支持，head_dim 灵活

**面试考点**：
- Q: FlashAttention 为什么能省显存？ → Fused kernel：不把完整注意力矩阵 (n×n) 写入 HBM，分块计算 + online softmax，只存 O(n) 的输出
- Q: H100 Hopper 架构带来了哪些对 LLM 重要的新特性？ → ① TMA（硬件级异步内存传输）；② WGMMA（新矩阵乘指令，吞吐远超 Ampere）；③ FP8 原生支持；④ NVLink 4.0（MoE 通信加速）
- Q: Speculative Decoding 的基本原理？ → 小草稿模型先快速生成多个 token，大模型一次验证（并行前向）；验证通过的 token 直接接受，不通过的回退重生成。等效推理速度提升 2-4x
