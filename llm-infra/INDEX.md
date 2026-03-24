# LLM 基础设施知识库导航 🧠

## 📊 领域概览

| 分类 | 文档数 | 描述 |
|------|--------|------|
| **Papers** (根级论文笔记) | ~80篇 | KV Cache、推理优化、MoE、RAG、Speculative Decoding |
| **Synthesis** (提炼总结) | 14篇 | LLM 推理、对齐、微调、RAG 标准化文档 |
| **总计** | ~94篇 | - |

---

## 📖 Synthesis (14篇 提炼总结)

### 标准化文档 (std_*)
- [对齐演进](./synthesis/std_llm_alignment_evolution.md) — RLHF → RLVR → DPO 演进路线
- [微调](./synthesis/std_llm_fine_tuning.md) — LoRA/QLoRA/全参微调
- [推理优化](./synthesis/std_llm_inference_optimization.md) — KV Cache/量化/Speculative Decoding
- [MoE 架构](./synthesis/std_llm_moe_architecture.md) — 混合专家模型
- [预训练](./synthesis/std_llm_pretraining.md) — 大模型预训练
- [RAG 系统](./synthesis/std_llm_rag_system.md) — 检索增强生成
- [Serving 系统](./synthesis/std_llm_serving_system.md) — vLLM/SGLang/部署

### 专题综合
- [KV Cache 压缩](./synthesis/20260320_kv_cache_compression.md)
- [MoE 分离式推理](./synthesis/20260320_moe_disaggregated_inference.md)
- [FlashAttention-3](./synthesis/20260321_flashattention3_llm_infra.md)
- [GRPO RL 对齐](./synthesis/20260321_grpo_rl_alignment.md)
- [LLM 效率三要素](./synthesis/20260322_llm_efficiency_trifecta.md)
- [KV Cache 与推理优化](./synthesis/20260323_kvcache_and_llm_inference_optimization.md)
- [RLVR vs RLHF 后训练](./synthesis/20260323_rlvr_vs_rlhf_posttraining.md)

---

## 📝 最后更新
- **最后更新**: 2026-03-24
- **总文档数**: ~94 篇
