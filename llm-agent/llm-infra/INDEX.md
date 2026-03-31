# LLM Infra 知识库导航

> 搜广推算法工程师的 LLM 基础设施学习路线

## 如何使用这个知识库

**新手路线（推荐顺序）：**
1. 先读 synthesis/ 中的综述文件（理解全貌）
2. 再根据感兴趣的方向看对应 papers/
3. 面试前重点看有 ⭐ 标记的文件

---

## Synthesis（提炼总结，从这里开始）

### LLM 推理优化

| 文件 | 内容 | 重要度 |
|------|------|--------|
| [KVCache与LLM推理优化全景](synthesis/KVCache与LLM推理优化全景.md) | KV Cache 原理、PagedAttention、压缩方案 | ⭐⭐⭐ |
| [FlashAttention3与LLM推理基础设施](synthesis/FlashAttention3与LLM推理基础设施.md) | FlashAttention 机制、IO 优化 | ⭐⭐⭐ |
| [LLMServing系统实践](synthesis/LLMServing系统实践.md) | vLLM/SGLang 对比、连续批处理 | ⭐⭐⭐ |
| [LLM推理效率三角](synthesis/LLM推理效率三角.md) | 延迟/吞吐/成本三角权衡 | ⭐⭐ |
| [LLM推理优化完整版](synthesis/LLM推理优化完整版.md) | KV Cache → Speculative Decoding → 分布式推理 | ⭐⭐ |
| [MoE推理解耦架构](synthesis/MoE推理解耦架构.md) | MoE 专家并行、解耦推理 | ⭐⭐ |
| [MoE架构设计](synthesis/MoE架构设计.md) | MoE 设计原则（与上文互补） | ⭐⭐ |

### 训练与微调

| 文件 | 内容 | 重要度 |
|------|------|--------|
| [LLM微调技术](synthesis/LLM微调技术.md) | LoRA/QLoRA/P-tuning 系统对比 | ⭐⭐⭐ |
| [GRPO大模型推理RL算法](synthesis/GRPO大模型推理RL算法.md) | GRPO 原理、vs PPO 对比 | ⭐⭐⭐ |
| [RLVR_vs_RLHF后训练路线](synthesis/RLVR_vs_RLHF后训练路线.md) | 两条后训练路线对比 | ⭐⭐ |
| [LLM对齐方法演进](synthesis/LLM对齐方法演进.md) | RLHF/DPO/PPO 演进 | ⭐⭐ |
| [LLM预训练技术演进](synthesis/LLM预训练技术演进.md) | 数据、架构、训练策略 | ⭐ |

### RAG 与 Agent

| 文件 | 内容 | 重要度 |
|------|------|--------|
| [RAG系统全景](synthesis/RAG系统全景.md) | RAG 完整流程、工业实践 | ⭐⭐⭐ |
| [RAG与Agent推理能力前沿综述](synthesis/RAG与Agent推理能力前沿综述.md) | Agent RAG 最新进展 | ⭐⭐ |
| [LLM推理优化与RAG_Agent前沿综述](synthesis/LLM推理优化与RAG_Agent前沿综述.md) | 推理优化 + RAG/Agent 综合 | ⭐⭐ |

### 综合/前沿

| 文件 | 内容 | 重要度 |
|------|------|--------|
| [LLM基础设施工程优化要点_2026](synthesis/LLM基础设施工程优化要点_2026.md) | 2026 最新工程实践 | ⭐⭐ |
| [LLM推理与RAG技术进展_20260326](synthesis/LLM推理与RAG技术进展_20260326.md) | 最新进展速览 | ⭐ |

---

## Papers（深入阅读）

### inference/（推理优化）

- [speculative_decoding_for_10x_faster_llm_inference_in_production](papers/inference/speculative_decoding_for_10x_faster_llm_inference_in_production.md) — 投机解码工业落地
- [double_speculative_parallelism](papers/inference/double_speculative_parallelism.md) — 双投机并行
- [lcd_low_bit_clustering_llm_quantization](papers/inference/lcd_low_bit_clustering_llm_quantization.md) — 低比特量化

### training/（训练/微调）

- [lora_based_fine_tuning_for_domain_specific_llm_recommendation_systems](papers/training/lora_based_fine_tuning_for_domain_specific_llm_recommendation_systems.md) — LoRA 领域微调推荐系统
- [limo_less_is_more_for_reasoning](papers/training/limo_less_is_more_for_reasoning.md) — 少量数据推理能力
- [qwen3_technical_report](papers/training/qwen3_technical_report.md) — Qwen3 技术报告

### architecture/（模型架构）

- [deepseek_r1_incentivizing_reasoning_capability_in_llms_via_rl](papers/architecture/deepseek_r1_incentivizing_reasoning_capability_in_llms_via_rl.md) — DeepSeek R1 推理能力强化学习

### rag-agent/（RAG 与 Agent）

- [rag_with_adaptive_retrieval_and_multi_hop_reasoning_for_complex_qa](papers/rag-agent/rag_with_adaptive_retrieval_and_multi_hop_reasoning_for_complex_qa.md) — 自适应检索 + 多跳推理
- [agentic_retrieval_augmented_generation_a_survey_on_agentic_rag](papers/rag-agent/agentic_retrieval_augmented_generation_a_survey_on_agentic_rag.md) — Agentic RAG 综述
- [collab_rag_white_box_and_black_box_llm_collaboration_for_rag](papers/rag-agent/collab_rag_white_box_and_black_box_llm_collaboration_for_rag.md) — 白盒/黑盒 LLM 协作 RAG
- [agent_framework_tool_use_reasoning_recommendation](papers/rag-agent/agent_framework_tool_use_reasoning_recommendation.md) — Agent 框架工具调用推荐

### daily/（归档：每日摘要 + 早期笔记，选读）

84 个文件，含 20260313–20260323 每日论文速读及早期课程笔记。

---

## 推荐阅读顺序

**面试备战（2 小时版）：**
KVCache全景 → FlashAttention3 → LLMServing系统实践 → GRPO算法 → LLM微调技术 → RAG系统全景

**深度学习（1 周版）：**
上述全部 synthesis/ ⭐⭐⭐ + ⭐⭐ → papers/inference/ → papers/rag-agent/

## 20260330 新增 Papers

| 文件 | 标题 | 关键结果 |
|------|------|---------|
| EAGLE3_scaling_inference_acceleration_training_time_test.md | EAGLE-3: 推测解码新 SOTA | 接受率 0.82，加速比 3.8x |
| LIMO_less_is_more_for_reasoning.md | LIMO: 少样本激活推理 | 817条 > 100K，AIME 57.1% |
| KV_Cache_optimization_strategies_scalable_efficient_LLM_inference.md | KV Cache 优化全景综述 | 量化/稀疏/卸载/复用四大策略 |
| MiniKV_2bit_KV_cache_compression_system_codesign.md | MiniKV: 2-bit KV 压缩 | 8x 压缩，精度损失 <2% |
| FlashAttention3_fast_accurate_attention_H100_GPUs.md | FlashAttention-3: H100 优化 | TMA 异步+WGMMA+FP8，75%+ MFU |

## 20260330 新增 Synthesis

| 文件 | 主题 | 类型 |
|------|------|------|
| synthesis/LLM推理加速与高效训练技术全景.md | 今日5篇llm-infra论文综述（含🎯核心洞察） | 新建+深度整合 |

## 2026-03-31 新增论文

| 文件 | 论文标题 | 关键词 |
|------|---------|--------|
| lcd_extreme_low_bit_clustering_llm.md | LCD: Extreme Low-Bit Clustering | 2-bit量化, 聚类量化, 知识蒸馏 |
| double_retrieval_speculative_parallelism.md | Double: Speculative Parallelism | 双源投机, 检索缓存, 并行验证 |
| fastmtp_multi_token_prediction_acceleration.md | FastMTP: Multi-Token Prediction | 多token并行, MTP头, 推理加速 |
| framework_formalizing_llm_agent_security.md | LLM Agent Security Framework | Agent安全, 形式化验证, 威胁模型 |
| google_agent_development_kit_adk.md | Google Agent Development Kit (ADK) | Agent框架, 声明式, 状态机编排 |

**Synthesis**: [LLM推理加速与Agent工程化](synthesis/20260331_llm_inference_optimization_and_agent_engineering.md)
