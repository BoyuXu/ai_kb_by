# LLM Infrastructure Papers Index

> 最后更新：20260328 | 总计：13 篇

## 推理能力激发

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [deepseek_r1_reasoning_via_rl.md](deepseek_r1_reasoning_via_rl.md) | DeepSeek-R1: Incentivizing Reasoning via RL | GRPO，无SFT激发推理，蒸馏小模型 | 20260328 |
| [limo_less_is_more_reasoning.md](limo_less_is_more_reasoning.md) | LIMO: Less is More for Reasoning | 817条数据超越83K条，LIMO假说 | 20260328 |

## 推理加速

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [double_speculative_parallelism.md](double_speculative_parallelism.md) | Double: Double Retrieval Speculative Parallelism | 无训练推测解码，5.3× 加速，突破理论上限 | 20260328 |
| [speculative_decoding_10x_faster_LLM_inference.md](speculative_decoding_10x_faster_LLM_inference.md) | Speculative Decoding: 10× Faster LLM Inference | 推测解码基础方法 | 早期 |

## 模型压缩与量化

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [lcd_low_bit_clustering_llm_quantization.md](lcd_low_bit_clustering_llm_quantization.md) | LCD: Extreme Low-Bit Clustering via KD | 2-3 bit量化，6.2× 推理加速，KD框架 | 20260328 |

## 模型架构与训练

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [qwen3_technical_report.md](qwen3_technical_report.md) | Qwen3 Technical Report | Thinking/Non-thinking统一，MoE 235B-A22B | 20260328 |
| [LoRA_finetuning_domain_specific_LLM_recommendation.md](LoRA_finetuning_domain_specific_LLM_recommendation.md) | LoRA Finetuning for Domain-Specific LLM | 低秩适配微调，推荐领域应用 | 早期 |

## RAG 系统

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [collab_rag_whitebox_blackbox_llm.md](collab_rag_whitebox_blackbox_llm.md) | Collab-RAG: White-Box & Black-Box Collaboration | 3B SLM + 黑盒LLM协作，Multi-hop QA提升14.2% | 20260328 |
| [rag_adaptive_retrieval_multihop.md](rag_adaptive_retrieval_multihop.md) | RAG with Adaptive Retrieval and Multi-Hop | 自适应检索触发，多跳推理链 | 20260328 |
| [RAG_adaptive_retrieval_multihop_reasoning.md](RAG_adaptive_retrieval_multihop_reasoning.md) | RAG with Adaptive Retrieval (v2) | 自适应RAG扩展版 | 早期 |
| [agentic_RAG_survey.md](agentic_RAG_survey.md) | Agentic RAG Survey | RAG Agent化综述 | 早期 |

## Agent 框架

| 文件 | 论文标题 | 关键贡献 | 日期 |
|------|----------|----------|------|
| [agent_framework_tool_use_recommendation.md](agent_framework_tool_use_recommendation.md) | Agent Framework with Tool-Use for Recommendation | ReAct推荐Agent，工具调用，个性化推理 | 20260328 |
| [agent_framework_tool_use_reasoning_recommendation.md](agent_framework_tool_use_reasoning_recommendation.md) | Agent Framework for Recommendation (v1) | Agent推荐早期版本 | 早期 |

---

## 综合总结

| 文件 | 主题 | 日期 |
|------|------|------|
| [../synthesis/LLM推理优化与RAG_Agent前沿综述.md](../synthesis/LLM推理优化与RAG_Agent前沿综述.md) | 推理优化+RAG/Agent综合综述，含10个Q&A | 20260328 |
