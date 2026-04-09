# Qwen2.5-Max: Alibaba新一代LLM with MoE架构

- **Type**: 闭源模型
- **URL**: https://qwenlm.github.io/blog/qwen2.5-max/

## 核心技术

- MoE架构：多专家子网络(~64个experts)，每token仅激活相关子集
- 预训练20T+ tokens，SFT + RLHF后训练
- 不同expert可专精不同领域（数学推理、代码理解、对话）

## 面试考点

MoE架构设计，expert routing策略，MoE vs Dense模型权衡。
