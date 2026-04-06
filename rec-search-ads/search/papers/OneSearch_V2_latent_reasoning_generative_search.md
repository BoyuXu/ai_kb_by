# OneSearch-V2: Latent Reasoning Enhanced Self-distillation Generative Search Framework

**ArXiv:** 2603.24422 | **Date:** 2026-03 | **Org:** Industrial (E-commerce)

## 背景
OneSearch 是已在工业生产部署的生成式搜索框架，V2 在其基础上引入隐式推理和自蒸馏机制。

## 三大核心创新

### 1. Thought-Augmented Query Understanding
用 LLM 为每个 query-user 对生成显式 Chain-of-Thought 推理，构建紧凑的关键词式 CoT 在推理时注入模型输入。

### 2. Reasoning-Internalized Self-Distillation
将推理能力蒸馏内化进模型，推理时无需额外 LLM 调用。

### 3. Behavior Feedback Preference Alignment
行为反馈偏好对齐，提升模型与用户实际行为的一致性。

## 核心优势
- 对长尾查询和歧义查询有显著改善
- 推理时注入 CoT 作为补充信号
- 无需额外推理计算开销（internalized）

## 工业落地
23 位作者，大规模工业部署验证。

## 面试考点
- 生成式搜索 vs 传统搜索的核心差异？
- 为什么长尾查询对 CoT 增强收益更大？
- Self-distillation 如何实现推理内化？

**Tags:** #search #generative-retrieval #cot #query-understanding #e-commerce
