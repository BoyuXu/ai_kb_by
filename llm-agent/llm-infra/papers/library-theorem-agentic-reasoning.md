# The Library Theorem: External Organization Governs Agentic Reasoning Capacity

- **Date**: 2026-03
- **Domain**: LLM-Infra/Agent Theory
- **URL**: https://arxiv.org/abs/2603.21272

## 核心贡献

形式化证明工具增强Agent使用索引外部记忆可实现指数级更低的检索成本：O(log_b N) vs Ω(N)页读取，O(T log_b T) vs Θ(T²)累积成本。

## 关键理论

- **Transformer上下文窗口 = I/O page**: 将上下文窗口形式化为I/O页
- **索引外部记忆**: 通过结构化检索实现高效推理
- **复杂度证明**: 索引检索 O(log_b N) vs 顺序扫描 Ω(N)

## 实验验证

- 控制查找benchmark：随机哈希、有序整数、百科条目
- 存储规模50-5000项
- 跨GPT-4o-mini和GPT-5.4两代模型复现

## 面试考点

- Agent外部记忆的理论分析框架
- 索引结构对推理效率的影响
- CoT与结构化检索的关系
