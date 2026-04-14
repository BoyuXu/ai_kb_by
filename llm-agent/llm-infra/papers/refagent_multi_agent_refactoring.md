# RefAgent: Multi-agent LLM-based Framework for Automatic Software Refactoring

- **Type**: Multi-Agent Framework
- **URL**: https://arxiv.org/abs/2511.03153
- **Date**: 2026-04

## 核心贡献

多智能体LLM框架（Planner/Generator/Compiler/Tester），自动完成代码重构并保持功能正确性。

## 关键技术

- **上下文感知规划**: 依赖分析 + 迭代编译反馈循环
- **自反思验证**: 通过测试驱动验证实现自反思
- **多智能体协作**: 专业化Agent分工（规划、生成、编译、测试）

## 核心结果

- 90% median单元测试通过率
- 52.5% 代码异味减少
- 8.6% 可复用性质量提升
- 79.15% F1与开发者重构对齐

## 面试考点

Multi-agent协作架构设计，LLM工具调用与验证循环，代码质量自动化评估。
