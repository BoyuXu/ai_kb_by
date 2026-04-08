# ALITA: Generalist Agent Enabling Scalable Agentic Reasoning

> 来源：arXiv 2025 | 领域：llm-infra/agent | 学习日期：20260408

## 问题定义

当前 AI Agent 过度依赖预定义工具和流程。

**核心问题**：如何构建最小预定义、最大自我演进的通用 Agent？

## 核心方法与创新点

1. **Minimal Predefinition**：
   - "简洁即终极精妙"原则
   - 仅一个核心组件：直接问题解决器

2. **Autonomous MCP Generation**：
   - 自主构建、优化和复用外部能力
   - 动态生成 MCP (Model Context Protocol) 工具

3. **Self-Evolution**：
   - 从任务经验中学习改进策略
   - 能力随使用增长

## 关键结果

- GAIA 基准：75.15% pass@1
- 无需大量预定义工具

## 工程启示

- Agent 设计的极简主义哲学
- MCP 作为 Agent 能力扩展协议的潜力
