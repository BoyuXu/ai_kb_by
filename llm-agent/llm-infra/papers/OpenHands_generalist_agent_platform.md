# OpenHands: An Open Platform for AI Software Developers as Generalist Agents

**ArXiv:** 2407.16741 | **Venue:** ICLR 2025 | **Date:** 2024-07

## 核心贡献
OpenHands（前身 OpenDevin）是一个开放平台，专为构建强大、灵活的 AI 软件开发 Agent 设计。Agent 可通过写代码、使用命令行、浏览网页与世界交互。

## 关键特性
- **Agent Hub**：包含 10+ 种 Agent 实现，核心为基于 CodeAct 架构的通用 Agent
- **评估框架**：支持 15 个基准测试（SWE-BENCH、WEBARENA 等）
- **社区驱动**：MIT 许可证，2,100+ 次 commit，188+ 贡献者
- **代码执行沙箱**：隔离环境中运行代码，安全可靠

## 架构要点
- CodeAct：将 Agent 行为统一为可执行代码
- 支持 Web 浏览与代码编辑专用 Agent
- 可插拔的 LLM 后端

## 工业意义
开源生态中最具影响力的 coding agent 框架，是学界和工业界研究 agent 能力的重要基准平台。

## 面试考点
- Agent 如何通过 CodeAct 统一工具调用？
- 与 SWE-agent、Devin 等闭源方案的差异？

**Tags:** #llm-infra #agent #coding-agent #open-source
