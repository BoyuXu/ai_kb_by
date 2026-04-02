# From Glue-Code to Protocols: A2A and MCP Integration for Scalable Agent Systems
> 来源：https://arxiv.org/abs/2505.03864 | 日期：20260319

## 问题定义
当前多Agent系统大量使用"胶水代码"（Glue Code）连接不同LLM、工具和Agent，导致系统脆弱、难以扩展、跨框架互操作性差。Google提出的A2A（Agent-to-Agent）协议和Anthropic提出的MCP（Model Context Protocol）旨在为Agent通信和工具调用建立标准化协议。本文分析两者的集成路径。

## 核心方法与创新点
1. **MCP（Model Context Protocol）**：
   - 标准化LLM与工具/数据源的接口
   - Client-Server架构：LLM是Client，工具是Server
   - 支持Resources（数据读取）、Tools（函数调用）、Prompts（模板）三类原语
2. **A2A（Agent-to-Agent Protocol）**：
   - 标准化Agent间通信：发现（Discovery）、委托（Delegation）、回调（Callback）
   - 基于HTTP/SSE，AgentCard描述Agent能力
   - 支持长时任务和流式输出
3. **集成架构**：MCP负责Agent-工具层，A2A负责Agent-Agent层，两者互补
4. **能力发现**：A2A的AgentCard + MCP的capability listing，实现动态Agent/工具发现
5. **安全机制**：OAuth2.0认证、权限范围（Scope）控制、审计日志

## 实验结论
- 基于A2A+MCP的系统比手写胶水代码减少约60%的集成代码量
- 跨框架互操作（LangChain Agent调用AutoGen Agent）在协议下实现
- 系统扩展性显著提升：新增工具/Agent无需修改现有代码

## 工程落地要点
- MCP已有大量服务端实现（GitHub、Slack、数据库等），优先使用现有Server
- A2A目前生态较新，建议在新项目中采用，存量系统可逐步迁移
- 本地开发用stdio传输，生产环境用HTTP+SSE或WebSocket
- AgentCard应描述Agent的能力边界、输入输出格式、速率限制

## 常见考点
**Q: MCP解决了什么核心问题？**
A: 解决LLM工具调用的碎片化问题。每个AI应用之前需要自己实现工具接口，MCP提供统一标准，工具厂商实现一次MCP Server，所有支持MCP的LLM客户端都能使用。

**Q: A2A与MCP的定位区别？**
A: MCP是LLM与工具/数据的接口（垂直方向）；A2A是Agent与Agent的接口（水平方向）。MCP让LLM用工具，A2A让Agent委托其他Agent。两者合作构建完整的Agent生态系统。

**Q: 多Agent系统中如何处理任务分配与协调？**
A: （1）中心化编排（Orchestrator统一分配）；（2）去中心化协作（Agent自主协商）；（3）市场机制（竞价/能力匹配）。A2A支持所有三种模式，通过AgentCard的能力描述实现自动匹配。
