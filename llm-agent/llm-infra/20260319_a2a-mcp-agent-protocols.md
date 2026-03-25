# From Glue-Code to Protocols: A2A and MCP Integration for Scalable Agent Systems
> 来源：https://arxiv.org/abs/2505.03864 | 日期：20260319

## 问题定义
随着AI Agent的规模扩大，Agent之间的通信（Agent-to-Agent, A2A）和Agent与工具/数据源的连接（Model Context Protocol, MCP）需要标准化协议。现有系统通常用"胶水代码"（glue code）连接不同Agent和工具，导致系统脆弱、难以扩展。本文研究A2A和MCP协议的集成架构。

## 核心方法与创新点
1. **A2A协议（Agent-to-Agent）**：
   - Google提出的Agent间通信标准
   - 定义Agent的能力声明（Agent Card）：描述Agent的功能、输入输出格式
   - 标准化任务委托：一个Agent可以发现并委托任务给另一个Agent
   - 支持同步/异步通信模式

2. **MCP协议（Model Context Protocol）**：
   - Anthropic提出的Agent-工具通信标准
   - 统一接口连接LLM与外部工具（数据库、API、文件系统）
   - Tools、Resources、Prompts三种原语
   - Client-Server架构：LLM是客户端，工具是服务端

3. **A2A + MCP集成架构**：
   - A2A负责Agent间的协作和任务分发
   - MCP负责每个Agent与其工具集的连接
   - 分层架构：顶层Agent通过A2A协调，底层Agent通过MCP调用工具
   - 标准化使Agent可组合、可替换

## 实验结论
- 集成架构相比临时胶水代码，Agent开发时间减少约40%
- 协议标准化使跨组织Agent协作成为可能
- 故障隔离更好：协议层错误不会传播到业务逻辑层
- 支持动态Agent发现：新Agent加入无需修改现有代码

## 工程落地要点
1. **Agent Card设计**：精心设计Agent能力描述，确保可被其他Agent正确理解和调用
2. **版本兼容**：协议版本演进时需要向后兼容，避免级联更新
3. **安全隔离**：Agent间通信需要认证和权限控制，防止恶意Agent
4. **可观测性**：记录所有A2A和MCP调用，便于调试和性能优化

## 面试考点
Q1: MCP（Model Context Protocol）解决了什么问题？
> 每个工具/数据源原来需要为每个AI框架写一次集成代码（N×M问题）。MCP定义标准接口，工具只需实现一次MCP服务端，所有支持MCP的LLM客户端都能使用。类似USB接口标准化硬件连接。

Q2: Agent系统的核心架构模式有哪些？
> (1) ReAct：推理+行动循环（Reason+Act），交替生成思维和工具调用；(2) Plan-and-Execute：先规划所有步骤，再执行；(3) Reflexion：自我反思和迭代改进；(4) Multi-Agent：多个专业Agent协作（如Planner+Executor+Critic）。

Q3: 如何保证Multi-Agent系统的可靠性？
> (1) 幂等操作：相同操作多次执行结果相同，支持安全重试；(2) 超时和重试：设置每个Agent调用的超时，失败时指数退避重试；(3) 人在回路（HITL）：关键决策由人类审批；(4) 沙箱隔离：Agent执行在受限环境，防止意外副作用；(5) 审计日志：完整记录所有Agent行为。
