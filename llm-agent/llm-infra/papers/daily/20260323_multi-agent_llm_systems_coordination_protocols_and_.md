# Multi-Agent LLM Systems: Coordination Protocols and Emergent Capabilities
> 来源：https://arxiv.org/search/?query=multi+agent+LLM+coordination+protocols&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
多个LLM Agent协作解决复杂任务时，需要有效的协调协议。本文研究多Agent系统的通信、协调、角色分工和涌现能力，为构建可靠的Multi-Agent系统提供设计指南。

## 核心方法与创新点
- 协调协议分类：并行（多Agent同步执行）、串行（流水线）、层级（orchestrator-worker）、辩论（多Agent互相评审）
- 角色专业化：不同Agent专注不同能力（代码/搜索/推理/验证）
- 涌现能力：多Agent系统通过协作展现单个Agent无法完成的能力
- 通信协议：结构化JSON消息vs自然语言，工具调用规范

## 实验结论
在复杂任务（SWE-bench、GAIA）上，3-5个专业化Agent协作比单Agent提升约15-25%；辩论协议（多Agent互相质疑）在事实性任务上降低幻觉约30%；但多Agent系统的错误会级联放大。

## 工程落地要点
- Agent间通信需要结构化格式（JSON Schema），避免自然语言导致的解析失败
- 需要全局状态管理（如共享memory），避免不同Agent的信息不一致
- 错误传播是主要风险，每个工具调用需要验证和重试机制

## 面试考点
1. **Q: 多Agent系统的主要协调模式？** A: 中央调度（orchestrator控制）、P2P通信、黑板模式（共享状态）、辩论模式
2. **Q: 为什么单Agent不如多Agent处理复杂任务？** A: 单Agent上下文有限、角色切换不自然、并行度受限
3. **Q: 多Agent系统的主要风险？** A: 错误级联（一个Agent的错误影响所有后续）、无限循环、成本失控
4. **Q: ReAct框架如何工作？** A: 交替Reasoning（思考下一步）和Acting（执行工具调用），形成推理-行动循环
5. **Q: 多Agent协作的"涌现能力"是什么意思？** A: 单独的弱模型组合后，在某些任务上展现出超过强大单模型的能力
