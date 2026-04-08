# Agentic AI 系统设计综合分析

> 综合日期：2026-04-08 | 涵盖论文/项目：ARTIST, ALITA, LiCoMemory, OpenViking, Open-SWE

## 技术演进路线

Agent 系统正从"预定义流程"走向"自适应进化"，三大支柱：推理、工具使用、记忆。

### 推理与工具使用
- **ARTIST**：RL 训练工具使用策略
  - Outcome-based RL，不监督中间步骤
  - 数学任务 +22%，学会自我纠正
- **ALITA**：极简主义 Agent 设计
  - "简洁即终极精妙"
  - 自主生成 MCP 工具，GAIA 75.15%

### 记忆系统
- **LiCoMemory**：图结构认知记忆
  - CogniGraph 分层图 + 时间/层级感知检索
  - 轻量级，低延迟更新
- **OpenViking**：文件系统范式记忆
  - 虚拟目录 + URI 替代向量存储
  - 确定性检索，可观察轨迹

### 工程实践
- **Open-SWE**：多 Agent 编码系统
  - Planner + Reviewer 架构
  - LangGraph 异步编排

## 设计模式对比

| 维度 | ARTIST | ALITA | LiCoMemory | OpenViking |
|------|--------|-------|------------|------------|
| 核心理念 | RL学习工具策略 | 极简自进化 | 认知图记忆 | 文件系统记忆 |
| 训练方式 | 强化学习 | 自我演进 | 监督+自更新 | 规则驱动 |
| 可扩展性 | 中等 | 高 | 高 | 高 |
| 适用场景 | 工具密集任务 | 通用Agent | 长期对话 | Agent基础设施 |

## 工业实践指南

1. **工具使用**：RL (ARTIST) 比 SFT 更适合训练灵活的工具使用策略
2. **Agent 架构**：ALITA 的极简设计 + MCP 扩展是可行的通用框架
3. **记忆选型**：
   - 需要关联推理 → LiCoMemory（图结构）
   - 需要确定性检索 → OpenViking（文件系统范式）
   - 两者可互补
4. **多 Agent 协作**：Open-SWE 的 Planner-Reviewer 模式适用于复杂任务

## 面试考点

1. **Agent 记忆设计**：图结构 vs 向量存储 vs 文件系统的 trade-off
2. **RL 在 Agent 中的应用**：为什么 outcome-based RL 优于 process-based？
3. **MCP 协议**：如何实现 Agent 能力的动态扩展？
4. **多 Agent 协作**：如何避免 Agent 间的冲突和冗余？
5. **记忆更新策略**：时间衰减、重要性加权、自迭代更新
