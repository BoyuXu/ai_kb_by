# AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Microsoft)

> 来源：arXiv 2023 | 领域：llm-infra | 学习日期：20260404

## 问题定义

单 LLM 调用的局限：
1. **上下文长度限制**：复杂任务需要长程规划，超过 Context Window
2. **技能专一化**：单模型无法同时精通代码、推理、检索
3. **验证困难**：模型输出无法自我验证，错误传播

**多智能体协作**：多个 LLM 实例（Agent）通过对话协作完成复杂任务。

## 核心方法与创新点

**AutoGen** 提供多 Agent 对话框架：

1. **可对话 Agent（ConversableAgent）**：
   - 基础 Agent 类：接收消息、执行动作、发送消息
   - 支持 LLM 回复、人工干预、代码执行三种模式
   - 灵活组合：不同 Agent 可选择不同模式

2. **角色特化 Agent**：
   - **AssistantAgent**：LLM 驱动，负责规划和推理
   - **UserProxyAgent**：代码执行 + 人工反馈
   - **GroupChatManager**：协调多 Agent 对话轮次
   
3. **对话驱动的任务分解**：
   - 任务 → GroupChat → 多 Agent 轮流贡献
   - 自然语言协商，不需要固定 DAG 工作流
   
```python
assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={...})
user_proxy.initiate_chat(assistant, message="实现一个排序算法并测试")
```

4. **反思循环（Reflection Loop）**：
   - 一个 Agent 生成 → 另一个 Agent 批评/验证 → 迭代改进
   - 代码生成：生成 → 执行 → 错误反馈 → 修复（自动 Debug）

## 实验结论

- 数学推理（MATH）: 多 Agent 反思 vs 单次生成: **+12%** 准确率
- 代码生成成功率: 自动 Debug 循环提升 **+35%**
- 复杂任务（需多工具协作）: AutoGen 完成率 **73%** vs 单 Agent **41%**

## 工程落地要点

- Agent 数量推荐 3-5（过多对话轮次爆炸）
- 终止条件：设置最大轮次（防止死循环）+ 成功标志检测
- 并发支持：AutoGen v0.4 支持异步 Agent（并行子任务）
- 安全沙箱：UserProxy 代码执行必须在沙箱中（Docker）

## 面试考点

1. **Q**: AutoGen 和 LangChain Agent 的核心区别？  
   **A**: LangChain：单 Agent + Tool 调用链（ReAct）；AutoGen：多 Agent 对话协作，Agent 间可以互相提问/批评/纠正，更适合需要多轮协商的复杂任务。

2. **Q**: 多 Agent 反思循环如何提升质量？  
   **A**: 生成 Agent 和审核 Agent 分工：生成 Agent 专注创造，审核 Agent 专注批判。角色分离比单 Agent 自我审核效果更好（避免确认偏误）。

3. **Q**: 多 Agent 框架的主要成本？  
   **A**: LLM 调用次数 × Token 数显著增加（每轮对话都是 LLM 调用）；对话历史累积导致 Context 变长；需要仔细设计终止条件防止无效循环。
