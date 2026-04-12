# Agent 系统完整指南：从单 Agent 到多智能体生产架构

> 综合自 5+ 篇 synthesis | 更新：2026-04-13 | 领域：LLM Agent/多智能体系统
> 关联：[[concepts/embedding_everywhere]] | [[03_RAG系统全景与决策框架]]

---

## Agent 技术演进三代

| 维度 | 早期 Agent (2023) | 工程化 Agent (2024) | 生产级 Agent (2025-2026) |
|------|------------------|-------------------|------------------------|
| 架构 | 单层 ReAct loop | LangChain/LlamaIndex 框架 | 声明式 Agent（Google ADK） |
| 安全 | 无安全考虑 | 基础错误处理 | 形式化安全框架（CIA 模型） |
| 工具 | 手动工具注册 | 自动工具发现 | MCP 协议 + 自主生成工具 |
| 规模 | 实验室 demo | 小规模部署 | 多 Agent DAG 编排 + 监控 |
| 记忆 | 无状态单轮 | 基础上下文窗口 | 分层持久化记忆（图+向量+文件） |

---

## 一、核心组件

Agent 系统的五大必备组件：

```
LLM 接口层 → 规划引擎 → 记忆管理 → 工具调用器 → 执行环境
```

### 规划与决策

| 框架 | 规划方式 | 特色 |
|------|---------|------|
| LangChain/LangGraph | 图结构工作流编排 | 条件分支+循环，精细控制 |
| AutoAgent | 基于意图的自动规划 | 零代码配置，门槛最低 |
| Google ADK | 声明式 YAML 配置 | 原生多 Agent 编排+内置安全 |
| MiroFish | 群体智能多智体模拟 | 涌现行为，复杂预测 |

### 记忆系统设计

**三层记忆架构**（工业实践推荐）：

1. **短期工作记忆**：当前上下文，容量有限
2. **中期记忆**：会话级别，支持检索
3. **长期知识库**：跨会话持久化，基于向量+符号索引混合

**关键方法对比**：

| 方法 | 核心机制 | 适用场景 |
|------|---------|---------|
| Cognee | 六阶段记忆管道（摄入→向量化→KG→索引→查询→回源） | 知识密集 |
| LiCoMemory | CogniGraph 分层图 + 时间/层级感知检索 | 长期对话 |
| OpenViking | 虚拟目录 + URI 替代向量存储 | 确定性检索 |
| HippoRAG 2 | PPR 图检索 + 非参数持续学习 | 多跳推理 |
| Hermes Agent | 多级记忆模型 + 自动技能生成 | 自主演化 |

### Library Theorem

为 Agent 记忆优化提供形式化理论基础：采用索引结构（层次化向量索引、B 树）将记忆查询复杂度从 $O(N)$ 降至 $O(\log N)$。

### MCP 协议规范

MCP（Model Context Protocol）要求工具提供者实现：
- Schema 自描述
- 类型校验
- 错误处理
- 幂等性保证

**ALITA** 的极简设计证明：Agent 可自主生成 MCP 工具，GAIA benchmark 达 75.15%。

---

## 二、Agent 失败模式与工程解法

### 失败概率公式

$$
P(\text{成功}) = p^n
$$

其中 $p$ 是单步成功率，$n$ 是步数。

| 单步成功率 | 5步任务 | 10步任务 | 20步任务 |
|-----------|---------|---------|---------|
| 95% | 77% | 60% | 36% |
| 99% | 95% | 90% | 82% |
| 99.9% | 99.5% | 99% | 98% |

**结论**：20 步任务成功率 >95%，单步成功率需 >99.7%。

### 五大失败模式

| 模式 | 现象 | 解法 |
|------|------|------|
| 级联失败 | 工具返回空结果，Agent 脑补继续 | 返回值校验 + Fail-fast + 强制重试 |
| 目标漂移 | 子任务替代原始目标 | 目标锚定 + 约束列表 + 每 5 步 Reflection |
| 幻觉工具调用 | 调用不存在的工具/参数 | JSON Schema 校验 + 结构化 Function Calling |
| Context 溢出 | 长任务中遗忘前期工作 | 阶段性总结 + 外部记忆 + 任务分解 |
| 过度思考 | CoT 模型简单任务花大量 token | 任务分类路由 + max_thinking_tokens 限制 |

---

## 三、多智能体工作流生产架构

### Workflow-as-DAG 抽象

将多 Agent 协作建模为有向无环图，三层架构：

```
Orchestration 层：DAG 编排 + 动态路由
State Management 层：Event Sourcing + 状态持久化
Resilience 层：Saga 补偿事务 + 幂等执行
```

### 调度优化目标

$$
\min_{s \in \mathcal{S}} \sum_{i=1}^{N} \left( \alpha \cdot C_{\text{latency}}(s_i) + \beta \cdot C_{\text{cost}}(s_i) + \gamma \cdot C_{\text{error}}(s_i) \right)
$$

三个维度（延迟、成本、错误率）对应生产系统三大核心诉求。

### 关键设计模式

**Saga 补偿事务**：
- 可补偿步骤（有明确逆操作，如写数据库）：标准 Saga 回滚
- 不可补偿步骤（如发送邮件）：先预览后确认的两阶段提交

**动态路由**：简单任务分配给小模型 → Token 成本降低 41%，成功率从 61% 提升至 95%。

### 单 Agent vs 多 Agent 决策

| 维度 | 评估标准 | 选多 Agent | 选单 Agent |
|------|---------|-----------|-----------|
| 任务分解性 | 子任务是否需不同专业能力 | 能力差异大，context 无法共载 | 同质化工作流 |
| 并行加速 | 有无可并行的独立子任务 | 串行延迟不可接受 | 任务有序依赖 |
| 容错粒度 | 是否需独立重试/补偿 | 单步失败不应影响全局 | 简单线性流程 |

**OneFlow 发现**：同质化工作流（同一基础模型+不同 prompt）用单 Agent 替代可省 15-40% 推理成本。

---

## 四、Agent 安全框架

### CIA 安全三角模型

| 维度 | 含义 | 主要威胁 |
|------|------|---------|
| Confidentiality | Agent 不应泄露私有数据 | Data Exfiltration |
| Integrity | 行为不应被操控 | Prompt Injection |
| Availability | 对抗环境中保持正常 | Tool Misuse |

### 形式化防御

$$
\text{Safe}(a_t | s_t) \Leftrightarrow \forall \text{policy}\ \pi: \pi(a_t | s_t) \in \mathcal{A}_\text{allowed}
$$

允许行为集合 $\mathcal{A}_\text{allowed}$ 由系统设计者静态定义，运行时约束 Agent 策略空间。

---

## 五、RL 驱动的 Agent 推理

### ARTIST：RL 训练工具使用策略

- Outcome-based RL，不监督中间步骤
- 数学任务 +22%，学会自我纠正

### Search-R1：RL 训练搜索推理

基于 GRPO，仅用最终答案正确性作为 Outcome Reward（0/1），训练 LLM 自主决定何时搜索、搜索什么。

$$
\nabla_\theta J(\theta) = \mathbb{E}_{q \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{T_i} \nabla_\theta \log \pi_\theta(a_t^i | s_t^i) \cdot \hat{A}_i \right]
$$

Search-R1-7B 在 HotpotQA 达 42.1% EM，超越单次搜索 RAG 的 31.5%。

### TongsearchQR：RL 训练查询改写

以 NDCG 作为 RL 奖励信号，实现查询改写与检索系统端到端对齐。

$$
R(q') = \text{NDCG}@K\left(\text{Retrieve}(q'), \mathcal{D}^+\right)
$$

MS MARCO 上 BM25 MRR@10 从 18.7 提升至 27.3（+46%）。

**两者互补**：Search-R1 解决"何时搜索"的宏观决策，TongsearchQR 解决"如何改写查询"的微观优化。

---

## 六、Agent 框架对比

| 框架 | 接口风格 | 多 Agent | 安全框架 | 特色 |
|------|---------|---------|---------|------|
| LangChain/LangGraph | 命令式代码 | 中心化 Supervisor | 无内置 | 生态最大 |
| Google ADK | 声明式 YAML | 原生编排 | 内置 CIA | 配置驱动 |
| Dify/Langflow | 可视化拖拽 | 半中心化队列 | 基础 | 低代码 |
| n8n | RPA + AI 混合 | 支持 | 基础 | 自动化编排 |
| AutoAgent | 意图识别 | 自动 | 无 | 零代码 |

### 工具调用 Token 成本

$$
\text{Cost}_{tool} = \text{token}_{call} + \text{token}_{result} + \text{token}_{parse} \approx 3\text{-}5x \text{ single generation}
$$

---

## 七、Agent 推理基础设施

### PLENA：分层 KV Cache

针对 Agentic 长上下文场景的硬件-软件协同设计：

| 层级 | 介质 | 延迟 | 存储内容 |
|------|------|------|---------|
| 热数据 | HBM 80GB | <1us | 当前活跃 Agent KV |
| 温数据 | DRAM 256GB | ~10us | 等待中 Agent KV |
| 冷数据 | NVMe SSD 2TB | ~100us | 历史 KV |

**Attention Score 时间衰减模型**：

$$
\text{Score}(q, k_i) \propto \exp\left(-\lambda \cdot (t - t_i)\right) \cdot \text{softmax}\left(\frac{q \cdot k_i^T}{\sqrt{d_k}}\right)
$$

80%+ 注意力权重集中在系统 prompt 和最近 3-5 次工具输出。128K 上下文下吞吐 3.2x，HBM 占用降低 72%。

---

## 面试高频 Q&A

### Q1: Agent 系统的核心组件？
**30秒**：LLM 接口层、规划引擎、记忆管理、工具调用器、执行环境。记忆和工具是差异化能力——前者决定知识积累速度，后者决定行动空间。

### Q2: Agent 失败概率为什么随步数指数增长？
**30秒**：$P(\text{成功})=p^n$，20 步任务需单步成功率 >99.7% 才能 >95% 整体成功率。解法：Fail-fast + 阶段性检查点 + 任务分解。

### Q3: 多 Agent 什么时候真正有价值？
**30秒**：需要不同专业模型（异质化）、可并行独立子任务、需要独立容错。同质化工作流（同模型不同 prompt）单 Agent 更高效，省 15-40% 成本。

### Q4: MCP 协议的作用？
**30秒**：标准化工具接口协议，要求 schema 自描述、类型校验、错误处理、幂等性。是 Agent 可靠性的基石。ALITA 证明 Agent 可自主生成 MCP 工具。

### Q5: Saga 补偿事务在 Agent 中的难点？
**30秒**：传统 Saga 补偿是确定性的（数据库回滚），但 Agent 步骤可能不可逆（已发邮件）。解法：分可补偿/不可补偿步骤，不可补偿用两阶段提交（先预览后确认）。

### Q6: Search-R1 和 TongsearchQR 的关系？
**30秒**：互补的两级优化——Search-R1 用 GRPO 训练"何时搜索+搜什么"（宏观），TongsearchQR 用 PPO+NDCG 训练"如何改写查询"（微观）。

### Q7: Agent 记忆系统如何选型？
**30秒**：需要关联推理选图结构（LiCoMemory），需要确定性检索选文件系统范式（OpenViking），两者可互补。三层架构：短期上下文+中期会话+长期知识库。

### Q8: PLENA 的分层 KV Cache 与 OS 虚拟内存的区别？
**30秒**：OS 用 LRU/LFU 通用策略，PLENA 利用 Attention Score 语义信息做预测性预取，利用 Agent DAG 拓扑信息预取即将激活的 Agent KV，命中率 83-89%。

---

## 记忆助手

- **Agent = 实习生 + 工具箱**：LLM 是大脑（规划），工具是手脚（执行），记忆是笔记本（经验积累）
- **$P=p^n$ = 多米诺骨牌**：每步 5% 失败率，20 步后成功率只剩 36%
- **Saga = 记账本**：每步操作都记录，失败时按记录逆序补偿
- **MCP = USB 标准**：统一工具接口，即插即用
- **多 Agent 决策口诀**：异质有价值，同质选单体

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[03_RAG系统全景与决策框架|RAG 系统与 Agent 检索]]
