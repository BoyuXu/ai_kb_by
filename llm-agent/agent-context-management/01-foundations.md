# Agent Context & Memory Management 奠基报告

> 作者: MelonEggLearn | 日期: 2026-03-23
> 来源: Lilian Weng博客、Anthropic Engineering、arXiv 多篇论文

---

## 一、为什么 Agent 需要记忆管理？

LLM 原生的"无状态"特性与 Agent 需要"长期执行复杂任务"之间存在根本矛盾：

- **Context Window 有限**：即使 200K tokens，长任务也会溢出
- **跨会话遗忘**：每次新对话从零开始，没有历史
- **信息噪声**：随着对话增长，早期无关信息稀释注意力
- **知识过时**：参数记忆（模型权重）无法实时更新

---

## 二、记忆的分类体系（类比人类认知）

| 人类记忆 | Agent 对应实现 | 特点 |
|---------|--------------|------|
| 感觉记忆（Sensory） | 原始输入的 Embedding 表示 | 极短暂，只保留特征 |
| 工作记忆（STM） | In-context（当前 context window） | 有限容量，约 7 个"组块" |
| 长期记忆（LTM）- 陈述性 | 外部向量数据库（Vector Store） | 无限容量，需检索 |
| 长期记忆（LTM）- 程序性 | 模型微调（LoRA / Full-FT） | 固化行为，不易更改 |
| 情节记忆（Episodic） | 对话历史文件、任务日志 | 按时间序列存储 |
| 语义记忆（Semantic） | 知识库文件（Markdown/JSON） | 事实型，结构化 |

---

## 三、主流记忆架构方案

### 3.1 In-Context Memory（短期记忆）

**原理**：直接把信息塞进 prompt。

**优化手段**：
- **滑动窗口**：只保留最近 N 轮对话，丢弃早期
- **摘要压缩**：定期将旧对话压缩成摘要再放入 context
- **重要性筛选**：对每条历史打分，只保留高分项
- **结构化注入**：用专门格式（如 XML/JSON）而非自由文本，减少 token 浪费

**适用**：短任务、单会话、实时性要求高的场景

---

### 3.2 External Memory（外部记忆）

**原理**：信息存入外部数据库，按需检索注入。

**核心组件**：
```
Write Path: 信息 → Embedding → Vector DB (Faiss/Weaviate/Pinecone)
Read Path:  Query → ANN Search → Top-K Chunks → Inject to Context
```

**关键技术 - MIPS（最大内积搜索）**：
- LSH（局部敏感哈希）：高维向量快速近似检索
- ANNOY（Spotify）：基于随机树的 ANN
- HNSW（Hierarchical NSW）：当前最主流，O(log N) 检索
- FAISS（Facebook）：GPU 加速，工业级

**优化手段**：
- **混合检索**：稠密检索（Embedding）+ 稀疏检索（BM25）结合
- **分级存储**：热数据内存 → 温数据本地文件 → 冷数据远端存储
- **记忆压缩**：写入前先提取关键信息，不存原文

---

### 3.3 Parametric Memory（参数记忆）

**原理**：通过微调把知识编码进模型权重。

- **LoRA**：低秩适应，轻量级注入领域知识
- **Continual Learning**：持续学习，防止灾难性遗忘
- **特点**：最慢更新，但推理时零检索开销

---

## 四、关键论文与框架

### 4.1 Reflexion（Shinn & Labash, 2023）
**核心思想**：让 Agent 在每次失败后写自我反思，存入 working memory，下次任务时读取。

```
执行 → 失败/低效 → 自我反思（语言形式）→ 存入 memory → 下次任务读取
```

**效果**：在 AlfWorld 环境提升显著，减少幻觉（repetitive actions）。
**局限**：反思最多保留 3 条，防止 context 过长。

---

### 4.2 Self-RAG（Asai et al., 2023）
**核心思想**：自适应检索——不是每次都检索，而是模型自己决定"需不需要检索"。

引入特殊 Reflection Tokens：
- `[Retrieve]`：是否需要检索
- `[IsRel]`：检索结果是否相关
- `[IsSup]`：检索内容是否支持生成
- `[IsUse]`：最终回答是否有用

**效果**：比固定 RAG 更精准，减少无关信息引入。

---

### 4.3 MemMA（2026.03，最新）
**核心思想**：多智能体协调的记忆生命周期管理。
- 记忆的创建、更新、遗忘由专门的 Memory Agent 负责
- In-situ Self-Evolution：边用边更新记忆

---

### 4.4 MemArchitect（2026.03，最新）
**核心思想**：策略驱动的记忆治理层（Policy-Driven Memory Governance）。
- 类似数据库的访问控制，对记忆读写有显式策略约束
- 防止记忆污染和过期信息"幽灵写入"

---

### 4.5 Graph-Native Cognitive Memory（Park, 2026.03）
**核心思想**：用图结构替代平坦向量库存储记忆。
- 正式信念修正语义（Formal Belief Revision Semantics）
- 版本化记忆架构（Versioned Memory）
- 能表达"A 曾经是 B，但现在是 C"的时序关系

---

## 五、工程实践优化方案

### 5.1 Anthropic 实践总结（Building Effective Agents）

> "最成功的实现不是使用复杂框架，而是简单、可组合的模式"

**核心原则**：
1. **能不用 Agent 就不用**：单次 LLM + Retrieval 往往够用
2. **可组合工作流优先**：Prompt Chaining > 复杂框架
3. **直接用 API，理解底层**：框架的额外抽象层是 bug 温床

**推荐工作流模式**：

| 模式 | 场景 |
|------|------|
| Prompt Chaining | 可分解为固定子步骤的任务 |
| Routing | 输入分类 → 路由到专用处理 |
| Parallelization | 子任务独立、需要多角度验证 |
| Orchestrator-Workers | 子任务动态确定、文件级别代码修改 |
| Evaluator-Optimizer | 有明确评估标准、迭代优化 |

---

### 5.2 工程层面的 6 大优化模式

#### ① 状态外化（State Externalization）
**问题**：Agent 依赖"记忆"记住做过的改动，记忆不可靠。
**方案**：把状态写入文件，每次从文件读取真实状态。

```markdown
# CURRENT_STATE.md（示例）
## 已删除的字段
- amount_rank: 2026-03-20 删除
- vol_raw: 2026-03-18 重命名为 vol_clean

## 当前生效规则版本
- 顶部大风车过滤器: v2（2026-03-23）
- vol_clean 验证: 非空强制

## 最近修复
- [2026-03-23] 修复 vol_clean 可能为 null 的边界情况
```

#### ② 变更日志追踪（Append-Only Changelog）
**方案**：每次代码改动强制写一行 CHANGELOG，Agent 可 grep 确认。

```
[2026-03-20] 删除 amount_rank 条件（用户要求）
[2026-03-21] 修复 vol_clean null 值（第1次）
[2026-03-22] vol_clean null 值再修复（第2次，边界遗漏）
[2026-03-23] vol_clean null 彻底修复，增加全字段非空断言
```

#### ③ 增量验证卡点（Incremental Validation Gates）
**方案**：每次改动先跑最小单元（单股票/单天），通过才继续。

```
改代码 → 跑 1 天 1 只股票验证 → 通过 → 跑 1 天全市场 → 通过 → 跑全年
```

**效果**：bug 在小规模时即被发现，不会等全年跑完才知道。

#### ④ Prompt 硬约束（Hard Constraints in Prompt）
**方案**：对反复出错的点写进系统 prompt，软记忆 → 硬规则。

```
❌ 禁止规则（系统级）：
- 不使用 amount_rank 字段（已删除）
- vol_clean 必须非空，遇到 null 直接报错而非静默跳过
- 顶部大风车：20日内最高价当日为阴线且成交量最大/次大 → 直接排除
```

#### ⑤ Memory Consolidation（记忆整合）
**方案**：定期把"工作日志"提炼成"精炼知识"，类似人类从短期到长期记忆的转化。

```
每日原始日志 → 每周整合一次 → 更新 MEMORY.md / knowledge base
```

**工具**: MemoryGPT / Mem0 等框架已实现自动化整合。

#### ⑥ 分层记忆路由（Hierarchical Memory Routing）
**方案**：按信息类型路由到不同存储层。

```
实时状态（当前任务）→ In-context（直接注入 prompt）
最近 N 次操作记录   → 滑动窗口（总结压缩）
领域知识/规则       → 结构化文件（按需读取）
历史案例/经验       → 向量数据库（语义检索）
固化行为模式        → 模型微调（长期）
```

---

## 六、常见失效模式（Anti-Patterns）

| 失效模式 | 表现 | 解决方案 |
|---------|------|---------|
| **记忆幻觉** | Agent 声称记得某操作，实际已过时 | 状态外化到文件，禁止依赖记忆 |
| **重复 Bug** | 同一 bug 修了 N 次还出现 | Append-only changelog + 单元验证 |
| **上下文污染** | 早期错误信息影响后续判断 | 定期摘要压缩，高重要性过滤 |
| **注意力稀释** | 长对话后 Agent 忽略早期关键指令 | 关键规则写入 system prompt |
| **记忆幽灵** | 已删除的字段/条件被重新引用 | 显式禁止列表 + 变更日志 |
| **无验证执行** | 大批量任务跑完才发现 bug | 增量验证门控 |

---

## 七、推荐工具栈（2026）

| 类别 | 工具 | 适用场景 |
|------|------|---------|
| 向量数据库 | Chroma, Qdrant, Weaviate | 语义检索长期记忆 |
| 记忆框架 | Mem0, MemoryGPT | 自动记忆管理 |
| 检索增强 | LlamaIndex, LangChain | RAG pipeline |
| 状态管理 | 文件系统（Markdown/JSON）| 轻量状态外化 |
| 图记忆 | Neo4j + LLM | 复杂关系型记忆 |

---

## 八、与本项目（回跑系统）的对应关系

| 理论概念 | 本项目实践 |
|---------|-----------|
| 状态外化 | CURRENT_STATE.md 记录已删条件 |
| 变更日志 | CHANGELOG.md 追踪每次代码改动 |
| 增量验证 | 先跑 3.20 一天，确认后再全年 |
| Prompt 硬约束 | 系统级禁止列表（不用 amount_rank）|
| 记忆整合 | 定期更新 MEMORY.md |

---

## 参考资料

1. Lilian Weng (2023). "LLM Powered Autonomous Agents". https://lilianweng.github.io/posts/2023-06-23-agent/
2. Zhang et al. (2024). "A Survey on the Memory Mechanism of Large Language Model based Agents". arXiv:2404.13501
3. Shinn & Labash (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning". arXiv:2303.11366
4. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique". arXiv:2310.11511
5. Anthropic (2024). "Building Effective Agents". https://www.anthropic.com/engineering/building-effective-agents
6. Lin et al. (2026). "MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning". arXiv:2026.03
7. Park (2026). "Graph-Native Cognitive Memory for AI Agents". arXiv:2026.03
