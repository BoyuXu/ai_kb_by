# Agent Context & Memory Management 知识库

> **方向定位：** LLM Agent 的上下文管理与记忆机制，是构建长期自主 Agent 的核心基础设施研究方向。

## 📚 核心子方向

| 子方向 | 关键词 | 代表工作 |
|--------|--------|---------|
| 长上下文扩展 | sparse attention, KV cache, 100M tokens | MSA (2026-03) |
| 持久化记忆层 | persistent memory, cross-session | Memori, MemArchitect (2026-03) |
| 终身记忆 | lifelong memory, dynamic topology | All-Mem (2026-03) |
| 跨 Agent 记忆协作 | memory sharing, trajectory distillation | MemCollab (2026-03) |
| 记忆 Benchmark | evaluation, multi-user, temporal consistency | VehicleMemBench, BeliefShift (2026-03) |
| 结构化记忆 | graph memory, self-evolving | GSEM (2026-03) |
| 上下文架构 | conversation tree, multi-branch | Conversation Tree Arch (2026-03) |
| 个性化记忆 | user preference, RAG + memory | User Pref Modeling (2026-03) |

## 📅 每日学习进度

| 日期 | 论文数 | 亮点 | 笔记链接 |
|------|--------|------|---------|
| 2026-04-03 | 7 精选 | Memory in LLM Era 综述(★★★★★), ContextBudget 预算感知上下文压缩(★★★★★), ByteRover LLM 分层记忆, Oblivion 衰减驱动遗忘 | [daily/2026-04-03.md](daily/2026-04-03.md) |
| 2026-04-01 | 5 精选 | MuSEAgent 原子决策经验库 (★★★★★), AdaptToken 熵驱动 token 选择, Marco DeepResearch 验证驱动轨迹 | [daily/2026-04-01.md](daily/2026-04-01.md) |
| 2026-03-31 | 5 精选 | HyDRA Hybrid Memory (HF🔥133), Trace2Skill 轨迹→技能蒸馏, Dual-Memory 架构, MCP代码记忆 | [daily/2026-03-31.md](daily/2026-03-31.md) |
| 2026-03-30 | 5 精选 | MemoryCD 跨域终身记忆benchmark, MemMA Memory Cycle多Agent, Dual-Memory 架构模式 | [daily/2026-03-30.md](daily/2026-03-30.md) |
| 2026-03-29 | 10 | MSA 100M-token memory, All-Mem 终身记忆, MemCollab 跨 Agent 协作 | [daily/2026-03-29.md](daily/2026-03-29.md) |

## 📂 目录结构

```
agent-context-management/
├── README.md          ← 本文件（进度总览）
├── daily/             ← 每日学习笔记
│   └── YYYY-MM-DD.md
└── papers/
    └── index.md       ← 值得深入的论文索引
```

## 🎯 与推荐系统的交叉点

- **User Preference Memory**：长短期偏好向量 → 与推荐系统的用户建模高度相关
- **Personalized Agent**：个性化 memory + RAG → 类似推荐系统的 user tower
- **Multi-User Memory**：VehicleMemBench 多用户场景 → 群组推荐的类比
- **Memory 检索**：外部向量 DB 检索 memory → 与召回系统机制相通

---
*最后更新：2026-04-03 | MelonEggLearn*


---
## Context 管理实战技巧

### 为什么 Context 管理比大多数人想的更重要？

128K token 的上下文不等于"能处理所有信息"。实验证明：
- 关键信息放在 context **最前面或最后面**，LLM 能很好利用
- 关键信息放在 **中间**（"Lost in the Middle"），利用率降至 50-60%

这意味着：塞满 context 反而比精心选择内容效果更差。

### Chunk 策略的实战选择

| 场景 | 推荐 Chunk 策略 | 原因 |
|------|--------------|------|
| 法律/合同文档 | 按段落/条款切分 | 条款是语义完整单元 |
| 代码库 | 按函数/类切分 | 函数是逻辑完整单元 |
| 新闻文章 | 滑动窗口（overlap 20%）| 段落间有语义连接 |
| 对话历史 | 按轮次切分 | 一轮对话是完整语境 |
| 表格数据 | 按行切分（带列头）| 每行需要完整的列头信息 |

### 最常见的 Context 管理错误

**错误1：把所有内容塞进一个 prompt**
```
你是一个助手，以下是关于用户的全部信息：[10000字用户历史]
请根据以上信息回答用户问题：[question]
```
问题：LLM 会"看不见"中间的信息。

**改进**：用 RAG，检索最相关的 3-5 段信息而不是全量。

**错误2：不给 context 设优先级**
```
系统提示：[500字]
检索结果：[5000字]
历史对话：[3000字]
用户问题：[20字]
```
问题：用户问题被淹没，LLM 权重不平衡。

**改进**：用户问题放在最后（紧接着 generation），系统提示保持精简（< 200字），历史对话只保留最近 3-5 轮。

**错误3：长任务不做 Checkpoint**
Agent 执行10步任务，第8步 context 满了 → 整个任务失败。

**改进**：每 3-5 步做一次"总结压缩"：
```
你已完成步骤1-5。请用200字总结已做的工作和当前状态，作为后续步骤的起点。
```
压缩后 context 释放空间，继续执行。
