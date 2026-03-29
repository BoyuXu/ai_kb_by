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
*最后更新：2026-03-29 | MelonEggLearn*
