# Agent Memory & Context Management — Learning Hub

> 面向搜广推算法工程师的 Agent 内存与上下文管理知识库

---

## 📖 学习目标

1. **理解 LLM Agent 内存的核心问题**
   - Context window 的有限性
   - 长期内存的存储与检索
   - 内存更新与一致性维护

2. **掌握前沿技术方案**
   - 持久化内存架构（Persistent Memory Layer）
   - 动态拓扑进化（Dynamic Topology Evolution）
   - 时间感知检索（Temporal-Aware Retrieval）
   - 双流程内存系统（Dual-Process Memory）

3. **应用到推荐系统**
   - Agent 在 CTR 预估中的内存需求
   - 多 Agent 召回协调
   - 用户交互历史的高效管理

4. **准备算法工程师面试**
   - 大模型 Agent 方向
   - 推荐系统设计
   - 知识图谱与内存表示

---

## 📂 目录结构

```
agent-context-management/
├── README.md                    # 本文件
├── daily/
│   ├── 2026-03-24.md          # 每日学习笔记
│   └── ...
├── papers/
│   ├── index.md               # 论文索引与阅读进度
│   └── summaries/             # （待建）论文总结
└── implementations/           # （待建）代码实现与案例
```

---

## 📊 进度表

### 学习日志

| 日期 | 论文数 | 关键发现 | 输出 | Status |
|------|--------|---------|------|--------|
| 2026-03-24 | 40+ | 15 核心论文，3 大技术方向 | daily/2026-03-24.md | ✅ |
| 2026-03-25 | - | - | - | 📅 |
| 2026-03-26 | - | - | - | 📅 |

### 论文阅读进度

**Tier-1（基础）：** 0/4
- [ ] Memori (2026-03-20)
- [ ] All-Mem (2026-03-19)
- [ ] D-Mem (2026-03-19)
- [ ] Graph-Native Cognitive Memory (2026-03-17)

**Tier-2（应用）：** 0/4
- [ ] Chronos (2026-03-17)
- [ ] AdaMem (2026-03-17)
- [ ] CraniMem (2026-03-03)
- [ ] NextMem (2026-02-26)

**Tier-3（深化）：** 0/4+
- [ ] MemMA (2026-03-19)
- [ ] D-MEM (2026-03-15)
- [ ] Structured Distillation (2026-03-13)
- [ ] MemArchitect (2026-03-18)

---

## 🎯 关键概念

### 内存类型
- **短期记忆（Short-term）：** Context window 内的对话历史
- **长期记忆（Long-term）：** 持久化存储，支持月度/年度尺度
- **工作记忆（Working Memory）：** 当前任务的活跃信息

### 检索方式
- **密集检索（Dense Retrieval）：** 语义相似度
- **稀疏检索（Sparse Retrieval）：** 关键词匹配
- **时间-事件索引（Temporal Event Indexing）：** Chronos 方式
- **图遍历（Graph Traversal）：** 知识图谱导航

### 内存效率
- **Token 开销：** 平均每次交互的 token 消耗
- **检索延迟：** ms 级检索时间
- **存储容量：** 支持的最大交互轮数/事件数

---

## 🔬 技术对标

### vs. 推荐系统
| 维度 | Agent 内存 | 推荐系统 |
|------|----------|--------|
| 数据 | 交互序列 + 对话 | 用户行为序列 |
| 时间性 | 实时更新 | 离线/准实时 |
| 复杂度 | 高（推理） | 中（排序） |
| 存储 | KB-MB/user | bytes-KB/user |

### vs. 知识图谱
| 维度 | Agent 内存 | KG |
|------|----------|-----|
| 结构 | 动态拓扑 | 静态图 |
| 更新频率 | 高频 | 低频 |
| 应用 | 推理+对话 | 知识查询 |

---

## 🚀 快速开始

### 本周任务（Week 1）
1. **周一-周二：** 阅读 Memori + D-Mem
2. **周三：** 阅读 All-Mem + CraniMem
3. **周四-周五：** 阅读 Chronos + AdaMem
4. **周末：** 综合笔记 + 知识图谱绘制

### 输出目标
- [ ] 10 篇论文的 1-2 页总结
- [ ] 技术对比表格（内存架构 vs 检索方式 vs 应用场景）
- [ ] 面试问题清单（5-10 个可能的提问）
- [ ] 代码示例（玩具实现，基于论文思想）

---

## 📚 相关资源

### 外部链接
- **ArXiv LLM Agent Memory:** https://arxiv.org/search/?query=LLM+agent+memory
- **GitHub - AlgoNotes:** https://github.com/shenweichen/AlgoNotes (推荐系统对标)
- **GitHub - fun-rec:** https://github.com/datawhalechina/fun-rec (推荐系统教程)

### 本地知识库
- `~/Documents/ai-kb/agent-context-management/daily/` — 每日笔记
- `~/Documents/ai-kb/agent-context-management/papers/` — 论文索引
- `~/Documents/ai-kb/agent-context-management/implementations/` — 代码示例（待建）

---

## ✅ 完成度追踪

**当前阶段：** 📌 初始化 + 论文收集

```
[████░░░░░░░░░░░░░░░░░░░░░░] 15% — Papers Collected & Indexed
[░░░░░░░░░░░░░░░░░░░░░░░░░░] 0% — Papers Read & Summarized
[░░░░░░░░░░░░░░░░░░░░░░░░░░] 0% — Implementation Examples
[░░░░░░░░░░░░░░░░░░░░░░░░░░] 0% — Interview Prep Materials
```

---

## 📞 更新日志

| 时间 | 事件 | 论文数 |
|------|------|--------|
| 2026-03-24 10:05 UTC+8 | 初始化知识库，收集 40+ 论文 | 40+ |
| - | - | - |

---

**Maintained by:** MelonEggLearn  
**Last Updated:** 2026-03-24 10:05  
**Next Update:** 2026-03-25
