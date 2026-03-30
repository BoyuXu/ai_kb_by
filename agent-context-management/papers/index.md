# 值得深入的论文索引

> 从每日扫描中筛选出值得精读的论文，按优先级排序。

---

## ⭐⭐⭐ 高优先级（建议精读）

### MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens
- **提交时间：** 2026-03-05
- **作者：** Yu Chen, Runkai Chen, ..., Lidong Bing, Tianqiao Chen
- **arXiv 搜索：** `Memory Sparse Attention 100M Tokens`
- **核心贡献：**
  - 提出 Memory Sparse Attention（MSA）架构
  - 将有效上下文长度扩展到 1亿 token
  - 端到端 memory 模型，无需外部 DB
- **关键技术：** 稀疏注意力 + memory 压缩 + 分层缓存
- **面试价值：** ★★★★★ — KV cache 优化 / 长上下文扩展面试高频考点
- **待读状态：** 🔲 未读

---

### All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution
- **提交时间：** 2026-03-19
- **作者：** Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou
- **arXiv 搜索：** `All-Mem Agentic Lifelong Memory Dynamic Topology`
- **核心贡献：**
  - 面向月/年级长期运行 agent 的 lifelong memory
  - 动态拓扑演化（节点增删改 + 关系更新）
  - 持续写入长期记忆而不丢失旧知识
- **关键技术：** Graph topology + 增量更新 + 记忆整合
- **面试价值：** ★★★★☆ — Agent 记忆机制设计题
- **待读状态：** 🔲 未读

---

### Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents
- **提交时间：** 2026-03-20
- **作者：** Luiz C. Borro, Luiz A. B. Macarini, Gordon Tindall, Michael Montero, Adam B. Struck
- **arXiv 搜索：** `Memori Persistent Memory Layer LLM Agents`
- **核心贡献：**
  - 提出 Memori — 轻量级持久化 memory 层
  - 高效跨会话记忆管理
  - 上下文感知检索（非向量相似度的精确语义匹配）
- **面试价值：** ★★★★☆ — Agent 工程实现参考
- **待读状态：** 🔲 未读

---

## ⭐⭐ 中优先级（选择性精读）

### MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation
- **提交时间：** 2026-03-24
- **作者：** Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin
- **核心：** 对比轨迹蒸馏实现跨 agent memory 共享
- **面试价值：** ★★★☆☆
- **待读状态：** 🔲 未读

### GSEM: Graph-based Self-Evolving Memory for Experience Augmented Clinical Reasoning
- **提交时间：** 2026-03-23
- **作者：** Xiao Han, Yuzheng Fan, Sendong Zhao et al.
- **核心：** 图结构自演化 memory + 经验复用
- **面试价值：** ★★★☆☆ — 结构化 memory 设计
- **待读状态：** 🔲 未读

### User Preference Modeling for Conversational LLM Agents (RAI)
- **提交时间：** 2026-03-21
- **作者：** Yuren Hao, Shuhaib Mehri, ChengXiang Zhai, Dilek Hakkani-Tür
- **核心：** 长短期用户偏好向量 + RAG + 弱奖励在线更新
- **面试价值：** ★★★★☆ — 对搜广推方向特别相关
- **待读状态：** 🔲 未读

### VehicleMemBench: Multi-User Long-Term Memory Benchmark
- **提交时间：** 2026-03-24
- **作者：** Yuhao Chen et al.
- **核心：** 可执行车载仿真环境的多用户 memory benchmark
- **面试价值：** ★★★☆☆ — Benchmark 设计参考
- **待读状态：** 🔲 未读

---

## 📊 统计

| 状态 | 数量 |
|------|------|
| ⭐⭐⭐ 高优先级 | 3 |
| ⭐⭐ 中优先级 | 4 |
| ✅ 已读 | 0 |
| 🔲 未读 | 7 |

---

*最后更新：2026-03-29 | MelonEggLearn*

---

## 📅 2026-03-30 新增

---

### MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization
- **提交时间：** 2026-03-26
- **作者：** Weizhi Zhang, Xiaokai Wei, Philip S. Yu et al.
- **arXiv ID：** 搜索 `MemoryCD lifelong cross-domain personalization`
- **核心贡献：**
  - 首个专门评估跨领域终身个性化用户记忆的 benchmark
  - 覆盖多领域（购物/娱乐/健康等）长上下文历史
  - 直接对比"百万 token 上下文 vs 外部记忆系统"的有效性
- **关键技术：** Long-context user memory + Cross-domain personalization benchmark
- **面试价值：** ★★★★★ — 推荐系统 + Agent Memory 最强交叉点
- **待读状态：** 🔲 未读

---

### MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution
- **提交时间：** 2026-03-19（HF Daily 2026-03-27）
- **作者：** Minhua Lin, Zhiwei Zhang, Hanqing Lu, Hui Liu, Xianfeng Tang, Qi He, Xiang Zhang, Suhang Wang
- **arXiv ID：** 2603.18718
- **核心贡献：**
  - 提出 Memory Cycle 统一框架（Write→Index→Retrieve→Use→Refine）
  - Multi-Agent 协同推理打通各环节
  - In-Situ Self-Evolution：memory 在使用中自动精炼
- **关键技术：** Memory Cycle + Contrastive Trajectory Distillation + Self-Evolution
- **面试价值：** ★★★★☆ — Agent 记忆系统整体设计面试必读
- **待读状态：** 🔲 未读
