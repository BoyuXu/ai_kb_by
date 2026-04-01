# Claude Code 源码学习知识库

这是一份关于 Claude Code（Anthropic 的 AI 编码助手）源码架构和设计模式的完整学习资源。

## 📚 文档结构

### 核心学习资源

1. **architecture.md** (193 行, 6KB)
   - Claude Code 的整体架构概览
   - 7 个核心组件详解
   - 关键技术细节和优化策略
   - 版本演进和已知限制

2. **design-patterns.md** (473 行, 12KB)
   - 7 个可复用的 Agent 设计模式
   - 每个模式包含：模式描述、实现要点、应用场景、Claude Code 实现
   - Session 状态管理、Tool 执行管道、LLM Context Assembly...
   - 快速选择表（根据场景选择模式）

3. **key-modules.md** (539 行, 19KB)
   - 12 个核心模块的详细清单
   - 每个模块的职责、接口、关键特性和依赖
   - 完整的模块架构图和数据流
   - 关键数据流程（启动、消息处理、工具执行）

4. **LEARNING_PLAN.md** (930 行, 27KB)
   - 完整的 5 周学习计划（84-112 小时）
   - 3 个学习阶段：基础理解(1周) → 深度设计(2-3周) → 系统集成(1-2周)
   - 每个阶段包含：学习目标、具体任务、实践实验、反思题、输出物
   - 4 个实践项目选项（可选择 1 个完成）
   - 学习成功标准和 FAQ

## 🎯 快速开始

### 方案 A：快速了解（1 天）
```bash
1. 读 architecture.md 的前两部分（30 分钟）
2. 读 design-patterns.md 的快速选择表（15 分钟）
3. 选 1 个设计模式深入学习（45 分钟）
4. 思考在自己的项目中的应用（30 分钟）
```

### 方案 B：扎实学习（1 周）
遵循 LEARNING_PLAN.md 的阶段 1 学习内容

### 方案 C：深度掌握（5 周）
完整学习整个 LEARNING_PLAN.md（包括 3 个阶段 + 实践项目）

## 📖 适用人群

- ✅ 想理解 Claude Code 架构的开发者
- ✅ 设计 Agent 系统的工程师
- ✅ 优化 LLM 应用的 AI 工程师
- ✅ OpenClaw 的贡献者和维护者
- ✅ 对分布式系统和错误恢复感兴趣的工程师

## 🔑 核心概念速览

| 概念 | 解释 | 关键文件 |
|------|------|--------|
| Session ID | 用于跨终端同步和消息追踪 | architecture.md § 1 |
| Checkpoint | 用于中断恢复的进度存储 | design-patterns.md § 1 |
| Tool Pipeline | 标准化的工具执行流程 | design-patterns.md § 2 |
| Token Budget | LLM 输入的内存预算管理 | design-patterns.md § 3 |
| Permission Rules | 动态的权限和功能过滤 | design-patterns.md § 4 |
| Long-term Memory | 跨会话的向量化记忆系统 | design-patterns.md § 5 |
| Error Classification | 智能的错误分类和重试 | design-patterns.md § 7 |

## 🚀 学习成果

完成全套学习后，你将能够：

- ✅ 理解复杂 Agent 系统的整体架构
- ✅ 识别和应用可复用的设计模式
- ✅ 设计支持中断恢复的会话系统
- ✅ 优化 LLM 应用的性能和成本
- ✅ 实现健壮的分布式错误处理
- ✅ 为团队设计和评审 Agent 系统

## 📋 文件清单

```
~/Documents/ai-kb/claude-code/
├── README.md (本文件)
├── architecture.md (6KB, 193 行)
├── design-patterns.md (12KB, 473 行)
├── key-modules.md (19KB, 539 行)
└── LEARNING_PLAN.md (27KB, 930 行)

总计: 64KB, 2135 行文档
```

## 📚 参考资源

- [Claude Code 官方文档](https://code.claude.com/docs)
- [Claude Code GitHub](https://github.com/anthropics/claude-code)
- [Anthropic API 文档](https://platform.anthropic.com/docs)
- [Claude 开发者 Discord](https://anthropic.com/discord)

## 🤝 贡献和反馈

如果你：
- 发现文档中的错误或遗漏
- 有改进建议
- 想分享学习心得

欢迎提交 issue 或 pull request！

## 📝 更新日志

- **2026-04-01**: 初版发布
  - 完成 architecture.md（7 个核心组件）
  - 完成 design-patterns.md（7 个设计模式）
  - 完成 key-modules.md（12 个核心模块）
  - 完成 LEARNING_PLAN.md（完整 5 周学习计划 + 4 个实践项目）

---

**开始学习 Claude Code 架构吧！** 🎯

建议的学习路径：
1. 先读 architecture.md（了解全貌）
2. 再读 design-patterns.md（学习设计思想）
3. 查 key-modules.md（理解实现细节）
4. 按 LEARNING_PLAN.md 实践（巩固和应用）

预计时间：
- 快速阅读：2-3 小时
- 深入学习：1 周
- 完整掌握：5 周

祝学习愉快！✨
