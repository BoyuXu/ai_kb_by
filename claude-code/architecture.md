# Claude Code 源码架构总览

## 概述

Claude Code 是一个 **AI 驱动的编码助手**，可运行在：
- **终端 CLI**（全功能）
- **VS Code 扩展**（内联 diff、@提及、计划审查、对话历史）
- **桌面应用**（可视化 diff、多会话并行、定时任务调度）
- **浏览器**（Web 版）
- **GitHub**（通过 @claude 提及）

## 核心架构组件

### 1. **Session 管理**
- **会话隔离**：每个用户会话有独立的 `SESSION_ID`（通过 `X-Claude-Code-Session-Id` 头传递）
- **状态维护**：
  - `--resume` 参数支持从上一个失败点恢复
  - 支持长会话恢复（处理 OOM/Crash）
  - 会话转录缓存（`transcript files`）

### 2. **Tool 执行框架**
```
User Command → Parse Input → Tool Selection → Execution → Result Handling
```

#### 核心工具：
- **Read** — 读文件（支持 `@` 文件提及，JSON 压缩编码减少 token）
- **Write/Edit** — 编辑文件（支持条件技能和权限规则）
- **Run** — 执行命令（支持 bash/shell）
- **Git** — Git 工作流自动化

#### Tool 生命周期：
1. **AskUserQuestion** → 等待用户输入
2. **PreToolUse Hooks** → 权限验证（支持条件 `if` 过滤）
3. **Tool Execution** → 实际执行
4. **Tool Result** — 返回结果（支持 `tool_parameters` 日志）
5. **Post-execution** → 内存管理、结果缓存

### 3. **Agent 生命周期**

#### 初始化阶段：
- 加载项目配置（`.claude/`, `CLAUDE.md`）
- 初始化 MCP 服务器连接（支持 RFC 9728 OAuth）
- 加载插件和技能（支持组织策略限制）
- 构建代码索引和文件缓存

#### 运行阶段（Main Loop）：
1. **Message Input** → 从用户/脚本获取消息
2. **Context Assembly** → 组织 LLM 上下文
   - 最近的对话历史
   - 代码片段和文件内容
   - 内存和技能描述（<=250 字符）
3. **LLM Inference** → 调用 Claude 模型
4. **Tool Call Processing** — 并行处理多个 tool calls
5. **Streaming Response** → 实时流式返回结果
6. **Memory Update** → 更新会话内存

#### 完成阶段：
- `/compact` 支持清理超大会话
- 保存转录和元数据
- 清理临时资源

### 4. **错误处理 & 恢复机制**

#### 内部错误恢复：
- **Tool 失败重试** — 内置 retry 逻辑（如 Git 操作失败）
- **Context 超限处理** — 自动触发 `/compact` 清理
- **MCP 连接异常** — 自动重连或降级
- **OOM 防护** — 内存缓存管理（markdown/highlight 渲染缓存去重）

#### 用户可见的恢复：
- `--resume` 恢复会话（检查 `tool_use ids` 和 `tool_result` 块完整性）
- 支持从 checkpoint 续跑（Cowork Dispatch）

### 5. **LLM 集成**

#### 模型支持：
- **Claude 3.5 Sonnet** — 默认（快速、平衡）
- **Claude 3 Opus 4.5** — 高性能（支持迁移工具）
- **Bedrock / Vertex / Foundry** — 企业方案

#### 优化技术：
- **Prompt Caching** — 使用 `claude-3-5-sonnet-20241022` 缓存提示（提高 hit rate）
- **Token 优化**：
  - 文件内容不再 JSON 转义（原始字符串）
  - 条件技能过滤减少 process 启动开销
  - 状态行缓存（避免重复赋值）
- **Streaming** — 实时流式返回，不等待完整生成

### 6. **插件系统 (Plugin Architecture)**

#### 插件类型：
- **Skill Plugins** — 自定义技能（添加新命令和代理）
- **MCP Plugins** — Model Context Protocol 服务器集成
- **Hook Plugins** — 生命周期钩子（PreToolUse, WorktreeCreate 等）

#### 插件管理：
- 支持官方市场 + 社区插件
- 组织策略可阻止特定插件（`managed-settings.json`）
- 条件激活（`if` 字段支持权限规则语法）

### 7. **VCS 集成**

#### Git 支持：
- 自动文件跟踪和 commit
- Worktree 管理
- 分支导航
- 排除列表（`.jj`, `.sl` 用于 Jujutsu/Sapling）

#### 其他 VCS：
- Jujutsu 支持
- Sapling 支持

## 关键技术细节

### 数据流
```
User Input
  ↓
[Parsing & Intent Detection]
  ↓
[Memory & Context Retrieval]  ← LanceDB Pro / 内存管理
  ↓
[Skill Selection & Filtering]  ← 权限规则、条件 if
  ↓
[LLM Prompt Assembly]  ← Prompt caching, token optimization
  ↓
[Model Inference]  ← Streaming, @-mentions, tool_use
  ↓
[Tool Execution Framework]
  ├─ Permission Check (PreToolUse hooks)
  ├─ Execute (parallel capable)
  ├─ Collect Results (tool_result blocks)
  └─ Post-process (caching, memory update)
  ↓
[Response Formatting & UI]  ← Inline diffs, highlights
  ↓
Output
```

### 性能优化

1. **启动优化**
   - Markdown 渲染缓存去重
   - MCP keychain 缓存扩展（5s → 30s）
   - 条件技能过滤减少进程启动

2. **Token 优化**
   - Prompt caching 提高 hit rate
   - 文件内容去 JSON 转义
   - 技能描述长度限制（250 字符）
   - `@` 提及的文件使用原始字符串

3. **内存优化**
   - 长会话自动 compact
   - 渲染缓存去重
   - 流式处理避免全量加载

## 会话间上下文传递

### 机制
- **Session ID** 通过 HTTP 头 `X-Claude-Code-Session-Id` 传递
- **转录文件** 保存在本地（支持 `--resume` 恢复）
- **内存系统** LanceDB Pro 支持跨会话的长期记忆

### 数据隐私
- 有限的数据保留期限
- 受限的会话数据访问
- 明确禁止用于模型训练的反馈
- OpenTelemetry 日志需要 `OTEL_LOG_TOOL_DETAILS=1` 启用

## 版本演进

### 近期重要变化（v2.1.85+）
- MCP OAuth RFC 9728 支持
- PreToolUse 钩子支持 `updatedInput`（无头集成）
- Cowork Dispatch 消息修复
- Session ID 头添加（代理聚合）
- 条件技能 `if` 字段支持
- `/compact` 上下文清理能力
- Worktree 支持改进

### 已知限制
- `--bare` 模式会丢弃 MCP 工具
- VSCode 扩展 8 小时后 OAuth 刷新问题
- 非 git 项目中 `--worktree` 失败

## 参考资源

- [官方文档](https://code.claude.com/docs)
- [开发者 Discord](https://anthropic.com/discord)
- [数据使用政策](https://code.claude.com/docs/en/data-usage)
- [GitHub 仓库](https://github.com/anthropics/claude-code)
