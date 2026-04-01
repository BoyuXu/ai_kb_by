# Claude Code 核心模块清单

## 核心模块架构

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
│  (Terminal CLI / VS Code / Desktop App / Browser / GitHub)
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Session Manager                        │
│  ├─ Session ID tracking (HTTP header)                   │
│  ├─ Conversation history management                     │
│  ├─ Resume/checkpoint system                            │
│  └─ State persistence                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
      ┌────────────────┼────────────────┐
      │                │                │
┌─────▼────────┐ ┌────▼──────┐ ┌──────▼──────┐
│   Message    │ │   Intent  │ │  Permission │
│   Parser     │ │ Detector  │ │   Manager   │
└─────┬────────┘ └────┬──────┘ └──────┬──────┘
      │                │               │
      └────────────────┼───────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            Skill & Plugin Resolution                    │
│  ├─ Skill registry and loading                          │
│  ├─ Plugin marketplace integration                      │
│  ├─ Conditional skill filtering (if expressions)       │
│  ├─ Hook system (PreToolUse, PostExecution)           │
│  └─ Organization policy enforcement                    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Tool Execution Pipeline                    │
│  ├─ Tool Registry                                       │
│  │  ├─ Read (files, with @ mentions)                    │
│  │  ├─ Write/Edit (with conditional rules)             │
│  │  ├─ Run (bash/shell execution)                       │
│  │  └─ Git (version control workflows)                  │
│  ├─ Permission Validation (PreToolUse hooks)           │
│  ├─ Parallel Execution Engine                          │
│  ├─ Result Aggregation                                 │
│  └─ Error Recovery & Retry Logic                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│             Context Assembly & Optimization             │
│  ├─ Token budget management                            │
│  ├─ File reference extraction                          │
│  ├─ Memory retrieval (LanceDB Pro)                     │
│  ├─ Prompt caching (Bedrock/Vertex)                    │
│  ├─ Tool definition filtering                          │
│  └─ @-mention resolution                               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                LLM Integration Layer                     │
│  ├─ Model selection (Sonnet/Opus/Haiku)                │
│  ├─ API client (Anthropic/Bedrock/Vertex)              │
│  ├─ Streaming manager                                  │
│  ├─ Tool use handler                                   │
│  ├─ Response parsing                                   │
│  └─ Streaming response producer                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Memory & Context Management                │
│  ├─ Short-term cache (LRU)                             │
│  ├─ Long-term storage (LanceDB Pro)                    │
│  ├─ Auto-capture system                                │
│  ├─ Auto-recall integration                            │
│  ├─ Scope-based access control                         │
│  └─ Embedding service (Jina)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Project & File Management                  │
│  ├─ Project config loader (.claude/, CLAUDE.md)        │
│  ├─ Code indexing & search                             │
│  ├─ File caching & metadata                            │
│  ├─ VCS integration (Git/Jujutsu/Sapling)             │
│  ├─ Worktree management                                │
│  └─ Gitignore parsing                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
└──────────────────────▼──────────────────────────────────┘
             System & Infrastructure
  ├─ Configuration management
  ├─ Logging & telemetry (OpenTelemetry)
  ├─ Terminal UI rendering
  ├─ File system operations
  └─ Process management
```

## 模块详解

### 1. **Session Manager** (会话管理)
**职责**: 维护用户会话状态，支持多终端同步和恢复

**主要接口**:
```typescript
interface SessionManager {
  createSession(projectPath: string): Promise<Session>;
  getSession(sessionId: string): Promise<Session>;
  saveCheckpoint(sessionId: string, step: string, artifacts: any[]): Promise<void>;
  resumeSession(sessionId: string, fromStep: string): Promise<void>;
  saveTranscript(sessionId: string, messages: Message[]): Promise<void>;
  compactSession(sessionId: string): Promise<void>;  // 清理超大会话
}
```

**关键特性**:
- Session ID 通过 HTTP 头 `X-Claude-Code-Session-Id` 传递
- 本地转录文件保存（支持离线恢复）
- Checkpoint 系统（支持中断和恢复）
- 自动 OOM 清理（`/compact` 命令）

**依赖**: ProjectManager, MemorySystem

---

### 2. **Message Parser & Intent Detector** (消息解析 & 意图检测)
**职责**: 解析用户输入，检测意图，提取 @ 提及

**主要接口**:
```typescript
interface MessageParser {
  parse(raw: string): ParsedMessage;
  detectIntent(message: string): 'code_edit' | 'analysis' | 'workflow' | ...;
  extractMentions(message: string): {
    files: string[];
    agents: string[];
    skills: string[];
  };
}
```

**关键特性**:
- @ 文件提及解析（如 `@README.md`）
- @ 代理提及（@ 其他 agent）
- 命令识别（`/command`）
- 上下文感知的解析

**依赖**: ProjectManager, FileManager

---

### 3. **Permission Manager** (权限管理)
**职责**: 控制工具访问权限，应用组织策略

**主要接口**:
```typescript
interface PermissionManager {
  checkPermission(tool: string, resource: string): PermissionLevel;
  evaluateRule(rule: PermissionRule, context: ExecutionContext): boolean;
  loadOrgPolicy(policyFile: string): void;
  getUnrestrictedSkills(permissions: PermissionRule[]): Skill[];
}
```

**关键特性**:
- 权限规则匹配（glob/regex）
- 条件表达式评估（`if` 字段）
- 组织策略应用（`managed-settings.json`）
- Hook-based 权限检查（PreToolUse）

**依赖**: SkillRegistry, ConfigManager

---

### 4. **Skill Registry & Plugin Manager** (技能注册 & 插件管理)
**职责**: 加载、管理、过滤技能和插件

**主要接口**:
```typescript
interface SkillRegistry {
  loadSkill(skillPath: string): Promise<Skill>;
  filterByPermission(skills: Skill[], perms: PermissionRule[]): Skill[];
  filterByCondition(skills: Skill[], context: ExecutionContext): Skill[];
  registerMCPServer(config: MCPConfig): Promise<void>;
  listAvailableSkills(): Skill[];
  uninstallPlugin(pluginId: string): Promise<void>;
}
```

**关键特性**:
- 官方市场 + 社区插件
- 条件激活（`if` 字段）
- Hook 系统（PreToolUse, WorktreeCreate）
- MCP (Model Context Protocol) 支持
- RFC 9728 OAuth 集成

**依赖**: PermissionManager, ConfigManager

---

### 5. **Tool Execution Pipeline** (工具执行管道)
**职责**: 标准化工具调用、执行、结果处理

**主要接口**:
```typescript
interface ToolExecutor {
  execute(toolCall: ToolCall): Promise<ToolResult>;
  executeParallel(toolCalls: ToolCall[]): Promise<ToolResult[]>;
  retry(toolCall: ToolCall, policy: RetryPolicy): Promise<ToolResult>;
}

interface ToolRegistry {
  get(name: string): Tool;
  register(name: string, tool: Tool): void;
}
```

**核心工具**:
- **Read** — 读取文件（支持 JSON 压缩编码）
- **Write** — 创建文件（支持条件规则）
- **Edit** — 编辑文件（精确替换）
- **Run** — 执行命令（bash/shell）
- **Git** — 版本控制操作

**关键特性**:
- 权限检查（PreToolUse 钩子）
- 并行执行能力
- 智能重试（transient error 分类）
- Result block 完整性验证
- 内存优化（原始字符串编码）

**依赖**: PermissionManager, ErrorRecovery

---

### 6. **Context Assembler** (上下文组织)
**职责**: 优化 LLM 输入，管理 token 预算

**主要接口**:
```typescript
interface ContextAssembler {
  assemble(state: AgentState): LLMContext;
  countTokens(context: LLMContext): number;
  truncateMessages(messages: Message[], tokenLimit: number): Message[];
  rankMemories(memories: Memory[], query: string, limit: number): Memory[];
  filterTools(skills: Skill[], permissions: PermissionRule[]): ToolDefinition[];
}
```

**优化策略**:
- Token 预算管理（180k limit, 分配给 response/reasoning/tools）
- 优先级分配（历史 40%, 代码 30%, 记忆 20%）
- 文件内容去 JSON 转义
- 技能描述长度限制（250 字符）
- @ 提及文件使用原始字符串

**依赖**: MemorySystem, ToolRegistry

---

### 7. **LLM Integration** (LLM 集成)
**职责**: 调用 Claude 模型，处理流式响应

**主要接口**:
```typescript
interface LLMClient {
  chat(request: ChatRequest, stream: boolean): Promise<ChatResponse> | AsyncIterable<Event>;
  selectModel(hint?: string): ModelConfig;
  parseToolUse(event: ContentBlockDelta): ToolCall;
  createPrompt(context: LLMContext): Message;
}
```

**支持的模型**:
- Claude 3.5 Sonnet (default)
- Claude 3 Opus 4.5
- Claude 3 Haiku

**API 提供商**:
- Anthropic API
- AWS Bedrock
- Google Vertex AI
- Custom Foundry

**关键特性**:
- Prompt Caching（提高 hit rate）
- 流式响应处理
- Tool use 自动解析
- 模型选择逻辑

**依赖**: ConfigManager, ContextAssembler

---

### 8. **Memory System** (内存系统)
**职责**: 短期和长期记忆管理，支持跨会话学习

**主要接口**:
```typescript
interface MemorySystem {
  capture(memory: Memory): Promise<void>;  // 自动采集
  recall(query: string, scope: Scope, limit: number): Promise<Memory[]>;  // 自动检索
  rankByRelevance(memories: Memory[], query: string): Memory[];
  clearScope(scope: Scope): Promise<void>;
}
```

**内存类型**:
- fact (事实)
- decision (决策)
- error (错误)
- pattern (模式)

**存储层**:
- **Short-term**: LRU Cache (1000 items, 单会话)
- **Long-term**: LanceDB Pro (向量化存储, 跨会话)

**关键特性**:
- 自动采集（autoCapture: true）
- 自动检索（autoRecall: true）
- 向量相似度搜索（Jina embedding）
- Scope 权限控制
- Token 优化的记忆编码

**依赖**: LanceDB, EmbeddingService

---

### 9. **Project Manager** (项目管理)
**职责**: 加载项目配置，管理代码索引

**主要接口**:
```typescript
interface ProjectManager {
  loadConfig(projectPath: string): Promise<ProjectConfig>;
  indexCode(projectPath: string): Promise<CodeIndex>;
  resolveFile(path: string): ResolvedPath;
  getVCSInfo(): VCSInfo;
}
```

**配置源**:
- `.claude/` 目录（命令和配置）
- `CLAUDE.md` (项目指导)
- `managed-settings.json` (组织策略)

**代码索引**:
- 文件树缓存
- 符号索引（函数、类、变量）
- 依赖关系图

**关键特性**:
- 多 VCS 支持（Git, Jujutsu, Sapling）
- Worktree 管理
- 文件排除列表（`.git`, `.jj`, `.sl`）

**依赖**: FileManager, VCSManager

---

### 10. **Error Recovery & Retry** (错误恢复)
**职责**: 分类错误，执行智能重试和恢复

**主要接口**:
```typescript
interface ErrorRecovery {
  classify(error: Error): ErrorLevel;
  retry(fn: () => Promise<T>, policy: RetryPolicy): Promise<T>;
  recover(error: Error, context: ErrorContext): Promise<RecoveryAction>;
}
```

**错误分类**:
- Transient (临时, 可重试)
  - 网络错误
  - 限流
  - 临时锁
- Permanent (永久, 不可重试)
  - 权限错误
  - 文件不存在
  - 无效语法
- Unknown (未知, 询问用户)

**恢复策略**:
- 指数退避重试
- 自动 `/compact` (Context 超限)
- Cowork Dispatch 消息重传
- Session 恢复（`--resume`）

**依赖**: Logger, MetricsCollector

---

### 11. **File Manager** (文件管理)
**职责**: 跨项目的文件操作，支持超大文件

**主要接口**:
```typescript
interface FileManager {
  read(path: string, range?: Range): Promise<string>;
  write(path: string, content: string): Promise<void>;
  edit(path: string, oldText: string, newText: string): Promise<void>;
  list(dir: string, recursive?: boolean): Promise<FileInfo[]>;
  getMetadata(path: string): Promise<FileMetadata>;
}
```

**特性**:
- 文件内容缓存（性能优化）
- 行号计数（token 优化）
- 编码检测
- 大文件分块读取

**依赖**: ProjectManager

---

### 12. **VCS Manager** (版本控制)
**职责**: Git/Jujutsu/Sapling 操作

**主要接口**:
```typescript
interface VCSManager {
  getStatus(): Promise<VCSStatus>;
  commit(message: string, files: string[]): Promise<CommitResult>;
  createWorktree(name: string, ref?: string): Promise<string>;
  switchBranch(name: string): Promise<void>;
  listBranches(): Promise<Branch[]>;
}
```

**支持的 VCS**:
- Git (主要)
- Jujutsu (新兴)
- Sapling (meta)

**依赖**: FileManager, ProcessManager

---

## 模块依赖图 (简化)

```
SessionManager ─────────────┐
                            │
ProjectManager ──────────┬──┼────┬─────────────────┐
                         │  │    │                 │
FileManager ─────────────┤  │    │                 │
                         │  │    │                 │
VCSManager ──────────────┤  │    │                 │
                         │  │    │                 │
MessageParser ───────────┤  │    │                 │
                         │  │    │                 │
SkillRegistry ────┬──────┤  │    │                 │
                  │      │  │    │                 │
PermissionManager ┤      │  │    │                 │
                  │      │  │    │                 │
ToolExecutor ─────┴──────┼──┼────┤                 │
                         │  │    │                 │
ContextAssembler ────────┼──┼────┤                 │
                         │  │    │                 │
MemorySystem ────────────┼──┼────┤                 │
                         │  │    │                 │
LLMClient ───────────────┼──┼────┤                 │
                         │  │    │                 │
ErrorRecovery ───────────┼──┼────┤                 │
                         ▼  ▼    ▼                 ▼
                    ┌────────────────────────────┐
                    │   Agent Main Loop          │
                    │   (Orchestration)          │
                    └────────────────────────────┘
```

## 关键数据流

### 1. 会话启动流程
```
User Command "claude"
  ↓
Session Manager: createSession()
  ↓
Project Manager: loadConfig()
  ↓
Skill Registry: loadSkills()
  ↓
Tool Registry: registerTools()
  ↓
Ready for input
```

### 2. 消息处理流程
```
User Input
  ↓
Message Parser: parse()
  ↓
Intent Detector: detectIntent()
  ↓
Permission Manager: checkPermissions()
  ↓
Skill Registry: filterSkills()
  ↓
Context Assembler: assemble()
  ↓
LLM Client: chat(stream=true)
  ↓
Tool Execution Pipeline
  ├─ PreToolUse hooks
  ├─ Parallel execute
  └─ Collect results
  ↓
Memory System: capture()
  ↓
Response streaming to UI
```

### 3. 工具执行流程
```
Tool Call
  ↓
Permission Check (PreToolUse hooks)
  ↓
If denied → Ask User
If allowed → Execute
  ↓
Try:
  Execute tool
  Collect result
Catch:
  Classify error
  If transient: retry with backoff
  If permanent: report error
  If unknown: ask user
  ↓
Tool Result block
  ↓
Memory: capture()
```

