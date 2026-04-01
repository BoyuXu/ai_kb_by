# Claude Code 设计模式速查表

## 1. Session 状态管理模式

### 模式描述
在分布式 Agent 系统中维护会话状态，支持断点恢复。

### 实现要点
```typescript
interface Session {
  id: string;           // UUID, 通过 HTTP 头传递
  state: 'init' | 'running' | 'suspended' | 'completed';
  lastCheckpoint: {
    step: string;       // 最后完成的步骤标识
    timestamp: number;
    artifacts: any[];   // 输出物列表
  };
  context: {
    history: Message[];
    memory: LongTermMemory[];
    projectConfig: Config;
  };
  resumeCmd?: string;   // 恢复指令
}

// 恢复流程
async function resume(sessionId: string, fromStep: string) {
  const session = await loadSession(sessionId);
  const checkpoint = session.lastCheckpoint;
  
  if (checkpoint.step !== fromStep) {
    throw new Error(`Checkpoint mismatch: expected ${fromStep}, got ${checkpoint.step}`);
  }
  
  return continueExecution(session, fromStep);
}
```

### 应用场景
- 长时间运行的 Agent 任务（>5 分钟）
- 需要支持断点恢复的工作流
- 分布式任务调度系统

### Claude Code 实现
- HTTP 头 `X-Claude-Code-Session-Id` 传递会话 ID
- `--resume` 参数启用恢复模式
- 本地 `transcript files` 保存会话转录
- `CURRENT_TASK.md` 中的 Checkpoint 块存储恢复点

---

## 2. Tool 执行管道模式

### 模式描述
标准化的工具调用执行流程，支持权限检查、并行执行、异常恢复。

### 实现要点
```typescript
enum ToolStatus {
  Pending = 'pending',
  PermissionWait = 'permission_wait',
  Executing = 'executing',
  Success = 'success',
  Failed = 'failed',
  Retry = 'retry'
}

interface ToolCall {
  id: string;
  name: 'Read' | 'Write' | 'Edit' | 'Run' | 'Git';
  args: Record<string, any>;
  status: ToolStatus;
  result?: any;
  error?: Error;
  retryCount: number;
}

// 执行管道
class ToolPipeline {
  async execute(toolCalls: ToolCall[]): Promise<ToolResult[]> {
    // Phase 1: Permission Check
    const permResults = await this.checkPermissions(toolCalls);
    
    // Phase 2: Parallel Execution
    const executing = toolCalls.map(async (call) => {
      try {
        return await this.executeTool(call);
      } catch (error) {
        if (isRetryable(error) && call.retryCount < MAX_RETRIES) {
          call.retryCount++;
          return await this.execute([call]); // Retry
        }
        throw error;
      }
    });
    
    // Phase 3: Collect Results
    return Promise.all(executing);
  }
}
```

### 应用场景
- 需要权限控制的工具执行
- 可能失败需要重试的操作（Git、网络请求）
- 支持并行执行的任务
- 多阶段处理流程

### Claude Code 实现
- PreToolUse 钩子进行权限验证
- 条件过滤减少进程启动（`if` 字段）
- Tool result 块确保 tool_use id 对应
- MCP 服务器集成作为工具提供者

---

## 3. LLM Context Assembly 模式

### 模式描述
高效地组织和优化发给 LLM 的上下文，支持 token 优化和缓存。

### 实现要点
```typescript
interface LLMContext {
  systemPrompt: string;
  messages: Message[];
  tools: ToolDefinition[];
  fileReferences: FileRef[];
  memories: Memory[];
}

class ContextAssembler {
  // Token 预算管理
  private readonly TOKEN_BUDGET = 180000;
  private reserved = {
    response: 8000,
    reasoning: 10000,
    tools: 5000
  };
  
  assemble(state: AgentState): LLMContext {
    const available = this.TOKEN_BUDGET - Object.values(this.reserved).sum();
    
    // 优先级分配
    const recentHistory = this.truncateMessages(
      state.history,
      available * 0.4  // 40% 用于历史
    );
    
    const codeSnippets = this.extractRelevantCode(
      state.projectFiles,
      state.currentTask,
      available * 0.3  // 30% 用于代码
    );
    
    const memories = this.rankMemories(
      state.memory,
      state.query,
      available * 0.2  // 20% 用于记忆
    );
    
    // 应用缓存优化
    return {
      systemPrompt: this.getCachedPrompt(state.model),
      messages: recentHistory,
      tools: this.filterTools(state.skills, state.permissions),
      fileReferences: codeSnippets,
      memories: memories
    };
  }
  
  // Token 计数（准确计数)
  countTokens(context: LLMContext): number {
    return enc.encode(JSON.stringify(context)).length;
  }
}
```

### 应用场景
- 长对话场景（需要管理上下文窗口）
- 需要实现 Prompt Caching 的系统
- 多模态输入（代码 + 图像 + 文本）
- Token-aware 的应用

### Claude Code 实现
- 文件内容不再 JSON 转义（节省 token）
- 技能描述长度限制（250 字符）
- Prompt Caching（Bedrock/Vertex/Foundry）
- 内存系统（LanceDB Pro）用于相关性排序

---

## 4. 条件技能执行模式

### 模式描述
支持根据权限规则和条件表达式动态过滤和执行技能。

### 实现要点
```typescript
interface Skill {
  name: string;
  description: string;
  permission: PermissionRule;  // 如 "Bash(git *)"
  condition?: ConditionalExpression;  // 如 "isGitRepo && hasChanges"
  execute: (args: any) => Promise<any>;
}

enum PermissionLevel {
  Allow = 'allow',
  Deny = 'deny',
  Ask = 'ask'
}

interface PermissionRule {
  tool: 'Bash' | 'Read' | 'Write' | 'Git' | 'Run';
  pattern: string;  // glob or regex
}

class SkillFilter {
  evaluateCondition(expr: ConditionalExpression, context: ExecutionContext): boolean {
    // 支持简单的布尔表达式
    // 如: "isGitRepo && !isDirty || force"
    const evaluator = new ConditionEvaluator(context);
    return evaluator.eval(expr);
  }
  
  filterSkills(
    skills: Skill[],
    userPermissions: PermissionRule[],
    context: ExecutionContext
  ): Skill[] {
    return skills.filter(skill => {
      // Check permissions
      const hasPermission = this.checkPermission(skill.permission, userPermissions);
      if (!hasPermission) return false;
      
      // Check conditions
      if (skill.condition) {
        return this.evaluateCondition(skill.condition, context);
      }
      
      return true;
    });
  }
}
```

### 应用场景
- 组织内的权限管理（限制某些命令）
- 条件激活功能（只在特定条件下可用）
- 减少进程启动开销（过滤不必要的工具）
- 安全的多用户环境

### Claude Code 实现
- `managed-settings.json` 存储组织策略
- PreToolUse 钩子支持 `if` 字段过滤
- Hook 系统用于生命周期管理（WorktreeCreate 等）

---

## 5. 内存管理 & 长期记忆模式

### 模式描述
支持跨会话的长期记忆，同时优化单会话的短期缓存。

### 实现要点
```typescript
interface Memory {
  id: string;
  type: 'fact' | 'decision' | 'error' | 'pattern';
  content: string;
  embedding?: number[];  // Jina embedding
  metadata: {
    scope: 'global' | 'agent:main' | 'agent:coder';
    createdAt: number;
    relevance: number;
    tags: string[];
  };
}

class MemorySystem {
  private shortTerm = new LRUCache<string, any>(1000);  // 单会话
  private longTerm = new LanceDB();  // 跨会话
  
  async recall(query: string, scope: MemoryScope, limit: number = 10): Promise<Memory[]> {
    // 向量相似度搜索
    const embeddings = await jina.embed([query]);
    const results = await this.longTerm.query(embeddings[0])
      .where(`scope IN [${scope}]`)
      .limit(limit)
      .toArray();
    
    return results.map(r => ({
      ...r,
      relevance: cosineSimilarity(embeddings[0], r.embedding)
    }));
  }
  
  async capture(memory: Memory): Promise<void> {
    // 自动采集（autoCapture: true）
    const embedding = await jina.embed([memory.content]);
    memory.embedding = embedding[0];
    
    await this.longTerm.add(memory);
    
    // 同时写入短期缓存
    this.shortTerm.set(memory.id, memory);
  }
}
```

### 应用场景
- 持久化学习（从历史任务中学习）
- 跨 Agent 协作（共享知识库）
- 错误防护（记录失败案例）
- 上下文增强（提升 LLM 决策质量）

### Claude Code 实现
- LanceDB Pro 插件支持向量化内存
- `autoCapture: true` 自动记录关键事件
- `autoRecall: true` 自动注入相关记忆
- Scope 权限控制（哪个 Agent 看哪些记忆）

---

## 6. 流式响应 & 渐进式 UI 模式

### 模式描述
支持实时流式输出，提升用户体验。

### 实现要点
```typescript
async function* streamResponse(
  modelStream: AsyncIterable<Event>
): AsyncGenerator<UIUpdate> {
  let buffer = '';
  let toolCalls: ToolCall[] = [];
  
  for await (const event of modelStream) {
    if (event.type === 'content_block_delta') {
      // 实时文本更新
      buffer += event.delta.text;
      
      // 渐进式解析（不等完整响应）
      yield {
        type: 'text_update',
        content: buffer,
        partial: true
      };
    }
    
    if (event.type === 'content_block_start') {
      if (event.content_block.type === 'tool_use') {
        yield {
          type: 'tool_start',
          toolName: event.content_block.name,
          toolId: event.content_block.id
        };
      }
    }
  }
  
  // 完整更新
  yield {
    type: 'text_complete',
    content: buffer,
    partial: false
  };
}
```

### 应用场景
- 提升 UX（不要等待完整生成）
- 实时工具执行反馈
- 长输出分块处理

### Claude Code 实现
- VS Code 扩展中的内联 diff（实时渲染）
- 终端流式输出
- 桌面应用中的实时状态更新

---

## 7. 错误恢复与重试模式

### 模式描述
分层次的错误处理和智能重试策略。

### 实现要点
```typescript
enum ErrorLevel {
  Transient = 'transient',     // 临时错误，可重试
  Permanent = 'permanent',     // 永久错误，不可重试
  Unknown = 'unknown'          // 未知错误，应询问用户
}

interface RetryPolicy {
  maxAttempts: number;
  backoffMs: number;  // 初始退避时间
  maxBackoffMs: number;
  exponentialBase: number;
}

class ErrorRecovery {
  async executeWithRetry<T>(
    fn: () => Promise<T>,
    policy: RetryPolicy = DEFAULT_POLICY,
    context: ErrorContext = {}
  ): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= policy.maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        const level = this.classifyError(error);
        
        if (level === ErrorLevel.Permanent) {
          throw error;  // 不重试
        }
        
        if (attempt < policy.maxAttempts) {
          const backoff = this.calculateBackoff(attempt, policy);
          await delay(backoff);
          continue;
        }
      }
    }
    
    // 所有重试都失败
    throw new ExhaustedRetriesError(lastError, context);
  }
  
  // 错误分类
  private classifyError(error: Error): ErrorLevel {
    if (error instanceof NetworkError) return ErrorLevel.Transient;
    if (error instanceof PermissionError) return ErrorLevel.Permanent;
    if (error instanceof ContextExceededError) {
      // 特殊处理：触发 /compact
      return ErrorLevel.Transient;
    }
    return ErrorLevel.Unknown;
  }
}
```

### 应用场景
- 网络不稳定的场景
- 限流和速率限制
- 超大 token 会话的 OOM 恢复
- Cowork Dispatch 消息丢失恢复

### Claude Code 实现
- Tool 失败自动重试（Git、网络操作）
- 超大会话 `/compact` 清理
- Cowork Dispatch 消息重新传递
- Session 恢复机制

---

## 快速选择表

| 场景 | 推荐模式 | 关键类 |
|------|--------|--------|
| 长时间 Agent 任务 | Session 状态管理 | Session, Checkpoint |
| 需要权限控制 | Tool 执行管道 + 条件过滤 | ToolPipeline, PermissionRule |
| 长对话上下文管理 | LLM Context Assembly | ContextAssembler, TokenBudget |
| 跨会话学习 | 内存管理 & 长期记忆 | MemorySystem, LanceDB |
| 改善用户体验 | 流式响应 | StreamResponse, UIUpdate |
| 不稳定环境 | 错误恢复与重试 | ErrorRecovery, RetryPolicy |
| 组织级权限 | 条件技能执行 | SkillFilter, ConditionalExpression |

