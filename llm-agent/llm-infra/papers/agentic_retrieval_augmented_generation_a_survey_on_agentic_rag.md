# Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG
> 来源：arxiv/2501.xxxxx | 领域：llm-infra | 学习日期：20260326

## 问题定义
传统 RAG（Retrieval-Augmented Generation）的局限：
- 单轮检索：一次性检索，无法处理复杂多步问题
- 静态检索策略：无法根据问题复杂度自适应调整
- 无反思能力：检索失败时不能重试或重新规划
- 工具调用受限：只能检索，无法执行代码/查询数据库/调用 API

## 核心方法与创新点
**Agentic RAG**：将 Agent 能力（规划/工具调用/自我反思）与 RAG 结合。

**四种 Agentic RAG 模式：**

**1. 单步 Agent RAG（Naive Agentic）：**
```
Query → Planner → [Retrieve, Execute] → LLM → Answer
```

**2. 迭代 RAG（Iterative Refinement）：**
```python
answer = None
for i in range(max_iter):
    context = retriever.search(query + answer)  # 用中间结果精化检索
    answer = llm.generate(query, context)
    if confident(answer): break
```

**3. 多 Agent RAG（Multi-Agent）：**
```
Orchestrator Agent
├── Researcher Agent  (检索+整合)
├── Coder Agent       (执行代码验证)
├── Critic Agent      (评估答案质量)
└── Synthesis Agent   (最终综合)
```

**4. 自适应 RAG（Adaptive）：**
```python
complexity = classifier(query)  # 简单/中等/复杂
if complexity == "simple":
    answer = direct_generation(query)
elif complexity == "medium":
    answer = single_rag(query)
else:
    answer = agentic_multi_step_rag(query)
```

**FLARE（Forward-Looking Active Retrieval）：**
```python
# 仅在生成不确定时触发检索
tokens = []
for position in generation:
    next_token, confidence = lm.generate(context + tokens)
    if confidence < threshold:
        new_docs = retrieve(query + "".join(tokens))
        context = update(context, new_docs)
    tokens.append(next_token)
```

## 实验结论
- 多跳 QA（HotpotQA）：Agentic RAG +15.3%（vs 单次 RAG）
- 复杂推理（FEVER）：+8.7%
- 代码生成+验证：Pass@1 +12.1%（执行反馈让 Agent 自纠错）
- 长文档理解（∞-Bench）：+18.4%（迭代检索关键段落）

## 工程落地要点
1. **工具调用标准化**：统一 Tool Schema（JSON Schema），支持 Retrieve/Execute/Search/API Call
2. **循环防止**：设置最大迭代次数（通常 5-10 轮），避免无限循环
3. **上下文管理**：多轮检索后 Context Window 膨胀，用 Summarizer 压缩历史
4. **成本控制**：每次 LLM 调用计费，复杂问题 Agentic RAG 成本是普通 RAG 的 5-20x
5. **可观察性**：记录每步的 Action/Observation，便于调试和改进

## 面试考点
**Q1: Agentic RAG 与传统 RAG 的核心区别是什么？**
A: 传统 RAG：单次检索 → 固定生成，无自适应能力。Agentic RAG：① 动态决策（何时检索/检索什么）②多步迭代（根据中间结果调整检索）③工具扩展（不只检索，还可执行/搜索/API调用）④自我反思（评估答案质量，不满足则重来）。

**Q2: FLARE 的核心思想是什么？**
A: 主动检索而非被动检索：LLM 在生成过程中实时监测生成置信度，当置信度低（说明知识不足）时才触发检索，并使用已生成的部分作为更精确的检索 query。这比事先检索更准确，比全程检索更高效。

**Q3: 多 Agent RAG 如何分工协作？**
A: 典型分工：Planner（分解问题为子任务）→ Researcher（检索相关信息）→ Executor（执行代码/API验证）→ Critic（评估质量/一致性）→ Synthesizer（综合答案）。关键：清晰的工具定义 + Orchestrator 的任务分配 + Agent 间的信息传递机制。

**Q4: Agentic RAG 的主要挑战？**
A: ①延迟：多步迭代 × LLM 推理延迟 = 高延迟（10-60秒）②成本：每步都是 LLM API 调用 ③可靠性：复杂 Agent 链中任何一步失败都可能级联失败 ④幻觉传播：错误的中间结果被后续步骤放大。

**Q5: 如何评估 Agentic RAG 系统的质量？**
A: 端到端：最终答案准确率（QA 任务）、用户满意度；中间过程：检索准确率（Retrieved docs 相关性）、工具调用成功率、步骤数效率；成本：平均 token 消耗、平均延迟；可靠性：错误率、循环检测率。
