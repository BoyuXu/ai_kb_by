# Tiny-Critic RAG: Empowering Agentic Fallback with Parameter-Efficient Small LMs

> 来源：arxiv | 日期：20260316 | 领域：llm-infra

## 问题定义

LLM 在面对知识缺口或不确定场景时会产生幻觉（hallucination）。RAG 通过检索补充信息缓解此问题，但检索本身也可能失败（检索器无法找到相关文档）。

本文提出 **Tiny-Critic**：用超轻量级模型（1-3B 参数）作为"评论家"，判断大模型的回答是否可信，若不可信触发 RAG fallback。相比直接调用 RAG，节省计算成本。

## 核心方法与创新点

1. **评论家任务定义**：
   - 输入：用户问题 + LLM 生成的初步答案（不带检索结果）
   - 输出：置信度分数（0-1）+ 判断是否需要检索（binary label）
   - 训练数据：标注问题-答案对的"是否需要检索"（用 oracle 判断：答案是否完全由 LLM 知识回答，还是需要额外检索）

2. **两级评估**：
   - **自信评估（Confidence Estimation）**：对已有答案的置信度评分。
   - **可检索性评估（Retrievability Assessment）**：预测问题是否适合用 RAG 改进（某些问题即使检索也无法改进）。

3. **轻量级设计**：
   - 用 DistilBERT / TinyBERT 类架构（1-3B 参数）。
   - 最小化输入（不拼接长文档），只用问题 + 答案摘要。
   - 推理耗时 < 100ms（相比大模型 RAG 的 1-5s）。

4. **Agentic Fallback 流程**：
```
user_query → LLM_generate_answer → Tiny_Critic_evaluate
                                    ↓
                            conf_score > threshold?
                            ↙                        ↘
                          YES                        NO
                        return                    RAG_retrieve
                        answer                   LLM_generate_with_docs
                                                 return refined_answer
```

## 实验结论

- 开放域 QA（Natural Questions）：相比始终调用 RAG，Tiny-Critic Fallback 减少 70% RAG 调用，同时保持答案准确率（仅损失 0.8%）。
- 推理延迟：端到端从 2.5s（总是 RAG）降低到 0.8s（平均，包括 30% 需要 RAG 的情况）。
- 置信度预测准确率 > 92%（在判断是否需要检索上）。

## 工程落地要点

- 训练数据构造：对现有 QA 数据集，用启发式判断（答案是否在预训练 cutoff 之后的知识，是否涉及实时信息）标注"是否需要检索"。
- 置信度阈值调优：业务权衡——降低阈值避免幻觉但增加 RAG 调用；提高阈值减少调用但接受少量错误。通常 0.6-0.7 之间。
- 批量推理：Tiny-Critic 推理可以在 CPU 上，不必占用 GPU（轻量级）。
- 动态阈值：根据用户 intent 或领域调整阈值（医学诊断严格，娱乐问答宽松）。

## 常见考点

- Q: 为什么评论家用小模型而不是大模型自我评估？
  A: (1) 小模型更容易训练（数据效率高）；(2) 推理快（成本低），适合额外的评估环节；(3) 大模型自评倾向于过度自信（特别是当被问"你确定吗"时，倾向说是）；(4) 小模型的评估结果更客观（因为它没见过那么多知识，更谦虚）。

- Q: 如何避免 Tiny-Critic 本身也产生错误的评估？
  A: (1) 用多个小评论家 ensemble；(2) 定期 audit 它的判断（上线后监控，当评论家拒绝检索但答案其实错了）；(3) 加入 oracle feedback 持续微调；(4) 保守设计：当不确定时倾向于建议 RAG（recall > precision）。

- Q: Tiny-Critic 和 Cascade Retrieval 的区别？
  A: Cascade Retrieval：先轻量级检索器，再精排。Tiny-Critic：先生成答案，再评估是否需要检索。前者优化"检索质量"，后者优化"是否值得检索"。它们可以结合使用。
