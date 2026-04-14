# 生成式搜索前沿进展综合 - 2026-04-07

## 综合论文
- OneSearch-V2 (2603.24422) - 潜在推理自蒸馏生成搜索
- GenFacet (2603.19665) - 生成式分面搜索
- Rich-Media Re-Ranker - 富媒体用户满意度重排
- GenIR Survey (2404.14851) - 生成式信息检索综述
- CTRL-Rec (2510.12742) - 自然语言控制推荐

---

## 一、技术演进脉络

```
传统搜索：BM25 + 倒排索引
  → Dense Retrieval（DPR, FAISS）
    → 生成式文档检索（DSI, TIGER）
      → 工业生成搜索（OneSearch）
        → OneSearch-V2：CoT + 自蒸馏 + 行为对齐
  → Faceted Search：规则 → 学习 → GenFacet（生成式端到端）
  → 重排：Pointwise/Pairwise → LLM 重排 → Rich-Media（富媒体+满意度）
  → 用户控制：对话系统 → CTRL-Rec（自然语言实时控制）
```

## 二、核心技术对比

| 子领域 | 方法 | 核心创新 | 效果 |
|--------|------|----------|------|
| 生成搜索 | OneSearch-V2 | CoT 增强+自蒸馏 | 长尾查询显著改善 |
| 分面搜索 | GenFacet | 联合生成+GRPO对齐 | Facet CTR +42%, UCVR +2% |
| 富媒体重排 | Rich-Media | VLM+LLM多维满意度 | 在线用户参与率提升 |
| GIR 综述 | GenIR Survey | GR+RAG统一框架 | 领域知识图谱 |
| 可控推荐 | CTRL-Rec | 训练时LLM模拟，部署时embedding | 实时自然语言控制 |

## 三、核心技术：CoT 增强搜索

### OneSearch-V2 的 CoT 注入机制
1. 离线：LLM 为 query-user 对生成显式 CoT
2. 压缩：将 CoT 压缩为关键词式紧凑表示
3. 在线：CoT 关键词作为补充信号注入模型输入
4. 效果：长尾和歧义查询显著改善

```
Query: "推荐一款适合旅行的背包"
CoT: "轻便, 防水, 大容量, 登机规格, 隐藏式口袋"
→ 注入搜索模型，提升语义理解
```

## 四、工业实践

1. **电商生成搜索**（OneSearch/GenFacet）：
   - CoT 内化减少推理时 LLM 调用成本
   - 行为反馈对齐（GRPO）是质量保证
   - 分面生成和查询改写联合优化

2. **富媒体搜索重排**：
   - VLM 视觉信号是差异化竞争力
   - 用户满意度多维建模（相关性、信息增益、新颖性）
   - Multi-task RL 提升场景泛化

3. **自然语言控制**（CTRL-Rec）：
   - 训练-推理解耦：训练时 LLM，推理时 embedding
   - 适合对透明度有要求的场景（新闻、信息流）

## 五、面试高频考点

**Q：生成式搜索 vs 传统搜索的优势和挑战？**
A：优势：端到端优化、更好的语义理解、易于整合 LLM 能力；挑战：生成质量控制、推理延迟、新 item 处理。

**Q：Faceted search 如何定义相关性？**
A：传统是 query-doc 相关性；GenFacet 额外考虑 facet-to-intent 对齐和 facet-to-query-rewrite 的完整性。

**Q：LLM 重排的主要挑战？**
A：延迟（LLM 慢）、上下文窗口限制、list-wise vs point-wise 的权衡、评估困难。

**Tags:** #synthesis #search #generative-retrieval #faceted-search #reranking #e-commerce

---

## 相关概念

- [[embedding_everywhere|Embedding 技术全景]]
- [[generative_recsys|生成式推荐统一视角]]
- [[multi_objective_optimization|多目标优化]]
