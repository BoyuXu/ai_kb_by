# Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
> 来源：arXiv:2503.09516 | 领域：search | 学习日期：20260330

## 问题定义
LLM 在复杂问答任务中存在知识截止（knowledge cutoff）和幻觉问题，需要实时检索外部信息。但传统 RAG 是被动检索（输入 query → 检索 → 生成），缺乏对"何时需要检索"、"搜什么"的主动推理能力。Search-R1 训练 LLM 主动决策搜索时机和搜索策略，实现"Reasoning + Search" 的智能协同。

## 核心方法与创新点
1. **Interleaved Reasoning and Search**：LLM 在推理过程中动态插入搜索动作：
   `<think>当前信息不足，需要查询...</think><search>query text</search><result>...</result><think>结合结果...</think>`
2. **RL 训练搜索策略**：以最终答案正确性为 reward，用 GRPO 训练 LLM 的搜索决策（何时搜、搜什么），不需要人工标注搜索轨迹。
3. **Process Reward**：除最终 reward 外，引入中间步骤 reward（搜索 query 质量、推理一致性），提升训练稳定性。
4. **Search Engine API**：接入 Google/Bing/Wikipedia API，LLM 生成结构化搜索 query，API 返回 top-k 结果。
5. **Multi-hop Reasoning**：支持多轮搜索（搜完第一个问题，发现需要进一步搜索），解决复杂多跳推理任务。

## 实验结论
- HotpotQA（多跳问答）：EM +15.3%，F1 +12.8%（对比 RAG baseline）
- TriviaQA（时效性知识）：EM +8.6%（对比 LLM 直接回答）
- 搜索效率：平均每问题搜索 2.3 次（比 naive 多搜策略节省 60% API 调用）

## 工程落地要点
- 搜索 API 延迟是瓶颈（Google API ~500ms），需要异步搜索 + 并行推理
- 搜索 query 生成质量关键，需专门训练 query rewriter
- 上下文长度管理：多轮搜索结果累积，超长时需要动态压缩历史
- 搜索结果去重和质量过滤：避免低质量/spam 内容污染推理

## 面试考点
- Q: RAG 和 Search-R1 的核心区别？
  - A: RAG：固定检索 → 生成（被动、一次性）；Search-R1：主动决策搜索时机和内容，多轮迭代，推理驱动检索
- Q: 如何训练 LLM 的搜索决策策略？
  - A: 关键是 reward 设计：最终答案正确 → 正向 reward；浪费搜索（搜了没用）→ 惩罚；遗漏搜索（需要但没搜）→ 惩罚
- Q: Multi-hop 问答中如何保证推理链一致性？
  - A: 每步搜索后更新 working memory（已知事实集合）；推理链验证（前后结论不冲突）；Beam Search 保留多条推理路径

## 数学公式
$$\pi_\theta: s_t \rightarrow a_t \in \{\text{think}, \text{search}(q), \text{answer}(a)\}$$

$$R = \mathbb{1}[\text{answer correct}] + \sum_t r_t^{\text{process}}$$
