# Collab-RAG: White-Box and Black-Box LLM Collaboration for RAG
> 来源：arxiv/2412.xxxxx | 领域：llm-infra | 学习日期：20260326

## 问题定义
RAG 系统中单一 LLM 的局限：
- 黑盒大模型（GPT-4）：能力强但成本高、无法微调、无法访问内部状态
- 白盒小模型（Llama-7B）：可微调但能力有限，检索质量差
- 无法同时获得：大模型的推理能力 + 小模型的可控性和低成本
- 复杂问题需要多跳推理，单模型 RAG 容易失败

## 核心方法与创新点
**Collab-RAG**：白盒小模型 + 黑盒大模型协作的 RAG 框架。

**分工架构：**
```
White-Box Small LLM（可微调）：负责检索质量提升
  - 查询改写（Query Rewriting）
  - 子问题分解（Sub-question Decomposition）
  - 检索结果评估（Retrieval Evaluation）

Black-Box Large LLM（GPT-4等）：负责最终推理生成
  - 复杂推理
  - 答案综合
  - 质量保证
```

**协作流程：**
```python
def collab_rag(question):
    # Step 1: 小模型分解问题
    sub_questions = small_llm.decompose(question)
    
    # Step 2: 小模型改写每个子问题（针对检索优化）
    retrieved_docs = []
    for sq in sub_questions:
        rewritten_q = small_llm.rewrite_for_retrieval(sq)
        docs = retriever.search(rewritten_q)
        quality = small_llm.evaluate_retrieval(sq, docs)
        if quality < threshold:
            docs = retriever.search(small_llm.rethink_query(sq, docs))
        retrieved_docs.extend(docs)
    
    # Step 3: 大模型综合生成（小模型提供高质量上下文）
    answer = large_llm.generate(question, retrieved_docs)
    return answer
```

**小模型微调目标：**
```python
# 微调小模型专注于检索辅助任务
L = L_decompose + L_rewrite + L_evaluate
# 用大模型生成的 Silver Label 训练
silver_data = large_llm.generate_training_data(raw_data)
small_llm_finetuned = finetune(small_llm, silver_data)
```

## 实验结论
- Multi-Hop QA（HotpotQA, MuSiQue）：
  - Exact Match：+6.8%（vs 单独使用 GPT-4 RAG）
  - 成本：比纯 GPT-4 少 60%（子任务由小模型处理）
- 单跳 QA（NQ, TriviaQA）：与 GPT-4 RAG 持平
- 小模型质量提升：微调后小模型的查询改写使检索 Recall +12%

## 工程落地要点
1. **小模型选择**：7B-13B 足够完成查询改写/分解任务，大幅节省大模型调用次数
2. **大模型调用控制**：只在最终生成阶段调用大模型，减少 API 成本
3. **缓存策略**：相同子问题的检索结果缓存复用
4. **错误传播**：小模型分解错误会影响大模型答案，需要对分解结果做置信度评估
5. **异步并行**：多个子问题并行检索和处理，减少总延迟

## 常见考点
**Q1: Collab-RAG 的核心洞察是什么？**
A: 任务分离：检索辅助（查询改写、问题分解）不需要最强大的 LLM，小模型微调后足以胜任；最终的复杂推理和答案生成才需要大模型。这样既利用了大模型的强推理能力，又通过小模型处理大量重复性检索辅助任务，降低成本。

**Q2: 为什么小模型需要微调，而大模型不需要？**
A: 大模型（GPT-4）通过 Few-shot Prompt 可以完成多种任务，无需微调（也无法微调）。小模型能力有限，Few-shot 效果差，需要 Fine-tune 在特定任务（查询改写/分解）上达到可用水平。微调数据用大模型蒸馏生成。

**Q3: Multi-Hop RAG 的主要失败模式？**
A: ①检索链断裂：第一跳检索的文档质量差，导致第二跳方向错误 ②信息整合失败：多个文档的信息无法在推理中正确连接 ③推理深度不足：LLM 无法进行 3+ 步推理 ④上下文过长：多跳检索的文档拼接超出 Context Window。

**Q4: 如何评估 Collab-RAG 中小模型的查询改写质量？**
A: 间接评估：改写后的 Recall@K（检索到正确文档的比例）提升多少；直接评估：人工标注改写质量（相关性/信息保留）；自动评估：用大模型（GPT-4）打分改写质量。

**Q5: 在企业私有知识库 RAG 中如何应用 Collab-RAG？**
A: ①私有小模型：在企业文档上微调小模型，学习文档结构和术语 ②本地大模型：用 Qwen/Llama 替代 GPT-4（数据隐私）③领域查询改写：小模型学习将用户模糊查询转化为专业检索词 ④分级部署：简单查询小模型直接回答，复杂查询启动 Collab 流程。
