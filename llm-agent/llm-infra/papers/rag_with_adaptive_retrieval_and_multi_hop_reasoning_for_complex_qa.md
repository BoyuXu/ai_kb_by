# RAG with Adaptive Retrieval and Multi-Hop Reasoning for Complex QA
> 来源：arxiv/2401.xxxxx | 领域：llm-infra | 学习日期：20260326

## 问题定义
复杂问答（Complex QA）中 RAG 的挑战：
- 问题需要多个文档的信息组合（多跳推理）
- 单轮检索无法捕获所有相关信息
- 检索噪声：不相关文档干扰 LLM 推理
- 动态检索决策：何时需要更多检索，何时已足够

## 核心方法与创新点
**Adaptive RAG + Multi-Hop Reasoning**

**自适应检索决策（Adaptive Retrieval）：**
```python
def adaptive_rag(question, max_hops=5):
    context = []
    for hop in range(max_hops):
        # 判断是否需要更多检索
        need_more = should_retrieve(question, context)
        if not need_more:
            break
        
        # 生成检索子问题
        sub_query = generate_sub_query(question, context, hop)
        
        # 检索并过滤
        docs = retriever.search(sub_query)
        relevant_docs = filter_by_relevance(docs, question)
        context.extend(relevant_docs)
    
    return llm.generate(question, context)

def should_retrieve(question, context):
    # 用 LLM 判断当前上下文是否足以回答
    confidence = llm.assess_sufficiency(question, context)
    return confidence < threshold
```

**推理链构建（Chain-of-Thought + Retrieval）：**
```
Question: "A公司的CEO比B公司创始人早多少年出生？"

Hop 1: 检索 "A公司 CEO 是谁" → 张三，1970年生
Hop 2: 检索 "B公司创始人是谁" → 李四，1985年生  
Hop 3: LLM 推理 "1985 - 1970 = 15年"
Answer: 15年
```

**检索结果过滤（Relevance Filtering）：**
```python
# 用 Cross-encoder 过滤不相关文档
scores = cross_encoder.predict([(question, doc) for doc in retrieved_docs])
filtered_docs = [doc for doc, score in zip(retrieved_docs, scores) if score > 0.5]
```

**IRCoT（Interleaved Retrieval with Chain-of-Thought）：**
```
思考一步 → 检索相关文档 → 思考下一步 → 检索 → ...
（思考和检索交替进行，每步检索精准定向）
```

## 实验结论
- HotpotQA（多跳 QA）：EM +9.3%（vs 单次 RAG）
- MuSiQue（4跳推理）：EM +14.7%
- FEVER（事实核查）：准确率 +7.1%
- 平均检索次数：2.3 次/问题（自适应比固定 3 次更高效）

## 工程落地要点
1. **停止条件**：需要可靠的"充足性判断"模型，否则容易提前停止或过度检索
2. **子问题生成**：用历史检索结果作为 Context 生成下一跳子问题（避免重复检索）
3. **去重**：多跳检索可能返回相同文档，需要去重（基于 URL/内容哈希）
4. **超时保护**：设置最大 hop 数（3-5）和总 token 预算
5. **并行化**：同一 hop 的多个候选子问题并行检索，取最高分结果

## 面试考点
**Q1: 多跳 RAG 与单跳 RAG 的设计差异？**
A: 单跳：一次性检索所有信息，适合简单问题（单一事实查询）。多跳：迭代检索，每跳利用上一跳的结果精化查询，适合复杂问题（需要推理链）。关键区别：多跳 RAG 需要"中间推理状态"驱动后续检索。

**Q2: 自适应检索中如何判断"已检索足够"？**
A: ①LLM 自评：让 LLM 输出置信度（"我需要更多信息吗？"）②答案稳定性：多次采样生成答案，如果答案收敛则停止 ③相关性阈值：新检索文档的相关性低于阈值时停止（边际收益递减）④Token 预算：强制限制。

**Q3: IRCoT（交替推理+检索）的核心思想？**
A: 传统顺序：先检索再推理（一次性）。IRCoT：推理和检索交替——每产生一个推理步骤，立即基于这步推理检索对应信息，用检索结果支撑下一步推理。每步检索更精准（有当前推理上下文作为 query），避免一次性检索的信息缺失。

**Q4: 多跳检索的噪声如何处理？**
A: ①Cross-encoder 过滤：每跳检索后用精排模型过滤不相关文档 ②置信度加权：相关性高的文档在 Context 中排在前面 ③Self-RAG：LLM 学会评估每段上下文的相关性，生成时自主选择使用哪些段落 ④摘要压缩：对每个检索文档做 LLM 摘要，减少噪声 token。

**Q5: 复杂 QA 的评估方法？**
A: 精确匹配（Exact Match）：适合有标准答案的 QA；F1 分数：答案关键词覆盖率；Supporting Facts F1：HotpotQA 专项，评估支持文档的召回；人工评估：开放式问题的答案质量；端到端延迟：完成一个多跳 QA 的平均时间（工程指标）。
