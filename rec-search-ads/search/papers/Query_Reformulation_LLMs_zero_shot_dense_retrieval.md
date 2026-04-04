# Query Reformulation with LLMs for Zero-Shot Dense Retrieval

> 来源：arXiv 2024 | 领域：search | 学习日期：20260404

## 问题定义

稠密检索在 Zero-Shot 场景（无领域标注数据）性能大幅下降：
- 用户查询口语化、模糊，与文档语言风格差异大（Vocabulary Mismatch）
- 新领域无法微调检索模型

**查询改写（Query Reformulation）**：将用户原始查询转化为更有利于检索的形式。

$$\text{Retrieve}(\text{Reform}(q)) \succ \text{Retrieve}(q)$$

## 核心方法与创新点

1. **LLM 驱动的查询扩展**：
   - 让 LLM 基于原始查询生成多个改写版本（不同表达方式）
   - Prompt 设计：`"Rewrite this query in {N} different ways: {query}"`
   
$$\mathcal{Q}_{\text{exp}} = \{q_1, q_2, \ldots, q_N\} = \text{LLM}(q_0)$$

2. **假设文档生成（Hypothetical Document Embedding, HyDE）**：
   - LLM 生成假设性答案文档（即使不正确，语言风格更接近真实文档）
   - 用假设文档 Embedding 检索（而非用查询 Embedding）

$$e_{\text{query}} \leftarrow e_{\text{LLM}(\text{"Answer: " + q})}$$

3. **多路召回融合（Multi-Query Fusion）**：
   - 对 N 个改写查询分别检索
   - 用 Reciprocal Rank Fusion（RRF）合并排名
   
$$\text{RRF}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{rank}_i(d)}, \quad k=60$$

4. **自适应改写策略**：
   - 短查询（<3 词）：扩展改写（加上下文）
   - 长查询（>10 词）：压缩改写（提取核心意图）
   - LLM 判断查询类型并选择策略

## 实验结论

- BEIR Zero-Shot NDCG@10: **+5.8%** vs 无改写 Dense Retrieval
- HyDE: 在科学文献检索中效果最佳（**+9.2%**）
- 多路召回 RRF: 比单路最佳改写高 **+2.3%**
- 查询扩展对长尾 / 专业领域提升最大（+12%）

## 工程落地要点

- LLM 改写延迟：GPT-3.5 ~200ms，建议缓存高频查询的改写结果
- 改写数 N=3-5（收益递减，超过 5 无显著提升）
- HyDE 生成长度控制 ≤ 100 tokens（防止引入噪声）
- RRF k=60 是经验最优值，无需调参

## 面试考点

1. **Q**: Vocabulary Mismatch 问题是什么？Dense Retrieval 如何解决？  
   **A**: 用户用口语（"手机壳子"），文档用专业词（"保护套"），BM25 完全匹配失败。Dense Retrieval 用语义向量相似度，在语义空间对齐，不受词汇差异影响。

2. **Q**: HyDE（假设文档）为什么有效？  
   **A**: 查询 Embedding 和文档 Embedding 来自不同分布（问题 vs 陈述句）。LLM 生成的假设文档 Embedding 与真实文档分布更近，检索精度更高。

3. **Q**: RRF（Reciprocal Rank Fusion）如何合并多路检索结果？  
   **A**: 每个文档的分数 = 各路排名的倒数之和（$\sum 1/(k+\text{rank})$），高的文档被多路召回时排名高，即使单路排名不高也能被合并提升。
