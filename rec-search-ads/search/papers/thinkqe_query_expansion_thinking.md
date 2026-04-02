# ThinkQE: Query Expansion via an Evolving Thinking Process
> 来源：https://arxiv.org/abs/2506.09260 | 领域：search | 学习日期：20260401

## 问题定义

查询扩展（Query Expansion, QE）是信息检索的经典技术，通过扩展用户查询（添加同义词、相关词、上下文信息）来提升召回率。现有 LLM-based 查询扩展方法的核心问题是：

**问题1：扩展不够多样化**
- 现有方法（如 HyDE、Query2Doc）通常只生成一个扩展，缺乏对查询多种可能含义的覆盖
- 例如查询"python"可能涉及：编程语言、蟒蛇、游戏框架等多个方向

**问题2：扩展与语料库脱节**
- 静态生成的扩展词不知道语料库的实际词汇分布，可能生成语料库中根本不存在的词
- 导致"扩展后反而降低召回"的问题（term mismatch）

**问题3：单次生成局限性**
- 一次性生成扩展，无法根据初始检索结果进行迭代优化

**ThinkQE** 提出了两个关键组件：
1. **Thinking-based Expansion**：通过 LLM 的推理过程（thinking/CoT）生成深度、多样化的扩展
2. **Corpus-Interaction Strategy**：迭代式反馈机制，根据检索结果精化扩展

论文发表于 **EMNLP 2025 Findings**。

## 核心方法与创新点

### 组件一：Thinking-Based Expansion（思维链扩展）

ThinkQE 利用具备推理能力的 LLM（如 DeepSeek-R1、QwQ）的"思维过程"而非仅仅使用最终输出：

**传统 LLM 扩展**：
```
Query: "climate change effects"
→ LLM Output: "global warming, greenhouse gas, carbon emissions, sea level rise"
```

**ThinkQE 思维链扩展**：
```
Query: "climate change effects"
→ <think>
  Let me consider multiple angles:
  1. Physical effects: temperature rise, extreme weather, ice melting
  2. Biological effects: biodiversity loss, species migration
  3. Economic effects: agriculture impact, insurance costs
  4. Social effects: climate refugees, health issues
  Let me explore each angle with specific technical terms...
  </think>
→ Expanded Query: [多角度、深层次的扩展词集合]
```

**关键优势**：thinking 过程促使 LLM 系统性地探索查询的多个方向，避免了单一视角的扩展偏差。

### 组件二：Corpus-Interaction Strategy（语料库交互策略）

这是 ThinkQE 区别于其他方法的核心创新：

```
迭代流程：
Step 1: 初始思维链扩展 → 生成扩展词集合 E₀
Step 2: 用 E₀ 检索语料库 → 获得 Top-K 文档 D₀
Step 3: LLM 分析 D₀ 中的词汇/概念 → 反思与修正
Step 4: 生成精化扩展 E₁（融合语料库反馈）
Step 5: 重复 2-4 直到收敛或达到迭代上限
```

**形式化描述**：

$$
E_t = \text{LLM}\left(Q, E_{t-1}, \text{Feedback}(D_{t-1})\right)
$$

其中 $\text{Feedback}(D_{t-1})$ 是从检索文档中提取的关键词分布摘要。

### 扩展多样性的量化

ThinkQE 引入了 **语义覆盖率（Semantic Coverage）** 指标衡量扩展多样性：

$$
\text{Coverage} = \frac{|\text{unique semantic facets covered by } E|}{|\text{total facets in relevant documents}|}
$$

思维链过程显著提升了语义覆盖率，这是性能提升的关键原因。

### 与现有方法对比

| 方法 | 多样性 | 语料库交互 | 训练需求 | 性能 |
|------|--------|-----------|---------|------|
| BM25 (baseline) | - | - | 无 | 基准 |
| HyDE | 低 | 无 | 无 | +中 |
| Query2Doc | 低 | 无 | 无 | +中 |
| SPLADE | 中 | 间接 | 有 | +中高 |
| **ThinkQE** | **高** | **显式迭代** | **无** | **+高** |

## 实验结论

### 评测基准

- **TREC DL 2019/2020**：MSMARCO 段落检索（标准 dense retrieval 场景）
- **BRIGHT**：多领域推理密集型检索基准（StackExchange 问答、编程、科学等）

### 主要结果（nDCG@10）

| 方法 | DL19 | DL20 | BRIGHT avg |
|------|------|------|-----------|
| BM25 | 49.3 | 47.7 | 4.0 |
| HyDE | 56.1 | 53.2 | 12.3 |
| Query2Doc | 55.8 | 52.9 | 11.8 |
| E5-Mistral (dense) | 70.5 | 70.8 | 22.5 |
| **ThinkQE** | **72.4** | **71.3** | **26.7** |

**关键发现**：
1. 在 **BRIGHT**（推理密集型）上提升最显著（+4.2% vs E5-Mistral），验证了思维链对复杂查询的价值
2. 无需训练数据，**zero-shot** 性能超过训练过的 dense retriever
3. 迭代次数 T=2-3 时效果最佳，T>3 收益递减

### 消融实验

- 去除思维链（直接输出扩展）→ BRIGHT 降低 3.1%
- 去除语料库交互 → BRIGHT 降低 2.8%
- 两个组件均去除 → 退化为标准 LLM 扩展，BRIGHT 降低 5.6%

## 工程落地要点

### 实现建议

```python
def thinkqe_expand(query, retriever, llm, max_iter=3):
    """ThinkQE 查询扩展实现伪代码"""
    expanded = query
    for t in range(max_iter):
        # Step 1: 检索当前扩展词的结果
        docs = retriever.search(expanded, topk=10)
        corpus_feedback = extract_key_terms(docs)
        
        # Step 2: 思维链扩展
        prompt = f"""
        Original query: {query}
        Retrieved document keywords: {corpus_feedback}
        Previous expansion: {expanded}
        
        Think step by step about multiple angles and facets of this query.
        Generate comprehensive query expansion terms.
        """
        thinking, new_expanded = llm.think_and_respond(prompt)
        
        if convergence_check(new_expanded, expanded):
            break
        expanded = new_expanded
    
    return final_retrieve(query + " " + expanded)
```

### 成本-性能权衡

- **迭代次数**：每增加 1 次迭代，延迟增加约 200-500ms（取决于检索速度）
- **推荐配置**：生产环境 T=2，研究/高质量场景 T=3
- **LLM 选择**：DeepSeek-R1-7B 或 QwQ-7B 足够，无需超大模型

### 适用场景

**高价值场景**：
- 复杂长尾查询（BRIGHT-style）
- 专业领域搜索（医疗、法律、学术）
- 用户意图模糊的对话式搜索

**不适用场景**：
- 实时搜索（延迟 < 100ms 的场景，迭代代价太大）
- 简单关键词查询（BM25 已足够）

### 与向量检索的结合

```
ThinkQE + 稀疏检索 (BM25)：最直接，扩展词直接用于 BM25
ThinkQE + 密集检索：扩展后 rewrite 为语义查询，用 dense encoder 编码
ThinkQE + 混合检索：分别扩展稀疏/密集两个方向的查询
```

## 常见考点

**Q1：ThinkQE 相比 HyDE 的核心区别是什么？**
A：HyDE 生成一个假设性文档（单一方向），ThinkQE 通过思维链系统性探索多个查询角度（多样性），且通过语料库交互迭代精化（自适应），是更动态的过程。

**Q2：为什么在 BRIGHT 推理密集型基准上 ThinkQE 提升最大？**
A：BRIGHT 的查询通常需要深度理解才能找到相关文档（如"什么 Python 库能处理这种 NumPy 错误"）。思维链过程能分析查询背后的技术意图，生成精准的技术术语扩展，而传统方法只能做表面词汇扩展。

**Q3：ThinkQE 的 corpus-interaction 策略解决了什么问题？**
A：解决了"扩展词与语料库词汇不匹配"的问题。传统扩展生成语料库中不存在的词（如过于专业的术语），导致扩展无效甚至有害。语料库交互让 LLM 看到实际检索结果，调整扩展词以覆盖语料库的实际词汇分布。

**Q4：如何判断迭代是否收敛？**
A：可以用扩展词集合的 Jaccard 相似度：当相邻两轮扩展的相似度 > 0.85 时认为收敛。实践中通常 2-3 轮已收敛。
