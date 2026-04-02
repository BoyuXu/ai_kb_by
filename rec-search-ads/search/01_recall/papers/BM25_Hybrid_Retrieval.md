# BM25 与语义检索融合：Hybrid Retrieval 最佳实践

> 来源：技术综述 | 日期：20260316 | 领域：search

## 问题定义

稀疏检索（BM25）和稠密检索（DPR/E5）各有优劣：
- BM25 精确词匹配，擅长专有名词、编程代码、精确短语匹配，零样本稳定。
- 稠密检索语义理解强，但词汇鸿沟问题已解决，需要训练数据。

**在工业系统中，单一方法都无法覆盖全部检索需求**，Hybrid Retrieval 成为最佳实践。

## 核心方法与创新点

### 1. RRF（Reciprocal Rank Fusion）

最简单有效的融合方法，无需训练：

```python
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

def hybrid_search(query, bm25_results, dense_results, k=60):
    scores = defaultdict(float)
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] += rrf_score(rank + 1, k)
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] += rrf_score(rank + 1, k)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

RRF 的优点：不需要归一化两路检索的分数尺度（BM25 分数和向量余弦值量纲不同）。

### 2. 加权线性融合

需要归一化后加权：
```python
score = α × norm(bm25_score) + (1-α) × norm(dense_score)
```

α 通常在 0.3-0.7 之间，需要在验证集上调参。归一化方法：Min-Max 或 Z-score。

### 3. 级联（Cascade）方式

```
BM25 召回 Top-K1 → Dense Rerank → Top-K2
Dense 召回 Top-K1 → BM25 Filter → Top-K2
```

节省计算：只对 BM25 召回的结果做昂贵的语义计算。

### 4. 稀疏-稠密统一模型（SPLADE）

**SPLADE（SParse Lexical AnD Expansion）**：
- 在 BERT 词表维度上输出 sparse 激活（大多数词维度为0）。
- 同时实现词汇匹配（稀疏）和语义理解（扩展）。
- 可以直接用倒排索引存储，兼顾效率和效果。

SPLADE 分数计算：
```
w_t = max(0, BERT_output_t)  # ReLU 稀疏化
doc_vector = log(1 + w_t)  # 对数压缩
score(q, d) = q_vector · d_vector  # 稀疏内积（高效）
```

### 5. BGE-M3：多功能统一检索

- **Dense**：标准 embedding 检索。
- **Sparse**：SPLADE 风格的词汇权重检索。
- **Multi-Vector（ColBERT 风格）**：每个 token 一个 embedding，MaxSim 算法。
- 三路融合，一个模型解决所有场景。

## 实验结论

- BEIR benchmark（零样本跨领域）：Hybrid（BM25+DPR，RRF）相比单纯 DPR：nDCG@10 +3-5%。
- SPLADE 在 MS-MARCO：MRR@10 比 BM25 +15%，接近 DPR 但推理速度快 3-5x（稀疏索引）。
- BGE-M3 在 MTEB benchmark：多语言场景全面领先，特别是低资源语言 +10-20%。

## 工程落地要点

- Elasticsearch 8.x 支持 Hybrid Search（BM25 + kNN vector）内置 RRF，无需额外开发。
- Qdrant/Weaviate 都支持 Hybrid 搜索模式，开箱即用。
- **k 参数调优**：RRF 中 k=60 是经验默认值，对大多数场景有效；如果一路检索质量显著高于另一路，可以减小 k（增大高质量结果的相对权重）。
- 性能监控：分别记录 BM25 和 dense 的召回率，判断哪路检索是瓶颈。

## 常见考点

- Q: 为什么 RRF 融合不需要分数归一化？
  A: RRF 只使用排名（rank），不使用原始分数，所以不同量纲的分数（BM25 是概率估计，余弦相似度是[-1,1]）不需要对齐。排名本身是无量纲的，1/(k+rank) 公式保证了两路结果的贡献对等。

- Q: 什么场景下 BM25 仍然比稠密检索强？
  A: (1) 精确短语查询（产品型号、代码片段）；(2) 领域专有词汇（医学术语、法律术语），稠密模型没见过；(3) 多语言低资源语言；(4) 长文档检索（BM25 的 IDF 对长文档归一化效果好）。

- Q: ColBERT 的 MaxSim 是什么？
  A: ColBERT 为 query 和 doc 的每个 token 分别生成 embedding（不是整体一个向量），相似度计算：score(q,d) = Σ_{t∈q} max_{t'∈d} (t·t')，即 query 每个 token 与 doc 中最相似 token 的得分之和。细粒度匹配，精度高，但存储开销大（每个 token 一个向量）。
