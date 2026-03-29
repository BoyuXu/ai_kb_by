# 稠密检索 DPR：原理、训练与工程实践

> 来源：技术综述（Facebook AI Research, 2020） | 日期：20260316 | 领域：search

## 问题定义

传统稀疏检索（BM25/TF-IDF）基于词频统计，无法处理：
- **词汇不匹配**：query "automobile" vs document "car"，语义相同但词不同。
- **语义理解不足**："best restaurant nearby"，"nearby"的空间语义 BM25 无法理解。

DPR（Dense Passage Retrieval）用预训练语言模型（BERT）学习 query 和 passage 的稠密向量表示，通过向量相似度检索，克服词汇鸿沟。

## 核心方法与创新点

### 双塔架构（Bi-encoder）

```python
# Query Encoder
q_embedding = BERT_Q([CLS] query [SEP])[:, 0, :]  # [CLS] token

# Passage Encoder  
p_embedding = BERT_P([CLS] passage [SEP])[:, 0, :]

# Similarity
score(q, p) = q_embedding · p_embedding  # 内积
```

两个独立的 BERT（或共享参数），query 和 passage 分别编码，**避免了 cross-encoder 的实时计算代价**。

### 训练方法

**对比学习训练目标（In-batch Negative）**：
```
L = -log( exp(sim(q, p+)) / (exp(sim(q, p+)) + Σ exp(sim(q, p-))) )
```
- 每个 batch 中，当前 query 的正例 passage + batch 内其他 query 的正例作为负例（**In-batch negatives**）。
- 大 batch size（512-2048）提供大量免费负样本，训练效率高。

**难负例（Hard Negatives）**：
- BM25 检索到的但不相关的 passage（看起来相关但实际不是，对模型挑战大）。
- DPR 训练集 = 正例 + 随机负例 + BM25 难负例（三类混合）。

### 索引与检索

1. 离线：用 Passage Encoder 将所有 passage 编码为向量，用 FAISS 建立 ANN 索引。
2. 在线：用 Query Encoder 编码 query，FAISS 检索 Top-K 最近邻 passage（毫秒级）。

## 实验结论

- Natural Questions (NQ)：DPR 相比 BM25：Top-20 recall 79.4% vs 59.1%（+20% 绝对提升）。
- TriviaQA：Top-20 recall 79.4% vs 66.9%。
- 端到端 QA（DPR + FiD）：NQ exact match 51.4%，当时 SOTA。

## 工程落地要点

- FAISS 索引：文档 <1M 用 Flat（精确但慢）；1M-100M 用 IVF-HNSW（balance）；>100M 用 IVF-PQ（压缩内存）。
- Embedding 维度：DPR 用 768（BERT-base），可以用 PCA/归一化降维到 256 减少内存。
- 增量更新：新文档只需编码并添加到 FAISS 索引，无需重训练（Bi-encoder 的最大优势）。
- 多语言：替换为 mBERT/XLM-R 作为 backbone，支持跨语言检索。

## 面试考点

- Q: DPR 和 BM25 应该用哪个？什么时候混合？
  A: 有标注数据（query-passage 对）时 DPR 更强；零样本或跨领域时 BM25 更稳。工业界最佳实践是 Hybrid 检索：BM25（词汇匹配）+ DPR（语义匹配），用 RRF（Reciprocal Rank Fusion）合并结果。

- Q: In-batch Negative 的局限是什么？
  A: Batch 内的负样本是随机的，大部分是简单负样本（和 query 明显不相关），难以训练模型区分边界困难案例。解决：加入 Hard Negatives（BM25 难例 / 模型预测错误的案例），提升模型判别能力。

- Q: FAISS 的 HNSW 和 IVF 有什么区别？
  A: HNSW（Hierarchical Navigable Small World）是图结构索引，查询精度高，支持实时添加新向量，但内存占用大（每个向量约 200-500 字节）。IVF（Inverted File）将向量聚类，查询时只搜索最近的几个簇，内存效率高，但需要预先训练聚类（不支持增量添加）。IVF-PQ 进一步压缩，内存最小但精度有损。
