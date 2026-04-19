# Mistral-SPLADE: LLMs for Better Learned Sparse Retrieval

> arXiv:2408.11119 | 2024-08 | 领域：稀疏检索 / LLM for IR

## 一句话总结

用 Mistral-7B 作为 backbone 替代 BERT 训练 SPLADE 式稀疏检索模型，利用 LLM 更强的语言理解能力提升 keyword expansion 质量，在 BEIR benchmark 上达到 learned sparse retrieval SOTA。

## 问题背景

SPLADE 系列基于 BERT 的 MLM head 做词汇扩展，但 BERT 的语言理解能力有限：
- 见过的数据量远少于 LLM
- Encoder-only 限制了上下文理解
- Keyword expansion 质量受限于 MLM 的预训练质量

**假设**：Decoder-only LLM 见过更多数据，能更好地理解"什么词和 query 相关但未出现"。

## 核心方法

### Echo Embeddings

Decoder-only 模型有因果注意力掩码，前面的 token 看不到后面的 → 直接用末尾 token 做 pooling 会丢失前部信息。

**解决方案**：Echo Embeddings
```
Input:  "what is retrieval"
Echo:   "what is retrieval what is retrieval"
                            ^^^^^^^^^^^^^^^^^
                            只用第二份的 token embeddings
```

- 第二份的每个 token 都经过了对完整序列的注意力
- 补偿了 causal masking 的信息损失

### 训练流程

```
Mistral-7B (frozen or fine-tuned)
    ↓ Echo input
Token Embeddings
    ↓ Linear projection → vocab size
    ↓ ReLU + log(1+x) sparsification
    ↓ Max pooling over tokens
Sparse Representation (倒排索引)
```

### 训练数据

使用 sentence-transformers 数据集（常用于训练 text embedding 模型），而非传统 IR 数据集。

## 实验结果 (BEIR Benchmark)

| 模型 | BEIR Avg NDCG@10 | Backbone |
|------|-----------------|----------|
| BM25 | 0.437 | 无 |
| SPLADE++ | 0.470 | BERT |
| SPLADE v3 | 0.480 | BERT |
| **Echo-Mistral-SPLADE** | **0.497** | Mistral-7B |

**Echo-Mistral-SPLADE 超越所有已有 LSR 方法，成为 BEIR 上的 SOTA learned sparse retrieval 模型。**

## 核心洞见

### 为什么 LLM backbone 更好？

1. **更丰富的世界知识** → 更好的同义词/相关词扩展
2. **更强的上下文理解** → 更精准的歧义消解
3. **更大的词汇表** → 扩展空间更大

### 成本 vs 收益

| 维度 | SPLADE (BERT) | Mistral-SPLADE |
|------|--------------|----------------|
| 推理延迟 | 低 | 高（7B 模型） |
| 索引大小 | 相同 | 相同（都是稀疏向量） |
| 检索延迟 | 相同 | 相同（倒排索引查找） |
| 效果 | 好 | 更好 |

**关键**：LLM 只在离线编码阶段增加成本，在线检索阶段（倒排索引查找）延迟不变。

## 与其他工作的关系

- SPLADE 系列演进见 [[SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M|SPLADE v3]]
- Dense-Sparse-Late Interaction 对比见 [[检索三角_Dense_Sparse_LateInteraction|检索三角]]
- Dense 转 Sparse 见 [[SPLATE_Sparse_Late_Interaction_Retrieval|SPLATE]]
- LLM Embedding 趋势见 [[embedding_everywhere|Embedding 全景]]
