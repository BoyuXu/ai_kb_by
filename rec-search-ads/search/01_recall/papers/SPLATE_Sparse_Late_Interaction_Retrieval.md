# SPLATE: Sparse Late Interaction Retrieval

> arXiv:2404.13950 | SIGIR 2024 | 领域：稀疏检索 / Late Interaction

## 一句话总结

SPLATE 在冻结的 ColBERTv2 token embedding 之上训练一个轻量 MLM adapter，将 dense late interaction 表示映射到 SPLADE 式的稀疏词汇空间，实现用传统倒排索引加速 ColBERT 的候选生成。

## 问题背景

ColBERT (Late Interaction) 精度高但候选生成阶段仍需专用引擎（如 PLAID），依赖 GPU + 特殊索引结构。能否用传统的稀疏检索（倒排索引）替代这一步？

## 核心方法

### 架构

```
Input Text
    ↓
ColBERTv2 Encoder (frozen)
    ↓ token embeddings (dense, 128d each)
MLM Adapter (trainable)
    ├── 2-layer MLP + residual connection
    └── Modified MLM Head
    ↓ sparse vocabulary-sized vectors
SPLADE-style max pooling over tokens
    ↓
Sparse document representation (倒排索引友好)
```

### 关键设计

1. **冻结 ColBERTv2**：不改动原始 dense 表示，保证兼容性
2. **轻量 adapter**：仅训练 MLM head + 2 层 MLP（参数量极少）
3. **Residual connection**：保留原始 token 信息
4. **SPLADE max pooling**：每个词汇维度取所有 token 的最大激活值

$$\mathbf{s}_j = \max_{i=1}^{N} \text{ReLU}(\text{MLM}(\mathbf{h}_i))_j$$

其中 $\mathbf{h}_i$ 是第 $i$ 个 token 的 ColBERTv2 embedding，$j$ 索引词汇表维度。

### 两阶段 Pipeline

```
Stage 1: SPLATE sparse retrieval (倒排索引, <10ms)
    → Top-50 candidates
Stage 2: ColBERTv2 MaxSim reranking (原始 dense 表示)
    → Final Top-K
```

## 实验结果 (MS MARCO)

| 方法 | MRR@10 | 延迟 |
|------|--------|------|
| PLAID + ColBERTv2 | 0.397 | ~50ms (GPU) |
| SPLATE + ColBERTv2 rerank | 0.397 | <10ms (CPU) + rerank |
| 纯 SPLADE | 0.380 | <10ms (CPU) |

**关键结论**：SPLATE 候选生成 + ColBERTv2 重排 ≈ PLAID 全流程效果，但延迟更低且可在 CPU 上运行。

## 核心价值

- **Dense → Sparse 桥梁**：把 ColBERT 的 dense 能力"蒸馏"到稀疏空间
- **CPU 友好**：候选生成不需要 GPU
- **兼容现有基础设施**：可以用 Lucene/Elasticsearch 的倒排索引
- **即插即用**：冻结 backbone，只训练 adapter

## 与其他工作的关系

- 检索范式演进见 [[检索三角_Dense_Sparse_LateInteraction|检索三角]]
- SPLADE 系列见 [[SPLADE_v3_Advancing_Sparse_Retrieval_with_Deep_Language_M|SPLADE v3]]
- ColBERT 系列见 [[ColBERT_v2_Effective_and_Efficient_Retrieval_via_Lightwei|ColBERTv2]]
- LLM backbone 做稀疏检索见 [[Mistral_SPLADE_LLMs_Better_Learned_Sparse_Retrieval|Mistral-SPLADE]]
