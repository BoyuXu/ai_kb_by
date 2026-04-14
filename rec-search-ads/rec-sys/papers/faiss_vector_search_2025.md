# FAISS: Efficient Similarity Search (2025-2026 Updates)

- **Type**: Open-Source Library
- **URL**: https://github.com/facebookresearch/faiss

## 核心技术

- **PagedAttention风格内存管理**: IVF, HNSW, CAGRA多种索引
- **NVIDIA cuVS加速**: IVF搜索延迟降低8.1x，CAGRA图索引快4.7x
- **v1.10**: 2025年5月发布，cuVS集成为核心特性

## 工业应用

Milvus/OpenSearch等向量DB底层引擎，推荐系统ANN召回核心组件。

## 面试考点

HNSW vs IVF索引对比，量化方法(PQ/SQ)，GPU加速原理。
