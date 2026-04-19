# Beyond Hard Negatives: Score Distribution in Knowledge Distillation for Dense Retrieval
> 来源：arXiv (April 2026) | 领域：search | 学习日期：20260419

## 核心方法
1. **超越硬负例**：传统KD关注hard negative mining，本文关注teacher模型的分数分布
2. **分数分布蒸馏**：让student模型学习teacher的完整分数分布（而非仅排序），保留更多信息
3. **稠密检索优化**：改进dual-encoder模型的训练效果

## 面试考点
- Q: 知识蒸馏在检索中的应用？
  - A: Cross-encoder (teacher) → Bi-encoder (student)，保持检索效率的同时逼近重排效果
