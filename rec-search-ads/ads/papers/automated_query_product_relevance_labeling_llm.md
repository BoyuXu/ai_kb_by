# Automated Query-Product Relevance Labeling using LLMs for E-commerce Search
> 来源：arXiv:2502.15990 | 领域：ads | 学习日期：20260419

## 核心方法
1. **LLM自动化标注**：用LLM替代人工标注query-product相关性
2. **CoT Prompting**：Chain-of-Thought引导LLM逐步推理相关性判断
3. **ICL + RAG**：In-context Learning提供标注示例，RAG补充商品知识
4. **大规模标注**：降低标注成本，扩大训练数据规模

## 面试考点
- Q: LLM标注 vs 人工标注的优劣？
  - A: LLM：成本低、速度快、一致性好；人工：边界case更准、领域知识更丰富
