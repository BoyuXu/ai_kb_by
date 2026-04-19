# LREF: A Novel LLM-based Relevance Framework for E-commerce Search
> 来源：arXiv:2503.09223 | 领域：ads | 学习日期：20260419

## 核心方法
1. **Rule Adherence Chain-of-Thought**：引导LLM遵循相关性规则进行推理
2. **双维度相关性**：将query-product相关性分解为product relevance（商品与query的匹配度）和modifier relevance（修饰词的满足度）
3. **可解释性**：CoT推理过程提供相关性判断的可解释依据

## 面试考点
- Q: 电商搜索相关性的粒度？
  - A: Exact（完全匹配）→ Substitute（替代品）→ Complement（互补品）→ Irrelevant（无关）
