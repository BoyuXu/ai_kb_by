# Are LLM-Based Retrievers Worth Their Cost? Efficiency, Robustness, and Reasoning Overhead
> 来源：arXiv (April 2026) | 领域：search | 学习日期：20260419

## 核心方法
1. **LLM检索器评估**：系统评估LLM-based检索器的成本效益
2. **三维度分析**：效率（latency/throughput）、鲁棒性（OOD泛化）、推理开销（compute cost）
3. **实证研究**：对比传统检索器 vs LLM检索器在不同场景下的表现

## 面试考点
- Q: 何时使用LLM检索器？
  - A: 复杂推理型查询（multi-hop QA）优势明显；简单关键词匹配任务传统方法更高效
