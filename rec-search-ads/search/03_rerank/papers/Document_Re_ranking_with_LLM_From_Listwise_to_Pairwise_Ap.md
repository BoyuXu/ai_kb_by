# Document Re-ranking with LLM: From Listwise to Pairwise Approaches
> 来源：https://arxiv.org/search/?query=document+reranking+LLM+listwise+pairwise&searchtype=all | 领域：search | 日期：20260323

## 问题定义
比较LLM在文档重排序（reranking）中的不同范式：逐点（pointwise）、成对（pairwise）和列表级（listwise），分析各自优缺点并提供最佳实践。

## 核心方法与创新点
- Listwise排序：LLM直接输出文档排列顺序（如SetRank、RankGPT）
- Pairwise比较：LLM比较两文档的相对相关性，用锦标赛制汇总
- Pointwise评分：LLM为每文档独立打分，相对排序不稳定
- 混合策略：先listwise粗排，再pairwise精排

## 实验结论
Listwise在TREC DL benchmark上NDCG@10最高，比pointwise高约4%；但listwise对prompt中文档顺序敏感（position bias）；pairwise更稳定但O(N²)开销；混合方案兼顾效果和效率。

## 工程落地要点
- Listwise最多处理约20个文档（受context length限制）
- Position bias是关键问题，需要随机shuffle文档顺序或多次采样取均值
- 工业场景通常：稀疏/稠密召回 → bi-encoder rerank → LLM精排（Top50→Top10）

## 常见考点
1. **Q: LLM重排序的position bias问题？** A: LLM倾向于认为prompt开头/结尾的文档更相关，与实际内容无关
2. **Q: RankGPT的listwise重排序如何工作？** A: 向LLM提供文档列表，要求直接输出相关性排序，用sliding window处理长列表
3. **Q: Pairwise重排序的O(N²)开销如何优化？** A: 冒泡排序近似（只比较相邻）、锦标赛排序（O(N log N)比较次数）
4. **Q: 搜索重排序的常用特征？** A: 查询-文档相关性分、BM25分、dense分、文档质量分、点击历史
5. **Q: LLM重排序在工业搜索中的实用价值？** A: 提升精排质量，但成本高；通常用于高价值查询或结合蒸馏降成本
