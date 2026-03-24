# Dense Retrieval vs Sparse Retrieval: A Unified Evaluation Framework for Large-Scale Product Search
> 来源：https://arxiv.org/search/?query=dense+sparse+retrieval+unified+product+search&searchtype=all | 领域：search | 日期：20260323

## 问题定义
在电商商品搜索场景，对稠密检索（dense retrieval，如DPR/bi-encoder）和稀疏检索（sparse retrieval，如BM25/SPLADE）进行系统化评估，提供统一的评估框架。

## 核心方法与创新点
- 统一评估框架：在相同benchmark上公平对比稠密和稀疏方法
- 电商专项评估维度：完全匹配（exact match）、语义相关性、多样性
- 混合方法评估：linear combination vs learned fusion vs cascade
- 工业规模测试：亿级商品库的检索效率和质量评估

## 实验结论
电商搜索场景：稀疏检索在精确品牌/型号搜索（exact match）上准确率高约15%；稠密检索在语义理解（同义词/意图识别）上优约20%；混合方法在综合指标上最优，提升约10%。

## 工程落地要点
- 稀疏检索（倒排索引）工程成熟度高，低延迟（<5ms），是默认选择
- 稠密检索需要ANN索引（HNSW/IVF），随物品量增大延迟增长
- 混合检索建议：稀疏召回TopK，稠密rerank，兼顾精度和延迟

## 面试考点
1. **Q: BM25的核心公式和参数含义？** A: TF-IDF变体，k1控制词频饱和，b控制文档长度归一化，简单高效
2. **Q: Dense retrieval为何需要ANN而不是精确搜索？** A: 精确搜索O(N×d)，N=亿级时不可接受；ANN牺牲少量精度换取O(log N)
3. **Q: 电商搜索的品牌词精确匹配为何稀疏方法更好？** A: "Nike Air Max 2024"是精确字面匹配，稠密模型可能把"Adidas"也检索出来
4. **Q: 如何评估搜索系统的检索质量？** A: Recall@K、NDCG@K、MRR（Mean Reciprocal Rank）、人工相关性标注
5. **Q: 电商搜索的查询改写（Query Rewriting）作用？** A: 纠正拼写错误、扩展同义词、分解复合查询，提升召回率
