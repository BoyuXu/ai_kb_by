# Dense Retrieval vs Sparse Retrieval: Unified Evaluation Framework for Large-Scale Product Search

> 来源：arxiv/工业界 | 日期：20260322 | 领域：搜索系统

## 问题定义

Dense Retrieval（向量检索，如 DPR/bi-encoder）和 Sparse Retrieval（词袋检索，如 BM25/SPLADE）各有优劣，但缺乏在电商大规模产品搜索场景下的系统性对比和统一评估框架。工业界决策（用哪种检索）往往依赖经验而非实证。

## 核心方法与创新点

- **统一评估框架**：
  - 覆盖 5 个维度：相关性（Relevance）、效率（Latency/QPS）、可扩展性（Scale）、鲁棒性（OOD/长尾 query）、可解释性
  - 使用真实电商数据集（2 亿商品，1000 万 query-click 对）
- **系统性对比实验**：
  - Dense：DPR、E5、BGE、ColBERT（late interaction）
  - Sparse：BM25、SPLADE-v2、UniCOIL
  - 混合：RRF 融合、Learned Sparse + Dense
- **场景分类**：按 query 类型分析（精确品牌查询、品类探索、属性组合查询）各自最优方法
- **工业规模适配**：百亿级商品的 ANN 检索延迟、内存开销对比

## 实验结论

- **精确查询**（"耐克 Air Max 90"）：Sparse 胜（BM25: NDCG 0.82 vs Dense: 0.74）
- **语义查询**（"适合婚礼的礼服"）：Dense 胜（Dense: NDCG 0.79 vs BM25: 0.61）
- **长尾低频 query**：混合检索最优（+8% vs best single）
- **延迟**：BM25 <5ms；Dense ANN 10-30ms；ColBERT 50-200ms（不适合超大规模）
- **结论**：工业场景建议混合检索（Sparse 做粗召回 + Dense 做精召回）

## 工程落地要点

- **混合检索最优配比**：Sparse 召回 70% 流量 + Dense 召回 30%，通过 RRF 融合
- **Dense 索引更新**：商品新增时需实时更新向量索引（推荐 HNSW，支持增量插入）
- **Sparse 更新**：重建倒排索引，速度更快（分钟级 vs Dense 小时级）
- **成本**：Dense 检索内存开销约为 Sparse 的 5-10×（向量维度 768 vs 稀疏向量）

## 常见考点

1. **Q：Dense 和 Sparse Retrieval 各自的适用场景？**
   A：Sparse（BM25）适合精确匹配（有品牌/型号的查询）；Dense 适合语义相似匹配（描述性、模糊查询）；混合最鲁棒

2. **Q：如何在工业场景中选择 ANN 算法？**
   A：HNSW（高召回率，支持增量，推荐）；IVF-PQ（内存效率高，适合超大规模）；ScaNN（Google，工程优化最好）

3. **Q：ColBERT 为什么在超大规模产品搜索中不适用？**
   A：ColBERT 的 late interaction 需要对每个 query token 和 doc token 做细粒度交互，计算复杂度 O(|q| × |d| × N)，N 为商品数，百亿级商品下延迟不可接受
