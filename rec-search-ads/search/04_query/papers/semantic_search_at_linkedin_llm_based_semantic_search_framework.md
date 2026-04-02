# Semantic Search At LinkedIn: LLM-based Semantic Search Framework
> 来源：arxiv/2401.xxxxx | 领域：search | 学习日期：20260326

## 问题定义
LinkedIn 搜索系统（职位/人才/内容）面临语义理解挑战：
- 职位搜索：用户搜索职能而非职位名称（"数据分析" vs "BI Analyst/Data Scientist"）
- 人才搜索：招聘方用技能描述搜索候选人（"熟练Python会机器学习"）
- 多语言：200+ 国家、数十种语言的跨语言搜索
- 专业术语：领英特有专业词汇（职位等级/公司名缩写）

## 核心方法与创新点
**LinkedIn Semantic Search Framework**：LLM 赋能的语义搜索体系。

**双阶段架构：**
```
Stage 1 - Semantic Retrieval（语义召回）：
  Query Encoder: LLM_finetune(query) → q_emb
  Doc Encoder:   LLM_finetune(job/profile) → d_emb
  ANN Retrieval: topK = FAISS.search(q_emb, index)

Stage 2 - Semantic Ranking（语义排序）：
  Cross-encoder: LLM_reranker(query ⊕ doc) → relevance_score
  Final Rank: weighted(retrieval_score, rerank_score, business_score)
```

**领英领域专属预训练：**
```python
# 领域数据持续预训练（Domain-Adaptive Pretraining, DAPT）
linkedin_data = [job_descriptions, profiles, articles, messages]
LLM_linkedin = continue_pretrain(LLM_base, linkedin_data)

# 任务微调（搜索相关性）
fine_tune_data = [(query, relevant_doc, score)]  # 人工标注 + 点击日志
LLM_search = fine_tune(LLM_linkedin, fine_tune_data, loss=ListwiseLoss)
```

**跨语言对齐：**
```
多语言 LLM (mBERT/XLM-R) 统一语义空间
相同含义的英文/中文/法文查询 → 相近 embedding
```

**查询改写（LLM）：**
```
"找找会Python做机器学习的人" 
→ LLM改写 → "Python Engineer, Machine Learning, Data Science"
→ 标准化查询 → 召回精度提升
```

## 实验结论
- LinkedIn 搜索系统（2023 上线）：
  - 相关结果率（Relevance Rate）+8.4%
  - 职位搜索 Apply 率（投递率）+3.2%
  - 人才搜索精准度（招聘方满意度）+5.1%
- 跨语言搜索（非英语查询）：相关性 +12%

## 工程落地要点
1. **Profile/Job 离线 Embedding**：1亿+实体，定期（每天）重新计算，批量 GPU 推理
2. **增量更新**：新发布职位/新注册用户的 embedding 实时计算（Kafka + GPU 服务）
3. **向量索引分层**：全量 IVF + 精确 HNSW（Top-1000 后 exact re-rank）
4. **业务规则融合**：语义分数 × 业务分数（职位薪资/公司规模/匹配程度）
5. **A/B 实验框架**：搜索实验需要 Interleaving（避免 novelty effect）

## 常见考点
**Q1: LinkedIn 搜索为什么特别适合语义搜索而非关键词搜索？**
A: 职业场景的词汇高度多样：同一工作有数十种叫法（"软件工程师/SWE/程序员/开发"）；技能描述方式不一致（"会Python/Python熟练/精通Python"）；跨语言查询占比大。语义搜索天然解决这些多样化表达问题。

**Q2: 领域自适应预训练（DAPT）如何提升搜索效果？**
A: 通用 LLM 不了解专业词汇（如 LinkedIn 特定的职位等级/公司文化词汇）。DAPT 在领英数据上持续预训练，使模型学习专业词汇的正确语义，提升同义词/专业词的 embedding 相似度计算准确性。

**Q3: 大规模（10亿+）实体的 embedding 索引如何维护？**
A: ①分层索引：热门实体（Top 10M）精确 HNSW，全量 IVF 粗索引 ②增量更新：新实体 online embedding + 插入 HNSW ③定期重建：embedding 模型更新后，全量批量重建索引（GPU 集群，通常隔夜完成）。

**Q4: 搜索系统中如何融合语义相关性和业务目标？**
A: 线性组合：score = α·semantic_relevance + β·business_score（点击率/转化率/时效性）。α/β 通过离线实验（NDCG 优化）或在线 A/B 学习。也可用 LambdaLoss 将业务信号作为 NDCG 权重。

**Q5: 搜索的 Interleaving 实验与 A/B 实验的区别？**
A: A/B 实验：用户分组，分别看到不同排序系统。Interleaving：同一用户同一次搜索，两个系统的结果交叉混合展示，用户点击哪个系统的结果更多则赢。Interleaving 统计效率更高（相同样本量下置信度更高），适合排序系统快速评估。
