# 知识卡片 #004：Query Rewriting & RAG 优化

> 📚 参考文献
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Document Re-Ranking With Llm From Listwise To Pair](../../search/03_rerank/papers/Document_Re_ranking_with_LLM_From_Listwise_to_Pairwise_Ap.md) — Document Re-ranking with LLM: From Listwise to Pairwise A...
> - [Dense Passage Retrieval For Open-Domain Questio...](../../search/01_recall/papers/Dense_Passage_Retrieval_for_Open_Domain_Question_Answerin.md) — Dense Passage Retrieval for Open-Domain Question Answerin...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-Unified-Eva...](../../search/01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dpr-Dense-Retrieval](../../search/01_recall/papers/dpr_dense_retrieval.md) — 稠密检索 DPR：原理、训练与工程实践
> - [W-Rag-Weakly-Supervised-Dense-Retrieval](../../search/01_recall/papers/W_RAG_Weakly_Supervised_Dense_Retrieval_in_RAG_for_Open_D.md) — W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-...
> - [Dense Vs Sparse Retrieval Eval](../../search/01_recall/papers/Dense_Retrieval_vs_Sparse_Retrieval_Unified_Evaluation_Fr.md) — Dense Retrieval vs Sparse Retrieval: Unified Evaluation F...

> 创建：2026-03-20 | 领域：搜索·RAG | 难度：⭐⭐⭐

## 📐 核心公式与原理

### 1. 推荐系统漏斗

$$
\text{全量} \xrightarrow{召回} 10^3 \xrightarrow{粗排} 10^2 \xrightarrow{精排} 10^1 \xrightarrow{重排} \text{展示}
$$

- 逐层过滤，平衡效果和效率

### 2. CTR 预估

$$
pCTR = \sigma(f_{DNN}(x_{user}, x_{item}, x_{context}))
$$

- 排序核心：预估用户点击概率

### 3. 在线评估

$$
\Delta metric = \bar{X}_{treatment} - \bar{X}_{control}
$$

- A/B 测试量化策略效果

---

## 🌟 一句话解释

用户原始查询往往不是检索最优形式，**Query Rewriting 通过改写/扩展/分解查询来提升检索召回质量**；RL 版本（RewriteGen）能自适应学习最优改写策略，无需人工标注。

---

## 🎭 生活类比

去图书馆查资料：
- **原始查询**：你问馆员"有没有关于那个很出名的苹果公司创始人的书"（表达模糊）
- **Query Expansion**：馆员帮你加关键词："史蒂夫·乔布斯 苹果公司 传记 创业"（扩展同义词）
- **Sub-query Decomposition**：把一个复杂问题拆成3个小问题分别去查，再合并答案
- **HyDE**：先请专家写一段"假设的标准答案"，再用这段文字去找相似文献（以答找答）
- **RL 改写**：馆员通过多次试错，学会了哪种改写最能找到你真正需要的书

---

## ⚙️ 核心机制

### 方法一：HyDE（Hypothetical Document Embeddings）
```
用户查询 Q
    │
    ▼
LLM 生成假设答案 H（不需要正确，只需要形式相似）
    │
    ▼
Embedding(H) → 向量检索 → 相关文档
    │
    ▼
相关文档 + Q → LLM 生成最终答案
```

### 方法二：Sub-query Decomposition
```
复杂问题 Q
    │
    ▼
LLM 分解为 [Q1, Q2, Q3]
    │
    ├─▶ Retrieve(Q1) → Docs1
    ├─▶ Retrieve(Q2) → Docs2
    └─▶ Retrieve(Q3) → Docs3
              │
              ▼
         合并文档 → LLM 综合回答
```

### 方法三：RL 改写（RewriteGen）
```
状态 S = (原始查询 + 检索历史 + 当前答案质量)
    │
    ▼
策略网络（LLM）→ 生成改写查询 A
    │
    ▼
执行检索 → 获取文档 → LLM 生成答案
    │
    ▼
奖励 R = 答案EM/F1 - 检索成本惩罚
    │
    ▼
PPO 更新策略网络（支持多轮迭代改写）
```

---

## 🔄 横向对比

| 方法 | 核心思路 | 优势 | 劣势 |
|------|---------|------|------|
| Query Expansion | 加同义词/相关词 | 简单快 | 可能引入噪声 |
| HyDE | 以答找答 | 对抽象问题效果好 | LLM 生成成本高 |
| Step-Back | 问题抽象化 | 通用知识检索佳 | 细节问题可能丢失 |
| Sub-query | 问题分解 | 复杂推理题效果佳 | 延迟高（多轮检索）|
| RL 改写 | 自动学习策略 | 泛化好，无需标注 | 训练成本高 |

---

## 🏭 工业落地

- **搜索引擎**：Google/百度的查询扩展（QE）是排序核心模块
- **企业知识库 RAG**：LangChain/LlamaIndex 内置多种改写策略
- **对话系统**：History-aware Query Rewriting，把上下文融入查询
- **电商搜索**：同义词典 + 品牌词标准化（"苹果手机" → "iPhone"）

**延迟优化策略：**
1. 查询缓存（Query Cache）：相似查询直接返回
2. 意图分类：判断是否需要检索（事实型才检索）
3. Context Compression：LLM 压缩检索到的长文档，只保留相关片段
4. 并行检索：多个子查询同时检索，异步合并

---

## 🎯 常见考点

**Q1（基础）：RAG 系统中 Query Rewriting 能解决什么问题？**
> 原始查询常见问题：①过短缺乏上下文（"苹果股价"）；②歧义性（"美国总统"）；③口语化（"那个很火的大模型框架"）；④多跳推理（"OpenAI 创始人毕业的大学的城市在哪"）。改写后的查询与知识库的语义分布更接近，检索 Recall 更高。

**Q2（中等）：HyDE 的核心假设是什么？为什么有效？**
> 核心假设：语义相似的文本在 Embedding 空间中距离近，而"假设答案"与真实相关文档在 Embedding 空间比原始查询更接近（因为假设答案和目标文档属于同一语义类型）。适合开放域 QA，但在事实精确性要求高的场景（如法律、医疗）需谨慎，因为 LLM 生成的假设答案可能包含幻觉。

**Q3（高难）：在 RL 改写框架中，如何设计奖励函数避免"奖励黑客"问题？**
> 纯用最终答案正确性作为奖励会导致：①改写得越离谱越可能碰巧答对（奖励 hack）；②训练信号稀疏（大多数改写结果为0奖励）。解决：①稠密奖励：检索到相关文档给中间奖励；②惩罚项：查询过长/检索轮数过多减分；③多信号融合：答案质量+检索相关性+查询自然度；④人类偏好对齐（RLHF）校准奖励。

---

## 🔗 知识关联

- 上游：意图理解（NLU）、Query 分析
- 同层：Dense Retrieval（DPR/ColBERT）、BM25 混合检索
- 下游：Reranking（Cross-encoder）、Context Compression、LLM 生成

### Q1: 面试项目介绍的 STAR 框架？
**30秒答案**：Situation（背景）→Task（任务）→Action（方案）→Result（结果）。关键：量化结果（AUC +0.5%, 线上 CTR +2%），突出个人贡献，准备 follow-up 追问。

### Q2: 算法面试如何展现系统性思维？
**30秒答案**：①先说全局架构再说细节；②主动分析 trade-off；③提及工程约束（延迟/资源）；④讨论 A/B 测试验证；⑤对比多种方案优劣。

### Q3: 面试中遇到不会的问题怎么办？
**30秒答案**：①诚实说不了解具体细节；②从已知相关知识推导思路；③说明学习路径（"我会从 XX 论文入手了解"）。比胡编强 100 倍。

### Q4: 简历中项目经历怎么写？
**30秒答案**：①每个项目 3-5 行；②突出方法创新点和业务效果；③用数字量化（AUC/CTR/时长提升 X%）；④技术关键词匹配 JD；⑤按相关度排序而非时间顺序。

### Q5: 如何准备系统设计面试？
**30秒答案**：①准备推荐/搜索/广告各一个完整系统设计；②每个系统能说清召回→排序→重排全链路；③准备 scalability 方案（如何从百万到亿级）；④准备 failure mode 和降级方案。

### Q6: 八股文和实际项目经验如何结合？
**30秒答案**：八股文提供理论框架，项目经验证明落地能力。面试时：先用八股文回答「是什么/为什么」，再用项目经验回答「怎么做/效果如何」。纯八股文没有竞争力。

### Q7: 面试中如何展示 leadership？
**30秒答案**：①描述自己在项目中的角色和贡献；②说明如何推动跨团队协作；③展示主动发现问题并推动解决的案例；④分享技术方案选型的决策过程。

### Q8: 被问到不会的论文怎么办？
**30秒答案**：①说清楚自己了解的相关工作；②从论文标题推断可能的方法（如 xxx for recommendation 可能是把 xxx 技术迁移到推荐）；③承认不了解但表达学习意愿。

### Q9: 算法岗面试的常见流程？
**30秒答案**：①简历筛选→②一面（算法基础+项目）→③二面（系统设计+深度追问）→④三面（部门 leader，考察思维+潜力）→⑤HR 面→Offer。每轮约 45-60 分钟。

### Q10: 如何准备不同公司的面试？
**30秒答案**：①字节：重工程实现+大规模系统+实际效果；②阿里：重业务理解+电商场景+系统设计；③腾讯：重算法深度+创新性+论文理解；④快手/小红书：重内容推荐+短视频场景+多模态。

## 📐 核心公式直观理解

### Query Expansion 的相关反馈

$$
q' = \alpha \cdot q + \beta \cdot \frac{1}{|D_r|}\sum_{d \in D_r} d - \gamma \cdot \frac{1}{|D_n|}\sum_{d \in D_n} d
$$

- $D_r$：相关文档集
- $D_n$：不相关文档集

**直观理解**：Rocchio 算法让 query 向"好结果"靠近、远离"坏结果"。如果搜索"苹果"返回了水果和手机的结果，用户点了手机相关的——下一轮 query 就会自动偏向"苹果手机"的语义方向。LLM 时代用 prompt 替代 Rocchio，但思想一致。

### HyDE（Hypothetical Document Embedding）

$$
\text{score}(q, d) = \cos(\text{Enc}(\text{LLM}(q)), \text{Enc}(d))
$$

**直观理解**：先让 LLM 根据 query 生成一篇"假文档"（可能包含幻觉），再用这篇假文档的 embedding 去检索真文档。妙处在于假文档和真相关文档在 embedding 空间更近（都是"答案风格的文本"），比 query（问题风格）直接检索效果好。

### Query 理解的多任务框架

$$
\mathcal{L} = \mathcal{L}_{\text{intent}} + \lambda_1 \mathcal{L}_{\text{NER}} + \lambda_2 \mathcal{L}_{\text{rewrite}}
$$

**直观理解**：一个查询同时需要理解意图（导航/信息/交易）、抽取实体（品牌/型号/地点）、改写为标准形式。多任务学习让底层表示共享三种理解能力的知识，比三个独立模型更高效也更准。

