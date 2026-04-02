# Dense Passage Retrieval for Open-Domain Question Answering with Contrastive Learning
> 来源：https://arxiv.org/abs/2004.04906 | 领域：search | 日期：20260323

## 问题定义
DPR（Dense Passage Retrieval）是开放域问答（ODQA）的基础工作，提出用双塔BERT编码器学习问题和段落的稠密表示，替代传统的TF-IDF/BM25稀疏检索。

## 核心方法与创新点
- 双塔BERT：独立编码问题和段落，内积计算相关性
- 对比学习训练：正样本（相关段落）和负样本（BM25硬负样本）对比训练
- In-batch负采样：同一batch内其他问题的段落作为负样本，提升训练效率
- FAISS向量索引：构建FAISS索引支持亿级段落的高效检索

## 实验结论
DPR相比BM25，在NQ（Natural Questions）数据集Top-20准确率从59.1%提升至78.4%，提升近20%；对比学习训练比简单微调提升约5%；成为后续稠密检索工作的重要基线。

## 工程落地要点
- FAISS索引构建需要大量内存（100M段落×768d×4bytes≈300GB）
- 需要定期重新编码所有文档（index refresh），动态文档更新成本高
- In-batch负采样在大batch size时效果更好，需要大GPU内存

## 常见考点
1. **Q: DPR为什么比BM25效果好那么多？** A: DPR学习语义相关性（同义词理解），BM25只做词汇匹配；NQ等数据集语义跳跃多
2. **Q: 对比学习的正负样本构造策略？** A: 简单负样本（随机）→难负样本（BM25）→挖掘型难负样本（小模型召回的假正样本）
3. **Q: DPR的推理时间如何？** A: 查询编码约5ms，FAISS检索1M段落约1ms，总约10ms，满足实时要求
4. **Q: 如何处理DPR不擅长的精确匹配场景？** A: 混合检索（BM25+DPR融合）、SPLADE等学习稀疏方法
5. **Q: DPR到现代检索（E5/BGE/GTE）的演进？** A: 更大预训练模型、多语言、指令微调、MTEB多任务评估
