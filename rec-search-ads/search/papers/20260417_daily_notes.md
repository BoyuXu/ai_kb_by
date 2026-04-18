# 搜索系统论文笔记 — 2026-04-17

## 1. Dense Passage Retrieval (DPR) with Contrastive Learning

**来源：** https://arxiv.org/abs/2004.04906 (Facebook AI)
**领域：** 稠密检索 × 对比学习
**核心定位：** 稠密向量检索的奠基性工作，双编码器框架取代稀疏词匹配

**核心贡献：**
- 双编码器架构学习 passage 稠密向量表征
- 对比学习最大化 query-passage 相关对相似度，最小化无关对
- Hard negative mining (ANCE) 异步更新 ANN 索引

**关键公式：**
- 对比损失: L = -log[exp(sim(q,p+)/τ) / Σ exp(sim(q,p-)/τ)]
- sim 使用点积，τ 为温度参数

**工业影响：** 现代稠密检索的基础；可扩展到十亿级文档集合

**面试考点：** 双编码器 vs 交叉编码器的效率-精度 trade-off、hard negative mining 原理、对比学习构建有意义嵌入空间

---

## 2. ColBERT & GTE-ModernColBERT-v1: Token-Level Multi-Vector Retrieval

**来源：** LightOn AI / Stanford
**领域：** 多向量检索 × 长文档
**核心创新：** Token 级多向量表征替代单向量聚合

**关键技术：**
- 每个 token 保留独立向量（而非聚合为单一向量）
- **MaxSim** 相似度函数：token-to-token 匹配
- ModernBERT backbone + 128 维 token embeddings
- 支持最长 8192 tokens 输入

**关键指标：** LongEmbed benchmark 88.39 分 (vs 竞品 78.82)

**面试考点：** token 级表征为何优于单向量、MaxSim late-interaction 的排序精度优势、长文档检索的可扩展性

---

## 3. Learning to Rank: Listwise Loss in Industrial Search

**领域：** 学习排序 × 工业搜索
**核心定位：** Listwise LTR 直接优化整个排序列表

**核心方法：**
- Listwise loss 建模完整排序分布（vs pointwise/pairwise）
- **LambdaMART：** 梯度提升树 + LambdaRank 梯度近似
- 直接融入 NDCG 等排序指标到训练目标

**工业影响：** 
- 主流搜索引擎有机结果的生产级排序
- 对标注噪声和部分相关性更鲁棒
- A/B 测试中持续优于 pointwise/pairwise

**面试考点：** pointwise vs pairwise vs listwise 三种方法对比、LambdaMART 成功的工程原因、listwise 对噪声的鲁棒性

---

## 4. Semantic Denoising for Cross-Lingual Search Ranking

**领域：** 跨语言检索 × 语义去噪
**核心问题：** 跨语言检索中翻译瓶颈和对齐噪声

**核心方法：**
- **Denoising Word Alignment (DWA)：** 平行句子预训练
- 多语言稠密检索器专为 CLIR 训练
- 词、短语、文档多层级对比学习
- Cross-encoder re-ranking

**工业影响：** 消除昂贵的文档翻译需求；降低延迟，提升多语言搜索相关性

**面试考点：** DWA 如何处理跨语言对齐噪声、语言特定 vs 通用多语言检索器、消除翻译瓶颈的生产优势

---

## 5. JaColBERT v2.5: Optimising Multi-Vector Retrievers

**来源：** https://arxiv.org/abs/2407.20750
**领域：** 多向量检索 × 日语 × 低资源优化
**核心贡献：** 在资源受限条件下优化 ColBERT 架构

**关键创新：**
- **Checkpoint Merging：** 融合微调和预训练模型的优势
- 系统性推理/训练设置优化
- 仅 110M 参数，4×A100 训练 <15 小时

**关键指标：**
- 平均分 0.754（较 v2 提升 4.5%）
- 较最佳日语单语模型 GLuCoSE 提升 60%
- 较多语言基线 BGE-M3 提升 5.32%

**面试考点：** checkpoint merging 的微调-泛化平衡、非英语检索器的特殊挑战、低成本高质量部署策略
