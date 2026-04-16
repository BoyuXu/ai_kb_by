# 搜索/信息检索论文笔记 — 2026-04-16

## 1. LACONIC: Dense-Level Effectiveness for Scalable Sparse Retrieval via a Two-Phase Training Curriculum

**来源：** https://arxiv.org/abs/2601.01684 (Jan 2026)
**领域：** 学习型稀疏检索
**核心问题：** 稀疏检索高效但效果与稠密检索有差距，如何让稀疏检索达到稠密检索水平？

**核心方法：**
- 基于 Llama-3 架构（1B/3B/8B）的学习型稀疏检索器
- **两阶段训练课程：**
  1. **弱监督预微调：** 适配因果 LLM 进行双向上下文化
  2. **高信号微调：** 使用精选困难负样本进行对比学习

**SOTA 结果：**
- LACONIC-8B 在 MTEB Retrieval 基准上达到 60.2 nDCG（截至 2026.1.1 排名第 15）
- 索引内存仅为等效稠密模型的 29%（节省 71%）
- 可在普通 CPU 硬件上高效运行

**关键价值：** 首次证明学习型稀疏检索可以在 MTEB 上与稠密检索模型竞争

**面试考点：** 稀疏 vs 稠密检索的效率-效果权衡、因果 LLM 双向适配的技术细节、倒排索引的工程优势

---

## 2. Generative Recall, Dense Reranking: Learning Multi-View Semantic IDs for Text-to-Video Retrieval

**来源：** https://arxiv.org/abs/2601.21193 (Jan 2026)
**领域：** 生成式检索 × 视频检索
**核心创新：** 结合生成式召回和稠密重排序两阶段，用多视角 Semantic ID 实现高效文本到视频检索

**方法设计：**
- **召回阶段：** 将视频编码为多视角 Semantic ID（视觉/语义/时序），通过解码文本查询生成 ID token 实现快速召回
- **重排阶段：** 用稠密表征进行精细重排
- **推理效率：** Semantic ID 方案实现近常数级推理和存储复杂度

**核心优势：** 将生成式检索从文本扩展到多模态（视频），解决视频嵌入维度高、存储大的痛点

**面试考点：** 多模态 Semantic ID 的设计、生成式召回的 beam search 策略、文本-视频对齐技术

---

## 3. A Survey of Model Architectures in Information Retrieval

**来源：** https://arxiv.org/abs/2502.14822 (Feb 2026)
**领域：** IR 模型架构综述
**核心定位：** 全面梳理信息检索中的模型架构演进

**架构谱系：**
1. **经典方法：** BM25、TF-IDF、语言模型
2. **稠密双编码器：** DPR、Contriever（零样本）
3. **后期交互模型：** ColBERT（token 级细粒度匹配）
4. **神经稀疏检索：** SPLADE、SPLARE
5. **交叉编码器：** query-doc 拼接深度交互，重排阶段使用
6. **生成式检索：** DSI、GENRE、Semantic ID 方法
7. **LLM 重排：** 利用 LLM 做 listwise/pointwise/pairwise 重排

**关键对比维度：** 效率 vs 效果、离线索引 vs 在线计算、可解释性

**面试考点：** 检索架构选型决策树、ColBERT 的 MaxSim 操作、各架构的时间/空间复杂度

---

## 4. The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models

**来源：** https://arxiv.org/abs/2512.16236 (Dec 2025)
**领域：** 重排序模型综述
**核心定位：** 追踪重排序模型从启发式方法到 LLM 的完整演进

**演进路径：**
1. **启发式阶段：** 基于规则的重排
2. **交叉编码器阶段：** BERT-based cross-encoder
3. **序列生成阶段：** T5-based（monoT5/duoT5）
4. **图神经网络阶段：** GNN-based 关系建模
5. **LLM 阶段：** GPT/LLaMA 做 zero-shot/few-shot 重排

**LLM 重排策略：**
- **Pointwise：** 逐文档评分
- **Pairwise：** 文档对比较
- **Listwise：** 全列表排序（如 RankGPT）

**效率优化：** 知识蒸馏（大模型→小模型）、量化、early exit

**在 RAG 中的应用：** 重排序成为 RAG pipeline 的关键组件，直接影响生成质量

**面试考点：** 三种 LLM 重排策略的优劣、知识蒸馏在重排中的应用、重排与 RAG 的结合

---

## 5. From Retrieval to Generation: Comparing Different Approaches

**来源：** https://arxiv.org/abs/2502.20245 (Feb 2026)
**领域：** 检索 vs 生成对比研究
**核心问题：** 传统检索（retrieve-then-read）与生成式检索（直接生成答案）各有优劣，如何选择？

**对比维度：**
- **准确性：** 检索式在事实性任务占优，生成式在创意任务占优
- **效率：** 检索式需要索引维护，生成式需要大模型推理
- **可解释性：** 检索式可追溯来源，生成式为黑盒
- **鲁棒性：** 检索式对检索质量敏感，生成式对提示词敏感

**混合方案：** RAG（检索增强生成）= 检索 + 生成的最佳实践

**面试考点：** 检索 vs 生成的选型决策、RAG 的核心设计原则、端到端 vs 分阶段系统的权衡

---

**今日 search 总结：** 5 篇论文覆盖信息检索的核心技术栈。关键趋势：(1) 学习型稀疏检索（LACONIC）首次在 MTEB 上追平稠密检索；(2) 生成式检索从文本扩展到多模态（视频）；(3) LLM 重排成为标准组件，推动 RAG pipeline 质量提升。
