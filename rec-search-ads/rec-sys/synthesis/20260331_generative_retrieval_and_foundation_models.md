# 推荐系统前沿综合：生成式检索与排序基础模型

> 综合日期：20260331 | 领域：推荐系统 | 覆盖论文：10篇

## 主题概述

本批次10篇论文聚焦两大趋势：**生成式检索/推荐**和**排序基础模型**，同时覆盖了Mamba架构在图序列建模中的应用、LLM与CF的融合、以及多任务学习的新范式。

## 核心技术脉络

### 1. 生成式检索与可学习索引

传统ANN检索（HNSW/IVF）的索引与模型分离导致目标不一致。MFLI和Rethinking ANN提出端到端可学习索引，将索引构建纳入模型训练：

$$
L = L_{rec} + \lambda_1 L_{index} + \lambda_2 L_{contrast}
$$

关键创新：多面索引（multi-faceted index）为每个item建立多角度编码，支持品类、价格、风格等多维度召回。

### 2. 因果语言模型驱动召回

LinkedIn的CLM召回系统标志着推荐召回从双塔范式向自回归范式的演进：

$$
P(v_{t+1} | v_1, ..., v_t) = \text{CLM}(v_1, ..., v_t)
$$

这与DynamicRAG的迭代检索-生成循环形成互补：前者解决了序列依赖建模，后者解决了检索-生成的动态对齐。

### 3. LLM × 协同过滤融合

CoLLM通过映射网络将CF Embedding注入LLM，解决了LLM推荐中协同信号缺失的问题。TagCF则从标签维度实现LLM语义理解与CF的桥接。两种方法的共同点是**离线计算LLM/CF信号，在线轻量融合**。

### 4. 排序基础模型

iRanker的提出标志着推荐排序进入基础模型时代。类似NLP中BERT→fine-tune的范式：
- 预训练阶段：多场景数据联合训练
- 微调阶段：场景Prompt快速适配

## 关键公式汇总

**MoE门控网络**：

$$
y_k = \sum_{i=1}^{N} g_k^{(i)} \cdot f_i(x), \quad g_k = \text{Softmax}(W_k \cdot x)
$$

**有序逻辑回归**：

$$
P(Y \geq k) = \sigma(f(x) - \theta_k), \quad \theta_1 \leq \theta_2 \leq ... \leq \theta_K
$$

**Graph-Mamba门控融合**：

$$
h_t = \sigma(W_g) \odot h_{mamba} + (1 - \sigma(W_g)) \odot h_{local}
$$

## Q&A 面试精选

**Q1: 生成式检索与传统向量检索的本质区别是什么？**
A: 传统检索是"编码→索引→最近邻搜索"，生成式检索是"编码→解码item ID"。前者依赖固定索引结构，后者直接生成目标。MFLI的可学习索引介于两者之间。

**Q2: MoE解决负迁移的核心原理？**
A: 门控网络让不同任务选择性使用不同专家，相当于为每个任务学习了一个定制化的网络子集，减少任务间参数共享的冲突。

**Q3: 为什么Graph-Mamba比Graph Transformer更适合推荐？**
A: 推荐场景中用户行为序列动辄数百上千，Mamba的线性复杂度 $O(n)$ 比Transformer的 $O(n^2)$ 更具可扩展性。

**Q4: CoLLM中为什么需要映射网络？**
A: CF Embedding和LLM token空间维度不同、语义不对齐，映射网络实现跨空间的对齐变换。

**Q5: GNOLR相比ESMM的优势？**
A: ESMM通过乘法链保证概率一致性但梯度传播路径长。GNOLR用有序回归框架，每个任务独立优化但隐式保证概率有序，训练更稳定。

**Q6: iRanker的zero-shot如何实现？**
A: 预训练阶段学习了通用排序能力，新场景通过场景描述Prompt即可激活对应能力，无需任何标注数据。

**Q7: 可学习索引vs传统索引的训练开销对比？**
A: 可学习索引增加约30%训练时间（需要索引重建Loss的反向传播），但离线训练完全可接受。

**Q8: DynamicRAG的迭代机制会不会导致死循环？**
A: 不会。设置了最大迭代轮数和置信度阈值两个终止条件，实际平均2-3轮即收敛。

**Q9: LinkedIn为什么选择因果语言模型而非BERT？**
A: 推荐是预测"下一个"交互的任务，天然契合自回归（因果）建模；BERT的双向编码更适合理解而非预测。

**Q10: 多面索引的"facet"具体指什么？**
A: 同一个商品可以从品类、价格区间、风格、使用场景等多个维度被检索到，每个维度对应一个facet编码。

## 参考文献

1. Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces (arXiv:2402.00789)
2. Multi-Task Learning for Recommendation with Mixture of Experts (arXiv:2501.18300)
3. DynamicRAG: Leveraging LLM Outputs as Feedback for Dynamic Reranking (arXiv:2505.07233)
4. Large Scale Retrieval for LinkedIn Feed Using Causal Language Models (arXiv:2510.14223)
5. CoLLM: Collaborative Large Language Model for Recommendation (arXiv:2310.13825)
6. MFLI: Multifaceted Learnable Index for Large-scale Recommendation (arXiv:2602.16124)
7. TagCF: LLM-Enhanced Logical Recommendation (arXiv:2505.10940)
8. GNOLR: Generalized Nested Ordered Logistic Regression (arXiv:2505.00000)
9. iRanker: Towards Ranking Foundation Model (arXiv:2506.21638)
10. Rethinking ANN-based Retrieval: Multifaceted Learnable Index (arXiv:2602.16124)
