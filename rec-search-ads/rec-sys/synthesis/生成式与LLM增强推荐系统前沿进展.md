# 生成式与 LLM 增强推荐系统前沿进展（2025）

> 综合总结 | 领域：rec-sys | 学习日期：20260404

## 主题概述

2025年推荐系统的核心趋势是 **生成式推荐（Generative Recommendation）** 与 **LLM 深度融合**。主要研究方向包括：冷启动解决方案、长序列建模效率、公平性优化、连续时间建模、以及 OOD 泛化。

---

## 一、冷启动问题的新解法

### 核心公式

语义 ID 初始化（Cold-Starts in Generative Recommendation）：

$$e_{\text{cold}} = \text{LLM-Encoder}(x_{\text{item}}), \quad P(i_{\text{cold}}) \propto e_{\text{cold}}$$

IDProxy 代理 ID（小红书）：

$$e_{\text{proxy}} = \sum_{k=1}^{K} \alpha_k \cdot e_{j_k}, \quad \alpha_k \propto \text{sim}(e_i^{\text{content}}, e_{j_k}^{\text{content}})$$

渐进迁移：

$$e(t) = (1-\lambda(t)) \cdot e_{\text{proxy}} + \lambda(t) \cdot e_{\text{id}}, \quad \lambda(t) = 1 - e^{-\mu N_{\text{clicks}(t)}$$

### 关键发现
- 语义 ID 初始化贡献约 60% 的冷启动提升
- IDProxy 在新品前 7 天 CTR AUC +3.1‰，D+1 CTR +12.3%
- LLM 推理链（CoT）能推断冷启动物品的目标用户群，+31.2% Recall@20

---

## 二、长序列建模效率

### PerSRec 记忆压缩

$$h_u(t) = \text{Attention}(Q_{\text{session}}, K_{M_u}, V_{M_u})$$

$$\text{复杂度}: O(n^2) \rightarrow O(nk), \quad k \ll n$$

### R2LED 检索+精炼

$$h_u^{\text{refined}} = \text{Transformer}(\text{TopK}_{i \in \text{hist}}[\text{sim}(e_a, e_i)], a)$$

### 工程对比

| 方法 | 序列处理 | 记忆存储 | 适用场景 |
|------|---------|---------|---------|
| SASRec | 全序列 Attention | 无 | 序列 ≤ 200 |
| PerSRec | 记忆 Attention | 用户记忆矩阵 | 序列 ≤ 2000 |
| R2LED | 检索+精炼 | Faiss Index | 序列 ≤ 100k |

---

## 三、公平性与 OOD 泛化

### UFO 自对弈 DPO

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_f|x)}{\pi_{\text{ref}}(y_f|x)} - \beta \log \frac{\pi_\theta(y_b|x)}{\pi_{\text{ref}}(y_b|x)}\right)\right]$$

UFO 3 轮迭代后人口统计组间 Recall 差距缩小 **67%**。

### LLM Graph Invariant Learning

$$\min_\phi \max_e \mathcal{L}(f(\mathcal{G}^{\text{inv}}_\phi), y^e)$$

跨地域泛化 NDCG@10 **+17.3%**，跨时段 **+12.1%**。

---

## 四、连续时间建模

Time-Guided Graph Neural ODE：

$$h_u(t) = h_u(t_0) + \int_{t_0}^{t} \text{GNN}(h_u(\tau), \mathcal{G}_u) \cdot e^{-\lambda(\tau - t_{\text{last}})} \, d\tau$$

ODE 建模用户兴趣连续演化，时间敏感场景（促销期）提升 **+15%**。

---

## 五、用户留存优化

三阶段信号融合：

$$\text{Score}(u, i) = \alpha \cdot P_{\text{CTR}} + \beta \cdot P_{\text{Save}} + \gamma \cdot P_{\text{Revisit}} + \delta \cdot P_{\text{Retain}}$$

生存分析处理延迟标签：

$$P_{\text{Retain}(t) = \exp\left(-\left(\frac{t}{\lambda}\right)^k\right) \quad \text{Weibull}$$

---

## 六、LLM 推荐系统的持续更新

LFU（Locate-Forget-Update）范式：

$$\mathcal{F}_\theta = \mathbb{E}\left[\left(\nabla_\theta \log p(y|x)\right)^2\right] \quad \text{(Fisher Information)}$$

$$\theta_{t+1} = \theta_t + \Delta\theta_{\text{forget}}^{\text{GR}} + \Delta\theta_{\text{update}}^{\text{LoRA}}$$

比全量 SFT 计算量减少 **94%**，旧任务性能保持 **97.3%**。

---

## 🎓 面试高频 Q&A（10题）

**Q1**: 生成式推荐和传统推荐的核心区别？  
**A**: 传统：打分 + 排序（二阶段）；生成式：直接生成物品 ID 序列（端到端），自然建模列表级交互，支持多样性约束。

**Q2**: 冷启动的终极解法是什么？  
**A**: 语义桥接（LLM 特征）+ 代理迁移（热门物品代理 Embedding）+ 渐进过渡（交互积累后迁移至精确 ID）。三者结合效果最佳。

**Q3**: 如何让推荐系统适应用户兴趣漂移？  
**A**: 短期：当前会话行为加权；中期：滑动窗口 + 时间衰减；长期：ODE 连续建模 + LFU 增量更新 LLM 参数。

**Q4**: 推荐系统公平性的技术实现？  
**A**: DPO 偏好优化（公平推荐 vs 偏置推荐的偏好对）+ Self-Play 迭代（渐进消除残余偏差）+ 多目标约束（帕累托优化）。

**Q5**: 什么是 OOD 推荐？主要解法？  
**A**: 在训练分布外场景（新地域/新时段/新用户群）性能下降。解法：不变特征学习（GIL）+ 多环境训练 + LLM 语义特征（比 ID 更稳定）。

**Q6**: 用户行为序列长度 vs 模型效果的关系？  
**A**: 通常正相关，但收益递减。长序列关键是效率：PerSRec（压缩记忆）解决计算；R2LED（检索精炼）解决 10k+ 场景。

**Q7**: ItemRAG 如何防止 LLM 推荐幻觉？  
**A**: Constrained Decoding（Token Trie）+ 结构化物品卡片（ID 锚点）+ 物品级 RAG（实时检索保证物品存在）。

**Q8**: 用户留存为什么比 CTR 更重要？  
**A**: CTR 优化短期行为，留存优化长期价值。过度优化 CTR（点击诱饵）损害留存（用户疲劳流失）。DAU 增长 > CTR 提升的商业价值。

**Q9**: LLM 推荐系统的部署挑战？  
**A**: 推理延迟（LLM 100ms+ vs 传统 <10ms）、知识截止（需 RAG/定期微调）、参数更新（LFU 增量 vs 全量 SFT）、成本（Token 消耗）。

**Q10**: 如何平衡推荐多样性和相关性？  
**A**: DPP（行列式点过程）重排、Token Controlled Re-ranking（列表级生成）、UFO 公平性约束、MMR（最大边际相关性）等。

---

## 📚 参考文献

1. Cold-Starts in Generative Recommendation (2025)
2. PerSRec: Efficient Sequential Recommendation (2025)
3. ItemRAG: Item-Based RAG for LLM Recommendation (2025)
4. GeoGR: Generative Retrieval for POI Recommendation (2025)
5. LLM Reasoning for Cold-Start Item Recommendation (2025)
6. UFO: Unfair-to-Fair via Self-Play Fine-tuning (2025)
7. Token-Controlled Re-ranking for Sequential Recommendation (2025)
8. Time Matters: Time-Guided Graph Neural ODEs (2025)
9. Save, Revisit, Retain: User Retention Framework (2025)
10. LLM Graph Invariant Contrastive Learning OOD (2025)
11. LLM-based Evolutional Recommendation: Locate-Forget-Update (2025)

---

## 相关概念

- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
