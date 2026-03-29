# 推荐/广告 A/B 测试与在线实验：从假设到决策

> 📚 参考文献
> - [Onerec Unifying Retrieve And Rank With Generative ](../../rec-sys/papers/OneRec_Unifying_Retrieve_and_Rank_with_Generative_Recomme_v2.md) — OneRec: Unifying Retrieve and Rank with Generative Recomm...
> - [Onerec Unifying Retrieve-And-Rank With Generative ](../../rec-sys/papers/OneRec_Unifying_Retrieve_and_Rank_with_Generative_Recomme.md) — OneRec: Unifying Retrieve-and-Rank with Generative Recomm...

> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：A/B Test 实践, Interleaving, Bandits, 因果推断系列

## 📐 核心公式与原理

### 1. 矩阵分解

$$
\hat{r}_{ui} = p_u^T q_i
$$

- 用户和物品的隐向量内积

### 2. BPR 损失

$$
L_{BPR} = -\sum_{(u,i,j)} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj})
$$

- 正样本得分 > 负样本得分

### 3. 序列推荐

$$
P(i_{t+1} | i_1, ..., i_t) = \text{softmax}(h_t^T E)
$$

- 基于历史序列预测下一次交互

---

## 🎯 核心洞察（4条）

1. **离线指标和线上效果经常不一致**：AUC 提升 0.5% 不一定带来线上 CTR 提升，A/B Test 是验证的唯一标准
2. **分流粒度决定实验质量**：用户级分流（同一用户只看一个版本）最常用，但有网络效应的场景需要地域/时间分流
3. **多重检验是常见陷阱**：同时看 10 个指标，期望有 1 个 p<0.05 是纯随机的。需要 Bonferroni 或 BH 校正
4. **Interleaving 比传统 A/B Test 灵敏 10x**：将两个排序算法的结果交替展示给同一用户，用点击偏好判断优劣，所需样本量远小于传统 A/B

---

## 🎓 面试考点（5条）

### Q1: A/B Test 的最小样本量怎么算？
**30秒答案**：`n = (Z_α/2 + Z_β)² × 2σ² / δ²`，α=0.05（显著性）、β=0.2（功效80%）、δ=预期效果大小、σ²=指标方差。CTR 类指标通常需要每组 10K-100K 用户。

### Q2: 常见的 A/B Test 陷阱？
**30秒答案**：①新奇效应（新功能初期点击高但很快消退）；②辛普森悖论（整体效果好但某些子群效果差）；③网络效应/SUTVA 违背（用户间互相影响）；④过早停止（p 值刚到 0.05 就停，实际可能是噪声）。

### Q3: Interleaving 实验怎么做？
**30秒答案**：将 A、B 两个排序结果交替合并（Team Draft 方法：轮流从 A、B 各取一个），展示给同一用户。根据用户点击的是 A 的结果还是 B 的结果来判断哪个更好。优势：不受用户个体差异影响。

### Q4: 推荐 A/B Test 看哪些核心指标？
**30秒答案**：①短期指标：CTR、CVR、人均时长、人均播放数；②长期指标：次日/7日留存、MAU；③业务指标：GMV、广告收入。注意防止短期指标好但长期指标差（如标题党提升 CTR 但降低留存）。

### Q5: 什么时候 A/B Test 不够用？
**30秒答案**：①网络效应场景（社交推荐、双边市场）→ 用 cluster-based 实验；②个性化治疗效应（不同用户反应不同）→ 用 HTE（异质性治疗效应）分析；③多策略交互效应 → 用析因实验设计。

---

### Q6: 推荐系统的实时性如何保证？
**30秒答案**：①用户特征实时更新（Flink 流处理）；②模型增量更新（FTRL/天级重训）；③索引实时更新（新物品上架）；④特征缓存+预计算降低延迟。

### Q7: 推荐系统的 position bias 怎么处理？
**30秒答案**：训练时：①加 position feature 推理时固定；②IPW 加权；③PAL 分解 P(click)=P(examine)×P(relevant)。推理时：设置固定 position 或用 PAL 只取 P(relevant)。

### Q8: 工业推荐系统和学术研究的差距？
**30秒答案**：①规模（亿级 vs 百万级）；②指标（商业指标 vs AUC/NDCG）；③延迟（<100ms vs 不关心）；④迭代（A/B 测试 vs 离线评估）；⑤工程（特征系统/模型服务 vs 单机实验）。

### Q9: 推荐系统面试中设计题怎么答？
**30秒答案**：按层回答：①明确场景和指标→②召回策略（多路）→③排序模型（DIN/多目标）→④重排（多样性）→⑤在线实验（A/B）→⑥工程架构（特征/模型/日志）。

### Q10: 2024-2025 推荐技术趋势？
**30秒答案**：①生成式推荐（Semantic ID+自回归）；②LLM 增强（特征/数据增广/蒸馏）；③Scaling Law（Wukong）；④端到端（OneRec 统一召排）；⑤多模态（视频/图文理解）。
## 🌐 知识体系连接

- **上游依赖**：统计检验、因果推断、实验设计
- **下游应用**：模型上线决策、策略迭代、产品优化
- **相关 synthesis**：偏差治理体系.md, 推荐系统排序范式演进.md
