# Embedding 学习：推荐系统的表示基石

> 📚 参考文献
> - [Linear-Item-Item-Session-Rec](../papers/Linear_Item_Item_Model_with_Neural_Knowledge_for_Session.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Diffgrm Diffusion Generative Rec](../../03_rerank/papers/DiffGRM_Diffusion_based_Generative_Recommendation_Model.md) — DiffGRM: Diffusion-based Generative Recommendation Model
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../01_recall/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Recgpt Generative Rec](../../04_multi-task/papers/RecGPT_A_Large_Language_Model_for_Generative_Recommendati.md) — RecGPT: A Large Language Model for Generative Recommendat...
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../01_recall/papers/GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Pinrec Outcome-Conditioned Multi-Token Generati...](../../01_recall/papers/PinRec_Outcome_Conditioned_Multi_Token_Generative_Retriev.md) — PinRec: Outcome-Conditioned Multi-Token Generative Retrie...
> - [Gems Long Sequence Generative Rec](../../01_recall/papers/GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Contrastive-Learning-Recsys](../../01_recall/papers/contrastive_learning_recsys.md) — 对比学习在推荐系统中的应用

> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：Word2Vec, Item2Vec, EGES, Node2Vec, Contrastive Learning, Semantic ID 系列

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

1. **Embedding 是推荐系统的"通用语言"**：用户、物品、特征都通过 Embedding 转化为同一向量空间，使得相似度计算、特征交叉、模型训练成为可能
2. **从 ID Embedding 到 Semantic Embedding 的演进**：ID Embedding（随机初始化、纯靠训练）→ 预训练 Embedding（Word2Vec/Graph2Vec 预训练）→ 内容 Embedding（BERT/CLIP 编码语义）→ Semantic ID（RQ-VAE 量化语义）
3. **对比学习是 Embedding 预训练的主流方法**：SimCLR/MoCo 的思想在推荐中广泛应用——同一用户的不同行为子序列是正样本对，不同用户是负样本对
4. **Embedding 维度、存储、更新是工程三大挑战**：十亿级物品 × 128 维 = 500GB+，需要分布式 Parameter Server + 增量更新

---

## 🎓 常见考点（5条）

### Q1: Item2Vec vs Graph Embedding 的区别？
**30秒答案**：Item2Vec 将用户行为序列类比句子，物品类比单词，用 Word2Vec 训练物品 Embedding。Graph Embedding（EGES/Node2Vec）在物品共现图上随机游走生成序列，再用 Word2Vec 训练。Graph 方法可以利用更丰富的图结构信息。

### Q2: 对比学习在推荐 Embedding 预训练中怎么用？
**30秒答案**：正样本构建——对用户行为序列做数据增广（随机裁剪/mask/reorder），同一用户的不同增广版本互为正样本。负样本——不同用户的行为。Loss = InfoNCE：最大化正样本相似度，最小化负样本相似度。

### Q3: Embedding 冷启动怎么处理？
**30秒答案**：新物品没有训练过 ID Embedding。方案：①内容 Embedding 代替（BERT/CLIP 编码物品内容）；②相似物品 Embedding 插值；③Hash Embedding（多个 hash function 的 Embedding 求和，类似 feature hashing）。

### Q4: 大规模 Embedding Table 的存储方案？
**30秒答案**：①分布式 PS（Parameter Server）：Embedding 按 hash 分片到多台机器；②混合精度：高频 ID 用 FP32，低频用 FP16/INT8；③Hash Collision：多个 ID 共享同一 Embedding（有精度损失）；④动态 Embedding：只为活跃 ID 分配 Embedding。

### Q5: Semantic ID 和传统 ID Embedding 的本质区别？
**30秒答案**：传统 ID Embedding 是"查表"——每个 ID 一个独立向量，ID 之间没有语义关系。Semantic ID 是"编码"——用 RQ-VAE 将物品内容编码为多级 token，相似物品的 ID 也相似，天然支持泛化和冷启动。

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

- **上游依赖**：Word2Vec/GNN、对比学习、VQ-VAE
- **下游应用**：向量召回、特征输入、Semantic ID 生成式推荐
- **相关 synthesis**：推荐系统召回范式演进.md, 推荐系统特征工程体系.md, 生成式范式统一视角.md
## 参考文献

- [BERT](../../papers/bert.md)
- [CLIP](../../papers/clip.md)

## 📐 核心公式直观理解

### Word2Vec (Skip-gram) 目标

$$
\max \sum_{(w, c)} \log \sigma(\mathbf{v}_w^T \mathbf{v}_c') - k \cdot \mathbb{E}_{n \sim P_n} \log \sigma(\mathbf{v}_w^T \mathbf{v}_n')
$$

**直观理解**：经常一起出现的词应该有相似的向量（"巴黎"和"法国"），很少一起出现的应该不同（"巴黎"和"显卡"）。推荐系统中把用户行为序列当"句子"，物品当"词"——连续点击的物品 embedding 会相近。

### Hash Embedding 压缩

$$
e(i) = \frac{1}{K}\sum_{k=1}^{K} E_k[h_k(i)]
$$

**直观理解**：不给每个物品单独存 embedding（数亿物品太贵），而是用 $K$ 个哈希函数映射到 $K$ 个小表中取平均。少量哈希冲突会让部分物品共享表示，但 $K$ 越大冲突影响越小——像布隆过滤器的思想。

### 对比学习 embedding

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z, z^+)/\tau)}{\exp(\text{sim}(z, z^+)/\tau) + \sum_{z^-} \exp(\text{sim}(z, z^-)/\tau)}
$$

**直观理解**："相似的拉近、不同的推远"。关键在于正负样本定义——用户的两次点击是正对（应该相近），不同用户的点击是负对（应该远离）。温度 $\tau$ 越小，模型对 hard negative 越敏感。

