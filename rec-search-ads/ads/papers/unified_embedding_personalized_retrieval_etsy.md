# Unified Embedding Based Personalized Retrieval in Etsy Search

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

电商搜索的召回阶段需要同时满足：
1. **相关性**：返回与查询语义相关的商品
2. **个性化**：结合用户历史行为返回偏好商品
3. **统一性**：避免多个独立召回通道（语义召回、协同过滤召回）造成的系统复杂性

Etsy 的传统做法是多个独立召回模型（BM25、CF、语义向量），召回结果合并后排序。问题：各通道相互独立，难以联合优化个性化与相关性。

**目标**：用统一的 embedding 空间同时建模 query 相关性和用户个性化，单个模型覆盖多种召回需求。

## 核心方法与创新点

### 统一 Query 表示

将 query 表示扩展为：
$$q_{unified} = f(q_{text}, u_{history}, u_{context})$$

- $q_{text}$：query 文本 embedding（BERT/Sentence-T5）
- $u_{history}$：用户历史点击/购买商品的 embedding 聚合
- $u_{context}$：用户实时上下文（地区、设备、时间等）

### 双塔模型（Bi-Encoder）

$$\text{Query Tower: } e_q = Encoder_Q(q_{text}, u_{history}, u_{context})$$
$$\text{Item Tower: } e_i = Encoder_I(title, desc, category, price, ...)$$

相似度：$s(q, i) = e_q \cdot e_i$

### 多任务训练目标

$$\mathcal{L} = \mathcal{L}_{click} + \lambda_1 \mathcal{L}_{purchase} + \lambda_2 \mathcal{L}_{relevance}$$

- 点击损失：用 in-batch negative sampling
- 购买损失：更高权重，捕捉转化信号
- 相关性损失：基于人工标注的 query-item 相关性

### Hard Negative Mining

除了随机负样本，加入：
1. **曝光未点击**（impression negatives）：难负样本，提升判别能力
2. **同品类负样本**：同品类但不相关的商品，增加语义区分
3. **流行偏差去除**：对高曝光商品降采样，避免模型偏向热门

## 实验结论

- 统一 embedding 在 recall@100 上比独立双塔提升约 8%
- 个性化版 query tower 比纯文本 query 提升 recall 约 12%
- 在 Etsy 搜索线上 A/B 测试，购买转化率提升 2-3%
- 硬负采样策略对 AUC 提升约 1.5%

## 工程落地要点

1. **向量索引**：线上使用 FAISS HNSW 索引，支持百亿级商品的实时 ANN 检索
2. **Query Tower 实时性**：用户历史需要实时更新，用 streaming feature store（Redis）
3. **Item Tower 离线**：商品 embedding 定期离线更新（每日），增量更新新品
4. **负样本质量**：in-batch negative 的 batch size 越大越好（4096+），提升对比学习质量
5. **多语言支持**：Etsy 是多国平台，item tower 需要多语言 encoder 或翻译预处理
6. **冷启动**：新品无曝光历史，用 content-based 特征作为 item embedding 的 fallback

## 面试考点

**Q：统一 embedding 和独立多路召回相比有什么优缺点？**
A：优点：联合优化个性化和相关性，单模型维护成本低，embedding 空间一致；缺点：单模型覆盖多目标可能相互竞争，调优复杂，对训练数据质量要求更高，且不同召回信号的权重需要精心设计。

**Q：双塔模型中为什么 item tower 可以离线计算而 query tower 需要实时？**
A：item 数量相对固定（数千万级），可以提前离线计算并建索引；query 是用户实时输入，且需要结合用户当前上下文（实时行为、当前 session），必须在线实时计算。

**Q：如何处理 in-batch negative 的假负样本问题（false negative）？**
A：1) 对训练集中已知的正样本对做去重（deduplication）；2) 用 clicked but not purchased 作为 soft negative（温和负样本）；3) 基于商品相似度做 negative 过滤，去掉与 anchor 相似度过高的 item。
