# GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation
> 来源：arxiv/2311.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
广告推荐系统中存在多个独立子系统（召回、粗排、精排、重排），各自优化目标不一致：
- 多阶段漏斗存在目标不对齐（召回高 recall，精排高 CTR）
- 子系统参数无法共享，维护成本高
- 无法利用大规模预训练的语义理解能力
- 跨场景、跨任务迁移困难

## 核心方法与创新点
**GPR (Generative Pre-trained Recommendation)**：统一多阶段推荐为单一生成式模型。

**核心思路：**
将推荐任务统一为条件生成问题：
```
P(item_id | user_context, task_prefix) = Π P(token_t | token_{<t}, context)
```

**Semantic ID 设计：**
- 用分层量化（RQ-VAE）将 item 编码为离散 token 序列
- 每个 item = [c_1, c_2, ..., c_k]，k 层码本
- 生成过程：逐 token 自回归解码

**预训练 + 微调框架：**
```
预训练目标：L_pretrain = -Σ log P(item_t | history_{<t})
微调目标：  L_finetune = -Σ λ_task · log P(item | context, task)
```

**任务前缀（Task-Conditioned）：**
- [RECALL]: 全库检索
- [RANKING]: 候选集重排
- [CTR]: 点击率预估

## 实验结论
- 统一模型 vs 独立多阶段：整体 GMV +3.2%
- 召回阶段 Recall@100：+5.8%
- 精排 AUC：持平（减少了 stage gap 损失）
- 模型参数量：比多套独立模型总和少 40%

## 工程落地要点
1. **Semantic ID 生成**：离线 RQ-VAE 训练，每日/每周更新码本
2. **Beam Search 检索**：生成式召回用 beam search，k=100 取 top 候选
3. **前缀路由**：线上按任务动态切换 prefix token，共享主干参数
4. **增量更新**：新 item 需及时生成 Semantic ID 并更新码本映射
5. **延迟控制**：生成式解码比向量检索慢，需 speculative decoding 加速

## 常见考点
**Q1: GPR 为什么能统一召回和排序？**
A: 通过任务前缀条件化，同一模型根据不同前缀 token 切换行为模式：召回时做全库 beam search 生成，排序时做候选集 scoring，共享底层语义表示。

**Q2: Semantic ID 相比 item ID 的优势？**
A: Semantic ID 捕获 item 语义（相似 item 有相近码字），天然支持跨 item 泛化，缓解长尾和冷启动问题。传统 ID 是独立哈希，无语义关联。

**Q3: 生成式召回的主要挑战？**
A: ①解码延迟高（需自回归多步）②新 item 的 Semantic ID 更新滞后 ③beam search 难以覆盖全库多样性。

**Q4: GPR 训练时如何避免"召回-排序"目标冲突？**
A: 多任务损失加权：L = λ_recall · L_recall + λ_rank · L_rank，不同任务有独立的训练样本（召回用全量曝光，排序用精选候选集）。

**Q5: 如何评估统一模型效果？**
A: 不能只看单阶段指标，需看端到端 GMV/RPM；同时对比各阶段独立模型的 stage gap（上游输出在下游的落差）。

## 模型架构详解

### 候选编码
- **Item 表示**：Semantic ID（层次化离散编码）或稠密向量 Embedding
- **编码方式**：RQ-VAE（残差量化）/ K-Means 聚类 / 端到端学习的 Token 序列
- **多模态融合**：文本/图片/行为信号的统一表示空间

### 检索机制
- **生成式检索**：自回归解码器逐步生成 Item Token 序列
- **向量检索**：双塔编码 + ANN 索引（HNSW/IVF-PQ）
- **混合召回**：多路检索结果的统一评分与去重

### 训练策略
- **正样本**：用户交互（点击/购买/收藏）
- **负采样**：In-batch Negatives + 难负例挖掘
- **对比学习**：InfoNCE Loss 拉近正样本、推远负样本
- **课程学习**：从简单到困难逐步增加负例难度

## 与相关工作对比

| 维度 | 生成式召回 | 双塔向量召回 | 传统倒排 |
|------|-----------|------------|---------|
| 冷启动 | 好（内容特征） | 中（需行为） | 差 |
| 索引维护 | 无需显式索引 | 需 ANN 索引 | 需倒排表 |
| 推理延迟 | 中（自回归） | 低（一次编码） | 低 |
| 可扩展性 | 亿级 | 亿级 | 百万级 |
| 多模态 | 原生支持 | 需要适配 | 困难 |

## 面试深度追问

- **Q: Semantic ID 的设计思路和优势？**
  A: 将 Item 映射为离散 Token 序列（类似自然语言），使推荐问题转化为序列生成。优势：1) 天然支持自回归生成；2) 层次化结构（粗→细）提升检索效率；3) 避免连续向量的 ANN 近似误差。

- **Q: 生成式召回如何处理新物品？**
  A: 1) 内容特征驱动的 Semantic ID 分配（新物品基于属性分配 Token）；2) 增量学习更新 Codebook；3) 备用的 Content-based 召回通道兜底。

- **Q: 多路召回的融合策略？**
  A: 1) 统一打分：所有通道候选用同一模型重新打分；2) 配额分配：各通道按历史表现分配固定配额；3) 加权融合：考虑通道多样性的加权排序。

- **Q: 如何衡量召回质量？**
  A: 离线：Recall@K, HR@K, NDCG@K。在线：端到端 CTR/GMV 提升 + 召回覆盖率 + 新颖性。注意 K 值要与下游排序的候选集大小匹配。
