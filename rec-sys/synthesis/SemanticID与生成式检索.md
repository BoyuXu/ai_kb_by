# Semantic ID + 生成式检索：推荐系统的"下一代召回"

> 📚 参考文献
> - [Variable-Length-Semantic-Ids-For-Recommender-Sy...](../../rec-sys/papers/Variable_Length_Semantic_IDs_for_Recommender_Systems.md) — Variable-Length Semantic IDs for Recommender Systems
> - [Spotify Unified Lm Search Rec](../../rec-sys/papers/A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Variable Length Semantic Id](../../rec-sys/papers/Variable_Length_Semantic_IDs_for_Recommender_Systems.md) — Variable-Length Semantic IDs for Recommender Systems
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Spotify Semantic Id Podcast](../../rec-sys/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Gems Long Sequence Generative Rec](../../rec-sys/papers/GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...


**一句话**：把商品/内容编码成一串"语义暗号"（Semantic ID），让推荐模型直接把用户想要的东西"说"出来，而不是在百万商品里找最像的。

**类比**：传统召回像"图书馆找书"——你有一张卡片（用户 embedding），拿去书架（ANN 索引）扫描哪本书最像。生成式检索像"直接问图书管理员"——管理员根据你以前借过的书，直接说"你应该去借书架 B-17-3 那本"（生成 Semantic ID）。

**核心机制**（5步）：
1. **Semantic ID 生成**：用 RQ-VAE 对物品 embedding 做残差量化，生成 K 个 token 的层次化编码（如 [42, 17, 8]），语义相近的物品共享前缀
2. **序列建模**：把用户历史行为翻译成 Semantic ID 序列，输入 Transformer
3. **自回归生成**：模型逐 token 生成目标物品的 Semantic ID
4. **前缀树约束解码**：维护合法 Semantic ID 的 Trie，Beam Search 时只走合法路径，避免幻觉
5. **多路融合**：和传统 ANN 召回合并，互补（生成式偏探索/冷启动，ANN 偏精确热门）

**今日三篇的技术演进**：
- **Spotify（基础版）**：标准 RQ-VAE + 固定长度（3-4 token），工业首次大规模验证（500万播客，Recall@20 +8.3%）
- **Variable-Length（创新版）**：自适应深度量化，简单物品 2 个 token，复杂物品 5+ token；长尾召回 +8.1%，总 token 数减少 30%
- **GEMs（扩展版）**：多流解码器解决长序列问题（短期流+长期流+类目流并行），不截断历史，重度用户 +15% HR

**和双塔召回的区别**：
| 维度 | 双塔 Dense 召回 | 生成式检索 |
|------|----------------|----------|
| 原理 | 用户/物品各自 embedding，ANN 找最近邻 | 自回归生成物品 Semantic ID |
| 冷启动 | 差（新物品无 embedding） | 好（Semantic ID 基于内容，不依赖交互）|
| 索引大小 | 随物品数线性增长（GB级向量索引）| Trie 远小于 ANN 索引 |
| 多样性 | 偏头部（ANN 精确近邻）| 偏探索（beam search 覆盖多路径）|
| 推理延迟 | ~5ms（GPU ANN）| ~45ms（beam size=10），较慢 |
| 大厂落地 | 主力召回（字节、阿里、腾讯）| 补充召回（Spotify、Amazon 等试验）|

**工业常见做法**：
- Semantic ID 用 3-4 个 token，codebook size 1024（约 10 亿物品空间）
- 必须 Constrained Decoding（前缀树），否则线上空结果率高
- 版本管理：embedding 更新 → Semantic ID 漂移，需灰度迁移
- 与 ANN 召回做多路融合，不替代

**面试考点**：
- Q: Semantic ID 为什么比整数 ID 好？ → 有层次语义（前缀共享）、冷启动友好、可利用语言模型知识
- Q: 为什么变长 Semantic ID 对长尾有帮助？ → 短 ID = 路径短 = beam search 更容易生成，长尾物品恰好信息量少，短 ID 反而更准确
- Q: 生成式检索最大的工程挑战？ → ① Constrained Decoding 实现；② 延迟（~45ms vs ANN ~5ms）；③ Semantic ID 版本管理


## 📐 核心公式与原理

### 1. 双塔相似度
$$score(u, i) = \frac{E_u^T E_i}{\|E_u\| \|E_i\|}$$
- 用户塔和物品塔的余弦相似度

### 2. Softmax 损失
$$L = -\log \frac{\exp(s_{u,i^+})}{\exp(s_{u,i^+}) + \sum_{j \in Neg} \exp(s_{u,j})}$$
- Sampled softmax 训练双塔模型

### 3. ANN 检索
$$\text{Top-K} = \text{HNSW}(E_u, \{E_i\}_{i \in \mathcal{I}})$$
- 近似最近邻从百万候选中检索

### Q1: 推荐系统的实时性如何保证？
**30秒答案**：①用户特征实时更新（Flink 流处理）；②模型增量更新（FTRL/天级重训）；③索引实时更新（新物品上架）；④特征缓存+预计算降低延迟。

### Q2: 推荐系统的 position bias 怎么处理？
**30秒答案**：训练时：①加 position feature 推理时固定；②IPW 加权；③PAL 分解 P(click)=P(examine)×P(relevant)。推理时：设置固定 position 或用 PAL 只取 P(relevant)。

### Q3: 工业推荐系统和学术研究的差距？
**30秒答案**：①规模（亿级 vs 百万级）；②指标（商业指标 vs AUC/NDCG）；③延迟（<100ms vs 不关心）；④迭代（A/B 测试 vs 离线评估）；⑤工程（特征系统/模型服务 vs 单机实验）。

### Q4: 推荐系统面试中设计题怎么答？
**30秒答案**：按层回答：①明确场景和指标→②召回策略（多路）→③排序模型（DIN/多目标）→④重排（多样性）→⑤在线实验（A/B）→⑥工程架构（特征/模型/日志）。

### Q5: 2024-2025 推荐技术趋势？
**30秒答案**：①生成式推荐（Semantic ID+自回归）；②LLM 增强（特征/数据增广/蒸馏）；③Scaling Law（Wukong）；④端到端（OneRec 统一召排）；⑤多模态（视频/图文理解）。

### Q6: 推荐系统的 EE（Explore-Exploit）怎么做？
**30秒答案**：①ε-greedy：ε 概率随机推荐；②Thompson Sampling：从后验分布采样；③UCB：置信上界探索；④Boltzmann Exploration：按 softmax 温度控制探索度。工业实践：对新用户多探索，老用户少探索。

### Q7: 推荐系统的负反馈如何利用？
**30秒答案**：①隐式负反馈：曝光未点击（弱负样本）、快速划过（中等负样本）；②显式负反馈：「不喜欢」按钮（强负样本）。处理：加大显式负反馈的权重，用 skip 行为做弱负样本。

### Q8: 多场景推荐（Multi-Scenario）怎么做？
**30秒答案**：同一用户在首页/搜索/详情页/直播间有不同推荐需求。方法：①STAR：场景自适应 Tower；②共享底座+场景特定头；③Scenario-aware Attention。核心：共享知识避免数据孤岛，同时保留场景差异。

### Q9: 推荐系统的内容理解怎么做？
**30秒答案**：①文本理解（NLP/LLM 提取标题、标签语义）；②图片理解（CNN/ViT 提取视觉特征）；③视频理解（时序模型提取关键帧+音频）；④多模态融合（CLIP-style 对齐文本和视觉）。

### Q10: 推荐系统的公平性问题？
**30秒答案**：①供给侧公平（小创作者也有曝光机会）；②需求侧公平（不同用户群体获得同等服务质量）；③内容公平（避免信息茧房）。方法：公平约束重排、多样性保障、定期公平性审计。
