# Embedding 技术全景：从 ID 到语义

> **一句话总结**：Embedding 就是把离散的东西（用户ID、商品ID、query文本）变成稠密的低维向量，让"相似的东西靠近"。从 one-hot 到 Semantic ID，Embedding 经历了 5 代演进。
>
> **为什么要学**：Embedding 是推荐、搜索、广告、LLM 四个领域的共同地基。你遇到的几乎每个模型，第一步都是 Embedding。

**相关概念页**：[Attention in RecSys](attention_in_recsys.md) | [序列建模演进](sequence_modeling_evolution.md) | [生成式推荐](generative_recsys.md) | [向量量化方法](vector_quantization_methods.md)

---

## 1. ID Embedding：最基础的表示

每个用户/物品分配一个 ID，用 Embedding 矩阵把 ID 映射为向量：

$$\mathbf{e}_i = \mathbf{E}[i] \in \mathbb{R}^d$$

**工程要点**：
- **维度选择**：8~128 维，经验公式 $d \approx 6 \times (\text{vocab\_size})^{1/4}$（Google 建议）
- **初始化**：Xavier/He 初始化，别用全零
- **OOV（Out-of-Vocabulary）**：新用户/新物品没有 ID → 用默认 embedding 或 hash bucket

**问题**：纯 ID embedding 学不到交叉信息，冷启动时是随机向量。

📄 详见 [rec-sys/02_rank/synthesis/Embedding学习_推荐系统表示基石.md](../rec-search-ads/rec-sys/02_rank/synthesis/Embedding学习_推荐系统表示基石.md)

---

## 2. Feature Interaction Embedding：特征交叉

FM/DeepFM 的核心贡献：让不同特征的 embedding 自动交叉。

**FM 二阶交叉**：

$$\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

关键洞见：$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 让从未同时出现过的特征组合也有分数（泛化能力）。

**演进路线**：
| 模型 | 交叉方式 | 特点 |
|------|---------|------|
| FM | 二阶内积 | 显式，可解释 |
| DeepFM | FM + DNN | 低阶+高阶并行 |
| DCN-V2 | Cross Network | 矩阵替代向量，任意阶 |
| AutoInt | Multi-head Self-Attention | attention 做交叉 |
| FiBiNET | SENet 压缩-激发 | 动态特征重要性 |

📄 详见 [rec-sys/02_rank/synthesis/CTR模型深度解析.md](../rec-search-ads/rec-sys/02_rank/synthesis/CTR模型深度解析.md)

---

## 3. Graph Embedding：结构化关系

当你有用户-物品交互图时，ID Embedding 可以融入图结构信号。

**典型方法**：
| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| DeepWalk/Node2Vec | 随机游走 + Word2Vec | 小规模图 |
| PinSAGE | 邻居采样 + GraphSAGE | 十亿级工业图（Pinterest） |
| LightGCN | 极简 GCN：只做邻居聚合 | 协同过滤主流 |

**LightGCN 核心公式**：

$$\mathbf{e}_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \mathbf{e}_i^{(l)}$$

**直觉**：如果你和很多人都喜欢同一个物品，你们的 embedding 会靠近（高阶协同信号）。

**工程挑战**：
- Mini-batch 邻居采样（全图训练不现实）
- 图更新延迟（新交互进来后需重新聚合）

📄 详见 [rec-sys/04_multi-task/synthesis/图神经网络在推荐中的应用.md](../rec-search-ads/rec-sys/04_multi-task/synthesis/图神经网络在推荐中的应用.md)

---

## 4. Semantic ID：从离散 ID 到语义编码

传统 ID Embedding 的根本问题：**ID 没有语义**——物品 37482 和物品 37483 可能完全不相关。

**Semantic ID 方案**（TIGER, Google 2023）：
1. 用 **RQ-VAE** 把物品的内容特征编码为一串离散 code: `[c1, c2, c3, c4]`
2. 每层 code 对应一个语义层次：`c1=电子产品, c2=手机, c3=旗舰, c4=具体型号`
3. 用 **自回归模型** 逐层生成 code → 天然支持生成式召回

**优势**：
- 冷启动：新物品只要有内容特征就能生成 Semantic ID
- 碰撞率低：多层码本组合空间巨大
- 可解释性：code 层次对应语义粒度

**后续演进**：
- **UniGRec**：连续向量替代离散 code，碰撞率 0
- **Spotify 部署**：工业级 Semantic ID 在音乐推荐中的落地
- **Prefix Ngram**（Meta 2025）：用于排序模型的 Semantic ID 变体，见下方 §4-B

📄 详见 [[SemanticID从论文到Spotify部署|rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md]] | [[generative_recsys|生成式推荐]] | [[vector_quantization_methods|向量量化四大方法]]

### 4-B. Semantic ID Prefix Ngram：解决 Embedding 不稳定性（Meta 2025）

> 📄 arXiv:2504.02137 | Meta Ads Ranking 生产部署

传统 Random Hash 把不相关 ID 随机碰撞到同一 bucket，导致 embedding 被噪声污染。Meta 的 **Semantic ID Prefix Ngram** 用层次化内容聚类替代随机哈希：

**核心机制**：
1. 用内容 embedding 层次化聚类，生成语义 ID：$[c_1, c_2, ..., c_L]$
2. 对每个前缀子序列建 embedding：$\mathbf{E}[c_1], \mathbf{E}[c_1,c_2], ..., \mathbf{E}[c_1,...,c_L]$
3. 聚合所有前缀 embedding 得到最终表示

**为什么有效**：
- **语义碰撞 > 随机碰撞**：相似物品共享前缀 embedding，碰撞是有意义的信息共享
- **长尾友好**：低频 ID 通过共享前缀获得更好的初始化（类似迁移学习）
- **表示稳定**：新 ID 诞生时，其前缀 embedding 已经被训练好，不是随机向量
- **Attention 协同**：在 attention-based 序列模型中，同前缀物品的注意力模式更一致

**和 TIGER 的区别**：TIGER 用 RQ-VAE 做生成式召回；Prefix Ngram 用层次聚类做排序模型的 embedding 表示。目标不同，但底层逻辑相同——**用语义结构替代随机 ID**。

📄 详见 [[SemanticID从论文到Spotify部署|rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md]]

---

## 5. LLM Embedding：大模型时代的表示

LLM 本身就是最强的 Embedding 提取器：

| 方法 | 原理 | 典型应用 |
|------|------|---------|
| 双编码器 (DPR) | BERT 分别编码 query/doc，内积相似度 | 搜索召回 |
| Cross-Encoder | BERT 联合编码 [query; doc]，全交互 | 搜索重排 |
| ColBERT | token 级延迟交互 | 精度-效率平衡 |
| Qwen3-Emb | LLM backbone 全量微调 | BEIR SOTA |
| E5-Mistral | 指令化 embedding | 多任务通用表示 |

**对推荐的影响**：
- **文本特征增强**：用 LLM embedding 替代 title/description 的 TF-IDF
- **跨模态对齐**：图文一起编码（CLIP 思路进入推荐）
- **Zero-shot 推荐**：不需要交互数据，纯语义匹配

📄 详见 [search/01_recall/synthesis/检索三角_Dense_Sparse_LateInteraction.md](../rec-search-ads/search/01_recall/synthesis/检索三角_Dense_Sparse_LateInteraction.md)

---

## 6. 工程实践速查

### 维度选择
| 场景 | 推荐维度 | 原因 |
|------|---------|------|
| 小规模 (<1M items) | 16-32 | 数据少，高维过拟合 |
| 中规模 (1M-100M) | 64-128 | 工业主流 |
| 大规模 (>100M) | 128-256 | 需要更多表达力 |
| LLM Embedding | 768-4096 | 跟着模型走 |

### 冷启动方案
| 方案 | 思路 | 适用 |
|------|------|------|
| 默认 Embedding | 所有新 ID 共享一个向量 | 最简单，效果一般 |
| 内容特征映射 | 用 side info (类目、标题) 生成初始 embedding | 有内容特征时 |
| Meta-Learning | MAML 快速适配新 ID | 研究导向 |
| Semantic ID | 内容编码生成 ID → 天然解决冷启动 | 生成式推荐 |

### OOV 处理
| 方案 | 原理 | 工程复杂度 |
|------|------|-----------|
| Hash Bucket | 多个 ID 共享一个 embedding | 低 |
| Feature Hashing | hash trick，碰撞后平均 | 低 |
| 动态扩表 | 在线新增 embedding 行 | 高 |

---

## 6. Reasoning Embedding：推理增强的表示（2025-2026 前沿）

传统 LLM Embedding 仍是"直接编码"——把文本直接映射为向量。当 query 和目标文档之间存在语义鸿沟（需要推理才能关联）时，direct embedding 力不从心。

**两种推理增强路径**：

| 方法 | 推理引入时机 | 核心思路 | 代表 |
|------|------------|---------|------|
| LREM | Inference-time | 先生成 CoT 推理链，再基于推理结果编码 embedding | 阿里电商搜索 |
| ReasonEmbed | Training-time | ReMixer 合成推理密集数据 + Redapter 自适应权重 | BRIGHT nDCG@10=38.1 |

**LREM 两阶段训练**：
1. SFT + InfoNCE：在 Query-CoT-Item 三元组上联合训练推理能力和编码能力
2. RL 精调：强化学习优化推理轨迹质量

**ReasonEmbed 的关键创新**：
- ReMixer 三阶段数据合成：条件化 query 生成 → 排除源文档的候选挖掘 → 推理增强标注（解决合成数据 triviality 问题）
- Redapter：推理越难的样本训练权重越高

**与前代的关系**：Reasoning Embedding 是 LLM Embedding 的自然进化——不仅用 LLM 做编码器，还让 LLM 的推理能力服务于表示学习。

📄 详见 [[20260411_dense_retrieval_and_reranking_advances.md|search/synthesis/20260411_dense_retrieval_and_reranking_advances.md]]

---

## 演进总结

```
One-Hot (稀疏，百万维)
    │
    ▼
ID Embedding (稠密，学习得到)
    │
    ├─ Feature Interaction ──── FM/DeepFM/DCN (特征交叉)
    │
    ├─ Graph Embedding ─────── PinSAGE/LightGCN (结构信号)
    │
    ├─ Semantic ID ─────────── RQ-VAE 层次编码 (语义+生成式)
    │   └─ Prefix Ngram ────── 层次聚类+前缀共享 (排序+稳定性)
    │
    └─ LLM Embedding ──────── BERT/LLM backbone (最强表示力)
        └─ Reasoning Embedding ── CoT推理+编码 (LREM/ReasonEmbed)
```

## 面试高频问题

1. **FM 和 DNN 做特征交叉有什么区别？** → FM 显式二阶可解释，DNN 隐式高阶但黑盒。DeepFM 二者并行互补。
2. **Graph Embedding 和 ID Embedding 有什么关系？** → Graph Embedding 可以作为 ID Embedding 的初始化，或者拼接使用，提供高阶协同信号。
3. **冷启动时 ID Embedding 怎么办？** → 默认值→内容映射→Meta-Learning→Semantic ID，按复杂度递进选择。
4. **为什么推荐系统的 Embedding 维度通常比 NLP 小？** → 推荐的 token 是 ID（语义简单），NLP 的 token 是词（语义复杂）；推荐需要实时推理，维度太大 ANN 检索变慢。
5. **Random Hashing vs Semantic ID Prefix Ngram？** → Random Hash 让不相关 ID 碰撞，污染 embedding；Prefix Ngram 让语义相似 ID 碰撞，碰撞变成信息共享。Meta 生产验证：长尾 AUC 显著提升，预测稳定性改善。
6. **Reasoning Embedding 和普通 LLM Embedding 的区别？** → 普通 LLM Embedding 直接编码文本，捕捉统计共现模式；Reasoning Embedding 先推理再编码（LREM）或用推理密集数据训练（ReasonEmbed），能跨越 query-doc 语义鸿沟。LREM 已部署于淘宝电商搜索。
