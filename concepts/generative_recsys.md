# 生成式推荐：从判别到生成的范式转变

> **一句话总结**：传统推荐是"从候选集中挑最好的"（判别式），生成式推荐是"直接创造出用户想要的"（生成式）。这个转变正在重塑召回、排序、创意生成整条链路。
>
> **为什么要学**：生成式推荐是 2024-2026 最热的推荐前沿方向，TIGER/HSTU/MTGR 等工作正在被大厂落地。也是面试中区分"了解前沿"和"只会经典模型"的分水岭。

**相关概念页**：[Attention in RecSys](attention_in_recsys.md) | [Embedding全景](embedding_everywhere.md) | [序列建模演进](sequence_modeling_evolution.md) | [多目标优化](multi_objective_optimization.md)

---

## 1. 判别式 vs 生成式：本质区别

| 维度 | 判别式推荐 | 生成式推荐 |
|------|-----------|-----------|
| 核心操作 | 给候选打分 → 排序 | 直接生成目标 ID/内容 |
| 候选集 | 需要提前构建 | 不需要（从全空间生成） |
| 数学形式 | $P(y|x)$（条件概率） | $P(x)$（联合概率/自回归） |
| 冷启动 | 依赖交互数据 | 可用内容特征生成 |
| 多样性 | 需要后处理（MMR/DPP） | 采样天然带多样性 |
| 代表方法 | DeepFM, DIN, 双塔 | TIGER, HSTU, MTGR |

**类比**：判别式像"从菜单上点菜"，生成式像"跟厨师描述你想吃什么，让他现做"。

---

## 2. 生成式召回：TIGER 与 Semantic ID

### 传统召回的瓶颈
双塔模型：分别编码 user 和 item → ANN 检索最近邻。
问题：① 表示瓶颈（内积无法建模复杂关系）② 索引更新延迟 ③ 冷启动

### TIGER（Google 2023）⭐ 必了解

**核心思路**：把物品编码为一串层次化的离散 token，然后用 Transformer 自回归生成。

**Step 1：构建 Semantic ID**（离线）
用 RQ-VAE（残差量化变分自编码器）把物品特征编码为多层离散 code：

```
物品特征 → Encoder → [c1=3, c2=17, c3=42, c4=8]
                      │      │       │       │
                      电子   手机    旗舰   Galaxy S25
```

每层 code 从粗到细，形成语义层次。

**Step 2：自回归生成**（在线）
给定用户行为序列 $[s_1, s_2, ..., s_T]$（每个 $s_t$ 是一个 Semantic ID），Transformer 自回归预测下一个物品的 Semantic ID：

$$P(\text{next item}) = P(c_1) \cdot P(c_2|c_1) \cdot P(c_3|c_1, c_2) \cdot P(c_4|c_1, c_2, c_3)$$

**Beam Search**：每层展开 top-K 候选 → 层次化剪枝 → 最终得到 top-N 物品。

**优势**：
- 无需 ANN 索引（直接生成 ID）
- 冷启动友好（新物品只要有特征就能编码）
- Beam Search 天然探索多样性

### 后续演进
| 方法 | 改进点 | 效果 |
|------|--------|------|
| UniGRec | 连续向量替代离散 code → 碰撞率 0 | Recall+5.8% |
| Spotify | 工业级 Semantic ID 在音乐推荐落地 | 真实上线 |
| LETTER | 多任务学习 Semantic ID | 多场景复用 |

📄 详见 [rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md](../rec-search-ads/rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md) | [Embedding全景](embedding_everywhere.md) §4

---

## 3. 生成式排序：HSTU 与 Scaling Law

### HSTU（Meta 2024）⭐ 必了解

**核心发现**：推荐系统也有 Scaling Law！参数从百万扩到 **1.5 万亿**，效果持续提升。

**架构改动**（相对标准 Transformer）：
| 改动 | 原因 |
|------|------|
| ReLU 替代 softmax | 推荐不需要概率归一化，ReLU 更稀疏更快 |
| 去掉 LayerNorm | 推荐场景数值稳定，去掉加速推理 |
| 行为序列直接当 token | 不做额外特征工程 |

**意义**：证明推荐系统不只是"特征工程+小模型"，大模型路线也可行。

### MTGR（美团 2024）

双流架构：
- **序列流**：Transformer 建模行为序列（类 HSTU）
- **特征交叉流**：传统 CTR 模型做特征交叉
- 两流融合输出排序分数

**实效**：GMV +2.1%，在美团外卖场景上线。

📄 详见 [rec-sys/synthesis/生成式推荐范式统一_20260403.md](../rec-search-ads/rec-sys/synthesis/生成式推荐范式统一_20260403.md)

---

## 4. 生成式重排：LLM 直接排序

### 思路
把 top-K 候选的描述拼成 prompt，让 LLM 直接输出排列顺序。

```
以下是用户可能感兴趣的商品：
A: iPhone 16 Pro
B: 小米14
C: 猫粮 10kg
D: 手机壳透明

请按用户兴趣从高到低排列：
→ A > B > D > C
```

### 挑战与解法
| 挑战 | 解法 |
|------|------|
| 延迟太高（LLM 推理慢） | 蒸馏到小模型（DEAR 方案） |
| 位置偏差（LLM 偏好列表前面的） | 随机打乱输入顺序 + 多次投票 |
| 无法处理大量候选 | 先精排 top-50 → LLM 重排 top-10 |

### PROMISE：测试时计算扩展
用 **过程奖励模型（PRM）** 在推理时多次采样、评分、选最优排列：
- 生成多条推理链 → PRM 逐步打分 → 选最优
- Recall@10 +9.1%

📄 详见 [rec-sys/03_rerank/synthesis/生成式重排与LLM推理增强.md](../rec-search-ads/rec-sys/03_rerank/synthesis/生成式重排与LLM推理增强.md)

---

## 5. 生成式创意：广告的生成式革命

广告系统中，"生成"最直接的应用是**创意生成**：

| 阶段 | 生成内容 | 方法 |
|------|---------|------|
| 文案生成 | 广告标题/描述 | LLM（GPT/Claude） |
| 图片生成 | 广告素材 | Diffusion（DALL-E/SD） |
| 视频生成 | 短视频广告 | Sora 类模型 |
| 落地页 | 个性化着陆页 | LLM + 模板 |

**核心挑战**：创意效果（CTR 提升）和品牌安全（不出错）的平衡。

📄 详见 [ads/05_creative/synthesis/广告创意优化.md](../rec-search-ads/ads/05_creative/synthesis/广告创意优化.md)

---

## 6. 生成式检索：搜索的生成式路线

搜索领域也在走生成式路线：

| 方法 | 思路 | 和推荐的联系 |
|------|------|-------------|
| DSI (Differentiable Search Index) | 把文档 ID 编码在模型参数里，直接生成 docid | 类似 TIGER |
| GENRE | 自回归生成实体名称 | 实体检索 |
| 生成式 QA | 直接生成答案而非检索文档 | RAG 的终极形态 |

**和推荐生成式召回的共性**：都是 "文档/物品 → 离散 ID → 自回归生成"。

📄 详见 [search/synthesis/2026-04-09_generative_retrieval_evolution.md](../rec-search-ads/search/synthesis/2026-04-09_generative_retrieval_evolution.md)

---

## 7. 全景图：生成式范式在搜广推的渗透

```
                    生成式范式
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    推荐系统          搜索系统          广告系统
        │               │               │
   ┌────┼────┐     ┌────┼────┐     ┌────┼────┐
   ▼    ▼    ▼     ▼    ▼    ▼     ▼    ▼    ▼
  召回  排序  重排  检索  问答  重排  CTR  出价  创意
   │    │    │     │    │    │          │    │
 TIGER HSTU LLM  DSI  RAG  LLM       RL  LLM/
       MTGR Rerank     生成式     AutoBid Diffusion
```

---

## 面试高频问题

1. **生成式召回和传统双塔召回的核心区别？** → 双塔是"编码→检索"（需要 ANN 索引），生成式是"自回归生成 ID"（不需要索引）。生成式还能解决冷启动（新物品可直接编码 Semantic ID）。

2. **Semantic ID 的 RQ-VAE 是怎么工作的？** → 多层量化：第一层把特征量化到最近码字 → 计算残差 → 第二层量化残差 → 循环。每层从粗到细，形成层次语义。

3. **HSTU 证明了什么？** → 推荐系统也有 Scaling Law：参数量从百万到万亿，效果持续提升。但需要特殊优化（ReLU 替代 softmax、去 LayerNorm）才能在推荐数据上 scale。

4. **生成式推荐的最大挑战是什么？** → 推理延迟（自回归生成比打分慢）、训练数据需求大、评估困难（生成的 ID 可能不存在）。工业落地需要在效果和效率间平衡。

5. **生成式重排靠谱吗？** → 离线效果好（LLM 理解力强），但在线延迟是硬伤。DEAR 蒸馏方案（7B→1.5B）是当前最实用的折中。
