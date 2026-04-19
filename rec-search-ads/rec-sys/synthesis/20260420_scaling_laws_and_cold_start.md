# Scaling Laws、生成式召回与冷启动：10 篇 Rec-Sys 前沿论文综合

> **日期**：2026-04-20
> **覆盖论文**：LLaTTE / Kunlun / TokenMixer-Large / MixFormer / Scaling New Frontiers / SID Staleness / Deep Research for RecSys / HyFormer / EmerG / ULIM
> **三大主题**：(A) 推荐系统 Scaling Laws (B) 生成式召回新进展 (C) 特征交互与冷启动

**相关概念页**：[[sequence_modeling_evolution|序列建模演进]] | [[generative_recsys|生成式推荐]] | [[attention_in_recsys|Attention in RecSys]] | [[embedding_everywhere|Embedding全景]]

---

## Theme A：推荐系统 Scaling Laws

### 背景

LLM 领域的 Scaling Laws（Kaplan 2020 / Chinchilla 2022）已成共识：$L(N,D) \propto N^{-\alpha} + D^{-\beta}$。推荐系统能否复制这一规律？2024 年 HSTU（Meta）首次证明推荐也有 Scaling Law，但具体形态、瓶颈和工程实践直到 2026 年才逐步清晰。

### 论文详解

#### 1. LLaTTE — Scaling Laws for Multi-Stage Sequence Modeling (Meta, 2601.20083)

| 维度 | 内容 |
|------|------|
| **问题** | 推荐模型能否像 LLM 一样展现 power-law scaling？语义特征的角色是什么？ |
| **方法** | LLM-Style Latent Transformers for Temporal Events (LLaTTE)：大规模 Transformer 序列模型 |
| **核心发现** | ① 推荐性能遵循 power-law scaling（与计算量的幂律关系）；② **语义特征是 scaling 的前提条件**——没有语义特征，模型无法有效利用更深更长的架构；③ Scaling law 是 content-aware 的，单看模型大小或计算量不够 |
| **架构** | 两阶段设计：重计算的大上下文模型异步运行在 upstream user model → 下游排序模型轻量使用 |
| **线上效果** | Facebook Feed + Reels **转化率 +4.3%**，是 Meta 最大的 user model |
| **工业启示** | 异步两阶段是解决延迟约束下 scaling 的关键模式 |

**关键公式**（power-law scaling）：

$$\text{Loss}(C) = \alpha \cdot C^{-\beta} + \gamma$$

其中 $C$ 为计算量（FLOPs），$\alpha, \beta, \gamma$ 为拟合参数。语义特征的引入使 $\beta$ 显著增大（曲线更陡 → scaling 更有效）。

#### 2. Kunlun — Establishing Scaling Laws for Massive-Scale RecSys (Meta, 2602.10016)

| 维度 | 内容 |
|------|------|
| **问题** | 推荐系统 scaling 效率低的根因是什么？如何系统性提升？ |
| **方法** | 统一架构设计，系统优化 Model FLOPs Utilization (MFU) |
| **核心发现** | 低 MFU（17%）和次优资源分配是 scaling 不可预测的主因 |
| **低层优化** | **GDPA** (Generalized Dot-Product Attention)：把 PFFN 重构为多头注意力风格，支持 FlashAttention-like 内核融合；**HSP** (Hierarchical Seed Pooling)：层次化序列压缩；**Sliding Window Attention** |
| **高层优化** | **CompSkip** (Computation Skip)：每隔一层跳过，FLOPs 减少 43.1%，QPS 提升 35%；**Event-level Personalization** |
| **MFU 提升** | 17% → **37%**（NVIDIA B200），scaling 效率翻倍 |
| **线上效果** | Meta Ads topline **+1.2%** |

**Kunlun 的 scaling coefficient 约为 InterFormer 的 2 倍**——同样增加计算量，Kunlun 获得的性能提升是前者的两倍。

#### 3. TokenMixer-Large — Scaling Up to 7B/15B (ByteDance, 2602.06563)

| 维度 | 内容 |
|------|------|
| **问题** | 现有推荐架构（Wukong/HiFormer/DHEN）在深层配置下梯度消失、硬件利用率低 |
| **方法** | 系统性演进 TokenMixer 架构，解决残差路径、MoE 稀疏化和扩展性瓶颈 |
| **核心创新** | ① **Mixing-and-Reverting**：混合操作后恢复原始表示，确保深层梯度稳定传播；② **Inter-layer Residuals + Auxiliary Loss**：跨层残差连接 + 辅助损失；③ **Sparse Per-token MoE**：按 token 级别稀疏路由，高效参数扩展 |
| **规模** | 在线流量达 **7B 参数**，离线实验 **15B 参数** |
| **线上效果** | 电商订单 +1.66%，GMV +2.98%；广告 ADSS +2.0%；直播收入 +1.4% |

#### 4. MixFormer — Co-Scaling Dense and Sequence (ByteDance, 2602.14110)

| 维度 | 内容 |
|------|------|
| **问题** | 序列建模和特征交互分离参数化 → 计算预算分配次优（co-scaling 难题） |
| **方法** | 统一 Transformer 骨架，在同一参数空间内联合建模序列行为和特征交互 |
| **核心创新** | ① 统一参数化消除 dense/sequence 分配权衡；② 深层交互：序列聚合直接受高阶特征语义影响；③ **User-Item 解耦策略**：减少冗余计算，降低推理延迟 |
| **线上效果** | 抖音 + 抖音极速版 A/B 测试，用户活跃天数和 App 使用时长均显著提升 |

#### 5. Scaling New Frontiers — Insights into Large Recommendation Models (2412.00714)

| 维度 | 内容 |
|------|------|
| **定位** | Survey/Vision paper，系统总结大规模推荐模型的发展方向 |
| **核心观点** | ① 传统推荐模型网络参数停滞在千万级，embedding table 已扩展到数十 TB，但网络规模未跟上；② HSTU 展示了扩展到万亿参数的可行性；③ 新范式：用创新结构扩展网络参数，实现持续性能提升 |
| **工业意义** | 明确了大推荐模型（Large Recommendation Model, LRM）的发展路线图 |

### Theme A 横向对比

| 论文 | 机构 | 核心 Scaling 手段 | 参数规模 | MFU/效率 | 线上效果 |
|------|------|-----------------|---------|---------|---------|
| **LLaTTE** | Meta | 两阶段异步 + 语义特征 | 未公开（最大 user model） | — | 转化 +4.3% |
| **Kunlun** | Meta | GDPA + CompSkip + MFU 优化 | — | 17%→37% | topline +1.2% |
| **TokenMixer-Large** | ByteDance | Mixing-Reverting + Sparse MoE | 7B-15B | — | 订单 +1.66% |
| **MixFormer** | ByteDance | 统一参数化 dense+sequence | — | — | 活跃天数↑ |
| **Scaling NF** | Survey | — | 万亿级 vision | — | — |

### Scaling Laws 关键结论

1. **推荐系统的 Scaling Law 是 content-aware 的**（LLaTTE）：语义特征决定 scaling 天花板
2. **MFU 是工业 scaling 的真正瓶颈**（Kunlun）：从 17% 到 37% 的提升直接翻倍 scaling 效率
3. **梯度稳定是深层扩展的前提**（TokenMixer-Large）：Mixing-Reverting + Inter-layer Residuals
4. **Dense 和 Sequence 必须联合 scaling**（MixFormer）：分离参数化导致次优分配
5. **异步两阶段是延迟约束下的通用解法**（LLaTTE）：重模型放 upstream，轻模型做实时排序

---

## Theme B：生成式召回新进展

### 6. Mitigating Collaborative Semantic ID Staleness (2604.13273)

| 维度 | 内容 |
|------|------|
| **问题** | 协同信息 SID 随用户交互漂移而过时（staleness），导致 SID 语义与最新日志不匹配 |
| **背景** | Content-only SID 稳定但忽略交互模式；Interaction-informed SID 更强但会 stale |
| **方法** | 轻量级、模型无关的 **SID 对齐更新**：从最新日志导出刷新后的 SID，对齐到现有 SID 词表，使检索器 checkpoint 保持兼容 |
| **核心创新** | ① 首次在严格时序评估下研究 SID staleness；② Alignment 而非 Rebuild——无需重训模型，只做词表映射 |
| **工业意义** | 解决生成式召回的「维护成本」问题：SID 可以低成本持续刷新 |

**和 QuaSID 的对比**：
- QuaSID（快手）关注 SID **碰撞**（同码不同义）
- SID Staleness 关注 SID **过时**（时间漂移导致语义失配）
- 两者互补：先用 QuaSID 解决碰撞质量，再用 Alignment 解决时间漂移

### 7. Deep Research for Recommender Systems (2603.07605)

| 维度 | 内容 |
|------|------|
| **问题** | 传统推荐只给 item list，用户需自行探索比较 → 被动过滤器 |
| **方法** | **RecPilot**：多智能体框架 = 用户轨迹仿真 Agent + 报告生成 Agent |
| **创新** | ① 自回归生成 exploration-to-decision 轨迹（点击→收藏→购买）；② Encoder-Decoder 架构解耦历史编码和未来行为生成；③ 从 item list → 综合报告的范式转变 |
| **训练** | 自回归生成轨迹，处理历史数据中的类不平衡问题 |
| **意义** | 推荐从「信息过滤」走向「决策辅助」，是推荐 + Agent 融合的前沿方向 |

**生成式召回发展脉络更新**：

```
TIGER (2023) — SID + 自回归生成
    │
    ├── QuaSID (2026) — SID 碰撞优化
    ├── SID Staleness (2026) — SID 时间对齐 ★ NEW
    ├── NEO/Spotify (2025) — 统一搜索+推荐
    │
    └── RecPilot (2026) — 从 item list → 决策报告 ★ NEW
```

---

## Theme C：特征交互与冷启动

### 8. HyFormer — Revisiting Sequence Modeling and Feature Interaction (ByteDance, 2601.12681)

| 维度 | 内容 |
|------|------|
| **问题** | 工业 LRM 的序列建模和特征交互采用解耦 pipeline（LONGER → RankMixer），限制表示能力和交互灵活性 |
| **方法** | 统一混合 Transformer，交替执行两个核心操作 |
| **核心组件** | ① **Query Decoding**：将非序列特征扩展为 Global Tokens，对长行为序列的 KV 表示做层级解码；② **Query Boosting**：通过高效 token mixing 增强跨 query 和跨序列的异构交互 |
| **评估** | 抖音搜索 CTR 预测，billion-scale 工业数据集 |
| **结果** | 在相同参数和 FLOPs 预算下持续优于 LONGER + RankMixer baseline，且 scaling 行为更优 |
| **部署** | 已全量部署 ByteDance，服务数十亿用户 |

**HyFormer vs MixFormer 对比**：

| 维度 | HyFormer | MixFormer |
|------|----------|-----------|
| 核心思路 | 交替 Query Decoding + Query Boosting | 统一参数化 dense + sequence |
| 融合方式 | 层级交替操作 | 同一骨架内联合 |
| 场景 | 抖音搜索 CTR | 抖音推荐 engagement |
| 侧重 | 长序列 + 异构特征交互 | dense scaling + 序列 scaling 均衡 |

### 9. EmerG — Warming Up Cold-Start CTR (KDD 2024, 2407.10112)

| 维度 | 内容 |
|------|------|
| **问题** | 新物品无交互记录，传统 CTR 模型用全局特征交互模式，新物品被热门物品淹没 |
| **方法** | HyperNetwork → Item-Specific Feature Graph → GNN |
| **核心创新** | ① **HyperNetwork 生成物品特定的特征图**：根据物品特征动态构建特征交互图（节点=特征，边=交互）；② **定制化 GNN**：可证明捕获任意阶特征交互（自定义消息传递机制）；③ **Meta-Learning 策略**：跨物品任务优化 HyperNet + GNN 参数，每个任务只调最小物品特定参数集 → 防止小样本过拟合 |
| **结果** | 在 0-shot / few-shot / sufficient-shot 三种场景下均 SOTA |

**EmerG 的核心公式**：

HyperNetwork 生成邻接矩阵：
$$A_i = \text{HyperNet}(\mathbf{x}_i) \in \mathbb{R}^{F \times F}$$

GNN 消息传递（k 阶交互）：
$$\mathbf{h}_f^{(k)} = \sigma\left(\sum_{g \in \mathcal{N}(f)} A_{fg} \cdot W^{(k)} \mathbf{h}_g^{(k-1)}\right)$$

Meta-learning 外循环：
$$\theta^* = \arg\min_\theta \sum_{\mathcal{T}_i} \mathcal{L}(\mathcal{T}_i; \theta - \alpha \nabla_\theta \mathcal{L}(\mathcal{T}_i; \theta))$$

### 10. ULIM — User Long-Term Multi-Interest Retrieval Model (RecSys 2025, 2507.10097)

| 维度 | 内容 |
|------|------|
| **问题** | 排序模型已能处理千级长序列，但召回模型因延迟预算和缺乏 target-aware 机制仍限于百级 |
| **方法** | ULIM：千级行为建模的召回框架 |
| **核心创新** | ① **Category-Aware Hierarchical Dual-Interest Learning**：按品类聚类切分长序列为多兴趣子序列，联合优化长短期兴趣；② **Pointer-Enhanced Cascaded Category-to-Item Retrieval**：PGIN (Pointer-Generator Interest Network) 先预测 next-category (top-K)，再在品类内做 next-item 召回 |
| **数据** | 使用 2 年用户历史行为 |
| **线上效果** | 淘宝秒杀：点击 **+5.54%**，订单 **+11.01%**，GMV **+4.03%** |

**ULIM 的两阶段级联架构**：

```
用户 2 年行为序列
      │
      ▼
[按品类聚类分组]
      │
      ▼
┌─────────────────────┐
│ Dual-Interest Model │
│  长期兴趣 + 短期兴趣 │
└─────────┬───────────┘
          │
          ▼
   PGIN: next-category top-K
          │
          ▼
   Category → Item 召回
          │
          ▼
     最终候选集
```

---

## 三大主题交叉分析

### 1. Scaling + 特征交互的交汇

LLaTTE 发现**语义特征是 scaling 的前提**，HyFormer 和 MixFormer 则从架构层面回答了「怎么在统一框架内同时 scale 特征交互和序列建模」：

| 层次 | 贡献 |
|------|------|
| 理论 | LLaTTE 证明 content-aware scaling law |
| 效率 | Kunlun 的 GDPA + CompSkip 提升 MFU |
| 架构 | HyFormer 的交替 Decode/Boost、MixFormer 的统一参数化 |
| 规模 | TokenMixer-Large 实现 7B-15B |

### 2. 生成式召回的成熟化

SID Staleness 论文标志着生成式召回从「能用」走向「好维护」——解决了工业部署中的持续维护难题。RecPilot 则探索了生成式推荐的终极形态：从 item list 到决策报告。

### 3. 冷启动的新思路

| 方法 | 冷启动策略 | 粒度 |
|------|-----------|------|
| **EmerG** | Item-specific feature graph + meta-learning | 特征交互级 |
| **TIGER/SID** | 内容特征 → SID，无需交互数据 | 物品 ID 级 |
| **ULIM** | 品类级聚类降低稀疏性 | 兴趣级 |

---

## 工业实践总结

### 1. 两阶段异步架构（LLaTTE 模式）

```
                    异步 (100ms+)           实时 (<10ms)
                  ┌──────────────┐      ┌──────────────┐
用户行为序列 ────→ │ 大 User Model │──→   │ 轻量 Ranker  │ ──→ 排序结果
                  │ (Transformer) │      │ (特征查表)    │
                  └──────────────┘      └──────────────┘
                      LLaTTE               下游模型
```

### 2. MFU 优化清单（Kunlun 经验）

| 优化 | FLOPs 减少 | QPS 提升 |
|------|-----------|---------|
| GDPA（PFFN → Attention 式内核融合） | — | 显著 |
| CompSkip（每隔一层跳过） | 43.1% | 35% |
| HSP（层次化序列压缩） | — | — |
| 综合 MFU | 17% → 37% | — |

### 3. 深层模型稳定性技巧（TokenMixer-Large）

- **Mixing-and-Reverting**：操作后恢复原始表示 → 梯度稳定
- **Inter-layer Residuals**：跨层连接（类似 DenseNet 思想）
- **Auxiliary Loss**：辅助损失防止深层退化
- **Sparse Per-token MoE**：按 token 路由 → 参数量 ↑ 但计算量可控

---

## 面试 Q&A（8 题）

### Q1: 推荐系统的 Scaling Law 和 LLM 的有什么不同？

**答**：两个关键区别。第一，推荐 Scaling Law 是 **content-aware** 的（LLaTTE 发现）——语义特征是 scaling 的前提条件，没有语义特征的模型再大也不 scale；而 LLM 的 Scaling Law 主要由参数量 N、数据量 D、计算量 C 决定。第二，推荐系统的 **MFU 极低**（Kunlun 发现只有 17%），远低于 LLM 训练的 30-50%，因为推荐模型有大量 embedding lookup、稀疏操作和异构特征处理。Kunlun 通过 GDPA 等优化提升到 37%，这直接翻倍了 scaling 效率。

### Q2: 如何在延迟约束下实现推荐模型的 scaling？

**答**：LLaTTE 提出的**两阶段异步架构**是关键模式。重计算的大 Transformer user model 异步运行（可以容忍 100ms+ 延迟），产出的 user representation 缓存后供实时排序模型查表使用（<10ms）。upstream 的改进可预测地传递到下游排序任务。ByteDance 的 MixFormer 也采用 user-item 解耦策略降低推理延迟。

### Q3: TokenMixer-Large 如何 scale 到 7B-15B？核心技术挑战是什么？

**答**：核心挑战是**深层模型的梯度退化**。TokenMixer-Large 三管齐下：① Mixing-and-Reverting 操作确保混合后恢复原始表示空间，保持梯度通道；② Inter-layer Residuals 跨层直连；③ Auxiliary Loss 监督中间层。参数扩展则靠 Sparse Per-token MoE——每个 token 只激活部分专家，实现参数量大增但计算量可控。

### Q4: HyFormer 和 MixFormer 都解决「序列+特征联合建模」，有何区别？

**答**：HyFormer 采用**交替优化**思路：Query Decoding（非序列特征→Global Tokens→长序列 KV 解码）和 Query Boosting（token mixing 增强异构交互）交替执行。MixFormer 采用**统一参数化**思路：在同一 Transformer 骨架内联合建模，消除 dense/sequence 的预算分配权衡。HyFormer 侧重抖音搜索（长序列+异构特征），MixFormer 侧重抖音推荐（dense scaling + sequence scaling 均衡）。

### Q5: Semantic ID 的 staleness 问题是什么？怎么解决？

**答**：Interaction-informed SID 基于用户-物品交互模式构建，随时间推移交互分布漂移，SID 的协同语义不再匹配最新日志——这就是 staleness。传统做法是全量重建 SID + 重训模型（成本高）。2604.13273 提出**轻量 SID 对齐**：从最新日志导出新 SID，通过词表映射对齐到旧 SID 空间，检索器 checkpoint 可直接复用无需重训。这解决了生成式召回的工业维护成本问题。

### Q6: EmerG 如何解决冷启动？和 Semantic ID 方式有什么区别？

**答**：两种思路互补。**EmerG**（KDD 2024）从**特征交互**入手：用 HyperNetwork 根据物品特征动态生成 item-specific 的特征交互图，GNN 处理后捕获任意阶交互，meta-learning 防止小样本过拟合。它改善的是**排序阶段**的冷启动 CTR 预测。**Semantic ID** 从**表示**入手：新物品只要有内容特征就能通过 RQ-VAE 编码为 SID，直接参与生成式召回。它解决的是**召回阶段**的冷启动。EmerG 做 item-specific 精细化，SID 做通用化快速入库。

### Q7: ULIM 如何在召回阶段做千级长序列建模？

**答**：两个关键技术。第一，**Category-Aware Hierarchical Dual-Interest Learning**：把 2 年的用户行为按品类聚类切分为多个兴趣子序列，在每个兴趣簇内联合优化长期和短期兴趣——这将超长序列降维为多个短序列。第二，**Pointer-Enhanced Cascaded Retrieval**：先用 PGIN 预测 next-category（top-K），再在预测品类内做 next-item 召回——两阶段级联将候选空间大幅收窄。淘宝秒杀 GMV +4.03%。

### Q8: RecPilot 的「Deep Research for RecSys」代表什么趋势？

**答**：从**被动过滤**到**主动决策辅助**的范式转变。传统推荐给 item list，用户自行比较；RecPilot 用多智能体框架生成完整的探索-决策报告。技术上，它自回归生成 exploration-to-decision 轨迹（点击→收藏→购买序列），用 Encoder-Decoder 解耦历史编码和未来行为生成。这代表推荐系统和 Agent 技术融合的前沿方向，也预示着推荐系统的输出形式可能从 ranked list 走向结构化报告。

---

## 相关 Synthesis

- [[推荐系统ScalingLaw_Wukong|推荐系统ScalingLaw_Wukong]]
- [[生成式推荐范式统一_20260403|生成式推荐范式统一_20260403]]
- [[SemanticID从论文到Spotify部署|SemanticID从论文到Spotify部署]]
- [[推荐系统冷启动|推荐系统冷启动]]
- [[长序列用户行为建模技术演进|长序列用户行为建模技术演进]]
- [[序列推荐高效注意力与状态空间模型_20260411|序列推荐高效注意力与状态空间模型_20260411]]
- [[精排模型进阶深度解析|精排模型进阶深度解析]]
- [[20260419_generative_reranking_and_ranking_evolution|20260419_generative_reranking_and_ranking_evolution]]
