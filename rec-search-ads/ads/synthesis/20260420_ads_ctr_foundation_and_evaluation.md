# 广告 CTR 基础模型、冷启动探索与评估基准 (2024-2026)

> 10 篇论文综合：Foundation Models for Ads + Cold-Start & Exploration + Evaluation & Benchmarking + Embedding Survey
> 日期：2026-04-20 | MelonEgg Ingest

---

## 总览

本文覆盖广告系统四大前沿方向：

| 主题 | 论文 | 核心贡献 |
|------|------|----------|
| **A: Foundation Models** | CADET, LFM4Ads, Generative CTR | Decoder-Only → 广告 CTR; 基础模型多粒度迁移; 生成式增强判别 |
| **B: Cold-Start & Exploration** | PAM, AuctionUCB-PBM | 流式冷启动 Meta-Learning; 拍卖动态下 MAB 探索 |
| **C: Evaluation & Benchmarking** | Bench-CTR, OPE-Embedding, Parallel Ranking | 统一 CTR 基准; 离线排序策略评估; 广告+创意并行排序 |
| **D: Embedding Survey** | Embedding in RecSys Survey, GACE | 嵌入技术全景; 图嵌入跨页冷启动 |

---

## Theme A: Foundation Models for Ads

### A1. CADET — Decoder-Only Transformer for Ads CTR (LinkedIn, 2026)

**论文**: CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer ([arXiv:2602.11410](https://arxiv.org/abs/2602.11410))

**问题**: 传统 DLRM (Deep Learning Recommendation Model) 采用 encoder + 多塔结构，无法建模用户行为序列内复杂依赖；post-scoring 信号（如广告位置）与 CTR 预测存在"鸡生蛋"问题。

**方法与创新**:

1. **Context-Conditioned Decoding**: 采用 decoder-only 架构 + 多塔预测头，将 ad position 等 post-scoring signal 作为条件输入，解耦排序与位置建模
2. **Self-Gated Attention**: 在 representation-level 和 interaction-level 引入门控机制，稳定大规模 Transformer 训练
3. **Timestamp-based RoPE**: 改进旋转位置编码，用时间戳而非序列位置，捕获秒级到月级的时间关系
4. **Session Masking**: 防止模型学习 in-session 不可获得事件的依赖，消除 train-serve skew
5. **工程优化**: Tensor packing + 自定义 Flash Attention kernel

**核心公式**:

Context-Conditioned CTR:
$$\hat{y} = \sigma(f_{\text{tower}_k}(h_{\text{decoder}} \| e_{\text{position}} \| e_{\text{context}}))$$

Self-Gated Attention:
$$\text{SGA}(Q,K,V) = \text{gate}_r(V) \odot \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} \odot \text{gate}_i(QK^T)\right)V$$

**结果**: 在线 A/B 测试 **+11.04% CTR** 提升 vs LiRank baseline。已部署 LinkedIn 全量。

**工业启示**: Decoder-Only 架构不再局限于 NLP/LLM，在广告 CTR 领域同样有效。关键在于：(1) 处理 post-scoring signals 的条件化设计; (2) session masking 解决 train-serve skew。

---

### A2. LFM4Ads — Large Foundation Model for Ads (Tencent, 2025)

**论文**: Large Foundation Model for Ads Recommendation ([arXiv:2508.14948](https://arxiv.org/abs/2508.14948))

**问题**: 现有基础模型迁移只抽取 User Representation (UR)，忽略 Item Representation (IR) 和 User-Item Cross Representation (CR)；UR 直接作特征无法弥合上下游 gap。

**方法 — All-Representation Multi-Granularity (ARMG) 框架**:

| 迁移粒度 | 方法 | 说明 |
|----------|------|------|
| Feature-level | Non-linear Adapter | UR/IR/CR → 下游特征空间非线性映射 |
| Module-level | Isomorphic Interaction Module | 保持上游交互结构，模块级迁移 |
| Model-level | Standalone Retrieval | 基础模型独立检索，结果融合 |

**结果**:
- 部署于腾讯广告平台（微信朋友圈、视频号等）
- 日处理百亿级样本，TB 级模型参数，数十亿 sparse embedding keys
- Q4 2024 上线后 **GMV 提升 2.45%**，年化增收数亿美元
- 10+ 成功上线场景

**工业启示**: 基础模型迁移不应只迁移 UR → 全表示迁移 (UR+IR+CR) + 多粒度 (feature/module/model) 才能充分利用预训练知识。这是目前工业界最完整的 LFM → Ads 迁移框架。

---

### A3. Generative CTR — 生成式增强判别 CTR (电商搜索广告, 2025)

**论文**: Generative Click-through Rate Prediction with Applications to Search Advertising ([arXiv:2507.11246](https://arxiv.org/abs/2507.11246))

**问题**: 传统判别式 CTR 模型表达能力有限，生成式模型（如 GPT）能否增强 CTR 预测精度？

**方法 — 两阶段训练**:

```
Stage 1: Generative Pre-training
  - Next-item prediction on user behavior sequence
  - Conditional Self-Condition Decoder
  - Conditional Negative Sampling

Stage 2: Discriminative Fine-tuning
  - Parameter Sharing: 共享生成式预训练参数
  - Model Integration: 生成式表示 + 判别式框架融合
```

**四大核心技术**:
1. **Conditional Self-Condition Decoder**: 以 item category 为条件的自回归解码
2. **Conditional Negative Sampling**: 基于条件的负采样策略，提升预训练质量
3. **Parameter Sharing**: 生成 → 判别阶段的参数共享
4. **Model Integration**: 将生成式模型学到的 representation 集成到判别式框架

**结果**: 部署于全球顶级电商平台搜索广告系统，服务数亿日活用户。开源了包含预训练和微调数据的真实流量数据集。

**工业启示**: "生成式增强判别式"是 CTR 领域重要范式 — 先用生成模型学行为序列中的 pattern，再迁移到判别模型。与 [[generative_recsys]] 中的趋势一致。

---

### Foundation Models 对比

| 维度 | CADET | LFM4Ads | Generative CTR |
|------|-------|---------|----------------|
| 架构 | Decoder-Only Transformer | Multi-Granularity Transfer | Generative Pre-train + Discriminative Fine-tune |
| 核心创新 | Context-Conditioned + Self-Gated Attn | UR+IR+CR 全表示迁移 | Conditional Self-Condition Decoder |
| 位置建模 | Timestamp RoPE | N/A | N/A |
| 部署方 | LinkedIn | 腾讯 (微信/视频号) | 顶级电商 |
| 在线收益 | +11.04% CTR | +2.45% GMV (年化数亿$) | 数亿 DAU |
| 范式 | End-to-End Decoder | Foundation → Downstream Transfer | Gen → Disc Transfer |

---

## Theme B: Cold-Start & Exploration

### B1. PAM — Popularity-Aware Meta-Learning for Cold-Start (KDD 2025)

**论文**: Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning ([arXiv:2411.11225](https://arxiv.org/abs/2411.11225))

**问题**: 流式数据场景下，传统冷启动方案（fine-tuning / knowledge transfer）计算开销大、不可行。新物品曝光不足导致行为特征缺失。

**方法**:

1. **Popularity-Based Task Division**: 按 item 热度阈值将流式数据划分为不同 meta-learning 任务
2. **Feature Reweighting**: 在不同热度任务中，区分并重新加权行为特征（高热度 → 行为特征重要）和内容特征（低热度 → 内容特征重要）
3. **Data Augmentation for Low-Popularity Tasks**: 利用高热度样本的 insight 增强低热度任务的训练
4. **Self-Supervised Loss**: 专为低热度任务设计的自监督损失，缓解监督信号不足

**核心思想**:

$$\mathcal{L}_{\text{PAM}} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_{\text{task}_k} + \lambda \cdot \mathcal{L}_{\text{SSL}}$$

其中 $k$ 为热度分桶，$w_k$ 为热度自适应权重，$\mathcal{L}_{\text{SSL}}$ 为自监督辅助损失。

**结果**:

| 数据集 | Recall@5 提升 | NDCG@5 提升 |
|--------|--------------|-------------|
| MovieLens | +20.06% ~ +64.51% | +20.23% ~ +74.09% |
| Yelp | 同上 | 同上 |
| Book | 同上 | 同上 |

**工业启示**: PAM 是 model-agnostic 的，可直接插入现有推荐/广告系统。关键 insight: **热度分桶 → 特征重要性自适应** 比统一的 meta-learning 更有效。与 [[embedding_everywhere]] 中的冷启动嵌入策略互补。

---

### B2. AuctionUCB-PBM — MAB under Auction Dynamics (2025)

**论文**: Optimizing Online Advertising with Multi-Armed Bandits: Mitigating the Cold Start Problem under Auction Dynamics ([arXiv:2502.01867](https://arxiv.org/abs/2502.01867))

**问题**: 新广告 CTR 预估不准（冷启动），但广告系统是拍卖机制 (auction)，探索成本不同于标准 MAB — 展示低 CTR 广告会损失拍卖收入。

**方法**:

1. **Position-Based Model (PBM)**: 将 CTR 分解为 $\text{CTR} = \text{attractiveness} \times \text{examination\_prob}(position)$
2. **UCB-like Algorithm for Auction**: 在 PBM 下设计 UCB 变体，考虑拍卖动态（竞价、位置分配）
3. **Budget Regret Bound**: 理论推导了 budget regret 上界
4. **Controlled Exploration-Exploitation**: 保证探索的同时维持短期收益

**核心公式**:

$$\text{UCB}_i(t) = \hat{\mu}_i(t) + \sqrt{\frac{2\ln t}{n_i(t)}} \cdot f(\text{auction\_dynamics})$$

其中 $f(\text{auction\_dynamics})$ 调整探索幅度以适应拍卖机制。

**结果**: 在合成数据和真实广告平台数据上验证，提升平台长期总收入，同时通过控制探索维持短期利润。

**工业启示**: 在广告拍卖场景下做探索 ≠ 标准 MAB。必须考虑：(1) 位置效应 (PBM); (2) 竞价机制对收入的影响; (3) 短期 vs 长期收益平衡。

---

### Cold-Start & Exploration 对比

| 维度 | PAM | AuctionUCB-PBM |
|------|-----|----------------|
| 场景 | 推荐/广告物品冷启动 | 广告平台新广告探索 |
| 方法 | Meta-Learning + 热度分桶 | UCB + 拍卖动态 |
| 理论保证 | 无（实验验证） | 有（Budget Regret Bound） |
| 核心 Insight | 热度→特征重要性自适应 | 拍卖成本感知的探索 |
| 适用性 | Model-agnostic | 拍卖 PPC 系统 |

---

## Theme C: Evaluation & Benchmarking

### C1. Bench-CTR — 统一 CTR 预测基准 (2025)

**论文**: Toward a Benchmark for CTR Prediction in Online Advertising ([arXiv:2512.01179](https://arxiv.org/abs/2512.01179))

**问题**: CTR 模型论文各自用不同数据集、指标、实验设置，缺乏统一基准，难以公平比较。

**贡献**:

1. **统一平台架构**: Bench-CTR 提供灵活接口，统一数据集加载和模型评估
2. **评估协议体系**: 涵盖 3 个公开数据集 + 2 个合成数据集，标准化指标分类和实验流程
3. **横跨传统到 LLM**: 评估从多变量统计到 LLM-based 的全系列模型

**三大发现**:

| 发现 | 含义 |
|------|------|
| 高阶模型 > 低阶模型 | 特征交叉阶数重要，但优势因数据集/指标而异 |
| LLM 模型数据效率极高 | **仅 2% 训练数据** 即达可比性能 |
| 2015-2016 后进展放缓 | CTR 模型性能提升进入瓶颈期 |

**代码**: [GitHub - Bench-CTR](https://github.com/NuriaNinja/Bench-CTR)

**工业启示**: (1) LLM-based CTR 模型可能是小数据/冷启动场景的突破口; (2) CTR 领域"军备竞赛"回报递减，应更关注系统工程和在线效果。

---

### C2. OPE-Embedding — 离线排序策略评估 (2025)

**论文**: Off-Policy Evaluation of Ranking Policies via Embedding-Space User Behavior Modeling ([arXiv:2506.00446](https://arxiv.org/abs/2506.00446))

**问题**: 排序策略的离线评估 (OPE) 面临大 action space 问题 — unique actions 数量大 + ranking 长度大 → 方差爆炸。

**方法**:

1. **Embedding-Space User Behavior Modeling**: 将用户行为建模从离散 action space 映射到连续 embedding space
2. **Large Action Space 处理**: 通过 embedding 降维缓解 IPS (Inverse Propensity Score) 方差问题
3. **Ranking-Aware Estimation**: 考虑排序位置对用户行为的影响

**核心思想**: 传统 OPE 的 IPS 估计在大 action space 下方差极大：

$$\hat{V}(\pi) = \frac{1}{n}\sum_{i=1}^n \frac{\pi(a_i|x_i)}{\pi_0(a_i|x_i)} r_i$$

通过 embedding space 映射，将 action $a$ 映射为 $e(a)$，在低维空间做 importance weighting，显著降低方差。

**工业启示**: 广告排序策略上线前必须做 OPE，embedding-space 方法为大规模广告系统提供了可行的离线评估方案。与 [[embedding_everywhere]] 中的 embedding 应用场景互补。

---

### C3. Parallel Ranking — 广告+创意并行排序 (AAAI 2024)

**论文**: Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems ([arXiv:2312.12750](https://arxiv.org/abs/2312.12750))

**问题**: AIGC 时代广告主可低成本生成大量创意素材，但传统系统串行排序 Ads → Creatives，创意模块受限于效果和效率。

**方法**:

```
传统 Serial:  Ads Ranking → Top-K Ads → Creative Ranking → Display
本文 Parallel: Ads Ranking ──┐
                              ├── Joint CTR Model → Display
               Creative Ranking┘
```

1. **Online Parallel Architecture**: 广告排序和创意排序并行执行，降低总延迟
2. **Offline Joint Optimization Model**: 广告和创意互感知 (mutual awareness)，协同优化 CTR
3. **Implicit Feedback Sorting Optimization**: 优化创意排序中的隐式反馈评估指标

**结果**: 离线和在线均验证了 response time、CTR、CPM 的提升。

**工业启示**: AIGC 创意爆发 → 系统架构必须从串行→并行。这不仅是工程优化，joint model 能捕获 ad-creative 交互信号提升 CTR。

---

### Evaluation 对比

| 维度 | Bench-CTR | OPE-Embedding | Parallel Ranking |
|------|-----------|---------------|------------------|
| 关注点 | 模型公平比较 | 离线策略评估 | 系统架构优化 |
| 方法类型 | 基准平台 | 统计估计 | 在线+离线联合 |
| 核心发现 | LLM 2%数据可比 | Embedding降方差 | 并行降延迟提CTR |
| 场景 | 通用 CTR | 排序策略迭代 | AIGC创意+广告 |

---

## Theme D: Embedding Survey 精华

### D1. Embedding in Recommender Systems: A Survey (2023-2025)

**论文**: Embedding in Recommender Systems: A Survey ([arXiv:2310.18608](https://arxiv.org/abs/2310.18608))

**全景分类**:

```
Embedding in RecSys
├── Matrix-based (协同过滤)
│   ├── MF / SVD / ALS
│   └── Neural CF (NCF, DeepFM)
├── Sequential (序列建模)
│   ├── RNN-based (GRU4Rec)
│   ├── Transformer-based (SASRec, BERT4Rec)
│   └── Self-Supervised (对比学习, 生成式)
├── Graph-based (图结构)
│   ├── node2vec, LINE
│   ├── GNN (GCN, GAT, GraphSAGE)
│   └── Knowledge Graph Embedding
├── Optimization (效率优化)
│   ├── AutoML (NAS for Embedding Dim)
│   ├── Hashing (LSH, Learning-to-Hash)
│   └── Quantization (PQ, VQ)
└── LLM-Enhanced
    ├── LLM as Encoder (text→embedding)
    ├── LLM as Feature Extractor
    └── Unified Generative Embedding
```

**关键洞察**:

| 趋势 | 说明 |
|------|------|
| Embedding Dim 选择 | AutoML (NAS) > 手动调参，不同特征应有不同维度 |
| 压缩技术 | Hashing 适合极大规模; Quantization (PQ/VQ) 适合相似性搜索 |
| LLM 增强 | 文本特征 embedding 提升 10-20%，但推理开销大 |
| 冷启动 | Graph-based embedding (如 GACE) 是冷启动最有效方案之一 |

**与 [[embedding_everywhere]] 的关系**: 本 survey 提供了更系统的分类和最新进展（特别是 LLM-Enhanced 部分），应更新概念页。

---

### D2. GACE — 图嵌入跨页广告冷启动 (Alipay, ICONIP 2023)

**论文**: GACE: Learning Graph-Based Cross-Page Ads Embedding For Click-Through Rate Prediction ([arXiv:2401.07445](https://arxiv.org/abs/2401.07445))

**问题**: (1) 多页面广告数据联合利用; (2) 新广告冷启动。

**方法**:

1. **Cross-Page Graph Construction**: 考虑语义和页面类型属性，构建加权无向图
2. **VAE Pre-training**: 设计变分自编码 (VAE) 预训练模块，为新旧广告生成 embedding
3. **Feature Fusion with Graph Guidance**: 图结构引导特征融合方向

**结果 (Alipay A/B Test)**:

| 指标 | 页面1 | 页面2 | 页面3 |
|------|-------|-------|-------|
| 整体 CTR 提升 | +3.6% | +2.13% | +3.02% |
| 冷启动 CTR 提升 | **+9.96%** | **+7.51%** | **+8.97%** |

**工业启示**: 图嵌入在广告冷启动场景效果显著（冷启动 CTR 提升 7-10%）。关键是跨页面图构建 — 将不同广告位的数据联合建模。

---

## 交叉主题分析

### 1. Foundation Model 在广告中的三条路径

```
路径1: End-to-End (CADET)
  输入→Decoder-Only Transformer→CTR
  优势: 统一建模，捕获复杂依赖
  代价: 需要从头训练/适配

路径2: Transfer Learning (LFM4Ads)
  预训练 LFM → UR+IR+CR 迁移 → 下游 CTR
  优势: 利用预训练知识，多场景复用
  代价: 需要维护预训练模型

路径3: Generative Enhancement (Generative CTR)
  生成式预训练 → 判别式微调
  优势: 捕获行为序列中的生成式 pattern
  代价: 两阶段训练复杂度
```

### 2. 冷启动方法谱

```
特征补全: GACE (图嵌入生成)
行为迁移: PAM (热度感知 meta-learning)
探索策略: AuctionUCB-PBM (拍卖感知 UCB)
基础模型: LFM4Ads IR 迁移 (item 表示)
```

### 3. Embedding 技术在广告全链路的应用

| 链路阶段 | Embedding 应用 | 代表方法 |
|----------|---------------|----------|
| 召回 | Item/User Embedding | Embedding Survey 全景 |
| 粗排 | Cross-Page Embedding | GACE |
| 精排 | Decoder Embedding | CADET |
| 创意选择 | Creative Embedding | Parallel Ranking |
| 离线评估 | Action Embedding for OPE | OPE-Embedding |
| 冷启动 | Graph/Meta Embedding | GACE + PAM |

---

## 面试 Q&A (8 题)

### Q1: Decoder-Only Transformer 用于广告 CTR 预估有什么优势？和传统 DLRM 的区别？

**A**: CADET (LinkedIn 2026) 展示了 Decoder-Only 在 CTR 的优势：
1. **统一序列建模**: 用户行为序列 + 候选广告统一编码，而 DLRM 需要手工设计特征交叉
2. **自回归因果注意力**: 天然建模时序依赖，避免信息泄露
3. **Context-Conditioned**: 通过条件化多塔头处理 post-scoring signal（如 position），解决 "chicken-and-egg" 问题
4. **结果**: +11.04% CTR 提升

区别: DLRM = embedding + MLP + 特征交叉; CADET = 序列化输入 + Decoder-Only + 条件化预测头。

---

### Q2: 基础模型迁移到广告推荐时，只迁移 User Representation 有什么问题？

**A**: LFM4Ads (Tencent) 指出三大问题：
1. **信息不完整**: 忽略 Item Representation (IR) 和 User-Item Cross Representation (CR)，丢失物品侧知识和交互模式
2. **上下游 Gap**: UR 直接作特征缺乏适配，预训练目标与下游目标不对齐
3. **迁移粒度单一**: 只在 feature-level 迁移，忽略 module-level 和 model-level 的迁移可能

解法: All-Representation (UR+IR+CR) + Multi-Granularity (feature/module/model) 迁移，在线 GMV +2.45%。

---

### Q3: 如何在流式数据场景下处理物品冷启动？

**A**: PAM (KDD 2025) 提出 Popularity-Aware Meta-Learning:
1. 按 item 热度将流式数据分桶为 meta-learning 任务
2. 高热度任务: 行为特征权重高; 低热度任务: 内容特征权重高
3. 高→低跨任务数据增强 + 自监督辅助损失
4. Model-agnostic，直接嵌入现有系统

关键 insight: **热度是特征重要性的天然代理变量** — 越冷的物品越依赖内容特征。

---

### Q4: 广告系统中做 MAB 探索和标准 MAB 有什么区别？

**A**: AuctionUCB-PBM 指出广告拍卖场景的特殊性：
1. **拍卖成本**: 展示低 CTR 广告损失竞价收入，探索成本非均匀
2. **位置效应 (PBM)**: CTR = attractiveness × examination_prob(pos)，position 影响 CTR 估计
3. **短期 vs 长期**: 需同时维持短期拍卖收入和长期 CTR 准确性
4. **理论保证**: 推导 Budget Regret Bound，平衡探索收益与竞价损失

---

### Q5: CTR 模型发展到现在，还有多少提升空间？

**A**: Bench-CTR 给出了令人深思的发现：
1. CTR 模型性能在 2015-2016 后进入**缓慢提升期**，跨数据集一致
2. 高阶特征交叉仍然重要，但边际递减
3. **LLM-based 模型仅用 2% 训练数据即达可比性能** — 数据效率是新突破方向
4. 启示: CTR 领域应从"模型创新"转向"系统工程+数据效率+在线优化"

---

### Q6: 如何不上线就评估新的排序策略？

**A**: OPE-Embedding 提出 Embedding-Space 方法：
1. 传统 IPS 在大 action space 下方差爆炸
2. 将 action 映射到 embedding space，在低维空间做 importance weighting
3. Ranking-aware: 考虑位置对用户行为的影响
4. 实际价值: 广告系统 A/B 测试成本极高，OPE 能在离线筛选 80% 不靠谱的策略

---

### Q7: AIGC 时代广告系统架构如何适应创意爆发？

**A**: Parallel Ranking (AAAI 2024) 提出并行架构：
1. **问题**: 传统串行 Ads→Creatives，创意模块受瓶颈限制
2. **方案**: 广告排序和创意排序并行执行，联合优化 CTR
3. **Joint Model**: 广告和创意互感知，捕获 ad-creative 交互
4. **收益**: 降低延迟 + 提升 CTR + 提升 CPM

---

### Q8: 广告冷启动为什么图嵌入方法效果好？

**A**: GACE (Alipay, ICONIP 2023) 展示了图嵌入的优势：
1. **跨页面信息传播**: 不同广告位的数据通过图结构关联，新广告可借邻居信息
2. **VAE 生成**: 为完全无曝光的新广告生成 embedding，而非用零向量
3. **语义+结构双通道**: 图边权重同时考虑语义相似和页面类型
4. **效果**: 冷启动 CTR 提升 **7-10%**，远超 warm-up 阶段整体 CTR 提升的 2-3%

---

## 参考文献

1. CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer. LinkedIn, 2026. [arXiv:2602.11410](https://arxiv.org/abs/2602.11410)
2. Large Foundation Model for Ads Recommendation. Tencent, 2025. [arXiv:2508.14948](https://arxiv.org/abs/2508.14948)
3. GACE: Learning Graph-Based Cross-Page Ads Embedding For CTR Prediction. Alipay, ICONIP 2023. [arXiv:2401.07445](https://arxiv.org/abs/2401.07445)
4. Generative Click-through Rate Prediction with Applications to Search Advertising. 2025. [arXiv:2507.11246](https://arxiv.org/abs/2507.11246)
5. Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning (PAM). KDD 2025. [arXiv:2411.11225](https://arxiv.org/abs/2411.11225)
6. Optimizing Online Advertising with Multi-Armed Bandits under Auction Dynamics. 2025. [arXiv:2502.01867](https://arxiv.org/abs/2502.01867)
7. Toward a Benchmark for CTR Prediction in Online Advertising (Bench-CTR). 2025. [arXiv:2512.01179](https://arxiv.org/abs/2512.01179)
8. Off-Policy Evaluation of Ranking Policies via Embedding-Space User Behavior Modeling. 2025. [arXiv:2506.00446](https://arxiv.org/abs/2506.00446)
9. Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems. AAAI 2024. [arXiv:2312.12750](https://arxiv.org/abs/2312.12750)
10. Embedding in Recommender Systems: A Survey. 2023-2025. [arXiv:2310.18608](https://arxiv.org/abs/2310.18608)

---

> 相关概念页: [[embedding_everywhere]] | [[attention_in_recsys]] | [[generative_recsys]] | [[sequence_modeling_evolution]]
> 相关 synthesis: [[01_CTR_CVR预估与校准全景]] | [[06_冷启动与偏差治理]] | [[08_广告创意优化]] | [[05_竞价与预算优化]]
