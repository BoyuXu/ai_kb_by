# Scaling Law + Cold-Start + CTR + 生成式推荐：10 篇前沿论文综合精读

> **日期**：2026-05-04
> **覆盖论文**：ULTRA-HSTU / LUM / MixFormer / ApEn Scaling / EmerG / IDProxy / Sparse Contrastive / DTN / CETNet / GR-LLMs Survey
> **四大主题**：(1) Scaling Law in Rec-Sys (2) Cold-Start with Modern Methods (3) CTR/Feature Interaction Evolution (4) Generative Recommendation Survey

**相关概念页**：[[推荐中的注意力机制]] | [[序列建模演进]] | [[生成式推荐]] | [[Embedding无处不在]] | [[多目标优化]]
**相关 synthesis**：[[20260420_scaling_laws_and_cold_start]] | [[20260503_scaling_sequence_multitask_frontier]] | [[20260420_feature_interaction_ctr_advances]] | [[20260414_LLM驱动生成式推荐前沿]]

---

## Section 1: Scaling Law in Rec-Sys

### 1.0 背景：从 LLM Scaling Law 到推荐系统

LLM 领域的 Scaling Laws（Kaplan 2020, Chinchilla 2022）已确立 $L(N,D) \propto N^{-\alpha} + D^{-\beta}$ 的范式。推荐系统能否复制？2024 年 HSTU（Meta）首次证明推荐也有 Scaling Law，但瓶颈在于：

1. **序列长度爆炸**：用户行为序列可达 10K-100K，self-attention $O(n^2)$ 不可承受
2. **特征异构性**：推荐不仅有序列，还有稠密特征、稀疏特征、交叉特征
3. **数据质量 vs 数据量**：推荐数据噪声大，单纯增加数据量未必有效
4. **工程约束**：推荐系统对延迟敏感（P99 < 50ms），不能像 LLM 随意增大模型

以下四篇论文分别从**模型效率**、**预训练范式**、**统一架构**和**数据质量度量**四个角度突破。

### 1.1 ULTRA-HSTU: Bending the Scaling Law Curve (Meta, arxiv 2602.16986)

| 维度 | 内容 |
|------|------|
| **问题** | HSTU 在超长序列上 scaling 效率低，self-attention $O(n^2)$ 无法扩展到 16K+ 序列 |
| **方法** | 端到端 model-system co-design，借鉴 DeepSeek-V2 的 MLA 思想 |
| **部署** | 18 层 self-attention，16K 用户行为序列，数百张 H100 训练，服务数十亿用户 |

**三大创新轴：**

```
ULTRA-HSTU = Input Sequence Optimization + Sparse Attention + Dynamic Topology

(1) Input Sequence Optimization
    - 对原始行为序列做源头压缩，减少有效序列长度
    - 类似 LLM 的 prompt compression，但针对推荐场景设计

(2) Recommender-Tailored Sparse Attention
    - 不同于 DeepSeek NSA 只用 local window
    - 同时设计 local window + global window
    - 长期行为对推荐至关重要 → global attention 不可省略
    - 实现线性复杂度 O(n)

(3) Dynamic Topological Design
    - 不是每层都处理全序列
    - 有利的深度 scaling 而不付出全序列代价
```

**Scaling 效率对比：**

$$\text{ULTRA-HSTU Speedup} = \begin{cases} 5\times & \text{Training scaling} \\ 21\times & \text{Inference scaling} \end{cases}$$

**核心洞察**：推荐系统的 scaling law 曲线是可以"弯曲"的——通过 model-system co-design，在相同计算预算下获得更陡的性能增长斜率。这与 LLM 领域追求 compute-optimal（Chinchilla）不同，推荐更关注 latency-optimal。

### 1.2 Large User Model (LUM): Three-Step Paradigm (Alibaba, arxiv 2502.08309)

| 维度 | 内容 |
|------|------|
| **问题** | 端到端生成式推荐牺牲了传统 DLRM 的成熟工程优势（特征工程、模块化、线上优化实践） |
| **方法** | 三步范式：Knowledge Construction → Knowledge Querying → Knowledge Utilization |
| **Scaling** | 模型可扩展到 7B 参数，性能持续提升 |

**三步范式详解：**

```
Step 1: Knowledge Construction (知识构建)
  - Transformer 架构 + 生成式预训练
  - 目标：捕获用户兴趣 + 物品协同关系
  - 类似 LLM 的 next-token prediction，但 token = 用户行为

Step 2: Knowledge Querying (知识查询)
  - 用预定义问题（prompts）查询 LUM
  - 提取用户特定信息（兴趣偏好、生命周期阶段等）
  - 类比 LLM 的 in-context learning

Step 3: Knowledge Utilization (知识利用)
  - LUM 输出作为补充特征注入传统 DLRM
  - 不替代现有系统，而是增强
  - 工程友好：增量接入，无需重构
```

**关键公式**（Scaling Law 验证）：

$$\text{Performance}(N) = a \cdot N^{-\alpha} + c, \quad N \in [100M, 7B]$$

其中 $N$ 是 LUM 参数量，$\alpha > 0$ 表明推荐领域确实存在 power-law scaling。

**工业启示**：LUM 的三步范式是一种"安全"的大模型落地路径——不需要替换整个推荐系统，而是将大模型作为特征增强器。这对工业落地极其友好。

### 1.3 MixFormer: Co-Scaling Dense and Sequence (ByteDance, arxiv 2602.14110)

| 维度 | 内容 |
|------|------|
| **问题** | 现有 Transformer 推荐模型中，序列建模和特征交互是分离的模块，存在 co-scaling 难题 |
| **方法** | 统一 Transformer 架构，joint modeling 序列行为 + 特征交互 |
| **部署** | 抖音 + 抖音极速版线上 A/B，提升活跃天数和使用时长 |

**核心 Co-Scaling 问题：**

```
传统架构（Decoupled Design）:
  ┌─────────────────┐  ┌─────────────────┐
  │  Sequence Model  │  │ Feature Interact │
  │  (Transformer)   │  │    (DCN/DNN)     │
  └────────┬────────┘  └────────┬────────┘
           └────────┬───────────┘
                  Concat

  问题：计算预算在两个模块间不可最优分配

MixFormer（Unified Design）:
  ┌─────────────────────────────┐
  │   Unified Transformer       │
  │   - Sequence tokens         │
  │   - Feature tokens          │
  │   - Cross-modal attention   │
  └─────────────────────────────┘

  优势：统一参数化，co-scaling 无需手工分配预算
```

**User-Item Decoupling（推理加速）：**
- 非序列特征分成 user-side ($N_U$ heads) 和 item-side ($N_G$ heads)
- Causal mask 设计：user-side heads 可跨请求复用
- 显著降低推理冗余计算和延迟

**与 ULTRA-HSTU 对比：**

| 维度 | ULTRA-HSTU | MixFormer |
|------|-----------|-----------|
| 解决问题 | 序列长度 scaling | 稠密+序列 co-scaling |
| 稀疏注意力 | Local+Global sparse | 统一 attention |
| 特征交互 | 序列内隐式 | 显式 joint modeling |
| 部署 | Meta（数十亿用户） | ByteDance 抖音 |
| 设计哲学 | System co-design | Architecture co-design |

### 1.4 Approximate Entropy Scaling Law for Sequential Recommendation (arxiv 2412.00430)

| 维度 | 内容 |
|------|------|
| **问题** | 传统 scaling law 只看数据量 $D$，忽略数据质量差异 |
| **方法** | 引入 Approximate Entropy (ApEn) 替代数据量作为 scaling law 中的数据因子 |
| **核心发现** | $D/\text{ApEn}$ 比单独 $D$ 更准确预测模型性能 |

**Performance Law（推荐版 Scaling Law）：**

$$\text{HR@K}(N, D, \text{ApEn}) = a \cdot \left(\frac{D}{\text{ApEn}}\right)^{-\alpha} + b \cdot N^{-\beta} + c$$

**Approximate Entropy (ApEn)**：
- 统计度量，量化时间序列数据的规律性和不可预测性
- ApEn 高 → 用户行为随机性强 → 数据"质量"低
- ApEn 低 → 用户行为有规律 → 数据"质量"高、更容易建模

**核心洞察**：
1. 推荐数据不像文本语料，质量差异极大（bot 流量、随机点击 vs 真实兴趣）
2. 用 $D/\text{ApEn}$ 作为有效数据量，比直接用样本数更精准
3. 为推荐系统的数据筛选提供了理论指导：应优先收集低 ApEn（高规律性）的用户行为

### 1.5 Scaling Law 主题总结

```
技术演进路线：
  HSTU (2024) — 首次证明推荐有 Scaling Law
  → LUM (2025) — 三步范式安全落地 7B 参数
  → ULTRA-HSTU (2026) — model-system co-design 弯曲 scaling 曲线
  → MixFormer (2026) — 统一架构解决 co-scaling
  → ApEn Scaling (2024) — 数据质量维度补充

核心规律：
  Performance ∝ f(Compute, DataQuality, ArchEfficiency)
  而不仅仅是 f(N, D)
```

**面试 Q&A 要点：**

**Q: 推荐系统的 Scaling Law 和 LLM 的有什么区别？**
A: 三大区别：(1) 推荐数据质量差异大，需要 ApEn 等指标补充数据量；(2) 推荐有严格延迟约束，不能无限放大模型，需要 latency-optimal 而非 compute-optimal；(3) 推荐特征异构（序列+稠密+稀疏），scaling 需要 co-scaling 策略。

**Q: 如何在工业推荐系统中实践 Scaling Law？**
A: 两条路径：(1) ULTRA-HSTU 路径——model-system co-design，通过稀疏注意力和硬件协同设计在固定延迟下扩大模型；(2) LUM 路径——大模型做特征增强器，不替代现有 DLRM 架构，增量接入更安全。

**Q: MixFormer 的 co-scaling 问题本质是什么？**
A: 在固定计算预算下，序列建模和特征交互模块独立参数化导致预算分配不可联合优化。MixFormer 通过统一 Transformer 架构消除这一瓶颈，让模型自动学习最优分配。

---

## Section 2: Cold-Start with Modern Methods

### 2.0 冷启动问题的本质

冷启动的核心矛盾：**CTR 模型依赖 ID Embedding，但新物品没有足够的行为数据来学习好的 Embedding**。

```
传统方法：
  (1) 全局默认 embedding — 无法区分新物品
  (2) 基于 side info 的 embedding 初始化 — 信息损失大
  (3) Meta-learning — 训练复杂，工业落地难

2024-2026 新方向：
  (1) EmerG — 用 HyperNetwork + GNN 学习 item-specific 特征交互图
  (2) IDProxy — 用多模态 LLM 生成 proxy embedding
  (3) Sparse Contrastive — 纯内容建模 + 稀疏对比学习
```

### 2.1 EmerG: Item-Specific Feature Interactions (KDD 2024, arxiv 2407.10112)

| 维度 | 内容 |
|------|------|
| **问题** | 现有冷启动方法用全局特征交互模式，稀疏新物品被丰富物品"淹没" |
| **方法** | HyperNetwork 生成 item-specific 特征图 → GNN 捕获任意阶特征交互 |
| **训练** | Meta-learning 策略，跨任务优化 HyperNet + GNN 参数 |

**架构解析：**

```
Item Features → HyperNetwork → Item-Specific Feature Graph
                                        ↓
                               Customized GNN (Message Passing)
                                        ↓
                               任意阶特征交互表示
                                        ↓
                                   CTR Prediction
```

**核心创新：**
1. **Item-Specific 特征图**：每个物品有自己的特征交互模式，而非全局共享
2. **GNN 消息传递**：通过定制化消息传递机制，可证明捕获任意阶特征交互
3. **Meta-Learning**：在不同物品的 CTR 预测任务间优化，仅调整最小化的 item-specific 参数，避免稀疏数据下过拟合

**特征交互阶数的可证明性：**

$$\text{GNN}^{(k)}(v) = \sigma\left(\sum_{u \in \mathcal{N}(v)} W^{(k)} \cdot h_u^{(k-1)}\right)$$

$k$ 层 GNN 等价于捕获 $k$ 阶特征交互，这是形式化可证明的。

### 2.2 IDProxy: Multimodal LLM for Cold-Start (Xiaohongshu, arxiv 2603.01590)

| 维度 | 内容 |
|------|------|
| **问题** | 新物品无 ID embedding，传统初始化方法信息有限 |
| **方法** | 多模态 LLM (MLLM) 从内容信号生成 proxy embedding，对齐到 ID 空间 |
| **部署** | 小红书 Explore Feed（内容推荐 + 展示广告），服务数亿用户 |

**两阶段 Coarse-to-Fine 框架：**

```
Stage 1: Coarse Proxy Generation
  ┌──────────────┐
  │ Item Content  │ → MLLM → Coarse Proxy Embedding
  │ (图片+文本)   │         (从多模态内容提取语义)
  └──────────────┘

Stage 2: End-to-End Alignment
  ┌──────────────┐     ┌──────────────┐
  │ MLLM Hidden  │ ──→ │ Alignment    │ ──→ Refined Proxy
  │ States       │     │ Module       │     (对齐到 ID 空间)
  └──────────────┘     └──────────────┘
                              ↕
                    CTR Ranker (Joint Optimization)
```

**关键设计决策：**
1. **为什么用 MLLM 而非简单内容 encoder？** 小红书是图文内容平台，MLLM 能理解图文联合语义
2. **为什么对齐到 ID 空间？** 现有 ranker 的所有特征交互都建立在 ID embedding 之上，proxy 必须与 ID 空间兼容
3. **为什么 end-to-end？** 分离训练会导致分布偏移（distribution shift），联合优化可消除

**与 EmerG 对比：**

| 维度 | EmerG | IDProxy |
|------|-------|---------|
| 信息源 | 结构化 side info | 多模态内容（图片+文本） |
| 技术路线 | HyperNet + GNN + Meta-Learning | MLLM + Embedding Alignment |
| 核心思想 | 学习 item-specific 交互模式 | 生成 proxy embedding 替代 ID |
| 适用场景 | 通用电商/新闻推荐 | 内容平台（图文/视频） |
| 工程复杂度 | 中等（需要 meta-learning） | 高（需要 MLLM 推理） |

### 2.3 Sparse Contrastive Learning for Cold Item Recommendation (arxiv 2604.12990)

| 维度 | 内容 |
|------|------|
| **问题** | 将 cold item 内容映射到 CF embedding 空间存在根本性信息鸿沟 |
| **方法** | 纯内容建模，用稀疏对比学习训练内容 encoder |
| **创新** | $\alpha$-entmax 替代 softmax，实现稀疏相似度估计 |

**核心思路转变：**

```
传统冷启动:  Content → Alignment → CF Embedding Space
  问题: CF 信号和内容特征之间存在根本性信息鸿沟

Sparse Contrastive:  Content → Sparse CL → Item-Item Similarity Space
  思路: 不做 CF 对齐，直接学习"内容相似 → 用户偏好相似"的映射
```

**$\alpha$-entmax 稀疏激活：**

$$p_i = \alpha\text{-entmax}(z_i) = \arg\max_{p \in \Delta^{K-1}} \left[ p^\top z - H_\alpha(p) \right]$$

其中 $H_\alpha(p) = \frac{1}{\alpha(\alpha-1)} \sum_i (p_i - p_i^\alpha)$ 是 Tsallis $\alpha$-entropy。

**关键性质**：当 $\alpha > 1$ 时，$\alpha$-entmax 可以输出精确的零概率（sparse），而 softmax 永远输出正值。这意味着：
- 不相关的 negative samples 可以得到精确的零梯度
- 避免了 uninformative negatives 对训练的干扰
- 更 sharp 的 item 相关性估计

### 2.4 Cold-Start 主题总结

```
方法演进：
  Global Embedding → Side Info Init → Meta-Learning (EmerG)
    → MLLM Proxy (IDProxy) → Pure Content CL (Sparse Contrastive)

趋势：
  (1) 从"修补 ID embedding"到"重新定义 item 表示"
  (2) 多模态大模型开始进入冷启动场景（IDProxy）
  (3) 对比学习 + 稀疏化是纯内容路线的新方向
```

**面试 Q&A 要点：**

**Q: 冷启动有哪些技术路线？各自的优缺点？**
A: 四条路线：(1) Meta-learning（EmerG）——学习如何从少量数据快速适配，通用性强但训练复杂；(2) MLLM Proxy（IDProxy）——利用多模态大模型理解内容语义生成 proxy embedding，效果好但推理开销大；(3) 对比学习（Sparse CL）——纯内容建模，不依赖 CF 信号，轻量但受限于内容信息量；(4) 传统 side info 初始化——简单但效果有限。

**Q: IDProxy 为什么要两阶段而不直接端到端？**
A: Coarse 阶段提供一个合理的初始化（MLLM 的内容理解能力），Fine 阶段通过 CTR 目标对齐到 ID 空间。直接端到端从随机初始化开始，MLLM 的梯度信号太弱，容易陷入局部最优。两阶段是 coarse-to-fine 的标准做法。

**Q: 为什么 Sparse Contrastive Learning 不对齐到 CF 空间？**
A: 因为 CF embedding 和内容特征之间存在根本性信息鸿沟——CF 编码的是协同过滤信号（谁和谁一起被消费），内容编码的是语义信息。强行对齐会丢失内容特征的独特价值。不如直接在内容空间学习 item-item 相似度。

---

## Section 3: CTR / Feature Interaction Evolution

### 3.0 特征交互的演进脉络

```
第一代：手工交叉
  LR + 人工特征交叉 → 工程量大但可解释

第二代：自动低阶交叉
  FM (二阶) → FFM (field-aware) → AFM (attention-weighted)

第三代：高阶交叉
  DCN (cross network) → xDeepFM (compressed interaction) → AutoInt

第四代：DNN 隐式交叉 + 显式交叉融合
  DeepFM → DCN-V2 → DHEN (hierarchical ensemble)

第五代（2024-2026）：任务感知 + 协同集成
  DTN (task-specific interactions) → CETNet (collaborative ensemble)

关键转变：从"如何交叉"到"为谁交叉"（task-specific）和"如何协同"（ensemble）
```

### 3.1 DTN: Deep Multiple Task-specific Feature Interactions (arxiv 2408.11611)

| 维度 | 内容 |
|------|------|
| **问题** | 现有 MTL 模型（MMoE, PLE）忽略了特征交互的优化，且不同任务对同一特征的重要性不同 |
| **方法** | 多种多样化的 task-specific 特征交互方法 + task-sensitive network |
| **部署** | 电商推荐，63 亿样本，CTR +3.28%，订单 +3.10%，GMV +2.70% |

**核心观察：**

> 同一特征在不同任务中的重要性可能显著不同。例如，"价格"对 CTR 影响小但对 CVR 影响大；"标题吸引力"对 CTR 影响大但对购买转化影响小。

**架构设计：**

```
Input Features
     ↓
┌────────────────────────────────────────┐
│  Multiple Diversified Feature          │
│  Interaction Methods                   │
│  ┌─────┐ ┌─────┐ ┌─────┐             │
│  │ FI-1│ │ FI-2│ │ FI-3│  ...        │
│  └──┬──┘ └──┬──┘ └──┬──┘             │
│     └────┬───┘──────┘                  │
│          ↓                             │
│  Task-Sensitive Network                │
│  (为每个任务选择/加权不同的 FI 方法)     │
│     ↓          ↓          ↓            │
│  Task-1      Task-2     Task-3        │
│  (CTR)       (CVR)      (...)         │
└────────────────────────────────────────┘
```

**与 MMoE/PLE 的本质区别：**

| 维度 | MMoE/PLE | DTN |
|------|----------|-----|
| 关注点 | Expert 网络的共享/专有分配 | 特征交互方法的 task-specific 选择 |
| 特征交互 | 不显式建模 | 多种 FI 方法并行 + task-sensitive 选择 |
| 信息流 | Bottom-up (shared → gated → task) | Parallel FI → Task-sensitive fusion |
| 创新层 | Expert 层 | Feature Interaction 层 |

### 3.2 CETNet: Collaborative Ensemble Framework (Meta, arxiv 2411.13700)

| 维度 | 内容 |
|------|------|
| **问题** | 单纯增大模型参数并不总能提升推荐性能 |
| **方法** | 多模型多 embedding 协同训练 + 置信度融合 |
| **组成** | InterFormer (序列交互) + DHEN (层次化特征交互)，各自独立 embedding table |

**三大核心组件：**

**组件 1: Multi-Embedding Paradigm**
```
Model A (InterFormer):  Embedding Table A → 序列+异构特征交互
Model B (DHEN):         Embedding Table B → 层次化特征交互

独立 embedding table 的意义：
  - 同一特征在不同模型中学到不同表示
  - 增加了表示多样性，避免同质化
```

**组件 2: Collaborative Learning (对称 KL 散度)**

$$\mathcal{L}_{collab} = \frac{1}{2}\left[D_{KL}(p_A \| p_B) + D_{KL}(p_B \| p_A)\right]$$

- 对称 KL 避免单方向"拉扯"导致的不均匀学习
- 两个模型互相学习对方的知识，而非一个教一个

**组件 3: Confidence-Based Fusion**

$$p_{final} = \sum_i w_i \cdot p_i, \quad w_i = \frac{\exp(H_i^{-1})}{\sum_j \exp(H_j^{-1})}$$

其中 $H_i = -\sum_k p_{i,k} \log p_{i,k}$ 是模型 $i$ 的预测熵。熵越低 → 置信度越高 → 权重越大。

**CETNet 的本质洞察：**
> 推荐系统的 scaling 不应只追求单模型参数增大，而是多模型多视角的协同。这与 LLM 领域的 MoE 思想异曲同工，但在 embedding 层面实现多样性。

### 3.3 CTR/Feature Interaction 主题总结

**面试 Q&A 要点：**

**Q: 为什么 MTL 推荐系统需要 task-specific 特征交互？**
A: 因为不同任务对同一特征的依赖模式不同。例如 CTR 更关注视觉吸引力特征，CVR 更关注价格/品质特征。共享特征交互会产生 task conflict（梯度冲突），导致 negative transfer。DTN 通过为每个任务选择不同的特征交互方法来缓解这一问题。

**Q: CETNet 的 collaborative learning 和 knowledge distillation 的区别？**
A: KD 是单向的（teacher → student），collaborative learning 是双向对称的。KD 假设 teacher 更好，CL 假设两个模型各有优势。CETNet 用对称 KL 散度确保两个模型互相受益，避免一个模型被另一个"碾压"。

**Q: 为什么 CETNet 用独立 embedding table 而非共享？**
A: 共享 embedding 会导致两个模型看到相同的特征表示，丧失多样性。独立 embedding 让同一特征在不同模型中有不同的"解读"，增加了集成的互补性。代价是 2x 的 embedding 存储，但推荐系统 embedding 通常分布式存储，可接受。

---

## Section 4: Generative Recommendation Survey (GR-LLMs)

### 4.1 GR-LLMs Survey 概览 (arxiv 2507.06507)

| 维度 | 内容 |
|------|------|
| **范围** | LLM-based 生成式推荐的全面综述 |
| **核心论点** | LLM-based GR 正在形成与判别式推荐截然不同的新范式 |
| **发表时间** | 2025 年 7 月（v2） |

### 4.2 两大范式分类

```
Paradigm 1: Generative Architecture (纯生成式)
  - 完全抛弃 DLR 框架
  - LLM 直接处理用户行为序列
  - 通过监督任务（如 CTR 预测）生成候选评分
  - 代表：HSTU, ULTRA-HSTU, 纯 LLM-based ranking

Paradigm 2: Hybrid Integration (混合集成)
  - 保留传统 DLR 框架
  - LLM 作为特征增强器/知识源
  - 代表：LUM (三步范式), IDProxy (冷启动增强)

工业现实：
  Paradigm 2 (Hybrid) 更容易落地
  原因：(1) 保留成熟工程实践 (2) 风险可控 (3) 增量接入
  但 Paradigm 1 是长期方向
```

### 4.3 GR 的核心挑战（工业场景）

| 挑战 | 描述 | 相关论文 |
|------|------|----------|
| **Scaling 效率** | LLM 推理延迟过高，推荐需要 ms 级响应 | ULTRA-HSTU, MixFormer |
| **特征兼容性** | 推荐有丰富的稀疏特征，LLM 不擅长处理 | LUM, DTN |
| **冷启动** | 生成式模型也需要足够的行为数据 | IDProxy, EmerG |
| **多目标** | 推荐需要同时优化 CTR/CVR/时长等多目标 | DTN |
| **可解释性** | 生成式推荐的"黑箱"程度更高 | - |
| **增量更新** | LLM 不易做在线增量学习 | - |

### 4.4 与本文其他论文的映射

```
GR-LLMs Survey 视角下的 10 篇论文定位：

                    ┌─────────────────────┐
                    │   Generative Rec     │
                    │   Survey (GR-LLMs)   │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   Scaling Law             Cold-Start            Feature Interaction
   ┌──────────┐           ┌──────────┐          ┌──────────┐
   │ULTRA-HSTU│           │ IDProxy  │          │   DTN    │
   │ LUM      │           │ EmerG    │          │ CETNet   │
   │MixFormer │           │Sparse CL │          └──────────┘
   │ApEn Scale│           └──────────┘
   └──────────┘

纯生成式路线: ULTRA-HSTU, MixFormer
混合路线:     LUM, IDProxy
传统增强路线: EmerG, DTN, CETNet, Sparse CL
理论补充:     ApEn Scaling Law
```

### 4.5 Generative Rec 面试 Q&A

**Q: 生成式推荐和判别式推荐的本质区别？**
A: 判别式推荐是 $P(\text{click} | \text{user}, \text{item})$ 的二分类问题；生成式推荐是 $P(\text{next item} | \text{history})$ 的序列生成问题。前者需要 explicit candidate set，后者可以直接生成 item。生成式的优势在于统一了召回和排序，但挑战在于 vocabulary 过大（百万级 item）时的计算效率。

**Q: 为什么工业界更倾向 Hybrid Integration 而非纯生成式？**
A: 三个原因：(1) 现有推荐系统有大量沉淀的特征工程和调优经验，纯生成式需要全部重来；(2) 纯生成式的延迟和计算成本在当前硬件下难以满足工业要求；(3) 混合路线风险可控，可以灰度发布、逐步迁移。LUM 的三步范式是典型代表。

**Q: 未来生成式推荐的发展方向？**
A: (1) 端到端生成式架构逐步替代 DLR（类似 Transformer 替代 RNN）；(2) 推理加速技术（稀疏注意力、KV Cache、量化）使纯生成式可行；(3) 多模态融合（IDProxy 方向）让生成式模型理解内容语义；(4) Scaling Law 指导最优模型配置（ULTRA-HSTU + ApEn 方向）。

---

## 附录：10 篇论文速查表

| # | 论文 | 机构 | 年份 | 主题 | 关键词 |
|---|------|------|------|------|--------|
| 1 | ULTRA-HSTU | Meta | 2026 | Scaling Law | Sparse Attention, System Co-design, 16K seq |
| 2 | LUM | Alibaba | 2025 | Scaling Law | Three-step, 7B params, Feature Enhancement |
| 3 | MixFormer | ByteDance | 2026 | Co-Scaling | Unified Transformer, User-Item Decoupling |
| 4 | GR-LLMs Survey | - | 2025 | Survey | Generative vs Hybrid, Industrial Challenges |
| 5 | EmerG | - | 2024 | Cold-Start | HyperNetwork, GNN, Meta-Learning |
| 6 | DTN | - | 2024 | Multi-Task CTR | Task-specific FI, Task-sensitive Network |
| 7 | IDProxy | Xiaohongshu | 2026 | Cold-Start | MLLM Proxy, Coarse-to-Fine Alignment |
| 8 | Sparse CL | - | 2026 | Cold-Start | α-entmax, Content-based, Item-Item Similarity |
| 9 | CETNet | Meta | 2024 | CTR Ensemble | Multi-Embedding, Symmetric KL, Confidence Fusion |
| 10 | ApEn Scaling | - | 2024 | Scaling Law | Approximate Entropy, Data Quality, Performance Law |
