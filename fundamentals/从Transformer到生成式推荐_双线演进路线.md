# 从 Transformer 到生成式推荐：架构线 x 特征线双线演进路线

> 标签：#transformer #generative-recommendation #architecture #feature-evolution #面试
> 关联：[[Transformer演进]] | [[序列建模演进]] | [[attention_in_recsys|推荐中的注意力机制]] | [[generative_recsys|生成式推荐]] | [[Semantic_ID演进知识图谱]] | [[生成式推荐范式统一_20260403]] | [[bytedance_recsys_transformer|字节系搜广推Transformer演进]]

---

## 0. 为什么需要双线视角？

推荐系统从传统 pipeline 演进到生成式推荐（Generative Recommendation, GR），本质上是**两条线同时推进、最终汇合**的过程：

- **架构线**：Transformer 模块如何被改造以适应推荐场景？（Attention/位置编码/效率优化/推荐特化）
- **特征线**：Item/User 的表示如何从"手工统计特征"进化到"可生成的 token 序列"？

**两线交汇点就是 GR**——架构上统一为 Transformer（或其变体），特征上统一为 token 序列，推荐变成了"给定用户行为 token 序列，自回归生成 item token 序列"。

本文用**时间线 + 双列对照**的方式展现这一演进过程。

---

## 1. 时间线总览：双线演进对照表

> **[P]** = Paradigm Shift（范式转换节点）

| 年份 | 架构线（Transformer 模块优化） | 特征线（表示与交互方式演进） | 交汇 |
|------|-------------------------------|-------------------------------|------|
| 2015 | — | GRU4Rec: RNN 序列建模 | — |
| 2017 | Transformer: Self-Attention + Sinusoidal PE | — | — |
| 2018 | **[P]** GPT/BERT: Decoder-Only / Encoder-Only 分化 | **[P]** DIN: Target Attention 引入推荐 | 推荐首次用 Attention |
| 2019 | MQA: KV 共享压缩 | DIEN: Attention + GRU 时序演化; SASRec: 自回归序列推荐 | Transformer 进入推荐 |
| 2020 | SwiGLU FFN; Scaling Law (Kaplan) | SIM: 两阶段检索式 Attention; BST: Transformer 做特征提取 | 长序列+特征交叉 |
| 2021 | RoPE 位置编码 | DLRM: 统计特征 + Embedding 交叉 | — |
| 2022 | **[P]** FlashAttention: IO-aware tiling; ALiBi; Chinchilla | **[P]** TIGER: RQ-VAE Semantic ID + T5 自回归生成 | **GR 元年** |
| 2023 | GQA (LLaMA); Sliding Window (Mistral); SwiGLU 成标配 | **[P]** Spotify GLIDE 工业部署; Variable-Length SID; BERT4Rec→SASRec 成熟 | GR 工业落地 |
| 2024 | **[P]** MLA (DeepSeek-V2): 低秩联合压缩 KV; FlashAttention-2 | **[P]** HSTU: 推荐 Scaling Law, 万亿参数; MTGR: 双流融合; OneRec: 召排统一 | 推荐系统的 Scaling 时代 |
| 2025 | MTP (DeepSeek-V3); FP8 训练; MoE 256专家 | OneRec-V2 (8B Lazy Decoder-Only); SID + CoT Reasoning; PROMISE (PRM test-time scaling) | GR + 推理增强 |
| 2026 | FlashAttention-3 (H100); Linear Attention (FuXi-Linear) | **[P]** OneTrans 统一架构; LONGER 万级序列; TokenMixer-Large 7B-15B; Mender 偏好感知 | **两线完全统一** |

---

## 2. 架构线：Transformer 模块优化

### 2.1 Attention 变体演进

```
Standard Self-Attention (2017)
    |
    v  问题：KV Cache 随 head 数线性增长
MHA → MQA (2019) → GQA (2023) → MLA (2024)
    |                                    |
    |  问题：推荐场景 softmax 归一化不必要
    v
HSTU Pointwise Aggregated Attention (2024)
    |  ReLU 替代 softmax，去掉 LayerNorm
    v
SparseCTR 三分支稀疏 Attention (2026)
    个性化分块 + Global/Transition/Local 三路
```

#### 2.1.1 从 MHA 到 MLA：KV Cache 压缩之路

| 变体 | 年份 | KV Cache/层 | 核心思想 | 代表模型 |
|------|------|------------|---------|---------|
| MHA | 2017 | $2hd_k n$ | 每个 head 独立 KV | Transformer, GPT-3 |
| MQA | 2019 | $2d_k n$ | 所有 head 共享 1 组 KV | PaLM |
| GQA | 2023 | $2gd_k n$ | 分 $g$ 组共享 | LLaMA-3, Mistral |
| MLA | 2024 | $d_c n$ ($d_c \ll hd_k$) | 低秩联合压缩 KV | DeepSeek-V2/V3 |

$$
\text{MHA: head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

$$
\text{GQA: head}_i = \text{Softmax}\left(\frac{Q_i K_{\lfloor i/(h/g) \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor i/(h/g) \rfloor}
$$

$$
\text{MLA: } c_t = W^{DKV} h_t, \quad K_t = W^{UK} c_t, \quad V_t = W^{UV} c_t
$$

**关键结论**：GQA 是工程简单的折中（LLaMA-3 用 $g=8$），MLA 更激进但需 absorption trick。推荐场景中 HSTU 走了另一条路——直接去掉 softmax。

#### 2.1.2 推荐特化的 Attention

推荐场景与 NLP 的关键差异：

| 维度 | NLP/LLM | 推荐系统 |
|------|---------|---------|
| Token 来源 | 词/子词 | 行为(点击/购买) + 属性(ID/类目) |
| 序列长度 | 4K-128K | 50-100K（行为序列） |
| 相关性分布 | 较均匀 | 极度稀疏（万级行为中仅几十个相关） |
| 归一化需求 | 需要（概率分布） | 不必要（稀疏激活更好） |

**因此推荐 Attention 的演进走向了不同路线**：

```
NLP 路线：保持 softmax，优化效率
  Sparse Attn → Linear Attn → FlashAttn → MLA

推荐路线：改变 Attention 语义
  DIN Target Attn (2018)
      → DIEN AUGRU (2019)：Attention 控制 GRU 门
      → SIM 两阶段 (2020)：先检索 top-K 再精算
      → SASRec/BST (2019-2020)：标准 Self-Attention
      → HSTU (2024)：ReLU 替代 softmax + 去 LayerNorm
      → SparseCTR (2026)：三分支结构化稀疏
      → LONGER (2026)：Token Merge + Global Token 万级序列
```

**DIN Target Attention（2018，阿里）**：

$$
\alpha_t = \text{MLP}\left([\mathbf{e}_t;\, \mathbf{e}_a;\, \mathbf{e}_t \odot \mathbf{e}_a;\, \mathbf{e}_t - \mathbf{e}_a]\right)
$$

$$
\mathbf{v}_u = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{e}_t
$$

- $\mathbf{e}_a$ 是候选物品，$\mathbf{e}_t$ 是历史行为
- 本质：候选物品作为 Query 去问历史行为"你跟我有多相关？"
- **与 Self-Attention 的区别**：Self-Attention 的 Q=K=V 来自同一序列，Target Attention 的 Q 来自候选

**HSTU Pointwise Aggregated Attention（2024，Meta）**：

$$
\text{Attn}(Q, K, V) = \text{ReLU}(QK^T) \cdot V
$$

- 去掉 softmax：推荐场景不需要概率归一化，ReLU 更稀疏且利用 GPU Tensor Core 更高效
- 去掉 LayerNorm：推荐场景数值稳定性足够
- 效果：推理加速 3-5 倍，支持 1.5T 参数 Scaling

> **[P] 范式转折**：HSTU 证明推荐系统不必照搬 NLP 的 Transformer 设计。softmax 和 LayerNorm 在推荐中是非必要的，去掉它们反而更好。

### 2.2 位置编码演进

| 方案 | 年份 | 类型 | 外推性 | 推荐适配 | 代表模型 |
|------|------|------|--------|---------|---------|
| Sinusoidal | 2017 | 绝对 | 差 | 差 | Transformer |
| Learned Absolute | 2018 | 绝对 | 无 | 弱 | GPT-1/2, BERT, SASRec |
| ALiBi | 2022 | 相对(隐式) | 好 | 中 | BLOOM |
| RoPE | 2022 | 相对(显式) | 好 | **强** | LLaMA 全系, DeepSeek |
| Timestamp PE | 2024+ | 时间戳 | — | **推荐专用** | CADET (LinkedIn) |
| Time-aware RoPE | 2025+ | RoPE+时间 | 好 | **推荐专用** | SynerGen (Amazon) |

$$
\text{Sinusoidal: } PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
\text{RoPE: } \langle f(q, m), f(k, n) \rangle = g(q, k, m-n) \quad \text{（内积只依赖相对位置）}
$$

**推荐场景的位置编码特殊需求**：

1. **时间间隔不均匀**：用户行为间隔可能是秒级（连续浏览）到月级（复购），简单的位置编号丢失时间信息
2. **周期性模式**：周末购物、午休刷视频等周期性行为
3. **超长序列外推**：万级行为序列需要良好的长度泛化

**SparseCTR 的复合时间编码（2026，美团）**：每个 attention head 有独立的时间偏置系数，同时编码顺序关系和周期关系。

**CADET 的 Timestamp RoPE（2025，LinkedIn）**：将 RoPE 的位置参数替换为行为发生的实际时间戳，让旋转频率反映真实时间间隔。

> **趋势**：NLP 领域 RoPE 已成为绝对标准；推荐领域在 RoPE 基础上叠加时间感知，形成 time-aware PE。

### 2.3 效率优化：FlashAttention → Linear Attention → SSM

| 方案 | 复杂度 | 精确/近似 | 内存 | 推荐适用性 |
|------|--------|----------|------|-----------|
| 标准 Attention | $O(n^2 d)$ | 精确 | $O(n^2)$ | 短序列(<200) |
| FlashAttention | $O(n^2 d)$ | **精确** | $O(n)$ | **通用标准** |
| Sliding Window | $O(nwd)$ | 近似 | $O(nw)$ | 中长序列 |
| Linear Attention | $O(nd^2)$ | 近似 | $O(nd)$ | $d \ll n$ 时 |
| FuXi-Linear | $O(nd^2)$ | 近似 | $O(nd)$ | **推荐专用** |
| Mamba/SSM | $O(nd)$ | — | $O(d)$ | 超长序列 |

**FlashAttention（2022，Dao et al.）**：

核心：IO-aware tiling——将 Q/K/V 分块加载到 SRAM，在 SRAM 中完成 online softmax，**从不显式存储 $n^2$ attention matrix**。

```
标准 Attention：Q,K → HBM → QK^T → 写 HBM(n²) → softmax → 写 HBM → ×V → output
FlashAttention：Q块,K块,V块 → SRAM → 计算+online softmax → 写 HBM(output块)
```

- 计算量不变（仍 $O(n^2d)$），但内存从 $O(n^2)$ 降到 $O(n)$，速度 2-4 倍提升
- **关键**：不是近似，是精确计算。改变的是计算顺序和内存访问模式

**FuXi-Linear（2026）**——推荐专用 Linear Attention：

标准 Linear Attention 在推荐中效果不佳，因为时间信号和语义信号在同一通道中串扰。

$$
\text{Output}_t = \frac{\phi(Q_t)^T \sum_{s \leq t} \phi(K_s) V_s^T}{\phi(Q_t)^T \sum_{s \leq t} \phi(K_s)}
$$

FuXi-Linear 的双通道设计：
- **Temporal Retention Channel**：独立的时间衰减矩阵
- **Linear Positional Channel**：可学习核函数编码位置信息

效果：prefill 加速 10 倍，decode 加速 21 倍，首次在千级 token 序列上验证推荐领域的 power-law scaling。

**Mamba/SSM（2023-2024）**——线性复杂度新范式：

$$
\mathbf{h}_t = \bar{A} \mathbf{h}_{t-1} + \bar{B} \mathbf{x}_t, \quad \mathbf{y}_t = C \mathbf{h}_t
$$

Mamba 让 $B, C, \Delta$ 依赖于输入（选择性机制），重要输入大步长多写入状态，不重要输入快速跳过。

| 维度 | Transformer | Mamba |
|------|-------------|-------|
| 训练复杂度 | $O(n^2)$ | $O(n)$ |
| 推理复杂度 | $O(n)$（KV Cache） | $O(1)$（固定状态） |
| 推理内存 | $O(n)$（线性增长） | $O(1)$（固定大小） |
| 长距离建模 | 直接交互 | 通过状态压缩（有损） |

推荐领域进展：SIGMA (AAAI'25) PF-Mamba 双向化 + M2Rec 多尺度 Mamba。截至 2026，Transformer + 剪枝仍是工业主流，Mamba 在研究阶段。

### 2.4 推荐特化架构演进

```
DIN (2018): Target Attention（候选问历史）
    |
    v
DIEN (2019): Attention + GRU（时序演化）
    |
    v  [P] Transformer 进入推荐
SASRec/BERT4Rec (2019): Self-Attention 序列推荐
    |
    v
BST (2019): Transformer 做特征提取（阿里搜索）
    |
    v
SIM (2020): 两阶段检索 + Attention（万级序列）
    |
    v  [P] 推荐 Scaling Law
HSTU (2024, Meta): 1.5T 参数，ReLU Attention，推荐 Scaling Law
    |
    ├── MTGR (2024, 美团): 双流融合（HSTU + DLRM）
    ├── OneTrans (2026, 字节): 统一 Transformer（特征+序列一体化）
    ├── OneRec-V2 (2026, 快手): 8B Lazy Decoder-Only
    └── LONGER (2026, 字节): Token Merge 万级序列

每个节点解决的问题和引入的新问题：
- DIN：解决→动态兴趣表示；引入→无时序建模
- DIEN：解决→时序演化；引入→O(T)串行，长序列受限
- SASRec：解决→并行+全局交互；引入→O(N²)长序列瓶颈
- SIM：解决→万级序列；引入→两阶段不端到端
- HSTU：解决→Scaling Law+效率；引入→丢失 side features
- MTGR：解决→side feature 融合；引入→双流复杂度
- OneTrans：解决→架构统一；引入→训练稳定性挑战
```

> 详见 [[attention_in_recsys|推荐中的注意力机制]] 和 [[bytedance_recsys_transformer|字节系搜广推Transformer演进]]

---

## 3. 特征线：从统计特征到可生成的 Token

### 3.1 特征表示演进总览

```
统计特征时代 (2015-2018)
  CTR/CVR/历史频次/类目分布 → FM/GBDT/Wide&Deep
    |
    v  [P] Embedding 取代手工特征
Embedding 时代 (2018-2021)
  ID Embedding + Side Info → DIN/DIEN/DLRM/DCN-V2
    |
    v  [P] 序列化取代拼接
序列建模时代 (2019-2023)
  行为序列 Token 化 → SASRec/BST/HSTU
    |
    v  [P] 从判别到生成
Token 生成时代 (2022-now)
  Semantic ID + 自回归生成 → TIGER/OneRec/生成式推荐
```

### 3.2 Seq vs 统计特征交互

| 时期 | 主导方式 | 代表 | 特征交互 |
|------|---------|------|---------|
| 2015-2017 | 纯统计特征 | FM/GBDT | 特征交叉（二阶/高阶） |
| 2018-2020 | 统计 + 序列并行 | DIN+DLRM | 序列提取兴趣向量，拼接统计特征后 MLP |
| 2021-2023 | 序列主导 + 统计辅助 | SASRec+side info | Transformer 建模序列，side features 作附加输入 |
| 2024-2025 | **[P]** 融合统一 | MTGR/OneTrans | 序列+统计在同一个 Transformer 中交互 |
| 2026 | **[P]** 全 Token 化 | HSTU/OneRec-V2 | 一切皆 Token，统计特征也被序列化 |

**DLRM（2019，Meta）**——传统特征交互的巅峰：

```
Dense Features → MLP底层 ─┐
                           ├─ Feature Interaction (点积/DCN) → MLP顶层 → CTR
Sparse Features → Embedding ┘
```

**OneTrans（2026，字节）**——统一 Transformer 的终局形态：

所有输入（静态特征 + 行为序列）统一 tokenize，通过 Heterogeneous Attention Mask 定义四种注意力模式：

| 模式 | Q 来源 | K/V 来源 | 意义 |
|------|--------|---------|------|
| Feature-to-Feature | 静态特征 | 静态特征 | 传统特征交叉 |
| Feature-to-Action | 静态特征 | 行为序列 | 特征感知序列信息 |
| Action-to-Action | 行为序列 | 行为序列 | 序列建模 |
| Action-to-Feature | 行为序列 | 静态特征 | 序列感知全局特征 |

效果：相比分离式架构（DCN-V2 + DIN），参数量减少 35%、推理延迟降低 20%，AUC 提升 0.38%。

> **[P] 范式转折**：OneTrans 证明特征交叉和序列建模不需要两套模块，一个 Transformer 配合异构 Mask 就够了。

### 3.3 User/Item 侧交互方式演进

```
阶段 1: 双塔独立编码 (2019-2021)
  User Tower ─── Item Tower
       |              |
    user emb      item emb
       └──── 内积/余弦 ────→ 相似度
  问题：user 和 item 完全独立编码，无交叉交互

阶段 2: 交叉注意力 (2022-2024)
  User Behavior ──┐
                   ├─ Cross-Attention → 融合表示 → 打分
  Item Features ──┘
  代表：M-FALCON (Meta), 搜索 Cross-Encoder
  问题：计算量大，每对 user-item 都要交叉，无法做候选集规模的计算

阶段 3: 统一序列建模 (2024-now)  [P]
  [user行为₁, user行为₂, ..., item属性₁, item属性₂, ...]
       └───── 统一 Transformer ─────→ 预测/生成
  代表：HSTU, OneTrans, OneRec
  核心变化：user 行为和 item 属性混合编码为同一个 token 序列
```

**M-FALCON（Meta，2024 开源）**：

通过 cross-attention 将物品多维特征显式注入序列表示，比纯 HSTU Recall@20 提升 2.1%。

**MTGR 双流融合（美团，2024）**：

$$
\text{Output} = \sigma(g) \cdot \text{SeqFlow}(x) + (1 - \sigma(g)) \cdot \text{FeatureFlow}(x)
$$

- 序列流：HSTU 风格的行为序列建模
- 特征交叉流：DLRM 风格的 side features 交叉
- 自适应 gating 动态融合

**核心洞察**：生成式推荐（HSTU）和传统特征交叉（DLRM）不是互斥的，而是互补的。MTGR 在美团外卖 GAUC +2.88pp，GMV +2.1%。

### 3.4 Tokenizer 设计演进

推荐中的 "Tokenizer" 经历了根本性变革：

| 阶段 | Token 化方式 | 带什么信息 | 问题 |
|------|-------------|-----------|------|
| 早期 | 无显式 Token 化 | 连续 Embedding | 无法做生成式推荐 |
| ID only | 物品 ID → Embedding lookup | 纯 ID | 冷启动失败 |
| ID + side info | ID + 类目/品牌/价格 | ID + 属性 | 属性融合方式粗糙 |
| **[P]** Semantic ID | RQ-VAE 离散化 | 层次化语义码字 | 量化误差/碰撞 |
| Soft ID | 连续向量序列 | 端到端可微 | 存储开销大 |
| Reasoning SID | SID + 推理 token | 语义 + 推理链 | 推理延迟 |

#### 3.4.1 Semantic ID 核心技术

**RQ-VAE（残差量化变分自编码器）**：

$$
c_l = \arg\min_{k \in \mathcal{C}_l} \|z_l - e_k\|_2, \quad z_{l+1} = z_l - e_{c_l}
$$

$$
P(\text{item} | \text{history}) = \prod_{l=1}^{L} P(c_l | c_1, \ldots, c_{l-1}, \text{history})
$$

- 每层从粗到细：$c_1$ 编码大类（电子产品），$c_2$ 编码子类（手机），$c_3$ 编码具体物品
- Beam Search 解码：每层展开 top-K，层次化剪枝

**Semantic ID 演进时间线**：

| 年份 | 工作 | 核心贡献 |
|------|------|---------|
| 2022 | **TIGER** (Google) | RQ-VAE + T5 自回归生成，GR 开山 |
| 2023 | **Spotify GLIDE** | 工业部署，Podcast Recall +25% |
| 2023 | Variable-Length SID | 冷启动长 ID + 热门短 ID，平均长度 -30% |
| 2024 | **UniGRec** (Soft ID) | 连续 soft token，碰撞率→0% |
| 2024 | SID + CoT Reasoning | 层间推理 token，Recall@10 +4.7% |
| 2025 | **DIGER** | Gumbel-Softmax 端到端，码本利用率 35%→88% |
| 2025 | **ETEGRec** | 推荐 Loss 直接反传到 ID 生成 |
| 2025 | QuaSID (快手) | 区分良性/有害碰撞，GMV +2.38% |
| 2026 | SID Staleness 对齐 | 轻量更新，不需全量重建 |
| 2026 | PROMISE (PRM) | test-time compute scaling，Recall +9.1% |

> 详见 [[Semantic_ID演进知识图谱]] 和 [[vector_quantization_methods|向量量化方法]]

#### 3.4.2 离散 SID vs 连续 Soft ID

| 维度 | 离散 Semantic ID | 连续 Soft ID (UniGRec) |
|------|-----------------|----------------------|
| 表示 | 层次化码字 [C1,C2,C3] | 连续向量序列 |
| 量化误差 | 有（RQ-VAE 重建损失） | 无（端到端可微） |
| 碰撞率 | ~3.2% | 0% |
| 存储 | 小（整数序列） | 大（千万物品 10-50GB） |
| Beam Search | 标准分类问题，天然兼容 | 需要 ANN 检索 |
| 工业部署 | Spotify 已落地 | 存储是障碍 |
| 精度 | 基线 | +5.8% Recall |

**未来方向**：混合方案——粗层用离散 token 做高效搜索，细层用连续向量做精确匹配。

### 3.5 从 Discriminative 到 Generative

> **[P] 这是整个推荐系统最大的范式转换。**

$$
\underbrace{P(y|\mathbf{x}, \text{item})}_{\text{判别式：给定候选打分}} \longrightarrow \underbrace{P(\text{item}|\mathbf{x})}_{\text{生成式：直接生成推荐}}
$$

| 维度 | 判别式推荐 | 生成式推荐 |
|------|-----------|-----------|
| 核心操作 | 给候选打分→排序 | 直接生成目标 ID |
| 候选集 | 需提前构建 | 不需要（从全空间生成） |
| 特征角色 | "拼接后打分" | "序列化后生成" |
| 冷启动 | 依赖交互数据 | 内容特征即可编码 SID |
| 多样性 | 需后处理（MMR/DPP） | 采样天然带多样性 |
| Scaling | 效果饱和较快 | 持续 Scaling (HSTU 1.5T) |

**特征从"拼接"到"序列化"的具体变化**：

```
判别式时代（拼接后打分）：
  [user_emb; item_emb; cross_features; context_features]
      → MLP/DCN → sigmoid → CTR score

生成式时代（序列化后生成）：
  [action₁_token, action₂_token, ..., actionₜ_token]
      → Transformer → autoregressive → [item_SID_c₁, c₂, c₃, c₄]
```

---

## 4. 两线交汇的关键节点

### 4.1 节点一：DIN/DIEN（2018-2019）——Attention 首次进入推荐

**架构线贡献**：Target Attention 让推荐模型首次具备"按需分配权重"的能力。
**特征线贡献**：从统计特征的定权加和，到动态加权的行为序列建模。
**交汇意义**：Attention 成为架构和特征的共同语言，为后续 Transformer 进入推荐铺路。

### 4.2 节点二：SASRec/BST（2019-2020）——Transformer 进入推荐

**架构线贡献**：Self-Attention 替代 GRU，序列建模从串行变并行。
**特征线贡献**：行为序列首次被当作"语言"来建模（next-item prediction ≈ next-token prediction）。
**交汇意义**：推荐模型开始与 NLP 模型架构趋同，但规模和效率仍有巨大差距。

### 4.3 节点三：HSTU（2024）——推荐 Scaling Law [P]

**架构线贡献**：ReLU Attention + 去 LayerNorm，推荐特化的 Transformer 可 scale 到 1.5T 参数。
**特征线贡献**：一切行为直接当 token，"统计特征"开始被序列化。
**交汇意义**：**首次证明推荐系统也有 Scaling Law**，参数和数据量的增长带来持续的效果提升。

$$
L(N, D) = \alpha N^{-\beta} + \gamma D^{-\delta} + \epsilon
$$

从 1.5B 到 1.5T 参数，性能持续提升且未饱和。

### 4.4 节点四：TIGER/GLIDE（2022-2023）——GR 元年 [P]

**架构线贡献**：T5 (Encoder-Decoder) 做自回归生成，Beam Search 做层次化解码。
**特征线贡献**：RQ-VAE Semantic ID 让物品有了"可生成"的离散表示。
**交汇意义**：**推荐正式变成了语言生成问题**——给定行为序列，自回归生成物品 token 序列。

### 4.5 节点五：OneTrans/OneRec-V2（2025-2026）——两线完全统一 [P]

**架构线贡献**：单一 Transformer 统一特征交叉 + 序列建模（OneTrans），或直接 Decoder-Only 端到端（OneRec-V2）。
**特征线贡献**：所有特征（静态属性、行为序列、上下文）统一 token 化，成为同一个序列的组成部分。
**交汇意义**：**架构线和特征线不再可区分**——模型就是一个 Transformer，输入就是一个 token 序列，输出就是生成的推荐。

---

## 5. 双线并行时期对照详解

### 5.1 2017-2019：萌芽期

| 架构线 | 特征线 |
|--------|--------|
| Transformer 原始架构提出 | DIN Target Attention 进入推荐 |
| MQA 首次提出 KV 共享 | DIEN 加入时序演化（GRU + Attention） |
| GPT/BERT 确立 Decoder/Encoder 范式 | SASRec 首次用 Transformer 做序列推荐 |
| **同期 LLM 侧**：参数从 100M→1.5B | **同期推荐侧**：仍以特征工程+浅层模型为主 |

**核心矛盾**：NLP 快速 scale，推荐还在"特征工程"时代。

### 5.2 2020-2022：加速期

| 架构线 | 特征线 |
|--------|--------|
| Scaling Law (Kaplan 2020) | SIM 万级序列两阶段方案 |
| SwiGLU 取代 ReLU/GELU FFN (Shazeer 2020) | DLRM 统计+Embedding 交叉成工业标准 |
| RoPE 位置编码 (Su 2021) | 推荐 Embedding: ID + side info 混合 |
| **[P]** FlashAttention (2022) | **[P]** TIGER: Semantic ID + T5 自回归 |
| Chinchilla: 数据 Scaling > 参数 Scaling | RQ-VAE 离散化物品 → 推荐变成生成 |

**核心矛盾**：架构线的效率优化（FlashAttention）让长序列 Transformer 成为可能，但特征线还在探索"什么东西值得 token 化"。

### 5.3 2023-2024：爆发期

| 架构线 | 特征线 |
|--------|--------|
| GQA (LLaMA-3): 工程简洁的 KV 压缩 | Spotify GLIDE: SID 首次工业部署 |
| Sliding Window (Mistral): 局部+全局 | Variable-Length SID: 冷启动/热门差异化 |
| **[P]** MLA (DeepSeek-V2): 低秩 KV 压缩 | **[P]** HSTU: 1.5T 推荐 Transformer |
| FlashAttention-2: 2x 加速 | MTGR: 序列流+特征交叉流双流融合 |
| MoE: 671B 总参但 37B 活跃参数 | UniGRec: Soft ID 消除量化误差 |
| | OneRec: 召回+排序统一为单一生成模型 |

**核心矛盾**：LLM 侧的架构创新（MLA/MoE）是否应该直接迁移到推荐？HSTU 的回答是"不完全——推荐需要自己的 Attention 设计"。

### 5.4 2025-2026：统一期

| 架构线 | 特征线 |
|--------|--------|
| MTP (DeepSeek-V3): 多 token 预测 | OneRec-V2: 8B Lazy Decoder-Only |
| FP8 训练: 算力成本 1/10 | SID + CoT Reasoning: 推理增强生成 |
| FlashAttention-3 (H100) | PROMISE: PRM test-time scaling |
| Linear Attention (FuXi-Linear) | OneTrans: 统一特征+序列 Transformer |
| Mamba for SeqRec (SIGMA/M2Rec) | LONGER/TokenMixer-Large: 万级序列 7B-15B |
| | Mender: 自然语言偏好条件生成 |

**核心趋势**：**架构和特征不再是两个独立问题**。OneTrans 的 Heterogeneous Mask、OneRec-V2 的 Lazy Decoder-Only、TokenMixer-Large 的统一参数化——都在说同一件事：一个 Transformer + 一个 Token 序列 = 整个推荐系统。

---

## 6. GR 为什么是两条线的自然终点？

### 6.1 架构线的收敛

从 Transformer 原始架构到推荐特化，经历了：

1. **Attention 简化**：softmax → ReLU（HSTU），因为推荐不需要概率归一化
2. **Norm 简化**：LayerNorm → 去掉（HSTU），因为推荐数值稳定
3. **效率优化**：FlashAttention → FuXi-Linear → Token Merge（LONGER），因为推荐序列极长
4. **架构统一**：双流→单 Transformer（OneTrans），因为特征交叉和序列建模本质上都是"token 间交互"

**最终形态**：一个高效的、推荐特化的 Transformer——没有 softmax、没有 LayerNorm、支持万级序列、统一处理所有类型的 token。

### 6.2 特征线的收敛

从统计特征到可生成的 Token，经历了：

1. **表示进化**：手工特征 → Embedding → Semantic ID → Soft ID
2. **交互进化**：特征拼接 → Target Attention → Self-Attention → 统一序列
3. **生成进化**：判别式打分 → 自回归生成物品 ID → 生成推荐列表

**最终形态**：一切皆 Token——用户行为是 token 序列，物品是 Semantic ID token，推荐就是"下一个 token 预测"。

### 6.3 GR = 架构收敛 x 特征收敛

$$
\boxed{\text{GR} = \underbrace{\text{推荐特化 Transformer}}_{\text{架构线终点}} + \underbrace{\text{万物 Token 化}}_{\text{特征线终点}} + \underbrace{\text{自回归生成}}_{\text{两线交汇}}}
$$

```
GR 的完整流程：

[用户行为 tokens] ─→ 推荐特化 Transformer ─→ 自回归生成 ─→ [物品 SID tokens]
   ↑ 特征线产物           ↑ 架构线产物          ↑ 两线交汇        ↑ 特征线产物
```

**为什么 GR 是"自然"终点而非人为设计**：

1. **架构线自然推动**：Transformer 越来越高效（FlashAttention/Linear Attention），可以处理越来越长的序列，使得"将整个用户历史当做 token 序列"成为可能
2. **特征线自然推动**：Embedding → Semantic ID 让物品有了离散表示，使得"推荐 = 语言生成"成为可能
3. **LLM 验证了路径**：NLP 领域证明了 Transformer + Token 序列 + 自回归生成 可以 scale 到极致，推荐系统沿同样路径前进是自然选择

### 6.4 GR 之后的演进方向

GR 不是终点，而是新起点：

| 方向 | 代表工作 | 核心创新 |
|------|---------|---------|
| 推理增强 | OneRec-Think, PROMISE | 在生成前加推理链，test-time scaling |
| 偏好感知 | Mender | 自然语言偏好条件生成 |
| Agentic | Agentic RecSys | 动态编排推荐流程 |
| 多模态统一 | — | 文本+图像+视频+行为统一 tokenize |
| World Model | — | 模拟用户偏好演化 |

**生成式推荐的四阶段**：
```
阶段 1 (2022-23): 生成式召回 — TIGER/GPR
阶段 2 (2024):    统一召排 — OneRec/MTGR
阶段 3 (2025):    推理增强 — OneRec-Think/PROMISE
阶段 4 (2025-26): 偏好感知+Agent — Mender/Agentic RecSys
```

---

## 7. 面试核心问题

### Q1: 推荐系统中 Transformer 的用法和 NLP 有什么区别？

**答**：三个核心区别。(1) **Attention 语义不同**：NLP 用 softmax 归一化做概率分布，推荐用 ReLU（HSTU）做稀疏激活——推荐场景中相关行为极度稀疏，softmax 反而模糊信号。(2) **输入异构**：NLP 的 token 同质（都是词），推荐的 token 异构（行为+属性+上下文+候选），OneTrans 用 Heterogeneous Attention Mask 处理。(3) **序列长度和分布**：NLP 4K-128K 均匀 token，推荐 50-100K 行为，相关行为极稀疏——SIM/SparseCTR 的"先检索再精算"或"结构化稀疏"是推荐特有的解法。

### Q2: HSTU 的 Pointwise Aggregated Attention 为什么用 ReLU 替代 softmax？

**答**：两个原因。(1) **效率**：softmax 的归一化需要额外的 reduction 操作，ReLU 直接元素级计算。在万级序列上，去 softmax 让推理加速 3-5 倍。(2) **效果**：推荐行为序列中，用户只对少数行为感兴趣（稀疏），softmax 的归一化强制所有权重和为 1，会给不相关行为分配不必要的权重。ReLU 的稀疏性（零输出）天然过滤无关行为。实验证明去掉 softmax 和 LayerNorm 在推荐场景中效果不降反升。

### Q3: Semantic ID 和传统 Item Embedding 的根本区别是什么？

**答**：三个核心区别。(1) **表示形式**：连续向量 vs 离散码字序列（层次化语义）。(2) **推荐范式**：向量相似度检索（ANN）vs 自回归序列生成。(3) **冷启动**：传统 Embedding 需历史交互，Semantic ID 只需内容特征即可编码。**本质区别**：Semantic ID 让推荐变成了"语言模型的 Next Token 预测"问题，打通了推荐和 NLP 的架构统一。

### Q4: 从 DIN 到 HSTU 的 Attention 演进逻辑是什么？

**答**：解决的问题链——DIN 解决"兴趣动态化"（Target Attention），DIEN 解决"时序演化"（AUGRU），SIM 解决"万级序列效率"（两阶段检索），SASRec 解决"全局并行交互"（Self-Attention），HSTU 解决"推荐 Scaling"（ReLU Attention + 去 LayerNorm）。每一步都是在前一步的基础上解决引入的新瓶颈。核心趋势：从"按需查询少数相关行为"到"全量建模整个行为序列"。

### Q5: OneTrans 统一架构为什么优于双流方案（MTGR）？

**答**：OneTrans 将特征交叉和序列建模统一到一个 Transformer 中，通过 Heterogeneous Attention Mask 让四种交互模式（Feature-Feature, Feature-Action, Action-Action, Action-Feature）在同一模型中共存。相比 MTGR 的双流方案：(1) 参数量少 35%（共享底层表示）(2) 推理延迟低 20%（一次 forward vs 两次）(3) 特征与序列的交互更直接（同一个 attention 内直接交互，vs 两流分别计算后 gating 融合）。选择建议：已有 DLRM 历史包袱选 MTGR（低风险迁移），新建选 OneTrans。

### Q6: 从判别式到生成式推荐，特征的角色如何变化？

**答**：判别式时代，特征是"拼接后打分"——user embedding、item embedding、交叉特征拼接起来送入 MLP 出一个分数。生成式时代，特征是"序列化后生成"——所有信息（行为、属性、上下文）被 token 化为一个序列，模型自回归生成目标物品的 token。核心变化：(1) 从"多种特征拼接"到"统一 token 序列" (2) 从"对每个候选逐一打分"到"直接从全空间生成" (3) 特征交叉从"显式设计"（FM/DCN）变成"Transformer 自动学习"。

### Q7: GR 为什么是架构线和特征线的自然交汇点？

**答**：架构线方向——Transformer 越来越高效（FlashAttention → Linear Attention），可以处理越来越长的 token 序列。特征线方向——物品表示越来越离散化（Embedding → Semantic ID），可以被自回归生成。当两条线交汇时：(1) 架构上有足够高效的 Transformer 处理万级行为序列 (2) 特征上有可生成的离散 token 表示物品 (3) 训练范式上有 Scaling Law 支撑 (4) NLP 领域已验证了同一路径。GR 不是人为设计的架构，而是两条独立演进路线自然收敛的结果。

### Q8: 推荐系统的 Scaling Law 和 LLM 的有什么异同？

**答**：**相同**：性能随参数量 $N$ 和数据量 $D$ 的 power law 增长；**不同**：(1) 推荐受延迟约束严格（P99 < 10-20ms），LLM 可容忍 100ms-1s (2) 推荐数据异构（行为+属性+上下文），scaling 效率系数不同 (3) 推荐推理成本与 QPS 成正比（每个请求独立推理），LLM 可以做 prefix caching。HSTU 从 1.5B 到 1.5T 验证了推荐 Scaling Law 存在，但工业落地需要 MoE 稀疏化 + KV Cache + 量化来控制推理成本。

### Q9: 如果你从零设计一个生成式推荐系统，技术栈怎么选？

**答**：分阶段实施。
- **Phase 1**：基于 Meta 开源 HSTU/M-FALCON 搭建基线，Transformer 4-8 层，序列长度 200-500
- **Phase 2**：根据场景选择 MTGR 路线（有 DLRM 历史）或 OneTrans 路线（新建），引入 SMES 做多任务扩展
- **Phase 3**：引入 Semantic ID（<1M 物品用 Soft ID，>10M 用离散 SID），端到端生成
- **Phase 4**：高价值场景引入 PROMISE test-time scaling（Beam Search K=8），试点 Agentic 推荐

### Q10: 推荐中的位置编码为什么需要时间感知？

**答**：NLP 中 token 间隔均匀（按位置递增），推荐中用户行为间隔极不均匀——可能是秒级（连续浏览）到月级（复购）。简单的位置编号丢失了时间信息。(1) SparseCTR 用复合时间编码，每个 head 有独立时间偏置系数，同时编码顺序和周期 (2) CADET 用 Timestamp RoPE，将 RoPE 的位置参数替换为实际时间戳 (3) SynerGen 用 time-aware RoPE，在搜索+推荐统一框架中融合时间信号。趋势：RoPE 是基座，时间感知是推荐的必要增强。

---

## 8. 关键论文索引

| 论文 | 年份 | 所属线 | 核心贡献 |
|------|------|--------|---------|
| Attention Is All You Need | 2017 | 架构 | Transformer 架构 |
| DIN (Zhou et al.) | 2018 | 特征 | Target Attention 进入推荐 |
| DIEN (Zhou et al.) | 2019 | 特征 | Attention + GRU 时序演化 |
| SASRec (Kang & McAuley) | 2019 | 架构+特征 | Self-Attention 序列推荐 |
| MQA (Shazeer) | 2019 | 架构 | KV 共享压缩 |
| SwiGLU Variants (Shazeer) | 2020 | 架构 | 门控 FFN，SwiGLU 成标配 |
| SIM (Pi et al.) | 2020 | 特征 | 万级序列两阶段检索 |
| Scaling Law (Kaplan et al.) | 2020 | 架构 | Scaling Law |
| DLRM (Naumov et al.) | 2019 | 特征 | 工业特征交叉标准 |
| RoPE (Su et al.) | 2021 | 架构 | 旋转位置编码 |
| FlashAttention (Dao et al.) | 2022 | 架构 | IO-aware tiling |
| Chinchilla (Hoffmann et al.) | 2022 | 架构 | 数据 Scaling Law |
| TIGER (Rajput et al.) | 2022 | 特征 | RQ-VAE Semantic ID + 自回归生成 |
| GQA (Ainslie et al.) | 2023 | 架构 | Grouped-Query Attention |
| Spotify GLIDE | 2023 | 特征 | SID 工业部署 |
| HSTU / Actions Speak Louder | 2024 | 架构+特征 | 1.5T 推荐 Transformer，Scaling Law |
| MLA / DeepSeek-V2 | 2024 | 架构 | 低秩联合压缩 KV |
| MTGR (美团) | 2024 | 特征 | 双流融合：HSTU + DLRM |
| OneRec (快手) | 2024 | 特征 | 召回+排序统一生成 |
| UniGRec | 2024 | 特征 | Soft ID，碰撞率→0% |
| DIGER | 2025 | 特征 | Gumbel-Softmax 端到端 SID |
| OneRec-V2 (快手) | 2025-26 | 架构+特征 | 8B Lazy Decoder-Only |
| OneTrans (字节) | 2026 | 架构+特征 | 统一 Transformer |
| LONGER (字节) | 2026 | 架构 | Token Merge 万级序列 |
| TokenMixer-Large (字节) | 2026 | 架构 | 7B-15B Sparse MoE |
| FuXi-Linear | 2026 | 架构 | 推荐专用 Linear Attention |
| SparseCTR (美团) | 2026 | 架构 | 三分支稀疏 Attention |
| PROMISE | 2025 | 特征 | PRM test-time scaling |
| Mender | 2025 | 特征 | 偏好感知条件生成 |

---

## 附录：概念页交叉引用

| 本文涉及概念 | 对应概念页 | 本文覆盖章节 |
|-------------|-----------|------------|
| Attention 变体 | [[attention_in_recsys\|推荐中的注意力机制]] | 2.1, 2.4 |
| 位置编码 | [[Transformer演进]] §3 | 2.2 |
| 效率优化 | [[Transformer演进]] §6 | 2.3 |
| Embedding 表示 | [[embedding_everywhere\|Embedding全景]] | 3.1-3.3 |
| 序列建模 | [[sequence_modeling_evolution\|序列建模演进]] | 2.4, 3.2 |
| Semantic ID | [[Semantic_ID演进知识图谱]] | 3.4 |
| 向量量化 | [[vector_quantization_methods\|向量量化方法]] | 3.4.1 |
| 生成式推荐全景 | [[generative_recsys\|生成式推荐]] | 3.5, 4, 6 |
| 字节 Transformer 演进 | [[bytedance_recsys_transformer]] | 2.4, 3.2, 5 |
| GR 范式统一 | [[生成式推荐范式统一_20260403]] | 4.3-4.5, 6 |

---

> **本文定位**：双线并行视角的模型演进路线图，聚焦"架构怎么变"和"特征怎么变"的同步对照。
> - 架构模块详解 → [[Transformer演进]]
> - 推荐 Attention 详解 → [[attention_in_recsys|推荐中的注意力机制]]
> - 序列建模详解 → [[sequence_modeling_evolution|序列建模演进]]
> - GR 技术全景 → [[generative_recsys|生成式推荐]] + [[生成式推荐系统技术全景_2026]]
> - SID 详解 → [[Semantic_ID演进知识图谱]]
> - 字节架构线 → [[bytedance_recsys_transformer|字节系搜广推Transformer演进]]
