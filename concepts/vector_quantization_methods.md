# 向量量化四大方法：从 Codebook 到 Codebook-Free

> **一句话总结**：向量量化（VQ）把连续向量压缩为离散 token，是生成式推荐（Semantic ID）和图像生成（VQ-GAN）的核心基座。四种主流方法在"是否需要学习码本"这个轴上形成两大阵营。
>
> **为什么要学**：面试问 Semantic ID 必问 RQ-VAE；问生成式推荐必问量化方法选型。理解 VQ 是打通推荐、搜索、多模态生成的底层公共知识。

**相关概念页**：[[embedding_everywhere|Embedding全景]] | [[generative_recsys|生成式推荐]] | [[sequence_modeling_evolution|序列建模演进]]

---

## 0. 为什么需要向量量化？直觉

想象你有一个 128 维的连续向量，表示一个物品的语义。你想用 Transformer 自回归地"生成"这个物品——但 Transformer 的 softmax 输出层只能在**离散词表**上做分类。

**核心矛盾**：连续表示 vs 离散生成。

**解法**：把连续向量"翻译"成一串离散 code（就像把图片压缩成 JPEG），每个 code 是码本里的一个索引。

```
连续向量 [0.3, -0.1, 0.7, ...] → 离散 code [42, 17, 8, 3]
                                    ↑    ↑    ↑   ↑
                                   粗粒度 ────→ 细粒度
```

**类比**：向量量化就像"地址编码"——"中国-北京-海淀-中关村"，每层从粗到细，用有限的"词"描述一个精确位置。

---

## 1. VQ-VAE（基础版，理解其他方法的前置）

### 核心思想

VQ-VAE（Van den Oord et al., 2017）是所有后续方法的起点。它在 VAE 的 latent space 中加一个**向量量化瓶颈**：

```
Input → Encoder → z_e → 量化(找最近码字) → z_q → Decoder → 重建
```

**码本**：一组可学习的向量 $\{e_1, e_2, ..., e_K\}$，$K$ 是码本大小（通常 256-8192）。

**量化操作**：$z_q = e_k$，其中 $k = \arg\min_j \|z_e - e_j\|_2$

### 关键问题：梯度怎么过 argmin？

argmin 不可导 → 用 **Straight-Through Estimator (STE)**：前向走量化路径，反向直接把梯度复制给 encoder。

$$\nabla_{\text{encoder}} \approx \nabla_{z_q} \quad (\text{直接拷贝})$$

### 损失函数

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{\|\text{sg}[z_e] - e_k\|^2}_{\text{码本学习}} + \underbrace{\beta \|z_e - \text{sg}[e_k]\|^2}_{\text{commitment loss}}$$

- sg 是 stop-gradient
- commitment loss 防止 encoder 输出离码字太远

### 问题：码本利用率崩塌

实践中只有少数码字被频繁使用，大量码字"死掉"（codebook collapse）。后续方法主要解决这个问题。

---

## 2. RQ-VAE（残差量化 VAE）⭐ 推荐系统重点

### 核心思想

**类比**：VQ-VAE 是"一次性找最近的词描述你"，RQ-VAE 是"先粗描再细调"——第一层给大方向，后续层修正误差。

```
z → 第1层量化 → 残差r1 → 第2层量化 → 残差r2 → 第3层量化 → ...
     [c1=42]              [c2=17]              [c3=8]
     "电子产品"             "手机"                "旗舰机"
```

### 数学公式

给定 encoder 输出 $z$：

$$r_0 = z$$
$$c_l = \arg\min_j \|r_{l-1} - e_j^{(l)}\|_2, \quad l = 1, ..., L$$
$$r_l = r_{l-1} - e_{c_l}^{(l)}$$

最终近似：$\hat{z} = \sum_{l=1}^{L} e_{c_l}^{(l)}$

每层码本独立，$L$ 层 $K$ 码字的组合空间 = $K^L$（例如 4 层 × 256 码字 = $256^4 \approx 43$ 亿种组合）。

### 在推荐系统中的应用：TIGER Semantic ID

TIGER（Google 2023）用 RQ-VAE 把物品内容特征编码为 Semantic ID：

1. **离线**：物品特征 → Encoder → RQ 量化 → `[c1, c2, c3, c4]`
2. **在线**：Transformer 自回归生成 $P(c_1) \cdot P(c_2|c_1) \cdot P(c_3|c_1, c_2) \cdot P(c_4|c_1, c_2, c_3)$

### 优缺点

| 维度 | 评价 |
|------|------|
| ✅ 优势 | 层次化语义（从粗到细）、组合空间大（碰撞率低）、工业验证（TIGER/Spotify/快手） |
| ❌ 劣势 | 码本需要学习（训练不稳定）、STE 梯度近似有偏、码本利用率需要特殊技巧（EMA/重置） |

📄 详见 [[SemanticID从论文到Spotify部署|../rec-search-ads/rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md]] | [[generative_recsys|生成式推荐]] §2

---

## 3. RQ-KMeans（残差量化 KMeans）

### 核心思想

**类比**：RQ-VAE 是"端到端学量化"，RQ-KMeans 是"先聚类好再用"——把 Encoder 和量化解耦。

```
Step 1: Encoder 提取特征 z（预训练好，冻结）
Step 2: 对 z 做多层 KMeans 残差量化（纯无监督聚类）
```

### 和 RQ-VAE 的区别

| 维度 | RQ-VAE | RQ-KMeans |
|------|--------|-----------|
| 码本学习 | 端到端反向传播 | KMeans 聚类（无梯度） |
| Encoder | 联合训练 | 预训练冻结 |
| 训练复杂度 | 高（需要 STE + commitment loss） | 低（标准 KMeans） |
| 码本质量 | 依赖训练技巧 | 依赖预训练特征质量 |
| 典型应用 | TIGER（端到端生成） | Spotify（两阶段部署） |

### 数学过程

与 RQ-VAE 的量化公式完全相同，区别在于码本 $\{e_j^{(l)}\}$ 通过 KMeans 离线聚类得到，而非梯度更新。

### 适用场景

- 已有高质量预训练 Encoder（如 CLIP、BERT）时，直接在特征上做 KMeans 更简单
- 不需要端到端优化重建损失
- Spotify 的 Semantic ID 系统（GLIDE/NEO）即使用此方案

---

## 4. FSQ（Finite Scalar Quantization）⭐ 码本利用率突破

### 核心思想

**类比**：RQ-VAE 是"从字典里找最像的词"（向量级查找），FSQ 是"每个维度独立四舍五入到最近的整数"（标量级截断）。

**核心创新**：完全去掉码本学习，用简单的 round 操作代替。

```
Encoder 输出 z = [0.7, -0.3, 1.2, ...] (d维连续向量)
                    ↓ 每维独立量化
FSQ 输出:       [1,   -1,   1,   ...]  (每维量化到有限取值)
```

### 数学公式

每个维度 $i$ 有 $L_i$ 个离散取值级别：

$$\hat{z}_i = \text{round}\left(\frac{L_i - 1}{2} \cdot \tanh(z_i)\right)$$

- $\tanh$ 把值压缩到 $[-1, 1]$
- 缩放到 $[-(L_i-1)/2, (L_i-1)/2]$
- round 取整

**隐式码本大小** = $\prod_i L_i$

例如：$d=6$ 维，每维 $L_i=5$ 个级别 → 码本大小 $5^6 = 15625$

### 为什么码本利用率高？

- **没有码本**，不存在"某些码字没被用到"的问题
- 每个维度独立量化 → 组合空间均匀覆盖
- Google 实验（Mentzer et al., 2023）：FSQ 在 ImageNet 上码本利用率接近 100%，而 VQ-VAE 通常只有 10-30%

### 优缺点

| 维度 | 评价 |
|------|------|
| ✅ 优势 | 无需码本学习、码本利用率极高（~100%）、训练简单（无 STE/commitment loss）、推理快 |
| ❌ 劣势 | 无层次语义结构（不像 RQ 从粗到细）、round 操作的 STE 仍有梯度偏差、需要精心设计每维级别数 |

### 在推荐中的潜在应用

- 适合对码本利用率敏感的场景（物品量巨大，不能浪费码字）
- 不需要层次化 Beam Search 的场景
- 可与 RQ 结合：外层 RQ 做层次结构，内层 FSQ 替代每层的 VQ 操作

📄 参考：Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple" (ICLR 2024)

---

## 5. LFQ（Lookup-Free Quantization）⭐ 最新趋势

### 核心思想

**类比**：FSQ 是"每维独立量化到多个级别"，LFQ 是"每维只量化到 ±1（二值化）"——最极端的简化。

```
Encoder 输出 z = [0.7, -0.3, 1.2, -0.8, ...]  (d维)
                    ↓ 每维取符号
LFQ 输出:       [+1,  -1,   +1,  -1,   ...]    (二值)
```

### 数学公式

$$\hat{z}_i = \text{sign}(z_i) = \begin{cases} +1 & \text{if } z_i \geq 0 \\ -1 & \text{if } z_i < 0 \end{cases}$$

**隐式码本大小** = $2^d$

$d=16$ → 码本 $2^{16} = 65536$；$d=18$ → 码本 $2^{18} = 262144$

### 和 FSQ 的关系

LFQ 是 FSQ 的特例（每维 $L_i=2$）。但 LFQ 有额外的训练技巧：

1. **Entropy Loss**：鼓励码字使用均匀分布
   $$\mathcal{L}_{\text{entropy}} = -\sum_k p(k) \log p(k)$$

2. **Commitment Loss**：鼓励 encoder 输出接近 ±1（减少量化误差）
   $$\mathcal{L}_{\text{commit}} = \|z - \hat{z}\|^2$$

3. **Diversity Loss**：防止所有样本量化到同一 code

### 为什么可行？

- **信息论**：$d$ 维二值 = $d$ bits，$d=18$ 就有 26 万种组合，足够编码大多数物品库
- **计算效率**：sign 操作 + bit 操作 → 极快
- **无 codebook lookup**：不需要计算与码字的距离 → 推理零开销

### 优缺点

| 维度 | 评价 |
|------|------|
| ✅ 优势 | 零 lookup 开销、推理最快、码本利用率高、实现极简 |
| ❌ 劣势 | 量化粒度粗（每维只有 2 个值）、需要更高维度补偿、训练需要辅助损失（entropy/diversity） |

### 应用场景

- 图像生成：MaskBit（Tencent 2024）用 LFQ 替代 VQ-GAN 的 codebook，FID 大幅提升
- 推荐系统：适合物品库不太大（<100K）、对推理速度要求极高的场景
- 多模态 Tokenizer：大模型统一离散化时，LFQ 的简洁性是优势

📄 参考：Yu et al., "Language Model Beats Diffusion — Tokenizer is Key to Visual Generation" (ICLR 2024)

---

## 6. 四种方法横向对比

| 维度 | RQ-VAE | RQ-KMeans | FSQ | LFQ |
|------|--------|-----------|-----|-----|
| **码本是否需要学习** | ✅ 端到端学习 | ⚠️ KMeans 聚类 | ❌ 无码本 | ❌ 无码本 |
| **训练复杂度** | 高（STE + EMA + commitment） | 中（KMeans + 冻结 Encoder） | 低（round + STE） | 低（sign + 辅助损失） |
| **码本利用率** | 低~中（10-60%，需技巧） | 中~高（取决于特征质量） | 高（~100%） | 高（~100%） |
| **推理速度** | 较慢（逐层查表） | 较慢（逐层查表） | 快（无查表） | 最快（sign 操作） |
| **语义层次结构** | ✅ 强（从粗到细） | ✅ 强（从粗到细） | ❌ 无（平坦结构） | ❌ 无（平坦结构） |
| **量化精度** | 高（多层残差修正） | 高 | 中（取决于级别数） | 较低（每维仅 2 值） |
| **梯度传播** | STE（有偏近似） | 无梯度（离线聚类） | STE（有偏近似） | STE（有偏近似） |
| **工业验证** | TIGER/快手/Meta | Spotify GLIDE/NEO | Google 图像生成 | MaskBit 图像生成 |
| **推荐系统适用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 7. 推荐系统中的选型建议

### 场景 → 方法映射

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **生成式召回（TIGER 范式）** | RQ-VAE | 需要层次化 Semantic ID + 端到端优化 + Beam Search 解码 |
| **已有强特征表示** | RQ-KMeans | Encoder 已经很好（CLIP/BERT），直接聚类更稳定 |
| **物品库巨大（>1B）** | RQ-VAE/RQ-KMeans | 层次结构 + 组合空间大，碰撞率最低 |
| **对码本利用率敏感** | FSQ | 不浪费码字，每个 code 都有对应物品 |
| **实时性要求极高** | LFQ | sign 操作无 lookup，推理零额外开销 |
| **多模态统一 Tokenizer** | FSQ/LFQ | 简洁 + 与 LLM 词表无缝拼接 |
| **排序模型 ID 表示** | RQ-KMeans → Prefix Ngram | 不需要生成，只需要好的 Embedding 初始化 |

### 工程实践建议

1. **先用 RQ-VAE**（推荐系统默认选择）：TIGER 范式已被广泛验证
2. **码本崩塌解法**：EMA 更新 + 死码字重置 + 正则化；或直接换 FSQ
3. **码本大小**：每层 256-1024 码字 × 4-8 层，总空间 $10^{10}$+ 足够
4. **碰撞优化**：参考 QuaSID（快手），区分良性碰撞（同品类共享）和有害碰撞（语义冲突排斥）

---

## 8. 面试高频问题

1. **RQ-VAE 的"残差"指什么？为什么要用残差？**
   → 残差是上一层量化误差。用残差做多层量化，每层修正上一层的误差，相当于从粗到细逐步逼近原始向量。类比 JPEG 的渐进式编码。

2. **VQ-VAE 的 STE 是什么？为什么需要它？**
   → Straight-Through Estimator。因为 argmin 不可导，前向走量化路径，反向直接把梯度"透传"给 encoder，让 encoder 能学习。这是有偏估计，但实践中有效。

3. **FSQ 和 LFQ 为什么不需要码本？**
   → 它们把量化简化为对每个标量维度的 round/sign 操作。码本是"隐式"的——所有可能的量化值组合构成隐式码本。好处是不需要存储/更新码本，不存在码本崩塌。

4. **码本利用率为什么重要？怎么解决崩塌？**
   → 码本利用率低意味着大量码字没被用到，相当于浪费了离散空间的表达力。解法：(1) EMA 更新码字 (2) 定期重置死码字 (3) 换 FSQ/LFQ (4) 加 entropy 正则化鼓励均匀使用。

5. **TIGER 用 RQ-VAE 而不用 FSQ/LFQ，为什么？**
   → TIGER 需要**层次化语义**（Beam Search 从粗到细搜索）。RQ 天然提供从粗到细的层次结构，FSQ/LFQ 是平坦结构，无法用 Beam Search 层次剪枝。

6. **如果让你设计一个新的 Semantic ID 系统，你会选哪种量化方法？**
   → 要看场景。默认选 RQ-VAE（层次语义 + 端到端 + 工业验证）。如果码本崩塌严重，可以在 RQ 的每层内部用 FSQ 替代 VQ（混合方案）。如果不需要生成式召回（只做 Embedding 初始化），用 RQ-KMeans 更简单稳定。

7. **四种方法的计算复杂度对比？**
   → RQ-VAE/RQ-KMeans 推理时需要逐层查表 $O(L \cdot K \cdot d)$（$L$ 层，$K$ 码字，$d$ 维），FSQ 只需 $O(d)$ 的 round 操作，LFQ 只需 $O(d)$ 的 sign 操作。训练时 RQ-VAE 最复杂（端到端 + STE），RQ-KMeans 最简单（离线 KMeans）。

---

## 演进关系图

```
VQ-VAE (2017) — 基础：单层向量量化
    │
    ├─ RQ-VAE (2022) ── 多层残差量化，层次语义
    │   └─ TIGER/NEO ── 生成式推荐 Semantic ID
    │
    ├─ RQ-KMeans ────── 解耦 Encoder，离线聚类
    │   └─ Spotify ──── 工业两阶段部署
    │
    ├─ FSQ (2023) ────── 去掉码本，标量量化
    │   └─ 图像生成 ──── 码本利用率 ~100%
    │
    └─ LFQ (2024) ────── 极简二值化，Lookup-Free
        └─ MaskBit ──── 视觉 Token 化 SOTA
```

---

## 相关 Synthesis

- [[rec-search-ads/search/synthesis/2026-04-09_generative_retrieval_evolution|2026-04-09_generative_retrieval_evolution]]
- [[rec-search-ads/rec-sys/synthesis/2026-04-09_llm_for_recsys_landscape|2026-04-09_llm_for_recsys_landscape]]
- [[rec-search-ads/ads/synthesis/20260407_CTR_scaling_advances_synthesis|20260407_CTR_scaling_advances_synthesis]]
- [[rec-search-ads/rec-sys/synthesis/20260407_GenRec_advances_synthesis|20260407_GenRec_advances_synthesis]]
- [[rec-search-ads/ads/synthesis/20260411_LLM驱动推荐推理_生成式召回_工业基础设施|20260411_LLM驱动推荐推理_生成式召回_工业基础设施]]
- [[rec-search-ads/ads/synthesis/20260411_llm_native_recsys_and_industrial_infra|20260411_llm_native_recsys_and_industrial_infra]]
- [[rec-search-ads/rec-sys/synthesis/20260411_sequential_and_generative_rec|20260411_sequential_and_generative_rec]]
- [[rec-search-ads/rec-sys/02_rank/synthesis/Boyu个人学习档案|Boyu个人学习档案]]
- [[rec-search-ads/ads/02_rank/synthesis/CTR预估模型工业级实践进展|CTR预估模型工业级实践进展]]
- [[rec-search-ads/rec-sys/02_rank/synthesis/Embedding学习_推荐系统表示基石|Embedding学习_推荐系统表示基石]]
- [[rec-search-ads/rec-sys/04_multi-task/synthesis/LLM增强推荐系统前沿综述|LLM增强推荐系统前沿综述]]
- [[llm-agent/llm-infra/synthesis/LLM推理优化与RAG_Agent前沿综述|LLM推理优化与RAG_Agent前沿综述]]
- [[rec-search-ads/rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署|SemanticID从论文到Spotify部署]]
- [[rec-search-ads/rec-sys/01_recall/synthesis/Semantic_ID演进知识图谱|Semantic_ID演进知识图谱]]
- [[rec-search-ads/rec-sys/01_recall/synthesis/召回系统工业界最佳实践|召回系统工业界最佳实践]]
- [[rec-search-ads/ads/synthesis/工业广告系统生成式革命_20260403|工业广告系统生成式革命_20260403]]
- [[rec-search-ads/ads/synthesis/广告竞价与CTR预估前沿进展|广告竞价与CTR预估前沿进展]]
- [[rec-search-ads/rec-sys/synthesis/推理链RL范式跨域整合_搜广推全景|推理链RL范式跨域整合_搜广推全景]]
- [[rec-search-ads/rec-sys/synthesis/推荐广告生成式范式统一全景|推荐广告生成式范式统一全景]]
- [[rec-search-ads/rec-sys/01_recall/synthesis/推荐系统召回范式演进|推荐系统召回范式演进]]
- [[rec-search-ads/rec-sys/01_recall/synthesis/生成式推荐系统技术全景_2026|生成式推荐系统技术全景_2026]]
- [[rec-search-ads/rec-sys/synthesis/生成式推荐范式统一_20260403|生成式推荐范式统一_20260403]]
- [[rec-search-ads/rec-sys/03_rerank/synthesis/生成式重排与LLM推理增强|生成式重排与LLM推理增强]]
- [[llm-agent/llm-infra/synthesis/知识蒸馏技术整体总结|知识蒸馏技术整体总结]]
- [[rec-search-ads/search/synthesis/端到端生成式搜索前沿_20260403|端到端生成式搜索前沿_20260403]]
