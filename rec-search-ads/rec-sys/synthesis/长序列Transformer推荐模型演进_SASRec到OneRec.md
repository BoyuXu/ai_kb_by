# 长序列 Transformer 推荐模型演进：从 SASRec 到 LONGER/OneRec

## 演进主线

```
SASRec (2018, O(n^2) baseline)
  │
  ├─→ 两阶段检索路线
  │     SIM (2020, 硬检索+精注意力)
  │     → ETA (2021, SimHash 端到端)
  │     → SDIM (2022, 哈希碰撞采样)
  │     → TWIN (2023, GSU-ESU 一致性)
  │     → TWIN V2 (2024, 聚类压缩至 10^6)
  │
  ├─→ 高效注意力路线
  │     LightSANs (2021, 低秩分解)
  │     LinRec (2023, L2 归一化线性注意力)
  │     Linformer/Performer (通用→推荐迁移)
  │
  ├─→ 自定义架构路线
  │     PinnerFormer (2022, Dense All-Action Loss)
  │     HSTU (2024, Meta, 非 Softmax + 万亿参数)
  │     LONGER (2025, 字节, Global Token + Token Merge)
  │
  └─→ 全流程统一路线
        OneRec (2025, 快手, Encoder-Decoder 取代多阶段)
```

---

## 1. SASRec — Baseline (2018)

论文：Self-Attentive Sequential Recommendation (ICDM 2018, UCSD)

核心思想：单向 Transformer 解码器 + 因果掩码，自回归预测下一项。

```
架构：Item Embedding + Learnable PE → Causal Self-Attention × L → FFN → Next Item Prediction
复杂度：O(n^2 * d)，n = 序列长度，d = 隐层维度
序列上限：通常 50-200，超过则截断
```

价值：奠定了序列推荐用 Transformer 的范式，KV Cache 增量推理使其成为工业主流 baseline。

局限：O(n^2) 使其无法处理千级以上的长序列。

---

## 2. SIM — 两阶段检索开山之作 (2020)

论文：Search-based User Interest Modeling with Lifelong Sequential Behavior Data (CIKM 2020, 阿里)

核心思想：先检索再注意力。将长序列问题转化为「检索 + 精排」两阶段。

```
Stage 1 — GSU (General Search Unit):
  从完整行为序列 (L ~ 54000) 中，用 category 硬匹配检索出 Top-K 相关行为子集
  复杂度：O(L) 简单查找

Stage 2 — ESU (Exact Search Unit):
  对 Top-K 子集 (K ~ 100) 做 Multi-Head Target Attention
  复杂度：O(K * d)，K << L

效果：CTR +7.1%（阿里线上）
```

关键缺陷：GSU 用 category 硬匹配，ESU 用 embedding 注意力，两阶段的相关性度量不一致，导致 GSU 可能漏掉真正相关的行为。

---

## 3. ETA — 端到端哈希检索 (2021)

论文：End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model (2021)

核心思想：用 SimHash + Hamming 距离替代内积检索，实现端到端可训练。

```
流程：
1. 对所有物品计算 m-bit SimHash 签名：sign(W_random * item_emb)
2. 用 Hamming 距离（XOR + popcount）替代内积，检索 Top-K
3. 对 Top-K 做标准 Target Attention

复杂度：
  检索：O(L * m) 位运算（vs SIM 的 O(L * d) 内积 或 O(L) 硬匹配）
  注意力：O(K * d)

优势 vs SIM：
  - 端到端可训练，GSU 和 ESU 联合优化
  - 比内积快得多（位运算 vs 浮点乘法）

效果：GMV +3.1%（电商线上）
```

---

## 4. SDIM — 哈希碰撞采样 (2022)

论文：Sampling Is All You Need on Modeling Long-Term User Behaviors (CIKM 2022, 美团)

核心思想：连距离计算都不要了。直接用哈希碰撞"免费"找到相关行为。

```
架构：
1. BSE (Behavior Sequence Encoding)：离线/异步对所有行为做多轮哈希
2. 推理时：计算候选物品哈希 → 直接查表找碰撞行为 → 作为特征输入 CTR 模型

复杂度：
  离线哈希：O(L)，完全不占在线延迟
  在线查表：O(1) per hash table × m tables

vs ETA：
  ETA 仍需 O(L*m) 在线 Hamming 距离计算
  SDIM 在线完全是 O(1) 查表

效果：与注意力方法持平，延迟显著降低。美团线上部署。
```

---

## 5. TWIN / TWIN V2 — 一致性两阶段 (2023/2024)

论文 V1：TWIN: TWo-stage Interest Network (KDD 2023, 快手)
论文 V2：TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling (CIKM 2024, 快手)

核心思想 V1：解决 SIM 的 GSU-ESU 不一致问题。让 GSU 使用和 ESU 完全相同的相关性度量。

```
V1 — CP-GSU (Consistency-Preserved GSU):
  将行为特征拆分为：
  (a) 视频固有特征 → 离线预计算，缓存 embedding
  (b) 用户-物品交叉特征 → 压缩为 1D bias 项

  这样 GSU 能以极低成本运行 "相同的" Target Attention 公式
  计算量减少 99.3%，但检索一致性大幅提升

V2 — 扩展到百万级序列:
  离线层次聚类压缩：相似物品合并为簇
  序列从 10^5 → 10^5/C（C 为压缩比）
  GSU 在压缩序列上运行

复杂度：
  V1: O(L) 线性投影(预缓存) + O(K*d) ESU
  V2: O(L/C) 压缩 GSU + O(K*d) ESU

效果：快手数亿 DAU 线上部署
```

---

## 6. LightSANs — 低秩兴趣分解 (2021)

论文：Lighter and Better: Low-Rank Decomposed Self-Attention Networks (SIGIR 2021, 微软/人大)

核心思想：不做 n*n 的物品间注意力，改为通过 k 个潜在兴趣向量做中转。

```
标准注意力：n*n（物品→物品）
LightSANs：n*k + k*n（物品→兴趣→物品），k 为兴趣数（10-20）

额外创新：解耦位置编码 (Decoupled PE)
  分离内容信号和位置信号，避免位置编码污染语义表示

复杂度：O(n * k * d)，k 为常数，对 n 线性
```

---

## 7. LinRec — L2 归一化线性注意力 (2023)

论文：LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems (SIGIR 2023)

核心思想：用 L2 归一化替代 Softmax，理论上保持点积注意力的等价性质，实现线性复杂度。

```
标准：Softmax(Q * K^T) * V          → O(n^2 * d)
LinRec：phi(Q) * (phi(K)^T * V)     → O(n * d^2)

其中 phi = L2 Normalization

关键数学洞察：
  先算 K^T * V（d*d 矩阵），再与 Q 相乘
  d 通常为 64-256，远小于 n（数千）

vs Performer/Linformer：
  LinRec 专门针对推荐特性设计 phi 函数
  通用线性注意力在推荐上效果往往不如专用方案
```

---

## 8. PinnerFormer — Dense All-Action Loss (2022)

论文：PinnerFormer: Sequence Modeling for User Representation at Pinterest (KDD 2022)

核心思想：不优化"预测下一项"，而是"从每个位置预测所有未来行为"，学习长期兴趣表示。

```
训练目标：
  SASRec: L = -sum_t log P(item_{t+1} | item_1..t)       # 只预测下一个
  PinnerFormer: L = -sum_t sum_{s>t} log P(item_s | item_1..t)  # 预测所有未来

架构：标准因果 Transformer，变化在 loss 不在结构

优势：
  - 用户 embedding 编码长期兴趣（不只是短期偏好）
  - 可以离线批量计算用户 embedding，缓存复用
  - 在线推理只需查缓存的 embedding

补充：TransAct (KDD 2023) 负责短期兴趣，与 PinnerFormer 的长期表示互补
```

---

## 9. HSTU — Meta 万亿参数方案 (2024)

论文：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (2024, Meta)

核心思想：为推荐专门设计的 Transformer 变体。抛弃 Softmax，用 Pointwise Gated Attention + 极致内存优化。

```
三个关键设计：

1. Pointwise Attention（非 Softmax）：
   用 SiLU 门控替代 Softmax：output = SiLU(Q*K^T) * V
   避免 Softmax 在超长序列上的数值不稳定

2. 相对时间偏置：
   Attention_bias = f(position_diff, timestamp_diff)
   同时编码位置距离和时间间隔

3. Raggified Attention（不规则注意力）：
   变长序列原生支持，无需 padding 浪费
   通过 Grouped GEMM 实现高效计算

性能：
  比 FlashAttention2 快 5.3-15.2x（8192 长度）
  每层只需 14d bfloat16 激活内存
  支持万亿参数规模
  NDCG 提升高达 65.8%（公开数据集）
```

---

## 10. LONGER — 字节端到端方案 (2025)

论文：LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders (RecSys 2025, 字节/抖音)

核心思想：拒绝两阶段检索的"上下游不一致"。通过三个机制直接端到端建模超长序列。

```
三个核心机制：

1. Global Token 全局令牌：
   可学习的辅助 token，能 attend 到所有位置
   稳定超长序列的注意力分布

2. Token Merge 令牌合并：
   InnerTransformer 将相邻 token 组压缩为紧凑表示
   降低有效序列长度，再做全局注意力

3. Hybrid Attention 混合注意力：
   局部窗口注意力 + 全局 token 注意力
   避免 O(n^2) 全注意力

vs 两阶段方法 (SIM/TWIN):
  无上下游不一致问题
  端到端联合优化

效果：抖音广告 CVR 预估，52 亿样本训练，线上部署
```

---

## 11. OneRec — 全流程统一 (2025)

论文：OneRec: Unifying Retrieve and Rank with Generative Recommender (2025, 快手)

核心思想：Encoder-Decoder 生成式架构，取代传统的 召回→粗排→精排→重排 多阶段流水线。

```
架构：
  Encoder: 编码超长用户行为序列
  Decoder: 自回归生成推荐物品序列
  + IPA (Iterative Preference Alignment): 奖励模型对齐用户偏好

效果：
  总观看时长 +1.68%
  平均观看时长 +6.56%
  运营成本降至传统流水线的 10.6%

V2 (2025): 开源基座模型 1.7B/8B（Qwen3 backbone）

意义：从"优化 Transformer 效率"进化到"用一个模型替代整个推荐系统"
```

---

## 全景对比表

```
模型          | 年份 | 来源    | 路线         | 序列规模  | 复杂度          | 核心技巧
-------------|------|--------|-------------|---------|----------------|------------------
SASRec       | 2018 | UCSD   | baseline    | ~200    | O(n^2*d)       | 因果掩码+KV Cache
SIM          | 2020 | 阿里   | 两阶段检索   | ~54000  | O(L)+O(K*d)    | Category硬检索+精注意力
LightSANs    | 2021 | 微软   | 高效注意力   | ~1000   | O(n*k*d)       | 低秩兴趣分解(k个latent interest)
ETA          | 2021 | -      | 哈希检索     | ~10000  | O(L*m)+O(K*d)  | SimHash+Hamming距离
PinnerFormer | 2022 | Pinterest| 训练目标   | ~1000   | O(n^2*d)       | Dense All-Action Loss
SDIM         | 2022 | 美团   | 哈希采样     | ~10^5   | O(1)在线查表    | 多轮哈希碰撞采样(离线哈希)
LinRec       | 2023 | -      | 线性注意力   | ~10000  | O(n*d^2)       | L2归一化核函数
TWIN         | 2023 | 快手   | 两阶段检索   | ~10^5   | O(L)+O(K*d)    | CP-GSU一致性检索
TWIN V2      | 2024 | 快手   | 两阶段检索   | ~10^6   | O(L/C)+O(K*d)  | 层次聚类压缩
HSTU         | 2024 | Meta   | 自定义架构   | ~8192+  | 5-15x快于FA2   | SiLU门控+Raggified+万亿参数
LONGER       | 2025 | 字节   | 端到端       | ~10^5   | 亚二次          | Global Token+Token Merge+Hybrid
OneRec       | 2025 | 快手   | 全流程统一   | ~10^5+  | Enc-Dec        | 生成式替代多阶段流水线
```

## 技术路线分类

```
路线 1: 两阶段检索（精度优先）
  SIM → ETA → SDIM → TWIN → TWIN V2
  核心矛盾：检索阶段和建模阶段的一致性
  演进方向：从硬匹配→哈希→一致性检索→聚类压缩

路线 2: 高效注意力（端到端优先）
  Linformer/Performer → LightSANs → LinRec → LONGER
  核心矛盾：降低复杂度 vs 保持注意力质量
  演进方向：通用线性注意力→推荐专用核→混合注意力

路线 3: 架构重设计（规模优先）
  SASRec → PinnerFormer → HSTU → OneRec
  核心矛盾：推荐场景需要什么样的注意力？
  演进方向：Softmax→Dense Loss→非Softmax→生成式
```

## 面试高频问题

Q: SIM 的 GSU-ESU 不一致问题是什么？TWIN 如何解决？
A: SIM 的 GSU 用 category 硬匹配检索行为，ESU 用 embedding 注意力建模。两者的"相关性"定义不同，GSU 可能漏掉语义相关但类目不同的行为。TWIN 提出 CP-GSU，将行为特征拆为"可预计算部分"和"压缩为 1D bias 的交叉部分"，使 GSU 能运行和 ESU 相同的 Target Attention 公式，计算量减少 99.3% 同时保证一致性。

Q: SDIM 为什么比 ETA 更快？
A: ETA 在线仍需 O(L*m) 的 Hamming 距离计算。SDIM 将哈希完全放到离线/异步，在线只需 O(1) 查哈希表找碰撞行为，本质是把"计算"变成了"查表"。

Q: HSTU 为什么不用 Softmax？
A: Softmax 在超长序列上有数值不稳定问题（指数运算溢出），且梯度饱和。HSTU 用 SiLU 门控做 Pointwise Attention，避免归一化的信息损失，配合 Raggified Attention 支持变长序列无 padding 浪费。在万亿参数规模下比 FlashAttention2 快 5-15 倍。

Q: LONGER 和 SIM/TWIN 相比，为什么选择端到端？
A: 两阶段方法的根本问题是"上下游不一致"——检索阶段的优化目标与最终推荐目标不完全对齐。LONGER 通过 Global Token（全局视野）+ Token Merge（压缩长度）+ Hybrid Attention（局部+全局）三个机制，在保持端到端可训练的同时，将复杂度降到亚二次。代价是工程实现更复杂。

Q: 线性注意力 (LinRec) 和两阶段方法 (TWIN)，工业上怎么选？
A: 取决于序列长度和系统约束：
- 序列 < 1000：LinRec 更简洁，端到端无额外系统组件
- 序列 10^4-10^5：TWIN 更成熟，工业验证充分（快手数亿 DAU）
- 极端长序列 10^6：TWIN V2 是目前唯一方案
- 追求架构简洁且有 GPU 资源：LONGER

Q: OneRec 的"统一"意味着什么？
A: 传统推荐系统是四阶段流水线（召回→粗排→精排→重排），每阶段独立优化。OneRec 用一个 Encoder-Decoder 模型端到端替代整个流水线。Encoder 编码用户行为，Decoder 直接生成推荐列表。运营成本降至传统方案的 10.6%，但对算力和模型能力要求极高。
