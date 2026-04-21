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

### 建模方式

- 架构类型：Decoder-only（因果自回归），与 GPT 同族
- 序列表示：每个历史物品通过 item embedding + learnable positional encoding 编码为一个 token，整个行为序列构成 token 序列
- 注意力模式：因果掩码（Causal Mask），位置 t 只能 attend 到位置 1..t，确保自回归性质。每一层都是标准 Multi-Head Self-Attention
- 推理方式：支持 KV Cache 增量推理，新物品进入序列时无需重算历史

### 优化目标/损失函数

```
损失函数：Binary Cross-Entropy (BCE) with Negative Sampling

L = -sum_t [ log σ(s(h_t, v_{t+1})) + sum_{j~Neg} log(1 - σ(s(h_t, v_j))) ]

其中：
  h_t = 位置 t 的隐层输出
  v_{t+1} = 下一个真实物品的 embedding
  v_j = 随机采样的负例物品 embedding
  s(·,·) = 内积打分
  σ = sigmoid

训练方式：每个位置独立预测下一项，正例 1 个 + 负例若干（通常 1-5 个）
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

### 建模方式

- 架构类型：两阶段 Retrieval + Attention，非标准 Transformer 架构
- 序列表示：用户完整生命周期行为存储于外部索引（可达 54000 条），不直接输入模型。GSU 检索出 Top-K 子序列后，ESU 用 Target Attention 将子序列聚合为用户兴趣向量
- 注意力模式：ESU 阶段为 Target Attention（以候选物品为 Query，用户子序列为 Key/Value），非自注意力。只做一次 query-to-sequence 的交叉注意力
- 特殊设计：GSU 和 ESU 是两个独立模块，分别优化

### 优化目标/损失函数

```
损失函数：Multi-class Cross-Entropy + 辅助 GSU Loss

主损失（ESU）：
  L_main = -log P(click | user_interest_vector, candidate)
  标准 CTR 预估的 binary CE

辅助损失（GSU）：
  L_gsu = -sum_k relevance(behavior_k, target) * log P_gsu(behavior_k)
  目的：让 GSU 的检索结果尽量接近 ESU 的注意力权重分布
  （soft-SIM 变体中使用，hard-SIM 中 GSU 无梯度）

总损失：L = L_main + α * L_gsu

注：hard-SIM 中 GSU 用类目匹配，无可学参数；soft-SIM 中 GSU 用 embedding 内积，有辅助 loss
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

### 建模方式

- 架构类型：两阶段 Hash Retrieval + Target Attention，但端到端可训练
- 序列表示：每个行为物品通过 embedding 层映射后，再经随机投影矩阵 + sign 函数转化为 m-bit 二进制哈希码。检索阶段在哈希空间操作，注意力阶段回到连续 embedding 空间
- 注意力模式：检索后对 Top-K 子集做标准 Target Attention（候选物品为 Query），与 SIM 的 ESU 相同
- 特殊设计：哈希函数通过 Straight-Through Estimator (STE) 实现梯度回传，使 sign 函数可微分

### 优化目标/损失函数

```
损失函数：BCE + 端到端哈希学习

L = L_ctr + λ * L_hash_reg

主损失：
  L_ctr = -[y * log(p) + (1-y) * log(1-p)]
  标准 CTR 预估 BCE

哈希梯度传播：
  前向：h = sign(W * e)，输出 {-1, +1}^m
  反向：∂L/∂e = ∂L/∂h * W（Straight-Through Estimator）
  即：反向传播时忽略 sign 的不可微性，直接将梯度穿透传回 embedding

正则项（可选）：
  L_hash_reg = ||h - sign(h)||  鼓励哈希码接近 {-1,+1}（减少量化误差）

关键创新：通过 STE 使得哈希检索阶段可以接收来自下游注意力和 CTR loss 的梯度，
实现真正的端到端联合优化（vs SIM 的 GSU 和 ESU 分离优化）
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

### 建模方式

- 架构类型：非参数化哈希采样 + 下游 CTR 模型（如 DIN/DCN），不含 Transformer 注意力层
- 序列表示：用户行为序列在离线阶段通过多组独立哈希函数编码到哈希表中。每个哈希表是一个 bucket → behavior_list 的映射。在线时，候选物品的哈希值直接索引到相关行为
- 注意力模式：无显式注意力计算。"相关性"由哈希碰撞隐式定义——同一个 bucket 中的行为被视为与候选物品相关。多组哈希表的碰撞取并集，等价于 Locality-Sensitive Hashing (LSH) 的近似近邻搜索
- 特殊设计：哈希函数为非参数化的（随机投影 + sign），不参与反向传播

### 优化目标/损失函数

```
损失函数：标准 BCE（CTR 预估损失）

L = -[y * log(p) + (1-y) * log(1-p)]

关键特点：
  - 哈希部分完全非参数化，没有可学习参数，不参与梯度计算
  - 梯度只流经下游 CTR 模型（如 DIN 的 attention pooling 层）
  - 哈希碰撞采样的"注意力"是离散的（命中/未命中），无需 softmax

vs ETA 的本质区别：
  ETA：哈希有参数（通过 STE 学习），需要端到端训练
  SDIM：哈希无参数（随机投影），只需要保证哈希函数的 LSH 性质

工程优势：模型训练和哈希索引构建完全解耦，可独立更新
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

### 建模方式

- 架构类型：两阶段 Consistent Retrieval + Target Attention
- 序列表示：V1 中将行为 embedding 分解为"物品固有部分"（离线缓存）和"交叉特征部分"（压缩为标量 bias）。V2 进一步将相似物品通过层次聚类合并为簇代表向量
- 注意力模式：
  - GSU：简化版 Target Attention，score = q^T * k_cached + bias（与 ESU 公式一致但计算量极低）
  - ESU：完整 Multi-Head Target Attention，对 Top-K 子集做精细建模
- 特殊设计：CP-GSU 的核心在于数学上证明了简化后的打分公式与完整 Target Attention 的排序一致性

### 优化目标/损失函数

```
损失函数：BCE + 一致性正则化

主损失（ESU CTR 预估）：
  L_main = -[y * log(p) + (1-y) * log(1-p)]

一致性正则（V1 核心创新）：
  L_consistency = KL(Attention_weights_ESU || Retrieval_scores_GSU)

  目的：强制 GSU 的检索排序与 ESU 的注意力权重分布对齐
  使得 GSU 检索到的 Top-K 恰好是 ESU 会给高注意力权重的行为

总损失：L = L_main + β * L_consistency

V2 额外损失：
  L_cluster = reconstruction loss，确保聚类压缩后信息损失最小
  L_total = L_main + β * L_consistency + γ * L_cluster

关键区别 vs SIM：
  SIM 的 GSU 和 ESU 各自优化自己的目标（或 GSU 无梯度）
  TWIN 通过一致性 loss 显式约束两阶段对齐
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

### 建模方式

- 架构类型：Encoder-only，低秩分解自注意力（非标准 Transformer）
- 序列表示：物品 embedding 序列经过低秩投影，先映射到 k 个可学习的"潜在兴趣原型"(Latent Interest Prototypes)，再从兴趣原型映射回物品空间。本质是 n 个物品 token 通过 k 个兴趣瓶颈做信息交换
- 注意力模式：
  - 标准自注意力：A = softmax(Q * K^T / sqrt(d))，shape n*n
  - LightSANs：A = softmax(Q * P^T) * softmax(P * K^T)，P 为 k 个兴趣原型，shape (n*k) * (k*n)
  - 本质是将 n*n 的注意力矩阵做低秩近似 A ≈ A1 * A2
- 特殊设计：Decoupled Position Encoding — 位置信息独立于内容做注意力计算，最后才融合

### 优化目标/损失函数

```
损失函数：BCE + 兴趣多样性正则

主损失：
  L_main = -sum_t [log σ(s(h_t, v_{t+1})) + log(1 - σ(s(h_t, v_neg)))]
  标准下一项预测 BCE（与 SASRec 相同）

兴趣多样性正则：
  L_diversity = ||P * P^T - I||_F^2

  其中 P 为 k 个兴趣原型向量构成的矩阵
  鼓励不同兴趣原型互相正交，覆盖不同的兴趣方向
  防止多个原型坍缩到相同的语义子空间

总损失：L = L_main + λ * L_diversity

λ 通常较小（0.01-0.1），主要起正则化作用
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

### 建模方式

- 架构类型：Decoder-only（因果），与 SASRec 相同框架，仅替换注意力计算方式
- 序列表示：与 SASRec 完全相同——item embedding + positional encoding 构成 token 序列
- 注意力模式：
  - 将 Q, K 做 L2 归一化后，利用结合律改变计算顺序
  - 标准：(Q * K^T) * V，先算 n*n 矩阵
  - LinRec：Q * (K^T * V)，先算 d*d 矩阵
  - 因果掩码通过 cumulative sum 实现：S_t = sum_{i=1}^{t} phi(k_i) * v_i^T
- 特殊设计：L2 归一化保证了注意力权重非负且有界，避免了一般线性注意力的"负注意力"问题

### 优化目标/损失函数

```
损失函数：BCE（与 SASRec 完全相同）

L = -sum_t [log σ(s(h_t, v_{t+1})) + log(1 - σ(s(h_t, v_neg)))]

关键点：
  - LinRec 的创新完全在注意力计算层面（用核近似替代 softmax）
  - 损失函数、训练策略、负采样方式均与 SASRec 一致
  - 这意味着 LinRec 可以作为 SASRec 的 drop-in replacement

理论保证：
  作者证明 L2 归一化核在推荐场景下满足：
  1. 非负性：phi(q)^T * phi(k) >= 0（避免负注意力权重）
  2. 有界性：注意力权重和有限，不会发散
  3. 近似质量：在推荐数据分布下，与 softmax 注意力的近似误差有理论上界
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

### 建模方式

- 架构类型：Decoder-only（因果 Transformer），与 SASRec 架构相同
- 序列表示：Pin（图片/内容）的 embedding 序列 + positional encoding，每个位置输出一个用户兴趣向量
- 注意力模式：标准因果自注意力（与 SASRec 相同），架构上没有任何修改
- 特殊设计：创新完全在训练目标上。每个位置的 hidden state 被训练为能预测该位置之后所有未来行为的"长期兴趣表示"，而非仅预测下一项

### 优化目标/损失函数

```
损失函数：Dense All-Action Loss（对所有未来行为的 BCE 求和）

L = -sum_{t=1}^{T} sum_{s=t+1}^{T} [log σ(h_t^T * v_s) + E_{j~Neg}[log(1 - σ(h_t^T * v_j))]]

展开理解：
  对于序列中每个位置 t：
    正例：位置 t 之后的 所有 物品 {v_{t+1}, v_{t+2}, ..., v_T}
    负例：随机采样的物品集合
    loss 是对所有正例和负例的 BCE 求和

vs SASRec 的 Next-Item Loss：
  SASRec：位置 t 只预测 v_{t+1}（1 个正例）
  PinnerFormer：位置 t 预测 v_{t+1}, v_{t+2}, ..., v_T（T-t 个正例）

直觉：
  - Next-Item Loss 训练模型捕捉"短期意图"（下一步做什么）
  - Dense All-Action Loss 训练模型捕捉"长期兴趣"（未来一段时间内会做什么）
  - 后者产生的 user embedding 更稳定，适合离线缓存和长期兴趣召回

工程价值：
  用户 embedding 不需要实时更新（因为编码的是长期兴趣）
  可以每隔几小时甚至每天离线刷新一次，极大降低在线计算成本
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

### 建模方式

- 架构类型：Decoder-only 变体（自定义 Sequential Transducer），非标准 Transformer
- 序列表示：用户行为序列中每个动作（点击、购买、停留等）编码为一个 token，支持多种行为类型混合。通过 Raggified 机制原生支持变长序列，不同用户的序列长度可以不同且无需 padding
- 注意力模式：
  - Pointwise Gated Attention：A = SiLU(Q * K^T + bias) 而非 Softmax(Q * K^T)
  - SiLU 门控是逐元素操作，避免了 softmax 的行归一化
  - 没有归一化意味着每个位置可以独立并行计算，无需 causal mask 的严格顺序依赖
  - 相对时间偏置让模型感知行为的时间间隔（而非仅靠位置编码）
- 特殊设计：Grouped GEMM 将不同长度的序列分组为 GEMM 操作，极大提升 GPU 利用率

### 优化目标/损失函数

```
损失函数：Sampled Softmax + In-batch Negatives + 多任务辅助损失

主损失（Sampled Softmax）：
  L_main = -log [exp(s(h_t, v_pos)) / (exp(s(h_t, v_pos)) + sum_{j∈Neg_batch} exp(s(h_t, v_j)))]

  负例来源：batch 内其他用户的正例物品（in-batch negatives）
  比随机负采样更难，提供更强的对比学习信号

多任务辅助损失：
  L_aux = sum_task w_task * L_task

  包括：
  - 下一项预测（点击）
  - 停留时长回归
  - 转化预测（购买）
  - 多种行为类型的联合预测

总损失：L = L_main + L_aux

关键设计选择：
  - 不用简单 BCE：因为万亿参数模型需要更强的对比信号来区分数十亿物品
  - In-batch negatives：计算效率高，无需额外负采样，且 batch 内的物品往往是"容易混淆"的
  - 多任务：充分利用推荐系统中丰富的多种反馈信号
  - Sampled softmax 在大词表（数十亿物品）上比 full softmax 高效得多
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

### 建模方式

- 架构类型：Encoder-only 变体，层次化混合注意力架构
- 序列表示：用户行为序列分为若干窗口（chunk），每个窗口内用 InnerTransformer 压缩为少量代表 token（Token Merge），压缩后的 token 序列再做全局注意力。同时引入 Global Token 作为全局信息汇聚节点
- 注意力模式：
  - 层次化设计：
    - 底层：局部窗口注意力（Window Attention），每个 chunk 内 O(w^2)
    - Token Merge：InnerTransformer 将 w 个 token 压缩为 r 个（r << w）
    - 顶层：Global Token + 压缩 token 之间做全局注意力
  - Global Token 可以 attend 到所有位置（类似 CLS token 但可学习且多个）
  - 有效复杂度：O(n*w + (n/w)*r^2 + g*(n/w)*r)，其中 g 为全局 token 数
- 特殊设计：Global Token 稳定超长序列的注意力分布，解决长序列中注意力"稀释"问题

### 优化目标/损失函数

```
损失函数：BCE + Token Merge 重建损失 + 辅助未来预测损失

主损失（CTR 预估）：
  L_main = -[y * log(p) + (1-y) * log(1-p)]
  标准 CTR 二分类 BCE

Token Merge 重建损失：
  L_merge = ||Reconstruct(merged_tokens) - original_tokens||^2

  目的：确保 Token Merge 压缩过程中不丢失关键信息
  InnerTransformer 的输出应能重建原始 token 序列
  （类似 autoencoder 的重建目标）

辅助未来预测损失：
  L_future = -sum_{t} log P(future_actions | h_t)

  对每个位置的 hidden state，预测未来若干步的行为
  增强模型对长期依赖的建模能力

总损失：L = L_main + α * L_merge + β * L_future

设计思路：
  - L_main 确保最终预测准确
  - L_merge 确保压缩层不是信息瓶颈（类似 VQ-VAE 的重建目标）
  - L_future 显式鼓励长期依赖建模（类似 PinnerFormer 的 Dense Loss 思想）
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

### 建模方式

- 架构类型：Encoder-Decoder（生成式），与 T5/BART 同族但用于推荐
- 序列表示：
  - Encoder 端：用户完整行为序列作为输入，双向注意力编码所有历史行为的上下文表示
  - Decoder 端：自回归生成推荐物品 ID 序列，每一步生成一个推荐物品
  - 物品通过 VQ-VAE 或 Semantic ID 离散化为 token
- 注意力模式：
  - Encoder：双向全注意力（Bidirectional），充分理解用户历史
  - Decoder：因果自注意力 + 对 Encoder 输出的交叉注意力（Cross-Attention）
  - Encoder 处理长序列可使用高效注意力（如 FlashAttention 或窗口注意力）
- 特殊设计：IPA (Iterative Preference Alignment) 通过奖励模型迭代对齐用户真实偏好，类似 LLM 领域的 RLHF/DPO

### 优化目标/损失函数

```
损失函数：Generative Loss + IPA (Iterative Preference Alignment)

阶段 1 — 生成式预训练：
  L_gen = -sum_{t=1}^{T} log P(item_t | item_1..t-1, user_history)

  标准自回归语言模型损失，但生成的是物品 ID 序列
  类似 GPT 的 next-token prediction，但 token 是物品而非文字

阶段 2 — IPA 偏好对齐：
  训练奖励模型 R(推荐列表, 用户):
    R = f(用户实际反馈信号: 观看时长、点击、转化等)

  对齐损失（类似 DPO）：
    L_ipa = -log σ(R(y_preferred) - R(y_rejected))

    y_preferred = 用户实际喜欢的推荐序列
    y_rejected = 用户不喜欢的推荐序列

  迭代过程：
    1. 用当前模型生成多组推荐列表
    2. 奖励模型打分
    3. 构造偏好对 (preferred, rejected)
    4. 用 DPO-style loss 更新模型
    5. 重复迭代

总训练流程：
  Pretrain(L_gen) → IPA_round_1(L_ipa) → IPA_round_2(L_ipa) → ...

关键创新：
  - 生成式 loss 使模型能"创造"推荐（而非从候选集中选择）
  - IPA 解决了生成式推荐的"exposure bias"和"目标不对齐"问题
  - 奖励模型可以建模复合目标（时长 + 多样性 + 新鲜度 + 生态健康）
  - 迭代对齐逐步逼近最优策略，避免一步到位的不稳定性

vs 传统 Loss 的本质区别：
  - BCE/CE：优化"当前物品是否被点击"（pointwise 或 pairwise）
  - Generative + IPA：优化"整个推荐列表是否让用户满意"（listwise + 长期）
```

---

## 全景对比表

```
模型          | 年份 | 来源    | 路线         | 序列规模  | 复杂度          | 核心技巧                      | 建模方式                          | 损失函数
-------------|------|--------|-------------|---------|----------------|------------------------------|----------------------------------|---------------------------
SASRec       | 2018 | UCSD   | baseline    | ~200    | O(n^2*d)       | 因果掩码+KV Cache             | Causal Decoder                   | BCE (next-item + neg sampling)
SIM          | 2020 | 阿里   | 两阶段检索   | ~54000  | O(L)+O(K*d)    | Category硬检索+精注意力        | Two-stage: 硬检索+Target Attn    | Multi-class CE + 辅助GSU loss
LightSANs    | 2021 | 微软   | 高效注意力   | ~1000   | O(n*k*d)       | 低秩兴趣分解(k个latent)        | Low-rank Encoder                 | BCE + diversity正则
ETA          | 2021 | -      | 哈希检索     | ~10000  | O(L*m)+O(K*d)  | SimHash+Hamming距离            | Hash Retrieval + Target Attn     | BCE + STE端到端哈希学习
PinnerFormer | 2022 | Pinterest| 训练目标   | ~1000   | O(n^2*d)       | Dense All-Action Loss          | Causal Decoder                   | Dense All-Action BCE
SDIM         | 2022 | 美团   | 哈希采样     | ~10^5   | O(1)在线查表    | 多轮哈希碰撞采样(离线)          | Non-parametric Hash + CTR模型    | BCE (标准CTR, 哈希无梯度)
LinRec       | 2023 | -      | 线性注意力   | ~10000  | O(n*d^2)       | L2归一化核函数                  | Causal Decoder (kernel approx)   | BCE (同SASRec)
TWIN         | 2023 | 快手   | 两阶段检索   | ~10^5   | O(L)+O(K*d)    | CP-GSU一致性检索               | Consistent Retrieval + Attn      | BCE + consistency正则
TWIN V2      | 2024 | 快手   | 两阶段检索   | ~10^6   | O(L/C)+O(K*d)  | 层次聚类压缩                   | Cluster Retrieval + Attn         | BCE + consistency + cluster重建
HSTU         | 2024 | Meta   | 自定义架构   | ~8192+  | 5-15x快于FA2   | SiLU门控+Raggified+万亿参数    | Pointwise Gated Decoder          | Sampled Softmax + 多任务辅助
LONGER       | 2025 | 字节   | 端到端       | ~10^5   | 亚二次          | Global Token+Token Merge       | Hierarchical Encoder (hybrid)    | BCE + merge重建 + future预测
OneRec       | 2025 | 快手   | 全流程统一   | ~10^5+  | Enc-Dec        | 生成式替代多阶段流水线          | Enc-Dec Generative               | Generative Loss + IPA (DPO-style)
```

## 损失函数演进总结

```
演进方向：Pointwise BCE → Dense BCE → Sampled Softmax → Generative + Alignment

第一代（BCE 主导）：
  SASRec/SIM/ETA/SDIM/TWIN/LightSANs/LinRec
  核心：每个样本独立的 binary 预测 → 优化单物品粒度的点击率
  局限：只建模即时反馈，无法捕捉长期兴趣和列表效应

第二代（Dense/多任务）：
  PinnerFormer (Dense All-Action), HSTU (多任务), LONGER (辅助预测)
  核心：每个位置预测多个目标 → 更丰富的监督信号
  优势：长期兴趣建模、多种行为联合建模

第三代（生成式+对齐）：
  OneRec (Generative + IPA)
  核心：列表级生成 + 奖励模型对齐 → 优化整体推荐质量
  优势：listwise 优化、可建模复合目标、类 LLM 的 scaling 能力
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

Q: SASRec 的损失函数和 BERT4Rec 有什么区别？为什么 SASRec 用 BCE 而 BERT4Rec 用 CE？
A: SASRec 是因果自回归模型，用 BCE + 负采样（每个位置预测下一项是/否为正例）。BERT4Rec 是 Masked Language Model，用 full softmax CE（从全物品空间中选出被 mask 的物品）。SASRec 的 BCE 适合在线增量推理（新行为来了直接续推），BERT4Rec 的 CE 需要重编码整个序列但理论上能建模双向依赖。工业上 SASRec 范式更主流（因为在线推理友好）。

Q: SIM 的 GSU-ESU 不一致问题是什么？TWIN 如何解决？
A: SIM 的 GSU 用 category 硬匹配检索行为，ESU 用 embedding 注意力建模。两者的"相关性"定义不同，GSU 可能漏掉语义相关但类目不同的行为。TWIN 提出 CP-GSU，将行为特征拆为"可预计算部分"和"压缩为 1D bias 的交叉部分"，使 GSU 能运行和 ESU 相同的 Target Attention 公式，计算量减少 99.3% 同时保证一致性。

Q: SDIM 为什么比 ETA 更快？
A: ETA 在线仍需 O(L*m) 的 Hamming 距离计算。SDIM 将哈希完全放到离线/异步，在线只需 O(1) 查哈希表找碰撞行为，本质是把"计算"变成了"查表"。

Q: PinnerFormer 的 Dense All-Action Loss 相比 Next-Item Loss 有什么好处？
A: Next-Item Loss 只优化"预测下一步"，模型倾向于捕捉短期意图。Dense All-Action Loss 让每个位置预测所有未来行为，强制模型编码长期兴趣。产生的 user embedding 更稳定，适合离线缓存（Pinterest 每天只需刷新一次用户表示）。代价是训练时间增加（正例数量从 T 变为 T^2/2）。

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

Q: OneRec 的"统一"意味着什么？IPA 和 RLHF 什么关系？
A: 传统推荐系统是四阶段流水线（召回→粗排→精排→重排），每阶段独立优化。OneRec 用一个 Encoder-Decoder 模型端到端替代整个流水线。IPA (Iterative Preference Alignment) 本质上就是推荐领域的 DPO——用奖励模型定义"好的推荐列表"，然后通过偏好对构造对比学习信号，迭代优化生成策略。与 LLM 的 RLHF/DPO 一脉相承，但奖励信号来自用户行为（观看时长、转化率等）而非人工标注。
