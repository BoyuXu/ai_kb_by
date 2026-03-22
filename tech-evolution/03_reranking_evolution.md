# 重排技术演进脉络（Reranking Evolution）

> 整理者：MelonEggLearn | 更新时间：2026-03-16
> 覆盖：规则时代 → 统计多样性 → 深度学习重排 → 生成式重排

---

## ASCII 时间线

```
2015          2017          2019          2021          2023          2025
 |             |             |             |             |             |
 |  规则时代   |  统计多样性  |   深度学习重排           |  生成式重排  |
 |             |             |             |             |             |
 +─────────────+─────────────+─────────────+─────────────+─────────────>
 │去重/打散    │MMR算法      │PRM模型      │SetRank      │LLM重排      │
 │业务规则     │DPP行列式    │DLCM         │上下文感知   │Listwise生成 │
 │多样性插入   │ε-greedy探索 │DeepFM+序列  │MISL/PGRank  │RLHF优化     │
```

---

## 阶段1：规则时代（2015–2017）

### 背景

2015年前后，推荐系统的召回+精排已经相对成熟，但工程师们发现一个普遍现象：**精排分高的若干条目大量重复（同类目、同作者、同话题），用户体验差，点击率反而下降**。重排（Re-ranking）作为独立模块被提出，介于精排与展示之间，承担最终的列表整形工作。

此时学术界和工业界都缺乏统一的理论框架，主要依赖产品经理和工程师的经验规则。

### 核心方法

#### 1.1 去重（De-duplication）
- 同一 item_id 或同一 source_id 出现多次时，保留分最高的，其余下沉或删除。
- 实现简单：`seen = set(); filtered = [x for x in list if x.id not in seen and not seen.add(x.id)]`

#### 1.2 业务规则打散（Business-rule Scatter）
- **类目打散**：同一 L2/L3 类目在列表中间隔 ≥ K 个位置（如 K=3）。
- **品牌打散**：同一品牌不超过连续 2 条。
- **时间打散**：内容发布时间过于接近的同来源内容，保留最新一条。
- 实现方式：贪心插入、滑动窗口约束。

#### 1.3 多样性插入（Diversity Injection）
- 人工维护"探索池"，强制在 Top-K 结果中插入占比约 10%–20% 的多样性条目（长尾、冷启动内容）。
- 典型策略：每 5 个 position 插入 1 个探索位。

#### 1.4 业务加权（Business Boosting）
- 对特定广告位、新品、平台战略品类进行人工加权或置顶。
- 规则形如：`score_final = score_model * boost_factor(item)`

### 局限性

| 问题 | 描述 |
|------|------|
| 规则爆炸 | 类目多时规则条数呈指数增长，维护困难 |
| 无法个性化 | 所有用户执行相同规则，无法感知用户兴趣 |
| 规则冲突 | 打散规则之间相互矛盾，优先级难以确定 |
| 缺乏全局最优 | 贪心策略只能保证局部约束，无法优化列表整体质量 |
| 冷启动友好性差 | 新品/新作者仅靠人工维护，成本高 |

### 代表工作

- **YouTube（2016）**：在精排后增加多样性约束层，使用 slot-filling 贪心算法，避免同频道内容扎堆。
- **淘宝（2017前）**：类目打散 + 品牌打散双重规则，结合运营干预。
- **今日头条**：内容标签频率约束 + 强制时间多样性。

### 面试必考点

1. **Q：规则打散的本质目的是什么？**
   A：在保持相关性的前提下，增加列表多样性，避免信息茧房，提升用户浏览深度（dwell time）和长期留存。

2. **Q：业务规则打散有哪些常见实现？**
   A：类目/品牌/来源的 sliding window 约束，贪心插入（优先高分，违约则跳过，填入次高分），或基于优先队列的多路归并。

3. **Q：规则方案最大的工程痛点是什么？**
   A：规则膨胀、冲突仲裁难、AB实验困难（规则变更影响面难评估）、无法自动适应流量变化。

---

## 阶段2：统计多样性（2018–2019）

### 背景

规则时代的瓶颈推动了更有理论依据的多样性优化方法。这一阶段两条主线并行发展：
- **信息检索领域** 的 MMR（Maximal Marginal Relevance）被引入推荐；
- **概率论/线性代数领域** 的 DPP（Determinantal Point Process）提供了严格的多样性建模框架。

这一阶段的核心思想：**用数学语言定义"多样性"，将多样性与相关性统一在一个优化目标中**。

---

### 2.1 MMR（Maximal Marginal Relevance）

#### 算法背景
MMR 最早由 Carbonell & Goldstein（1998）提出，用于文本摘要，2018–2019 年被大量引入推荐重排。

#### 核心公式

```
MMR(d_i) = λ · Sim₁(d_i, q) - (1-λ) · max_{d_j ∈ S} Sim₂(d_i, d_j)
```

其中：
- `d_i`：候选 item
- `q`：用户 query / 用户兴趣向量
- `S`：已选入结果集的 items
- `Sim₁(d_i, q)`：item 与用户兴趣的相关性（可用精排分代替）
- `Sim₂(d_i, d_j)`：候选 item 与已选 item 的相似度（余弦相似度）
- `λ ∈ [0,1]`：相关性与多样性的权重平衡因子

#### 迭代选择过程

```
算法：MMR贪心选择
Input: 候选集 R, 用户查询 q, 参数 λ, 输出数量 k
Output: 有序结果列表 S

S = []
while len(S) < k:
    d* = argmax_{d_i ∈ R\S} [λ·Sim₁(d_i,q) - (1-λ)·max_{d_j∈S} Sim₂(d_i,d_j)]
    S.append(d*)
return S
```

#### 关键性质
- **贪心近似**：每步选择局部最优，不保证全局最优，但实践效果良好。
- **λ 调参**：λ=1 退化为纯相关性排序；λ=0 退化为纯多样性最大化。
- **时间复杂度**：O(k·n)，n 为候选集大小，k 为输出列表长度。
- **相似度定义灵活**：可用 item embedding 余弦、类目one-hot jaccard、文本 BM25 等。

#### 工业实践变体
- **加权 MMR**：不同特征维度（类目、品牌、内容）设置不同 λ。
- **位置感知 MMR**：前几个位置强调相关性，后续位置加大多样性权重。
- **用户级 λ**：根据用户多样性偏好动态调整 λ（新用户 λ 大，老用户 λ 小）。

---

### 2.2 DPP（Determinantal Point Process）行列式点过程

#### 算法背景
DPP 是概率论中的排斥过程模型，最早用于量子物理，由 Kulesza & Taskar（2012）系统化后引入机器学习。**核心直觉：用行列式度量子集的"体积"，高质量且多样的子集行列式值更大**。

#### 核心数学

设 `L` 为半正定核矩阵（Kernel Matrix），`L ∈ ℝⁿˣⁿ`：

```
L_ij = quality(i) · similarity(i,j) · quality(j)
```

对于子集 Y，DPP 的概率正比于其核子矩阵的行列式：

```
P(Y) ∝ det(L_Y)
```

其中 `L_Y` 是 `L` 关于子集 Y 对应行列的主子矩阵。

#### 行列式的几何意义

```
det(L_Y) = (vol of parallelotope spanned by {v_i : i ∈ Y})²
```

向量之间越正交（越不相似）→ 体积越大 → 行列式越大 → 该子集被选中概率越大。

这天然编码了**多样性**：相似的 item 对应近似平行的向量，压缩体积，降低行列式。

#### 分解核矩阵（Decompose L）

实践中将 L 分解为质量分和多样性两部分：

```
L_ij = q_i · φ_i^T · φ_j · q_j
```

- `q_i = f(relevance_score_i)`：item 质量（精排得分的单调变换）
- `φ_i`：item 的归一化 embedding 向量（内容、类目、风格等）
- `φ_i^T · φ_j`：item 间相似度（∈[-1,1]）

#### MAP 推断（最大后验子集选择）

从 DPP 中找最优 k-DPP 子集（最大化 `det(L_Y)`, `|Y|=k`）是 NP-hard，工业界常用**贪心近似**：

```python
# 贪心 k-DPP
S = {}
for _ in range(k):
    best = argmax_{i ∉ S} det(L_{S∪{i}}) / det(L_S)
    S.add(best)
```

利用 Cholesky 分解可以将单步更新加速到 O(|S|²)。

#### 工业实践（阿里、腾讯等）

- **阿里 2018**：将 DPP 引入淘宝首页重排，同类目商品 MMR 仅靠分数差异不足，DPP 通过行列式全局衡量集合多样性，AUC 提升约 0.3%，CTR +1.2%。
- **腾讯微信看一看**：文章主题 embedding + DPP，有效降低同话题扎堆，用户浏览深度 +8%。

#### DPP vs MMR 对比

| 维度 | MMR | DPP |
|------|-----|-----|
| 多样性建模 | 局部（与已选集最大相似度） | 全局（子集行列式）|
| 最优性 | 贪心近似 | MAP 近似 |
| 计算复杂度 | O(kn) | O(k²n) |
| 理论基础 | 启发式 | 概率论 |
| 实现难度 | 低 | 中 |
| 质量-多样性联合建模 | 显式加权 λ | 隐式通过核矩阵 |

### 局限性

| 问题 | 描述 |
|------|------|
| 无法感知用户偏好 | 相似度矩阵对所有用户相同，无个性化 |
| embedding质量依赖 | φ 的质量决定多样性度量准确性 |
| 长列表计算昂贵 | DPP行列式计算 O(k³) 串行，难以工程优化 |
| 静态优化 | 不考虑 item 在列表中的位置效应 |
| 冷启动 | 新 item 无可靠 embedding |

### 代表工作

- **Carbonell & Goldstein (1998)**：MMR 原始论文（IR领域）
- **Kulesza & Taskar (2012)**：DPP in Machine Learning（综述）
- **阿里 DPP 重排（2018）**：首次大规模工业 DPP 应用
- **Chen et al. (2018) Fast Greedy MAP Inference for DPP**：工业可用的快速 DPP 推断

### 面试必考点

1. **Q：DPP 行列式为什么能衡量多样性？**
   A：行列式等于向量组张成的平行多面体体积的平方。相互正交的向量体积最大；高度相似（接近线性相关）的向量体积趋近于0。因此，子集内 item embedding 越不相似，行列式越大，该子集被 DPP 选中的概率越高。

2. **Q：MMR 的 λ 如何设置？线上如何调参？**
   A：离线通过多样性指标（ILD, Intra-List Diversity）和相关性指标（NDCG）的帕累托前沿确定候选范围；线上用 AA/AB 实验，多段时间窗口观察用户浏览深度、7日留存等长期指标，因为短期 CTR 可能与多样性负相关。

3. **Q：DPP 在工业界如何解决计算效率问题？**
   A：① 贪心近似替代精确 MAP；② Cholesky 分解增量更新，复用已选集的分解结果；③ 候选截断（只对精排 Top-200 做 DPP，而非全量）；④ GPU 并行矩阵运算；⑤ 部分公司用 FPGA 加速行列式计算。

---

## 阶段3：深度学习重排（2020–2022）

### 背景

2020年开始，重排进入深度学习时代。核心动机：

1. **上下文感知**：一个 item 的展示价值取决于它**和哪些 item 一起展示**，这是 pointwise/pairwise 精排模型无法建模的。
2. **个性化多样性**：不同用户对多样性的偏好差异巨大，需要从数据中学习个性化的多样性权衡。
3. **序列建模能力**：Transformer 在 NLP 的成功激发了将 item 列表视为"序列"进行建模的想法。

关键突破：**从"对单个 item 打分"转变为"对 item 列表整体建模"**。

---

### 3.1 DLCM（Deep Listwise Context Model，2018）

#### 核心思想
DLCM 使用 RNN 对精排结果列表进行顺序编码，捕捉列表内 item 之间的上下文关系，然后重新打分。

#### 模型架构

```
精排列表 [x₁, x₂, ..., xₙ]（按精排分排序）
         ↓
    GRU/LSTM 顺序编码
         ↓
  h₁, h₂, ..., hₙ（上下文感知表示）
         ↓
    Softmax 打分层
         ↓
  重排得分 s₁', s₂', ..., sₙ'
```

#### 损失函数

使用 Attention-based Listwise Loss（近似 NDCG 优化）：

```
L = -∑ᵢ wᵢ · log P(σ|s')
```

其中 `σ` 为目标排列（按相关性降序），`P(σ|s')` 为给定打分的列表似然。

#### 局限
- GRU 编码顺序敏感，输入顺序不同结果不同，不够稳定。
- 序列长度受限，难以建模长列表。

---

### 3.2 PRM（Personalized Re-ranking Model，2019，阿里）

#### 核心思想
PRM 是工业界影响力最大的深度重排模型之一。将 Transformer 引入重排，**以用户历史行为序列 + 候选 item 序列为输入，通过 Self-Attention 建模 item 间的上下文交互**。

#### 输入设计

```
PRM 输入 = {
    PV: 个性化向量（用户 embedding，从精排模型提取）
    IP: item 特征向量（id, 类目, 价格, 精排分...）
}

输入序列：[PV; IP₁], [PV; IP₂], ..., [PV; IPₙ]
```

#### 模型架构

```
Input Layer: [PV||IPᵢ] for each item i
      ↓
Transformer Encoder（多头自注意力）
      ↓
每个 position 的上下文感知表示 hᵢ
      ↓
MLP 打分 → 重排得分 sᵢ'
      ↓
列表级损失（Listwise Loss）优化
```

#### Self-Attention 的意义

```
Attention(Q, K, V) = softmax(QKᵀ/√d) · V
```

item_i 的注意力权重 = 它与列表中其他 item 的相关程度 → 捕捉"这个商品在这个列表中的价值"。

#### 训练目标

优化列表概率：

```
L = -∑_{(u,L,y)} ∑ᵢ yᵢ · log P(sᵢ'|L, u)
```

其中 `L` 为候选列表，`yᵢ` 为 item_i 的点击标签。

#### 工业效果（阿里 2019）
- **离线**：AUC +0.003，NDCG@5 +1.8%
- **线上 AB**：CTR +1.0%，RPM +1.5%
- 延迟：P99 < 20ms（GPU serving）

---

### 3.3 SetRank（2020）

#### 核心思想
SetRank 认为重排输出应该对**输入 item 的排列顺序不变（permutation invariant）**，改用 Deep Sets 架构替代 Transformer 的位置敏感性。

#### 关键公式

```
SetRank(S) = ρ(∑ᵢ φ(xᵢ))
```

- `φ`：item 级特征变换（MLP）
- `∑`：对集合做置换不变聚合
- `ρ`：集合级表示 → 打分（MLP）

#### 优点
- 真正的 Permutation Invariant，避免输入顺序的 spurious correlation。
- 集合内 item 数量可变，更灵活。

---

### 3.4 上下文感知重排（Context-aware Re-ranking）

#### MISL（Multi-Interest Self-supervised Learning，2022，快手）

```
核心：对用户多兴趣建模 + 候选列表的 Item-User-Item 三元交互
输入：用户多兴趣向量（K个） + 候选列表
Self-Attention 交叉编码 → 上下文打分
辅助任务：对比学习，拉近同兴趣 item，推远不同兴趣 item
```

#### PGRank（Policy Gradient Re-ranking，2021，美团）

```
将重排建模为 MDP：
State: 已选 item 集合 S_t + 用户状态
Action: 从候选集选下一个 item
Reward: 列表展示完毕后的总 CTR/GMV
Policy: Pointer Network（注意力机制选择动作）
优化：REINFORCE / Actor-Critic
```

#### 上下文感知的关键创新

| 方法 | 上下文建模方式 | 个性化 | 适用场景 |
|------|--------------|--------|----------|
| DLCM | RNN 顺序 | 弱 | 短列表 |
| PRM | Transformer Self-Attention | 强 | 工业主流 |
| SetRank | Deep Sets 置换不变 | 中 | 无序场景 |
| PGRank | RL + Pointer Network | 强 | 列表生成 |

### 局限性

| 问题 | 描述 |
|------|------|
| 训练-推断不一致 | 训练用精排列表，推断用实际候选，分布偏移 |
| 曝光偏差 | 用户只对已展示列表有反馈，未展示排列无标签 |
| 延迟瓶颈 | Transformer O(n²) 自注意力，n=200 时已显著 |
| 标注稀疏 | 列表级标签需要更多展示才能收集 |
| 长期价值忽视 | 训练目标以 CTR 为主，忽略用户长期满意度 |

### 代表工作

- **Ai et al. (2018) DLCM**：Deep Listwise Context Model，列表上下文建模开山之作
- **Pei et al. (2019) PRM**：阿里个性化重排，工业影响力最大
- **SetRank (2020)**：置换不变集合推断
- **PGRank (2021) 美团**：强化学习重排
- **MISL (2022) 快手**：多兴趣自监督对比学习

### 面试必考点

1. **Q：PRM 为什么要将用户向量 PV 和 item 向量 IP concat 后再送入 Transformer？**
   A：PRM 的核心思路是"个性化上下文建模"。若只用 item 特征，Self-Attention 只能捕捉 item 之间的通用关系，无法区分不同用户。将 PV 注入每个 position，使得 Attention 权重能够反映"对当前用户而言，item_i 和 item_j 的关联程度"，实现真正的个性化重排。

2. **Q：重排模型的训练样本如何构造？有什么难点？**
   A：难点在于**曝光偏差（Exposure Bias）**。模型训练时只能看到历史已展示的列表及其点击，但大量"如果展示这个排列，用户会怎么点击"的数据无法获取。缓解方案：① 用倒置倾向得分（IPS）加权纠偏；② 自然实验法（Position Randomization）；③ Off-policy RL 校正历史策略偏差。

3. **Q：SetRank 的"置换不变"有什么工程意义？**
   A：DLCM/PRM 对输入顺序敏感（同一候选集不同输入顺序 → 不同重排结果），导致工程实现中需要固定输入顺序（通常按精排分降序），引入了 spurious correlation（精排分靠前的 item 因位置获得额外加成）。SetRank 用 sum-aggregation 使结果完全不依赖输入顺序，实现更公平的打分。

4. **Q：PGRank 如何定义 MDP 的 Reward？**
   A：常见方案：① 稀疏 reward：用户完成全列表浏览后，将总 CTR/GMV 作为最终奖励；② 密集 reward：每个 position 展示后以该 item 是否被点击作为即时奖励（需折扣因子 γ 平衡长短期）；③ 复合 reward：CTR + 多样性惩罚项 + 用户停留时长。工业实践以密集 reward 为主，因为稀疏 reward 导致方差过大，训练不稳定。

---

## 阶段4：生成式重排（2023–今）

### 背景

2023年 ChatGPT 引爆 LLM 热潮后，推荐系统领域开始探索：**能否用大语言模型直接生成推荐列表排列，而非逐个打分？** 这一阶段的核心转变：

1. **从打分到生成**：不再逐 item 打分然后排序，而是直接生成有序列表（Listwise Generation）。
2. **世界知识注入**：LLM 携带的语义知识可以弥补 ID-based 系统的语义鸿沟。
3. **指令可控**：通过 Prompt 工程动态调整排序目标（多样性、新鲜度、商业化等）。

---

### 4.1 LLM 直接重排（Zero-shot / Few-shot）

#### 核心思路

将重排问题转化为 NLP 生成任务：

```
Prompt:
"Given user's click history: [item_A, item_B, item_C]
Candidate items: [item_1, item_2, ..., item_20]
Please rank these items for the user from most to least relevant.
Output format: [item_id_1, item_id_2, ..., item_id_20]"

LLM → 生成有序列表
```

#### 代表工作：Is ChatGPT a Good Recommender?（2023）

- 评估 ChatGPT 在直接推荐、排序、解释任务上的表现。
- 发现：零样本下 LLM 排序能力 < 专门精排模型，但在冷启动和长尾场景显著更优。
- 问题：① 幻觉（生成不存在的 item id）；② 位置偏差（倾向于输出靠近 prompt 末尾的 item）；③ 无法利用协同过滤信号。

#### 工程挑战

| 挑战 | 描述 | 缓解方案 |
|------|------|----------|
| 延迟 | LLM 推断 >100ms，难以满足在线要求 | 离线预排/异步重排 |
| 候选太多 | LLM 上下文窗口限制，无法处理 200+ 候选 | 精排截断至 Top-20 |
| 幻觉 | 输出不合法 item_id | 输出后处理 + 约束解码 |
| 成本 | Token 计费，大规模推断成本高 | 蒸馏小模型 |

---

### 4.2 Listwise 生成重排

#### 核心思路：将重排建模为 Seq2Seq 任务

```
输入序列：[user_context; item₁; item₂; ...; itemₙ]
输出序列：[item_σ(1); item_σ(2); ...; item_σ(n)]  （重排后的排列 σ）
```

使用 Encoder-Decoder 或 Decoder-only LLM 直接生成排列。

#### 代表工作：LLMRank（2023，复旦 + 阿里）

```
关键设计：
1. Recency-focused prompting：用户最近交互的 item 放在 prompt 末尾（LLM 更关注末尾）
2. In-context learning：提供 3-5 个高质量排列示例
3. Permutation bootstrapping：多次推断取集成，降低输出随机性

实验结果（MovieLens / Amazon Reviews）：
- 零样本 LLM 重排 vs BM25：NDCG@10 +15%
- 零样本 LLM 重排 vs SASRec：部分数据集仍有差距，冷启动场景领先
```

#### 代表工作：RankGPT（2023）

```
Sliding Window Strategy：
由于 LLM 输入窗口限制，将候选列表分窗口滑动排序：
Window 1: [item₁₋₂₀] → 局部排序
Window 2: [item₁₁₋₃₀] → 局部排序（与前窗口重叠）
...
最终：bubble-sort 式全局整合
```

---

### 4.3 强化学习优化排列（RLHF for Re-ranking）

#### 核心思路

将列表生成视为 RL 问题，用人类反馈或在线反馈优化：

```
框架：RLHF-Reranking
1. SFT Phase: 用历史点击数据对 LLM 做监督微调，学习基础重排能力
2. Reward Model: 用用户行为（点击、购买、收藏）训练奖励模型
3. PPO Phase: 用 PPO 算法，以 Reward Model 信号优化排列策略
```

#### 奖励函数设计

```
R(π) = α · CTR_reward(π)
      + β · Diversity_reward(π)
      + γ · Freshness_reward(π)
      - δ · KL(π || π_ref)   // 防止与参考策略偏离过远（PPO clip）
```

#### 代表工作：InstructRec（2023）

```
将推荐指令对齐到 LLM：
Input: "Find me diverse tech news articles from the past week, 
        avoiding content from sources I've seen recently."
LLM → 解析指令 → 生成满足约束的重排列表

优势：Zero-shot 处理复杂业务约束（之前需要手写规则）
```

#### 代表工作：P5（2022）/ M6-Rec（2022）

```
将推荐任务统一为文本生成：
Input Template: "Given user history: {history}, 
                 Rate item {item}: "
Output: "Highly Relevant / Somewhat Relevant / Not Relevant"

→ 可以 Zero-shot 泛化到新领域
```

---

### 4.4 小模型蒸馏 + 端侧重排

#### 背景
LLM 重排延迟高、成本大，工业界转向"用 LLM 生成训练数据，蒸馏到小模型"路线：

```
Pipeline：
1. LLM 离线生成高质量排列标注（Listwise Labels）
2. 用标注数据训练轻量级重排模型（BERT-base 或更小）
3. 小模型部署到线上，<10ms 延迟
```

#### 代表工作：TALLRec（2023）

```
Two-stage Adaptation:
Stage1: 推荐任务的 instruction tuning（1k-10k 样本）
Stage2: 蒸馏到 LLaMA-7B → 进一步蒸馏到 BERT-level 模型

效果：在少样本场景（<100 items/user history）显著优于传统方法
```

---

### 局限性

| 问题 | 描述 |
|------|------|
| 延迟 | LLM 推断 100-500ms，远超精排<10ms 要求 |
| 可解释性 | LLM 决策黑盒，业务监控困难 |
| 幻觉与不一致 | 同一输入不同推断结果可能不同 |
| ID信号缺失 | LLM 不擅长利用 item_id 的协同过滤信号 |
| 训练数据 | Listwise 标注成本高，需要大量用户行为日志 |
| 冷启动LLM本身 | LLM fine-tune 需要大量推荐领域数据 |

### 代表工作汇总

| 工作 | 机构 | 年份 | 核心贡献 |
|------|------|------|----------|
| LLMRank | 复旦+阿里 | 2023 | 首批系统评估 LLM 重排 |
| RankGPT | 微软 | 2023 | Sliding Window 排序策略 |
| InstructRec | 阿里 | 2023 | 指令可控推荐重排 |
| TALLRec | 华中科大 | 2023 | 轻量 LLM 推荐适配 |
| BIGRec | 清华 | 2023 | 基于 ID 的 Grounding 对齐 |
| PALR | 华为 | 2023 | 个性化对齐 LLM 重排 |

### 面试必考点

1. **Q：LLM 用于重排的核心挑战是什么，工业界如何解决？**
   A：三大挑战：① **延迟**（缓解：离线预计算 + 异步更新，或蒸馏到小模型）；② **幻觉**（缓解：约束解码，只允许输出候选集内的 item_id，后处理校验）；③ **成本**（缓解：只对高价值流量用 LLM 重排，低价值流量仍用传统方法）。

2. **Q：Listwise 生成和 Pointwise 打分的根本区别？**
   A：Pointwise 独立对每个 item 打分，忽略 item 间的相互影响；Listwise 将整个列表作为输出对象，天然建模了"这个 item 在这个列表位置的价值"，即上下文感知。但 Listwise 的训练需要列表级标注（即对整个排列的评价），数据获取成本更高。

---

## 技术演进总结对比表

| 维度 | 规则时代 | 统计多样性 | 深度学习重排 | 生成式重排 |
|------|---------|-----------|------------|----------|
| 代表方法 | 打散/去重 | MMR/DPP | PRM/SetRank | LLMRank/RLHF |
| 个性化 | ❌ | 弱 | ✅ | ✅ |
| 理论基础 | 无 | 概率论 | 深度学习 | 生成模型 |
| 计算复杂度 | O(n) | O(kn)~O(k²n) | O(n²) Attention | O(n·token) |
| 延迟 | <1ms | 1-5ms | 10-50ms | 100-500ms |
| 工程难度 | 低 | 中 | 高 | 极高 |
| 多样性建模 | 显式规则 | 数学公式 | 隐式学习 | 指令控制 |
| 上下文感知 | ❌ | 弱 | ✅ | ✅ |
| 可解释性 | 高 | 中 | 低 | 极低 |

---

## 面试高频问题

> 以下 8 道题覆盖重排方向面试高频考点，建议每题都能流利回答。

### Q1：重排模块在整个推荐系统中的定位是什么，为什么不直接把重排能力放到精排里？

**参考答案框架：**

重排是精排之后的最后一道处理，主要职责：
1. **集合优化（Set Optimization）**：精排是 pointwise/pairwise，优化单个 item 的相关性；重排优化整个展示列表的整体效用（多样性 + 相关性 + 商业目标的综合最优）。
2. **快速响应业务需求**：运营打散、类目约束、新品扶持等规则，放在精排层会影响全局模型，放在重排层可以快速迭代。
3. **延迟预算分配**：精排处理 500-2000 个候选，重排只处理 20-200 个，可以使用更复杂的模型（Transformer等）而不影响整体 RT。

不能合并的原因：精排需要快速处理大量候选，无法对全量候选运行 O(n²) 的注意力计算；重排的 listwise 目标函数与精排的 pointwise 损失难以统一优化。

---

### Q2：MMR 和 DPP 都能做多样性，实际工程中怎么选？

**参考答案框架：**

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 快速上线，资源有限 | MMR | 实现简单，λ 单参数调节，无需额外 embedding |
| 追求理论最优，有 embedding 资源 | DPP | 全局多样性度量，效果上限更高 |
| 类目/标签多样性 | MMR | 离散特征的 Jaccard 相似度易于计算 |
| 语义多样性（内容推荐）| DPP | 连续 embedding 空间中行列式更有意义 |
| 实时计算 | MMR | DPP 的 Cholesky 分解难以全实时 |

实践中两者可以结合：先用 DPP 做内容多样性，再用 MMR 规则做类目约束。

---

### Q3：PRM 的训练数据如何构造？如何处理曝光偏差？

**参考答案框架：**

**训练数据构造：**
- 正样本：历史曝光列表中，有点击的 item 对应 position 标为 1。
- 负样本：同一列表内，曝光未点击的 item 标为 0。
- 列表构造：保持历史精排顺序（或加入轻微随机扰动避免顺序 spurious correlation）。

**曝光偏差处理：**
1. **IPS 加权**：根据 item 被展示的倾向得分（Propensity Score）对训练样本加权，降低高频位置样本的权重。
2. **Position Debiasing**：在模型中加入位置特征，并在推断时将位置特征 mask 掉（设为平均值），使模型学习的是内容偏好而非位置偏好。
3. **随机曝光实验**：定期用 ε-greedy 策略随机展示，获取无偏样本作为训练集的补充。

---

### Q4：如何评估重排模块的效果？离线指标和线上指标有哪些？

**参考答案框架：**

**离线指标：**
- **NDCG@K（Normalized Discounted Cumulative Gain）**：衡量排序质量，考虑位置折扣。
- **ILD（Intra-List Diversity）**：列表内 item embedding 的平均余弦距离，衡量多样性。
- **MRR（Mean Reciprocal Rank）**：第一个相关 item 的平均倒数排名。
- **Coverage**：推荐列表覆盖的长尾 item 比例。

**线上 AB 指标（短期）：**
- CTR、CVR、GMV（主要业务指标）
- 浏览深度（用户滑动了几屏）、停留时长

**线上 AB 指标（长期）**（重排尤其需要关注）：
- 7日/30日留存率（多样性优化长期效果）
- 复访率、用户满意度调查
- 内容生态健康度（长尾 item 曝光比例）

**注意**：重排的多样性优化可能在短期降低 CTR（用户更多看到不熟悉的内容），但长期提升留存，需要设置足够长的实验窗口（≥2周）。

---

### Q5：深度学习重排的训练-推断不一致问题（Exposure Bias）如何解决？

**参考答案框架：**

训练时，模型看到的是历史已部署策略生成的列表；推断时，模型要对新策略生成的列表重排 → 训练-推断数据分布不同。

解决方案：
1. **Scheduled Sampling**：训练时以一定概率用模型自己的输出替代真实历史列表，逐步缩小分布差异。
2. **Online Distillation**：将线上新策略生成的候选列表实时采样，持续更新训练集。
3. **Off-policy Learning with Importance Sampling**：用重要性采样权重修正历史策略和当前策略之间的分布差异。
4. **Augmented Negative Sampling**：除了曝光负样本，额外采样"模型预测高分但未曝光"的 hard negative，提升模型鲁棒性。

---

### Q6：重排中的强化学习（如 PGRank）相比监督学习有什么优势和挑战？

**参考答案框架：**

**优势：**
1. 可以直接优化非可微目标（如 NDCG、用户留存率）而非 CTR 的代理指标。
2. 自然建模序列决策：每选一个 item 都考虑之前已选的 context。
3. 可以探索历史数据中未出现的新排列组合（exploration）。

**挑战：**
1. **奖励稀疏**：用户行为反馈延迟（用户需要完整浏览列表后才能获得信号），导致方差大。
2. **Off-policy 问题**：历史数据由不同策略生成，直接用于 on-policy RL 训练存在分布偏移。
3. **训练不稳定**：PPO/A3C 等算法对超参敏感，工业环境噪声大。
4. **探索成本**：线上探索有 CTR 损失风险，需要保守的 exploration 策略（如 ε 很小）。

**工业实践**：通常用离线 RL（Batch RL / CQL）+ 少量线上探索，而非纯在线 RL。

---

### Q7：LLM 用于推荐重排的工业化路径是什么？

**参考答案框架：**

当前主流工业化路径（2024 主流）：

```
阶段1（离线增强）：
  LLM 作为"数据飞轮"
  → LLM 离线生成高质量排列标注
  → 用标注训练/增强传统重排模型
  → 延迟不变，质量提升

阶段2（轻量部署）：
  LLM 蒸馏 → BERT-base 级别重排模型
  → 保留 LLM 的语义理解能力
  → 延迟降至 <30ms
  
阶段3（异步 LLM）：
  高价值用户/场景：异步调用 LLM 重排
  → 结果缓存（TTL=5min）
  → 实时 serving 走缓存，异步更新
  
阶段4（端侧小模型）：
  端侧轻量 LLM（1B-3B）+ 本地重排
  → 隐私保护 + 超低延迟
```

---

### Q8：如何设计一个完整的重排 A/B 实验方案？

**参考答案框架：**

**实验设计要点：**

1. **流量分割**：按 user_id hash 而非 request_id，保证同一用户始终在同一桶，避免同用户 cross-contamination。

2. **实验组设置**：
   - Control：当前线上策略（规则打散 or 现有重排模型）
   - Treatment：新重排方案
   - 建议同时跑 AA 实验验证流量一致性

3. **观测指标优先级**：
   - 核心：CTR、CVR、GMV（主要业务指标，不能显著下降）
   - 重排专项：浏览深度、多样性指标（ILD）、长尾曝光比
   - 长期：7日留存（设置 holdout 组持续观测）

4. **实验时长**：≥2周（覆盖完整用户行为周期，避免新奇效应）

5. **分析方法**：
   - 主指标：双侧 t-test，显著性 p<0.05
   - 多重检验校正：Bonferroni 修正或 FDR 控制
   - 异质性分析：分用户活跃度、类目偏好分析实验效果

6. **上线策略**：
   - 灰度：先 1% → 10% → 50% → 100%
   - 回滚条件：CTR 下降超过 0.5% 或 P99 延迟超标立即回滚

---

*文档完成时间：2026-03-16 | 作者：MelonEggLearn*
*参考：PRM论文(2019)、DPP in ML(2012)、LLMRank(2023)、RankGPT(2023)、阿里SIGIR/RecSys工作*
