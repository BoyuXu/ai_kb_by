# 搜广推算法工程师 · 项目经历（面试版）

> 整理者：MelonEggLearn | 更新时间：2026-03-16
> 适用方向：推荐系统 / 搜索 / 广告算法工程师
> 使用说明：根据目标公司调整数据规模和技术栈细节，量化数字基于合理范围，面试时用自己语言描述

---

## 项目1：电商推荐召回系统重构（双塔+HNSW向量检索）

### 背景

- **公司/团队背景**：某头部电商平台推荐算法团队，负责首页"猜你喜欢"场景，日均 PV 约 8 亿，在线服务用户 2 亿+。
- **问题痛点**：
  - 原有召回以协同过滤（UserCF/ItemCF）为主，依赖共现矩阵，**长尾商品召回率低**（尾部 60% 商品几乎不被召回）。
  - 新用户冷启动问题严重，CF 无法处理历史行为 < 5 次的用户。
  - ItemCF 索引全量更新周期 6 小时，**实时性差**，新品上架后 6 小时内基本无曝光。
  - 多路召回结果简单截断，**没有统一的向量相似度检索层**，召回质量参差不齐。

### 技术方案

#### 核心算法：双塔模型（Two-Tower / DSSM）

```
用户塔                          商品塔
-----------                   -----------
User ID embedding              Item ID embedding
行为序列（Attention池化）       类目/品牌 embedding
人口属性特征                    价格分桶 embedding
实时行为特征（过去1h）           图文内容特征（ResNet/CLIP）
       ↓                              ↓
    MLP + BN                      MLP + BN
       ↓                              ↓
  用户向量 u ∈ ℝ¹²⁸           商品向量 v ∈ ℝ¹²⁸
       ↓____________________________↓
              cosine similarity
              训练目标：In-batch Negative Sampling
```

**训练目标（Sampled Softmax）**：

```
L = -log( exp(u·v⁺/τ) / (exp(u·v⁺/τ) + Σⱼ exp(u·vⱼ⁻/τ)) )
```

- τ=0.07（温度系数），batch_size=4096，In-batch 负样本 + 随机负样本
- 负样本采样：混合 batch 内负样本（70%）+ 全局随机负样本（20%）+ Hard Negative（10%，精排分 top20% 但未点击）

#### 向量检索：HNSW（Hierarchical Navigable Small World）

```
数据规模：商品向量库 3000 万条，128 维
HNSW 参数：M=16, efConstruction=400, ef=200
QPS：单机 2 万 QPS，P99 延迟 < 5ms
索引大小：约 18GB，部署在内存（Redis 集群或 Milvus）
```

增量更新策略：新品上架后 5 分钟内完成向量计算并写入索引（Kafka + Flink 实时 pipeline）。

#### 关键技术细节

- **序列建模**：用户行为序列（点击/购买/收藏）用 Target Attention（与目标商品做 attention pooling），序列长度 50。
- **多场景统一塔**：引入 Scene Embedding，单模型覆盖首页/搜索/活动页三个场景的召回。
- **特征工程**：实时特征（过去 1h 点击序列）通过 Feature Server（Redis）实时拉取，延迟 < 2ms。

### 实施过程

#### 数据准备

- 正样本：过去 30 天用户点击/购买日志，去除 bot 流量后约 50 亿条。
- 负样本：In-batch + 全局随机（按商品曝光量的 0.75 次方采样，降低热门商品过采样偏差）。
- 特征：离线特征 Hive 生成（T+1），实时特征 Flink 聚合写入 Feature Store（Redis Cluster）。

#### 模型训练/迭代

| 迭代 | 主要变更 | 离线 AUC |
|------|---------|---------|
| v1.0 | 基础双塔，ID特征 | 0.712 |
| v1.1 | 加入行为序列（Mean Pooling） | 0.731 |
| v1.2 | Target Attention 序列建模 | 0.748 |
| v1.3 | Hard Negative Mining | 0.756 |
| v1.4 | 实时特征接入 | 0.762 |

#### AB 实验设计

- **基线**：ItemCF + UserCF 双路召回
- **实验组**：双塔召回替换 ItemCF，与 UserCF 融合
- **流量**：10% 用户，持续 2 周
- **主要观测**：召回率（@50/100/200）、精排层 CTR、覆盖率（长尾商品占比）

### 量化结果（STAR格式）

- **Situation**：首页召回以 CF 为主，长尾商品召回率低，新品冷启动差。
- **Task**：重构召回链路，引入双塔+向量检索，提升召回质量和实时性。
- **Action**：设计并上线双塔模型，构建 HNSW 向量索引，打通实时特征 pipeline。
- **Result**：
  - 召回层 Recall@200 提升 **+23.4%**（尤其长尾商品，长尾召回率 +41%）
  - 精排层 CTR **+1.8%**（召回质量提升带动精排效果）
  - 新品平均首次曝光时间从 6 小时缩短到 **12 分钟**
  - 冷启动用户（< 5 次行为）的点击率提升 **+12%**
  - 线上 P99 召回延迟 **< 8ms**（含向量检索）

### 面试亮点

**可深挖的技术点（3条）**：

1. **In-batch Negative Sampling 的频率偏差问题**：热门商品在 batch 中出现频率高，作为负样本的概率也高，导致模型对热门商品的 embedding 过度压制。解决：引入按曝光频次的降采样修正（`logQ correction`），在 loss 中减去 `log(q(v))`（`q(v)` 为商品被采样为负样本的概率）。

2. **Hard Negative Mining 的度量标准**：Hard Negative 不能太难（精排 top1，可能是真正相关但未点击）也不能太简单（随机商品）。实践中选取精排分 rank 在 10%-30% 区间的未点击商品，或用余弦相似度 > 0.6 但未点击的商品。

3. **向量索引增量更新策略**：全量重建 HNSW 代价高（3000w商品 ~2小时），实践采用`增量追加 + 定期全量重建`：新商品实时追加到索引（支持动态插入），每天凌晨重建全量索引（覆盖过期商品的删除）。

**可能被追问的问题+参考答案框架**：

- **Q：双塔模型怎么处理用户-商品交叉特征？**
  A：标准双塔不建模交叉特征（用户塔和商品塔独立）这是其可大规模部署的代价。解决思路：① 将交叉特征提取到用户塔（如用户对某类目的偏好分作为用户特征）；② 在精排阶段用 DIN 等模型处理细粒度交叉；③ 探索轻量交互层（如 MIND、ComiRec）在召回阶段做有限交叉。

- **Q：HNSW 和 Faiss IVF 怎么选？**
  A：HNSW：图结构，查询精度高（Recall@10 > 95%），支持动态插入，内存占用大（约 60 bytes/vector）。IVF：倒排量化，内存占用小，需要离线重建（不支持动态增量），适合超大规模（>1亿）静态库。电商新品更新频繁，选 HNSW；搜索静态文档库用 IVF+PQ。

---

## 项目2：CTR精排模型迭代（DIN → DIEN升级）

### 背景

- **公司/团队背景**：某头部短视频平台推荐团队，负责信息流精排，日均排序请求约 30 亿次，精排候选约 200 个/次。
- **问题痛点**：
  - 精排模型以 DeepFM 为基础，**用户历史行为仅做简单 Pooling**，无法捕捉兴趣演化。
  - 用户历史行为最长 100 条，但均值池化导致早期兴趣和近期兴趣权重相同，**时序信息丢失**。
  - 发现用户对不同类型视频有明显交替兴趣（如刷完搞笑视频后想刷知识视频），**兴趣漂移建模缺失**。

### 技术方案

#### DIN（Deep Interest Network）

```
核心：对历史行为序列做 Target-Attention，计算与目标 item 的相关性权重

Attention score: e_i = f(h_i, e_target) = sigmoid(W · [h_i; e_target; h_i-e_target; h_i*e_target])
用户兴趣向量: u = Σᵢ aᵢ · hᵢ  （aᵢ = softmax(eᵢ)）

h_i: 第i条历史行为embedding
e_target: 目标item embedding
```

#### DIEN（Deep Interest Evolution Network）

在 DIN 基础上增加 GRU 序列建模，捕捉兴趣演化：

```
第1层 GRU（兴趣提取层）：
h'_t = GRU(h'_{t-1}, e_t)   // 顺序建模历史行为

第2层 AUGRU（带注意力的 GRU，兴趣进化层）：
u'_t = α_t * u_t             // 注意力门控，α_t = attention(h'_t, e_target)
h_t = (1 - u'_t) ⊙ h_{t-1} + u'_t ⊙ h̃_t
最终兴趣表示: h_T（最后时刻隐状态）
```

关键创新：AUGRU 的 Update Gate 由目标 item 的注意力分数控制，使兴趣进化方向与目标相关。

#### 辅助损失（Auxiliary Loss）

```
在每个 GRU 时间步引入辅助 loss：
L_aux = -1/T Σ_{t=1}^{T} [y_t log σ(h'_t · e_{t+1}) + (1-y_t) log(1-σ(h'_t · e_{t+1}))]

e_{t+1}: t+1时刻实际交互item的embedding（正样本）
负样本: 随机采样的未交互item
```

辅助 loss 督促每个时间步的隐状态能预测下一步行为，避免梯度消失。

#### 关键技术细节

- 行为序列长度：50（点击）+ 20（点赞）+ 10（评论/分享），分字段建模。
- 训练数据：7 天点击日志，约 50 亿样本，每天增量训练。
- 特征：用户 ID、商品 ID、行为序列、上下文（时间、位置、设备）、视频时长分桶。
- 批量大小：4096，Adam 优化器，学习率 1e-4（warmup 1000步）。

### 实施过程

#### 数据准备

- 正样本：点击（主）、点赞/评论（加权）、完播（视频完播率 > 80% 作为强正样本）。
- 负样本：曝光未点击（去除误点，停留时长 < 0.5s 的点击视为误点，降级为负样本）。
- 特征预处理：高频 ID 截断（用户行为 ID 保留频次 > 50），低频 ID 归入 OOV bucket。

#### 模型训练/迭代

| 版本 | 核心变更 | 线上CTR | AUC |
|------|---------|---------|-----|
| DeepFM | 基础模型，行为Mean Pooling | baseline | 0.734 |
| +DIN | Target Attention替换MeanPooling | +0.8% | 0.741 |
| +DIEN | GRU序列 + AUGRU进化层 | +1.5% | 0.748 |
| +辅助Loss | DIEN + Auxiliary Loss | +1.8% | 0.751 |
| +实时行为 | 加入过去30min实时行为序列 | +2.3% | 0.755 |

#### AB 实验设计

- 对照组：DeepFM + DIN（线上已有）
- 实验组：DIEN + Auxiliary Loss
- 流量：20%，实验周期 14 天
- 核心指标：CTR、完播率、用户次日留存

### 量化结果（STAR格式）

- **Situation**：精排模型对用户兴趣演化建模不足，行为 Pooling 丢失时序信息。
- **Task**：将精排模型从 DIN 升级为 DIEN，引入序列兴趣演化建模。
- **Action**：设计 DIEN 架构，引入 AUGRU + 辅助损失，处理行为序列长度不一致问题（Masking）。
- **Result**：
  - 线上 CTR 提升 **+1.8%**（相对提升）
  - 完播率提升 **+2.1%**
  - AUC 从 0.734 提升到 **0.751**（+0.017）
  - 用户次日留存提升 **+0.4%**
  - 模型训练时间：T+2 → T+1（训练加速，从全量重训改为增量训练）

### 面试亮点

**可深挖的技术点（3条）**：

1. **AUGRU 的注意力门控 vs 标准 GRU Update Gate**：标准 GRU 的更新门由当前输入和前一隐状态决定，与目标无关。AUGRU 将目标 item 的注意力分数乘到更新门上，当历史行为与目标高度相关时，更新门激活更大，该时刻的兴趣演化对最终表示的贡献更大。这使得 DIEN 对不同 query（目标 item）返回不同的兴趣演化路径。

2. **辅助损失解决长序列梯度消失**：DIEN 的 GRU 序列可能很长（50步+），梯度需要 BPTT 反传，容易消失。辅助损失在每一步都提供监督信号（预测下一步行为），相当于在每个时间步注入梯度，避免长序列梯度消失问题。

3. **行为序列的 Padding 和 Masking 处理**：不同用户行为序列长度不同，需要 Padding 到固定长度。Attention 计算时需要对 Padding 位置 Mask（设为 -inf 再 softmax 归零），否则 Padding 的 embedding 会干扰注意力权重。GRU 则用 `sequence_length` 参数实现有效步的掩码计算。

**可能被追问的问题+参考答案框架**：

- **Q：DIEN 和 SIM（Search-based Interest Model）的区别？**
  A：DIEN 对全部历史行为做 GRU 序列建模，计算复杂度 O(T)。SIM 面向超长行为序列（1000+），分两步：① 用 GSU（General Search Unit）基于目标 item 从超长序列中检索 Top-K 相关行为（O(T) 检索）；② 对 Top-K 行为做精细的注意力建模（O(K)，K<<T）。SIM 解决了 DIEN 在超长序列下的计算瓶颈。

- **Q：如何处理用户的多兴趣问题？**
  A：DIEN 产出单一兴趣向量，难以表达用户的多种兴趣（如同时喜欢科技和美食）。解决：① MIND（Multi-Interest Network with Dynamic Routing）：用胶囊网络产出 K 个兴趣向量；② ComiRec：Attention-based 多兴趣聚类；③ 对不同行为类型（点击/购买/收藏）分别建立兴趣塔，Concat 后输入后续网络。

---

## 项目3：多任务学习优化（ESMM+MMoE联合训练）

### 背景

- **公司/团队背景**：某头部电商平台广告与推荐融合团队，负责搜索结果页的商品排序，需要同时优化点击率（CTR）和购买转化率（CVR）。
- **问题痛点**：
  - CTR 模型和 CVR 模型**独立训练**，CVR 模型训练样本仅限"点击后是否购买"，**样本选择偏差（SSB）**严重（只有被点击的商品才有 CVR 标签）。
  - CVR 训练样本量仅为 CTR 的 1/20，**数据稀疏性严重**，长尾商品 CVR 预估不准。
  - 线上 ranking 分 = CTR * CVR * Price，两个模型独立产出导致**校准（Calibration）不一致**，乘积后的量纲混乱。
  - 后续新增 wishlist（收藏率）、cart（加购率）目标，多目标扩展困难。

### 技术方案

#### ESMM（Entire Space Multi-task Model，阿里 2018）

核心思想：在全空间（所有曝光样本）建模 CTCVR = CTR × CVR，解决 SSB 和数据稀疏问题：

```
架构：
                  共享 Embedding 层
                /                  \
         CTR Tower              CVR Tower
        (MLP + Sigmoid)        (MLP + Sigmoid)
              |                      |
           p(CTR)                 p(CVR|click)
              └─────────────────────┘
                        ×
                    p(CTCVR) = p(CTR) × p(CVR|click)

训练 Loss：
L = L_CTR + L_CTCVR  （不单独训练 CVR，用 CTCVR 联合监督）
L_CTR = BCE(y_ctr, p_ctr)
L_CTCVR = BCE(y_ctcvr, p_ctcvr)

其中 y_ctcvr = y_ctr AND y_cvr（曝光且点击且购买=1，否则=0）
→ 训练样本从"点击样本"扩展为"全曝光样本"，SSB 消除
```

#### MMoE（Multi-gate Mixture of Experts，Google 2018）

解决多任务之间的目标冲突问题（如 CTR 和 CVR 任务的梯度方向可能冲突）：

```
架构：
Input Features → K 个 Expert（各自独立 MLP）
                    ↓
     Task-k 专属 Gate（Softmax over K experts）
                    ↓
     Task-k 的加权 Expert 输出
                    ↓
     Task-k Tower（MLP → 预测头）

数学：
f_k(x) = Σᵢ g_k(x)ᵢ · eᵢ(x)
g_k(x) = softmax(W_{gk} · x)  // 第k个任务的gate
eᵢ(x) = MLP_i(x)              // 第i个expert
```

#### 联合方案：ESMM + MMoE

```
数据规模：
- 训练样本：30天全曝光日志，约 300 亿条
- 特征：用户ID、商品ID、搜索词、上下文（时间/位置/设备）
- Expert 数量 K=8，Tower 数=4（CTR/CVR/Wishlist/Cart）

任务权重：
L_total = w₁·L_CTR + w₂·L_CTCVR + w₃·L_Wishlist + w₄·L_Cart
权重：w₁=1.0, w₂=2.0, w₃=0.5, w₄=0.5（按任务重要性和难度调整）
```

### 实施过程

#### 数据准备

- 曝光样本：搜索结果页所有曝光记录（包含未点击商品）。
- 标签定义：
  - y_ctr: 是否点击
  - y_cvr: 点击后 24h 内是否购买
  - y_ctcvr: y_ctr AND y_cvr（全曝光空间的购买标签）
- 负采样：全量曝光 300 亿无需负采样，但增加 Hard Negative（曝光但从未点击的优质商品）提高区分度。

#### 模型训练/迭代

- 共享 Embedding：用户 ID Embedding（512维）、商品 ID Embedding（256维）所有任务共享。
- Expert：8 个 256 维 MLP，每个 Expert 独立参数。
- 增量训练：每日增量 fine-tune（新数据 1 轮），每周全量重训（7天数据）。

#### AB 实验设计

- **对照**：独立训练的 CTR 模型 + CVR 模型，线上 rank score = pCTR × pCVR × price
- **实验**：ESMM+MMoE 联合模型，rank score = pCTCVR × price（直接用全空间 CVR）
- 实验周期：21 天（包含大促前中后完整周期）

### 量化结果（STAR格式）

- **Situation**：CVR 模型样本偏差严重，多任务扩展困难，各模型校准不一致。
- **Task**：引入 ESMM 解决 SSB，引入 MMoE 支持多任务，统一全空间建模框架。
- **Action**：设计 ESMM+MMoE 联合架构，处理 300 亿全曝光样本的分布式训练，接入 4 个业务目标。
- **Result**：
  - CTR AUC: 0.741 → **0.748**（+0.007）
  - CVR AUC: 0.679 → **0.694**（+0.015，显著改善，SSB 消除效果明显）
  - 线上 GMV 提升 **+2.4%**（核心业务指标）
  - 新增 Wishlist/Cart 任务无需重新训练独立模型，**迭代效率提升 60%**
  - 大促期间（流量 3x）模型稳定性显著优于独立模型（AUC 方差降低 40%）

### 面试亮点

**可深挖的技术点（3条）**：

1. **ESMM 为什么能解决 SSB（Sample Selection Bias）？**：传统 CVR 模型只在"点击样本"上训练，但线上推断在"全曝光"样本上进行，训练与推断空间不匹配。ESMM 将 CVR 的监督信号转化为 CTCVR（全曝光空间），CVR Tower 的梯度来自 CTCVR 的 loss（通过链式法则：∂L/∂CVR = ∂L/∂CTCVR · CTR），使 CVR 塔在全曝光空间得到间接训练。

2. **MMoE 的 Expert Utilization 监控**：实践中需要监控各 Expert 的激活情况（每个 task 的 Gate 的 Softmax 权重分布），避免出现"Expert Collapse"（所有 task 的 gate 都主要激活少数几个 Expert，其他 Expert 几乎不被使用）。解决：引入 Expert Diversity Loss 或辅助的 Load Balancing 机制（类似 MoE 语言模型）。

3. **多任务权重的动态调整**：静态权重需要手动调，常见优化：① GradNorm：根据各任务梯度范数自动调整权重；② Uncertainty Weighting（Kendall 2018）：将任务权重建模为可学习的不确定性参数；③ 实践中按线上指标贡献每月重新标定一次。

**可能被追问的问题+参考答案框架**：

- **Q：PLE（Progressive Layered Extraction）相比 MMoE 有什么改进？**
  A：MMoE 的所有 Expert 对所有 Task 共享，当任务相关性低时，Expert 很难学到任务特有的模式（被拉扯）。PLE 引入任务专属 Expert（只服务特定任务）和共享 Expert（跨任务共享），分层提取：低层更多共享知识，高层更多任务专有知识，避免 seesaw problem（一个任务涨另一个任务必跌）。

- **Q：CTCVR 联合训练时，如果 CTR 和 CVR 的梯度方向冲突怎么办？**
  A：这正是 MMoE 要解决的问题。另外可以：① 梯度手术（Gradient Surgery）：投影掉互相冲突的梯度分量；② 分阶段训练：先预训练 CTR Tower，再 fine-tune CVR Tower；③ 降低冲突任务的 loss 权重，牺牲次要任务保证主要任务。

---

## 项目4：Auto Bidding 智能出价系统（强化学习出价）

### 背景

- **公司/团队背景**：某头部广告平台投放优化团队，为中小广告主提供自动化出价服务，管理广告主日均消耗约 5 亿元。
- **问题痛点**：
  - 广告主手动出价，需要不断根据竞争态势调价，**运营成本高**，中小广告主缺乏专业运营能力。
  - 规则出价（固定 CPC/eCPM）无法应对**流量竞争的动态变化**（不同时段、不同地域竞争强度差异大）。
  - 简单比例出价策略（target CPA × pCVR）忽略了**跨时段的预算分配**问题（早上用完预算，下午黄金时段无法曝光）。
  - 广告主有 ROI 约束（ROAS ≥ 3.0），预算约束（日预算 X 元），两个约束同时满足困难。

### 技术方案

#### 问题建模：约束马尔可夫决策过程（CMDP）

```
State S_t = {
    t: 当前时刻（0~24h离散化为96个时间片）
    B_t: 剩余预算
    spend_rate_t: 过去1h消耗速率
    competitor_intensity_t: 竞争强度（对手出价分布的统计量）
    cvr_forecast_t: 当前时段预估 CVR
    historical_roi_t: 过去1h实际 ROI
}

Action a_t = bid_ratio ∈ [0.5, 2.0]  // 出价倍率（相对于基准出价）

Reward r_t = conversion_value_t - λ · constraint_violation_t

约束：
- 日预算约束: Σ spend_t ≤ Budget
- ROI约束: Σ value_t / Σ spend_t ≥ ROAS_target
```

#### 模型架构：Actor-Critic（PPO）

```
State Encoder（MLP）
         ↓
  共享特征表示 h_t
    /           \
Actor（出价策略）  Critic（价值函数）
  μ_θ(s_t)      V_φ(s_t)
  σ_θ(s_t)
    ↓
  Gaussian Policy: a_t ~ N(μ_θ, σ²_θ)
  Clip: bid = clip(base_bid × a_t, min_bid, max_bid)
```

**PPO 目标函数**：

```
L_PPO = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)] - c₁L_VF + c₂H[π_θ]
```

**约束处理（Lagrangian Relaxation）**：

```
L_CMDP = L_PPO - λ_budget · max(0, spend - budget) - λ_roi · max(0, roi_target - actual_roi)
λ 通过对偶梯度法（Dual Gradient Descent）自动更新：
λ_{t+1} = λ_t + α · constraint_violation_t
```

#### 训练环境：离线模拟器

```
模拟器组件：
1. 出价拍卖模拟器（GSP / VCG 拍卖机制）
   输入：本方出价 + 竞争对手出价分布（历史统计）
   输出：是否赢得竞价 + 实际扣费

2. 转化模拟器
   输入：赢得曝光的 ad impression
   输出：是否转化（pCVR 模型打分 + 噪声）

3. 竞争对手模拟器
   POMDP：对手出价行为用历史分布采样建模

数据规模：
- 训练：3个月历史竞价日志，约 1000 亿次竞价记录
- 模拟器 + RL 训练：离线训练 200 epoch（GPU集群，约 48h）
```

### 实施过程

#### 数据准备

- 竞价日志：bid_request, bid_price, won, cost, conversion, value。
- 竞争对手建模：用竞争对手出价分布（按时段、地域、受众分层建模）构建模拟环境。
- Reward 计算：conversion_value = 成交金额 × GMV_weight。

#### 模型训练/迭代

| 迭代 | 方案 | ROI满足率 | 预算消耗率 |
|------|------|----------|----------|
| v1（规则出价）| target CPA × pCVR | 71% | 82% |
| v2（DQN离散动作）| 出价分 10 档 | 79% | 87% |
| v3（PPO连续出价）| Gaussian Policy | 84% | 91% |
| v4（PPO+约束）| CMDP + Lagrangian | **89%** | **94%** |

#### AB 实验设计

- 对照：规则出价（target CPA × pCVR，有人工调价）
- 实验：Auto Bidding RL（全自动，无人工干预）
- 广告主分层：按日预算（<1万/1-10万/>10万）分层实验，21天

### 量化结果（STAR格式）

- **Situation**：广告主手动出价运营成本高，规则出价无法动态适应竞争变化。
- **Task**：构建 Auto Bidding 系统，在满足 ROI 和预算约束下自动化出价。
- **Action**：设计 CMDP 建模 + PPO 出价策略 + Lagrangian 约束处理 + 离线模拟器训练。
- **Result**：
  - 广告主 ROI 满足率从 71% 提升到 **89%**（约束满足率）
  - 预算消耗率从 82% 提升到 **94%**（更充分利用广告主预算）
  - 广告主 GMV 平均提升 **+18%**（在相同预算下）
  - 广告主运营人力节省 **60%**（无需频繁手动调价）
  - 系统管理广告主数量从 500 扩展到 **5000+**（自动化带来规模效应）

### 面试亮点

**可深挖的技术点（3条）**：

1. **离线模拟器与在线环境的分布偏移（Sim-to-Real Gap）**：离线模拟器基于历史竞价数据，无法准确反映实时竞争态势（对手策略在变化）。缓解：① 定期用最新在线数据更新竞争对手模型；② Domain Randomization：在模拟器中随机化竞争强度参数，提高策略鲁棒性；③ Conservative Policy：部署时设置保守探索（σ 较小），先上线 + 小范围 Fine-tune。

2. **Lagrangian 对偶法的收敛性**：λ 的梯度更新步长 α 的选择很关键。α 太大：λ 震荡，约束满足不稳定；α 太小：约束满足慢。实践用自适应 λ 更新（Adam 更新 λ），并对 λ clip 到非负（约束为单向不等式）。

3. **奖励设计中的 delayed reward 问题**：广告转化可能在点击后数小时才发生（后归因），奖励存在延迟。解决：① 引入即时代理奖励（pCVR 打分，实时可得）；② 延迟奖励归因（统计学上将24h转化归因回展示时刻）；③ Multi-step Return（n-step TD）减少延迟影响。

**可能被追问的问题+参考答案框架**：

- **Q：为什么选 PPO 而不是 DQN？**
  A：出价是连续动作空间（出价倍率 0.5~2.0），DQN 只能处理离散动作（需要离散化，损失精度）。PPO 天然支持连续动作（Gaussian Policy），且 on-policy 的 clipped objective 比 DQN 的 off-policy 更新更稳定（DQN 在高维状态空间容易发散）。另外 PPO 的超参数更少，工程上更易调试。

- **Q：如何保证 Auto Bidding 不会学到"恶意"策略（如先抬价再降价，操纵市场）？**
  A：① 出价约束：bid 范围 clip 在 [0.5×base, 2.0×base]，防止极端出价；② 奖励设计不包含竞争对手损失（纯以广告主自身 ROI/GMV 为目标）；③ 在线监控：出价策略异常检测（如同一广告主在某时段出价突然 10x），触发告警和自动熔断。

---

## 项目5：实时特征工程平台建设

### 背景

- **公司/团队背景**：某头部内容平台基础算法团队，负责为推荐、搜索、广告三条业务线提供统一的特征服务，支持日均 200 亿次特征查询。
- **问题痛点**：
  - 各业务线**特征重复建设**（推荐团队和广告团队分别维护类似的 CTR 统计特征，逻辑不一致）。
  - 离线特征 T+1 延迟，**实时热点（Breaking News、直播开播）无法及时反映在特征中**，导致实时内容推荐效果差。
  - 特征口径不一致：训练时用 T-1 的统计特征，线上 serving 用实时统计，**训练-推断特征穿越**问题严重。
  - 特征 serving 延迟高（P99 > 50ms），成为精排延迟瓶颈。

### 技术方案

#### 整体架构：Lambda 架构（批+流融合）

```
数据流：
用户行为事件 → Kafka → [Flink实时处理] → Redis Feature Store (实时特征)
                    ↓
              [Spark批处理(T+1)] → Hive → [特征同步] → Redis Feature Store (离线特征)

Feature Server：
- 统一 API：feature_get(user_id, item_id, feature_keys)
- 多级缓存：Local Cache (L1, 1ms) → Redis Cluster (L2, 3ms) → Hive (L3, fallback)
- 延迟目标：P50 < 2ms, P99 < 10ms
```

#### 实时特征计算（Flink）

```
核心实时特征：
1. 用户实时行为统计（滑动窗口）：
   - 过去5min/30min/1h 的点击/曝光/点赞数量
   - 实时类目偏好（过去1h点击的L2类目分布）

2. Item实时统计：
   - 过去5min/1h 的曝光/点击/CTR
   - 实时热度分（基于曝光量指数衰减）

3. 用户-Item交叉特征：
   - 用户对该作者历史互动次数（滑动1h窗口）
   - 用户对该类目近期兴趣分（实时更新）

Flink 设计：
- Window Type: Sliding Window（5min/30min/1h，步长1min）
- State Backend: RocksDB（支持大状态持久化）
- 消息队列：Kafka，分区数=64，消费延迟<200ms
- 输出：Redis HSET，TTL=2h（实时特征设短TTL自动过期）
```

#### 特征穿越防护（Point-in-time Correct）

```
问题：训练时 "2024-01-15 10:30" 的样本，误用了 "2024-01-15 11:00" 的特征（未来特征）

解决：时间分区特征存储
- 每小时打快照：feature_snapshot_20240115_10 = {user_id: {ctr_1h: 0.05, ...}}
- 训练时样本按时间戳查找对应快照，严格 point-in-time 对齐
- 验证：训练集中随机抽样，检查特征时间戳 < 样本时间戳

工程实现：
- Hive 特征表按小时分区
- 训练样本 JOIN 时用 asof_join（时间最近的过去时刻）
```

#### 特征重要性分析与裁剪

```
定期分析：
1. SHAP（Shapley Additive exPlanations）计算每个特征的贡献度
2. 去除 SHAP 贡献 < 阈值 的特征（减少存储和计算成本）
3. 相关性分析：去除与已有特征相关性 > 0.95 的冗余特征

特征数量变化：
优化前：1832 个特征（含大量冗余）
优化后：847 个特征
延迟降低：50ms → 23ms（特征数量减少，网络传输和特征提取时间减少）
```

### 实施过程

#### 数据准备

- 实时数据源：用户行为事件（click/view/like），经 Kafka 实时消费。
- 历史数据：Hive 存储 90 天行为日志，支持回溯特征计算（补齐历史样本特征）。
- 数据量：日均 500 亿事件，Kafka 峰值 50 万 QPS。

#### 建设过程

1. **统一特征注册中心**：所有特征定义（名称、类型、计算逻辑、TTL）注册到元数据服务。
2. **跨团队特征共享**：推荐、广告、搜索通过统一 Feature Store 调用，消除重复建设。
3. **训练特征回放服务**：离线训练时，样本按时间戳查找历史快照，确保 Point-in-time 正确。

#### AB 实验设计

- 对照：原有特征体系（T+1 离线特征 + 少量规则实时特征）
- 实验：新实时特征平台（全面实时特征 + 穿越防护）
- 核心指标：精排 AUC（离线）、线上 CTR、延迟

### 量化结果（STAR格式）

- **Situation**：特征重复建设、实时性差、穿越问题、serving 延迟高。
- **Task**：建设统一实时特征平台，支持批流融合，解决穿越问题。
- **Action**：基于 Flink+Redis+Hive 构建 Lambda 架构特征平台，建立穿越防护机制。
- **Result**：
  - 精排 AUC 提升 **+0.009**（引入实时特征后，实时热点内容精排效果提升）
  - 线上 CTR 提升 **+1.5%**（实时特征更准确反映用户当前兴趣）
  - 特征 serving P99 延迟从 **50ms → 9ms**（多级缓存 + 特征裁剪）
  - 特征开发效率提升 **3x**（统一平台，复用率高，跨团队共享）
  - 消除 **23 处** 训练-推断特征穿越问题（系统性检测后发现）
  - 热点内容实时特征更新延迟 **< 30 秒**（Flink 实时计算）

### 面试亮点

**可深挖的技术点（3条）**：

1. **Flink 实时统计的精确性 vs 吞吐的权衡**：精确的滑动窗口（Sliding Window）需要保存窗口内所有事件，状态大、内存压力高。实践中对 count 类特征用近似算法：Count-Min Sketch（估算频次，误差 <1%，内存节省 10x）；对 CTR 类特征用指数移动平均（EMA）替代精确滑动窗口，`ctr_t = α·click_t + (1-α)·ctr_{t-1}`，α 对应时间衰减系数。

2. **特征穿越的自动检测**：手工检查不可扩展，建立自动化检测：对训练集随机采样 1000 条样本，用样本时间戳查询 Feature Store，比较实际存储时间戳与样本时间戳，若存在"特征时间 > 样本时间"则报警。集成到特征注册流程（每次新特征上线前必须通过穿越检测）。

3. **Redis Cluster 的热 Key 问题**：热门 item（如热门直播、爆款商品）的特征被高频查询，单个 Redis 节点 QPS 达到上限。解决：① 本地缓存热 Key（Local Cache，TTL=1s，命中率 80%+）；② 热 Key 副本（多备份到不同节点，查询时随机选择）；③ 异步预热（预测下一时间段热 Key，提前写入本地缓存）。

**可能被追问的问题+参考答案框架**：

- **Q：训练时用的特征和线上 serving 用的特征，如何保证一致性？**
  A：核心是"训练特征离线化"策略：① 训练样本生成时，同步将特征值存储到样本 log（样本携带特征）；② 线上特征经过相同的预处理 pipeline（归一化、分桶参数一致）；③ 特征版本管理：模型版本和特征版本绑定，线上 serving 用与训练对应的特征版本；④ 实时特征对齐检验：定期对比离线特征分布（训练集统计）和线上特征分布（实时监控），若 PSI > 0.1 触发特征漂移告警。

- **Q：如何选择哪些特征做实时化，哪些保持离线？**
  A：决策矩阵：① 时效性需求高（如热度分、实时 CTR）→ 实时；② 计算复杂度低（可以在 Flink 中完成）→ 实时；③ 数据量小（状态可以放入内存）→ 实时；④ 反之（复杂的图特征、大规模统计）→ 离线（T+1 或 T+6h）。一般经验：用户行为统计实时化（高时效），item 内容特征离线计算（低频变化），user-item 交叉特征按场景决定。

---

## 项目6：A/B实验平台建设与因果推断应用

### 背景

- **公司/团队背景**：某头部互联网公司数据平台团队，负责全公司 A/B 实验平台的建设和维护，支撑推荐、搜索、广告、产品等 50+ 个团队，同期在线实验数量 300+。
- **问题痛点**：
  - 各团队**各自实现 AB 分流逻辑**，分流桶互相干扰（同一用户在不同实验中对照/实验组混杂），实验结果不可信。
  - **辛普森悖论**：某推荐算法实验全量 CTR 提升，但分用户活跃度分层后发现高活跃用户 CTR 下降，原因是实验组中高活跃用户比例更高，导致汇总指标被混淆。
  - **实验间干扰（Interference）**：用户 A 看到新推荐算法，与用户 B 的社交互动影响 B 的行为，违反了 SUTVA（Stable Unit Treatment Value Assumption），导致实验结果偏估。
  - **Novelty Effect**：新功能上线后短期 CTR 虚高，2 周后回落，团队容易在 Novelty Effect 消退前就决策上线。

### 技术方案

#### 实验平台核心架构

```
实验管理系统：
├── 实验配置（名称、流量、分桶策略、指标）
├── 分流服务（Hashing + 分层实验框架）
├── 曝光日志收集（按 user_id + exp_id 打标）
└── 数据分析（指标计算 + 统计检验 + 报告）

分层实验框架（解决实验干扰）：
Layer 1: 召回层实验（流量桶 A1-A10）
Layer 2: 精排层实验（流量桶 B1-B10）
Layer 3: 重排层实验（流量桶 C1-C10）
Layer 4: UI/产品实验（流量桶 D1-D10）

同一用户在每层独立 Hash，各层实验互不干扰：
bucket_i = Hash(user_id + layer_salt_i) % 100
```

#### 统计检验框架

```
基础检验：双侧 t-test（连续指标）/ Chi-square（比例指标）
显著性水平：α = 0.05（需 Bonferroni 修正多重检验）
功效要求：power = 0.80（最小样本量计算）

最小样本量公式：
n = 2 × (z_{α/2} + z_β)² × σ² / δ²
z_{α/2} = 1.96（双侧 α=0.05）
z_β = 0.842（power=0.80）
σ: 指标的标准差（历史数据估算）
δ: 最小可检测效应大小（MDE，业务定义）

方差缩减（CUPED）：
使用实验前的历史指标 X 作为协变量，控制组间初始差异：
Y_cuped = Y - θ · (X - E[X])
θ = Cov(Y, X) / Var(X)  // OLS 估计

CUPED 效果：在不增加样本量情况下，方差缩减约 20-40%，相当于样本量增加 1.2-1.7x
```

#### 因果推断应用

**场景1：多臂老虎机（MAB）替代 AA/AB 传统实验**

```python
# Thompson Sampling 动态分配流量
def thompson_sampling(arms):
    samples = []
    for arm in arms:
        alpha = arm.conversions + 1
        beta = arm.impressions - arm.conversions + 1
        samples.append(np.random.beta(alpha, beta))
    return np.argmax(samples)

# 效果：同等置信度下，比固定 AB 多发现 15% 的显著实验
# 损失（Regret）比传统 AB 低 23%
```

**场景2：双重差分（DiD）评估自然实验**

```
场景：某城市的直播功能灰度（非随机，仅开放给特定城市），
     评估对该城市 GMV 的影响

DiD 模型：
Y_it = α + β·Treat_i + γ·Post_t + δ·(Treat_i × Post_t) + ε_it

Treat_i: 是否为处理城市（1=是，0=对照城市）
Post_t: 是否为政策后时期（1=是，0=政策前）
δ: DiD 估计量（真实因果效应）

前提：平行趋势假设（Parallel Trend）验证：
- 处理前，处理城市和对照城市的 GMV 趋势平行
- 用处理前多期数据做安慰剂检验
```

**场景3：PSM（倾向得分匹配）处理非随机化实验**

```python
# 场景：评估"开启通知推送"功能对留存的影响
# 问题：开启通知的用户本身就更活跃（自选择偏差）

# 步骤1：估计开启通知的倾向得分（Logistic Regression）
ps_model = LogisticRegression()
ps_model.fit(X_confounders, treatment)  # X: 用户活跃度、注册天数等
propensity_score = ps_model.predict_proba(X)[:, 1]

# 步骤2：按倾向得分匹配（1:1 最近邻匹配）
matched_pairs = match(treatment_group, control_group, propensity_score, caliper=0.05)

# 步骤3：在匹配后的样本上估计 ATE
ATE = mean(Y_treatment) - mean(Y_control)  # 匹配后样本

# 结果：控制自选择偏差后，开启通知的真实留存提升为 +3.2%（vs 未校正的虚高 +8.1%）
```

### 实施过程

#### 平台建设阶段

1. **分层实验框架**：解决多实验干扰，支持 100+ 并行实验不互相干扰。
2. **CUPED 方差缩减**：标准化集成到所有实验报告，实验决策速度提升。
3. **因果推断工具集**：封装 DiD、PSM、IV 等方法，提供给各业务团队自助使用。
4. **Novelty Effect 检测**：引入分时段分析，自动识别实验效果的时间衰减趋势。

#### AB 实验设计的实践规范

```
实验 Checklist：
□ AA 实验先验证（分流一致性检验，p > 0.05）
□ 最小样本量计算（基于历史指标方差和 MDE）
□ 实验周期 ≥ 2 周（覆盖用户行为周期）
□ SRM 检验（Sample Ratio Mismatch，实验组与对照组比例是否符合预期）
□ 多重检验校正（Bonferroni 或 BH-FDR）
□ 异质性分析（按用户分层、设备、地域）
□ Novelty Effect 监控（分周观察效果变化）
```

### 量化结果（STAR格式）

- **Situation**：各团队自建 AB，实验干扰严重，统计方法不规范，决策不可信。
- **Task**：建设统一 AB 实验平台，引入因果推断工具，提升实验可信度和效率。
- **Action**：设计分层实验框架，集成 CUPED 方差缩减，建立因果推断工具集，制定实验规范。
- **Result**：
  - 支持同期 **300+** 并行实验，实验干扰问题降低 **90%**（SRM 告警减少）
  - CUPED 实施后，同等样本量下实验敏感度提升 **35%**（等效于样本量增加 1.35x）
  - 实验平均决策周期从 **21 天 → 12 天**（统计效能提升 + 规范流程）
  - 发现并纠正 **15 个**历史实验的错误结论（因重新引入分层控制和 CUPED）
  - PSM 因果推断应用：纠正了 3 个产品功能评估中的自选择偏差，防止了错误的大规模推广决策
  - MAB 替代部分 AB 实验：在快速迭代场景下，Regret 降低 **23%**，实验效率提升 **15%**

### 面试亮点

**可深挖的技术点（3条）**：

1. **SRM（Sample Ratio Mismatch）的检测和原因分析**：SRM 是实验组/对照组实际用户比例与设计比例不符，是实验结果失效的重要信号。检测：Chi-square 检验（observed vs expected 比例）。常见原因：① 机器人流量过滤不一致；② 新用户 Cookie 赋值延迟（用户在赋予 exp_id 前就有行为，记录到了错误的桶）；③ 实验代码逻辑 Bug（某些条件下用户被强制分到对照组）。

2. **CUPED 的局限性和适用条件**：CUPED 要求协变量 X（实验前指标）与结果变量 Y（实验后指标）有线性相关关系，且协变量不受实验处理影响。若 X 与 Y 相关性低（< 0.3），CUPED 方差缩减效果有限。此外，若实验时间很短（<3天），没有足够的"实验前期"数据构建协变量，CUPED 无法使用。替代方案：Stratified Sampling（按用户活跃度分层随机分配）。

3. **双重差分的平行趋势假设验证**：这是 DiD 最重要的前提。验证方法：① 视觉检验：画出处理组和对照组在处理前多期的趋势曲线，目视是否平行；② 安慰剂检验（Placebo Test）：假设处理时间提前（用处理前数据），若 DiD 估计量显著不为 0，说明平行趋势可能不成立；③ 预期检验（Pre-trend Test）：用处理前数据做回归，检验各期 DiD 系数是否统计上不显著。

**可能被追问的问题+参考答案框架**：

- **Q：推荐系统实验中常见的 SUTVA 违反是什么，如何处理？**
  A：SUTVA 要求一个用户受到的处理只影响自己，不影响他人的潜在结果。推荐系统中的违反：① **网络效应**：用户 A 看到新内容后转发，影响好友 B 的行为（社交场景）；② **供给侧干扰**：推荐算法变化影响创作者的创作策略，进而影响所有用户；③ **库存效应**：实验组用户大量购买某商品导致对照组用户无法购买。解决方案：① 按地理/社区聚类分配（Cluster Randomization，同一社区用户在同一组）；② 双边实验（同时实验用户侧和供给侧）；③ 网络去偏（Graph-based Debiasing）。

- **Q：如何判断一个实验结果是否值得推全？**
  A：决策框架：① **统计显著性**：p < 0.05（经多重检验校正），置信区间不含 0；② **业务显著性（Practical Significance）**：效应大小超过 MDE（最小可检测效应，通常为线上成本可接受的最小收益）；③ **异质性分析**：无明显用户群受损（分层后无显著负效应）；④ **Novelty Effect 排除**：实验效果在第 2-3 周没有明显衰减；⑤ **工程评估**：延迟影响、系统稳定性；以上全部满足，推荐推全。

---

---

## 项目7：生成式召回系统建设（Semantic ID + Generative Retrieval）

### 背景

- **公司/团队背景**：某头部短视频平台推荐算法团队，负责首页信息流召回模块，item 库约 2 亿条视频，日均 PV 约 30 亿。
- **问题痛点**：
  - 现有召回以双塔为主，**user-item 特征交叉能力受限**（双塔结构决定了 user 侧和 item 侧独立编码，只有最后 dot product 才有交叉）。
  - ANN 检索（HNSW）在 item 量超过亿级后，**内存开销极大**（2亿×128维×4 bytes ≈ 100GB），扩容成本高。
  - 新视频上线后需要先入 ANN 索引再能被召回，**从上传到可召回存在约 3-5 分钟的延迟**。
  - 双塔召回的 item 多样性有限：余弦相似度接近的 item 集中在少数类目，**长尾内容曝光不足**。

### 技术方案

#### 核心思路：Generative Retrieval（生成式召回）

不再用 ANN 检索，而是让模型直接"生成" item 的 Semantic ID token 序列：

```
输入: 用户行为序列 [v1, v2, v3, ..., vn]
    ↓  Encoder（Transformer）
    ↓  Decoder（自回归）
输出: token1 → token2 → token3  →  映射回 item_id
```

#### Item Tokenization：残差 K-Means 量化

> 改进自 TIGER 的 RQ-VAE，解决 Hourglass 现象

```
Step 1: 预训练多模态 Item Embedding
- 视频帧: ViT-Large（16帧均值）
- 文字标题: BERT 中文
- 行为对齐: 用 user-item co-click 对做对比学习微调 embedding
- 最终 item embedding: 256维，对齐语义与行为分布

Step 2: 残差 K-Means 分层量化
第1层: K=4096 簇，item → 最近簇 c1_id，残差 r1 = e - center_1
第2层: K=4096 簇，对 r1 → 最近簇 c2_id，残差 r2 = r1 - center_2
第3层: K=4096 簇，对 r2 → 最近簇 c3_id

item_i 的 Semantic Token = (c1_id, c2_id, c3_id)
token 空间: 4096^3 ≈ 680亿 种组合，足够覆盖 2亿 item
```

与 RQ-VAE 的关键区别：每层 K-Means 强制平衡（每个簇的 item 数量相近），避免 code 分布不均匀。

#### Encoder-Decoder 生成模型

```
Encoder:
- 输入: 用户历史 50 条行为的 token 序列（每条行为 = 3个 token）
- Position Encoding: 行为时序
- Transformer 6层, d_model=512

Decoder（自回归）:
- 第1步: cross-attention(user_rep) → 生成 c1_id（beam_size=10）
- 第2步: 在 c1_id 约束下生成 c2_id（beam_size=10）
- 第3步: 在 (c1_id, c2_id) 约束下生成 c3_id（beam_size=10）
- 最终: beam search 得到 top-100 个 token 序列 → 映射回 top-100 item

约束 beam search（Constrained Decoding）：
- 每一步只在合法 token（有对应 item 的 token）中做 softmax
- 避免生成不存在的 item
```

#### 训练策略

```
训练目标: 给定用户历史，预测下一个有效互动 item 的 token 序列
Loss: Cross-Entropy（每个 token 位置独立）

L = -1/T Σ_t [log P(c1_t|H_u) + log P(c2_t|H_u, c1_t) + log P(c3_t|H_u, c1_t, c2_t)]

数据规模: 30天点击+完播日志，约 200亿 (user, item) 对
训练: 8×A100，约 36h 完成一轮

新 item 冷启动:
- 新视频上传后立即计算多模态 embedding
- K-Means 量化（无需重训，用已有 codebook 直接映射）→ 得到 token
- 立即可参与生成式召回（无需 ANN index rebuild）
```

### 实施过程

#### 对比实验（生成式召回 vs 双塔召回）

| 指标 | 双塔+HNSW | 生成式召回 | 变化 |
|------|-----------|-----------|------|
| Recall@50 | 38.2% | 42.7% | +4.5% |
| Recall@200 | 61.4% | 67.8% | +6.4% |
| 长尾 item 召回率（底部30%）| 12.3% | 19.1% | +55% |
| 新 item 首次召回延迟 | 3-5 min | **< 1 min** | -80% |
| 内存占用（索引部分）| ~100GB | ~8GB（codebook）| -92% |
| 召回延迟（P99）| 6ms | 18ms | +3x（工程优化后可降至 10ms）|

#### AB 实验

- 召回阶段：生成式召回替换一路双塔召回（保留另一路双塔+用户协同过滤）
- 流量：20%，3周
- 线上结果：
  - CTR +0.9%（召回质量提升）
  - 长尾内容曝光占比 +12%（多样性改善）
  - 用户停留时长 +1.3%

### 量化结果（STAR格式）

- **S**：双塔召回内存成本高、新 item 冷启动慢、长尾多样性不足。
- **T**：探索生成式召回技术，替换一路双塔，提升召回质量和新 item 时效。
- **A**：设计残差 K-Means tokenization，训练 Encoder-Decoder 生成模型，实现约束 beam search 召回。
- **R**：Recall@200 **+6.4%**，长尾召回率 **+55%**，新 item 首次召回延迟从 3-5 分钟降至 **< 1 分钟**，ANN 索引内存降低 **92%**，线上 CTR **+0.9%**，停留时长 **+1.3%**。

### 面试亮点

**可深挖的技术点（3条）**：

1. **为什么选残差 K-Means 而非 RQ-VAE？**
RQ-VAE（VQ-VAE的残差版）在推荐场景存在"Hourglass 现象"：第1层 codebook 的某些 code 被大量 item 使用，而第 2、3 层的区分度极低，导致不同 item 的 token 后几位几乎相同，模型难以区分。残差 K-Means 每层强制均匀分配（用 balanced K-Means，每个簇 item 数相近），保证每层 code 都有足够区分度。实验对比：相同 token 长度下，残差 K-Means 的 item 区分度（token 碰撞率）比 RQ-VAE 低 60%。

2. **约束 Beam Search 的工程实现**：
Decoder 每步都要在"合法 token 空间"做 softmax（非全词表，只有当前前缀下存在 item 的 token 才合法）。实现：预构建前缀树（Trie Tree），key=(c1, c2)，value=合法的 c3 列表。每步 decode 时查 Trie 得到合法 token 集合，在此集合内做 top-k 选择。Trie 大小约 4GB（2亿 item × 3层），加载到内存后查询延迟 < 0.1ms。

3. **生成式召回的 Exposure Bias 问题**：
训练时 Decoder 用真实 token 序列（Teacher Forcing），推理时用自身生成的 token（自回归），训练推理不一致（Exposure Bias）。缓解：Scheduled Sampling（训练时以概率 p 用模型预测 token 替代真实 token）；p 从 0 逐步增大到 0.3，让模型学会纠错。

**可能被追问的问题+参考答案框架**：

- **Q：生成式召回和双塔召回能直接比较吗？各自适合什么场景？**
A：不能简单替代，应该互补并行。双塔：延迟极低（ANN < 5ms），对热门 item、高曝光商品效果稳定，工程成熟；适合对延迟要求极高、item 库相对静态的场景。生成式召回：延迟较高（beam search ~15ms），但冷启动优秀、多样性好、不需要大规模 ANN 索引；适合 item 更新频率高（短视频/新闻）、多样性要求高的场景。实践：两路并行，最后 merge 去重送粗排。

- **Q：Token 冲突问题：不同 item 有相同 token 怎么处理？**
A：K-Means 量化后必然存在 token 碰撞（相同 token 映射多个 item）。处理方式：① 三层 token（4096^3）的碰撞率 < 0.001%，基本可忽略；② 碰撞的 item 都出现在 beam search 结果中，downstream ranking 会区分；③ 对热门 item 可加一层 item-specific 后缀 token（第4层，仅热门 item 专属），降低 top item 的碰撞。

---

## 项目8：LLM 语义特征增强推荐系统

### 背景

- **公司/团队背景**：某综合电商平台推荐算法团队，负责商品详情页"相关推荐"和首页"猜你喜欢"，商品库约 5000 万 SKU，类目 3 万+，日均 PV 15 亿。
- **问题痛点**：
  - 传统推荐强依赖行为信号（点击/购买），但 **60% 的 SKU 是长尾商品**，历史行为极少（< 10 次），CF 信号稀疏，排序效果差。
  - 商品标题/描述包含大量语义信息（材质、场景、风格），但原有模型只用分词后的 bag-of-words 特征，**深层语义理解能力不足**（例如"透气运动鞋"和"跑步鞋"语义相关，但词汇不同导致无法关联）。
  - 用户端：用户的真实购买意图往往藏在浏览行为序列的语义模式里（"最近看了3双跑鞋→用户在选跑步装备"），**行为语义归因能力弱**。
  - 新兴品类（如"新中式服装"）上线后无历史数据，**跨类目冷启动**基本失败。

### 技术方案

#### 整体思路：LLM 作为语义特征提取器（不替换 CF，而是增强）

```
架构：
                     ┌──────────────────────────────┐
用户行为序列(IDs)  →  │    协同过滤模型（DIN/SIM）     │ → CF embedding (256d)
                     └──────────────────────────────┘
                               ↕ 对齐
用户历史行为文本描述  →  LLM  → 语义 embedding (256d)
                               ↕ 融合
商品ID → CF item embedding    最终表示 → 排序打分
商品文本(标题+属性) → LLM → 语义 item embedding
```

#### LLM 语义表示提取

**模型选择**：`bge-large-zh-v1.5`（通义千问 embedding，中文优化，1024 维，本地部署）

**用户语义表示构建**：

```python
# 把用户最近 20 条行为转化为自然语言摘要（离线，每日更新）
prompt = f"""
用户最近浏览的商品（按时间倒序）：
1. {item1_title} - 品类：{category1} - 价格区间：{price_range1}
2. {item2_title} - 品类：{category2} - 价格区间：{price_range2}
...
请用一段话总结该用户当前的购物兴趣和意图：
"""
# 用 Qwen-7B-Chat 生成摘要（离线批处理）
summary = llm.generate(prompt)
# 用 embedding 模型提取向量
user_semantic_emb = bge_model.encode(summary)  # 1024维
# 降维到256维（PCA/Linear Projection）
user_semantic_emb = projection(user_semantic_emb)  # 256维
```

**商品语义表示构建**：

```python
# 商品侧：标题 + 属性 + 描述 → 拼接 → embedding
item_text = f"{item.title} | {item.category_path} | 材质:{item.material} | 适用场景:{item.scene} | {item.description[:200]}"
item_semantic_emb = bge_model.encode(item_text)  # 1024维 → 降维256维

# 数据规模：5000万 SKU，批量离线处理
# GPU：4×A100，约 18h 完成全量计算
# 增量：每日新商品实时触发 embedding 计算（Kafka trigger）
```

#### 语义对齐（Semantic-CF Alignment）

CF embedding 和 LLM embedding 来自不同空间，需要对齐：

```
对齐目标：
相同用户的 CF embedding 和 LLM embedding 应该相近（InfoNCE Loss）
相同 item 的 CF embedding 和 LLM embedding 应该相近

L_align = -log(sim(cf_u, sem_u) / Σ_j sim(cf_u, sem_uj⁻))

对齐方式：在 DIN 模型基础上，加一个 Alignment Head：
- 输入：DIN 的 user representation（CF侧）
- 目标：拉近对应用户的 LLM semantic embedding（语义侧）
- 反向：同时更新 CF model 参数，让 CF embedding 向语义空间靠拢
```

#### 融合策略（三种，按场景选择）

```
方案A（特征拼接）：
final_user_rep = concat([cf_user_emb, semantic_user_emb])  // 512维
适用：训练数据充足的热门品类

方案B（MoE门控融合）：
gate = sigmoid(W · [cf_user_emb; semantic_user_emb])
final = gate * cf_user_emb + (1-gate) * semantic_user_emb
适用：行为稀疏的中等品类（语义权重自适应）

方案C（纯语义）：
仅用 semantic_user_emb，丢弃 CF
适用：极度冷启动（< 3次行为）或新品类上线
```

#### RAG 增强的 Item 相关推荐

```
场景：商品详情页"相关推荐"
传统：i2i 协同过滤（共现矩阵）
新方案：语义 ANN 检索

query = item_semantic_emb (当前商品)
candidates = HNSW.search(query, top_k=500)  // 语义相似商品
合并 i2i 候选 + 语义候选 → 去重 → 粗排
```

### 实施过程

#### 数据准备

- 商品文本：SKU 标题（平均 35 字）+ 属性（平均 15 个 KV 对）+ 描述（截断 200 字）。
- 用户行为摘要：每日 T+1 批处理，Qwen-7B-Chat 生成 20 字摘要（日均处理 5000 万活跃用户）。
- Embedding 更新频率：商品全量每周重算（新商品实时），用户每日增量。

#### 模型训练/迭代

| 版本 | 方案 | 长尾商品AUC | 全量AUC |
|------|------|------------|---------|
| baseline | DIN（CF only）| 0.682 | 0.749 |
| v1 | +语义 item embedding（拼接）| 0.701 | 0.754 |
| v2 | +语义 user embedding（拼接）| 0.718 | 0.758 |
| v3 | +对齐 Loss（Alignment）| 0.729 | 0.762 |
| v4 | MoE 门控融合 | **0.736** | **0.764** |

#### AB 实验设计

- 对照：原有 DIN 模型（CF + 简单文本特征 bag-of-words）
- 实验：DIN + LLM 语义特征（MoE 融合）
- 流量：20%，实验期 21 天
- 分层分析：按商品行为频次（热门/中等/长尾）分别统计

### 量化结果（STAR格式）

- **S**：长尾商品行为稀少，CF 无能为力；语义理解能力弱，跨类目推荐质量差。
- **T**：引入 LLM 语义特征增强推荐，重点提升长尾和冷启动商品效果。
- **A**：bge 提取商品/用户语义 embedding，Qwen-7B 生成用户购物意图摘要，语义-CF 对齐，MoE 融合。
- **R**：
  - 全量 AUC：0.749 → **0.764**（+0.015）
  - 长尾商品（< 50次历史行为）AUC：0.682 → **0.736**（+0.054，提升最显著）
  - 跨类目推荐 CTR **+8.3%**（语义理解打通了近义类目）
  - 新品（上线 < 7天）首周 CTR **+31%**（语义冷启动效果显著）
  - 线上全量 CTR **+1.2%**，GMV **+1.7%**
  - Embedding 服务 P99 延迟 < 5ms（HNSW 语义 ANN 检索）

### 面试亮点

**可深挖的技术点（3条）**：

1. **LLM Embedding 的训练-推断一致性**：LLM 生成的 embedding 是离线批处理的（T+1 或实时），而模型训练时也是离线特征。潜在的一致性风险：LLM 模型版本更新后，embedding 空间发生 shift，导致旧 embedding 和新 embedding 不可比。解决：embedding 版本管理（每次 LLM 更新后全量重新计算，并同步更新 HNSW 索引）+ 灰度上线（先替换 10% 流量验证 embedding 一致性）。

2. **Qwen 生成用户意图摘要的质量控制**：LLM 生成的摘要质量不稳定，可能产生幻觉或噪声。质控：① 设置最大/最小 token 限制（10-50 字）；② 温度参数设为 0（确定性输出，减少幻觉）；③ 人工抽样 500 条评估摘要质量（ROUGE vs 真实购买商品的相关性）；④ 对摘要质量低的用户（LLM 置信度低）回退到 CF only。

3. **语义 embedding 的流行度偏差**：LLM 的预训练数据里，知名品牌/热门品类的描述更丰富，导致其 embedding 质量更高，长尾小众商品反而在语义空间里离群（embedding 质量差）。缓解：① 在 embedding 训练阶段用推荐业务数据做微调（fine-tune bge，使 embedding 更贴近用户行为分布）；② 长尾商品的语义特征权重适度降低（在 MoE 融合时，长尾商品的 gate 对 semantic 的权重给 cap 上限）。

**可能被追问的问题+参考答案框架**：

- **Q：为什么不直接用 LLM 做排序打分（cross-encoder），而要提取 embedding？**
A：cross-encoder（每个 user-item 对单独输入 LLM 做打分）精度最高，但线上延迟不可接受（7B LLM 单次 inference ~100ms，精排要处理 200 个候选 → 总延迟 20s，不可用）。bi-encoder（分别编码 user 和 item，embedding 离线）牺牲部分精度，但延迟 < 5ms，可落地。工程 tradeoff 的结论：LLM 做离线 embedding，线上轻量 MLP 做融合打分。

- **Q：LLM 提取的 user embedding 和 item embedding，如何保证语义空间一致？**
A：关键是用同一个 embedding 模型处理 user 描述和 item 描述，且用 Alignment Loss 在业务数据上对齐两个空间（user 向量应该和他购买的 item 向量相近）。具体：用点击/购买对 (user_emb, item_emb) 做对比学习 fine-tune（InfoNCE），确保 user 的语义偏好和 item 的语义内容在同一空间里可以做相似度计算。

---

## 项目经历汇总表

| 项目 | 核心技术 | 主要亮点 | 关键指标 | 时期 |
|------|---------|---------|---------|------|
| 召回重构 | 双塔+HNSW | 解决长尾和冷启动 | 召回率+23.4% | 2021-2022 |
| CTR精排迭代 | DIN→DIEN | 兴趣演化建模 | CTR+1.8%, AUC+0.017 | 2021-2022 |
| 多任务学习 | ESMM+MMoE | 解决SSB，多目标联合 | GMV+2.4% | 2021-2022 |
| Auto Bidding | PPO+CMDP | 约束RL，全自动出价 | ROI满足率89% | 2022-2023 |
| 实时特征平台 | Flink+Redis | 穿越防护，特征统一 | P99延迟50→9ms | 2022-2023 |
| AB实验平台 | CUPED+因果推断 | 分层框架，统计规范 | 实验周期-43% | 2022 |
| **生成式召回** | **残差K-Means+Encoder-Decoder** | **新item冷启动<1min，ANN内存-92%** | **Recall@200+6.4%，CTR+0.9%** | **2023-2024** |
| **LLM语义增强** | **bge/Qwen+语义对齐+MoE融合** | **长尾商品AUC+0.054，跨类目打通** | **全量CTR+1.2%，GMV+1.7%** | **2024-2025** |

---

*文档更新时间：2026-03-23 | 作者：MelonEggLearn*
*注：所有数据为合理范围内的虚构数字，面试时请结合个人实际经历调整描述*
