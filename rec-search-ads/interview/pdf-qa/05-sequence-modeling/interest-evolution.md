# 兴趣演化建模：DIN / DIEN / DSIN / SIM

## 1. 从池化到注意力：兴趣建模的演进

传统方法：对用户历史行为做 sum/mean pooling → 固定的用户表示
问题：用户兴趣是多样的，单一向量无法表达

演进路线：
```
Sum Pooling → DIN(候选感知注意力) → DIEN(兴趣演化GRU) → DSIN(Session级建模) → SIM(长序列检索)
```

## 2. DIN (Deep Interest Network)

### 2.1 核心思想

用户兴趣是多样的，不同候选物品应该激活不同的历史行为。用候选物品作为 Query，对历史行为做加权求和。

### 2.2 模型结构

```
候选物品 embedding: e_a (Query)
用户历史行为: [e_1, e_2, ..., e_n]

注意力权重: a_i = f(e_i, e_a)
用户兴趣表示: v_u = sum(a_i * e_i)

f 是 Local Activation Unit:
  concat(e_i, e_a, e_i - e_a, e_i * e_a) → MLP → 1 维标量
```

### 2.3 关键设计：不做 Softmax 归一化

传统注意力：权重经 Softmax 归一化，和为 1
DIN：直接用 MLP 输出作为权重，不做归一化

为什么？
- 保留用户兴趣强度信息
- 如果候选物品与所有历史行为都不相关：
  - Softmax 后权重仍然和为 1，被迫分配注意力（即使全不相关）
  - 不归一化则所有权重接近 0，v_u 接近零向量，反映「用户对此不感兴趣」
- 更符合实际：兴趣的绝对强度比相对强度更重要

### 2.4 Local Activation Unit 的输入设计

```
输入 = concat(e_i, e_a, e_i - e_a, e_i * e_a)
```

- e_i, e_a：原始表示
- e_i - e_a：差异信息（不同在哪）
- e_i * e_a：相似性信息（element-wise product 捕捉同维度的匹配）

这比简单的点积更有表达力，能捕捉更复杂的相关性模式。

### 2.5 DIN 的其他贡献

- Dice 激活函数：数据自适应的 PReLU 变体，根据数据分布调整拐点
- Mini-batch Aware Regularization：根据特征在 mini-batch 中的出现频率调整正则化强度，缓解稀疏特征过拟合

## 3. DIEN (Deep Interest Evolution Network)

### 3.1 核心思想

DIN 只做了静态的注意力加权，没有建模兴趣随时间的演化过程。DIEN 引入两层 GRU 结构，显式建模兴趣的提取和演化。

### 3.2 模型结构

```
第一层 - 兴趣提取层 (Interest Extractor):
  行为序列 [e_1, ..., e_n] → GRU → 隐状态 [h_1, ..., h_n]
  辅助损失: 用 h_t 预测下一个点击行为 e_{t+1}

第二层 - 兴趣演化层 (Interest Evolution, AUGRU):
  输入: 隐状态 [h_1, ..., h_n]
  注意力分数: a_t = softmax(h_t^T * e_a)  (与候选物品的相关性)
  AUGRU: 用 a_t 调制 GRU 的更新门

最终: AUGRU 的最后一个隐状态作为用户兴趣表示
```

### 3.3 AUGRU (Attention-based GRU with Update gate)

标准 GRU 的更新门：
```
u_t = sigmoid(W_u * [h_{t-1}, x_t])
h_t = (1 - u_t) * h_{t-1} + u_t * h_tilde_t
```

AUGRU 的改进：
```
u'_t = a_t * u_t    # 用注意力分数缩放更新门
h_t = (1 - u'_t) * h_{t-1} + u'_t * h_tilde_t
```

效果：
- a_t 高（与候选物品相关）→ 正常更新，该行为影响演化轨迹
- a_t 低（与候选物品无关）→ u'_t 接近 0，隐状态几乎不更新，相当于「跳过」该行为
- 实现了候选物品感知的兴趣演化——只追踪与当前候选相关的兴趣演化路径

### 3.4 辅助损失 (Auxiliary Loss)

```
L_aux = -sum_t [log sigmoid(h_t^T * e_{t+1}) + log sigmoid(-h_t^T * e_neg)]
```

- 用下一个真实点击 e_{t+1} 作为正样本
- 随机采样未点击物品作为负样本
- 目的：加强兴趣提取层的监督信号，确保 h_t 确实捕捉了当前兴趣

### 3.5 DIEN 的三种演化方案对比

DIEN 论文尝试了三种方案：
```
AIGRU: 直接用注意力分数缩放输入 x'_t = a_t * h_t → 效果一般
AGRU:  用注意力替换 GRU 的更新门 u_t = a_t → 过于粗暴
AUGRU: 用注意力缩放更新门 u'_t = a_t * u_t → 最优
```

AUGRU 最好，因为保留了 GRU 自身的门控机制，只是在此基础上叠加注意力调制。

## 4. DSIN (Deep Session Interest Network)

### 4.1 核心思想

将用户行为按 session 切分，session 内用 Transformer 编码，session 间用 BiLSTM 建模演化。

### 4.2 Session 切分

- 时间间隔超过阈值（如 30 分钟）→ 新 session
- 或基于行为类型变化（如从浏览切到搜索）
- session 反映了用户某一次集中的交互意图

### 4.3 模型结构

```
用户行为序列 → 按 session 切分 → [S_1, S_2, ..., S_m]

Session 内编码:
  S_k = [e_1, ..., e_j] → Multi-Head Self-Attention → s_k (session 表示)

Session 间演化:
  [s_1, s_2, ..., s_m] → BiLSTM → [h_1, h_2, ..., h_m]

Session 兴趣激活:
  与候选物品 e_a 做注意力，得到最终用户表示
```

### 4.4 适用场景

- 行为稀疏、session 边界明确（如旅游、房产）
- session 内行为高度相关（同一购物意图）
- session 间有明显的兴趣切换

不适用：行为密集、无明显 session 边界（如短视频连续滑动）

## 5. SIM (Search-based Interest Model)

### 5.1 核心问题

用户历史行为可能有上万条，全部做注意力计算不现实（O(n) 且 n 极大）。

### 5.2 两阶段检索

```
第一阶段 - GSU (General Search Unit):
  用类目、品牌等硬匹配规则快速筛选
  从上万条行为中筛出与候选物品相关的几百条
  复杂度: O(1) 的哈希查找

  hard-search: item_category == candidate_category
  soft-search: cos(e_i, e_a) > threshold

第二阶段 - ESU (Exact Search Unit):
  对筛选后的子集做精确的注意力计算
  标准的 DIN-style 注意力
  复杂度: O(K)，K << N
```

### 5.3 时间信息编码

SIM 特别强调时间信息：
```
input_t = item_emb + time_emb + position_emb
```

time_emb 编码行为发生的时间距今的间隔，帮助模型区分近期行为和远期行为。

### 5.4 工业落地

- 阿里妈妈提出，在淘宝定向广告中验证
- GSU 阶段可以预计算并缓存（离线构建用户行为索引）
- ESU 阶段在线实时计算
- 支持百万级用户行为历史的实时推荐

### 5.5 SIM 的两种检索模式

Hard-search SIM:
- 精确匹配：类目 ID 一致
- 速度最快，但可能遗漏跨类目的相关行为

Soft-search SIM:
- 向量相似度匹配：用预训练 embedding 的 cosine 相似度
- 更灵活，但需要在线计算相似度（或离线预计算 top-K）

## 6. 兴趣建模对比总结

```
模型    核心机制            解决的问题              复杂度
DIN     候选感知注意力       用户兴趣多样性          O(n)
DIEN    双层GRU+AUGRU       兴趣动态演化            O(n)
DSIN    Session分层建模      会话级兴趣切换          O(m*l)
SIM     两阶段检索          超长序列计算效率         O(K)
```

## 7. 实践中的设计选择

### 7.1 何时用哪个模型？

- 序列长度 < 50，行为密集 → DIN 够用
- 需要建模兴趣随时间变化 → DIEN
- session 边界清晰，行为稀疏 → DSIN
- 序列长度 > 1000 → SIM

### 7.2 工业简化

很多公司的实际做法：
- 基础版：DIN + 最近 50 个行为 → 已经比 pooling 好很多
- 进阶版：多窗口 DIN（最近 10 + 最近 50 + 最近 200 分别做注意力，拼接）
- 高阶版：SIM 的检索思路 + DIN/DIEN 的精排

### 7.3 与 Transformer 方案的结合

- DIN/DIEN 的注意力是 target-aware（候选物品参与注意力计算）
- SASRec 的注意力是 self-attention（序列内部交互）
- 可以结合：SASRec 做序列编码 → DIN-style target attention 做候选物品匹配

## 8. 面试高频问题

Q: DIN 中注意力为什么不做 softmax 归一化？
A: 保留用户兴趣强度信息。Softmax 归一化后权重和为 1，即使候选物品与所有历史行为都不相关，也被迫分配注意力。不归一化则权重可以全部接近 0，反映「用户对此不感兴趣」，更符合实际。

Q: DIEN 的 AUGRU 是怎么工作的？
A: 用注意力得分 a_t 调制 GRU 的更新门 u'_t = a_t * u_t。与候选物品无关的行为 a_t 低，更新门接近 0，隐状态几乎不更新，相当于「跳过」。保留了 GRU 自身的门控能力，只叠加候选感知的过滤。

Q: DIEN 的辅助损失有什么作用？
A: 加强兴趣提取层 GRU 的监督信号。只靠最终 loss 反传，GRU 底层很难得到有效梯度。辅助损失用 h_t 预测 e_{t+1}，迫使每个时刻的隐状态都必须捕捉当前兴趣。

Q: SIM 的两阶段检索为什么有效？
A: 用户上万条历史行为中，与当前候选物品相关的可能只有几十条。GSU 用 O(1) 的规则快速筛选出这个子集，ESU 在小子集上做精确注意力。复杂度从 O(N) 降到 O(K)，K << N，同时保留最相关行为的信息。

Q: DIN 和 Transformer 序列推荐的注意力有什么区别？
A: DIN 是 target-aware 注意力（候选物品作为 Query 对历史行为做加权），关注「历史行为与候选的相关性」。Transformer 是 self-attention（序列内部行为互相交互），关注「行为之间的关系和依赖」。两者可以结合使用。

Q: DSIN 什么时候比 DIN/DIEN 好？
A: 当用户行为有明显的 session 结构时——session 内高度相关（同一购物意图），session 间有明显切换。旅游、房产、低频电商等场景。如果行为密集且无明显 session 边界（如短视频），DSIN 的优势不明显。
