# 注意力排序模型：DIN 到序列建模全链路

## 1. DIN（Deep Interest Network）

### 核心思想：Target Attention

```
传统做法：对用户历史行为做 sum/mean pooling → 固定用户向量
  问题：用户兴趣是多样的，固定向量丢失了与当前候选物品相关的兴趣

DIN 改进：根据候选物品动态提取用户兴趣
  attention(i) = f(e_i, e_target)    // 行为 i 与候选物品的相关性
  user_interest = sum( attention(i) * e_i )

  f 通常是一个小型 MLP：
    f(e_i, e_target) = MLP([e_i, e_target, e_i - e_target, e_i * e_target])

效果：
  候选是"篮球鞋" → 历史中"运动鞋"行为权重高，"零食"权重低
  候选是"薯片"   → 历史中"零食"行为权重高，"运动鞋"权重低
```

### DIN 架构详解

```
输入层：
  用户画像特征（age, gender, city）
  用户行为序列（clicked_item_1, ..., clicked_item_N）
  候选物品特征（item_id, category, brand）
  上下文特征（time, position）

              ┌────────────────┐
              │   Output MLP   │
              └───────┬────────┘
                      │
         Concat(user_profile, user_interest, candidate, context)
                      │
              ┌───────┴────────┐
              │ Target         │
              │ Attention      │
              │ Pooling        │
              └───────┬────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
  e_1 × a_1      e_2 × a_2      e_N × a_N
    │                 │                 │
  item_1            item_2           item_N
  (用户历史行为序列)
```

### DIN 训练技巧

```
1. Dice 激活函数（替代 PReLU）：
   Dice(x) = p(x) * x + (1 - p(x)) * alpha * x
   其中 p(x) = sigmoid((x - E[x]) / sqrt(Var[x] + eps))
   数据自适应的激活函数，效果优于 PReLU

2. Mini-Batch Aware 正则：
   对 Embedding 只正则化当前 batch 出现的参数
   避免对未出现的低频特征施加无意义的正则

3. 注意力权重不做 softmax 归一化：
   保留原始权重大小信息
   允许所有行为权重都很低（与候选不相关时）
```

---

## 2. DIEN（Deep Interest Evolution Network）

### 核心改进：兴趣演化建模

```
DIN 局限：不建模序列的时序信息（注意力只看相关性，不看顺序）

DIEN 引入两层 GRU：

第一层：Interest Extractor（兴趣提取）
  普通 GRU，从行为序列提取兴趣状态
  h_t = GRU(h_{t-1}, e_t)
  辅助损失：用 h_t 预测下一个行为 e_{t+1}
    → 确保隐状态真正捕获用户兴趣

第二层：Interest Evolution（兴趣演化）
  AUGRU（Attention-based GRU）
  用候选物品的注意力信号调制 GRU 更新门
  u'_t = a_t * u_t    (a_t 是 target attention 权重)
  → 只演化与候选物品相关的兴趣
```

### AUGRU 公式

```
标准 GRU：
  u_t = sigmoid(W_u * [h_{t-1}, e_t])       // 更新门
  r_t = sigmoid(W_r * [h_{t-1}, e_t])       // 重置门
  h_t' = tanh(W * [r_t ⊙ h_{t-1}, e_t])    // 候选隐状态
  h_t = (1 - u_t) ⊙ h_{t-1} + u_t ⊙ h_t'  // 输出

AUGRU 改动：
  a_t = attention(h_t, e_target)             // target attention
  u'_t = a_t * u_t                           // 注意力调制更新门
  h_t = (1 - u'_t) ⊙ h_{t-1} + u'_t ⊙ h_t' // 用 u'_t 替代 u_t

效果：与候选不相关的行为 → a_t 小 → u'_t 小 → 隐状态几乎不更新
     与候选相关的行为 → a_t 大 → 正常更新兴趣状态
```

---

## 3. 长序列处理

### 问题

```
用户历史可能有数千条行为
全序列做 attention / GRU：
  - Attention: O(n) 计算 + 存储
  - GRU: O(n) 串行计算，延迟线性增长
  - 在线推理延迟不可接受（要求 < 50ms）
```

### 解决方案矩阵

```
方案              原理                    复杂度    信息损失   适用场景
─────────────────────────────────────────────────────────────────────
序列截断          只用最近 N 条            O(N)     高(丢远期)  简单baseline
时间衰减采样      按时间权重采样子集        O(K)     中         中等序列
类目采样          按与候选同类目采样        O(K)     中         电商
局部注意力        只对相关子序列计算        O(K)     低         精度要求高
线性注意力        O(n) 近似 softmax att    O(n)     低         长序列
GRU 压缩          RNN 压缩为固定向量       O(n)     中         在线延迟敏感
知识蒸馏          Teacher 全序列           O(N_s)   低         工程成本可接受
               Student 短序列

实际工业方案（如 SIM）：
  1. 检索阶段：用候选物品从长序列中检索 top-K 相关行为
     → 倒排索引 / 类目匹配 / Embedding 近邻
  2. 建模阶段：对 top-K 子序列做精确 attention
  延迟可控 + 信息损失小
```

---

## 4. Transformer vs GRU 序列建模

### 对比

```
维度            Transformer              GRU
──────────────────────────────────────────────────
并行性          完全并行（矩阵运算）      串行（时间步依赖）
长距离依赖      Self-Attention 直接建模   门控衰减，远距离弱
参数量          较大（Q/K/V 矩阵）        较小
训练速度        快（并行）                慢（序列依赖）
推理延迟        一次前向（并行）           逐步解码（串行）
位置信息        需要位置编码              隐式包含时序
内存占用        O(n^2)（注意力矩阵）      O(n)（逐步）
适用场景        长序列 / 离线训练          短序列 / 在线延迟敏感
```

### 推荐场景选择

```
选 Transformer：
  - 离线训练，序列长度中等（< 200）
  - 需要捕获长距离依赖
  - 多头注意力可捕获多种兴趣模式
  - BST（Behavior Sequence Transformer）是典型应用

选 GRU/LSTM：
  - 在线推理延迟敏感（可增量更新隐状态）
  - 序列很长但只需最终状态
  - 模型参数量有限制
  - DIEN 使用 GRU 的原因：兴趣演化需要时序递推

混合方案：
  - 离线用 Transformer 训练 → 知识蒸馏到 GRU Student
  - 或用线性注意力 Transformer 降低复杂度
```

---

## 5. 多模态行为序列

### 融合策略

```
场景：用户行为序列中包含文本（标题）、图像（封面）、ID 特征

1. 特征拼接（Early Fusion）：
   e_item = Concat(e_id, e_text, e_image)
   → 简单直接，但不同模态尺度/分布不同

2. 跨模态注意力（Cross-Modal Attention）：
   Q = 文本特征, K = V = 图像特征
   → 让文本关注图像中的相关信息
   适合模态间有对应关系的场景

3. 对比对齐（Contrastive Alignment）：
   用 InfoNCE 损失对齐不同模态表示
   L = -log(exp(sim(v_text, v_image) / tau) / sum(exp(sim(v_text, v_neg) / tau)))
   → 对齐后的表示可直接拼接或加权

4. 门控融合（Gated Fusion）：
   gate = sigmoid(W * [e_text, e_image])
   e_fused = gate * e_text + (1 - gate) * e_image
   → 自适应选择模态贡献
```

### 文本/图像特征处理

```
文本特征：
  - 预训练模型提取：BERT / sentence-transformers
  - 工业方案：离线提取 → 存入特征服务 → 在线查表
  - 降维：768d → 64d / 128d（线性投影或 PCA）
  - 注意：文本特征更新频率低，适合缓存

图像特征：
  - 预训练 CNN/ViT 提取（ResNet-50 / CLIP）
  - 2048d → 128d 降维
  - 工业中通常离线提取存好，不在线计算
  - CLIP 的多模态对齐特征效果更好
```

---

## 6. 行为序列作为特征的设计

### 序列特征构造

```
直接序列：
  - 最近 N 个点击 item_id → Embedding → Attention Pooling
  - 最近 N 个点击 category → 同上
  - 分时段序列：近 1 小时 / 近 1 天 / 近 1 周

统计特征（从序列衍生）：
  - 类目分布：各类目点击占比
  - 时间模式：活跃时段分布
  - 行为强度：点击/收藏/购买次数
  - 序列多样性：类目 entropy

交叉特征：
  - 候选 item 的 category 在用户历史中的点击次数
  - 候选 item 的 brand 在用户购买历史中的占比
  - 这些特征在 DIN 之前的时代非常重要
```

### 架构设计考虑

```
Q: 排序模型的输入应该包含哪些部分？
A: 四大块特征：

  1. 用户画像：demographics + 长期偏好统计
  2. 物品属性：ID / category / brand / 统计（曝光数、CTR）
  3. 用户行为序列：近期行为 → Attention/GRU 建模
  4. 上下文：时间、设备、位置、请求来源

  拼接方式：
  final = Concat(user_profile, item_features,
                 sequence_interest, context)
  → MLP → sigmoid → CTR score
```

---

## 7. 面试高频问题

```
Q1: DIN 的 attention 和 Transformer 的 self-attention 有什么区别？
A: DIN 是 target attention（候选物品 vs 行为序列），单向查询
   Transformer 是 self-attention（序列内部互相关注）
   DIN 不建模行为间的关系，Transformer 建模序列内部依赖

Q2: DIEN 为什么用 AUGRU 而不是直接在 GRU 后面加 attention？
A: 直接加 attention = 先演化再筛选（DIN + GRU 的简单拼接）
   AUGRU = 演化过程中就筛选（注意力信号深度融入状态更新）
   后者能让兴趣演化路径本身就聚焦于与候选相关的方向

Q3: 在线推理时，用户行为序列实时更新怎么处理？
A: 1. GRU 方案：增量更新隐状态（新行为 → 一步 GRU → 新 h_t）
   2. Attention 方案：新行为追加到序列缓存，重新计算
   3. 工程优化：用户侧特征异步更新 + 缓存

Q4: 多模态特征会增加在线延迟吗？
A: 不会，如果做对了：
   - 文本/图像特征离线提取，存入特征服务
   - 在线只做查表 + 拼接，不做模型推理
   - 额外延迟 < 1ms（纯内存查询）

Q5: SIM 的两阶段检索怎么做？
A: Stage 1（Search）：用候选 item 的 category 做硬检索
   或用 Embedding 近邻做软检索，得到 top-K 相关行为
   Stage 2（Modeling）：对 top-K 子序列做精确 target attention
   解决了长序列（万级）的在线延迟问题
```
