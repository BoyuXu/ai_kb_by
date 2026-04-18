# Transformer 序列推荐：SASRec / BERT4Rec / 位置编码 / 长序列

## 1. 自注意力机制在序列推荐中的作用

### 1.1 基本计算流程

Self-Attention 的核心步骤：
```
1. 线性变换生成 Q, K, V:
   Q = X * W_Q,  K = X * W_K,  V = X * W_V

2. 计算注意力分数:
   score = Q * K^T / sqrt(d_k)

3. Softmax 归一化:
   alpha = Softmax(score)

4. 加权求和:
   output = alpha * V
```

在序列推荐中的意义：
- 每个交互项通过自注意力与序列中所有其他项直接交互
- 不依赖固定距离，捕捉任意两项间的长程依赖
- 并行计算所有位置，训练效率远高于 RNN
- 多层堆叠可学习从具体行为到抽象偏好的层次化表示

### 1.2 与 RNN 的本质区别

RNN (GRU/LSTM)：
- 顺序处理，当前状态 = f(上一状态, 当前输入)
- 长距离信息在逐步传播中衰减（梯度消失）
- 不可并行训练
- 天然包含顺序信息

Self-Attention：
- 并行处理所有位置
- 任意两个位置间路径长度为 1（直接交互）
- O(n^2) 复杂度但可并行，实际更快
- 需要额外位置编码注入顺序信息

### 1.3 多头注意力 (Multi-Head Attention)

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O
head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
```

多头的价值：不同头可以关注不同类型的依赖关系：
- 某些头关注近期行为（短期兴趣）
- 某些头关注远期相似行为（长期偏好）
- 某些头关注同类目行为（主题聚焦）

## 2. SASRec (Self-Attentive Sequential Recommendation)

### 2.1 模型结构

单向 Transformer 解码器，使用因果掩码（causal mask）。

```
输入：物品 embedding 序列 [e_1, e_2, ..., e_n] + 位置编码

Transformer Block × L:
  - Masked Multi-Head Self-Attention  (因果掩码：位置 t 只能看到 1..t)
  - Layer Norm
  - Point-wise Feed-Forward Network
  - Layer Norm + Residual Connection

输出：每个位置 t 的表示 h_t 用于预测下一项 item_{t+1}
```

### 2.2 训练目标

自回归 Next Item Prediction：
```
L = -sum_t log P(item_{t+1} | item_1, ..., item_t)
P(item_{t+1}) = softmax(h_t * E^T)  # E 是物品 embedding 矩阵
```

实际中用负采样加速（不对全部物品做 softmax）。

### 2.3 关键设计

因果掩码：
- 下三角矩阵，确保位置 t 只能 attend 到位置 1..t
- 物理意义：预测未来物品不能看到未来信息
- 训练时所有位置可以同时计算（teacher forcing）

位置编码：
- 使用可学习的位置嵌入（Learnable Position Embedding）
- 直接加到物品 embedding 上：input = item_emb + pos_emb

Dropout 策略：
- Embedding dropout + Attention dropout + FFN dropout
- 推荐系统中序列通常较短，dropout 需要适度（0.1-0.3）

### 2.4 在线推理优势

增量推理：用户新增一个行为时，只需计算新位置与历史的注意力，历史位置的 KV cache 可以复用。不需要重新编码整个序列。

```
已有 cache: K_1..t, V_1..t
新行为 item_{t+1}:
  Q_{t+1} = embed(item_{t+1}) * W_Q
  只需计算 Attention(Q_{t+1}, [K_1..t, K_{t+1}], [V_1..t, V_{t+1}])
```

## 3. BERT4Rec

### 3.1 模型结构

双向 Transformer 编码器，无因果掩码。

```
输入：物品 embedding 序列，部分位置被替换为 [MASK] token

Transformer Block × L:
  - Multi-Head Self-Attention  (双向，所有位置互相可见)
  - Layer Norm
  - Feed-Forward Network
  - Layer Norm + Residual Connection

输出：被 mask 位置的表示用于预测原始物品
```

### 3.2 训练目标

Masked Item Prediction（类似 BERT 的 MLM）：
```
随机 mask 序列中 15% 的物品位置
L = -sum_{masked positions} log P(item_t | context)
```

Mask 策略（沿用 BERT 80/10/10）：
- 80% 替换为 [MASK] token
- 10% 替换为随机物品
- 10% 保持不变

### 3.3 双向注意力的优势

- 每个位置可以看到前后所有上下文
- 更全面地捕捉用户行为模式
- 例如：用户买了手机壳和充电器，中间的手机即使被 mask 也能从两侧上下文推断

### 3.4 BERT4Rec 的局限

推理效率低：
- 每次推理必须对整个序列重新编码（双向依赖）
- 无法像 SASRec 那样用 KV cache 做增量推理
- 在线服务的延迟要求下不太实际

训练与推理不一致：
- 训练时有 [MASK] token，推理时没有
- 推理时通常在序列末尾加 [MASK] 预测下一项
- 这种 gap 可能影响效果

## 4. SASRec vs BERT4Rec 全面对比

```
维度          | SASRec              | BERT4Rec
结构          | 单向解码器           | 双向编码器
掩码          | 因果掩码(下三角)     | 随机掩码(15%)
训练目标      | 自回归预测下一项      | 掩码物品预测
上下文        | 只看过去              | 看前后所有
推理效率      | 高(KV cache增量推理)  | 低(需重编码)
训练效率      | 每个位置都有监督信号   | 只有 mask 位置有信号
短序列效果    | 好                    | 可能更好(双向)
长序列效果    | 较好                  | 计算开销大
工业落地      | 主流选择              | 较少直接使用
适合场景      | 在线实时推荐          | 离线特征挖掘/预训练
```

工业选择建议：
- 在线服务对延迟敏感 → SASRec
- 离线预训练用户表示 → BERT4Rec
- 折中方案：用 BERT4Rec 预训练，SASRec 微调和在线服务

## 5. 位置编码方法

### 5.1 可学习位置嵌入 (Learnable PE)

```
pos_emb = Embedding(max_len, d_model)
input_t = item_emb_t + pos_emb[t]
```

特点：
- 最简单有效，SASRec/BERT4Rec 默认使用
- 只能处理训练时见过的最大长度
- 不同位置的嵌入完全独立学习，缺乏位置间的泛化

### 5.2 正弦位置编码 (Sinusoidal PE)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

特点：
- 固定函数，无需学习参数
- 理论上可泛化到任意长度
- 推荐系统中效果通常不如可学习 PE

### 5.3 相对位置编码 (Relative PE)

```
Attention(Q, K) = Q * K^T + Q * R  # R 是相对位置嵌入
```

关注行为间的距离而非绝对位置。例如「上一个点击」比「第 5 个位置」更有意义。

优势：
- 更好的泛化性（不依赖绝对位置）
- 适合变长序列
- Transformer-XL 和 ALiBi 都属于此类

### 5.4 时间感知位置编码 (Time-aware PE)

```
time_emb = Embedding(time_bucket(delta_t))
或: time_emb = W * log(1 + delta_t)   # 对数时间间隔
input_t = item_emb_t + pos_emb[t] + time_emb_t
```

融入实际时间间隔信息（而非仅位置顺序），捕捉时间维度模式。

适用场景：
- 新闻推荐（时间衰减明显）
- 电商（季节性、促销周期）
- 行为间时间间隔差异大的场景

### 5.5 选择建议

- 电商会话推荐：可学习 PE（序列短，位置固定）
- 新闻/信息流：时间感知 PE（时效性强）
- 长序列（上千行为）：相对 PE（泛化性好）
- 通用 baseline：先用可学习 PE，不够再换

## 6. 长序列处理策略

标准自注意力 O(n^2) 在长序列上不可行。以下是工业级解决方案：

### 6.1 序列截断 / 动态截断

最简单有效：只保留最近 K 个行为。

动态截断改进：
- 保留最近 K1 个行为 + 历史中与候选物品最相关的 K2 个行为
- SIM 模型的思路：先检索再精排

### 6.2 滑动窗口注意力

```
位置 t 只 attend 到 [t-w, t] 的局部窗口
复杂度: O(w * n)，w << n
```

代表：Longformer 的 sliding window attention
缺陷：无法捕捉超出窗口的长距离依赖

### 6.3 LSH 注意力 (Locality-Sensitive Hashing)

```
用 LSH 将 Q/K 分桶
只在同一桶内计算注意力
复杂度: O(n * log(n))
```

代表：Reformer
适用：超长序列（数千以上），但实现复杂

### 6.4 分层 Transformer

```
原始序列 → 分段（每段 L 个行为）
段内: 用标准 Transformer 编码 → 段表示
段间: 用另一个 Transformer 建模段之间的关系
```

优势：两层 Transformer 都是标准结构，实现简单
代表：DSIN 的思路（段 = session）

### 6.5 循环 Transformer (Transformer-XL)

段间传递隐状态，类似 RNN 的思路：
```
段 1: 标准 Transformer → 保存 KV cache
段 2: 注意力范围扩展到段 1 的 KV cache
```

可以在固定计算开销下扩大有效上下文窗口。

### 6.6 线性注意力 (Linear Attention)

```
标准: Softmax(QK^T) * V    → O(n^2)
线性: phi(Q) * (phi(K)^T * V)  → O(n * d)
```

用核函数近似 Softmax，将复杂度降到线性。效果通常不如标准注意力。

## 7. 工业级优化实践

### 7.1 推理加速

- KV Cache：SASRec 的因果结构天然支持
- 模型蒸馏：大 Transformer 蒸馏到小模型
- 量化：FP16/INT8 量化减少计算量
- 提前退出：简单样本用浅层结果，复杂样本用深层

### 7.2 噪声过滤

- 时间衰减注意力：attention_weight *= decay(delta_t)
- 行为类型加权：购买 > 收藏 > 点赞 > 浏览 > 曝光
- 去噪自编码器预训练：随机 mask + 重建，让模型学会忽略噪声行为

### 7.3 冷启动

用户冷启动：
- 新用户无序列 → 用全局平均 embedding 或人口统计特征初始化
- 少量行为时序列极短 → 融合 side information（类目、品牌等）

物品冷启动：
- 新物品无 embedding → 用内容特征（文本、图片）生成 embedding
- 对比学习：拉近新物品与其类目中心的距离

## 8. 面试高频问题

Q: SASRec 为什么用因果掩码而不是双向注意力？
A: 因为序列推荐的目标是预测下一项（next item），天然是自回归任务。因果掩码确保预测时只使用历史信息，符合在线推理的约束。双向注意力在推理时无法使用未来信息。

Q: BERT4Rec 的 mask 比例为什么是 15%？
A: 沿用 NLP 中 BERT 的经验值。过低（如 5%）训练信号太稀疏，收敛慢；过高（如 50%）上下文信息太少，预测难度过大。15% 是经验上的平衡点，但在推荐场景中可以微调（序列短时可适当提高）。

Q: 位置编码为什么重要？Transformer 不加位置编码会怎样？
A: Self-Attention 是置换不变的（permutation invariant），不加位置编码则模型无法区分「先看A后看B」和「先看B后看A」。行为序列的顺序包含关键的兴趣演化信息，必须通过位置编码注入。

Q: 如何处理超长用户行为序列（上万条）？
A: 工业主流方案是 SIM 的两阶段检索——GSU 先用类目等硬规则快速筛选出相关行为子集（几百条），ESU 再对子集做精确注意力计算。其他方案包括分层 Transformer、Transformer-XL、线性注意力，但 SIM 在工业中验证最充分。

Q: 自注意力机制如何捕捉用户兴趣的动态变化？
A: 自注意力让每个行为与所有其他行为直接交互。当用户连续点击某类物品时，这些行为互相赋予高注意力权重，在表示中强化该兴趣。它不依赖固定距离，能灵活识别长期稳定偏好和短期突发兴趣。多层堆叠可学习从底层具体行为到高层抽象概念的层次化演变。
