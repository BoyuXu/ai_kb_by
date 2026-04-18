# Ch8 用户序列行为建模

## 子主题索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [interest-evolution.md](interest-evolution.md) | DIN/DIEN/DSIN/SIM 兴趣演化建模、注意力设计、AUGRU、两阶段检索 | ~250 |
| [transformer-rec.md](transformer-rec.md) | SASRec/BERT4Rec、位置编码方法、长序列处理策略、工业优化 | ~260 |
| [session-rec.md](session-rec.md) | Session 推荐、长短期兴趣分离、多模态序列建模、数据预处理 | ~250 |
| [long-seq-transformer-evolution.md](long-seq-transformer-evolution.md) | 长序列 Transformer 演进全景：SASRec→SIM→TWIN→HSTU→LONGER→OneRec，12 模型对比 | ~350 |

## 1. 核心概念

用户行为序列建模的核心目标：从用户历史交互序列中捕捉动态兴趣演化，预测下一个可能感兴趣的物品。

关键挑战：
- 序列中包含噪声（误点击、临时兴趣）
- 长短期兴趣需要分离建模
- 超长序列的计算效率问题
- 多模态行为信息的融合

## 2. 经典序列模型演进

### 2.1 RNN/LSTM/GRU 系列

GRU4Rec 是最早将 RNN 应用于会话推荐的模型：
- 将用户点击序列视为 session，用 GRU 编码
- 使用 session-parallel mini-batch 训练
- 引入 BPR loss 和 TOP1 loss 优化排序
- 局限：难以捕捉长距离依赖，训练速度慢
- 详见 [session-rec.md](session-rec.md)

LSTM vs GRU 对比：
- LSTM：三个门（输入/遗忘/输出），参数更多，表达力更强
- GRU：两个门（重置/更新），参数少，训练快，效果相当
- 推荐系统中 GRU 更常用（序列较短，GRU 足够）

### 2.2 注意力机制系列

DIN（Deep Interest Network）：
- 核心思想：用户兴趣是多样的，不同候选物品应激活不同的历史行为
- 用候选物品作为 Query，对历史行为做加权求和（Local Activation Unit）
- 关键设计：注意力权重不做 softmax 归一化（保留兴趣强度信息）
- 详见 [interest-evolution.md](interest-evolution.md)

DIEN（Deep Interest Evolution Network）：
- 两层 GRU 结构：兴趣提取 + 兴趣演化（AUGRU）
- 用注意力得分调制 GRU 更新门，跳过无关行为
- 辅助损失加强底层 GRU 的监督信号
- 详见 [interest-evolution.md](interest-evolution.md)

DSIN（Deep Session Interest Network）：
- 将用户行为按 session 切分
- Session 内用 Transformer 编码，Session 间用 BiLSTM 建模演化
- 适合行为稀疏、session 边界明确的场景
- 详见 [interest-evolution.md](interest-evolution.md)

SIM（Search-based Interest Model）：
- 针对超长序列（上千甚至上万行为）的工业方案
- GSU 快速筛选 + ESU 精确注意力的两阶段检索
- 详见 [interest-evolution.md](interest-evolution.md)

### 2.3 Transformer 系列

SASRec（Self-Attentive Sequential Recommendation）：
- 单向 Transformer 解码器结构（因果掩码）
- 训练目标：自回归预测下一项
- 支持 KV cache 增量推理，工业主流
- 详见 [transformer-rec.md](transformer-rec.md)

BERT4Rec：
- 双向 Transformer 编码器结构
- 训练目标：随机掩码物品预测（Masked Item Prediction）
- 上下文信息更全面，但推理效率低
- 详见 [transformer-rec.md](transformer-rec.md)

## 3. 关键技术点

### 3.1 位置编码方法

四种常见方案：
1. 可学习位置嵌入（Learnable PE）：简单有效，适合固定长度序列
2. 相对位置编码（Relative PE）：关注行为间距离而非绝对位置，泛化性更好
3. 时间感知位置编码（Time-aware PE）：融入时间间隔信息，捕捉时间维度模式
4. 正弦位置编码（Sinusoidal PE）：固定函数，可泛化到任意长度

选择建议：电商会话序列用可学习 PE；新闻推荐（时间稀疏）用时间感知 PE
详见 [transformer-rec.md](transformer-rec.md)

### 3.2 长序列处理策略

- 序列截断/动态截断：保留最近和最相关的行为
- 滑动窗口注意力：限制注意力在局部窗口内
- LSH 注意力：利用哈希分桶近似全局注意力
- 分层 Transformer：先对片段编码，再对片段级表征做全局建模
- 循环 Transformer (Transformer-XL)：片段间传递隐状态
- 线性注意力：用核函数近似 Softmax
- 详见 [transformer-rec.md](transformer-rec.md)

### 3.3 噪声处理

数据预处理层面：
- 基于规则过滤（极短停留、重复点击）
- 引入交互权重（浏览 < 点赞 < 收藏 < 购买）
- 无监督异常检测

模型结构层面：
- 注意力机制本身可以降权不重要行为
- 噪声感知注意力（辅助网络预测噪声概率并降权）
- 对比学习去噪

详见 [session-rec.md](session-rec.md)

## 4. 自注意力机制详解

详见 [transformer-rec.md](transformer-rec.md)，包括：
- Self-Attention 完整计算流程
- 与 RNN 的本质区别（并行 vs 顺序、路径长度）
- 多头注意力的价值
- 工业级优化（KV cache、量化、提前退出）

## 5. 多模态序列建模

详见 [session-rec.md](session-rec.md)，包括：
- 行为序列 + 文本信息的交叉注意力融合
- ID Embedding vs 内容 Embedding 的权衡
- 行为类型融合（behavior_type_embedding）
- 多粒度兴趣建模（长/短/瞬时）

## 6. 面试高频问题

Q: DIN 中注意力为什么不做 softmax 归一化？
A: 保留用户兴趣强度信息。softmax 后权重和为 1，不相关时也被迫分配注意力；不归一化则权重可接近 0。

Q: DIEN 的 AUGRU 是怎么工作的？
A: 用注意力得分 a_t 调制更新门 u'_t = a_t * u_t。无关行为的 a_t 低，更新门接近 0，相当于跳过。

Q: 为什么 SASRec 比 BERT4Rec 更适合线上服务？
A: SASRec 的单向结构支持 KV cache 增量推理，新增行为只需计算新位置的注意力。BERT4Rec 双向结构每次需要重新编码整个序列。

Q: SIM 模型的两阶段检索为什么有效？
A: 上万条历史中相关行为只有几十条。GSU 用 O(1) 硬规则筛选子集，ESU 在子集上做精确注意力，复杂度从 O(N) 降到 O(K)。

Q: 如何处理用户的多层次兴趣？
A: 瞬时（最近几次点击）、短期（当前 session）、长期（全部历史），分别建模后门控/注意力融合。

Q: 自注意力机制如何捕捉用户兴趣的动态变化？
A: 每个行为与所有其他行为直接交互，连续同类点击互相强化。不依赖固定距离，灵活识别长期偏好和短期突发兴趣。多层堆叠学习从具体行为到抽象概念的层次化演变。
