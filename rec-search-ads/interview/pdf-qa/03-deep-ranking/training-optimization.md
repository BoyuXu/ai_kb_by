# 排序模型训练优化：从样本到特征到损失函数

## 1. 样本不均衡处理

### 问题描述

```
推荐系统中正负样本比例极度不均衡：
  CTR 场景：正样本 1-5%，负样本 95-99%
  CVR 场景：正样本 0.1-1%，更极端

不处理的后果：
  - 模型偏向预测负类（全预测 0 也有 95% 准确率）
  - 正样本梯度被海量负样本梯度淹没
  - 预测概率整体偏低
```

### 解决方案

```
1. 负采样（Negative Sampling）：
   从全量负样本中按比例随机采样
   采样率 r = 采样负样本数 / 全量负样本数

   关键：概率校准（不校准 → 线上预测偏高）
   p_true = p_pred / (p_pred + (1 - p_pred) / r)

   例：采样率 r=0.1，模型预测 p_pred=0.5
   → p_true = 0.5 / (0.5 + 0.5/0.1) = 0.5 / 5.5 ≈ 0.09

2. Focal Loss：
   FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

   gamma > 0：降低易分类样本（高置信度）的权重
   gamma = 0：退化为标准交叉熵
   gamma = 2, alpha = 0.25 是常用配置

   效果：自动聚焦于难分类样本
   优势：不需要手动调采样率

3. 类别权重（Class Weight）：
   正样本 loss × w_pos（w_pos = neg_count / pos_count）
   简单但粗糙，不区分难易样本

4. SMOTE / 过采样：
   对正样本做插值生成新样本
   推荐系统中较少用（行为数据难以插值）
```

---

## 2. Pointwise / Pairwise / Listwise

### 三种学习范式对比

```
Pointwise：
  输入：单个 (query, item) 对
  标签：点击/未点击（0/1）或评分
  损失：交叉熵 / MSE
  L = -[y * log(p) + (1-y) * log(1-p)]
  优势：简单，可直接预测 CTR
  劣势：不考虑文档间相对关系

Pairwise：
  输入：(query, item_pos, item_neg) 三元组
  目标：正样本得分 > 负样本得分
  损失：BPR Loss / Hinge Loss
  L = -log(sigmoid(s_pos - s_neg))         // BPR
  L = max(0, margin - (s_pos - s_neg))     // Hinge
  优势：直接优化排序关系
  劣势：样本对组合爆炸，训练慢

Listwise：
  输入：query 下的完整排序列表
  目标：优化整个列表的排序质量
  损失：Softmax Cross-Entropy / LambdaRank / ApproxNDCG
  L = -sum_i y_i * log(exp(s_i) / sum_j exp(s_j))
  优势：直接优化排序指标（NDCG）
  劣势：需要完整列表，计算量大
```

### 适用场景选择

```
场景              推荐范式        原因
────────────────────────────────────────────────────
CTR 预估          Pointwise      需要绝对概率值（竞价/融分）
搜索排序          Pairwise/List  关注相对排序质量
推荐精排          Pointwise      输出需要是校准概率
重排序            Listwise       需要考虑列表多样性
召回模型          Pairwise       对比学习天然匹配
```

---

## 3. 过拟合防御

### 正则化方法

```
L1 正则（Lasso）：
  L_total = L_data + lambda * sum(|w_i|)
  效果：稀疏化参数 → 特征选择
  适用：特征多但有效特征少

L2 正则（Ridge / Weight Decay）：
  L_total = L_data + lambda * sum(w_i^2)
  效果：防止参数过大，平滑决策边界
  适用：所有特征都有贡献但权重需约束

Dropout：
  训练时：每个神经元以概率 p 随机失活
  推理时：所有神经元激活，权重乘以 (1-p)
  推荐系统常用 p = 0.1 ~ 0.5
  Embedding 层也可加 Dropout（特征 mask）

Early Stopping：
  监控验证集 AUC/logloss
  连续 N 个 epoch 无提升 → 停止训练
  恢复到验证集最优的模型参数

数据增强：
  - 特征 Mask：随机 mask 部分特征（类似 Dropout 但在输入层）
  - 行为序列截断：随机截断部分历史行为
  - 行为序列重排：打乱部分行为顺序
  - 正样本复制：对正样本重复采样
```

---

## 4. 特征工程

### 数值特征处理

```
归一化：
  Z-Score: x' = (x - mean) / std
    适用：特征近似正态分布
  Min-Max: x' = (x - min) / (max - min)
    适用：特征有明确范围

分桶（Bucketization）：
  等距分桶：[0,10), [10,20), [20,30), ...
  等频分桶：每桶相同数量样本（更常用）
  自定义分桶：根据业务含义（如年龄 [0,18), [18,25), ...）

  分桶后转为类别特征 → Embedding
  优势：非线性能力、鲁棒性（对异常值不敏感）

对数变换：
  x' = log(1 + x)
  适用：长尾分布（如播放次数、价格）
  效果：压缩高值区间，拉伸低值区间

组合使用：
  常见方案：对数变换 → 分桶 → Embedding
  如：item_price → log → 10 等频桶 → 8d Embedding
```

### 类别特征处理

```
One-Hot（低基数）：
  性别 → [1, 0, 0]（男/女/未知）
  适用：基数 < 100

Embedding（高基数）：
  item_id（千万级）→ Embedding 表 → 64d 向量
  维度选择：dim = min(50, ceil(n^0.25))

特征哈希：
  hash(feature) % bucket_size
  优势：固定内存，无需维护词表
  劣势：哈希冲突（不同特征映射到同一桶）
  适用：超高基数 + 在线特征（如用户搜索词）

Target Encoding：
  用该类别下目标变量的均值替代
  如：city_A 的平均 CTR = 0.05 → 编码为 0.05
  注意：严格做 K-fold 避免数据泄露！
```

### 高基数特征优化

```
问题：item_id 有千万级别，Embedding 表占用数 GB 内存

方案：
1. 频率截断：
   出现 < 5 次的 item → 映射到 "default" Embedding
   效果：词表从千万缩到百万级

2. 特征哈希：
   hash(item_id) % 1000000
   内存固定，但有冲突（实测影响 < 0.1% AUC）

3. Field-Aware：
   不同 field 用不同 Embedding
   如 item_id 在"点击场景"和"购买场景"有不同表示

4. 混合维度 Embedding：
   高频 item → 大维度（64d）
   低频 item → 小维度（16d）
   参数量显著降低
```

---

## 5. CTR 涨但 CVR 跌的排查

```
场景：AB 实验中，新模型 CTR 提升但 CVR 下降

排查思路（6.4 节重点）：

1. 检查点击质量：
   - 新模型是否推荐了更多"标题党"内容？
   - 点击后停留时间是否缩短？
   - 分品类看 CTR/CVR 变化

2. 检查用户群分布：
   - 新模型是否偏向了"爱点击但不转化"的用户群？
   - 按用户分层（新/老、高/低活跃）分析

3. 检查样本空间变化：
   - CVR = 转化数 / 点击数
   - CTR 涨 → 点击数涨 → CVR 分母变大
   - 新增的点击可能本身就是低转化意图

4. 检查位置偏差：
   - 排序变化 → 物品展示位置变化 → 点击率变化
   - 前排位置本身 CTR 高但 CVR 不一定高

5. 综合看 GMV / 总转化数：
   - 如果总转化数涨了，CVR 下降可能只是分母效应
   - 最终看业务大盘指标
```

---

## 6. 分布式训练

### 参数服务器架构

```
适用：Embedding 表超大（数 GB ~ TB）

架构：
  Worker 1 ──┐
  Worker 2 ──┤── Parameter Server（存 Embedding）
  Worker 3 ──┘

流程：
  1. Worker 从 PS pull 本 batch 需要的 Embedding
  2. Worker 本地前向 + 反向
  3. Worker 向 PS push 梯度
  4. PS 更新参数

优化：
  - 异步 SGD：Worker 不等待，吞吐高但有梯度延迟
  - 同步 SGD：所有 Worker 对齐，精度好但慢
  - 混合：Dense 参数同步，Sparse（Embedding）异步
```

### 数据并行 vs 模型并行

```
数据并行：
  每个 GPU 持有完整模型，不同数据分片
  适用：模型放得下单卡

模型并行：
  模型拆分到多个 GPU
  适用：模型太大放不下单卡（如超大 Embedding 表）

推荐系统常用混合方案：
  - Embedding 表：模型并行（按 hash 分片到不同 PS）
  - DNN 部分：数据并行（每个 Worker 一份）
```

---

## 7. 面试高频问题

```
Q1: 负采样后为什么需要概率校准？不校准会怎样？
A: 负采样降低了负样本比例，模型学到的概率偏高。
   不校准 → 线上预测概率 > 真实 CTR → 竞价出价偏高 / 融分偏移
   校准公式：p_true = p_pred / (p_pred + (1-p_pred)/r)

Q2: Focal Loss 的 gamma 越大越好吗？
A: 不是。gamma 太大会过度忽略易分类样本，导致训练不稳定。
   gamma=2 是经验最优。gamma>5 通常效果变差。

Q3: L1 和 L2 正则有什么区别？什么时候用哪个？
A: L1 → 稀疏化（产生零权重）→ 特征选择
   L2 → 平滑化（小权重）→ 防过拟合
   工业中 DNN 用 L2 + Dropout 居多
   Wide 部分可用 L1（FTRL 优化器自带 L1）

Q4: 等频分桶 vs 等距分桶？
A: 等频：每桶样本数相同，对长尾分布更友好
   等距：桶边界等间距，高频区过密低频区过疏
   推荐系统数值特征多为长尾 → 首选等频分桶

Q5: Target Encoding 怎么防止数据泄露？
A: 用 K-fold：对第 k 折的样本，用其余 k-1 折计算均值
   还可以加贝叶斯平滑：
   encode = (count * mean + global_count * global_mean) / (count + global_count)
   低频类别 → 回退到全局均值

Q6: 分布式训练时，异步 SGD 的梯度延迟怎么处理？
A: 1. 限制最大延迟步数（超过则 Worker 等待）
   2. 学习率补偿：延迟越大，学习率越小
   3. 对 Dense 参数同步更新，仅 Sparse 异步
   4. 实践中 2-5 步延迟影响不大
```
