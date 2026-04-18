# Wide&Deep 家族：从手动交叉到自动高阶交互

## 1. DNN vs LR：为什么深度学习取代逻辑回归

```
LR 局限：
  - 依赖手动特征交叉，组合爆炸（n 个特征 → C(n,2) 二阶交叉）
  - 线性模型，无法捕捉非线性关系
  - 高维稀疏特征下泛化能力差

DNN 优势：
  - Embedding 层：高维稀疏 → 低维稠密（自动学习语义表示）
  - 多层非线性：自动学习任意阶特征交叉
  - 端到端学习：无需手动特征工程
  - 模型容量：参数量级差几个数量级

关键区别：
  LR：y = sigmoid(w^T * x)           → 线性决策边界
  DNN：y = sigmoid(f(f(f(x))))       → 任意复杂决策边界
```

### 面试角度
```
Q: DNN 完全替代 LR 了吗？
A: 没有。LR 在以下场景仍有优势：
  1. 可解释性要求高（金融风控）
  2. 训练数据少（DNN 容易过拟合）
  3. 线上延迟极敏感（LR 推理 < 1ms）
  4. 特征已经做好充分交叉工程
  实践中常用 LR 做 baseline，DNN 做主模型
```

---

## 2. Wide&Deep

### 核心架构

```
                 ┌──────────┐
                 │  Output  │
                 │ sigmoid  │
                 └────┬─────┘
                      │
              logits_wide + logits_deep
                 ╱              ╲
        ┌───────┐            ┌───────┐
        │ Wide  │            │ Deep  │
        │ (GLM) │            │ (DNN) │
        └───┬───┘            └───┬───┘
            │                    │
    手动交叉特征          Embedding → Concat → Dense
   (如 city x category)     (稀疏特征自动学习)
```

### Wide 和 Deep 各自的职责

```
Wide 部分 → 记忆能力（Memorization）
  - 广义线性模型 y = w^T * [x, cross(x)] + b
  - 记住高频共现模式（如"上海用户常买咖啡"）
  - 需要手动构造交叉特征（领域知识）
  - 典型特征：user_city x item_city, gender x category

Deep 部分 → 泛化能力（Generalization）
  - 稀疏特征 → Embedding → 拼接 → 多层 DNN
  - 发现低频/未见过的特征组合
  - 自动学习特征间的隐式关系
  - 能泛化到训练集中未出现的模式
```

### Joint Training vs Ensemble

```
Joint Training（Wide&Deep 实际做法）：
  - Wide 和 Deep 同时训练，共享 loss
  - 反向传播同时更新两部分参数
  - Wide 部分可以用更小的模型（因为 Deep 补充了泛化）

Ensemble（对比）：
  - 两个模型独立训练
  - 预测时加权融合
  - 每个模型都需要足够大以独立工作

关键区别：Joint Training 让 Wide 和 Deep 互补，
         Ensemble 是两个独立模型的拼凑
```

### 线上线下不一致的排查

```
常见原因（6.2 节重点）：
  1. 特征穿越：训练用了未来信息（如当天 CTR 统计特征）
  2. 特征计算差异：离线 Spark vs 在线 Java 实现不同
  3. 数据分布漂移：训练集时间段 vs 线上实时分布
  4. 负采样校准缺失：离线做了负采样但没校准概率
  5. 模型版本不同步

排查流程：
  Step 1: 固定一批样本，逐个对比离线/在线特征值
  Step 2: 对比离线/在线模型预测分数分布
  Step 3: 检查特征时间戳，排除穿越
  Step 4: 检查负采样率和概率校准公式
  Step 5: AB 实验配置核查
```

---

## 3. DeepFM

### 核心改进：FM 替代手动交叉

```
Wide&Deep 痛点：Wide 部分需要人工设计交叉特征
DeepFM 解法：用 FM 层自动学习所有二阶特征交叉

架构：
  FM 部分（替代 Wide）：
    y_FM = w0 + sum(wi * xi) + sum_i<j <vi, vj> * xi * xj

    其中 <vi, vj> 是第 i 和第 j 个特征的 Embedding 内积
    自动捕获所有二阶交叉，无需手动设计

  Deep 部分：与 Wide&Deep 相同
    Embedding → Concat → Dense layers

关键创新：
  FM 和 Deep 共享同一套 Embedding 参数
  → 参数效率更高
  → FM 的交叉信号可以增强 Embedding 质量
  → 避免两套 Embedding 的不一致
```

### FM 复杂度优化

```
朴素实现：O(k * n^2)  — 遍历所有特征对
  sum_i<j <vi, vj> * xi * xj

数学变换后：O(k * n)
  = 1/2 * sum_f [ (sum_i vi,f * xi)^2 - sum_i (vi,f * xi)^2 ]

  即：先求和再平方 - 先平方再求和
  只需一次遍历即可计算所有二阶交叉
```

### Wide&Deep vs DeepFM 对比

```
                  Wide&Deep           DeepFM
──────────────────────────────────────────────────
低阶交叉          手动设计交叉特征     FM 自动二阶交叉
高阶交叉          DNN                 DNN
特征工程          需要领域知识         无需手动
Embedding 共享    Wide/Deep 各自独立   FM/Deep 共享
参数效率          较低                 更高
迁移成本          需重新设计交叉特征   即插即用
适用场景          有强先验知识时       通用场景
```

---

## 4. DCN（Deep & Cross Network）

### Cross Network 核心公式

```
x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l

分解理解：
  x_l^T * w_l    → 标量（对当前层做加权求和）
  x_0 * scalar   → 与初始输入做缩放
  + x_l          → 残差连接

每层参数量：O(d)（仅 w_l 和 b_l）
  对比 DNN 每层：O(d^2)

L 层 Cross Network → 最高 L+1 阶特征交叉
```

### Cross Network 特性

```
优势：
  1. 参数高效：每层只有 O(d) 参数
  2. 显式交叉：可追踪哪些特征被交叉
  3. 有界阶数：L 层 = 最高 L+1 阶，可控
  4. 残差连接：训练稳定

局限：
  1. 始终与 x_0 交叉（交叉模式受限）
  2. bit-wise 交叉：标量级别操作
     → 不保持 Embedding 的向量语义
  3. 交叉形式固定：x_0 * scalar + residual
```

### DCN-V2 改进

```
DCN-V1: x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
  w_l 是向量 → 只能做 rank-1 交叉

DCN-V2: x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
  W_l 是矩阵 → 更丰富的交叉模式
  ⊙ 是逐元素乘法
  参数量增加到 O(d^2)，但可用低秩分解优化
```

---

## 5. xDeepFM / CIN

### CIN（Compressed Interaction Network）

```
核心思想：vector-wise 交叉（保持 Embedding 语义）

与 DCN 对比：
  DCN  = bit-wise 交叉（标量级别，破坏 Embedding 结构）
  CIN  = vector-wise 交叉（向量级别，保持 Embedding 语义）

CIN 操作类似 CNN：
  - 在特征维度（field 维度）上做类卷积操作
  - 每层显式建模有界阶特征交叉
  - 输出接 sum pooling → 全连接

优势：
  - 显式 + vector-wise 交叉
  - 交叉阶数可控

劣势：
  - 计算复杂度较高：O(H * m * D * H_prev)
  - 实际工业部署较少
```

---

## 6. 模型演进全景对比

```
模型        低阶交叉       高阶交叉      交叉方式    参数效率   工程难度
──────────────────────────────────────────────────────────────────────
LR          手动           无            -          低        低
Wide&Deep   手动           DNN隐式       -          中        中
DeepFM      FM自动二阶     DNN隐式       bit-wise   高        低
DCN         Cross显式      DNN隐式       bit-wise   高        低
xDeepFM     CIN显式        DNN隐式       vector     中        高
AutoInt      -             Self-Att      vector     中        中
```

### 面试高频问题

```
Q1: 如何从 Wide&Deep 迁移到 DeepFM？
A: 去掉 Wide 侧手动交叉特征，用 FM 层替代。
   关键：FM 和 Deep 共享 Embedding，直接复用 Deep 的 Embedding 表。
   迁移后无需维护交叉特征工程 pipeline。

Q2: DCN 的 Cross Network 每层参数量是多少？
A: O(d)，其中 d 是输入维度。对比 DNN 每层 O(d^2)。
   L 层 Cross Network 最高建模 L+1 阶交叉。

Q3: bit-wise vs vector-wise 交叉的区别？
A: bit-wise（DCN）：在 Embedding 的每个标量维度上独立交叉
   vector-wise（CIN/xDeepFM）：以整个 Embedding 向量为单位交叉
   vector-wise 更符合 Embedding 语义（每个 field 的 Embedding 是一个整体）

Q4: 高基数特征（如 item_id 千万级）怎么处理？
A: 四种方案：
   1. 频率截断：低频特征合并为 "other"（如出现 < 5 次）
   2. 特征哈希：hash(feature) % bucket_size（有冲突但实用）
   3. 动态 Embedding：按频率分配不同维度（高频大维度）
   4. 双塔预训练：先用对比学习预训练 item Embedding

Q5: 为什么实际业务中 DeepFM 比 DCN 更常用？
A: 1. DeepFM 结构简单，调参少
   2. FM 的二阶交叉在多数场景已足够
   3. DCN 的 bit-wise 交叉理论优势不总能转化为业务收益
   4. DeepFM 共享 Embedding 的参数效率优势
```

---

## 7. 工程实践要点

```
Embedding 表设计：
  - 维度选择：dim = min(50, ceil(cardinality^0.25))
  - 频率截断：出现 < 5 次 → default Embedding
  - 多值特征：sum/mean pooling 后再拼接
  - 共享 Embedding：同类特征（如 user_city / item_city）可共享

训练技巧：
  - Wide 部分用 FTRL 优化器（稀疏更新）
  - Deep 部分用 Adam/AdaGrad
  - Embedding 学习率可设为全连接层的 0.1-0.5 倍
  - 梯度裁剪：max_norm = 5.0

线上部署：
  - Embedding 表占内存最大（可达数 GB）
  - 热门 item 的 Embedding 缓存到内存
  - 冷门 item 查 Redis/特征服务
  - 模型更新频率：全量日更 + 增量小时更
```
