# 模型压缩与部署优化：从训练到上线全链路

## 1. 模型压缩三板斧

### 剪枝（Pruning）

```
结构化剪枝：
  移除整个 channel / layer / attention head
  优势：直接减少计算量，无需特殊硬件支持
  方法：
    1. 按权重大小：移除 L1-norm 最小的 channel
    2. 按梯度敏感度：移除对 loss 影响最小的 channel
    3. 迭代剪枝：剪 → 微调 → 再剪 → 再微调

非结构化剪枝：
  移除单个权重（置零）
  可达到 90%+ 稀疏率
  但需要稀疏计算硬件支持（如 NVIDIA Ampere 的稀疏 Tensor Core）
  实际工业应用较少（硬件支持不成熟）

推荐系统中的剪枝：
  - Embedding 表剪枝：低频 item 共享 Embedding
  - MLP 层剪枝：按 channel 重要度裁剪
  - 注意力头剪枝：移除贡献小的 attention head
```

### 量化（Quantization）

```
精度降低：FP32 → FP16 → INT8 → INT4

训练后量化（PTQ - Post Training Quantization）：
  训练完成后直接转换精度
  优势：简单，无需重新训练
  劣势：精度损失较大（尤其 INT8 以下）
  适用：对精度不敏感的场景

量化感知训练（QAT - Quantization Aware Training）：
  训练中模拟量化噪声
  前向：权重 → 量化 → 反量化 → 正常计算
  反向：用 STE（Straight-Through Estimator）近似梯度
  优势：精度损失小（通常 < 0.1% AUC）
  劣势：训练时间增加 20-50%

效果：
  FP32 → FP16：速度 ~2x，内存 ~0.5x，精度几乎无损
  FP32 → INT8：速度 ~3-4x，内存 ~0.25x，精度轻微下降
  FP32 → INT4：速度 ~5-6x，但精度损失明显

混合精度：
  Embedding 表 → FP16（占内存最大，收益最大）
  MLP 权重 → INT8（计算密集）
  BatchNorm / LayerNorm → FP32（对精度敏感）
```

### 知识蒸馏（Knowledge Distillation）

```
基本框架：
  Teacher（大模型）→ soft label → Student（小模型）

损失函数：
  L = alpha * L_hard + (1 - alpha) * L_soft

  L_hard = CE(student_output, hard_label)     // 与真实标签的交叉熵
  L_soft = KL(student_output, teacher_output) // 与 Teacher 输出的 KL 散度

  temperature T 控制软标签的平滑度：
  soft_label = softmax(logits / T)
  T 越大 → 分布越平滑 → 传递更多"暗知识"

推荐系统蒸馏实践：
  1. 模型蒸馏：大模型 → 小模型
     Teacher: 12 层 Transformer + 全序列
     Student: 2 层 MLP + 截断序列

  2. 特征蒸馏：复杂特征 → 简单特征
     Teacher: 使用交叉特征 + 序列特征 + 实时特征
     Student: 仅使用 ID 特征 + 少量统计特征
     适用：在线推理需要极低延迟

  3. 序列蒸馏：长序列 → 短序列
     Teacher: 全部 1000 条行为
     Student: 只用最近 50 条
     目标：Student 输出逼近 Teacher 的得分分布
```

---

## 2. A/B 测试指标体系

### 离线指标

```
AUC（Area Under ROC Curve）：
  排序能力评估，不受正负样本比例影响
  AUC = P(score_pos > score_neg)
  注意：AUC 对全局排序敏感，但对头部排序不敏感

logloss（Log Loss / Cross Entropy）：
  概率校准评估
  L = -1/N * sum[y*log(p) + (1-y)*log(1-p)]
  AUC 好但 logloss 差 → 排序对但概率不准

GAUC（Group AUC）：
  按用户分组计算 AUC 再加权平均
  GAUC = sum(impression_i * AUC_i) / sum(impression_i)
  比全局 AUC 更贴合推荐场景

NDCG（Normalized Discounted Cumulative Gain）：
  考虑排序位置的指标
  DCG = sum(rel_i / log2(i+1))
  NDCG = DCG / IDCG
```

### 在线指标

```
核心业务指标：
  CTR = 点击数 / 曝光数
  CVR = 转化数 / 点击数
  GMV = 成交金额
  人均播放时长（视频推荐）
  次日留存率（长期价值）

系统指标：
  P50/P99 延迟
  吞吐量（QPS）
  GPU 利用率
  模型更新延迟

注意：
  离线 AUC 涨不一定线上指标涨
  原因：样本分布差异、特征不一致、位置偏差等
  必须做线上 AB 测试验证
```

---

## 3. 在线推理优化

### 算子融合

```
将多个连续算子合并为单个 GPU kernel：
  例：MatMul → BiasAdd → ReLU → 3 次 kernel launch
  融合后：MatMul_BiasAdd_ReLU → 1 次 kernel launch

效果：
  减少 kernel launch 开销
  减少显存读写（中间结果不落显存）
  推理延迟降低 20-40%

工具：
  TensorRT：自动算子融合 + INT8 量化
  ONNX Runtime：图优化 + 算子融合
  XLA：TensorFlow 的编译优化
```

### 动态 Batching

```
问题：推荐请求到达时间不均匀
  逐条推理 → GPU 利用率低（大量空闲）
  固定 batch → 高延迟（等凑够 batch）

动态 Batching：
  设置最大等待时间（如 5ms）和最大 batch size（如 64）
  任一条件满足就触发推理
  GPU 利用率 ↑ 3-5x，延迟可控

Triton Inference Server 原生支持：
  - 配置 max_batch_size / max_queue_delay_microseconds
  - 自动合并来自不同请求的样本
```

### 异步特征获取

```
传统：特征获取 → 模型推理（串行，总延迟 = T1 + T2）

优化：
  1. 用户特征和物品特征并行获取
  2. 特征获取与上一批次推理并行（pipeline）
  3. 非关键特征异步获取（缺失时用默认值）

效果：总延迟 ≈ max(T_feature, T_inference)
```

### 多级缓存

```
L1：热门物品打分缓存（秒级 TTL）
  命中率 ~30-50%（热门物品请求集中）
  Key: user_segment + item_id → cached_score

L2：用户特征缓存（分钟级 TTL）
  用户画像、近期行为统计
  更新策略：行为发生时异步刷新

L3：Embedding 缓存
  高频 item Embedding 缓存到 GPU 显存
  低频 item 从 CPU 内存 / Redis 查询

缓存失效策略：
  时间失效 + 事件失效（用户新行为 → 清 L1/L2）
```

---

## 4. 分布式服务架构

### 典型架构

```
客户端请求
    │
    ▼
API Gateway（限流、鉴权、路由）
    │
    ▼
负载均衡（Nginx / Envoy）
    │
    ├── 模型服务实例 1（GPU）
    ├── 模型服务实例 2（GPU）
    └── 模型服务实例 N（GPU）
    │
    ├── 特征服务（Feature Store）
    │     ├── Redis（实时特征）
    │     └── HBase（离线特征）
    │
    └── Embedding 服务
          └── 分片存储（按 hash 分 N 片）
```

### 模型分片

```
问题：超大 Embedding 表放不下单机

方案：
  按 item_id hash 分片到 N 台服务器
  shard_id = hash(item_id) % N

  请求处理：
  1. 解析请求中涉及的 item_id 列表
  2. 按 shard 分组，并行发送到对应服务器
  3. 收集各 shard 返回的 Embedding
  4. 本地完成 DNN 推理

优化：
  - 热门 item 全量缓存到每台机器（避免跨机查询）
  - 一致性哈希（扩缩容时最小化数据迁移）
```

### 推理框架对比

```
框架                优势                     劣势
──────────────────────────────────────────────────────────
TF Serving         TF 原生支持              仅 TF 模型
                   热更新模型
                   版本管理

Triton             多框架（TF/PyTorch/ONNX）配置复杂
                   动态 batching
                   模型集成（pipeline）

ONNX Runtime       跨框架（转 ONNX）        模型转换可能有精度差
                   图优化强
                   边缘端部署友好

TensorRT           极致性能（INT8/FP16）    仅 NVIDIA GPU
                   算子融合自动化            模型转换受限
```

---

## 5. 性能瓶颈定位

### 排查清单

```
1. 延迟分解：
   总延迟 = 特征查询 + 模型推理 + 网络传输 + 排队等待
   先定位哪个阶段是瓶颈

2. 特征查询瓶颈：
   - Redis 慢查询 → 检查 key 设计、网络延迟
   - 特征数量太多 → 特征裁剪、异步获取
   - 批量查询优化（pipeline / multi-get）

3. 模型推理瓶颈：
   - GPU 利用率低 → 增大 batch size
   - 模型太大 → 量化 / 蒸馏 / 剪枝
   - 算子效率低 → TensorRT 优化

4. 内存瓶颈：
   - Embedding 表太大 → 分片 / 量化 / 频率截断
   - 显存不足 → FP16 推理 / 模型分片

5. 吞吐瓶颈：
   - 单机 QPS 不够 → 水平扩容
   - GPU 排队 → 动态 batching 调优
   - 网络带宽 → 特征压缩
```

---

## 6. 前沿方向：因果推断 + 强化学习

### 因果推断消除偏差

```
曝光偏差（Selection Bias）：
  只观察到展示过的样本 → 模型学到的是 P(Y|X, 被展示)
  IPW（逆倾向加权）：
    loss_i = loss_i / P(展示 | 特征)
    给低曝光样本更大权重，纠正采样偏差

  双重稳健估计（Doubly Robust）：
    结合 IPW 和回归模型
    任一个估计正确 → 结果无偏
    比纯 IPW 更稳定

位置偏差（Position Bias）：
  用户倾向于点击靠前位置的物品
  解法：
    训练时引入位置特征（position embedding）
    推理时去除位置特征（或设为默认值）
    → 模型学到的是去除位置影响后的真实偏好

流行度偏差（Popularity Bias）：
  热门物品被推荐更多 → 曝光更多 → 点击更多 → 更被推荐
  因果图建模：
    do-calculus 干预：P(Y | do(X)) ≠ P(Y | X)
    后门调整：P(Y|do(X)) = sum_Z P(Y|X,Z) * P(Z)
    Z 是混淆因子（如物品流行度）
```

### 强化学习

```
MDP 建模：
  状态 S：用户画像 + 历史行为 + 上下文
  动作 A：推荐物品（列表）
  奖励 R：即时奖励（点击）+ 延迟奖励（留存、付费）
  转移 P：用户状态变化

方法：
  Q-Network：学习 Q(s, a) 值函数
    Q(s,a) = r + gamma * max_a' Q(s', a')
    适用：离散动作空间

  Policy Gradient：直接学习推荐策略 pi(a|s)
    优势：可处理连续/大动作空间

探索 - 利用：
  epsilon-Greedy：以 epsilon 概率随机探索
  UCB：score = mean + c * sqrt(ln(N) / n_i)
    平衡均值和不确定性

  Thompson Sampling：从后验分布采样
    Beta 分布：Beta(alpha + successes, beta + failures)
    每次采样一个 CTR 估计值
    自然探索：不确定性大的 item 更容易被采样

挑战：
  1. 动作空间巨大（item 数千万）→ 需要动作压缩
  2. 奖励稀疏 + 延迟 → 需要 reward shaping
  3. 在线探索风险 → 离线 RL（Batch RL）
  4. 训练不稳定 → 保守估计（CQL/BCQ）
```

---

## 7. 面试高频问题

```
Q1: 离线 AUC 涨但线上指标不涨，怎么排查？
A: 按优先级排查：
   1. 特征一致性：对比离线/在线特征值
   2. 样本分布：训练集 vs 线上数据分布差异
   3. 负采样校准：线上概率是否校准
   4. 位置偏差：模型是否学到了位置信号
   5. AB 实验配置：分桶是否均匀，样本量是否足够

Q2: 模型上线后延迟从 20ms 涨到 80ms，怎么优化？
A: 1. 分解延迟：特征 / 推理 / 网络各占多少
   2. 模型侧：量化（FP32→FP16/INT8）、剪枝、蒸馏
   3. 系统侧：动态 batching、算子融合、异步特征
   4. 缓存：热门 item 打分缓存、用户特征缓存
   5. 架构：模型分片、Embedding 服务独立部署

Q3: Triton vs TF Serving 怎么选？
A: 纯 TF 模型 → TF Serving（简单、原生）
   多框架混合 → Triton（支持 TF/PyTorch/ONNX）
   需要极致性能 → Triton + TensorRT 后端
   边缘端 → ONNX Runtime

Q4: IPW 的倾向分数怎么估计？
A: 训练一个独立的展示概率模型：
   P(展示 | 用户特征, 物品特征, 上下文)
   常用 LR / 小型 DNN
   注意：倾向分数过小会导致方差爆炸
   → 截断：max(propensity, 0.01)

Q5: 探索（Exploration）和利用（Exploitation）怎么平衡？
A: 三种常用方法：
   1. epsilon-Greedy：简单但粗糙，epsilon 随时间衰减
   2. UCB：理论保证，但需要维护每个 item 的统计量
   3. Thompson Sampling：最推荐，自然平衡，实现简单
   实践中 TS 效果最好：对不确定性大的 item 自动多探索
```
