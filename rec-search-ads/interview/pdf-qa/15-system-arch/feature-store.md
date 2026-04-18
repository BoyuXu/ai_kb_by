# 特征存储架构：实时/离线特征、Feature Store、特征一致性

## 1. 特征存储的核心问题

### 1.1 为什么需要 Feature Store？
```
推荐系统的特征管理痛点：

痛点一：训练-推理不一致（Training-Serving Skew）
  训练时用离线批量计算的特征（Hive/Spark）
  推理时用在线实时计算的特征（Flink/Redis）
  两套独立的特征计算逻辑 → 计算结果不一致 → 模型效果下降

痛点二：特征重复计算
  推荐模型、广告模型、搜索模型各自维护一套特征计算 Pipeline
  相同特征（如用户近 30 天购买次数）被多个团队重复实现

痛点三：特征血缘不可追踪
  特征从哪张表来的？用了什么计算逻辑？版本变更记录？
  缺乏统一管理 → Debug 困难 → 上线事故难以定位

Feature Store 的价值：
  一次定义 → 训练推理共用 → 统一血缘管理 → 跨团队复用
```

### 1.2 Feature Store 架构
```
                Feature Registry（特征注册中心）
                    |  特征定义/血缘/版本
                    v
  ┌────────────────────────────────────────────┐
  │            Feature Store                     │
  │                                              │
  │  离线特征 Pipeline          实时特征 Pipeline  │
  │  Spark/Hive 批量计算        Flink 流式计算     │
  │      |                          |             │
  │      v                          v             │
  │  离线存储                    在线存储          │
  │  (Hive/S3/DFS)              (Redis/DynamoDB)  │
  │      |                          |             │
  │      +────── 统一特征服务 API ──+              │
  │              gRPC/REST                        │
  └─────────────|──────────────|──────────────────┘
              训练                推理

主流开源实现：
  Feast：Google 开源，轻量级，社区活跃
  Tecton：Feast 商业版，企业级功能
  Hopsworks：开源全功能 Feature Store
  自研：大厂通常自研（字节、阿里、美团）
```

---

## 2. 实时特征工程

### 2.1 实时特征类型
```
类型一：滑动窗口聚合特征
  - 近 30 分钟用户点击次数
  - 近 1 小时物品曝光量
  - 近 24 小时品类偏好分布
  计算：Flink 时间窗口聚合

类型二：序列特征
  - 用户最近 N 次行为的物品 ID 列表
  - 存储：Redis Sorted Set（按时间戳排序）
  - 读取：ZREVRANGE 取最新 N 条
  - 用于 DIN/DIEN 等序列模型

类型三：实时交叉统计特征
  - 用户-品类交叉 CTR
  - 用户-作者交叉互动率
  - 计算：Flink 中维护 MapState

类型四：实时上下文特征
  - 当前时段、设备类型、网络状况
  - 当前会话已浏览物品数
  - 直接从请求上下文提取
```

### 2.2 实时特征 Pipeline（Kafka + Flink + Redis）
```
全链路流程：

用户行为（点击/购买/浏览）
    |
    v
客户端 SDK 上报事件
    |
    v
Kafka（消息队列，削峰/解耦）
    |  topic: user_behavior_events
    v
Flink 流处理引擎
    |  ┌─ 事件时间语义（EventTime + Watermark）
    |  ├─ 状态管理（RocksDB StateBackend）
    |  ├─ 窗口计算（滑动窗口/会话窗口）
    |  └─ 异步 IO（查维表/外部数据）
    v
Redis Cluster（在线特征存储）
    |  Key 设计：feature:{user_id}:{feature_name}
    |  TTL 设计：实时特征 TTL=24h，会话特征 TTL=30min
    v
特征服务 API（gRPC）
    |  批量查询 + Pipeline + 本地缓存
    v
推荐模型推理

关键设计点：
  幂等性：Flink 的 Exactly-Once 语义 + Kafka 事务
  延迟目标：事件 → Redis 可读 < 1 秒（P99）
  高可用：Flink Checkpoint + Kafka 多副本
```

### 2.3 特征延迟与降级
```
特征查询延迟要求：P99 < 5ms

延迟优化：
  1. Redis Pipeline 批量查询（N次网络 → 1次网络）
  2. 本地缓存热特征（Caffeine/Guava，TTL=10s）
  3. 预计算：常用特征在 Flink 阶段就计算好，不在推理时计算
  4. 异步并行：多类特征并行查询

降级策略（特征不可用时）：
  Level 1：用最近一次缓存值（陈旧但可用）
  Level 2：用离线特征替代实时特征
  Level 3：用默认值填充（全局均值/中位数）
  Level 4：跳过使用该特征的模型分支

监控告警：
  特征缺失率 > 1% → P1 告警
  特征延迟 P99 > 5ms → P2 告警
  特征值分布异常（PSI > 0.2）→ P2 告警
```

---

## 3. 离线特征工程

### 3.1 离线特征计算
```
计算引擎：Spark / Hive / BigQuery
调度：Airflow / DolphinScheduler
频率：每日全量 + 每小时增量

典型离线特征：
  - 用户画像：历史总购买金额、注册天数、品类偏好向量
  - 物品统计：历史 CTR、30天购买转化率、评分均值
  - 交叉特征：用户-品类历史偏好矩阵
  - Embedding 特征：离线训练的 user/item Embedding

产出流程：
  Hive 表（离线计算结果）
      |
      v
  批量导入脚本（每日/每小时）
      |
      v
  在线存储（Redis/DynamoDB）
      |  同时写入离线存储（训练用）
      v
  模型训练 + 在线推理 共用同一份特征
```

### 3.2 特征存储选型
```
存储方案         适用场景               延迟      容量      成本
Redis Cluster   用户实时特征/热数据     <1ms     TB 级     高
Cassandra       用户历史特征/大规模     5-10ms   PB 级     中
DynamoDB        Serverless/低运维       5ms      无限      中
本地缓存        超热数据/全局统计       <0.1ms   GB 级     低
HBase           大规模离线特征         10-50ms   PB 级     低

最佳实践（三级缓存）：
  L1：本地缓存（全局统计量，如全局热门物品 top100）
  L2：Redis Cluster（用户级热特征，如实时兴趣序列）
  L3：Cassandra/HBase（冷特征，如用户长期画像）

缓存命中率目标：L1 > 90%，L1+L2 > 99%
```

---

## 4. 特征一致性

### 4.1 Training-Serving Skew
```
不一致来源：

来源一：计算逻辑不一致
  训练：SQL 写的特征 JOIN
  推理：Python/Java 实现的特征计算
  → 微妙的边界条件差异（NULL 处理、时间窗口对齐）

来源二：时间穿越（Data Leakage）
  训练时误用了事件发生后的特征
  例：用明天的 CTR 预测今天的点击
  → 离线指标虚高，上线效果暴跌

来源三：特征延迟
  训练时用的是精确特征（离线批量计算）
  推理时用的是近似特征（实时流式计算，有延迟）
  → 小差异但积累导致效果偏差

来源四：特征版本不同步
  模型训练用 v1 版特征定义
  上线后特征升级到 v2
  → 模型输入分布与训练时不一致
```

### 4.2 一致性保障方案
```
方案一：特征日志回放（Feature Logging & Replay）
  在线推理时：将查询到的特征值原样写入日志
  离线训练时：不重新计算特征，直接用日志中的特征值
  → 训练用的特征 = 推理时实际使用的特征（完全一致）

  代价：存储量大（每次推理都要记录全量特征）
  优化：只记录变化的特征 + 压缩存储

方案二：Feature Store 统一计算
  训练和推理共用同一份特征计算逻辑
  用 DSL（领域特定语言）定义特征：
    feature "user_30d_purchase_count":
      source: user_purchase_events
      window: 30d
      aggregation: count
      key: user_id
  → 一次定义，批处理和流处理自动生成计算代码

方案三：Shadow Mode 验证
  新模型上线前，在影子模式运行：
  - 实际推理用旧模型
  - 新模型并行推理但不输出结果
  - 对比新旧模型的输入特征是否一致
  - 对比新旧模型的输出分布是否符合预期
  验证通过后才正式切流

方案四：特征版本管理
  每个特征有唯一版本号
  模型绑定特定版本的特征集
  特征升级时：新版本共存 → 新模型用新版 → 旧模型下线 → 旧版本清理
```

### 4.3 时间点一致性（Point-in-Time Correctness）
```
问题：训练数据中每条样本应使用该事件发生时刻的特征快照
      不能用事件之后才有的特征（时间穿越）

实现：
  特征表设计带时间戳：
    (entity_id, feature_value, effective_timestamp)

  训练时做 Point-in-Time JOIN：
    SELECT features.value
    FROM features
    WHERE features.entity_id = events.entity_id
      AND features.timestamp <= events.timestamp
    ORDER BY features.timestamp DESC
    LIMIT 1

  对每条训练样本，取事件发生前最近的特征版本

  Feature Store 的自动化支持：
  Feast 的 get_historical_features() 自动处理 PIT JOIN
```

---

## 5. 缓存策略

### 5.1 缓存问题三兄弟
```
缓存穿透（Cache Penetration）：
  问题：查询不存在的 Key → 每次都打到数据库
  原因：恶意请求/数据删除后缓存未同步
  解决：
    1. 缓存空值（Null Object）+ 短 TTL（5min）
    2. 布隆过滤器：请求先查布隆过滤器，不存在的 Key 直接返回
    3. 接口层面的参数校验和限流

缓存击穿（Cache Breakdown）：
  问题：热 Key 过期瞬间大量请求涌入数据库
  原因：热门物品/用户的特征 Key 同时过期
  解决：
    1. 互斥锁：只放一个请求回源，其他等待
    2. 永不过期 + 异步更新：逻辑过期而非物理过期
    3. 提前续期：在 TTL 到期前异步刷新

缓存雪崩（Cache Avalanche）：
  问题：大量 Key 同时过期 → 数据库压力突然暴增
  原因：同一时间批量写入的 Key 使用相同 TTL
  解决：
    1. TTL 加随机偏移：TTL = base_ttl + random(0, delta)
    2. 多级缓存：L1 本地缓存兜底
    3. 限流降级：数据库层面的请求限流
```

### 5.2 推荐系统缓存设计
```
缓存分层：

Layer 1 - 本地缓存（进程内）：
  内容：全局统计特征、热门物品 Embedding
  实现：Caffeine/Guava Cache
  TTL：10-60 秒
  容量：< 1GB
  延迟：< 0.1ms

Layer 2 - 分布式缓存：
  内容：用户实时特征、物品实时统计
  实现：Redis Cluster（16-64 分片）
  TTL：1-24 小时
  容量：TB 级
  延迟：< 1ms

Layer 3 - 后端存储：
  内容：用户完整画像、物品历史特征
  实现：Cassandra / DynamoDB
  TTL：无限（按需清理）
  容量：PB 级
  延迟：5-50ms

推荐结果缓存：
  Key: user_id + scene_id + timestamp_bucket
  Value: 推荐列表（物品 ID 列表）
  TTL: 5-10 分钟
  作用：重复请求直接返回，避免重复计算
  注意：TTL 不能太长，否则影响推荐实时性
```

---

## 6. 面试高频问题

```
Q: Feature Store 解决什么核心问题？
A: 训练-推理特征不一致（Training-Serving Skew）。
   统一特征定义和计算逻辑，一次定义训练推理共用，
   消除两套独立计算导致的偏差。

Q: 如何保证实时特征的延迟 < 5ms？
A: 四层优化：1) Flink 预计算（不在推理时计算）
   2) Redis Pipeline 批量查询 3) 本地缓存热特征
   4) 特征不可用时用默认值降级。

Q: 特征日志回放 vs Feature Store 统一计算，怎么选？
A: 日志回放更简单直接，完全消除不一致但存储成本高。
   Feature Store 统一计算更优雅，但 DSL 定义的批/流
   转换不一定完全等价（浮点精度等）。
   实践中：日志回放做 ground truth + Feature Store 做日常管理。

Q: 时间穿越（Data Leakage）如何检测？
A: 1) 检查特征时间戳 vs 事件时间戳的先后关系
   2) 离线 AUC 异常高（> 0.95）时高度怀疑
   3) 用 Point-in-Time JOIN 强制保证特征时间正确性
   4) Shadow Mode 对比在线/离线特征值差异。

Q: 缓存穿透、击穿、雪崩的区别和解决方案？
A: 穿透=查不存在的Key（用布隆过滤器/缓存空值），
   击穿=热Key过期时并发请求（用互斥锁/永不过期），
   雪崩=大量Key同时过期（TTL加随机偏移/多级缓存）。
```
