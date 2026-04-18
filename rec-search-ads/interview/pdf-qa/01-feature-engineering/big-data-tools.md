# Ch4 大数据处理工具与应用

## 核心概念总览

本章覆盖 Spark 和 Flink 两大数据处理框架，以及数据倾斜、实时特征等工程实战问题。

---

## 1. Spark 核心架构

### RDD vs DataFrame vs Dataset

```
RDD:
  - 最底层抽象，强类型，惰性计算
  - 支持任意 transformation/action
  - 缺点：无 Catalyst 优化、序列化开销大

DataFrame:
  - 带 schema 的分布式表（行+列）
  - 经过 Catalyst 优化器 + Tungsten 引擎
  - 缺点：弱类型，编译期不检查列名

Dataset:
  - DataFrame + 强类型（Scala/Java）
  - 兼具类型安全和 Catalyst 优化
  - Python 中 DataFrame == Dataset[Row]
```

### 宽依赖 vs 窄依赖

```
窄依赖（Narrow）：父 RDD 每个分区最多被一个子分区使用
  - 例：map, filter, union
  - 可 pipeline 执行，无需 shuffle

宽依赖（Wide）：父 RDD 分区被多个子分区依赖
  - 例：groupByKey, reduceByKey, join
  - 触发 shuffle，产生新 Stage 边界
```

### Catalyst 优化器

四阶段流程：
1. 解析（Parsing）→ 未解析逻辑计划
2. 分析（Analysis）→ 解析列名、类型
3. 逻辑优化 → 谓词下推、列裁剪、常量折叠
4. 物理计划 → 选择最优执行策略（如 BroadcastHashJoin vs SortMergeJoin）

### Tungsten 引擎

- 堆外内存管理，避免 GC
- 缓存感知的数据布局（行式 → UnsafeRow 紧凑二进制）
- 全阶段代码生成（Whole-Stage CodeGen）：将多个算子融合为单个 JVM 函数

---

## 2. Spark SQL 实战

### 窗口函数

```sql
-- 连续登录天数计算（经典面试题）
SELECT user_id, login_date,
  login_date - ROW_NUMBER() OVER (
    PARTITION BY user_id ORDER BY login_date
  ) * INTERVAL '1 day' AS grp
FROM logins
-- 同组 = 连续登录，再 GROUP BY grp 计 COUNT
```

### 常用优化手段

- 分区裁剪：WHERE 条件命中分区列
- Broadcast Join：小表 < 10MB 自动广播
- AQE（Adaptive Query Execution）：运行时自动合并小分区、切换 Join 策略

---

## 3. Spark Streaming vs Flink

```
对比维度          Spark Streaming        Flink
──────────────────────────────────────────────────
计算模型          微批（Micro-Batch）     连续流（Continuous）
延迟              秒级（批间隔）          毫秒级
状态管理          基于 RDD 快照           原生 KeyedState
窗口支持          处理时间为主            事件时间 + Watermark
Exactly-Once     通过 WAL + 幂等输出     Checkpoint（Chandy-Lamport）
背压处理          限速接收                Credit-Based 自然背压
适用场景          准实时 ETL/报表         实时特征/风控/CEP
```

---

## 4. Flink 核心机制

### Watermark 机制

- 解决乱序数据的事件时间处理
- Watermark = 当前最大事件时间 - 允许延迟
- 当 Watermark 越过窗口结束时间 → 触发窗口计算
- 迟到数据处理：allowedLateness + sideOutput

### State Backend

```
MemoryStateBackend：状态存内存，小状态用
FsStateBackend：状态存内存，checkpoint 写文件系统
RocksDBStateBackend：状态存 RocksDB（磁盘）
  - 支持超大状态（TB级）
  - 增量 checkpoint
  - 生产首选
```

### Checkpoint（Chandy-Lamport）

1. JobManager 注入 Barrier 到数据流
2. 算子收到所有输入的 Barrier → 快照当前状态
3. 对齐模式：缓存先到 Barrier 通道的数据（Exactly-Once）
4. 非对齐模式：直接处理，牺牲一致性换低延迟（At-Least-Once）

---

## 5. 数据倾斜处理

### 识别信号

- Spark UI 中某个 Task 耗时远超其他（长尾）
- Shuffle Read/Write 严重不均

### 解决方案

```
1. 加盐打散（Salting）
   - 热 key 拼接随机前缀 → 打散到多分区
   - 对应维表也需要膨胀（笛卡尔 × 盐值数）

2. Broadcast Join
   - 小表直接广播，避免 shuffle
   - 适用：大表 JOIN 小表（< 几百MB）

3. 两阶段聚合
   - 第一阶段：加随机前缀 → 局部聚合
   - 第二阶段：去前缀 → 全局聚合
   - 适用：groupBy + agg 场景

4. 热 key 单独处理
   - 识别 Top-N 热 key → 单独走 Broadcast Join
   - 其余正常 shuffle join

5. Bucket Join
   - 预先按 join key 分桶存储
   - 避免运行时 shuffle
```

---

## 6. Lambda / Kappa 架构

### Lambda 架构

```
          ┌─ 批处理层（Batch Layer）──→ 批视图
原始数据 ──┤
          └─ 速度层（Speed Layer）──→ 实时视图
                                       ↓
                              服务层（合并查询）
```

- 优点：准确性（批）+ 实时性（流）
- 缺点：维护两套代码，逻辑一致性难保证

### Kappa 架构

- 统一用流处理引擎（Flink）
- 重放 Kafka 消息替代批处理
- 优点：架构简单，一套代码
- 缺点：对流处理引擎要求极高

---

## 7. 特征工程与 Feature Store

### 推荐系统特征分类

```
用户侧：用户画像（年龄/性别）、历史行为统计、实时行为序列
物品侧：物品属性、统计特征（CTR/热度）、内容特征
交叉特征：用户×物品（历史交互、类目偏好匹配度）
上下文：时间、位置、设备、网络
```

### Feature Store 架构

```
离线特征 → Hive/Spark 计算 → 写入 Redis/HBase
实时特征 → Flink 计算 → 写入 Redis
                                  ↓
               在线服务 ← 统一 Feature API 查询
```

关键要求：
- 离线在线一致性（同一份特征定义）
- 低延迟（P99 < 10ms）
- 特征版本管理和回溯能力

---

## 面试高频考点

1. RDD/DataFrame/Dataset 区别 → 关注 Catalyst 优化能否生效
2. 宽窄依赖 → 影响 Stage 划分和容错
3. Spark vs Flink 流处理 → 延迟、一致性、状态管理三维对比
4. Flink Checkpoint 原理 → Chandy-Lamport + Barrier 对齐
5. 数据倾斜 → 先识别（UI/日志），再按场景选方案
6. 窗口函数连续登录 → ROW_NUMBER 差值分组法
7. Feature Store → 离线在线一致性是核心挑战
