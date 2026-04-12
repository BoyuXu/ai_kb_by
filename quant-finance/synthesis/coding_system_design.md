# 量化编程与系统设计

> 量化开发的技术栈横跨 Python 数据分析、SQL 数据工程、C++ 低延迟编程，以及系统架构设计。本文从编程工具到系统设计，覆盖量化面试中最常考的工程能力点。

---

## 1. Python 量化栈

### 1.1 NumPy：向量化计算基石

```python
import numpy as np

# broadcasting：不同形状数组的自动扩展
prices = np.array([[100, 200, 150],    # day1: stock A, B, C
                    [102, 198, 153]])   # day2
weights = np.array([0.5, 0.3, 0.2])    # 一维权重自动广播
portfolio_value = (prices * weights).sum(axis=1)  # [145, 146.7]

# 向量化 vs 循环：百倍性能差距
returns = np.diff(prices, axis=0) / prices[:-1]  # 一行搞定收益率矩阵
```

**关键点：**
- `np.einsum` 处理多维张量运算（因子矩阵 x 权重矩阵）
- `np.nan` 系列函数（`nanmean`, `nanstd`）天然处理缺失值
- `dtype` 选择：`float32` 省内存，`float64` 保精度

### 1.2 Pandas：时间序列核心

```python
import pandas as pd

# 多因子横截面计算示例
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=500).repeat(100),
    'stock_id': list(range(100)) * 500,
    'close': np.random.randn(50000).cumsum() + 100,
    'volume': np.random.randint(1e6, 1e8, 50000),
    'market_cap': np.random.uniform(1e9, 1e12, 50000)
})

# 1) 时序因子：20日动量
df['mom_20'] = df.groupby('stock_id')['close'].pct_change(20)

# 2) 横截面标准化（每天截面z-score）
df['mom_zscore'] = df.groupby('date')['mom_20'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 3) 行业中性化（回归残差法）
# 假设 industry 列已有
# df['mom_neutral'] = df.groupby(['date','industry'])['mom_20'].transform(
#     lambda x: x - x.mean()
# )

# 4) 分组回测：按因子分5组看收益
df['quintile'] = df.groupby('date')['mom_zscore'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)

# 5) resample：日频转周频
weekly = df.set_index('date').groupby('stock_id')['close'].resample('W').last()
```

> **面试常问**：`transform` vs `apply` 的区别？`transform` 返回与输入同形状的 Series，适合添加列；`apply` 返回聚合结果。

### 1.3 SciPy：优化与统计

```python
from scipy.optimize import minimize
from scipy import stats

# 组合优化：最小化风险（给定收益约束）
def neg_sharpe(weights, ret_matrix, cov_matrix):
    port_ret = ret_matrix @ weights
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    return -port_ret / port_vol

result = minimize(neg_sharpe, x0=np.ones(10)/10,
                  args=(expected_returns, cov_matrix),
                  constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                  bounds=[(0, 0.3)] * 10)  # 单股上限30%

# Kolmogorov-Smirnov 检验收益分布
ks_stat, p_value = stats.kstest(daily_returns, 'norm')
```

### 1.4 性能优化

```python
from numba import njit

# numba JIT：数值密集型循环加速100x
@njit
def ewma(data, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

# 内存优化：category 类型节省字符串内存
df['industry'] = df['industry'].astype('category')  # 内存减少90%+

# 避免 apply：用 numpy 向量化替代
# 慢: df.apply(lambda row: row['a'] * row['b'], axis=1)
# 快: df['a'].values * df['b'].values
```

> 🔄 **与搜广推的关联**：推荐系统特征工程同样依赖 pandas groupby + transform 做用户/物品统计特征，numba 加速实时特征计算。

---

## 2. 向量化回测 vs 事件驱动回测

### 2.1 向量化回测

一次性对整个时间序列计算信号和收益，核心思想是矩阵运算。

```python
def vectorized_ma_strategy(prices: pd.Series, short=5, long=20):
    """向量化均线策略"""
    ma_short = prices.rolling(short).mean()
    ma_long = prices.rolling(long).mean()

    # 信号矩阵：1=多头, -1=空头, 0=观望
    signal = np.where(ma_short > ma_long, 1, -1)

    # 收益 = 信号 * 次日收益率（信号滞后一天避免未来信息）
    daily_returns = prices.pct_change()
    strategy_returns = pd.Series(signal, index=prices.index).shift(1) * daily_returns

    return strategy_returns
```

### 2.2 事件驱动回测

逐 bar 处理，维护虚拟账户状态，更接近实盘逻辑。

```python
class EventDrivenBacktest:
    def __init__(self, initial_capital=1_000_000):
        self.capital = initial_capital
        self.position = 0
        self.trades = []

    def on_bar(self, timestamp, open_, high, low, close, ma_short, ma_long):
        """逐bar处理"""
        if ma_short > ma_long and self.position <= 0:
            # 买入信号
            shares = int(self.capital * 0.95 / close)
            cost = shares * close * 1.0003  # 含手续费
            self.capital -= cost
            self.position = shares
            self.trades.append(('BUY', timestamp, close, shares))

        elif ma_short < ma_long and self.position > 0:
            # 卖出信号
            revenue = self.position * close * 0.9997  # 含手续费+印花税
            self.capital += revenue
            self.trades.append(('SELL', timestamp, close, self.position))
            self.position = 0

    def run(self, df):
        for _, row in df.iterrows():
            self.on_bar(row['date'], row['open'], row['high'],
                       row['low'], row['close'], row['ma5'], row['ma20'])
```

### 2.3 优劣对比

| 维度 | 向量化回测 | 事件驱动回测 |
|------|-----------|-------------|
| 速度 | 极快（秒级跑完10年） | 慢（分钟级） |
| 灵活性 | 低（难处理复杂逻辑） | 高（任意逻辑） |
| 手续费模拟 | 粗糙（估算） | 精确（逐笔计算） |
| 滑点模拟 | 困难 | 容易 |
| 资金管理 | 难以模拟 | 天然支持 |
| 实盘对接 | 需重写 | 策略代码可复用 |
| 适用场景 | 因子研究、快速验证 | 策略开发、实盘模拟 |

### 2.4 混合方案（工业实践）

```
信号生成（向量化）→ 信号矩阵 → 执行模拟（事件驱动）
     快速                            精确
```

- **Stage 1**：用向量化方式批量计算因子值和信号（快速筛选）
- **Stage 2**：将信号喂入事件驱动引擎，精确模拟执行（手续费、滑点、资金管理）

> 🔄 **与搜广推离线/在线的类比**：向量化回测 ≈ 离线批量评估（AUC/NDCG），事件驱动 ≈ 在线 A/B 测试。两者互补，先用离线快速筛选，再用在线精确验证。推荐系统也是「离线训练 + 在线 serving」的两阶段架构。

---

## 3. SQL 在量化中的应用

### 3.1 窗口函数处理时序数据

```sql
-- 计算20日动量因子
SELECT
    trade_date,
    stock_code,
    close,
    close / LAG(close, 20) OVER (
        PARTITION BY stock_code ORDER BY trade_date
    ) - 1 AS momentum_20d,
    -- 20日波动率
    STDDEV(daily_return) OVER (
        PARTITION BY stock_code
        ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) * SQRT(252) AS volatility_20d
FROM daily_quotes;

-- 横截面排名（每天排名）
SELECT
    trade_date,
    stock_code,
    momentum_20d,
    PERCENT_RANK() OVER (
        PARTITION BY trade_date ORDER BY momentum_20d
    ) AS mom_rank,
    NTILE(5) OVER (
        PARTITION BY trade_date ORDER BY momentum_20d
    ) AS quintile
FROM factor_momentum;
```

### 3.2 行业中性化 SQL 实现

```sql
-- 行业中性化：减去行业均值
WITH industry_avg AS (
    SELECT
        trade_date,
        industry_code,
        AVG(factor_value) AS ind_mean,
        STDDEV(factor_value) AS ind_std
    FROM factor_raw
    GROUP BY trade_date, industry_code
)
SELECT
    f.trade_date,
    f.stock_code,
    (f.factor_value - ia.ind_mean) / NULLIF(ia.ind_std, 0) AS factor_neutral
FROM factor_raw f
JOIN industry_avg ia
  ON f.trade_date = ia.trade_date
 AND f.industry_code = ia.industry_code;
```

### 3.3 大表优化

- **分区策略**：按 `trade_date` 做范围分区（每月/每季一个分区），查询自动裁剪
- **索引策略**：`(stock_code, trade_date)` 复合索引覆盖最常见查询模式
- **物化视图**：日频因子表预计算，避免重复窗口函数计算
- **列式存储**：ClickHouse / DolphinDB 列存天然适合因子宽表查询

> 🔄 **与搜广推特征工程 SQL 的相似之处**：推荐系统中用户行为统计（7天点击数、30天购买数）本质上也是窗口函数 + 分组聚合。区别在于推荐特征通常是 user_id 分组，量化因子是 stock_code 分组。两者的 SQL 模式高度复用。

> **面试常问**：如何高效计算全市场 3000 只股票 x 100 个因子的日度截面数据？答：列式数据库 + 分区表 + 向量化执行引擎（如 DolphinDB 的 SQL 向量化），避免逐行计算。

---

## 4. 因子平台架构设计

### 4.1 分层架构

```
┌─────────────────────────────────────────────────┐
│              服务层 (API + 可视化)                │
│    因子查询 API │ 回测服务 │ Dashboard │ Jupyter  │
├─────────────────────────────────────────────────┤
│              计算层 (因子引擎)                    │
│    因子表达式引擎 │ 调度系统 │ 增量/全量计算      │
├─────────────────────────────────────────────────┤
│              存储层 (因子库)                      │
│    因子值存储 │ 因子元数据 │ 因子血缘关系         │
├─────────────────────────────────────────────────┤
│              数据层 (原始数据)                    │
│    行情数据 │ 财务数据 │ 另类数据 │ 舆情数据      │
└─────────────────────────────────────────────────┘
```

### 4.2 因子库存储设计

**宽表方案**：每行一只股票一天，每列一个因子

| trade_date | stock_code | mom_20 | vol_20 | bp_ratio | ... |
|------------|-----------|--------|--------|----------|-----|
| 2024-01-02 | 000001    | 0.05   | 0.23   | 1.2      | ... |

- 优点：查询快（一次读取多因子），适合截面分析
- 缺点：加新因子需要 ALTER TABLE，schema 不灵活

**长表方案**：每行一个因子值

| trade_date | stock_code | factor_name | factor_value |
|------------|-----------|-------------|-------------|
| 2024-01-02 | 000001    | mom_20      | 0.05        |

- 优点：schema 灵活，加因子无需改表
- 缺点：多因子联合查询需要 PIVOT，性能差

**实践选择**：核心因子用宽表（查询性能优先），实验因子用长表（灵活性优先）。

### 4.3 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 因子计算 | Spark / DolphinDB | 分布式 + 时序原生支持 |
| 因子存储 | ClickHouse / HDF5 | 列存高压缩 + 快速读取 |
| 调度 | Airflow / DolphinDB 内置 | DAG 依赖管理 |
| 元数据 | PostgreSQL | 因子定义、血缘、权限 |
| API | FastAPI | 高性能异步查询 |

> **面试常问**：设计一个支持 5000 只股票 x 500 个因子 x 10 年日频数据的因子库，怎么存？答：约 5000 x 500 x 2500 = 62.5 亿行（长表）或 1250 万行 x 500 列（宽表）。宽表 + 按月分区 + 列式存储（ClickHouse），压缩后约 50-100GB，单机可承载。

---

## 5. 实时行情系统设计

### 5.1 需求分析

- **低延迟**：交易所到策略引擎 <1ms（高频），<100ms（中低频）
- **高吞吐**：全市场 5000+ 标的 x 3秒/tick ≈ 1700 msg/s（A股），期货/期权更高
- **数据一致性**：不丢包、不乱序、不重复
- **高可用**：主备切换 <1s

### 5.2 系统架构

```
交易所行情源 ─→ 网关(解码) ─→ 消息总线 ─→ 策略引擎
     │                           │           │
     │                           ├→ 行情存储  │
     │                           ├→ 监控告警  │
     └── 备用源(灾备) ──────────┘           └→ 风控引擎
```

### 5.3 消息队列选型

| 方案 | 延迟 | 吞吐 | 适用场景 |
|------|------|------|---------|
| Kafka | ~ms级 | 极高 | 中低频、日志、回放 |
| ZeroMQ | ~10us | 高 | 中频策略、进程间通信 |
| 共享内存 | ~100ns | 极高 | 高频、同机进程通信 |
| DPDK + 自研 | ~1us | 极高 | 超高频、做市 |

### 5.4 延迟优化技术栈

1. **网络层**：kernel bypass（DPDK / Solarflare OpenOnload），跳过内核协议栈
2. **序列化**：FlatBuffers / SBE（零拷贝反序列化），避免 Protobuf 的序列化开销
3. **内存**：预分配 + 对象池，避免运行时 malloc
4. **CPU**：核绑定（isolcpus + taskset），避免上下文切换
5. **NUMA**：本地内存访问，避免跨 NUMA 节点

> **面试常问**：设计一个实时行情分发系统，支持 10000 个客户端同时订阅。答：行情网关解码后写入共享内存 ring buffer；分发层通过 ZMQ PUB-SUB 或 UDP 组播推送；客户端按 topic（股票代码）订阅过滤；历史数据回放通过 Kafka 持久化层提供。关键指标：端到端 <5ms，丢包率 <0.001%。

---

## 6. C++ 低延迟编程要点

### 6.1 内存管理

```cpp
// 对象池：预分配避免运行时 malloc
template<typename T, size_t N>
class ObjectPool {
    std::array<T, N> pool_;
    std::array<bool, N> used_;
    size_t next_ = 0;

public:
    T* allocate() {
        for (size_t i = 0; i < N; ++i) {
            size_t idx = (next_ + i) % N;
            if (!used_[idx]) {
                used_[idx] = true;
                next_ = (idx + 1) % N;
                return &pool_[idx];
            }
        }
        return nullptr;  // 池满
    }

    void deallocate(T* ptr) {
        size_t idx = ptr - pool_.data();
        used_[idx] = false;
    }
};
```

### 6.2 Lock-Free Ring Buffer

```cpp
#include <atomic>
#include <array>

// 单生产者单消费者 (SPSC) 无锁环形缓冲区
template<typename T, size_t Size>
class SPSCRingBuffer {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");

    alignas(64) std::atomic<size_t> head_{0};  // 生产者写
    alignas(64) std::atomic<size_t> tail_{0};  // 消费者读
    std::array<T, Size> buffer_;

public:
    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & (Size - 1);
        if (next == tail_.load(std::memory_order_acquire))
            return false;  // 满
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return false;  // 空
        item = buffer_[tail];
        tail_.store((tail + 1) & (Size - 1), std::memory_order_release);
        return true;
    }
};
```

**关键设计点：**
- `alignas(64)` 避免 false sharing（head 和 tail 在不同 cache line）
- `Size` 为 2 的幂，用位运算替代取模
- `memory_order_acquire/release` 最小化内存屏障开销

### 6.3 CPU 亲和性与 NUMA

```bash
# 隔离 CPU 核心给策略进程
# /etc/default/grub: GRUB_CMDLINE_LINUX="isolcpus=4,5,6,7"

# 绑定进程到特定核心
taskset -c 4 ./strategy_engine

# NUMA 感知：绑定到本地内存节点
numactl --cpunodebind=0 --membind=0 ./strategy_engine
```

### 6.4 性能量级参考

| 操作 | 延迟 |
|------|------|
| L1 cache 访问 | ~1ns |
| L2 cache 访问 | ~4ns |
| L3 cache 访问 | ~12ns |
| 主内存访问 | ~60ns |
| mutex lock/unlock | ~25ns（无竞争） |
| 系统调用 | ~1us |
| 内核网络栈 (TCP) | ~10us |
| DPDK 收包 | ~1us |
| 上下文切换 | ~3-5us |

> **面试常问**：为什么高频交易用 C++ 而不是 Java/Python？答：(1) 确定性延迟——无 GC 停顿；(2) 内存布局控制——cache 友好的数据结构；(3) 零开销抽象——模板在编译期展开；(4) 直接硬件访问——kernel bypass、SIMD 指令。

> 🔄 **与搜广推推理优化的关联**：推荐系统在线推理也追求低延迟，使用的技术有交集——模型量化（类似数值精度选择）、batch inference（类似向量化）、模型缓存（类似对象池）。但量化高频的延迟要求（微秒级）比推荐系统（毫秒级）高 1000 倍。

---

## 参考与交叉引用

- [[factor_investing]] — 因子投资理论，因子计算的业务背景
- [[strategy_development]] — 策略开发流程，回测框架的上层应用
- [[ml_in_quant]] — 机器学习在量化中的应用，与因子平台的数据流关系
- [[../rec-search-ads/rec-sys/synthesis/feature_engineering]] — 搜广推特征工程，SQL 模式复用
- [[concepts/embedding_everywhere]] — Embedding 技术，在因子表示学习中的应用
