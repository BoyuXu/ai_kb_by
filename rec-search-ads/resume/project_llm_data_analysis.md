# 广告数据智能分析 Agent：基于 NL2SQL + LLM 的自动化投放洞察系统

> 一句话定位：让运营用自然语言提问，系统自动查询 ClickHouse、检测异常、生成分析报告，将日常 2~3小时数据分析压缩到 5分钟

**核心技术栈：** NL2SQL | Chain-of-Thought | ClickHouse | LLM Agent | Prophet | STL 分解 | 异常检测 | 多维归因

---

## 1. 项目背景与痛点

### 业务规模

| 指标 | 规模 |
|------|------|
| ClickHouse 日志量 | 每天 100亿行曝光/点击日志 |
| 总数据量 | 10TB+（保留 90天） |
| 广告数据维度 | 渠道 × 创意 × 人群 × 时段 = 数百万组合 |
| 运营团队 | 5~20人，每人管理 50~200个广告主 |

### 核心痛点

**痛点一：数据维度爆炸，人工分析效率低**
- 一个广告账户有：10个渠道 × 50个创意 × 20个人群包 × 24个时段 = 24万维度组合
- 运营每天手工翻看报表，花费 2~3小时，仍然可能遗漏关键问题
- SQL 能力参差不齐，复杂分析依赖数据分析师，响应周期长（1~2天）

**痛点二：异常发现滞后**
- 广告投放异常（CTR 骤降/预算跑飞/转化率突变）发现时间平均滞后 2~4小时
- 传统监控依赖固定阈值告警，误报率高，运营容易「告警疲劳」
- 根因定位靠经验猜测，定位时间长

**痛点三：数据到洞察的链路断裂**
- 运营能看到数据下降，但不知道「为什么」
- 需要人工逐维度下钻（先看渠道，再看创意，再看人群），费时费力

**解决思路：** 自然语言提问 → NL2SQL 自动查询 → 异常检测 + 统计分析 → LLM 生成自然语言洞察报告

---

## 2. 系统架构

### 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                  广告数据智能分析 Agent                            │
│                                                                  │
│  用户输入                                                         │
│  「为什么今天上午 10点 ROI 下降了 30%？」                          │
│                   │                                              │
│                   ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Query Planner                           │  │
│  │  LLM 分析问题类型：                                         │  │
│  │  1. 诊断类（找原因）→ 触发多步下钻分析                       │  │
│  │  2. 查询类（查数据）→ 直接 NL2SQL                           │  │
│  │  3. 预测类（看趋势）→ 时序预测                               │  │
│  │  4. 对比类（A vs B）→ 分组聚合 + 差异分析                   │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   NL2SQL 模块                             │    │
│  │  Schema 注入 → Few-shot 示例 → SQL 生成 → 执行 → 验证     │    │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             ▼                                    │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │  ClickHouse  │  │  异常检测模块    │  │  归因分析模块         │ │
│  │  执行查询    │  │  Prophet/STL    │  │  维度贡献度计算       │ │
│  └──────┬──────┘  └────────┬────────┘  └──────────┬───────────┘ │
│         └─────────────────┬┘───────────────────────┘            │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │               LLM 分析报告生成                              │  │
│  │  结构化数据 + 统计结论 → 自然语言解读 → 可执行建议           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           ▼                                      │
│  输出：文字报告 + 数据表格 + 异常定位 + 优化建议                   │
└──────────────────────────────────────────────────────────────────┘
```

### 多步推理流程（Chain-of-Thought）

```
问题：「今天上午 10点 ROI 下降了 30%，原因是什么？」

Step 1: 确认下降事实
  SQL: SELECT hour, sum(revenue)/sum(cost) as roi
       FROM ad_stats WHERE date=today() GROUP BY hour
  结果：10点 ROI=210%，9点 ROI=301%，确实下降 30%

Step 2: 渠道维度下钻
  SQL: SELECT channel, sum(revenue)/sum(cost) as roi
       FROM ad_stats WHERE date=today() AND hour=10 GROUP BY channel
  结果：信息流 ROI=195%（-35%），搜索广告 ROI=298%（正常）
  → 问题聚焦在信息流渠道

Step 3: 创意维度下钻
  SQL: 在信息流下，按 creative_id 分组，对比 9点 vs 10点
  结果：creative_001 的 CVR 从 2.1% 降到 0.8%（-62%）
  → 具体创意异常

Step 4: 外部因素排查
  查询：竞争对手估算出价（平台竞价密度指标）
  结果：10点竞价密度上升 40%（大促期间竞争加剧）
  → 流量质量下降，非素材问题

Step 5: LLM 生成报告
  结论：「今日 10点 ROI 下降主要由信息流渠道的 creative_001 CVR 骤降引起，
         该时段竞价密度上升 40%，推测是竞品加价抢量导致本方获得的流量质量下降。
         建议：（1）对 creative_001 在信息流降价保 ROI；（2）检查落地页是否异常。」
```

---

## 3. NL2SQL 技术细节

### 3.1 Schema 注入策略

面对数百张表的 schema，不能全部注入 LLM（token 太多）：

```python
class SchemaRetriever:
    """动态检索相关 Schema，避免全量注入"""
    
    def __init__(self, schema_embeddings, table_descriptions):
        self.schema_embeddings = schema_embeddings
        self.descriptions = table_descriptions
    
    def get_relevant_tables(self, query: str, top_k: int = 5) -> list:
        query_embedding = embed(query)
        similarities = cosine_similarity(query_embedding, self.schema_embeddings)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.descriptions[i] for i in top_indices]

# Schema 描述格式（紧凑型，节省 token）
SCHEMA_TEMPLATE = """
表名: {table_name}
描述: {description}
关键字段:
{columns}
常用查询示例: {examples}
"""

# 实际注入示例
table_schema = """
表名: ad_impression_daily
描述: 广告曝光日汇总表，按广告组+日期聚合
关键字段:
  - date DATE: 日期
  - campaign_id STRING: 广告组ID
  - impressions INT64: 曝光次数
  - clicks INT64: 点击次数
  - conversions INT64: 转化次数
  - cost DECIMAL: 消耗金额（元）
  - revenue DECIMAL: 带来的收入（元）
常用查询: SELECT date, sum(clicks)/sum(impressions) as ctr FROM ad_impression_daily WHERE date >= today()-7 GROUP BY date
"""
```

### 3.2 少样本示例动态选择

```python
class FewShotSelector:
    """基于问题相似度动态选择最相关的 few-shot 示例"""
    
    def __init__(self, example_pool):
        # example_pool: [{question, sql, description}, ...]
        self.examples = example_pool
        self.example_embeddings = embed_batch([e['question'] for e in example_pool])
    
    def select(self, query: str, n: int = 3) -> list:
        query_emb = embed(query)
        scores = cosine_similarity(query_emb, self.example_embeddings)
        top_n = np.argsort(scores)[-n:][::-1]
        return [self.examples[i] for i in top_n]

# Few-shot 示例池（覆盖常见查询类型）
EXAMPLE_POOL = [
    {
        "question": "查询过去7天每天的点击率趋势",
        "sql": """
            SELECT date, 
                   sum(clicks) / sum(impressions) as ctr
            FROM ad_impression_daily
            WHERE date >= today() - 7
            GROUP BY date
            ORDER BY date
        """,
        "type": "trend_query"
    },
    {
        "question": "找出昨天 ROI 最差的5个广告组",
        "sql": """
            SELECT campaign_id,
                   sum(revenue) / sum(cost) as roi,
                   sum(cost) as total_cost
            FROM ad_impression_daily
            WHERE date = yesterday()
            GROUP BY campaign_id
            HAVING total_cost > 100
            ORDER BY roi ASC
            LIMIT 5
        """,
        "type": "ranking_query"
    },
    # ... 更多示例
]
```

### 3.3 SQL 执行安全机制

```python
class SafeSQLExecutor:
    
    FORBIDDEN_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
    MAX_RESULT_ROWS = 10000
    QUERY_TIMEOUT_SECONDS = 30
    
    def validate_sql(self, sql: str) -> tuple[bool, str]:
        sql_upper = sql.upper()
        
        # 只允许 SELECT 语句
        if not sql_upper.strip().startswith('SELECT'):
            return False, "只允许 SELECT 查询"
        
        # 禁止写操作关键词
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw in sql_upper:
                return False, f"包含禁止关键词: {kw}"
        
        # 字段名验证（防止访问不存在的表/字段）
        parsed = sqlparse.parse(sql)[0]
        tables = extract_tables(parsed)
        if not all(t in ALLOWED_TABLES for t in tables):
            return False, "包含未授权的表"
        
        return True, "OK"
    
    def execute_with_retry(self, sql: str, llm_client, max_retry: int = 3):
        for attempt in range(max_retry):
            valid, msg = self.validate_sql(sql)
            if not valid:
                # 让 LLM 修正 SQL
                sql = llm_client.fix_sql(sql, error_message=msg)
                continue
            
            try:
                result = clickhouse_client.execute(
                    sql,
                    settings={'max_execution_time': self.QUERY_TIMEOUT_SECONDS,
                              'max_result_rows': self.MAX_RESULT_ROWS}
                )
                return result
            except ClickHouseException as e:
                if attempt < max_retry - 1:
                    sql = llm_client.fix_sql(sql, error_message=str(e))
                else:
                    raise
```

### 3.4 幻觉检测

```python
def validate_sql_hallucination(sql: str, schema: dict) -> list:
    """检测 SQL 中是否引用了不存在的字段或表"""
    errors = []
    parsed = parse_sql(sql)
    
    # 检查字段是否存在
    for table, columns in parsed.table_columns.items():
        if table not in schema:
            errors.append(f"表 '{table}' 不存在")
            continue
        for col in columns:
            if col not in schema[table]['columns']:
                errors.append(f"字段 '{table}.{col}' 不存在")
    
    # 检查逻辑一致性
    if 'GROUP BY' in sql.upper() and parsed.has_non_aggregate_select:
        errors.append("SELECT 中有非聚合字段但未在 GROUP BY 中")
    
    return errors
```

---

## 4. 异常检测算法

### 4.1 时序异常检测

**方案一：Prophet 分解**

```python
from prophet import Prophet

def detect_anomaly_with_prophet(metric_series: pd.DataFrame) -> dict:
    """
    metric_series: columns=[ds(日期), y(指标值)]
    返回：异常时间点列表 + 异常程度
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05  # 变点检测灵敏度
    )
    model.fit(metric_series[:-48])  # 用最近48小时前的数据训练
    
    # 预测并计算置信区间
    future = model.make_future_dataframe(periods=48, freq='H')
    forecast = model.predict(future)
    
    # 检测超出置信区间的点
    actual = metric_series.tail(48)
    anomalies = []
    for _, row in actual.iterrows():
        pred = forecast[forecast.ds == row.ds]
        if row.y < pred.yhat_lower.values[0] or row.y > pred.yhat_upper.values[0]:
            anomaly_score = abs(row.y - pred.yhat.values[0]) / pred.yhat.values[0]
            anomalies.append({'time': row.ds, 'actual': row.y,
                              'expected': pred.yhat.values[0],
                              'score': anomaly_score})
    return anomalies
```

**方案二：STL 分解 + 3σ 规则**

STL 将时序分解为趋势 + 季节性 + 残差：

$$
y_t = T_t + S_t + R_t
$$

对残差 $R_t$ 应用 3σ 规则：

$$
\text{anomaly} = \left| R_t - \bar{R} \right| > 3\sigma_R
$$

```python
from statsmodels.tsa.seasonal import STL

def detect_anomaly_stl(series, period=24):  # period=24（小时级数据，24小时为一周期）
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    residuals = result.resid
    
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    
    anomaly_mask = np.abs(residuals - mean_r) > 3 * std_r
    return anomaly_mask, result.trend, result.seasonal
```

### 4.2 多维下钻归因

**差值分析法（Contribution Analysis）：**

问题：指标 M（如 ROI）从 T-1 下降到 T，找出哪个维度贡献最大？

$$
\Delta M = M_T - M_{T-1} = \sum_{d} \text{contribution}(d)
$$

维度 $d$（如渠道、创意）的贡献度：

$$
\text{contribution}(d) = \frac{\text{cost}_d}{\text{total\_cost}} \times (ROI_d^T - ROI_d^{T-1})
$$

```python
def contribution_analysis(df_t: pd.DataFrame, df_t1: pd.DataFrame,
                          dimension: str, metric: str = 'roi') -> pd.DataFrame:
    """
    df_t, df_t1: 当前时刻和前一时刻的数据，按维度分组
    返回：每个维度值对总体指标变化的贡献度
    """
    merged = df_t.merge(df_t1, on=dimension, suffixes=['_cur', '_prev'])
    total_cost = merged['cost_cur'].sum()
    
    merged['weight'] = merged['cost_cur'] / total_cost
    merged['delta_metric'] = merged[f'{metric}_cur'] - merged[f'{metric}_prev']
    merged['contribution'] = merged['weight'] * merged['delta_metric']
    
    # 按贡献度（绝对值）排序，找出影响最大的维度
    result = merged[[dimension, 'contribution', f'{metric}_cur', f'{metric}_prev']]
    result = result.sort_values('contribution', key=abs, ascending=False)
    
    return result
```

### 4.3 LLM 增强归因

```python
def llm_attribution(statistical_results: dict, context: dict) -> str:
    """
    将统计分析结果转化为 LLM 可理解的格式，生成自然语言归因
    """
    prompt = f"""
你是广告数据分析专家。以下是对 ROI 下降问题的统计分析结果：

## 分析背景
- 时间：{context['time']}
- ROI 变化：{context['roi_change']}%（从 {context['roi_before']} 降到 {context['roi_after']}）

## 维度分析结果
{format_contribution_results(statistical_results['contributions'])}

## 异常检测结果
{format_anomalies(statistical_results['anomalies'])}

## 外部因素
- 竞价密度变化：{context['auction_density_change']}%
- 节假日/大促：{context['is_event_day']}

请：
1. 给出最可能的根因（按置信度排序，最多3个）
2. 区分「确定的」和「推测的」原因
3. 给出 2~3个可执行的优化建议
4. 标注哪些结论需要进一步数据验证

注意：只基于提供的数据给出结论，不要凭空假设未经数据支持的原因。
"""
    return llm.generate(prompt)
```

---

## 5. 数据规模与优化

### ClickHouse 表设计

```sql
-- 广告曝光日志原始表（每天 100亿行）
CREATE TABLE ad_impressions (
    date          Date,
    datetime      DateTime,
    campaign_id   String,
    creative_id   String,
    user_id       UInt64,
    channel       LowCardinality(String),    -- 低基数用 LowCardinality 节省存储
    is_click      UInt8,
    is_convert    UInt8,
    cost          Decimal32(4),
    revenue       Decimal32(4)
)
ENGINE = MergeTree()
PARTITION BY date                             -- 按日分区，便于裁剪
ORDER BY (campaign_id, datetime)              -- 排序键，加速广告组维度查询
TTL date + INTERVAL 90 DAY                   -- 90天自动过期删除

-- 预聚合物化视图（按小时 + 广告组维度）
CREATE MATERIALIZED VIEW ad_stats_hourly
ENGINE = SummingMergeTree()
PARTITION BY date
ORDER BY (date, hour, campaign_id, channel)
AS SELECT
    date,
    toHour(datetime) as hour,
    campaign_id,
    channel,
    sum(1) as impressions,
    sum(is_click) as clicks,
    sum(is_convert) as conversions,
    sum(cost) as total_cost,
    sum(revenue) as total_revenue
FROM ad_impressions
GROUP BY date, toHour(datetime), campaign_id, channel
```

### 查询性能

```
查询类型和预期延迟：

简单聚合查询（单表，有索引）：< 1秒
  示例：今日某广告组的 CTR
  SELECT sum(clicks)/sum(impressions) FROM ad_stats_hourly
  WHERE date=today() AND campaign_id='xxx'

中等复杂度（多维聚合，时间范围 7天内）：1~5秒
  示例：过去7天每日 ROI 趋势
  SELECT date, sum(revenue)/sum(cost) FROM ad_stats_hourly
  WHERE date >= today()-7 GROUP BY date

复杂分析（原始日志 + 多表 JOIN）：10~30秒
  示例：分析某创意的用户画像分布
  需要访问原始 ad_impressions + 用户属性表

优化手段：
  1. 预聚合物化视图：覆盖 80% 的常见查询，延迟降低 10倍
  2. 分区裁剪：WHERE date= 让 ClickHouse 只扫描相关分区
  3. 结果缓存（Redis）：相同查询 24小时内直接返回缓存结果
  4. 查询改写：LLM 生成的 SQL 经规则引擎优化（自动添加分区条件）
```

---

## 6. 面试高频考点

**Q：NL2SQL 准确率怎么评估？如何处理歧义查询？**

A：
- **准确率评估**：
  - Execution Accuracy（执行准确率）：生成的 SQL 执行结果与标准 SQL 结果一致的比例
  - Exact Match（精确匹配）：生成 SQL 与标准 SQL 完全一致（较严格，实用性低）
  - 线上评估：用户对结果的显式反馈（点赞/点踩）
  - 实测：在广告领域特定数据集上，GPT-4 配合 Schema + Few-shot 可达 70~80% 的 Execution Accuracy
- **歧义处理**：
  - 检测歧义：LLM 先判断问题是否有歧义（"昨天的数据"是哪个时区？"ROI"是哪种计算口径？）
  - 主动澄清：有歧义时返回澄清问题，而非猜测执行
  - 默认假设：对常见歧义建立默认规则（如默认时区、默认 ROI 计算公式），并在结果中展示假设

---

**Q：如何防止用户通过自然语言注入恶意 SQL？**

A：
1. **只读权限**：数据库用户权限配置为只读（GRANT SELECT），从根本上防止写操作
2. **SQL 白名单验证**：只允许 SELECT 开头的语句，检测 DROP/DELETE 等危险关键词
3. **LLM Prompt 隔离**：系统 Prompt 中明确声明"只生成 SELECT 查询"，用户输入作为数据传入，不直接拼接到 SQL
4. **参数化查询**：用户提供的具体值（如广告主 ID）通过参数化传入，不嵌入 SQL 字符串

---

**Q：多步推理中，中间步骤出错了怎么办？有没有回溯机制？**

A：
1. **错误检测**：每步执行后验证结果合理性（如查询结果为空、数值异常）
2. **自动修复**：SQL 报错 → 将错误信息反馈给 LLM 重新生成（最多 3次）
3. **回溯策略**：若某步骤执行失败且无法修复，回溯到上一个成功的步骤，改变推理路径（类似 Tree-of-Thought）
4. **部分失败容忍**：4步分析中 1步失败，仍可基于其他 3步的结果给出部分结论，并标注"以下分析因数据获取失败可能不完整"
5. **Human Escalation**：超过 3次重试仍失败，将问题转给人工数据分析师，并附上已完成的分析步骤

---

**Q：对比传统 BI 工具（如 Tableau），这个方案的优势和劣势？**

A：

| 维度 | 传统 BI | LLM 智能分析 |
|------|---------|------------|
| 使用门槛 | 需要学习 BI 工具，SQL 能力有要求 | 自然语言提问，零门槛 |
| 灵活性 | 受限于预定义报表和维度 | 可回答任意临时问题 |
| 分析深度 | 展示数据，不解释原因 | 自动下钻归因，给出解读 |
| 异常发现 | 需要人工设置告警规则 | 自动检测异常，智能归因 |
| 成本 | 低（BI 工具授权费） | 较高（LLM API 调用费） |
| 准确性 | 100%（执行确定性 SQL） | ~75~85%（NL2SQL 有失误） |
| 可解释性 | 数据透明 | 需要展示中间 SQL 增加可信度 |

---

**Q：如何做冷启动（新广告主问自己的数据）？**

A：
1. **Schema 快速感知**：为新广告主的数据表生成自动描述，加入 Schema 库
2. **元数据引导**：先让系统探索该广告主的数据结构（有哪些广告组、哪些渠道），再回答具体问题
3. **通用模板迁移**：新广告主的问题大多类似（"昨天花了多少钱/ROI 多少"），用通用 SQL 模板即可覆盖
4. **渐进式个性化**：收集该广告主使用过的查询和反馈，逐步建立专属 few-shot 示例库

---

**Q：LLM 生成的分析结论不准确时如何处理？**

A：
1. **展示数据来源**：每个结论都附上对应的 SQL 和查询结果（可验证）
2. **置信度标注**：区分「数据直接支持」和「LLM 推断」，用不同标识区分
3. **反馈机制**：用户对结论点踩 → 记录为负样本 → 用于 Prompt 优化和 Fine-tuning
4. **不确定性提示**：LLM 在无充分数据支撑时，主动说"数据不足，无法确定，建议进一步查看XX"

---

## 7. 项目效果（量化指标）

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|------|------|---------|
| 日常数据分析时间 | 2~3小时/天 | 20~30分钟/天 | 减少 85% |
| 异常发现响应时间 | 2~4小时 | 15~30分钟 | 减少 87% |
| 运营人员自助查数率 | ~30%（其余依赖 DA）| ~80% | 提升 50pp |
| 数据分析师处理 ad-hoc 需求 | 40%工时用于取数 | 10%工时 | 释放 30% 工时 |
| NL2SQL 准确率 | - | ~78% | 基于内部测评集 |

---

## 8. 技术亮点总结（面试用）

1. **多步推理 + 动态规划**：不是一次 NL2SQL，而是根据问题类型设计多步分析链路
2. **归因量化**：维度贡献度公式，将「哪里有问题」变成「谁的责任」，可量化
3. **安全第一**：只读权限 + 字段验证 + 参数化查询三重防护，防止 SQL 注入
4. **LLM 增强统计**：统计模型（Prophet/STL）+ LLM 的互补，避免纯 LLM 的不可靠
5. **工程化落地**：预聚合物化视图、结果缓存、查询超时保护，可在生产环境稳定运行

---

*文档版本：v1.0 | 适用场景：搜广推算法工程师面试 | 业务规模：中小型广告平台*
