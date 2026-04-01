# 广告投放决策 Agent：基于 LLM 的目标驱动自动投放优化系统

> 一句话定位：用 LLM Agent 替代人工经验，实现「设定目标→自动分析→执行优化」的闭环广告投放决策系统

**核心技术栈：** LLM Agent | ReAct Framework | Function Calling | ClickHouse | Redis | oCPC/oCPA | PID Control | Human-in-the-loop

---

## 1. 项目背景与痛点

### 业务规模

| 指标 | 规模 |
|------|------|
| 广告主数量 | 1万~5万（中小平台） |
| 广告组总数 | 10万~50万 |
| 每日新增广告 | 8000~10000条 |
| 峰值 QPS | 2万~5万 |

### 核心痛点

**痛点一：人工经验依赖严重**
- 广告优化师凭经验调整出价，策略难以复用和沉淀
- 新手优化师学习成本高，老手离职导致策略断层
- 同一广告主的优化策略无法跨账户共享

**痛点二：调整频率不足**
- 人工调整频率：每天 1~2次
- 广告市场变化频率：分钟级（竞争对手出价实时变化）
- 结果：广告主的 ROI 目标在高峰期未能达成，低峰期过度消耗预算

**痛点三：多约束难以兼顾**
- 同时满足：ROI ≥ 目标值 + 日预算消耗完 + 关键时段跑量
- 人工调整往往顾此失彼，无法同时优化多个目标

**核心问题：** 能否用 LLM Agent 模拟「资深优化师」的决策过程，实现 24x7 自动优化？

---

## 2. 系统架构设计

### 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     广告投放决策 Agent                         │
│                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────┐    │
│  │  广告主目标  │    │           LLM Planner              │    │
│  │  ROI >= 300%│───▶│  (GPT-4 / Claude API)             │    │
│  │  日预算 1万  │    │  Thought → Action → Observation   │    │
│  └─────────────┘    └──────────┬───────────────────────┘    │
│                                │ Function Calling             │
│            ┌───────────────────┼───────────────┐             │
│            ▼                   ▼               ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  数据查询工具 │  │  出价调整工具 │  │   预算分配工具    │   │
│  │  (ClickHouse)│  │  (Ads API)   │  │   (Budget API)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    Memory 模块                          │  │
│  │  短期: 当前对话上下文 | 中期: 投放周期历史 | 长期: 偏好  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Human-in-the-Loop 安全层                   │  │
│  │  高风险操作（出价变动>50%）→ 人工确认 → 执行/拒绝        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 调度流程

```
每小时触发一次 Agent 决策循环：

广告主目标设定
       │
       ▼
  OODA 循环开始
  ┌────────────┐
  │  Observe   │ ←── 查询过去1小时的曝光/点击/转化/消耗数据
  └─────┬──────┘
        ▼
  ┌────────────┐
  │  Orient    │ ←── LLM 分析：当前 ROI 趋势/预算剩余/时段特征
  └─────┬──────┘
        ▼
  ┌────────────┐
  │  Decide    │ ←── LLM 生成调整方案（出价/预算/定向）
  └─────┬──────┘
        ▼
  ┌────────────┐
  │   Act      │ ←── 安全验证 → 人工确认（必要时）→ API 执行
  └────────────┘
        │
        └──── 记录到 Memory，等待下次循环
```

### 核心组件详解

**Memory 模块设计**

| 层次 | 存储内容 | 存储介质 | 保留时间 |
|------|---------|---------|---------|
| 短期记忆 | 当前对话上下文，最近 10次 Action | In-context (LLM) | 单次会话 |
| 中期记忆 | 本次投放周期的所有决策和效果 | Redis（TTL 30天） | 1个投放周期 |
| 长期记忆 | 广告主的偏好策略、历史经验规律 | PostgreSQL | 永久 |

**Human-in-the-Loop 触发条件**

```python
HIGH_RISK_CONDITIONS = {
    "bid_change_ratio": 0.5,      # 出价变动超过 50% 需确认
    "budget_cut_ratio": 0.3,      # 预算削减超过 30% 需确认
    "new_targeting": True,        # 新增定向人群需确认
    "daily_budget_exceed": True,  # 超出日预算上限需确认
}
```

---

## 3. 核心算法细节

### 3.1 ReAct 框架实现

ReAct（Reasoning + Acting）是 Agent 的核心推理范式：

```
循环执行直到目标达成或达到最大迭代次数：
  Thought:  分析当前状态，推断下一步行动
  Action:   调用工具（Function Calling）
  Observation: 获取工具执行结果
  → 判断是否需要继续循环
```

**实际运行示例：**

```
Thought: 当前 ROI 为 280%，低于目标 300%。过去 1 小时点击率正常但转化率
         下降了 15%。可能原因：竞争加剧导致流量质量下降，或落地页问题。
         先查看各广告组的具体数据。

Action: query_campaign_metrics(campaign_id="all", time_range="1h",
        metrics=["ctr", "cvr", "roi", "cost"])

Observation: {
  "camp_001": {"ctr": 0.032, "cvr": 0.018, "roi": 320, "cost": 3200},
  "camp_002": {"ctr": 0.028, "cvr": 0.009, "roi": 201, "cost": 4800},
  "camp_003": {"ctr": 0.041, "cvr": 0.022, "roi": 380, "cost": 1200}
}

Thought: camp_002 的 CVR 只有 0.9%，ROI 仅 201%，是拖累整体 ROI 的主因。
         camp_003 表现优秀，ROI 380%，但消耗只有 1200 元。
         建议：降低 camp_002 出价，提高 camp_003 预算。

Action: adjust_bid(campaign_id="camp_002", new_bid=0.8*current_bid,
        reason="CVR 显著低于均值，ROI 低于目标，降价减少低质流量")

Observation: {"status": "success", "new_bid": 4.8, "old_bid": 6.0}

Action: reallocate_budget(allocation=[
  {"campaign_id": "camp_003", "budget_ratio": 0.35},
  {"campaign_id": "camp_001", "budget_ratio": 0.40},
  {"campaign_id": "camp_002", "budget_ratio": 0.25}
])
```

### 3.2 Function Calling 格式设计

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_campaign_metrics",
            "description": "查询广告组的实时指标数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": "广告组 ID，传 'all' 查询所有"
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["1h", "6h", "24h", "7d"],
                        "description": "查询时间范围"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要的指标列表"
                    }
                },
                "required": ["campaign_id", "time_range"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_bid",
            "description": "调整广告出价，单次调整幅度不超过 30%",
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_id": {"type": "string"},
                    "new_bid": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 500.0
                    },
                    "reason": {
                        "type": "string",
                        "description": "必须提供调整原因，用于审计日志"
                    }
                },
                "required": ["campaign_id", "new_bid", "reason"]
            }
        }
    }
]
```

### 3.3 防幻觉机制

LLM 可能产生超出合理范围的建议，需要多层防护：

**第一层：输出格式验证**
```python
def validate_bid_action(action: dict) -> bool:
    bid = action.get("new_bid", 0)
    campaign_id = action.get("campaign_id", "")
    
    # 字段完整性检查
    if not campaign_id or bid <= 0:
        return False, "参数缺失"
    
    # 出价范围检查（基于历史数据的合理范围）
    current_bid = get_current_bid(campaign_id)
    if bid > current_bid * 1.3 or bid < current_bid * 0.3:
        return False, f"出价变动幅度异常: {bid} vs {current_bid}"
    
    # 绝对值范围检查
    if bid < MIN_BID or bid > MAX_BID:
        return False, f"出价超出平台允许范围 [{MIN_BID}, {MAX_BID}]"
    
    return True, "验证通过"
```

**第二层：多数投票（对高风险决策）**
```python
# 对于大额预算调整，调用 LLM 3次，取多数一致的决策
def robust_decision(context, n=3):
    decisions = [llm_decide(context) for _ in range(n)]
    # 如果3次决策不一致，降级到保守策略
    if not decisions_consistent(decisions):
        return conservative_fallback()
    return majority_vote(decisions)
```

**第三层：规则引擎兜底**
```python
# 强硬规则，不可被 LLM 覆盖
HARD_RULES = [
    "daily_cost <= daily_budget",           # 绝不超预算
    "bid >= platform_min_bid",              # 不低于平台最低出价
    "single_adjustment <= 30%",             # 单次调整幅度限制
    "adjustment_interval >= 30min",         # 调整频率限制
]
```

---

## 4. 出价优化算法

### 4.1 CVR/CTR 预估模型

基于历史数据训练的轻量级预估模型（非实时 RTB，用于策略层）：

$$
\text{CTR}}_{\text{{pred}} = \sigma\left(\mathbf{w}^T \mathbf{x}_{ad} + \mathbf{v}^T \mathbf{x}_{user} + b\right)
$$

其中特征 $\mathbf{x}$ 包括：广告类目、历史点击率、用户兴趣标签、时段特征

$$
\text{CVR}}_{\text{{pred}} = \frac{\text{转化次数}}{\text{点击次数}} \cdot \text{平滑系数}
$$

平滑处理（防止小样本过拟合）：

$$
\text{CVR}}_{\text{{smooth}} = \frac{n \cdot \text{CVR}}_{\text{{obs}} + m \cdot \text{CVR}}_{\text{{prior}}}{n + m}
$$

其中 $n$ 为样本量，$m$ 为平滑参数（通常取 10~50），$\text{CVR}}_{\text{{prior}}$ 为类目均值

### 4.2 传统 PID 控制器的局限性

传统 PID 出价调整：

$$
\text{bid}}_{\text{{t+1}} = \text{bid}}_{\text{t + K}}_{\text{p \cdot e}}_{\text{t + K}}_{\text{i \sum}}_{\text{{j=0}}^{t} e_j + K_d (e_t - e_{t-1})
$$

其中误差 $e_t = \text{ROI}}_{\text{{target}} - \text{ROI}}_{\text{{actual}}$

**PID 的局限性：**
1. 只对单一指标（如 ROI）进行控制，无法处理多约束
2. 参数 $K_p, K_i, K_d$ 需要人工调优，不同广告主需要不同参数
3. 无法理解「为什么 ROI 下降」，只会机械调价
4. 对突发事件（竞品促销、节假日）反应迟钝

**LLM 的改进：**
- 理解因果关系：分析 ROI 下降是竞争加剧还是素材疲劳
- 多约束协调：同时兼顾 ROI、预算消耗率、跑量目标
- 上下文感知：节假日/大促期间自动调整策略

### 4.3 多约束出价优化

目标函数：

$$
\max_{\text{bid}} \quad \text{Conversions}(\text{bid})
$$

约束条件：

$$
\text{ROI}(\text{bid}) = \frac{\text{Revenue}}{\text{Cost}} \geq \text{ROI}}_{\text{{target}}
$$

$$
\text{Cost}(\text{bid}) \leq \text{Budget}}_{\text{{daily}}
$$

$$
\text{Impressions}(\text{bid}) \geq \text{Volume}}_{\text{{min}} \quad \text{（跑量约束）}
$$

### 4.4 oCPC/oCPA 目标出价公式

oCPC（目标转化出价）：

$$
\text{出价}}_{\text{{\text{oCPC}}} = \text{目标CPA} \times \text{预估CVR}
$$

$$
\text{出价}}_{\text{{\text{oCPC}}} = \text{CPA}}_{\text{{target}} \times p(\text{convert} | \text{click})
$$

oCPA（目标行为出价，平台侧竞价）：

$$
\text{ecpm} = \text{出价}}_{\text{{\text{oCPA}}} \times \text{pCVR} \times 1000
$$

其中 $\text{ecpm}$ 用于广告竞价排序，平台保证广告主按 CPA 计费

**LLM Agent 的作用：**
- 根据历史数据动态调整 CPA 目标值
- 当前 CVR 高于预期时，适当提高出价抢量
- 当前 CVR 低于预期时，降低出价保 ROI

---

## 5. 计算资源估算

### LLM API 成本

```
每次决策消耗：
  - 输入 tokens：~1500（系统 prompt + 历史上下文 + 数据）
  - 输出 tokens：~500（分析 + 决策 + 工具调用参数）
  - 合计：~2000 tokens/次

调用频率：
  - 每位广告主每小时 1次决策
  - 活跃广告主数量：5000（高峰期）
  - 峰值调用：5000次/小时 = 83次/分钟

成本估算（GPT-4 Turbo）：
  - 输入：$0.01/1K tokens × 1.5K × 5000次 = $75/天
  - 输出：$0.03/1K tokens × 0.5K × 5000次 = $75/天
  - 合计：约 $150/天
  
优化后（批量调用 + Prompt 压缩）：约 $50~80/天
```

### 数据层架构

```
数据流向：
广告日志 → Kafka → Flink 实时计算 → Redis（实时指标，秒级）
                 → ClickHouse（历史分析，分钟级聚合）
                 
Redis 存储：
  - 当前出价、预算剩余、过去 1小时指标
  - 内存占用：约 10GB（5万广告主 × 200字节/广告主）
  - 读取延迟：< 1ms

ClickHouse 查询：
  - 7天历史数据，按广告组聚合
  - 查询延迟：< 100ms（使用物化视图）
```

### 端到端延迟分析

```
一次完整决策的时间分解：
  数据查询（Redis + ClickHouse）：100ms
  LLM API 调用（包括网络往返）：2000~5000ms
  安全验证 + 格式解析：50ms
  人工确认等待（高风险操作）：0ms（异步，不阻塞常规流程）
  广告 API 执行：200ms
  ─────────────────────────────
  合计：约 2.5~5.5 秒

结论：适合策略层（分钟级/小时级决策），
      不适合 RTB（要求 < 50ms 响应）
```

---

## 6. 优化方案

### 6.1 Prompt 压缩

```python
# 原始 Prompt（冗长版）
raw_prompt = """
广告组 camp_001 在过去 24 小时内的数据如下：
- 曝光次数：120,000 次
- 点击次数：3,840 次
- 点击率（CTR）：3.2%
- 转化次数：69 次
- 转化率（CVR）：1.8%
- 总消耗：3,200 元
- 总收入：9,660 元
- ROI：301.875%
...（10个广告组各需要这样描述）
"""

# 压缩版 Prompt（结构化压缩）
compressed_prompt = """
[当前时刻 14:00, 距天结束 10h, 剩余预算 6800元]
广告组数据（格式：ID|imp|clk|ctr|cvr|cost|roi）：
camp_001|120k|3.8k|3.2%|1.8%|3200|302%
camp_002|95k|2.7k|2.8%|0.9%|4800|201%
camp_003|58k|2.4k|4.1%|2.2%|1200|380%
目标：ROI>=300%, 今日预算消耗率>=90%
"""
# Token 节省：约 60%
```

### 6.2 批量决策

```python
# 将多个广告主的优化请求合并为一次 LLM 调用
def batch_optimize(advertiser_list, batch_size=10):
    batches = chunk(advertiser_list, batch_size)
    for batch in batches:
        combined_context = format_batch_context(batch)
        decisions = llm.batch_decide(combined_context)
        for advertiser, decision in zip(batch, decisions):
            execute_decision(advertiser, decision)

# 节省效果：10个广告主合并 → Token 节省约 50%（共享 system prompt）
```

### 6.3 缓存策略

```python
# 场景相似度判断（基于规则的快速匹配）
def check_cache(current_state):
    cache_key = compute_state_hash(
        roi_bucket=bucket(current_state.roi, [250, 280, 300, 320]),
        budget_remaining_ratio=bucket(current_state.budget_ratio, [0.1, 0.3, 0.5, 0.8]),
        time_of_day=current_state.hour // 3,  # 3小时为一个时段
    )
    if cache_key in decision_cache:
        return decision_cache[cache_key]
    return None
```

### 6.4 降级策略

```
LLM 服务异常降级链：
Level 1: LLM API 超时重试（3次，间隔 1s）
Level 2: 切换备用 LLM（Claude → GPT-4 → Gemini）
Level 3: 使用缓存的上次决策
Level 4: 降级到规则引擎
  - ROI < 250%：出价降低 10%
  - ROI > 350%：出价提高 10%
  - 预算消耗率 < 50% 且距天结束 < 4h：出价提高 15%
```

---

## 7. 面试高频考点

**Q：Agent 如何保证决策的安全性，不会乱改出价？**

A：三层安全机制：
1. **硬约束层**：规则引擎强制执行，不可被 LLM 覆盖。包括：出价范围绑定（不低于 30% 当前值，不超过 130%），调整频率限制（每30分钟最多1次），绝对值上下限（平台最低出价~平台最高出价）
2. **输出验证层**：解析 LLM 输出的 JSON，验证字段合法性、值域合理性，任何不合格的指令直接拦截并记录
3. **人工审核层**：变动幅度超过 50%、涉及新定向策略、日消耗超过历史均值 2倍等高风险操作，需发送钉钉/邮件等待人工确认，超时自动降级到保守策略

---

**Q：如何评估 Agent 的决策质量？用什么指标？**

A：
- **在线指标**：与历史同期（或对照组）对比，ROI 提升比例、预算达成率、优化师介入频率降低比例
- **决策质量指标**：决策一致性（相同场景下建议的方差）、Regret（实际效果 vs 最优策略的差距）
- **操作安全指标**：被安全层拦截的操作比例、人工覆盖 Agent 决策的频率
- **A/B 测试**：随机将广告主分为 Agent 组和人工组，对比 30 天的 ROI 和预算达成率

---

**Q：Memory 模块怎么设计？短期/长期记忆如何切换？**

A：
- **短期记忆**（In-context）：直接放在 LLM 的上下文窗口里，包含当次对话的所有 Thought/Action/Observation，窗口满了之后进行摘要压缩（Summarize）后存入中期记忆
- **中期记忆**（Redis）：当次投放周期（通常是1个月）的决策历史摘要。查询时通过相似度检索最近 5条相关决策注入 Prompt
- **长期记忆**（PostgreSQL）：广告主的偏好规则（如"XX广告主倾向于保 ROI 而非冲量"）、有效策略模板、异常事件记录。每次大促/季报时人工审核更新

---

**Q：如果 LLM 给出了不合理的建议，系统如何处理？**

A：
1. **格式层**：JSON Schema 验证失败 → 要求 LLM 重新生成（最多 3次）
2. **值域层**：出价超出安全范围 → 截断到安全值，记录警告日志
3. **逻辑层**：如建议同时提价 + 降预算（自相矛盾）→ 触发二次验证请求，要求 LLM 解释矛盾
4. **后验层**：执行后 30 分钟评估效果，若 ROI 显著恶化 → 自动回滚 + 标记该决策为负样本，用于后续 Prompt 优化

---

**Q：这个方案和传统规则引擎相比优势在哪里？**

A：

| 维度 | 规则引擎 | LLM Agent |
|------|---------|-----------|
| 策略覆盖 | 需要人工穷举规则 | 自然语言描述目标，泛化能力强 |
| 上下文理解 | 无法理解"为什么" | 可推断因果，给出可解释的理由 |
| 多约束 | 规则冲突难处理 | 整体优化，自然权衡 |
| 新场景适应 | 需要人工新增规则 | 零样本泛化 |
| 可维护性 | 规则数量爆炸（千条以上难维护） | Prompt 修改即可调整策略 |

---

**Q：如何处理冷启动（新广告主没有历史数据）？**

A：
1. **类目相似迁移**：从同行业、同规模广告主的成功策略中提取初始参数
2. **分阶段策略**：前 3天使用保守的固定出价策略收集数据，第 4天开始 Agent 接管
3. **Meta-learning 思路**：将广告主特征（行业、预算量级、历史平台表现）作为输入，让 LLM 推断适合的初始策略
4. **兜底设置**：平台建议的行业均值出价 × 系数（通常 1.0~1.2倍）作为启动出价

---

## 8. 项目效果（量化指标）

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|------|------|---------|
| 广告主平均 ROI | 基线 | +15%~25% | 行业参考：头条/快手类似系统 10~20% |
| 日预算达成率 | ~70% | ~85% | +15pp |
| 人工调整频次 | 每天 2~3次/人 | 每天 0.5次/人 | 减少 75% |
| 异常预警响应时间 | 2~4小时 | 15分钟 | 减少 87% |
| 广告主留存率（月） | 基线 | +8%~12% | 间接效果 |

**注意：** 实际效果因业务场景差异较大，以上数据基于行业公开报告（字节跳动、百度等的 AutoBidding 系统论文中的参考范围）。

---

## 9. 技术亮点总结（面试用）

1. **ReAct + OODA 融合**：不是简单调用 LLM，而是设计了多轮推理-验证-执行的完整循环
2. **安全优先设计**：三层防护机制，确保 LLM 幻觉不会造成真实损失
3. **成本工程**：通过 Prompt 压缩 + 批量调用，将 API 成本控制在合理范围
4. **优雅降级**：LLM 不可用时无缝切换规则引擎，系统零中断
5. **可解释决策**：每次调整都有 LLM 生成的原因说明，便于广告主理解和信任

---

*文档版本：v1.0 | 适用场景：搜广推算法工程师面试 | 业务规模：中小型广告平台*
