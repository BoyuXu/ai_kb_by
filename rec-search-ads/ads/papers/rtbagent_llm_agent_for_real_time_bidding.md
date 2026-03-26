# RTBAgent: LLM Agent for Real-Time Bidding
> 来源：arxiv/2404.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
实时竞价（RTB）系统的智能化决策挑战：
- 传统 RTB 策略（线性出价/规则）无法处理复杂市场动态
- RL 方法需要大量在线探索，风险高
- 广告主需要对竞价策略有解释性和可控性
- 多个广告主同时竞价，需要博弈感知

## 核心方法与创新点
**RTBAgent**：基于 LLM 的自主竞价 Agent，具备工具调用能力。

**Agent 架构：**
```
RTBAgent = LLM_Brain + Tools + Memory + Planner

工具集：
  - market_analyzer: 分析当前市场竞价环境
  - budget_calculator: 计算剩余预算和消耗速率
  - cvr_predictor: 预测当前请求的转化率
  - bid_executor: 执行出价
  - strategy_reviewer: 回顾历史策略效果

记忆：
  - 短期记忆：当前 session 的竞价历史
  - 长期记忆：历史策略效果库（RAG）

规划：
  - 高层目标：日预算×目标ROI
  - 低层决策：每个竞价请求的出价
```

**ReAct（Reason + Act）出价流程：**
```
Observation: 当前竞价请求特征
Thought: "当前CTR预估0.03，CVR 2.5%，预计每转化成本=bid/0.03/0.025=133×bid元；
          目标CPA=50元，因此最高出价=50×0.03×0.025×1000=0.375元（CPM）"
Action: bid_executor(0.35)  # 略低于最高出价，留余量
Observation: 竞价结果（赢/输），记录到 memory
```

**多轮策略调整：**
```python
# 每小时检查一次策略效果
review = strategy_reviewer.check({
    "current_cpa": actual_cpa,
    "target_cpa": target_cpa,
    "budget_used": spent / total_budget,
    "time_elapsed": hours / 24
})
# LLM 根据 review 调整出价系数
adjustment = llm.decide_adjustment(review)
```

## 实验结论
- 模拟竞价环境（iPinYou 数据集）：
  - 转化量 +11.3%（vs 固定出价）
  - CPA 达标率 91.2%（vs RL: 83.4%）
  - 预算消耗率 98.1%（vs 规则策略 76%）
- 可解释性评估：广告主对 RTBAgent 决策推理过程满意度 4.2/5

## 工程落地要点
1. **延迟约束**：RTB 要求 <100ms，LLM 推理约 500ms → 缓存策略模板，实时快速调用
2. **工具调用缓存**：cvr_predictor/market_analyzer 结果缓存（TTL 1min）
3. **安全出价层**：LLM 输出的出价经规则层 Clip，确保不超预算上限
4. **批量决策**：将连续 100ms 内的竞价请求批量处理，减少 LLM 调用次数
5. **异步调整**：策略审查（hourly）异步执行，不影响实时竞价 latency

## 面试考点
**Q1: RTBAgent 相比传统 RTB 方法的核心优势？**
A: 推理能力：RTBAgent 能结合市场分析、预算状态、转化预测进行多步推理，做出更合理的出价决策。同时具备可解释性（能输出推理过程）和可控性（广告主可通过自然语言调整策略目标）。

**Q2: LLM Agent 在 RTB 场景中最大的工程挑战？**
A: 延迟：RTB 要求 100ms 内响应，LLM 推理需 300-1000ms。解决：①策略缓存：预计算常见场景的出价策略 ②轻量化 LLM：7B 量化模型 ③Speculative Execution：并行预测最可能的几种场景。

**Q3: RTBAgent 的 Memory 模块如何设计？**
A: 短期记忆（当前 session）：循环缓冲区存储最近 N 次竞价的（特征, 出价, 结果）三元组。长期记忆（历史策略）：向量数据库存储历史高效策略，新决策前 RAG 检索相似情境的历史策略作为参考。

**Q4: 如何确保 RTBAgent 满足预算约束不超支？**
A: 多层保护：①LLM Prompt 中明确当前剩余预算 ②硬限制：bid × predicted_requests > remaining_budget 时强制降价 ③速率控制：实时监控消耗速率，预计超支时触发保守出价模式 ④熔断机制：预算消耗率 >110% 暂停竞价。

**Q5: 博弈场景中 RTBAgent 如何应对竞争对手策略变化？**
A: ①市场感知：实时监控 win_rate 变化（win_rate↓ → 竞争加剧 → 可能需要提价）②竞争对手建模：通过历史竞价数据推断竞争对手的出价分布 ③策略多样化：随机化出价幅度，避免对手学习并反制自己的策略。
