# Enhancing Generative Auto-bidding with Prompt-Based Constraints
> 来源：arxiv/2403.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
自动出价（Auto-bidding）系统中约束满足难题：
- 广告主设定 ROI/CPA/预算 等约束，自动出价系统需在满足约束下最大化转化
- 传统 RL 出价：约束作为硬限制导致可行域过小，探索困难
- 多约束冲突：同时满足 CPA ≤ target 和 Budget ≤ B 且最大化 CVR
- 动态市场：竞价环境实时变化，静态策略难以自适应

## 核心方法与创新点
**Prompt-Based Constraints for Generative Auto-bidding**：将出价约束编码为 LLM 提示，生成满足约束的出价策略。

**核心框架：**
```
Constraint Prompt: 
  "广告主 A: 目标 CPA=50元, 日预算=10000元, 当前消耗=3000元
   当前时间: 14:00, 历史转化率: 2.3%, 当前竞价环境: 竞争激烈
   请生成最优出价策略..."

Policy LLM → bid = f(context, constraints)
```

**生成式出价策略：**
```python
# LLM 直接输出出价决策和推理过程
response = llm.generate(
    prompt=constraint_prompt + context_features,
    format="json"  # 强制输出 {"bid": X, "reasoning": "..."}
)
bid = response["bid"]
```

**约束感知生成（Constrained Decoding）：**
```python
# 解码时约束出价范围
valid_bid_range = compute_valid_range(
    remaining_budget=budget - spent,
    target_cpa=target_cpa,
    predicted_cvr=cvr_model.predict(context)
)
bid = clip(llm_bid, valid_bid_range.min, valid_bid_range.max)
```

**训练数据构建：**
```
正样本：历史高 ROI 的出价决策 + 对应的约束上下文
负样本：违反约束的出价 + 低 ROI 的出价
训练：SFT（学习约束遵循）+ DPO（偏好对齐：合规且高效的出价）
```

## 实验结论
- 某广告平台 auto-bidding 实验：
  - CPA 约束满足率：+8.2%（vs 传统 RL 方法）
  - 约束满足条件下的转化量：+4.1%
  - 多约束场景（CPA + 预算 + ROI）：约束满足率 89%（vs RL 73%）
- 动态市场适应：竞价环境突变（竞争加剧）后，恢复时间减少 40%

## 工程落地要点
1. **Prompt 模板标准化**：将广告主约束参数自动填入标准化 Prompt 模板
2. **约束违反惩罚**：加入约束违反检测，触发强制回退到保守出价
3. **LLM 推理延迟**：RTB 要求 100ms 内，需要小型化 LLM（7B 量化）或 Speculative Decoding
4. **历史策略微调**：用广告主历史高效出价序列 SFT，个性化策略
5. **安全回退**：LLM 输出解析失败时，回退到规则出价

## 常见考点
**Q1: 为什么用 LLM 做自动出价而不是传统 RL？**
A: LLM 优势：①理解复杂约束（多条件组合，自然语言表达）②强大的上下文推理（结合历史表现、市场信息、时段特点）③Few-shot 适应：新广告主提供几个示例即可生成合理策略。传统 RL 在多约束、动态环境中探索困难。

**Q2: Prompt-Based 出价的主要风险？**
A: ①幻觉：LLM 可能输出超出合理范围的出价 ②延迟：RTB 场景 100ms 约束，LLM 推理常超时 ③可解释性差：LLM 的出价推理不总是正确 ④分布外：训练数据没有覆盖的市场场景可能出错。需要多层安全措施。

**Q3: 如何构建自动出价 LLM 的训练数据？**
A: 从历史竞价日志中提取：①约束参数（目标 CPA、预算）②市场特征（竞价时间、竞争强度）③出价决策（实际出价）④结果标签（转化量、CPA、ROI）。正样本：CPA 达标且 ROI 高；负样本：CPA 超标或 ROI 低。

**Q4: Constrained Decoding 如何保证约束满足？**
A: 数学约束转硬限制：bid_max = min(remaining_budget/remaining_requests, target_cpa × predicted_cvr × 1000)；LLM 生成出价后 clip 到 [bid_min, bid_max]。保证「即使 LLM 输出不合理，最终出价也在约束范围内」。

**Q5: 自动出价中 CPA 和 ROI 约束的区别与统一？**
A: CPA（Cost Per Action）= 转化成本约束；ROI = 收益/成本约束。统一：ROI = revenue_per_conversion / CPA。如果知道每次转化的收益 value，则 target_ROI → target_CPA = value / target_ROI。两者都可转化为出价公式：bid = predicted_cvr × value_per_conversion × ROI_coefficient。
