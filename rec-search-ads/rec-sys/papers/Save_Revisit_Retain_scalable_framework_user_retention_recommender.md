# Save, Revisit, Retain: A Scalable Framework for Enhancing User Retention in Large-Scale Recommender Systems

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

推荐系统通常优化即时指标（CTR、CVR），忽略**用户留存（Retention）**：用户是否在 D+7、D+30 后仍活跃。即时满意 ≠ 长期留存，过度推送点击诱饵可能造成用户疲劳与流失。

$$\text{Retention}_{D+k} = P(\text{user active at } t+k | \text{active at } t)$$

## 核心方法与创新点

**SRR（Save-Revisit-Retain）框架** 将用户留存分解为三个可优化的行为信号：

1. **Save（收藏）**：用户主动保存表示深度兴趣
2. **Revisit（回访）**：用户多次访问某类内容，表示持续偏好
3. **Retain（留存）**：直接优化 D+7 留存率

三阶段建模：

$$\text{Score}(u, i) = \alpha \cdot P_{\text{CTR}} + \beta \cdot P_{\text{Save}} + \gamma \cdot P_{\text{Revisit}} + \delta \cdot P_{\text{Retain}}$$

权重 $(\alpha, \beta, \gamma, \delta)$ 通过多目标帕累托优化学习。

**长期价值网络（LTV Network）**：
- 输入：用户短期行为序列（14天）
- 输出：预测 D+7 留存概率
- 训练信号：历史真实留存标签（延迟标签问题用 survival model 处理）

$$P_{\text{Retain}} = \text{Sigmoid}(\text{MLP}([e_u, e_i, e_{\text{context}}]))$$

**生存分析处理延迟标签**：
- 留存标签 D+7 才能获得，但当天需要训练
- 用 Weibull 生存模型估计未来留存概率

## 实验结论

- D+7 留存率提升 **+3.2%**（在线 A/B，亿级用户）
- Save 率提升 +8.5%，Revisit 率提升 +5.1%
- 整体 DAU 增长 **+1.8%**（核心业务指标）
- CTR 略降 -1.2%（接受的权衡）

## 工程落地要点

- 延迟标签（D+7）需要专门的训练数据管道（历史数据）
- 多目标权重 $(\alpha, \beta, \gamma, \delta)$ 建议每周离线调参，不宜实时变化
- 收藏/回访信号需去噪（机器人/误点过滤）
- 帕累托优化：推荐 MOO-SVGD 或 MGDA 算法

## 面试考点

1. **Q**: 短期指标（CTR）和长期留存有时是矛盾的，如何平衡？  
   **A**: 多目标优化（CTR + Save + Revisit + Retain），通过帕累托最优找平衡点；线上根据业务阶段调整权重。

2. **Q**: 留存标签是延迟的，如何用于当天训练？  
   **A**: 生存分析（如 Weibull 模型）：用已观察到的留存事件估计未来时间点的生存概率，作为训练信号。

3. **Q**: "收藏" 行为为什么比 "点击" 更能预测留存？  
   **A**: 收藏是主动行为，表示用户有强烈的再访意图；点击是被动触发，包含大量误点和好奇心点击，噪声更多。
