# Ch9 A/B 实验设计与效果评估

## 文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [experiment-design.md](experiment-design.md) | 实验设计原则、分桶策略、辛普森悖论、多重检验校正、新奇效应、网络效应、CUPED、MAB 自适应实验 | ~280 |
| [metrics-analysis.md](metrics-analysis.md) | 指标体系（短期/长期/生态）、统计显著性检验、指标冲突分析、长期效果评估、Interleaving、因果推断 | ~280 |

## 知识点覆盖

### experiment-design.md
- A/B 测试基本原理与标准流程
- 样本量估算（四参数公式与实践要点）
- 辛普森悖论（定义、推荐系统场景、识别方法）
- 多重检验校正（Bonferroni vs BH-FDR）
- 新奇效应与 Peeking Problem
- 分桶策略与分层实验框架（Google 方案）
- 网络效应实验（集群随机化、Switchback）
- CUPED 方差缩减原理
- MAB 自适应实验（Epsilon-Greedy / UCB / Thompson Sampling）

### metrics-analysis.md
- 推荐系统四类指标体系（行为/体验/生态/商业）
- North Star Metric 与主指标+护栏指标设计
- 指标间权衡关系量化（综合得分/帕累托前沿/约束优化）
- 统计假设检验（t 检验流程、置信区间解读）
- 结果不显著的诊断五步法
- 指标冲突分析框架（CTR vs 停留时长实战案例）
- 长期效果评估（Holdout/代理指标/LTV 建模/留存队列）
- Interleaving 实验（Team Draft / Probabilistic）
- 因果推断纠偏（IPW / Doubly Robust / 因果森林）
- 实验文化与常见反模式

## 对应 PDF 章节
- 第 9 章全部 6 节，约 40 题
- PDF 页码：640-708
