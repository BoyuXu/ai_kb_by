# Cold-Starts in Generative Recommendation: A Reproducibility Study

**ArXiv:** 2603.29845 | **Date:** 2026-03

## 核心贡献
首个系统性研究生成式推荐在冷启动协议下的可复现性研究，建立统一评估框架。

## 研究设计
对代表性生成式推荐模型进行跨冷启动场景复现和评估：
- **User cold-start**：新用户无历史数据
- **Item cold-start**：新 item 无交互历史

## 三大设计维度分析

### 1. 模型规模（Model Scale）
**发现**：规模提升仅带来边际改善，无法从根本上弥合冷启动差距。

### 2. 标识符设计（Identifier Design）
**发现**：
- 文本标识符（Textual ID）：大幅改善 item 冷启动，但损害 warm 和 user 冷启动性能
- 组合语义编码方案：更好的鲁棒性

### 3. 训练策略（Training Strategy）
**发现**：强化学习不能持续改善，甚至在冷启动下会降级性能。

## 关键结论
标识符设计是决定性因素，其影响远超模型规模和训练策略。

## 面试考点
- 生成式推荐中冷启动的特殊挑战？
- 为什么文本 ID 对 item 冷启但损害 user 冷启？
- SID 设计的权衡（atomic vs semantic vs textual）？

**Tags:** #rec-sys #cold-start #generative-recommendation #sid #reproducibility
