# Ch6 深度学习在排序阶段的应用

## 核心概念总览

本章覆盖深度排序模型全链路：从经典模型（Wide&Deep/DeepFM/DCN）到序列建模（DIN/DIEN）、
训练优化、模型部署、以及因果推断和强化学习等前沿方向。

共 7 节 43 题，拆分为 4 个专题文件。

---

## 专题索引

### [1. Wide&Deep 家族](wide-deep-family.md)
```
覆盖内容：
  - DNN vs LR 优势对比
  - Wide&Deep 架构（Memorization + Generalization）
  - DeepFM（FM 自动二阶交叉 + 共享 Embedding）
  - DCN / DCN-V2（Cross Network 公式 + bit-wise 交叉）
  - xDeepFM / CIN（vector-wise 交叉）
  - 模型演进全景对比表
  - 高基数特征处理方案

对应 PDF 章节：6.1, 6.2, 6.3
```

### [2. 注意力排序模型](attention-models.md)
```
覆盖内容：
  - DIN Target Attention 机制 + Dice 激活 + Mini-Batch 正则
  - DIEN 兴趣演化（Interest Extractor + AUGRU 公式）
  - 长序列处理方案矩阵（截断/采样/SIM 两阶段检索）
  - Transformer vs GRU 序列建模对比
  - 多模态行为序列融合策略
  - 行为序列特征设计 + 排序模型输入架构

对应 PDF 章节：6.5
```

### [3. 训练优化](training-optimization.md)
```
覆盖内容：
  - 样本不均衡（负采样 + 概率校准 + Focal Loss）
  - Pointwise / Pairwise / Listwise 学习范式
  - 过拟合防御（L1/L2/Dropout/Early Stopping）
  - 特征工程（数值/类别/高基数特征处理）
  - CTR 涨但 CVR 跌的排查思路
  - 分布式训练（参数服务器 + 数据/模型并行）

对应 PDF 章节：6.4
```

### [4. 模型压缩与部署](model-compression.md)
```
覆盖内容：
  - 剪枝（结构化/非结构化）
  - 量化（PTQ vs QAT + 混合精度）
  - 知识蒸馏（模型/特征/序列蒸馏）
  - A/B 测试指标体系（离线 AUC/GAUC + 在线 CTR/CVR）
  - 在线推理优化（算子融合/动态 Batching/异步特征/多级缓存）
  - 分布式服务架构 + 推理框架对比
  - 因果推断消除偏差（IPW/位置偏差/流行度偏差）
  - 强化学习（MDP 建模 + UCB/Thompson Sampling）

对应 PDF 章节：6.6, 6.7
```

---

## 面试高频考点速查

```
排序    题目                                     详见文件
──────────────────────────────────────────────────────────────────
 1     Wide&Deep 的 Wide/Deep 分别负责什么        wide-deep-family.md
 2     DeepFM vs Wide&Deep 核心差异               wide-deep-family.md
 3     DCN Cross Network 公式及参数复杂度          wide-deep-family.md
 4     bit-wise vs vector-wise 交叉区别            wide-deep-family.md
 5     高基数特征（千万级 item_id）处理             wide-deep-family.md
 6     DIN Target Attention 机制                  attention-models.md
 7     DIEN AUGRU 为什么不是 GRU + Attention       attention-models.md
 8     长序列处理的工程方案（SIM 等）               attention-models.md
 9     Transformer vs GRU 选型                    attention-models.md
10     负采样后的概率校准公式                       training-optimization.md
11     Focal Loss 公式及 gamma 的作用              training-optimization.md
12     Pointwise vs Pairwise vs Listwise          training-optimization.md
13     CTR 涨但 CVR 跌的排查思路                   training-optimization.md
14     模型压缩三板斧：剪枝、量化、蒸馏             model-compression.md
15     离线 AUC 涨但线上不涨的排查                  model-compression.md
16     IPW 消除曝光偏差的原理                       model-compression.md
17     探索-利用平衡（UCB/Thompson Sampling）       model-compression.md
```
