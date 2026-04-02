# DIN: Deep Interest Network for Click-Through Rate Prediction

> 来源：KDD 2018, Alibaba | 年份：2018 | 领域：ads/02_rank（CTR预估）

## 问题定义

传统 CTR 模型（Wide&Deep, DeepFM）对用户历史行为序列做简单 pooling（avg/sum），将变长序列压缩为定长向量。这种方式存在根本缺陷：

- **兴趣多样性丢失**：一个用户可能同时对"电子产品"和"户外运动"感兴趣，avg pooling 把两类兴趣混合成一个模糊表示
- **与目标广告的关联性丢失**：当用户看到一个"运动鞋广告"时，只有历史中"运动相关"的行为才相关，但 avg pooling 无法区分
- **业务背景**：淘宝广告系统每天处理数十亿次展示，用户历史行为序列长度 50-200，需要高效且精准的兴趣建模

**核心问题**：如何让模型动态关注与当前目标广告最相关的历史行为，实现"局部兴趣激活"？

## 模型结构图

```
┌─────────────────────────────────────────────────────┐
│                    Output: pCTR                      │
│                       ↑                              │
│                   σ(FC Layer)                         │
│                       ↑                              │
│              ┌────────┴────────┐                     │
│              │   Concat Layer   │                    │
│         ┌────┴────┐    ┌───────┴──────┐             │
│         │ Context  │    │  User Interest│             │
│         │ Features │    │  Representation│            │
│         └─────────┘    └───────┬──────┘             │
│                                ↑                     │
│                    Weighted Sum (不用softmax)         │
│                    ┌───────────┼───────────┐         │
│                   w₁          w₂          w₃        │
│                    ↑           ↑           ↑         │
│              ┌─────┴─────┬─────┴─────┬─────┴─────┐  │
│              │ Attention │ Attention │ Attention │  │
│              │  Unit     │  Unit     │  Unit     │  │
│              └─────┬─────┴─────┬─────┴─────┬─────┘  │
│                    ↑           ↑           ↑         │
│              ┌─────┴───┐ ┌────┴────┐ ┌────┴────┐   │
│              │ Behav e₁ │ │ Behav e₂│ │ Behav e₃│   │
│              └─────────┘ └─────────┘ └─────────┘   │
│                         ↑ Target Ad Embedding        │
│                    (cross with each behavior)        │
└─────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：Attention Score（局部激活单元）

$$a_i = f(e_i, e_a) = \sigma(W \cdot [e_i; e_a; e_i - e_a; e_i \odot e_a] + b)$$

**解释：**
- $e_i$：第 $i$ 个历史行为的 embedding 向量
- $e_a$：目标广告的 embedding 向量
- $[;]$：向量拼接操作
- $e_i - e_a$：差值向量，显式建模差异
- $e_i \odot e_a$：逐元素乘积，显式建模交互
- $\sigma$：sigmoid 激活函数（**不是 softmax**）
- $W, b$：可学习参数

### 公式2：用户兴趣表示

$$\mathbf{v}_U = \sum_{i=1}^{N} a_i \cdot e_i = \sum_{i=1}^{N} f(e_i, e_a) \cdot e_i$$

**解释：**
- $N$：用户历史行为序列长度
- $a_i$：第 $i$ 个行为对目标广告的注意力权重
- $\mathbf{v}_U$：加权聚合后的用户兴趣向量

### 公式3：Dice 激活函数

$$\text{Dice}(x) = p(x) \cdot x + (1 - p(x)) \cdot \alpha x, \quad p(x) = \frac{1}{1 + e^{-\frac{x - E[x]}{\sqrt{Var[x] + \epsilon}}}}$$

**解释：**
- $E[x], Var[x]$：当前 mini-batch 的均值和方差
- $\alpha$：可学习参数，控制负半轴斜率
- 相比 PReLU 的固定拐点 $x=0$，Dice 的拐点自适应于数据分布

### 公式4：Mini-Batch Aware Regularization

$$L_{reg} = \sum_{(x,y) \in B} \sum_{j=1}^{K} \frac{\|w_j\|^2}{n_j}$$

**解释：**
- $B$：当前 mini-batch
- $w_j$：第 $j$ 个特征的 embedding 参数
- $n_j$：该特征在 batch 中出现的次数
- 高频特征正则化强度小，低频特征正则化强度大，避免长尾 ID 过拟合

## 与基线方法对比

| 方法 | 核心区别 | 优势 | 劣势 |
|------|---------|------|------|
| **Avg Pooling** | 等权聚合所有行为 | 简单、计算快 | 丢失兴趣多样性和相关性 |
| **Wide&Deep** | Wide侧手工交叉特征 | 可解释性 | 需要特征工程，无序列建模 |
| **DeepFM** | FM自动二阶交叉 | 无需手工交叉 | 行为序列仍是avg pooling |
| **DIN** | 目标广告驱动的注意力 | 动态兴趣激活 | 仅建模静态兴趣，无时序 |
| **DIEN** | GRU + 注意力 | 捕捉兴趣演化 | 计算复杂度高 |
| **SIM** | 检索 + DIN | 支持超长序列(10000+) | 系统复杂度高 |

## 实验结论

- **淘宝广告 CTR 预估**：AUC 提升 0.6%（工业界 0.1% 即显著）
- **vs Wide&Deep**：参数量仅增加约 5%，AUC 提升 0.4%
- **消融实验**：
  - Attention 机制贡献最大（+0.3% AUC）
  - Dice 激活函数贡献 +0.1% AUC
  - Mini-Batch Aware Reg 防止过拟合，AUC +0.1%
- **在线 A/B**：CTR 提升 10%，RPM（千次展示收入）提升 3.8%

## 工程落地要点

1. **序列长度选择**：原文使用最近 50 个行为，工业实践中 50-200 为常见范围；超过 200 需要考虑 SIM 等检索方案
2. **Attention 计算复杂度**：$O(L \cdot d)$，L 为序列长度，d 为 embedding 维度，适合实时 serving（<10ms）
3. **特征去噪**：用户行为序列需过滤误点击、机器行为、停留时间极短的行为
4. **Embedding 维度**：商品 ID embedding 通常 16-64 维，类目 embedding 8-16 维
5. **负样本处理**：展示未点击为负样本，注意负采样比例对 Attention 权重分布的影响

## 面试考点

**Q1：DIN 的 Attention 为什么不用 softmax？**
> 用户对不同类目的兴趣可以同时存在（非互斥），softmax 强制归一化为概率分布会抑制多兴趣共存。sigmoid 允许每个历史行为独立评分，多个行为可以同时获得高权重。这也是"局部激活"而非"全局竞争"的核心设计。

**Q2：DIN vs Transformer Self-Attention 的区别？**
> DIN 是目标广告 vs 历史行为的 cross-attention（单向查询），复杂度 $O(L)$；Transformer 是序列内部的 self-attention，复杂度 $O(L^2)$。DIN 更轻量适合实时推断；Transformer（如 BST）建模更丰富但计算更重。

**Q3：如何处理用户超长行为序列（1000+）？**
> SIM（Search-based Interest Model）方案：先基于目标广告属性从长序列中检索 TopK 相关行为（General Search Unit），再对 TopK 做 DIN Attention（Exact Search Unit）。检索阶段用倒排索引或类目匹配，复杂度从 $O(L)$ 降到 $O(K)$。

**Q4：Dice 激活函数相比 PReLU 的优势是什么？**
> PReLU 的拐点固定在 $x=0$，Dice 的拐点由数据分布的均值决定（$E[x]$），因此能自适应不同层、不同神经元的数据分布。在广告场景中数据分布随时间漂移，Dice 的自适应性更重要。

**Q5：DIN 中 Attention 的输入为什么要拼接 $e_i - e_a$ 和 $e_i \odot e_a$？**
> 显式提供差异信息和交互信息，帮助 Attention 网络更容易学到"相似"和"差异"模式。仅拼接 $[e_i; e_a]$ 理论上 MLP 也能学到，但显式提供这些特征加速收敛且效果更好（类似 NLP 中 ESIM 的做法）。

**Q6：Mini-Batch Aware Regularization 为什么对广告系统特别重要？**
> 广告系统有极大的 ID 特征空间（数亿商品 ID），大部分是长尾低频 ID。传统 L2 正则对所有参数等同对待，高频 ID 被过度正则化，低频 ID 正则不足。MBA-Reg 按 batch 内出现频次自适应调整正则强度，解决了长尾分布问题。

**Q7：DIN 能否用于召回阶段？**
> DIN 本身是精排模型（需要计算每个候选广告的注意力权重）。但其思想可以迁移到召回：MIND（Multi-Interest Network with Dynamic Routing）用胶囊网络生成多个兴趣向量，每个兴趣向量独立做 ANN 检索，实现了"多兴趣召回"。
