# GNOLR: Generalized Neural Ordinal Logistic Regression for Multi-Task Recommendation (KDD 2025)
> 来源：arXiv:2502.07894 | 领域：rec-sys | 学习日期：20260330

## 问题定义
推荐系统中多目标优化（CTR、CVR、时长、满意度）通常用多个独立二分类头，忽略了目标之间的有序性（点击→购买→复购是有序的正向行为链）。GNOLR 将有序逻辑回归（Ordinal Logistic Regression）推广到神经网络 + 多任务场景，建模标签的自然序关系。

## 核心方法与创新点
1. **Ordinal Logistic Regression 扩展**：传统 OLR 假设 $P(Y>k) = \sigma(f(x) - \theta_k)$，GNOLR 用神经网络参数化 $f(x)$，支持高维稀疏特征。
2. **共享有序阈值**：多个任务共享一个有序打分函数，不同任务用不同阈值 $\theta_k^{(t)}$ 划分，减少参数冗余。
3. **Multi-Task 兼容**：在 MMOE/PLE 架构上叠加 GNOLR 输出层，各专家网络输出统一传给 GNOLR 打分头。
4. **训练稳定性**：引入 monotonicity constraint（$\theta_1 < \theta_2 < ... < \theta_K$）用 softplus 参数化保证有序性。

## 实验结论
- 淘宝推荐线上：GMV +0.8%，购买率 +0.5%（对比独立二分类多任务）
- 离线 AUC 平均提升 0.3%（CTR/CVR/GMV 三个任务）
- 参数量减少 ~15%（共享打分函数），训练收敛更快

## 工程落地要点
- 有序阈值 $\theta_k$ 需初始化合理（建议用数据分位点），否则早期梯度消失
- 与 MMOE/PLE 集成时，专家输出维度建议统一（128/256）
- 线上打分仍是单次前向，不增加推理开销
- 需要标签有自然序关系才适用，强制对无序标签使用效果可能下降

## 常见考点
- Q: 多任务学习中如何处理任务冲突（Task Conflict）？
  - A: MMOE 用门控网络选择专家；PLE 用独占+共享专家分离；GradNorm/PCGrad 在梯度层面平衡
- Q: 有序标签建模 vs 二分类建模的区别？
  - A: OLR 利用标签序关系，共享打分函数减少参数且更鲁棒；但要求标签确实有序，否则假设不成立
- Q: GNOLR 中 monotonicity constraint 如何实现？
  - A: $\theta_k = \theta_1 + \sum_{i=2}^{k} \text{softplus}(\delta_i)$，确保 $\theta_k$ 单调递增

## 数学公式

$$
P(Y > k | x) = \sigma(f_\theta(x) - b_k), \quad b_1 < b_2 < ... < b_K
$$

$$
\mathcal{L} = -\sum_{t=1}^{T} \sum_i \log P(y_i^{(t)} | x_i)
$$
