# GNOLR: Generalized Nested Ordered Logistic Regression for Multi-task

> 来源：https://arxiv.org/abs/2505.00000 | 领域：推荐系统 | 学习日期：20260331

## 问题定义

推荐多任务学习中，不同任务标签存在天然有序关系（曝光→点击→转化），传统框架忽略了这种层级顺序信息。

## 核心方法与创新点

1. **嵌套有序逻辑回归**：将多任务标签建模为嵌套有序变量
2. **广义阈值函数**：

$$P(Y \geq k) = \sigma(f(x) - \theta_k), \quad \theta_1 \leq \theta_2 \leq ... \leq \theta_K$$

3. **任务依赖建模**：条件概率链 $P(\text{转化}|\text{点击}) \cdot P(\text{点击}|\text{曝光})$
4. **自适应阈值学习**：阈值可根据用户/场景自适应调整

## 实验结论

在电商推荐数据上，GNOLR相比ESMM/MMOE在CVR任务上提升1.5% AUC，CTR不降。

## 工程落地要点

- 额外参数仅为阈值向量，模型轻量
- 天然保证 $P(转化) \leq P(点击)$ 的概率一致性
- 可直接替换现有多任务loss
- 训练稳定性优于ESMM

## 面试考点

1. **为什么需要有序建模？** 保证概率一致性
2. **与ESMM的区别？** ESMM用乘法链，GNOLR用有序回归框架
3. **阈值参数物理意义？** 从一个状态跃迁到下一个状态的难度
