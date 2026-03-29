# HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou

> 来源：arxiv 2408.05430 | 领域：rec-sys | 学习日期：20260328 | 机构：快手 Kuaishou

## 问题定义

工业推荐系统需要同时预测**几十个行为任务**（点击、点赞、关注、转发、负反馈等），广泛使用**混合专家（MoE）多任务架构**。快手团队在实际迭代中发现MoE架构的三大系统性问题：

### 异常现象1：Expert Collapse（专家坍塌）
- 发现：部分专家的输出分布高度分化，某些专家在ReLU激活后有**>90%的零值**
- 根因：训练不均匀导致部分专家几乎不激活，门控网络难以分配合理权重
- 影响：有效专家数量减少，多任务表达能力受损

### 异常现象2：Expert Degradation（专家退化）
- 发现：设计为共享的shared expert被某个特定任务占据，退化为该任务的专属专家
- 根因：梯度更新不均衡，某些任务数据量大，dominant task的梯度"劫持"了shared expert
- 影响：shared expert失去多任务泛化能力

### 异常现象3：Expert Underfitting（专家欠拟合）
- 发现：数据稀疏的任务（如"关注"行为）的specific expert几乎不被使用
- 根因：稀疏任务样本少，specific expert梯度更新稀少，shared expert提供了更多训练信号
- 影响：稀疏任务预测质量差，specific expert欠拟合

## 核心方法与创新点

HoME（**Ho**mogeneous **M**ulti-gate **E**xperts）提出**层次化MoE设计**，针对性解决三个异常：

### 架构总览
```
输入特征
    ↓
[Homogeneous Expert Layer]  ← 解决Expert Collapse
    ↓
[Hierarchical Gating]       ← 解决Expert Degradation  
    ↓
[Task-Specific Layers]      ← 解决Expert Underfitting
    ↓
各任务预测头
```

### 1. Homogeneous Expert Layer（同质化专家层）
解决Expert Collapse：
$$\mathbf{h}_e = \text{Norm}(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e), \quad \forall e \in \{1,...,E\}$$

- 对所有专家的输出做**对抗性正则化**，拉近专家输出分布
- 使用**专家间的KL散度损失**惩罚过度分化：
$$\mathcal{L}_{homo} = \sum_{e_i \neq e_j} \text{KL}(P_{e_i} || P_{e_j})$$

### 2. Hierarchical Gating（层次化门控）
解决Expert Degradation：
- 将专家划分为**多层次**：任务组级共享 → 任务对级共享 → 任务专属
- 门控机制约束：每层的门控权重受上层监督，防止低层门控被单任务dominant
$$\mathbf{g}^{task} = \text{softmax}(\mathbf{W}_g \cdot [\mathbf{h}_{shared}, \mathbf{h}_{specific}])$$

### 3. Task-Adaptive Expert Assignment（任务自适应专家分配）
解决Expert Underfitting：
- 对稀疏任务动态分配更多专家资源（通过元学习学习分配策略）
- 使用任务重要性权重调整各任务在total loss中的比例

## 实验结论

在快手短视频推荐平台（百亿级日志数据）：
- 相比MMoE基线：各任务AUC平均提升**0.2-0.5%**（工业场景极显著）
- Expert Collapse现象消除：各专家激活率趋于均匀
- 稀疏任务（关注、评论）的预测AUC改善最为显著

## 工程落地要点

1. **专家数量选择**：经验规则：共享专家数≈任务数的2倍，specific专家数≈1-2个/任务
2. **同质化损失权重**：$\mathcal{L}_{homo}$的权重需小心调节（过大影响各专家分化表达，过小效果不明显）
3. **层次设计依据**：按任务相似性聚类（如互动类任务一组、消费类任务一组），相似任务共享更多专家
4. **梯度不均衡处理**：使用GradNorm或Uncertainty Weighting对各任务loss动态加权，防止大任务dominant
5. **监控指标**：线上监控每个专家的激活频率，若某专家激活率<5%则触发报警和人工干预

## 面试考点

**Q1：MMoE和HoME的主要区别是什么？**
A：MMoE（Multi-gate MoE）：每个任务有独立门控，从同一批专家中选取不同权重组合。HoME在此基础上增加了：(1)专家同质化正则防止Collapse；(2)层次化门控防止shared expert退化；(3)任务自适应分配解决稀疏任务Underfitting。HoME是对MMoE工业落地问题的系统性修复。

**Q2：如何检测和量化Expert Collapse？**
A：指标：专家激活率（batch内被激活的样本比例）、专家输出向量的方差、门控权重的熵（越低说明分配越集中）。工程实现：在训练过程中每N步记录每个专家的平均激活概率，若任一专家持续低于阈值（如5%）则触发处理。

**Q3：多任务学习中如何处理不同量级的正样本比例（CTR高vs关注率低）？**
A：常用策略：(1) Focal Loss：降低易分类样本权重，让稀疏任务的困难样本贡献更多梯度；(2) 过采样稀疏任务的正样本；(3) 独立调整各任务的loss权重（uncertainty weighting自动学习）；(4) HoME的任务自适应专家分配，给稀疏任务分配专属更多专家。

**Q4：为什么shared expert会被dominant task"劫持"？如何在工程上预防？**
A：根因：dominant task（如CTR，样本多）每步都更新shared expert的梯度，而小任务（如关注，样本少）更新频率低。shared expert逐渐"记住"了dominant task的模式。预防：(1) 梯度裁剪（只允许shared expert接受来自各任务的梯度和，不超过阈值）；(2) GradNorm动态调整各任务梯度强度；(3) 定期重置shared expert（Periodic Reset）。

**Q5：工业推荐中多任务学习的最佳实践是什么？**
A：(1) 任务关系分析先行：用梯度相似度分析任务是否适合共享（正相关任务共享有益，负相关任务强制共享反而互相干扰）；(2) 从简单到复杂：先跑MMoE baseline，识别问题任务，再针对性引入HoME等修复；(3) 任务权重调优：用Optuna等工具自动搜索loss权重；(4) 监控每个任务的AUC变化，防止提升主任务牺牲其他任务。
