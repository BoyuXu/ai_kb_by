# PreferRec: Learning and Transferring Pareto Preferences for Multi-objective Re-ranking

> 来源：arxiv 2603.22073 | 领域：rec-sys | 学习日期：20260328

## 问题定义

推荐系统重排（Re-ranking）需要同时满足**多个相互冲突的业务目标**：
- 点击率（CTR）vs 时长（Duration）
- 多样性（Diversity）vs 相关性（Relevance）  
- 商业变现（Revenue）vs 用户体验（UX）

现有多目标重排方法的缺陷：
1. **固定权重**：人工设定各目标权重，无法适应不同用户/场景
2. **Scalarization局限**：将多目标加权求和为单目标，丢失Pareto前沿信息
3. **迁移困难**：在一个场景学到的多目标平衡策略无法迁移到新场景

**核心问题**：如何学习并迁移Pareto最优的多目标偏好策略？

## 核心方法与创新点

### 1. Pareto偏好建模
$$\mathcal{F}_{pareto} = \{f(\mathbf{x}) | \nexists \mathbf{x}' : f_i(\mathbf{x}') \geq f_i(\mathbf{x}) \forall i, \text{ with at least one } >\}$$

PreferRec将重排问题转化为在Pareto前沿上选择操作点的问题：
- 不是找单个最优解，而是学习整条Pareto前沿
- 通过**偏好向量**（Preference Vector）$\mathbf{w} = [w_1, ..., w_K]$ 参数化前沿上的不同点
- 推理时给定偏好向量，模型输出该偏好下的最优重排序列

### 2. Pareto Preference Learning
$$\mathcal{L}(\theta, \mathbf{w}) = -\sum_{k=1}^{K} w_k \cdot r_k(\pi_\theta)$$

训练策略：
- 从偏好向量分布 $\mathbf{w} \sim \text{Dirichlet}(\alpha)$ 中随机采样
- 每批训练使用不同的偏好向量，让模型学会在整条Pareto前沿上操作
- 使用**超网络（HyperNetwork）**将偏好向量映射为模型参数的调制信号

### 3. Pareto Preference Transfer
$$\theta_{target} = \text{Transfer}(\theta_{source}, \Delta_{domain})$$

迁移机制：
- 源域预训练学到的Pareto前沿结构可迁移到目标域
- 只需在目标域少量数据上fine-tune偏好向量到模型参数的映射
- 大幅减少新场景冷启动时间

## 实验结论

- 在多个电商/短视频数据集上，PreferRec在Pareto超体积（Hypervolume）指标上超越SOTA多目标重排方法
- 偏好迁移实验：仅用目标域10%数据，即可达到从头训练80%的性能
- 在线A/B测试：相比固定权重多目标模型，用户满意度（显式反馈）提升显著

## 工程落地要点

1. **偏好向量的业务映射**：将运营指标权重（如"今日主推GMV"）自动映射为模型偏好向量，支持实时调整
2. **HyperNetwork轻量化**：偏好向量→参数调制网络保持小规模（通常2-3层MLP），不显著增加推理延迟
3. **Dirichlet采样训练**：α参数控制偏好向量的分散程度，α<1鼓励极端偏好（专家化），α>1鼓励均匀偏好
4. **Pareto前沿可视化**：提供运营dashboard展示当前策略在Pareto前沿的位置，辅助业务决策
5. **迁移流程**：源域训练（大数据集，充分学习Pareto结构）→ 目标域迁移（少量数据微调偏好映射）→ 线上部署

## 面试考点

**Q1：什么是Pareto最优？在推荐重排中如何理解？**
A：Pareto最优指在不降低任何一个目标的情况下，无法继续提升另一个目标的状态集合。推荐重排中：若A方案CTR比B高，时长比B低，两者均可能是Pareto最优点。Pareto前沿是所有这样方案的集合，业务需根据当前目标在前沿上选取操作点。

**Q2：为什么用Dirichlet分布采样偏好向量？**
A：Dirichlet分布是多项分布的共轭先验，采样结果自然满足$\sum_k w_k = 1, w_k \geq 0$（概率单纯形约束）。通过调整α参数可控制偏好向量的分散程度，确保训练覆盖整条Pareto前沿，不遗漏极端偏好场景。

**Q3：多目标优化中Scalarization（加权求和）的局限性是什么？**
A：(1) 对非凸Pareto前沿，固定权重的线性加权无法找到前沿上的所有解；(2) 权重对结果极敏感，工程上调参困难；(3) 无法描述目标间的非线性权衡关系；(4) 不同量纲的目标需要归一化，引入额外超参。

**Q4：HyperNetwork在PreferRec中的作用是什么？**
A：HyperNetwork接收偏好向量w作为输入，输出主网络（重排模型）的参数调制信号（如scale/shift），让同一套网络权重在不同偏好向量下行为不同。避免为每个偏好点训练独立模型，大幅节省存储和训练成本。

**Q5：多目标推荐中如何衡量模型在Pareto前沿上的覆盖效果？**
A：常用**Hypervolume（超体积）指标**：给定参考点（最差可接受值），计算Pareto前沿与参考点围成的超体积，值越大说明前沿越优越、覆盖越广。此外可用Spread（分散度）衡量前沿分布均匀性，用IGD（反转广义距离）衡量与真实Pareto前沿的近似误差。
