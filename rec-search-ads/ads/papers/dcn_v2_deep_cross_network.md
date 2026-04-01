# DCN V2: Improved Deep & Cross Network for Feature Cross Learning

> 来源：https://arxiv.org/abs/2008.13535 | 领域：计算广告 | 学习日期：20260331

## 问题定义

DCN V1的Cross Network表达能力受限于参数结构，无法有效建模高阶特征交叉；Cross与Deep网络融合方式不够灵活。

## 核心方法与创新点

1. **矩阵式Cross层**：向量外积替换为全矩阵参数

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

参数量从 $O(d)$ 到 $O(d^2)$，表达能力大幅提升。

2. **低秩分解**：$W = U V^T$，控制参数量 $O(d \times r)$
3. **Stacked与Parallel结构**：Cross和Deep的串行/并行两种融合方式
4. **混合专家交叉**：MoE版本Cross层

## 实验结论

在Criteo和生产数据上AUC提升0.1-0.3%，低秩版本参数量减少60%性能几乎不降。

## 工程落地要点

- 低秩分解版本适合线上部署
- Cross层可作为即插即用模块
- 训练时间与DNN相当
- Google内部已大规模部署

## 面试考点

1. **DCN-V1局限？** 向量参数Cross层只能学习特定形式交叉
2. **低秩分解必要性？** $O(d^2)$ 工业场景不可接受
3. **Stacked vs Parallel？** Stacked适合序列特征交叉，Parallel适合异构特征
