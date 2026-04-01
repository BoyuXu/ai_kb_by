# MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask
> 来源：https://arxiv.org/abs/2102.07619 | 领域：ads | 学习日期：20260329

## 问题定义

DNN CTR 排序模型（FNN、DeepFM、xDeepFM 等）普遍依赖**加性特征交互**（additive interactions），即 MLP 的加法运算。研究发现：

1. **加性交互的局限性**：前馈神经网络的加性操作在捕获特征交互方面效率不足
2. **缺乏乘性操作**：真实世界中特征之间的交互往往是乘性的（如"性别=女"×"品类=化妆品"→高 CTR）
3. **特征交互的实例依赖性**：同样的特征对，对不同的请求（instance）重要性不同，需要动态调整

## 核心方法与创新点

### Instance-Guided Mask（实例引导掩码）

**核心思想**：基于当前请求实例（instance），动态生成特征掩码，对 Embedding 和 MLP 层做逐元素乘法（element-wise product）。

$$
\mathbf{V}_{mask} = \text{LayerNorm}(\mathbf{M} \odot \mathbf{V})
$$

其中 $\mathbf{M}$ 是由输入实例动态生成的掩码向量，$\odot$ 是逐元素乘法。

**掩码生成过程：**
```
当前请求 Instance X
    ↓
MLP(X) → 掩码向量 M
    ↓
M ⊙ Feature Embedding V → 加权特征表示
M ⊙ Hidden Layer H → 加权隐藏层输出
```

### MaskBlock：基本构建模块

**三组件结构：**
1. **LayerNorm**：稳定乘性操作的梯度（防止数值爆炸）
2. **Instance-Guided Mask**：动态特征重加权
3. **Feed-Forward Layer**：传统 MLP 层

$$
\text{MaskBlock}(X) = \text{FFN}(\text{LayerNorm}(\text{Mask}(X) \odot X))
$$

**MaskNet 两种配置：**

```
串行 MaskNet (Serial MaskNet):
MaskBlock₁ → MaskBlock₂ → ... → MaskBlockN → 输出

并行 MaskNet (Parallel MaskNet):
         ┌── MaskBlock₁ ──┐
Input → ─┤── MaskBlock₂ ──├─→ Concat → 输出
         └── MaskBlockN ──┘
```

### 创新点总结

| 创新点 | 说明 |
|--------|------|
| 乘性特征交互 | 引入 element-wise product，补充加性操作的不足 |
| 实例动态性 | 掩码基于当前请求生成，同一特征对不同请求有不同权重 |
| LayerNorm 稳定性 | 解决乘性操作的训练不稳定问题 |
| 通用构建块 | MaskBlock 可即插即用到现有 DNN 架构 |

## 实验结论

在三个真实数据集上，MaskNet 显著超越 SOTA：

| 对比模型 | 数据集 AUC 提升 |
|---------|--------------|
| DeepFM | +0.1% ~ +0.5% |
| xDeepFM | +0.05% ~ +0.3% |
| DCN | +0.1% ~ +0.4% |

工业 CTR 系统中 **0.1% AUC 提升** 即有显著业务价值（通常对应约1%的 RPM 提升）。

发表于 DLP-KDD 2021。

## 工程落地要点

1. **LayerNorm 的必要性**：乘性操作必须配合 LayerNorm，否则训练极不稳定（梯度爆炸/消失）
2. **掩码生成网络**：建议使用轻量级 MLP（1-2层，隐层维度 = embedding 维度），避免参数爆炸
3. **串行 vs 并行配置**：串行配置更深，适合特征交互复杂场景；并行配置更宽，适合特征空间大的场景
4. **即插即用**：MaskBlock 可以直接替换现有 DNN 中的 FFN 层，迁移成本低
5. **计算开销**：掩码生成额外增加约 20-30% 计算量，需评估延迟影响

## 面试考点

**Q1: 为什么说前馈神经网络（FFN）在捕获特征交互方面效率不足？**
A: FFN 的基础操作是加权求和（矩阵乘法=加性操作），从数学上讲，两个特征的乘性交互 f₁×f₂ 需要无限多个加性隐含单元才能精确表示。FM 和 MaskNet 直接引入乘法操作，更自然地建模乘性特征关系。

**Q2: Instance-Guided Mask 的"实例引导"体现在哪里？为什么重要？**
A: 掩码向量 M 由当前请求的特征 X 动态生成（M = MLP(X)），不同请求生成不同掩码。重要性：同样是"美妆产品"特征，对"20岁女性用户"和"50岁男性用户"的重要性完全不同；实例动态掩码让模型根据每个请求的上下文自适应调整特征权重。

**Q3: MaskBlock 为什么必须引入 LayerNorm？放在掩码之前还是之后？**
A: 乘性操作（element-wise product）可能导致值域爆炸或消失（梯度不稳定）。LayerNorm 放在掩码之后：`LayerNorm(M ⊙ V)`，将乘性结果归一化到稳定的数值范围，再输入 FFN。归一化应在乘法之后，保留掩码的缩放效果的同时稳定数值。

**Q4: 串行 MaskNet 和并行 MaskNet 适合什么场景？如何选择？**
A: 串行：多个 MaskBlock 顺序堆叠，每层的输出作为下一层的输入，适合需要深层特征交互、特征关系复杂的场景（如搜索、广告精排）。并行：多个 MaskBlock 同时处理输入，Concat 后输出，适合特征维度高、需要多视角特征表示的场景（如内容推荐）。通常离线 NE 实验决定。

**Q5: MaskNet 与 SENet（Squeeze-and-Excitation Network）有什么区别和联系？**
A: 联系：都使用特征重加权（channel-wise/feature-wise 乘法）。区别：①SENet 的权重基于全局统计（squeeze全局平均），MaskNet 基于当前实例（instance-guided）→ MaskNet 的权重更具动态性 ②SENet 主要用于 CV，关注 channel 关系；MaskNet 专门为 CTR 特征 embedding 设计 ③MaskNet 同时在 embedding 层和 FFN 层施加掩码，SENet 通常只在 embedding 层
