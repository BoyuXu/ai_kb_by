# MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

传统 CTR 模型（DeepFM、DCN 等）通过加法操作（sum pooling、concatenation）聚合特征，忽略了不同样本（instance）对特征重要性的差异化需求。MaskNet 提出：
- 不同用户、不同上下文下，同一特征的重要性不同
- 需要一种能够根据输入实例动态调整特征权重的机制
- 乘法操作（multiplication）比加法更适合捕捉这种实例级别的特征调制

## 核心方法与创新点

### Instance-Guided Mask

核心思想：用 instance 的特征向量生成一个 mask，对所有特征做 element-wise 乘法：

$$V_{mask} = LayerNorm(f(e) \odot V_{emb})$$

其中：
- $e$ 是当前 instance 的 embedding 向量（通过 MLP 映射得到 mask）
- $V_{emb}$ 是所有特征的 embedding 拼接
- $f(e)$ 是由 instance 生成的 mask 向量
- LayerNorm 稳定训练

### MaskBlock

每个 MaskBlock 包含：
1. **Instance-guided mask**：动态调制特征
2. **隐藏层**：带激活函数的全连接
3. **LayerNorm**：归一化

$$MaskBlock(V) = LayerNorm(Linear(V_{mask}))$$

### 两种组合方式

1. **Serial MaskNet**：多个 MaskBlock 串联，逐层精炼
2. **Parallel MaskNet**：多个 MaskBlock 并联，输出聚合

### 与 SENet 的区别

SENet 做的是全局平均池化后的 channel-wise 权重；MaskNet 是 instance-specific 的 element-wise 调制，粒度更细。

## 实验结论

- 在 Criteo、Avazu、MovieLens 数据集上超越 DeepFM、DCN、xDeepFM、AutoInt
- Serial MaskNet 通常优于 Parallel MaskNet
- 乘法操作对于捕捉用户意图的细粒度差异效果显著
- 在微博广告系统线上 A/B 测试中，CTR 显著提升

## 工程落地要点

1. **LayerNorm 关键**：Mask 后的乘法结果需要 LayerNorm 稳定，否则训练不收敛
2. **Mask 生成网络**：建议 1-2 层 MLP，过深反而引入噪声
3. **串联深度**：3-4 个 MaskBlock 通常足够，更多收益递减
4. **与 FFM 结合**：MaskNet 可以看作动态 FFM 的一种实现
5. **计算开销**：相比普通 DNN 增加约 20-30% 计算，但收益显著
6. **冷启动**：新用户无足够历史，Mask 生成质量下降，需要回退策略

## 面试考点

**Q：MaskNet 与 SENet 的核心区别是什么？**
A：SENet 做全局 squeeze-excitation，权重对同一 batch 内所有样本相同（channel-wise）；MaskNet 的 mask 由每个 instance 的特征动态生成，每个样本有不同的 mask，粒度更细是 instance-specific 的。

**Q：为什么用乘法而不是加法？**
A：乘法可以实现"门控"效果，当 mask 值接近 0 时可以完全抑制某个特征维度；加法只能做线性偏移，无法实现这种强调/抑制的效果。

**Q：Serial 和 Parallel MaskNet 分别适合什么场景？**
A：Serial 适合需要逐步精炼特征表示的场景，每层 mask 基于上一层的输出；Parallel 适合希望捕获多视角特征表示的场景，适合特征多样性强的数据集。
