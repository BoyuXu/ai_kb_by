# DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

工业级 CTR 预估中，单一特征交叉模块（Cross Network、Bilinear 等）往往只能捕获特定类型的特征关系。现有的 ensemble 方法通常是简单加权融合多个模型，缺乏深层次的层级交互。DHEN 旨在：
1. 设计一种能够在多个抽象层次上 ensemble 不同交叉模块的框架
2. 让多种交叉模块在层级结构中互相增强
3. 在 Meta 广告系统大规模落地

## 核心方法与创新点

### 层级 Ensemble 框架

DHEN 的核心：在每一层同时运行多个交叉模块，输出经过聚合后作为下一层的输入：

$$h^{(l+1)} = Aggregate(\{f_k(h^{(l)})\}_{k=1}^{K})$$

每层的多个模块可以包括：
- **Cross Network**（DCN 类的显式多项式交叉）
- **Bilinear Interaction**（双线性特征交叉）
- **Self-Attention**（Transformer 风格）
- **MLP Block**（隐式高阶交叉）

### 聚合方式

$$Aggregate = LayerNorm\left(\sum_k \alpha_k \cdot f_k(h^{(l)})\right)$$

权重 $\alpha_k$ 可学习（类似 MoE），或使用等权重求和。

### 深层层级结构

$$\text{Layer 1: } [Cross, Bilinear, Attn, MLP] \rightarrow h^{(1)}$$
$$\text{Layer 2: } [Cross, Bilinear, Attn, MLP] \rightarrow h^{(2)}$$
$$\cdots$$

每层的不同模块捕获不同类型的特征交叉，层与层之间形成层级的组合特征。

### 与 AutoML/NAS 结合

DHEN 的结构搜索可以和 NAS 结合，自动学习每层最优的模块组合。

## 实验结论

- 在 Meta 广告数据集（数十亿规模）上，NE (Normalized Entropy) 显著优于 DLRM、DCN V2、DeepFM 等基线
- 层级 ensemble 显著优于单层 ensemble（flat ensemble）
- 模块多样性越高，最终效果越好（不同交叉机制互补）
- 在 Meta 线上系统部署，显著提升广告 CTR 和业务指标

## 工程落地要点

1. **模块选择**：建议包含至少一个显式交叉（Cross/Bilinear）和一个隐式交叉（MLP/Attention）
2. **层数控制**：2-3 层层级 ensemble 即可，更多层计算成本过高
3. **参数共享**：同一模块在不同层之间不共享参数，每层独立学习
4. **内存开销**：多模块并行计算，内存占用是单模型的 K 倍，需要精心设计 batch size
5. **稀疏特征处理**：大规模 ID 类特征需要独立的 embedding 层，不进入多模块交叉
6. **蒸馏压缩**：线上部署可以用 DHEN 作 teacher，蒸馏到轻量 student 模型

## 面试考点

**Q：DHEN 的层级 ensemble 与普通 model ensemble 有什么区别？**
A：普通 ensemble 是多个独立模型各自训练后融合预测，没有层间交互；DHEN 的层级 ensemble 是在每个网络层内多个模块并行，输出作为下一层的输入，模块间形成层级组合，捕获更复杂的特征关系。

**Q：为什么不同类型的交叉模块同时使用效果更好？**
A：Cross Network 擅长显式多项式特征交叉；Self-Attention 擅长捕捉特征间的相关性；MLP 擅长学习隐式非线性关系；Bilinear 擅长两两特征交互。各模块有互补的归纳偏置，组合使用能覆盖更广泛的特征关系。

**Q：大规模系统中 DHEN 的计算开销如何控制？**
A：1) 用 Low-rank 近似压缩各模块参数；2) 减少层数（2层通常够用）；3) 只对 dense 特征做多模块交叉，sparse ID 特征用普通 embedding；4) 用蒸馏将 DHEN 的知识迁移到轻量模型线上推理。
