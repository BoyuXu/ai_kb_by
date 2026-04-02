# SFG: Supervised Feature Generation Framework for CTR Prediction
> 来源：https://arxiv.org/abs/2512.14041 | 领域：ads | 学习日期：20260401

## 问题定义

CTR 预测的主流范式是**判别式特征交互（Discriminative Feature Interaction）**：
- 输入原始 ID Embedding（用户ID、商品ID、类别ID等）
- 通过 MLP、DCN、Transformer 等进行特征交叉
- 输出点击概率

这一范式存在两个根本性缺陷：

**问题1：Embedding 维度坍塌（Embedding Dimensional Collapse）**
- 在纯判别式训练下，不同 ID 的 Embedding 往往收敛到相近的方向
- 导致 Embedding 空间的有效维度远低于设计维度，特征区分能力弱

**问题2：信息冗余（Information Redundancy）**
- 原始 ID Embedding 包含大量冗余信息（两个相近用户的 Embedding 几乎相同）
- 模型需要从冗余信息中提取有用特征，学习效率低
- 特征交叉（Feature Interaction）在冗余 Embedding 上效果受限

**根本原因**：过度依赖原始 ID Embedding 进行特征交叉，没有利用监督信号来优化 Embedding 本身的表达质量。

## 核心方法与创新点

### Supervised Feature Generation（SFG）范式转换

SFG 提出将 CTR 预测从**"特征交互"**范式转换为**"特征生成"**范式：

> **核心思想**：不仅要预测点击，还要**重建/生成**所有特征的 Embedding，迫使模型学到更好的内部表示。

### SFG 框架结构

**Encoder（编码器）**：
为每个输入特征 $x_i$ 构建隐藏表示（Hidden Embedding）：

$$
h_i = \text{Encoder}(e_i)
$$

其中 $e_i$ 是原始 ID Embedding，$h_i$ 是经过编码的隐表示。

**Decoder（解码器）**：
从各特征的隐表示 $\{h_i\}$ 重建所有特征的原始 Embedding：

$$
\hat{e}_j = \text{Decoder}(\{h_i\}_{i=1}^N)
$$

**监督损失（Supervised Loss）**：
与传统生成方法使用自监督损失（重建损失 $\|e - \hat{e}\|^2$）不同，SFG 引入**监督信号（click/no-click）**：

$$
\mathcal{L}_{SFG} = \mathcal{L}_{CTR}(\hat{y}, y) + \lambda \cdot \mathcal{L}_{gen}(\hat{e}, e)
$$

- $\mathcal{L}_{CTR}$：标准 BCE 点击率预测损失（利用 click 标签）
- $\mathcal{L}_{gen}$：特征生成重建损失（自监督或条件生成）

**为什么监督损失更好**：
传统自监督只优化重建质量，而监督信号确保生成的特征表示直接服务于点击率预测任务，两者协同优化使 Embedding 空间更有判别力。

### 与现有工作的区别

| 方法 | 范式 | 监督信号 | 生成目标 |
|------|------|----------|----------|
| DIN/DIEN/DCN | 判别式交互 | Click | N/A |
| BERT4Rec | 生成式 | 自监督 MLM | 序列补全 |
| VAE-CF | 生成式 | 自监督 | 用户行为重建 |
| **SFG** | **生成式** | **Click（监督）** | **特征 Embedding 重建** |

### 可插拔设计（Plug-and-Play）

SFG 是一个**框架（Framework）**而非特定模型：
- 可以无缝集成到现有 CTR 模型（DIN、DCN、DeepFM 等）
- 只需在原有模型基础上增加 Encoder-Decoder 结构和生成损失
- 不改变推理时的网络结构（Decoder 只在训练时使用）

## 实验结论

**公开数据集**：在 Criteo、Avazu、MovieLens、KuaiRec 等标准 CTR 预测数据集上：
- AUC 提升 +0.1%~+0.3% vs 强基线
- Logloss 降低

**Embedding 分析（为什么有效）**：
- 可视化显示 SFG 的 Embedding 维度利用率更高（无坍塌）
- 不同类别商品的 Embedding 在空间中分离度更好

**与不同基础模型集成效果**：
- SFG + DeepFM > DeepFM
- SFG + DCN > DCN
- SFG + Transformer > Transformer  
- 证明了框架的通用性

## 工程落地要点

1. **Decoder 仅训练时使用**：推理阶段不运行 Decoder，不增加在线延迟
2. **损失权重 λ 调参**：CTR 损失 vs 生成损失的权重需要在验证集上调整
3. **特征选择**：并非所有特征都需要重建，优先重建主 ID（用户ID、商品ID）
4. **计算开销**：增加了 Encoder-Decoder，训练时间增加约 30-50%，但推理无额外开销
5. **与预训练结合**：Encoder-Decoder 可以先无监督预训练再 CTR 微调
6. **Embedding 维度**：SFG 对低维 Embedding（如 16D）提升更大，高维（512D）提升较小

## 常见考点

**Q1: 什么是 Embedding Dimensional Collapse？为什么会发生？**
A: 判别式训练时，优化目标只关心类别间的决策边界，导致 Embedding 向量趋于相似方向（秩降低）。这是因为对决策边界没有贡献的维度会被梯度忽略，最终退化到低维流形。

**Q2: SFG 和自监督预训练（BERT-style MLM）的区别？**
A: BERT MLM 用自监督信号（被 mask 的 token 作为标签），与下游任务目标不一致；SFG 使用真实 click 标签作为监督信号，生成损失和 CTR 损失联合优化，特征表示直接服务于点击率预测。

**Q3: CTR 预测从判别式到生成式的范式转换意义是什么？**
A: 生成式范式要求模型能"理解"特征的完整信息（不仅是判别边界），迫使 Embedding 学到更丰富的语义，减少坍塌和冗余。这本质上是引入了信息瓶颈约束，提高了特征表示的信息质量。

**Q4: SFG 框架的局限性是什么？**
A: （1）训练成本增加（Decoder 计算量）；（2）λ 权重难以自动调整；（3）对冷启动 ID 效果有限（没有历史交互，Embedding 依然随机）；（4）Decoder 需要知道所有特征的 Embedding 维度，扩展到新特征时需要修改架构。
