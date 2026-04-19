# FiiNet: A Click-Through Rate Prediction Method Based on Cross-Importance of Multi-Order Features

> 来源：https://arxiv.org/abs/2405.08852 | 领域：ads | 学习日期：20260420

## 问题定义

CTR 预估中特征交叉的两个核心问题：
1. **交叉阶数选择困难**：不同样本需要不同阶数的特征交叉，但现有模型统一处理所有阶数
2. **交叉重要性忽视**：现有模型平等对待所有特征交叉组合，缺乏对"哪些交叉更重要"的建模

## 核心方法与创新点

### 1. SKNet（Selective Kernel Network）用于动态多阶特征交叉

FiiNet 借鉴 CV 领域的 SKNet 思想，将其应用于特征交互重要性的动态学习：

**多阶特征交叉构建**：显式构建 2 阶、3 阶、...、K 阶特征交叉：

$$
z^{(k)} = \text{CrossLayer}_k(x), \quad k = 1, 2, ..., K
$$

**动态重要性选择**：通过 attention 机制对不同阶数的特征交叉赋予自适应权重：

$$
\alpha_k = \frac{\exp(W_k \cdot s)}{\sum_{j=1}^K \exp(W_j \cdot s)}, \quad s = \text{GAP}(\sum_k z^{(k)})
$$

$$
z_{\text{out}} = \sum_{k=1}^K \alpha_k \cdot z^{(k)}
$$

其中 GAP 为全局平均池化，$\alpha_k$ 为第 $k$ 阶交叉的重要性权重。

### 2. 细粒度交互重要性

不仅在阶数维度做选择，还在特征对维度做细粒度重要性评估：

$$
w_{ij} = \text{MLP}(e_i \odot e_j), \quad \text{Interaction}(i,j) = w_{ij} \cdot (e_i \odot e_j)
$$

## 实验结果

- KuaiRec-big 和 Book-Crossing 数据集上 AUC 超过 DeepFM、DCN-V2、FiBiNET 等基线
- LogLoss 也一致更低

## 与同类模型对比

| 模型 | 特征交叉方式 | 阶数感知 | 交互重要性 |
|------|------------|---------|-----------|
| FM | 二阶内积 | 固定 2 阶 | 无 |
| DeepFM | FM + DNN | FM 固定 2 阶 + DNN 隐式高阶 | 无 |
| DCN-V2 | 矩阵 Cross Layer | 逐层递增但无选择 | 无 |
| FiBiNET | SENET 特征重要性 + Bilinear | 无显式阶数 | 特征级重要性 |
| **FiiNet** | **SKNet 多阶 + attention** | **动态阶数选择** | **交叉组合级** |

## 核心 Insight

1. **"不是所有特征交叉都同等重要" 是 CTR 模型的下一个前沿** —— DCN-V2 解决了"高效高阶交叉"，FiiNet 解决了"哪些交叉更值得关注"
2. **SKNet 从 CV 到 CTR 的跨领域迁移** —— SKNet 在 CV 中做多尺度卷积核选择，在 CTR 中做多阶交叉选择，核心思想一致：动态路由
3. **细粒度 > 粗粒度**：交叉组合级重要性 > 特征级重要性 > 无重要性建模

## 面试考点

- Q: FiiNet 与 FiBiNET 的核心区别？
  > FiBiNET 用 SENET 评估**单特征**的重要性，再做 bilinear 交叉；FiiNet 直接评估**特征交叉组合**的重要性，且支持多阶数动态选择。FiiNet 的建模粒度更细。
- Q: 多阶特征交叉的工程问题？
  > K 阶交叉的组合数爆炸（$\binom{n}{k}$），FiiNet 用 Cross Layer 隐式构建而非显式枚举，但高阶仍有计算成本。工业中通常 K ≤ 4。

---

## 相关链接

- [[DLF_dynamic_low_order_aware_fusion_CTR_prediction]] — 动态阶数感知
- [[sfg_supervised_feature_generation_ctr]] — 生成式特征交叉
- [[concepts/attention_in_recsys]] — Attention 在搜广推中的演进
