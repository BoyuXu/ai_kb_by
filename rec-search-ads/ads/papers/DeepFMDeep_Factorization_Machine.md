# DeepFM：深度因子分解机（Deep Factorization Machine）

> 来源：工程实践 / IJCAI 2017 华为 | 日期：20260317

## 问题定义

广告 CTR 预估的核心挑战是**高阶特征交叉**。用户-广告交互依赖复杂的跨特征组合（如"女性+25岁+购物App"更容易点击"美妆广告"），手动构造交叉特征耗时且不完备。FM 能自动学习二阶交叉但无法建模高阶；DNN 能学习高阶但忽略低阶显式交叉；Wide&Deep 需要手动设计 Wide 侧特征。DeepFM 统一 FM 的显式低阶交叉和 DNN 的隐式高阶交叉，无需手工特征工程。

## 核心方法与创新点

1. **FM 组件（显式二阶交叉）**
   $$y_{FM} = \langle w, x \rangle + \sum_{i < j} \langle v_i, v_j \rangle x_i x_j$$
   - 所有特征共享 embedding $V \in \mathbb{R}^{n \times k}$

2. **Deep 组件（隐式高阶交叉）**
   - 将所有 field 的 embedding 拼接：$a^{(0)} = [e_1, e_2, ..., e_m]$
   - 经过多层 MLP：$a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)})$

3. **共享 Embedding（关键创新）**
   - FM 组件和 Deep 组件共享同一套 Embedding $V$
   - 无需预训练，端到端联合训练，FM 梯度和 DNN 梯度共同更新 embedding
   - Wide&Deep 中 Wide 侧需要手工特征，DeepFM 完全自动化

4. **输出层合并**
   $$\hat{y} = \sigma(y_{FM} + y_{DNN})$$

## 实验结论

- Criteo 数据集 AUC: DeepFM 0.8007 vs Wide&Deep 0.8000 vs FM 0.7940
- Company* 数据集（华为 App Store）AUC 提升 0.25%，转化率提升 3.1%（上线效果）
- 训练速度比 Wide&Deep 快（无手工交叉特征处理）

## 工程落地要点

1. **Field 划分**：每个语义独立的特征维度作为一个 field（用户ID、广告ID、类目等），field 间 embedding 维度可不同
2. **Embedding 维度**：通常 4~16 维，广告/用户 ID 可更大（16~64）
3. **Dense 特征处理**：连续特征需离散化（分桶）后作为 field，或直接输入 DNN 绕过 FM 组件
4. **正则化**：Dropout 在 DNN 层有效，Embedding L2 正则防止过拟合
5. **工程变体**：xDeepFM（显式高阶交叉）、DCN（Cross Network）、AutoInt（Attention 交叉）

## 面试考点

- **Q: DeepFM 相比 Wide&Deep 的核心优势？**
  A: DeepFM 的 FM 侧不需要手工构造交叉特征，且 FM 和 DNN 共享 Embedding，参数更高效，梯度信号更充分。Wide&Deep 的 Wide 侧需要领域知识设计交叉特征。

- **Q: FM 中为什么用内积 $\langle v_i, v_j \rangle$ 而不是直接学习 $w_{ij}$？**
  A: 直接学 $w_{ij}$ 参数量是 $O(n^2)$，稀疏数据下大多数 $(i,j)$ 对出现次数极少无法学习。用 latent vector 内积把参数量降到 $O(nk)$，且通过共现泛化到未见过的特征对。

- **Q: DeepFM 中 FM 组件和 DNN 组件学到的是什么？**
  A: FM 学习**显式二阶特征交叉**（如用户性别和商品类目的交互）；DNN 通过深层非线性变换学习**隐式高阶特征交叉**（多个特征的复杂组合模式）。
