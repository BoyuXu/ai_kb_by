# DIN：深度兴趣网络（Deep Interest Network）

> 来源：工程实践 / KDD 2018 阿里巴巴 | 日期：20260317

## 问题定义

电商推荐中，用户历史行为序列（点击、购买）蕴含丰富兴趣信息。传统 DNN 将行为序列 pooling 为固定向量，**忽略了用户对不同候选商品的兴趣是多样且局部激活的**。用户历史 1000 条行为中，真正与当前候选相关的可能只有 10~20 条，统一 pooling 会引入大量噪声。

## 核心方法与创新点

1. **注意力机制（Attention-based Pooling）**
   - 对每条历史行为 $e_i$，计算与候选商品 $e_a$ 的相关性得分：

$$
a_i = \text{Attention}(e_i, e_a) = \text{MLP}([e_i; e_a; e_i \odot e_a; e_i - e_a])
$$

   - 加权求和：$v_u = \sum_i a_i \cdot e_i$
   - 不做 softmax normalization（保留原始得分幅度，反映绝对兴趣强度）

2. **局部激活（Local Activation）**
   - 每次预测时根据候选动态计算权重，不预计算用户表征
   - 有效捕获"用户最近看了什么就对什么感兴趣"的局部激活特性

3. **训练技巧**
   - **GAUC（Group AUC）**：按用户分组计算 AUC 再平均，更能反映推荐个性化能力
   - **Mini-batch Aware Regularization**：只对 mini-batch 中出现的 Embedding 做正则化，避免稀疏更新时 L2 正则过度惩罚未出现 ID
   - **Adaptive Activation（Dice）**：替代 PReLU，自适应决定折点位置

4. **特征工程**
   - 行为序列：商品ID、类目ID、品牌ID 拼接
   - 候选商品：同维度特征
   - Cross feature：历史行为与候选的交叉特征作为 attention 输入

## 实验结论

- 淘宝 CTR 任务 GAUC 提升约 0.5%（相对提升显著）
- 与 Base DNN 相比，在长序列场景提升更明显（序列长度>50）
- Dice 激活函数相比 PReLU 提升约 0.1% AUC

## 工程落地要点

1. **序列长度截断**：通常取最近 50~200 条行为，过长序列在线计算 attention 代价高
2. **离线/在线一致性**：Attention 计算在线进行，需保证特征抽取与训练时一致
3. **Embedding 维度**：商品 ID embedding 通常 64~128 维，类目 16~32 维
4. **候选物 Lookup 缓存**：候选 embedding 可预先 batch lookup，减少在线 IO
5. **延伸**：DIN → DIEN（引入序列演化 GRU）→ DIN-SQL（长序列用 SIM 检索）

## 常见考点

- **Q: DIN 中 attention 为什么不用 softmax？**
  A: softmax 会归一化权重之和为 1，使得绝对兴趣强度信息丢失。若用户对候选无兴趣，softmax 仍会给某些行为高权重；不 softmax 可以保留"用户总体兴趣强度"信息。

- **Q: GAUC 和 AUC 的区别？**
  A: AUC 全局计算，高活跃用户样本多会主导指标；GAUC 先按用户分组分别计算 AUC，再按用户样本量加权平均，更能反映个性化效果，消除用户间偏差。

- **Q: Mini-batch Aware Regularization 解决什么问题？**
  A: 推荐系统 Embedding 极度稀疏，若对所有 Embedding 做 L2 正则，每个 step 都会对未出现的 ID 做梯度下降，导致这些 ID 的 embedding 不断缩小趋向 0。Mini-batch Aware 只对本 batch 出现的 ID 做正则。
