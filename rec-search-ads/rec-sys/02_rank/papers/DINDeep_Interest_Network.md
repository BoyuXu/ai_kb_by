# DIN：深度兴趣网络（Deep Interest Network）

> 来源：KDD 2018, Alibaba | 年份：2018 | 领域：rec-sys/02_rank（CTR预估/序列建模）

## 问题定义

电商推荐中，用户历史行为序列（点击、购买）蕴含丰富兴趣信息。传统 DNN 将行为序列 pooling 为固定向量，**忽略了用户对不同候选商品的兴趣是多样且局部激活的**。

- 用户历史 1000 条行为中，真正与当前候选相关的可能只有 10-20 条
- 统一 avg pooling 会引入大量噪声，且不同候选应激活不同的历史行为
- **核心思想**：用候选商品作为 query，对历史行为做 attention，动态提取与当前候选最相关的兴趣

## 模型结构图

```
┌──────────────────────────────────────────────────────┐
│                  DIN 模型架构                          │
│                                                      │
│                   Output: pCTR                       │
│                      ↑                               │
│                  σ(FC Layers)                         │
│                      ↑                               │
│             ┌────────┴────────────┐                  │
│             │      Concat          │                 │
│        ┌────┴────┐          ┌─────┴──────┐          │
│        │ Other   │          │ User       │          │
│        │ Features│          │ Interest   │          │
│        └────────┘          │ v_u        │          │
│                             └─────┬──────┘          │
│                              Σ(aᵢ × eᵢ)             │
│                          ┌────┼────┐                │
│                         w₁   w₂   w₃               │
│                          ↑    ↑    ↑                │
│                    ┌─────┴────┴────┴─────┐          │
│                    │  Activation Unit     │          │
│                    │  a = σ(MLP([eᵢ;eₐ;  │          │
│                    │   eᵢ-eₐ; eᵢ⊙eₐ]))  │          │
│                    └──────────┬───────────┘          │
│                               ↑                     │
│                    ┌──────────┼──────────┐           │
│                    ↑          ↑          ↑           │
│                 [e₁]       [e₂]       [e₃]          │
│                 Behav₁     Behav₂     Behav₃        │
│                                ↑                     │
│                          Target Item eₐ              │
└──────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：Attention Score（局部激活单元）

$$
a_i = f(e_i, e_a) = \sigma(W \cdot [e_i; e_a; e_i - e_a; e_i \odot e_a] + b)
$$

**解释：**
- $e_i$：第 $i$ 个历史行为的 embedding
- $e_a$：候选商品的 embedding
- $[;]$：向量拼接
- $e_i - e_a$：差值向量，显式建模差异
- $e_i \odot e_a$：逐元素乘积，显式建模交互
- $\sigma$：sigmoid（**不是 softmax**）
- 不用 softmax 保留绝对兴趣强度信息

### 公式2：用户兴趣表示

$$
v_u = \sum_{i=1}^{N} a_i \cdot e_i
$$

**解释：**
- $N$：历史行为序列长度
- $a_i$：attention 权重（非归一化）
- 不同候选商品产生不同的 $v_u$，实现"局部兴趣激活"

### 公式3：Dice 激活函数

$$
\text{Dice}(x) = p(x) \cdot x + (1-p(x)) \cdot \alpha x
$$

$$
p(x) = \frac{1}{1 + e^{-\frac{x-E[x]}{\sqrt{Var[x]+\epsilon}}}}
$$

**解释：**
- 相比 PReLU 的固定拐点 $x=0$，Dice 拐点自适应于数据分布均值 $E[x]$
- $\alpha$：可学习参数，控制负半轴斜率

### 公式4：Mini-batch Aware Regularization

$$
L_{reg} = \sum_{(x,y) \in B} \sum_{j=1}^{K} \frac{\|w_j\|^2}{n_j}
$$

**解释：**
- $n_j$：特征 $j$ 在 batch 中出现的次数
- 高频特征正则化强度小，低频特征正则化强度大
- 解决推荐系统 embedding 极度稀疏的长尾问题

## 与基线方法对比

| 方法 | 序列建模 | 动态兴趣 | 复杂度 | 适用场景 |
|------|---------|---------|--------|---------|
| **Avg Pooling** | 等权聚合 | 无 | $O(1)$ | 简单场景 |
| **DIN** | Target Attention | 有 | $O(L)$ | 精排（L≤200） |
| **DIEN** | GRU + Attention | 有+时序 | $O(L)$ | 精排（兴趣演化） |
| **BST** | Transformer | 有 | $O(L^2)$ | 精排（丰富交互） |
| **SIM** | 检索 + DIN | 有 | $O(K)$ | 精排（L>10000） |

## 实验结论

- 淘宝 CTR 任务 GAUC 提升约 0.5%（相对提升显著）
- 长序列场景（序列长度>50）提升更明显
- Dice 激活函数 vs PReLU：+0.1% AUC
- Mini-batch Aware Reg vs 标准 L2：+0.1% AUC
- 在线 A/B：CTR +10%，RPM +3.8%

## 工程落地要点

1. **序列长度截断**：通常取最近 50-200 条行为，超长序列需要 SIM 等检索方案
2. **离线/在线一致性**：Attention 在线计算，需保证特征抽取与训练时一致
3. **Embedding 维度**：商品 ID embedding 通常 64-128 维，类目 16-32 维
4. **候选 Embedding 缓存**：候选 embedding 可 batch lookup 预加载，减少在线 IO
5. **行为去噪**：过滤误点击、机器行为、停留极短的行为

## 面试考点

**Q1：DIN 中 attention 为什么不用 softmax？**
> softmax 归一化权重之和为 1，使绝对兴趣强度信息丢失。若用户对候选无兴趣，softmax 仍会给某些行为高权重。不 softmax 保留"总体兴趣强度"信息，是"局部激活"而非"全局竞争"的设计。

**Q2：GAUC 和 AUC 的区别？**
> AUC 全局计算，高活跃用户样本多会主导指标。GAUC 先按用户分组分别计算 AUC，再按用户样本量加权平均，更能反映个性化效果，消除用户间偏差。

**Q3：Mini-batch Aware Regularization 解决什么问题？**
> 推荐系统 Embedding 极度稀疏，标准 L2 正则对所有 Embedding 做梯度下降，未出现的 ID 的 embedding 不断缩小趋向 0。MBA-Reg 只对 batch 内出现的 ID 做正则，且频次越低正则越强。

**Q4：DIN 的 attention 输入为什么拼接 $e_i - e_a$ 和 $e_i \odot e_a$？**
> 显式提供差异和交互信息，帮助 attention 网络更容易学到"相似"和"差异"模式。仅拼接 $[e_i; e_a]$ 理论上 MLP 能学到，但显式特征加速收敛且效果更好（类似 NLP 中 ESIM）。

**Q5：DIN → DIEN → SIM 的演进路线？**
> DIN：引入 target attention，解决兴趣多样性。DIEN：加入 GRU 捕捉兴趣时序演化（AUGRU = attention-based GRU）。SIM：用检索（倒排索引/类目匹配）从超长序列（10000+）中检索 TopK 相关行为，再做 DIN attention。

**Q6：DIN 能否用于召回阶段？**
> DIN 本身是精排模型（需每个候选分别计算 attention）。但其思想可迁移到召回：MIND 用胶囊网络生成多个兴趣向量，每个向量独立做 ANN 检索，实现"多兴趣召回"。
