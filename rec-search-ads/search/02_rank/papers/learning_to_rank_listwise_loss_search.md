# Learning to Rank: Listwise Loss Functions Comparison for Search
> 来源：工业论文/arXiv | 领域：搜索 | 学习日期：20260327

## 问题定义
搜索排序（Learning to Rank, LTR）的损失函数选择直接影响模型效果：
1. **Pointwise**：每个文档独立打分，忽略文档间关系（MSE/交叉熵）
2. **Pairwise**：相对顺序正确即可（RankNet/SVM-Rank），近似NDCG但精度有限
3. **Listwise**：直接优化整个列表的质量指标（NDCG等），最难但理论最优

**目标**：系统比较SoftmaxCE（Softmax Cross-Entropy）和LambdaRank等主流Listwise损失函数，找到工业落地的最佳方案，NDCG+2%。

## 核心方法与创新点

### 1. SoftmaxCE（Softmax交叉熵）
将多文档排序转化为多分类：

$$
\mathcal{L}_{SCE} = -\sum_{i \in \text{rel}} \log \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}
$$

τ 为温度参数，控制分布的平滑程度，较小的τ使分布更尖锐（更关注Top-1）。

**优点**：简单，梯度稳定，容易实现。

### 2. LambdaRank
直接优化NDCG的近似梯度：

$$
\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}}_{\text{{ij}}|
$$

$|\Delta \text{NDCG}}_{\text{{ij}}|$ 表示交换文档i和j后NDCG的变化量，赋予排序错误但NDCG影响大的pair更大的梯度权重。

**优点**：与NDCG高度相关，Top-K文档质量更好。

### 3. ListMLE
最大化正确排列的似然：

$$
\mathcal{L}_{ListMLE} = -\sum_{i=1}^{n} \log \frac{e^{s_{\pi(i)}}}{\sum_{j=i}^{n} e^{s_{\pi(j)}}}
$$

其中 π 是相关性标注定义的理想排序。

### 4. 实验对比框架
在相同的特征、数据、模型架构下，只改变损失函数，控制变量比较：
- 数据集：工业搜索日志（10亿级别）
- 特征：BM25、语义相似度、点击率、新鲜度等
- 模型：2层MLP + 批归一化

## 实验结论
- **SoftmaxCE**：NDCG@10最佳，实现简单，工业推荐首选（+2%相比Pointwise）
- **LambdaRank**：Top-1精度更高，适合强调首位展示的场景
- **ListMLE**：NDCG@5最好，但对标注质量要求高，实现复杂
- 温度参数τ=0.1时SoftmaxCE效果最佳

## 工程落地要点
1. **SoftmaxCE的Query-level归一化**：每个Query的文档集合独立计算Softmax，不同Query间不互相影响
2. **LambdaRank的NDCG截断**：通常只计算Top-10的NDCG变化量，对深位置的文档不施加大梯度
3. **标注质量**：Listwise损失对标注噪声更敏感（点击行为有位置偏差），需要去偏处理
4. **batch构建**：每个batch包含多个Query，每个Query有N个候选文档（N=10~100）
5. **在线评估**：离线NDCG与在线效果（CTR、成交率）可能不一致，以在线A/B为准

## 常见考点
Q1: 为什么Listwise方法比Pointwise和Pairwise更好？
A: Pointwise：每个文档独立打分，完全忽略文档间的相对排序关系，本质上是一个分类或回归问题，与最终的排序指标（NDCG）相关性较弱。Pairwise：只关注相对顺序（文档A排在文档B前面），没有考虑排名位置的重要性（第1名vs第2名的差异远大于第100名vs第101名）。Listwise：直接优化完整列表质量（NDCG），考虑了位置权重，训练目标与评估指标对齐，理论上最优。

Q2: SoftmaxCE中温度参数τ的作用？
A: τ越小，Softmax分布越尖锐，越关注得分最高的文档（相当于强调Top-1的相关性）。τ越大，分布越平滑，对所有文档都有梯度（防止梯度消失）。工业实践：τ=0.1适合强调Top结果质量的搜索场景；τ=1.0适合需要多样性的推荐场景；τ作为超参数需要在验证集上调优。

Q3: 搜索排序中的位置偏差（Position Bias）如何处理？
A: 用户更倾向点击靠前的结果，导致Top位置的文档点击率虚高（不一定因为真正更好）。处理方案：(1)IPW（Inverse Propensity Weighting）：用位置展示概率对点击率进行逆加权修正；(2)TrustBias模型：同时建模审视概率（用户是否看了这个结果）和点击意愿；(3)随机对照实验：定期随机打乱排序，用真实相关性标注消除位置偏差；(4)在损失函数中引入去偏权重。
