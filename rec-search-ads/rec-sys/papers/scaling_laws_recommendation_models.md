# Scaling Laws for Recommendation Models

> 来源：arxiv 2502.07560 | 领域：rec-sys | 学习日期：20260328

## 问题定义

Scaling Law在语言模型领域已成为基础理论：OpenAI的Chinchilla法则给出了计算量C、参数量N、数据量D的最优分配关系。但**推荐系统是否遵循类似的Scaling Law？** 这是工业界极其关心的问题，因为它直接决定了：

1. 是否值得继续扩大模型规模？（边际收益是否下降）
2. 如何最优地分配计算预算（模型 vs 数据 vs 特征工程）？
3. 小规模实验的结论能否外推到大规模？

**本论文的目标**：系统研究推荐模型的Scaling行为，建立推荐场景的Scaling Law方程。

## 核心方法与创新点

### 1. 推荐Scaling的维度定义

与NLP不同，推荐系统有独特的Scaling维度：

| 维度 | NLP | 推荐系统 |
|------|-----|---------|
| 模型规模 | Transformer参数量 | MLP/MoE参数量 |
| 数据量 | Token数量 | 交互记录数 |
| **特征维度** | N/A | Embedding table大小、特征数量 |
| **ID规模** | N/A | User/Item空间大小 |

### 2. Scaling Law方程推导
$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

其中：
- $L$ 为验证集loss（对数损失/logloss）
- $N$ 为模型参数量
- $D$ 为训练数据量
- $\alpha, \beta$ 为各维度的Scaling指数
- $L_\infty$ 为理论最优（不可压缩的真实随机性）

**推荐系统关键发现**：
- $\alpha_{rec} \approx 0.07$（远小于LLM的$\alpha \approx 0.34$）
- $\beta_{rec} \approx 0.28$（接近LLM的$\beta \approx 0.29$）
- **结论**：推荐模型对模型规模的回报远低于LLM，对数据量的回报与LLM相当

### 3. Embedding Table的独特Scaling效应
$$L_{emb}(V, d) \approx C - \gamma \log(V \cdot d)$$

- $V$：词表大小（item/user数）
- $d$：embedding维度
- Embedding table的Scaling效益显著，且与模型参数Scaling相互独立

### 4. 最优计算分配公式
给定计算预算C（FLOPs），最优分配：
$$N^* = G \cdot C^{a}, \quad D^* = G' \cdot C^{b}$$

实验发现推荐系统最优分配：60%预算用于数据，30%用于embedding，只有10%用于模型参数（与NLP的50-50分配显著不同）。

## 实验结论

在工业级推荐数据集上（亿级样本，千万级item）：
- 模型参数10亿→100亿：AUC提升约**+0.05%**（收益递减明显）
- 数据量10亿→100亿：AUC提升约**+0.15%**（收益更稳定）
- Embedding维度32→512：AUC提升约**+0.12%**（边际效益高）
- **最有价值的投资**：增加高质量训练数据 > 扩大Embedding > 扩大网络参数

## 工程落地要点

1. **计算预算分配指导**：根据Scaling Law，给定GPU计算预算，先扩大训练数据量，再扩大Embedding维度，最后考虑增加模型参数
2. **小实验外推**：在1%数据上跑不同规模模型，拟合Scaling曲线，外推全量数据的预期收益，避免盲目大规模实验
3. **Embedding规模管理**：Item embedding table随业务增长，需定期清理低频item（tail cutting），维持有效参数密度
4. **数据质量的重要性**：Scaling Law假设数据独立同分布，工业推荐数据存在大量噪声（误点击）和分布漂移，需要数据清洗
5. **监控Scaling效益**：上线新模型时，记录模型规模/数据量/AUC的三元组，持续验证是否仍在Scaling Law预测的范围内

## 面试考点

**Q1：推荐系统的Scaling Law与LLM的Scaling Law最大的区别是什么？**
A：关键区别：推荐系统对模型参数的Scaling指数α≈0.07，远小于LLM的0.34，意味着继续扩大模型参数的边际效益更低。推荐系统有独特的Embedding Table Scaling维度（user/item规模驱动），LLM没有对应概念。此外，推荐数据有时效性（concept drift），简单堆数据不如LLM那么有效。

**Q2：如何用Scaling Law指导推荐系统的资源投入决策？**
A：核心思路：在1-10%的数据规模上，分别实验不同模型大小（100M/1B/10B参数），记录AUC，拟合$L = A/N^\alpha + C$曲线，外推全量数据下的收益。若外推的AUC提升<0.01%，则说明扩大该维度性价比低，应转而投入数据或特征工程。

**Q3：为什么推荐系统模型参数的Scaling效益低于LLM？**
A：LLM需要记忆大量世界知识（历史事件、语言规律），更多参数=更多记忆容量，效益高。推荐系统的"知识"主要存在于ID embedding中（用户/item的行为模式），增加网络参数不增加记忆容量，而是增加拟合能力——但推荐任务的固有随机性（用户行为本质上是随机的）限制了拟合上限。

**Q4：Embedding Table大小（V×d）如何影响推荐效果？**
A：V（item/user数）增大：能记忆更多个体差异，减少hash collision；d（维度）增大：每个entity的表征更丰富，能区分更细粒度的特征。两者的Scaling效益相互独立，通常$V \cdot d$的log成正比于性能提升。工业实践中，维度从64增到256效益显著，256到512边际减少，512以上几乎无收益。

**Q5：数据量Scaling在推荐系统中有哪些特殊挑战？**
A：(1) 时效性：旧数据分布漂移，不能无限累积；(2) 正负样本不均衡：大量曝光未点击，随机采样导致正样本比例极低；(3) 反馈延迟：购买行为可能在曝光数天后才发生，实时Scaling困难；(4) 数据噪声：机器人流量、误点击污染数据质量。因此推荐的"有效数据量"比原始日志量小得多。
