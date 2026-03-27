# Align3GR: Multi-Level Alignment for Generative Recommendation with LLM
> 来源：arXiv/工业论文 | 领域：推荐系统 | 学习日期：20260327

## 问题定义
LLM用于生成式推荐面临**对齐（Alignment）**问题：
1. **Item-level对齐**：LLM理解的"好物品"与用户真实偏好不对齐
2. **List-level对齐**：生成的物品列表内部缺乏多样性和互补性
3. **User-level对齐**：生成结果未充分考虑用户的个性化长期偏好
**目标**：设计三层对齐机制，让LLM生成的推荐结果在item、list、user三个粒度上都与用户真实需求对齐。

## 核心方法与创新点

### 1. Item-Level对齐
使用对比学习对齐LLM的item表示与推荐信号：

$$
\mathcal{L}_{item} = -\log \frac{\exp(\text{sim}(h_u, h_{i^+})/\tau)}{\sum_{j} \exp(\text{sim}(h_u, h_{i^j})/\tau)}
$$

正样本 $i^+$ = 用户实际点击/购买的物品，负样本来自曝光未点击。

### 2. List-Level对齐
推荐列表应满足：
- **相关性**：每个item与用户需求相关
- **多样性**：列表内item不过于相似

$$
\mathcal{L}_{list} = \mathcal{L}_{relevance} - \lambda \cdot \text{Diversity}(List)
$$

多样性通过MMR（Maximal Marginal Relevance）度量：

$$
\text{MMR} = \lambda \cdot \text{Relevance}(i, u) - (1-\lambda) \cdot \max_{j \in S} \text{Sim}(i, j)
$$

### 3. User-Level对齐（RLHF）
使用PPO算法，以用户长期满意度为奖励：

$$
r(List) = \text{CTR} \cdot w_1 + \text{Retention} \cdot w_2 + \text{Diversity} \cdot w_3
$$

其中Retention为用户次日留存率，捕获长期满意度。

### 4. 三层联合训练

$$
\mathcal{L}_{total} = \mathcal{L}_{item} + \alpha \mathcal{L}_{list} + \beta \mathcal{L}_{user}
$$

三层损失联合反向传播，层次化对齐LLM的推荐能力。

## 实验结论
- **NDCG@10**：+12%（相比标准LLM推荐基线）
- **多样性**：列表内部相似度降低0.15（更多样化）
- **次日留存**：+3.2%（长期满意度提升）
- 消融实验：三层对齐缺少任一层，效果均下降2-4%

## 工程落地要点
1. **三层训练顺序**：建议先item-level预训练，再list-level微调，最后user-level RLHF
2. **奖励设计**：次日留存奖励需要延迟收集（T+1天），训练时使用离线历史数据的代理奖励
3. **多样性与相关性trade-off**：λ需要根据业务场景调整（内容平台重多样性，电商重相关性）
4. **计算成本**：三层对齐的训练成本约为基线的3x，需要充足GPU资源
5. **在线评估**：多样性指标和长期留存不能只看离线AUC，必须做在线A/B测试

## 面试考点
Q1: 为什么需要List-Level对齐而不只是Item-Level对齐？
A: Item-Level对齐只保证每个单独的item与用户相关，但不考虑列表整体。一个全是篮球鞋的Top-10推荐列表，每个item都很相关，但用户可能只需要一双，其余展示空间被浪费。List-Level对齐通过MMR等方法确保列表内的多样性，让有限的展示位展示更多样的选择，提升整体点击率和满意度。

Q2: User-Level对齐与Item/List-Level的区别？
A: Item/List-Level对齐优化即时反馈（点击），容易陷入短期优化：用户点了很多但很快离开（"好看但没用"）。User-Level对齐引入长期信号（留存率、使用频次），优化用户长期满意度，避免"标题党"式推荐。实践中，长期信号的收集和归因比短期CTR复杂得多。

Q3: 如何处理三个对齐目标的冲突？
A: (1)层次化训练：先稳定lower-level对齐，再加入higher-level；(2)动态权重：用帕累托优化找到多目标的非支配解；(3)硬约束+软目标：设置相关性下限（item-level硬约束），在此基础上最大化多样性和留存（软目标）；(4)定期重新校准：根据在线实验结果调整权重α和β。
