# 冷启动、GNN 与因果推断

> **创建日期**: 2026-04-13 | **合并来源**: 推荐系统冷启动, 图神经网络在推荐中的应用, 推荐系统因果推断, 推荐广告AB测试与在线实验
>
> **核心命题**: 冷启动/GNN/因果推断/AB 测试是推荐系统的四大横切关注点，贯穿召回到重排全链路

---

## 一、冷启动

### 1.1 冷启动分类

| 场景 | 问题 | 核心解法 |
|------|------|---------|
| 新用户 | 无历史行为 | 人口统计特征 + 热门推荐 + Exploration |
| 新物料 | 无交互数据 | 内容特征 + 流量扶持 + 生成式方法 |
| 新场景 | 无训练数据 | 跨域迁移 + Meta-Learning |

### 1.2 新用户冷启动

**Exploration 策略**：
- $\varepsilon$-greedy：以 $\varepsilon$ 概率随机探索
- UCB (Upper Confidence Bound)：$\text{score} = \hat{\mu} + c\sqrt{\frac{\ln t}{n}}$，平衡利用与探索
- Thompson Sampling：从后验分布采样，贝叶斯探索

**特征替代**：用人口统计特征（年龄/性别/城市）+ 设备信息 + 注册来源构建初始用户画像。

### 1.3 新物料冷启动

**内容特征方案**：
- 文本 embedding（标题/描述 → BERT/LLM encoding）
- 图像 embedding（封面图 → ResNet/CLIP encoding）
- 类目/标签/属性等结构化特征

**DropoutNet**：训练时随机 dropout 物品 ID embedding，迫使模型学习利用内容特征。推理时新物品无 ID 也能工作。

**IDProxy (小红书)**：多模态 LLM 生成代理 embedding，对齐到 warm ID 空间。冷启动 CTR +8.5%。

**生成式冷启动**：
- Semantic ID 使相似物品共享前缀（新物品前缀匹配已知类别），冷启动效果 +12%
- 关键发现：标识符设计是关键，非模型规模

### 1.4 新场景冷启动

**Foundation Model 迁移**：大规模预训练 + Adapter 快速 finetune，减少 70% 标注需求。

**Meta-Learning**：MAML 风格，在多个场景上学习快速适应能力，新场景只需少量数据微调。

---

## 二、图神经网络在推荐中的应用

### 2.1 核心模型

| 模型 | 年份 | 核心创新 | 应用场景 |
|------|------|---------|---------|
| NGCF | 2019 | 图卷积建模协同信号 | 通用推荐 |
| LightGCN | 2020 | 简化 NGCF（去掉变换+激活） | 通用推荐 |
| PinSage | 2018 | 工业级图采样 + GraphSAGE | Pinterest 20亿 pin |
| EGES | 2018 | Side Info 图嵌入 | 阿里冷启动 |
| SGL | 2021 | 图对比学习增强 | 数据稀疏场景 |
| BiGEL | 2025 | 多行为图嵌入 | 多行为多任务 |

### 2.2 LightGCN

$$
e_u^{(l+1)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)|} \sqrt{|\mathcal{N}(i)|}} e_i^{(l)}
$$

$$
e_u = \sum_{l=0}^{L} \alpha_l e_u^{(l)}
$$

**核心简化**：去掉 NGCF 中的特征变换矩阵和非线性激活，只保留邻域聚合。实验证明这些"标配"组件在协同过滤场景中反而有害。

### 2.3 PinSage — 工业级图推荐

- **Random Walk 采样**：从目标节点出发随机游走，按访问频率选 Top-K 邻居（比全邻居聚合高效）
- **GraphSAGE 聚合**：Mean/LSTM/Pool 三种聚合函数
- **工业规模**：20亿 pin，数十亿边，分布式训练
- **MapReduce 兼容**：不需要整图存储在内存中

### 2.4 SGL — 图对比学习

$$
\mathcal{L}_{\text{SGL}} = -\log \frac{\exp(\text{sim}(z_u, z_u') / \tau)}{\sum_{v} \exp(\text{sim}(z_u, z_v') / \tau)}
$$

三种图增强方式：
- Node Dropout：随机删除节点
- Edge Dropout：随机删除边
- Random Walk：随机子图采样

在数据稀疏场景下有效，提供正则化效果。

### 2.5 BiGEL — 多行为图嵌入

- 级联门控反馈：利用行为间层级关系（view → click → purchase）
- R-GCN 图嵌入捕捉多跳协同信号
- 购买行为（最稀疏）AUC +1.8%，说明级联门控让稀疏行为借用丰富行为信息
- 可作为生成式推荐模型的增强输入

### 2.6 GNN 与生成式推荐的结合

- BiGEL 图嵌入作为 side feature 注入 HSTU
- 或用图嵌入初始化 UniGRec 的 soft identifiers
- 图嵌入离线预计算（每小时更新），不增加在线延迟

---

## 三、因果推断

### 3.1 推荐系统中的偏差类型

| 偏差类型 | 原因 | 影响 |
|---------|------|------|
| Position Bias | 位置本身影响点击 | 高位物品 CTR 虚高 |
| Exposure Bias | 只观测被曝光样本 | 训练数据分布偏移 |
| Selection Bias | 用户自选行为导致 | 无法估计反事实结果 |
| Popularity Bias | 热门物品过度曝光 | 长尾物品被忽视 |
| Conformity Bias | 从众效应 | 高分物品更容易获得好评 |

### 3.2 IPS (Inverse Propensity Scoring)

$$
\hat{R} = \frac{1}{n} \sum_{(u,i) \in \mathcal{O}} \frac{r_{u,i}}{p_{u,i}}
$$

$p_{u,i}$: 物品 $i$ 被展示给用户 $u$ 的倾向分数。对低曝光物品赋予更高权重，校正 exposure bias。

**问题**：$p_{u,i}$ 很小时方差爆炸。

### 3.3 SNIPS (Self-Normalized IPS)

$$
\hat{R}_{\text{SNIPS}} = \frac{\sum_{(u,i) \in \mathcal{O}} r_{u,i} / p_{u,i}}{\sum_{(u,i) \in \mathcal{O}} 1 / p_{u,i}}
$$

自归一化降低方差，但引入少量偏差。

### 3.4 DR (Doubly Robust)

$$
\hat{R}_{\text{DR}} = \frac{1}{n} \sum_{(u,i)} \left[\hat{r}_{u,i} + \frac{o_{u,i}}{p_{u,i}}(r_{u,i} - \hat{r}_{u,i})\right]
$$

结合 IPS 和 imputation model（$\hat{r}_{u,i}$）。只要倾向分数或 imputation 之一正确，估计就无偏。

### 3.5 DML (Double Machine Learning)

用 ML 模型估计 nuisance parameters（倾向分数和 outcome model），再用 Neyman 正交条件保证因果效应估计的一致性。

### 3.6 Position Bias 处理

**快手 D2Q**：将 position feature 从 document 侧解耦，训练时带入 position feature，推理时置为统一默认值。

**抖音位置去偏**：position tower 独立建模位置效应，主 tower 学习真实的物品-用户匹配。

### 3.7 因果竞价

普通回归出价存在选择偏差——只观测"用当时策略出价 b"的结果。因果竞价引入 IPW + DR 估计反事实：如果出价不同会怎样？

---

## 四、A/B 测试与在线实验

### 4.1 核心方法

| 方法 | 适用场景 | 优劣 |
|------|---------|------|
| 经典 A/B | 独立流量分桶 | 金标准，但需要大流量 |
| Interleaving | 搜索/排序对比 | 所需样本量小，灵敏度高 |
| MAB (Multi-Armed Bandit) | 持续优化 | 探索+利用同时进行 |
| Switchback | 双边市场 | 处理 interference |

### 4.2 样本量计算

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}
$$

- $\delta$: 最小可检测效应 (MDE)
- $\sigma^2$: 指标方差
- $z_{\alpha/2}, z_\beta$: 显著性和功效的临界值

### 4.3 常见陷阱

| 陷阱 | 说明 | 解法 |
|------|------|------|
| Peeking | 提前看结果导致假阳性 | 用 Sequential Testing |
| Novelty Effect | 新策略因新鲜感短期指标好 | 延长实验周期 |
| Network Effect | 用户间相互影响 | Cluster-based 分桶 |
| SRM (Sample Ratio Mismatch) | 实验/对照比例偏移 | 数据质量检查 |
| Simpson's Paradox | 分组看和整体看结论相反 | 分层分析 |

### 4.4 离线 vs 在线指标 Gap

离线指标（AUC/NDCG）和在线指标（CTR/GMV/留存）经常不一致。原因：
- 离线评估用历史数据，存在 exposure bias
- 在线用户行为受多因素影响（UI、网速、心情）
- 位置偏差在离线中难以完全校正

最佳实践：离线筛选 → 小流量 A/B → 大流量验证 → 全量上线

### 4.5 Exploration 价值评估

用 Counterfactual Estimation：在历史出价分布下估计"如果以不同策略会发生什么"。
- IPW 校正 selection bias
- 用长期 LTV 而非短期 CTR 评估

---

## 五、面试高频考点

**Q1: 新物料冷启动的三层解决方案？**
A: (1) 内容特征替代 ID（文本/图像 embedding）；(2) DropoutNet 训练时随机 dropout ID，迫使学习内容特征；(3) IDProxy 用多模态 LLM 生成代理 embedding 对齐到 warm 空间。Semantic ID 使相似物品共享前缀，冷启动 +12%。

**Q2: LightGCN 为什么去掉变换和激活？**
A: 协同过滤场景中，用户-物品交互图的信号主要在图结构中（谁买了什么），特征变换和非线性激活反而引入噪声。LightGCN 只保留邻域聚合，实验证明更简洁更有效。

**Q3: PinSage 如何处理 20 亿节点的图？**
A: (1) Random Walk 采样选 Top-K 邻居（不是全邻居）；(2) Mini-batch 训练，不需要整图在内存；(3) MapReduce 兼容的分布式架构；(4) 硬负例挖掘提升区分度。

**Q4: IPS 的核心问题及改进？**
A: 核心问题：倾向分数很小时方差爆炸。改进：(1) SNIPS 自归一化降低方差；(2) DR Doubly Robust 结合 IPS 和 imputation，只要一个正确就无偏；(3) 倾向分数裁剪（clipping）防止极端值。

**Q5: Position Bias 和 Exposure Bias 区别？**
A: Position Bias：相同物品不同位置 CTR 不同（位置本身影响点击）。Exposure Bias：模型只学习被曝光样本，未曝光的表现未知（训练数据分布偏差）。两者都需要 IPS/反倾向加权校正。

**Q6: A/B 测试中 SRM 是什么？为什么重要？**
A: Sample Ratio Mismatch，实验/对照组样本比例与预期不一致。可能原因：分流 bug、数据管道丢失、用户行为差异。SRM 存在时实验结论不可信，必须先排查修复。

**Q7: 为什么离线指标好但在线指标没提升？**
A: (1) Exposure bias：离线用历史数据，分布与在线不同；(2) 位置偏差离线难完全校正；(3) 用户行为受多因素影响；(4) 离线评估的候选集与在线不同。最佳实践：离线筛选 + 小流量 A/B 验证。

**Q8: GNN 如何与生成式推荐结合？**
A: 图嵌入捕捉全局协同信号，与 HSTU 的序列建模互补。两种结合方式：(1) 作为 side feature 注入（FFN 融合到 token 表示）；(2) 作为 soft token 初始化（UniGRec 框架）。图嵌入离线预计算，不增加在线延迟。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
