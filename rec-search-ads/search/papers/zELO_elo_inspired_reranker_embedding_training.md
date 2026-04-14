# zELO: ELO-inspired Training Method for Rerankers and Embedding Models

> 来源：arXiv:2509.12541 | 领域：search/reranking | 学习日期：2026-04-11

## 问题定义

现有重排器和 Embedding 模型的训练依赖人工标注的相关性标签，成本高且难以覆盖多领域。如何从**无标注数据**中高效训练出超越闭源方案的开源重排器？

## 核心方法

zELO 的核心洞察：排序任务在统计上等价于 **Thurstone 模型**（正态分布噪声假设，优于 Gumbel 分布的 Bradley-Terry 模型，由中心极限定理支撑）。

训练流程：
1. **稀疏成对偏好收集**：用多个大 LLM 集成（ensemble）对 query-document 对进行成对比较
2. **Thurstone 模型转换**：将稀疏成对偏好转换为每个文档的**绝对相关性分数**（zELO score）
3. **Pointwise 微调**：用 zELO 分数端到端训练 pointwise 重排器

$$
\text{zELO}(d|q) = \text{Thurstone}^{-1}(\text{PairwisePrefs}_{LLM}(q, d_i, d_j))
$$

## 关键创新

- **无需人工标注**：完全从 LLM 集成的成对偏好中提取监督信号
- **Thurstone > Bradley-Terry**：正态噪声假设比 Gumbel 更符合排序数据特征
- **Pointwise 训练**：将成对信号转为绝对分数，避免 pairwise 训练的 O(n²) 组合爆炸
- **多领域零样本泛化**：在金融、法律、代码、STEM 等领域均 SOTA

## 实验亮点

- 数据规模：112K queries × 100 docs/query，纯无标注
- 训练成本：< 10,000 H100-hours（端到端）
- zerank-1 在 NDCG@10 和 Recall 上超越所有闭源重排器
- 零样本迁移到私有客户数据集，性能不退化

## 工业价值

解决了"好的重排器需要昂贵标注"的核心痛点。LLM-as-Judge + Thurstone 模型的组合，为无监督训练高质量重排器提供了标准化流程。

[[推理增强检索技术综述]] | [[embedding_everywhere]]
