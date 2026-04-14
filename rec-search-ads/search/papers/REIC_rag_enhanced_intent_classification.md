# REIC: RAG-Enhanced Intent Classification at Scale

> 来源：arXiv:2506.00210 | 领域：search/classification | 学习日期：2026-04-11
> 发表于 EMNLP 2025 Industry Track (Amazon)

## 问题定义

客服路由需要精准的意图分类，但企业产品线扩张导致意图类别持续增长、不同业务线的分类体系各异。传统微调方法需要频繁重训，无法适应规模化动态变化。

## 核心方法

REIC 用 RAG 替代传统分类器的静态参数化，实现**无需重训的动态意图分类**：

1. **索引构建（Index Construction）**：将标注数据集中的 (query, intent) 对用 sentence transformer 编码为密集向量，存入向量索引
2. **候选检索（Candidate Retrieval）**：新 query 编码后在索引中检索 Top-K 相似样本
3. **意图概率计算（Intent Probability）**：基于检索到的 K 个近邻的意图分布计算最终分类概率

$$
P(\text{intent}_i | q) = \frac{\sum_{k=1}^{K} \text{sim}(q, d_k) \cdot \mathbb{1}[\text{intent}(d_k) = i]}{\sum_{k=1}^{K} \text{sim}(q, d_k)}
$$

## 关键创新

- **免重训适应新意图**：新增意图只需往索引中添加样本，无需重新训练模型
- **跨垂域泛化**：同一模型通过不同索引服务不同业务线
- **RAG 用于分类（非生成）**：将 RAG 范式从问答/生成扩展到分类任务

## 实验亮点

- 在大规模真实客服数据上超越传统 fine-tuning、zero-shot、few-shot 方法
- In-domain 和 out-of-domain 场景均有效
- 显著降低模型更新频率和运维成本

## 工业价值

REIC 的核心启示：**向量索引可以替代分类头**。在意图动态变化的场景（客服/搜索意图识别），用检索+投票的方式比训练分类器更灵活。这与推荐系统中的"双塔召回即分类"异曲同工。

[[LLM增强信息检索与RAG技术进展]] | [[embedding_everywhere]]
