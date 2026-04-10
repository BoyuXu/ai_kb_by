# ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval

> 来源：arXiv:2510.08252 | 领域：search/embedding | 学习日期：2026-04-11

## 问题定义

推理密集型检索（科学/编程/数学领域）中，文档相关性取决于**复杂语义或逻辑关联**而非直接文本重叠。现有 Embedding 模型在 BRIGHT benchmark 上表现不佳，核心瓶颈是训练数据的「平凡性问题」——合成数据中 query-document 对太容易匹配，模型学不到深层推理关联。

## 核心方法

三大技术贡献：

### 1. ReMixer（数据合成）
解决合成训练数据的平凡性问题，三阶段流程：
- **条件化 Query 生成**：基于文档内容生成需要推理才能关联的 query
- **源排除候选挖掘**：挖掘与 query 语义相近但非源文档的 hard negative
- **推理增强相关性标注**：用 LLM 推理链判断真实相关性

产出 82K 高质量训练样本。

### 2. Redapter（自适应学习）
动态调整每个训练样本的权重，基于其「推理强度」：

$$
w_i = f(\text{reasoning\_intensity}(q_i, d_i))
$$

推理强度高的样本获得更大权重，让模型聚焦于真正需要推理能力的困难样本。

### 3. 多骨干实现
在不同规模骨干（含 Qwen3-8B）上实现，均超越同规模基线。

## 关键创新

- **ReMixer 解决数据平凡性**：这是推理 Embedding 训练的关键瓶颈，之前的方法（如 ReasonIR）未专门解决
- **Redapter 自适应加权**：不是所有样本都需要推理能力，动态加权比均匀采样更高效
- **全部开源**：数据 + 模型 + 代码

## 实验亮点

- **BRIGHT benchmark NDCG@10 = 38.1**（ReasonEmbed-Qwen3-8B），显著超越所有现有 Embedding 模型
- 多规模骨干一致性提升，证明方法通用性

## 面试关键点

ReasonEmbed vs O1-Embedder：O1-Embedder 在推理链上做 query 增强（推理时开销），ReasonEmbed 在训练数据和学习算法上下功夫（训练时开销，推理时零额外成本）。

[[推理增强检索技术综述]] | [[concepts/embedding_everywhere]] | [[concepts/attention_in_recsys]]
