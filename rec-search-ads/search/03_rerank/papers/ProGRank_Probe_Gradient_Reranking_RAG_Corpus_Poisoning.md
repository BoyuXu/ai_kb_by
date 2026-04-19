# ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning

> arXiv:2603.22934 | 2026-03 | 领域：RAG安全/重排序

## 一句话总结

ProGRank 是一种 training-free 的重排序防御方法，通过对 query-passage 对施加随机扰动并分析 probe gradient 的不稳定性来检测 corpus poisoning 攻击注入的恶意文档。

## 问题背景

RAG 系统通过检索外部文档增强生成可靠性，但引入了新攻击面：**corpus poisoning**——攻击者注入/修改文档，使其被检索到 Top-K 并影响下游生成。

现有防御的局限：
- 需要重新训练检索器（成本高）
- 需要标注的 poison 样本（不现实）
- 仅防御特定攻击类型

## 核心方法

### 关键直觉

优化驱动的 poison 文档将检索能力集中在少数 perturbation-sensitive 的匹配信号上 → 在随机扰动下，梯度响应比正常文档更不稳定。

### 两个不稳定性信号

1. **Representational Consistency（表示一致性）**：poison 文档在扰动下 embedding 变化更大
2. **Dispersion Risk（离散风险）**：poison 文档的梯度在扰动下方差更大

### 流程

```
Query + Top-K Passages
    ↓ 对每个 passage
    施加 M 次随机扰动
    ↓
    提取 probe gradients（仅小部分参数）
    ↓
    计算 consistency score + dispersion score
    ↓
    Score Gate：综合两个信号
    ↓
    Rerank Top-K，降低可疑 passage 排名
```

### Surrogate 变体

当部署的检索器不可用时（黑盒场景），可用替代模型的梯度进行近似检测。

## 实验结果

- 3 个数据集 + 3 个 dense retriever backbone
- 检索阶段和端到端生成阶段均有效
- 提供了**鲁棒性-效用 trade-off**：在防御 poison 的同时不显著降低正常检索质量

## 核心价值

| 特性 | 说明 |
|------|------|
| Training-free | 无需重训检索器 |
| Post-hoc | 作为重排步骤即插即用 |
| 攻击无关 | 不针对特定攻击类型 |
| 内容保持 | 不修改原始文档 |
| 黑盒兼容 | 支持 surrogate 模型 |

## 与其他工作的关系

- 与 [[检索三角_Dense_Sparse_LateInteraction|检索三角]] 中的 dense retrieval 防御相关
- RAG 安全与 [[2026-04-09_rag_systems_evolution|RAG 系统演进]] 中的鲁棒性维度对应
- 重排序技术见 [[搜索Reranker演进|搜索Reranker演进]]
