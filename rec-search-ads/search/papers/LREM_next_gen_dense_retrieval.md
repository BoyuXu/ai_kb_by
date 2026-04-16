# Large Reasoning Embedding Models: Towards Next-Generation Dense Retrieval Paradigm

> 来源：arXiv:2510.14321 | 领域：search/dense_retrieval | 学习日期：2026-04-11
> 已部署于中国最大电商平台（2025年8月上线）

## 问题定义

当前密集检索模型（BERT→LLM backbone）仍采用**直接编码方式**（单次前向传播生成 Embedding），大量依赖对比学习对齐正样本。这导致模型倾向捕捉统计共现模式，偏向浅层词汇/语义匹配。对于 query 与目标 item 存在显著词汇差异的困难查询，性能严重退化。

## 核心方法：LREM 两阶段训练

**核心思想**：对困难 query，先进行推理以深度理解，再生成推理增强的 query embedding。

### Stage 1: SFT + InfoNCE 联合训练
在精心构造的 Query-CoT-Item 三元组上：
- **SFT 损失**：教模型生成高质量推理链（CoT）
- **InfoNCE 损失**：对比学习对齐推理增强后的 query embedding 与 item embedding

$$
\mathcal{L}_1 = \mathcal{L}_{\text{SFT}}(\text{CoT} | q) + \lambda \cdot \mathcal{L}_{\text{InfoNCE}}(e_{q+\text{CoT}}, e_{\text{item}})
$$

### Stage 2: RL 精化推理轨迹
用强化学习进一步优化推理链质量，使其生成的 embedding 更利于检索。

$$
\pi^* = \arg\max_\pi \mathbb{E}[\text{RetrievalReward}(q, \text{CoT}_\pi, \text{item})]
$$

## 关键创新

- **推理嵌入范式**：从"直接编码"到"先推理后编码"，密集检索的范式转变
- **两阶段训练**：SFT 建立初步推理+嵌入能力，RL 精化推理轨迹
- **电商实战验证**：不是纯学术 benchmark，而是在真实电商搜索系统中部署并验证

## 实验亮点

- 大量离线和在线实验验证有效性
- 2025年8月部署到中国最大电商平台（推测为淘宝/天猫）
- 对 query-item 词汇差异大的困难查询提升显著

## 与相关工作的关系

| 方法 | 推理时机 | 额外延迟 | 训练方式 |
|------|---------|---------|---------|
| O1-Embedder | Query 编码前推理 | 50-100ms | SFT |
| ReasonEmbed | 训练时数据增强 | 0ms | 自适应加权 |
| **LREM** | Query 编码前推理 | 有（CoT生成） | SFT + RL |
| CRE-T1 | 推理链增强 | 有 | GRPO |

LREM 的独特贡献在于**两阶段（SFT→RL）训练范式**和**电商工业验证**。

[[推理增强检索技术综述]] | [[embedding_everywhere]] | [[sequence_modeling_evolution]]
