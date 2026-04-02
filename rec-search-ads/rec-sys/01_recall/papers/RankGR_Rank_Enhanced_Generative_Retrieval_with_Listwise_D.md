# RankGR: Rank-Enhanced Generative Retrieval with Listwise DPO in Recommendation

> 来源：arxiv | 日期：20260316 | 领域：rec-sys

## 问题定义

生成式检索（Generative Retrieval, GR）直接用自回归模型预测 item ID，避免了传统 ANN 检索的索引瓶颈。但现有 GR 方法存在 **排序感知不足** 的问题：逐 token 生成时模型不能感知候选集的全局相对顺序，导致召回质量受限。RankGR 尝试将排序信息注入生成式检索训练过程。

## 核心方法与创新点

1. **Listwise DPO（Direct Preference Optimization）**：将多候选排序列表转化为偏好对，用 DPO 目标约束模型生成顺序与真实排序一致。传统 DPO 是 pairwise（好/坏各一条序列），Listwise DPO 同时比较多个候选，信号更密集。

2. **Rank-aware Beam Search**：推理时将候选 beam 的排名置信度与生成概率联合打分，避免"高频 token 陷阱"。

3. **SID（Semantic Item Descriptor）编码**：将 item 语义特征编码进 ID 结构，使生成过程同时利用语义与 ID 信息。

架构：Encoder-Decoder（如 T5-base）+ Listwise DPO Fine-tuning + Rank-aware Decoding。

## 实验结论

- 在 Amazon Review（Beauty/Sports）和 MovieLens 数据集上，相比 TIGER、GENRE 等基线，NDCG@10 提升 **3-8%**。
- Listwise DPO 相比 Pairwise DPO 额外提升约 **1-2%** NDCG@10。
- 数据稀疏场景（cold-start items）收益更显著，提升达 **12%**。

## 工程落地要点

- 生成式检索适合 **item 空间中等规模**（<10M）的场景，超大规模需 hierarchical SID。
- DPO 训练需要高质量偏好数据，可用 BM25/CF 得分构造 listwise 排序作为 offline 标签。
- 推理延迟高于 FAISS ANN，需结合 speculative decoding 或量化加速。
- 与 HNSW 混合：GR 负责语义召回，ANN 负责高频热门 item，融合后 QPS 可接受。

## 常见考点

- Q: 生成式检索和传统双塔检索的核心区别是什么？
  A: 双塔检索分离 query/item 编码后做向量点积，GR 直接自回归生成 item ID 序列。GR 天然建模 multi-hop 关系，但推理慢；双塔简单高效但建模能力有限。

- Q: DPO 是什么？与 RLHF 的关系？
  A: DPO（Direct Preference Optimization）直接用 Bradley-Terry 偏好模型将 RLHF 目标转化为分类损失，绕开 RL 训练不稳定的问题。公式：L_DPO = -log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))。

- Q: Listwise DPO 和 Pairwise DPO 有何区别？
  A: Pairwise 每次只比较一对 (win, lose)，Listwise 同时比较整个排序列表，利用 Plackett-Luce 模型定义列表级偏好概率，信号更丰富，训练效率更高。
