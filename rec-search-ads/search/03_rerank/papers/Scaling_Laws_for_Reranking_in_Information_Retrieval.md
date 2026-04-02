# Scaling Laws for Reranking in Information Retrieval

> 来源：arXiv | 日期：20260317

## 问题定义

Scaling Laws（规模律）在 LLM 领域已被深入研究（Chinchilla 等），但在**信息检索的重排（Reranking）任务**中尚不明确。重排模型（Cross-Encoder）的性能如何随模型大小、训练数据量、计算量扩展？是否存在类似 LLM 的幂律关系？这对选择模型规模、训练策略有重要指导意义。

## 核心方法与创新点

1. **实验设计**
   - 在多种规模的 Cross-Encoder 模型（BERT-tiny ~ T5-11B）上系统评估
   - 固定其他变量，分别分析：模型参数量 N、训练样本量 D、计算量 C（FLOPs）
   - 数据集：MSMARCO Passage、BEIR benchmark

2. **主要发现**
   - 重排性能（nDCG@10）与模型参数量呈**幂律关系**：$\text{Performance} \propto N^{\alpha}$
   - 最优计算分配（Compute-optimal）：给定 FLOPs 预算，建议同时扩大模型和数据，比例约 1:1
   - 数据量扩展的边际收益递减速度慢于模型规模扩展

3. **Reranking 特有发现**
   - Query-Document 交叉注意力在模型扩展时收益显著（vs 仅 Query 编码）
   - 小模型 + 大量蒸馏数据 ≈ 大模型效果（知识蒸馏可弥补规模差距）
   - 检索质量（First-stage Recall）对重排的最终性能影响大于模型规模

4. **与 LLM Scaling 的对比**
   - Reranking 的规模律指数 $\alpha$ 略小于语言建模任务（边际收益略快递减）
   - Reasoning 能力（对 long-context 相关性判断）在大模型中涌现

## 实验结论

- MSMARCO: 模型规模从 110M→11B 参数，nDCG@10 提升约 3.5%（绝对值）
- Compute-optimal frontier 上，11B 模型比相同 FLOPs 下的 110M 模型效果好约 2%
- BEIR zero-shot：大模型泛化能力更强，跨域提升约 4%

## 工程落地要点

1. **模型选型**：根据延迟预算选择最大可承受规模；延迟 <50ms → BERT-base（110M）；延迟 <500ms → T5-3B
2. **蒸馏策略**：如预算有限，优先用大模型蒸馏数据训练小模型，比直接训练小模型效果好
3. **检索质量优先**：提升第一阶段召回 Recall@1000 比升级重排模型 ROI 更高
4. **批量推理优化**：Cross-Encoder 每个 Query 需对 K 个 Document 串行推理，批量推理可提升 GPU 利用率

## 常见考点

- **Q: 什么是 Cross-Encoder 和 Bi-Encoder，各自适合哪个阶段？**
  A: Bi-Encoder（双塔）将 Query 和 Document 分别编码，相似度用向量内积，适合召回（可预算 doc embedding）；Cross-Encoder 将 Query+Document 拼接输入，交叉注意力建模，精度更高但无法预算，适合重排。

- **Q: 为什么重排阶段的 Scaling Law 指数比 LLM 小？**
  A: 重排任务的信号相对明确（相关性判断），不需要像语言模型一样学习世界知识；而且任务本身的难度上限较低（相关性不是无限复杂的任务），大模型的额外能力（推理、世界知识）在简单相关性判断上边际收益有限。

- **Q: 检索质量对重排的影响为什么大于模型规模？**
  A: 重排只能对召回的候选集重新排序，如果真正相关的文档没有被召回，再好的重排模型也无法找到它。Recall@K 是重排的上界。
