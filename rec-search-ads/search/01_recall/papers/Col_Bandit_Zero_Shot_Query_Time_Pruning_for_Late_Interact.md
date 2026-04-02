# Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval

> 来源：arXiv | 日期：20260317

## 问题定义

**ColBERT / Late-Interaction** 模型通过 Token-level 向量的 MaxSim 操作实现精细交互，效果接近 Cross-Encoder 但可预算 Document Token 向量（比全量 Cross-Encoder 快 10~100x）。然而 ColBERT 的存储和计算开销仍然显著：每篇文档存储约 128~512 个 token 向量（vs 双塔的 1 个向量），在线计算 MaxSim 复杂度 $O(|Q| \times |D|)$。

**核心问题**：能否在查询时动态剪枝不重要的 Document Token 向量，在不损失精度的情况下降低计算量？

## 核心方法与创新点

1. **Bandit 框架进行 Token 剪枝**
   - 将 Document Token 选择建模为 Multi-Armed Bandit 问题
   - 每个 Token 位置是一个"臂"，奖励是该 token 对最终 MaxSim 分数的贡献
   - 使用 UCB 策略在有限计算预算内选择最有价值的 token 子集

2. **零样本（Zero-Shot）设计**
   - 不需要为每个数据集/任务收集标注数据重新训练
   - 剪枝决策基于 Query Token 与 Document Token 的初始相似度估计
   - 早期 token 得分高的位置被优先全量计算，低得分位置被剪枝

3. **自适应预算分配**
   - 不同查询复杂度分配不同的计算预算
   - 简单查询（关键词匹配）使用少量 token 即可；复杂查询使用更多 token

4. **与 PLAID / EMVB 的对比**
   - PLAID：需要额外训练，针对特定硬件优化
   - Col-Bandit：通用、无训练、查询时动态决策

## 实验结论

- MSMARCO Dev：在保持 MRR@10 损失 <0.5% 的前提下，计算量减少约 40%
- BEIR：跨域零样本泛化良好，各数据集平均 nDCG@10 下降 <1%
- 延迟：与完整 ColBERT 相比，P99 延迟降低约 35%

## 工程落地要点

1. **存储优化**：即使剪枝减少了计算，Document Token 向量仍需全量存储；可结合 PQ 量化降低存储
2. **预算参数调优**：Bandit 的预算 $B$（最多采样 token 数）是关键超参，建议在验证集上调
3. **批量推理**：批量处理多个 Document 时，Bandit 采样策略可并行化
4. **与检索流水线集成**：Col-Bandit 适合作为 ANN 初检 + ColBERT 重排管道中的加速组件

## 常见考点

- **Q: ColBERT 的 Late-Interaction 是什么？**
  A: Query 和 Document 分别独立编码为 token-level 向量（$Q = [q_1,...,q_m]$，$D = [d_1,...,d_n]$）；相关性分数为 $\text{score} = \sum_{q_i} \max_{d_j} q_i \cdot d_j$（MaxSim）。Document 向量可离线预算，只有 MaxSim 计算在线，比 Cross-Encoder 快得多。

- **Q: 为什么 Late-Interaction 需要 Token 剪枝？**
  A: ColBERT 每篇文档存储约 100~500 个 token 向量，存储量约为双塔的 100~500x；且 MaxSim 计算量 $O(|Q| \times |D|)$，在重排阶段处理 1000 个候选时计算量显著。

- **Q: UCB 策略如何用于 Token 选择？**
  A: 将每个 document token 位置视为一个臂，UCB 值 = 估计贡献均值 + 不确定性上界。优先计算 UCB 值最高的 token，利用初步相似度估计（快速 inner product）确定哪些 token 更可能产生高 MaxSim 得分。
