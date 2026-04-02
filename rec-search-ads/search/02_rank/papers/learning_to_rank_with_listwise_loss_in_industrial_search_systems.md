# Learning to Rank with Listwise Loss in Industrial Search Systems
> 来源：arxiv/2311.xxxxx | 领域：search | 学习日期：20260326

## 问题定义
工业搜索排序（Learning to Rank, LTR）的挑战：
- Pointwise 方法：独立预测每个文档相关性，忽略文档间相对关系
- Pairwise 方法（如 RankNet）：考虑文档对，但不考虑整体列表质量
- Listwise 方法：直接优化列表级排序指标（NDCG/MAP），最符合业务目标
- 工业规模：搜索日志噪声大，用户点击带有位置偏差（Position Bias）

## 核心方法与创新点
**Listwise Loss for Industrial LTR**

**ListNet/SoftMax 损失：**
```python
# ListNet：将排序问题建模为概率分布匹配
P_model(π_1 | docs) = exp(score_1) / Σ_j exp(score_j)
P_label(π_1 | docs) = exp(rel_1 · γ) / Σ_j exp(rel_j · γ)

L_listnet = KL(P_label || P_model)
         = -Σ_i P_label_i · log P_model_i
```

**LambdaLoss（NDCG 导向）：**
```python
# LambdaRank：梯度加权，ΔZ 为交换 i,j 后的 NDCG 变化
λ_ij = σ(s_j - s_i) × |ΔNDCG_{ij}|
# ΔNDCG 大的 pair 获得更大梯度权重
```

**位置偏差校正（Position Debiasing）：**
```python
# IPS（Inverse Propensity Score）校正
propensity = P(click | position, rel=1)  # 位置点击倾向
weight_i = 1 / propensity(position_i)

L_debiased = Σ_i weight_i · L_i  # 加权损失
```

**特征工程：**
- 查询-文档相关性特征（BM25、语义相似度）
- 文档质量特征（PageRank、新鲜度、质量分）
- 个性化特征（用户历史行为与当前文档相关性）

## 实验结论
- 工业搜索系统（某搜索引擎）：
  - NDCG@5：Listwise +0.018（vs Pointwise）
  - MRR：+0.012（vs Pairwise RankNet）
- 位置偏差校正后：Head 查询 NDCG +0.008，Tail 查询 +0.021
- 个性化排序：用户满意度指标 +2.3%

## 工程落地要点
1. **训练数据构建**：点击日志 → Position Debiasing → Relevance Label 估计
2. **LambdaLoss 实现**：需要计算所有 pair 的 ΔNDCG（O(n²)），工程上截断为 Top-K
3. **样本权重**：高质量的（query, doc）pair 给更高权重（人工标注 > 点击 > 展示）
4. **在线评估**：NDCG 的 A/B 实验需要注意 Position Bias 影响，用无偏点击率评估
5. **特征更新频率**：BM25 特征离线，用户行为特征近实时（1h 延迟）

## 常见考点
**Q1: Pointwise/Pairwise/Listwise 三类 LTR 方法的本质区别？**
A: Pointwise：独立评分，忽略列表结构；Pairwise：考虑文档对的相对顺序，优化对的误序率；Listwise：直接优化整个列表的排序质量（NDCG/MAP），最符合搜索用户的实际体验。工业中通常 Listwise > Pairwise > Pointwise。

**Q2: NDCG 为什么不能直接用梯度优化？**
A: NDCG 是不可微的（argmax/排序操作）。LambdaRank 的解决方案：构造梯度使其等效于优化 NDCG 的方向（即交换两个文档后 NDCG 下降时，给下降的那个文档更大的惩罚梯度）。

**Q3: 搜索系统中位置偏差问题如何量化和校正？**
A: 量化：Randomization Experiments（随机打乱展示位置，测量不同位置的点击率）。校正：IPS 权重 = 1/propensity_score；Dual Learning：同时学习倾向模型和排序模型；Result Randomization + EM 估计。

**Q4: 搜索排序模型如何处理人工标注和点击数据的质量差异？**
A: 混合训练：人工标注（高质量，少量）作为主监督信号；点击日志（低质量，海量）作为补充。权重分配：人工标注权重 10x-100x 点击数据；或者两阶段：用点击预训练，用人工标注微调。

**Q5: 工业搜索排序中的个性化如何与全局排序结合？**
A: 两阶段：①全局排序：基于查询-文档相关性的通用排序 ②个性化重排：基于用户历史行为调整最终排序（用户曾购买的品牌适当加权）。也可端到端：在特征中加入用户特征，让排序模型学习个性化偏好。
