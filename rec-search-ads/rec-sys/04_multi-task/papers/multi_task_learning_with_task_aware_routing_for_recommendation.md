# Multi-Task Learning with Task-Aware Routing for Recommendation
> 来源：arxiv/2312.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
推荐系统多任务学习（MTL）中的核心挑战：
- **跷跷板效应**：优化一个任务往往损害另一个任务（CTR↑ vs CVR↑ 冲突）
- **负迁移**：任务差异大时，共享参数导致相互干扰
- **静态共享**：Hard Parameter Sharing 无法根据输入动态调整
- **任务相关性建模**：不同 item/用户对各任务的相关性不同

## 核心方法与创新点
**Task-Aware Routing MTL**：动态路由机制，根据输入自适应选择专家网络。

**Mixture of Experts (MoE) + Task-Aware Gating：**
```python
# 共享专家池
experts = [Expert_1, Expert_2, ..., Expert_K]  # K个专家网络

# 任务感知门控
gate_task_t(x) = Softmax(W_t · [x; task_embed_t])  # 输入+任务embedding
expert_output = Σ_k gate_task_t(x)[k] · Expert_k(x)

# 任务专属 Tower
output_t = Tower_t(expert_output)
```

**动态路由策略：**
- **Soft Routing**：所有专家加权求和（可微，默认）
- **Hard Routing（Top-K）**：只激活 Top-K 专家（稀疏，高效）
- **Hierarchical Routing**：先粗后细，两级门控

**任务相关性正则：**
```
L_corr = -Σ_{t1,t2} corr(t1,t2) · cosine(gate_t1, gate_t2)
# 相关任务应有相似的路由模式
```

**样本不均衡处理：**
- CVR 样本远少于 CTR → 类别权重 or 过采样
- 使用 ESMM 框架：P(CVR) = P(CTR→CVR) × P(CTR)

## 实验结论
- 美团外卖推荐系统 A/B：
  - CTR +1.4%，CVR +2.1%，订单量 +1.8%
- 离线评估（AUC）：CTR +0.003，CVR +0.005（vs 静态 MMoE）
- 消融：Task-Aware Routing > 随机路由 > 固定路由

## 工程落地要点
1. **专家数量选择**：通常 4-8 个专家，过多增加计算量，过少无法区分任务
2. **专家负载均衡**：Auxiliary Loss 防止所有样本都路由到少数专家
3. **任务 Embedding**：可训练的 task embedding（8-16 维），与输入特征拼接
4. **在线推理优化**：Top-K Hard Routing（K=2）大幅减少计算量
5. **任务损失权重**：根据业务优先级设置（GMV > CTR > 停留时长）

## 常见考点
**Q1: MMoE 相比 Shared Bottom 多任务模型的优势？**
A: Shared Bottom 强制所有任务共享同一表示，对差异大的任务（CTR/CVR）会导致负迁移。MMoE 通过 K 个专家 + 任务专属门控，允许不同任务使用不同的特征组合，减少冲突。

**Q2: 跷跷板效应的根本原因？**
A: 多任务学习中，任务之间的梯度方向可能相反。当任务 A 的梯度更新某参数时，这个更新可能对任务 B 有害。Task-Aware Routing 让不同任务使用不同的参数子集，从而物理隔离冲突。

**Q3: ESMM（Entire Space Multi-task Model）解决了什么问题？**
A: CVR 模型的样本选择偏差：传统 CVR 只在点击样本上训练，但线上需要在全部曝光上预测。ESMM 通过 CTR × CVR = CTCVR 的乘积关系，在全空间训练 CVR，消除样本偏差。

**Q4: 专家负载均衡为什么重要？如何实现？**
A: 没有负载均衡，门控网络会收敛到只使用少数专家（因为激活专家的梯度更大）。解决：Auxiliary Load Balance Loss = CV(expert_usage)²，惩罚专家使用不均匀。

**Q5: 任务相关性如何量化并用于 MTL 设计？**
A: ①Pearson 相关：历史数据上 CTR 和 CVR 的皮尔逊相关系数 ②梯度相似度：两任务梯度向量的余弦相似度 ③任务聚类：相关任务共享更多层/专家。任务相关性低 → 需要更多独立参数。

## 模型架构详解

### 共享层
- **底座网络**：共享 Embedding 层 + 底部 DNN 提取通用特征表示
- **专家网络**：MoE（Mixture of Experts）或 PLE（Progressive Layered Extraction）分离任务特有/共享知识
- **门控机制**：Softmax 门控决定各专家对不同任务的贡献权重

### 任务特有层
- **Tower 网络**：每个任务独立的 DNN Tower + 输出头
- **任务关系建模**：显式建模任务间的依赖关系（如 CTR→CVR 的因果链）
- **损失函数**：加权多任务损失 = Σ wᵢ × Lᵢ，权重可固定或动态调整

### 训练策略
- **动态权重**：GradNorm / Uncertainty Weighting 自适应调整任务权重
- **梯度冲突处理**：PCGrad / CAGrad 处理任务间的梯度方向冲突
- **渐进训练**：先训练简单任务（CTR），再加入复杂任务（CVR/LTV）

## 与相关工作对比

| 维度 | 本文方法 | 独立模型 | Shared-Bottom | MMoE |
|------|---------|---------|-------------|------|
| 参数效率 | 高 | 低 | 高 | 中高 |
| 负迁移 | 可控 | 无 | 严重 | 可控 |
| 可扩展性 | 好 | 好 | 差 | 好 |
| 训练复杂度 | 中 | 低 | 低 | 中 |

## 面试深度追问

- **Q: 多任务学习中的负迁移（Negative Transfer）是什么？**
  A: 不同任务的梯度方向冲突，联合训练反而比单独训练效果差。例如 CTR 任务偏好点击频繁的热门物品，而多样性任务偏好长尾物品。

- **Q: MMoE 和 PLE 的区别？**
  A: MMoE 所有专家对所有任务共享，用门控选择；PLE 区分共享专家和任务特有专家，并通过渐进层次避免浅层信息污染。PLE 在任务差异大时优于 MMoE。

- **Q: 如何判断两个任务是否适合联合训练？**
  A: 1) 任务相关性分析（目标正相关/互补）；2) 数据重叠度（共享用户/物品越多越好）；3) 梯度方向分析（余弦相似度 > 0 说明不冲突）；4) A/B Test 验证。

- **Q: LLM 增强推荐的典型方案？**
  A: 1) 特征增强：LLM 提取文本/知识特征作为推荐模型输入；2) 知识蒸馏：LLM 作为 Teacher 指导推荐模型学习；3) 端到端：LLM 直接输出推荐结果。
