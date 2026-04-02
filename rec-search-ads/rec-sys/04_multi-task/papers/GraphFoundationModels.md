# 图基础模型用于推荐：综合综述

> 来源：Graph Foundation Models for Recommendation: A Comprehensive Survey | 日期：20260318 | 领域：rec-sys

## 问题定义

推荐系统中的图神经网络（GNN）已经非常成熟（LightGCN、PinSage 等），但每个任务都需要从头训练，无法复用。与此同时，LLM 领域"预训练大模型 + 少量微调"范式取得巨大成功。**图基础模型（Graph Foundation Model, GFM）** 试图将"预训练+泛化"思想迁移到图上：能否预训练一个通用图编码器，然后低成本适配到新的推荐任务/平台？

## 核心方法与创新点

综述梳理了三类 GFM 用于推荐的路线：

1. **图预训练 + 下游微调**：
   - 在大规模用户-物品交互图上预训练 GNN（对比学习、掩码节点预测等自监督目标）。
   - 下游任务只需微调顶层 MLP 或 adapter，类似 BERT fine-tuning。
   - 代表：GraphMAE、SimGRACE。

2. **LLM 作为图节点特征编码器**：
   - 用 LLM 编码物品文本描述（标题、类目、属性），生成丰富语义 embedding 替代 ID embedding。
   - GNN 在 LLM embedding 之上传播协同信号。
   - 代表：LLMRec、RLMRec。

3. **图结构 + LLM 统一建模**：
   - 将图游走序列序列化为文本喂给 LLM（GraphGPT、InstructGLM），统一处理图结构和语义。
   - 灵活性高，但计算成本最大。

**关键挑战**：图异构性（不同平台图结构差异大）、节点对齐（跨平台实体没有共同 ID）、可扩展性（百亿节点图上的预训练）。

## 实验结论

- LLM 文本 embedding 替代 ID embedding：在冷启动场景（新物品）Recall@20 提升 20-40%，暖场景提升约 5%。
- 图预训练 + 微调 vs 从头训练：数据量少时（< 10% 训练数据）提升显著（+10-15% NDCG），数据充足时优势缩小。
- 统一建模方案效果最好，但推理延迟比 GNN 高 10-100x。

## 工程落地要点

- **冷启动优先用 LLM embedding**：上线即有效，不依赖历史交互，适合长尾物品和新品首发。
- **ID + 语义双 embedding**：生产中通常拼接 ID embedding（协同信号）和文本 embedding（语义信号），两者互补。
- **离线预计算**：LLM 编码物品 embedding 可离线完成并缓存，在线检索只做向量查找。
- **图预训练的迁移**：跨平台迁移时需注意特征空间对齐，可用 adapter 层做域适配。

## 常见考点

**Q: GNN 在推荐中的核心作用是什么？**
- 传播高阶协同信号：用户的用户的偏好（二阶邻居）可以被聚合；解决数据稀疏问题（通过邻居补充信息）。

**Q: LightGCN 为什么去掉了特征变换矩阵和激活函数？**
- 推荐场景中 ID embedding 的线性聚合已经足够，非线性变换容易过拟合且增加训练难度；消融实验表明简化后效果更好。

**Q: Graph Foundation Model 和 LLM for Rec 的区别？**
- GFM 重点在图结构的预训练泛化；LLM for Rec 重点在语义理解和生成能力；两者可结合（用 LLM 编码节点特征，GNN 传播图结构）。
