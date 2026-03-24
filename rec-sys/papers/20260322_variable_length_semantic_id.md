# Variable-Length Semantic IDs for Recommender Systems

> 来源：arxiv (https://arxiv.org/abs/2602.16375) | 日期：20260322 | 领域：推荐系统

## 问题定义

现有 Semantic ID 方法（如 RQ-VAE）对所有 item 生成固定长度的 ID 序列，但不同 item 的信息量差异巨大：热门 item 有丰富的交互和内容信息，用更短的 ID 即可区分；冷启动 item 则需更长的 ID 来精确表示。固定长度造成表达浪费或信息不足。

## 核心方法与创新点

- **自适应 ID 长度**：基于 item 的内容复杂度和交互丰富度，动态分配 1~L 层量化码字
- **长度预测器**：轻量级分类器预测最优 ID 长度，输入：item 内容嵌入 + 交互统计（曝光量、点击率等）
- **变长 RQ-VAE**：
  - 训练时：用重建损失 + 长度预测损失联合训练
  - 推理时：根据预测长度截断 codebook 查询
- **生成式推荐适配**：序列生成模型使用特殊 [EOS] token 标记 ID 结束，自然支持变长解码
- **压缩率**：平均 ID 长度比固定长度方案减少 30%，降低解码延迟

## 实验结论

- 相比固定长度 Semantic ID（TIGER），Recall@10 提升 5.8%，Hit Rate@50 提升 7.2%
- 冷启动 item（<10 次交互）改善最明显：+15% Recall，因为长 ID 提供了更丰富的内容区分
- 热门 item（>1000 次交互）使用短 ID，解码速度提升 1.4×，且精度不损失
- 索引大小减少 28%，ANN 检索内存占用显著降低

## 工程落地要点

- **长度分布规划**：建议预先统计 item 长度分布，设置最大 ID 长度上限（通常 4-6），避免超长 ID 影响批量训练效率
- **Padding 策略**：训练时对短 ID 进行右 padding，但 attention mask 屏蔽 padding 位置
- **增量更新**：新 item 通过长度预测器 + encoder 直接得到 ID，无需全量重训
- **与 beam search 配合**：变长 ID 使 beam search 的剪枝更自然（命中 [EOS] 即停止该 beam）

## 面试考点

1. **Q：为什么热门 item 用短 ID 而冷启动 item 用长 ID？**
   A：热门 item 有大量协同过滤信号，模型已能通过少量 token 精确定位；冷启动 item 协同信号弱，需要更多内容特征 token 来区分

2. **Q：变长 ID 对模型训练有什么挑战？**
   A：批量训练需要 padding/masking；序列生成时 EOS token 的学习需要特别设计损失函数；长度分布不均匀可能导致梯度不稳定

3. **Q：如何评估 Semantic ID 质量？**
   A：重建精度（能否从 ID 还原 item 内容表示）；聚类质量（同类 item 共享前缀比例）；下游推荐任务的 Recall/NDCG
