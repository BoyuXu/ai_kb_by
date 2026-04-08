# ReaRec: Enhancing Sequential Recommendation with Latent Reasoning

> 来源：arXiv 2025 | 领域：llm-infra/rec-sys | 学习日期：20260408

## 问题定义

序列推荐模型直接从用户行为序列预测下一个交互物品，缺乏显式推理过程：
- 用户兴趣可能随时间演变
- 短期行为与长期偏好需要不同的推理深度

**核心问题**：如何在推荐推理时引入 test-time scaling，通过推理增强用户表征？

## 核心方法与创新点

**ReaRec 推理时计算框架**：

1. **潜在空间自回归推理**：
   - 不在文本空间推理，而在潜在表示空间进行多步推理
   - 用户表示 $h_u$ 通过 K 步推理迭代增强：$h_u^{(k)} = f(h_u^{(k-1)})$

2. **Ensemble Reasoning Learning (ERL)**：
   - 多个推理头并行，集成结果
   - 降低单一推理路径的方差

3. **Progressive Reasoning Learning (PRL)**：
   - 推理步数渐进增加
   - 训练时从简单到复杂，提升稳定性

## 关键结果

- 在多个序列推荐骨干网络上提升 30%-50%
- 推理步数增加时性能持续提升（test-time scaling）

## 面试考点

- 推荐系统中的 test-time compute scaling
- 潜在空间推理 vs 文本空间推理的 trade-off
