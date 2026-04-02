# EST: Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling

> 来源：arXiv 2602.07890 | 领域：ads | 学习日期：20260402

## 问题定义

CTR 预估模型的参数量持续增长，但缺乏系统性的 Scaling Laws 指导资源分配。盲目增大模型可能导致边际收益递减。EST 提出统一建模框架来建立 CTR 预估的高效 Scaling Laws。

## 核心方法与创新点

1. **统一建模框架**：将不同 CTR 模型架构（DNN、DCN、Transformer）统一到一个可参数化的框架中
2. **计算最优 Scaling**：给定计算预算，找到模型参数量、训练数据量和训练步数的最优分配
3. **架构搜索集成**：在 Scaling 过程中同步优化模型架构，避免固定架构的次优 Scaling
4. **稀疏特征 Scaling**：特别研究 Embedding 表的 Scaling 规律，发现与稠密参数不同的幂律指数

## 实验结论

- 相同计算预算下，EST 指导的 Scaling 比均匀 Scaling AUC 高 0.8%
- 发现 CTR 模型的最优计算分配比 LLM（Chinchilla Law）更倾向增大数据量
- 稀疏参数的 Scaling 指数约为稠密参数的 0.6 倍

## 工程落地要点

- **预算规划**：根据 Scaling Laws 可以预估不同投入的预期收益
- **架构选择**：统一框架支持自动选择最优架构
- **训练效率**：计算最优分配可节约 30-50% 训练成本

## 常见考点

1. **Q：CTR 模型 Scaling Laws 和 LLM 的主要差异？**
   A：CTR 模型有大量稀疏参数（Embedding），其 Scaling 指数不同于稠密参数。且 CTR 最优分配更侧重数据量。
2. **Q：为什么 Embedding 的 Scaling 指数更小？**
   A：Embedding 的容量瓶颈主要在维度而非规模，增大表大小的边际收益递减更快。
