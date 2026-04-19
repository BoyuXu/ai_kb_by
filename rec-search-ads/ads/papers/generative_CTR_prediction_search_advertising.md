# Generative Click-through Rate Prediction with Applications to Search Advertising
> 来源：arXiv:2507.11246 | 领域：ads | 学习日期：20260419

## 核心方法
1. **两阶段训练**：生成预训练（next-item prediction）→ 判别微调（CTR prediction）
2. **生成式预训练**：学习用户行为序列的生成式表征，捕捉全局模式
3. **判别微调**：在生成表征基础上进行CTR分类任务微调
4. **应用于搜索广告**：结合query理解和广告匹配

## 面试考点
- Q: 生成式预训练如何提升CTR预估？
  - A: 生成式预训练学习更丰富的用户行为表征（因果关系、时序模式），为下游CTR预估提供更好的初始化
