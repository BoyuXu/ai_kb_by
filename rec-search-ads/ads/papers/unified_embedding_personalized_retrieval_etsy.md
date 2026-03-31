# Unified Embedding Based Personalized Retrieval in Etsy Search

> 来源：https://arxiv.org/abs/2306.04833 | 领域：计算广告 | 学习日期：20260331

## 问题定义

Etsy搜索中传统文本匹配无法捕获用户个性化偏好，而独立个性化模型难以与搜索相关性统一优化。

## 核心方法与创新点

1. **统一Embedding模型**：query、user、item统一编码到同一向量空间
2. **三元组训练**：(user, query, positive_item) vs (user, query, negative_item)
3. **个性化注意力**：用户历史行为通过注意力融入query表征

$$e_{query}^{personal} = \text{Attention}(e_{query}, [e_{h_1}, ..., e_{h_n}])$$

4. **混合检索**：文本匹配分+向量相似度分加权融合

## 实验结论

Etsy线上AB实验中搜索转化率提升3.1%，GMV提升2.4%。

## 工程落地要点

- 用户Embedding需实时/近实时更新
- ANN索引需支持filter（品类筛选）
- 需平衡个性化与搜索相关性权重
- 适合中小型电商搜索个性化升级

## 面试考点

1. **为什么统一Embedding？** 避免多个独立模型的维护成本和不一致
2. **query意图与个性化冲突？** 注意力权重自动学习平衡
3. **Etsy的特殊挑战？** 长尾手工商品、卖家多样性
