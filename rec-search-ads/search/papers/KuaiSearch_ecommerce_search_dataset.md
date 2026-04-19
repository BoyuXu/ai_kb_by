# KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance
> 来源：arXiv:2602.11518 | 领域：search | 学习日期：20260419

## 数据集概况
- **规模**：33万用户、1800万商品、250万真实搜索query
- **来源**：快手电商平台真实用户搜索交互
- **覆盖**：召回（Recall）、排序（Ranking）、相关性（Relevance）三个核心搜索阶段
- **开源**：https://github.com/benchen4395/KuaiSearch

## 挑战
- 高度模糊的query（如"好看的衣服"）
- 商品文本噪声大、语义序弱
- 用户偏好多样性
- 冷启动用户和长尾商品

## 面试考点
- Q: 电商搜索 vs 网页搜索的核心差异？
  - A: ①商品文本短且噪声大（标题堆砌关键词）；②query intent更商业化；③评价指标多元（相关性+转化+多样性）；④实时性更强（库存/价格变化）
