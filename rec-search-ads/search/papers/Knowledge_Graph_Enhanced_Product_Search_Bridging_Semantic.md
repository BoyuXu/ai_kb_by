# Knowledge Graph Enhanced Product Search: Bridging Semantic Gaps in E-Commerce
> 来源：https://arxiv.org/search/?query=knowledge+graph+product+search+semantic&searchtype=all | 领域：search | 日期：20260323

## 问题定义
电商搜索中，用户查询和商品描述之间存在语义鸿沟（vocabulary mismatch）。知识图谱（KG）包含实体关系（品牌→产品系列、材质→适用场景），可用于桥接这种语义gap。

## 核心方法与创新点
- KG增强查询理解：从KG中提取查询相关实体的属性和关系
- 图神经网络（GNN）：在KG上传播实体embedding，丰富商品表示
- 知识驱动扩展：利用KG的关系路径推断相关商品类别
- 多跳推理：处理复杂查询（"适合夏天海滩的防晒服"需要多跳推理）

## 实验结论
在JD/Taobao电商搜索数据集，KG增强相比纯文本检索，Recall@100提升约10%；多跳查询处理准确率提升约20%；特别是垂类搜索（电子/美妆）提升更显著。

## 工程落地要点
- KG构建和维护成本高，需要持续的知识抽取和审核
- GNN推理在大规模KG上效率是关键挑战，需要mini-batch采样
- 电商KG需要特别处理品牌词保护和竞品屏蔽

## 面试考点
1. **Q: 电商知识图谱包含哪些典型实体和关系？** A: 实体：品牌、类目、属性值；关系：belongs-to、compatible-with、substitute-for
2. **Q: GNN在知识图谱上如何传播信息？** A: 每个节点聚合邻居节点的表示，经过多层传播获取k跳邻域信息
3. **Q: KG如何帮助处理搜索中的词汇鸿沟？** A: "运动鞋"→KG→Nike/Adidas/New Balance，扩展精确品牌召回
4. **Q: 电商KG的自动构建方法？** A: 商品详情页结构化抽取、用户评论挖掘、属性值对齐、跨平台实体链接
5. **Q: KG增强推荐与KG增强搜索的差异？** A: 搜索：KG辅助查询理解和扩展；推荐：KG辅助用户-物品关系推断
