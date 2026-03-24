# 图神经网络在推荐中的应用：从 PinSage 到 GNN4Rec

> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：PinSage, EGES, LightGCN, NGCF, GraphSAGE 系列

---

## 🎯 核心洞察（4条）

1. **图结构自然适配推荐场景**：用户-物品交互是天然的二部图，物品间的共现/属性关系形成知识图谱
2. **GNN 的核心优势是高阶关联**：2 层 GNN 就能捕捉"A 和 B 都被 C 买过"的二阶关系，比协同过滤更高效
3. **LightGCN 的"越简单越好"哲学**：去掉 feature transformation 和非线性激活，只保留 neighborhood aggregation，效果反而更好
4. **工业落地面临可扩展性挑战**：Pinterest 的 PinSage 用 random walk + mini-batch 训练解决了十亿节点图的训练问题

---

## 🎓 面试考点（4条）

### Q1: GNN 在推荐中的典型应用？
**30秒答案**：①召回：PinSage/EGES 在物品图上学 embedding → ANN 检索；②排序：用户-物品交互图的 node embedding 作为额外特征；③知识图谱增强：物品属性图谱（品牌→类目→材质）的 GNN embedding 增强冷启动。

### Q2: LightGCN vs NGCF 的区别？
**30秒答案**：NGCF 在每层做 feature transformation + 非线性激活 + neighborhood aggregation；LightGCN 发现 transformation 和非线性对推荐无用甚至有害，只保留加权邻居聚合 + 多层 embedding 求和。

### Q3: PinSage 怎么在十亿级图上训练？
**30秒答案**：①Random Walk 采样邻居（不遍历全图）；②Mini-batch 训练（每次只取一个子图）；③Importance Sampling（按 PageRank 权重采样重要邻居）；④MapReduce 分布式计算。

### Q4: 图推荐 vs 向量推荐的选择？
**30秒答案**：图推荐——有明确图结构（社交网络、知识图谱）、需要高阶关系的场景（关联推荐"买了A的人还买了B"）。向量推荐——通用场景、工程简单、延迟要求高。两者可以作为多路召回互补。

---

## 🌐 知识体系连接

- **上游依赖**：图神经网络（GNN）、图采样算法、分布式图计算
- **下游应用**：多路召回、知识图谱推荐、社交推荐
- **相关 synthesis**：std_rec_recall_evolution.md, std_rec_embedding_learning.md
