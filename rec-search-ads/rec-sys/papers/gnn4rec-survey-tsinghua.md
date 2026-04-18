# GNN4Rec: Graph Neural Networks for Recommender Systems Survey

> 来源: 清华大学 FIB Lab (Chen Gao, Yu Zheng et al.)
> 发表: ACM Transactions on Recommender Systems (TORS)
> 代码: https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems
> 处理日期: 2026-04-18

## 核心贡献

本文是 GNN 在推荐系统中应用的综合性 Survey，系统性地分析了：

### 1. GNN 方法分类
- **谱方法 (Spectral Models)**: 基于图拉普拉斯矩阵的频域滤波
- **空间方法 (Spatial Models)**: 基于邻居聚合的消息传递机制

### 2. 应用 GNN 到推荐系统的核心动机
- **高阶连通性 (High-order Connectivity)**: 通过多跳邻居捕获协同过滤信号
- **数据的结构性质 (Structural Property)**: 用户-物品交互天然形成图结构
- **增强的监督信号 (Enhanced Supervision Signal)**: 图结构提供额外的自监督信号

### 3. 关键技术挑战
- 图构建 (Graph Construction): 如何构建有效的交互图
- 嵌入传播与聚合 (Embedding Propagation/Aggregation): 信息如何在图中流动
- 模型优化 (Model Optimization): 训练效率与收敛性
- 计算效率 (Computation Efficiency): 大规模图上的可扩展性

### 4. 代表性工作
- **LightGCN**: 简化 GCN，去除非线性变换和特征变换
- **NGCF**: Neural Graph Collaborative Filtering
- **PinSage**: Pinterest 工业级 GraphSAGE 应用
- **SURGE**: Sequential Recommendation with Graph Neural Networks (SIGIR 2021)

## 面试考点
- GNN 在推荐中的优势 vs 传统 CF
- LightGCN 为什么去掉非线性变换反而效果更好
- 大规模图推荐的工程挑战（采样、分布式训练）
- 图构建策略对推荐效果的影响
