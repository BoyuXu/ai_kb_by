# GNN4Rec: Graph Neural Networks for Recommendation Survey

> **来源**: Tsinghua FIB Lab — ACM TORS (2023)
> **GitHub**: https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems
> **领域**: rec-sys / graph-based recommendation

## 核心内容

清华大学 FIB Lab 的 GNN 推荐系统综述，覆盖 GNN 在推荐中的全景应用。

### 动机：为什么推荐需要 GNN

1. **高阶连通性** — 用户-物品交互天然形成图结构，GNN 可以捕获多跳邻居信息
2. **数据结构性** — 社交网络、知识图谱等辅助信息天然是图
3. **增强监督信号** — 通过图结构传播标签，缓解交互稀疏问题

### GNN 方法分类

| 类别 | 方法 | 代表模型 |
|------|------|---------|
| 谱方法 (Spectral) | 基于图傅里叶变换 | SpectralCF, ChebNet |
| 空间方法 (Spatial) | 基于邻居聚合 | GCN, GAT, GraphSAGE |
| 推荐专用 | 面向 user-item 二部图 | LightGCN, NGCF, PinSage |

### 关键挑战

- **图构建** — 如何定义边（隐式反馈/显式评分/知识图谱）
- **聚合策略** — mean/attention/LSTM 聚合各有优劣
- **可扩展性** — 工业场景亿级节点，需采样（PinSage neighbor sampling）
- **过平滑** — 层数增加导致节点表示趋同

### 面试考点

- Q: LightGCN 为什么去掉了特征变换和非线性？A: 实验发现推荐场景中这些操作不仅无用反而引入噪声，纯线性传播 + 层间加权平均效果最好。
- Q: GNN-based 推荐 vs CF 的核心优势？A: 捕获高阶协同信号（user→item→user→item 路径），传统 CF 只看一阶交互。

> **关联 synthesis**: [[20260421_generative_retrieval_and_long_sequence]] | [[20260513_fm_llm_genrec_survey_landscape]]
