# Ch7 Embedding 技术与向量召回

## 子主题索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [word2vec-item2vec.md](word2vec-item2vec.md) | Word2Vec原理(Skip-gram/CBOW)、负采样、Item2Vec行为序列构造 | ~220 |
| [graph-embedding.md](graph-embedding.md) | DeepWalk, Node2Vec, LINE, GraphSAGE, Metapath2Vec, LightGCN 详细对比 | ~250 |
| [dual-tower.md](dual-tower.md) | 双塔模型架构、训练技巧(温度系数/难负例)、在线更新、工业实践 | ~250 |
| [ann-retrieval.md](ann-retrieval.md) | HNSW, IVF-PQ, ScaNN 等ANN索引对比、双索引架构、工程选型 | ~270 |

---

## 核心概念总览

本章覆盖 Embedding 全技术栈：Word2Vec/Item2Vec 基础、图嵌入（DeepWalk/Node2Vec/GraphSAGE）、
双塔向量召回系统、ANN 检索、多模态 Embedding、在线学习与更新、评估优化。

---

## Embedding 基础速查

### Embedding vs One-Hot

```
              Embedding              One-Hot
──────────────────────────────────────────────
维度          低维（64-256）          高维（=词表大小）
语义信息      包含（相似物品向量近）  不包含（正交）
泛化能力      强（相似物品共享语义）  无
冷启动        可通过侧信息初始化      无法处理新物品
```

### 冷启动 Embedding 方案

```
1. 内容特征生成: BERT编码文本 / ResNet提取图像 → 映射到 Embedding 空间
2. 图神经网络: 新物品加入交互图 → GNN 消息传递聚合邻居信息
3. 元学习(MAML): 学习好的初始化参数，少量交互即可微调
4. 非 Embedding 补充: 新品召回通道（规则/热门）
```

---

## 多模态 Embedding（概要）

### 对比学习统一空间

```
核心: InfoNCE 损失
  L = -log( exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau) )

流程: 正样本对(同一物品的图片+文本) → 模态编码器 → 投影层 → 对比损失
机理: 迫使模型学习跨模态共享的高层语义信息
```

### 融合方法对比

```
融合策略      时机              优点                缺点                适用场景
──────────────────────────────────────────────────────────────────────────
早期融合      特征级拼接后      模态交互最充分      对齐要求严格        模态关联强且对齐好
                               端到端训练          数据缺失敏感        (视频音画/商品主图+标题)

中期融合      Cross-Attention   灵活+交互兼顾      计算开销较大        跨模态检索
                                                                     (图搜商品/文搜商品)

晚期融合      各自出Embedding   最灵活              无法深层交互        生产环境常用
              后加权            支持模态缺失        可解释性较好        (稳定/可独立迭代)
```

---

## 在线学习与更新（概要）

### 在线更新路径对比

```
                    梯度下降在线学习              ANN 增量更新
────────────────────────────────────────────────────────────────
更新粒度        实时（秒级）                  定期（小时/天级）
优点            捕捉实时兴趣变化              系统稳定，模型服务解耦
缺点            灾难性遗忘，易发散            延迟高，无法捕捉实时变化
适用场景        新闻/短视频                   电商/长视频

生产首选: 混合架构（在线处理热点 + 离线全量修正偏差）
```

### 防灾难性遗忘策略

```
1. 自适应学习率: 高频物品小LR，新物品大LR
2. L2 正则化: 约束参数偏离旧值幅度
3. 回放缓冲区: 混入历史样本缓解遗忘
4. 分阶段更新: 固定基础 Embedding，仅更新上层
5. 监控+回滚: AUC/相似度异常时切回稳定版本
```

---

## Embedding 评估与优化（概要）

### 评估体系

```
内在评估:
  1. 相似度相关性: Embedding余弦相似度 vs 人工标注 → 斯皮尔曼相关系数
  2. 最近邻重合度: K近邻 vs 业务定义的相似集合
  3. 降维可视化: t-SNE/UMAP 观察聚类结构

外在评估:
  1. 召回率 Recall@K: 最直接的业务指标
  2. 分类/聚类: NMI/ARI/F1
  3. MAP/NDCG: 排序质量

线上验证:
  AB 测试: CTR, CVR, 停留时长, GMV
  重点关注冷启动和长尾物品的提升
```

---

## 技术选型速查

```
场景                          推荐方案
──────────────────────────────────────────────────
有明确行为序列                Item2Vec (Skip-gram)
需要图结构信息                Node2Vec / GraphSAGE
异构图（多类型节点）          Metapath2Vec / HGNN
冷启动严重                    多模态 Embedding + GNN
实时性要求高                  在线学习 + 增量索引
十亿级向量库                  IVF-PQ（一级） + HNSW（二级）
需要跨模态检索                对比学习（InfoNCE）
召回阶段主模型                双塔模型 (DSSM/YouTube DNN)
```

---

## 面试高频考点

```
1. Skip-gram vs CBOW → 为什么推荐更用 Skip-gram → 详见 word2vec-item2vec.md
2. Item2Vec 序列构建 → 四种方法优缺点 → 详见 word2vec-item2vec.md
3. DeepWalk vs Node2Vec → p/q 参数 BFS/DFS → 详见 graph-embedding.md
4. GraphSAGE 归纳学习 → 为什么能冷启动 → 详见 graph-embedding.md
5. 双塔训练技巧 → 温度系数/困难负例/In-batch → 详见 dual-tower.md
6. HNSW vs IVF-PQ → 内存/精度/速度权衡 → 详见 ann-retrieval.md
7. 向量索引更新策略 → 双索引+蓝绿发布 → 详见 ann-retrieval.md
8. 对比学习统一多模态空间 → InfoNCE 损失
9. 早期 vs 晚期融合 → 适用场景分析
10. 在线学习的灾难性遗忘 → 回放缓冲区/正则化约束
```
