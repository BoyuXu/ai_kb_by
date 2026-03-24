# 推荐系统知识库导航 🎬

## 📊 领域概览

| 分类 | 文档数 | 描述 |
|------|--------|------|
| **Papers** (学术论文笔记) | 69篇 | 召回、排序、重排、协同过滤等 |
| **Practices** (工业实践案例) | 0篇 | Netflix、YouTube、TikTok 等大厂案例（待补充） |
| **Synthesis** (提炼总结) | 26篇 | 推荐系统演进、LLM 融合框架、标准化文档 |
| **总计** | 80篇 | - |

---

## 🚀 快速导航

### 📚 按学习阶段查找

1. **推荐系统基础** → 推荐系统概览（待补充）
2. **召回算法** → 召回系统设计
3. **排序算法** → 排序与重排
4. **LLM 时代** → [LLM 集成框架](./synthesis/llm_integration_framework.md)
5. **系统设计** → [推荐演进](./synthesis/recommendation_evolution.md)

### 🎯 研究方向

#### 🔍 召回 (Recall)
- 多路召回设计
- 召回算法对标
- 近似最近邻搜索

#### 🔄 排序 (Ranking)
- LTR (Learning-to-Rank)
- 神经排序
- 多目标排序

#### 🎭 重排 (Re-ranking)
- 多样性控制
- 重排模型
- 序列优化

#### 🧠 协同过滤 (Collaborative Filtering)
- 用户-物品矩阵
- 显式反馈
- 隐式反馈

#### 🤖 深度学习推荐
- 深度神经网络
- 注意力机制
- 图神经网络

#### 📊 多目标优化
- 多目标平衡
- 帕累托优化

#### 💾 LLM 集成
- LLM 作为排序器
- LLM 作为召回器
- LLM 生成式推荐

---

## 📚 完整文档列表

### 📄 Papers (69篇 学术论文笔记)

**注**: 当前知识库主要包含推荐系统的学术论文。具体文件列表可通过以下命令查看：

```bash
ls -la ~/Documents/ai-kb/rec-sys/papers/
```

主要覆盖方向：
- 召回算法设计
- 排序模型
- 用户行为预测
- 协同过滤方法
- 图神经网络应用
- 多目标优化
- 离线评估方法
- 在线实验设计

### 🏢 Practices (0篇 工业实践案例 - 待补充)

建议补充大厂案例：
- Netflix 推荐系统架构
- YouTube 推荐系统
- TikTok 推荐系统
- Alibaba 推荐系统
- Amazon 推荐系统
- Meta 推荐系统

### 📖 Synthesis (26篇 提炼总结)

#### 标准化文档 (std_*)
- [冷启动](./synthesis/std_rec_cold_start.md)
- [Embedding 学习](./synthesis/std_rec_embedding_learning.md)
- [特征工程](./synthesis/std_rec_feature_engineering.md)
- [图神经网络](./synthesis/std_rec_graph_neural_network.md)
- [在线实验](./synthesis/std_rec_online_experiment.md)
- [排序演进](./synthesis/std_rec_ranking_evolution.md)
- [召回演进](./synthesis/std_rec_recall_evolution.md)
- [重排与多样性](./synthesis/std_rec_rerank_diversity.md)
- [用户行为建模](./synthesis/std_rec_user_behavior_modeling.md)

#### 专题综合
- [Semantic ID 生成式检索](./synthesis/20260321_semantic_id_generative_retrieval.md)
- [生成式推荐范式对比](./synthesis/20260322_generative_rec_paradigm_comparison.md)
- [Semantic ID 全景](./synthesis/20260322_semantic_id_full_picture.md)
- [生成式推荐全谱](./synthesis/20260323_generative_rec_full_spectrum.md)
- [推荐系统综合](./synthesis/20260323_rec_sys_synthesis.md)
- [Wukong 推荐 Scaling Law](./synthesis/20260323_recommendation_scaling_law_wukong.md)
- [LLM 集成框架](./synthesis/llm_integration_framework.md)

#### 基础总结
- [概览](./synthesis/00_overview.md)
- [CTR 模型深度](./synthesis/01_ctr_models_deep_dive.md)
- [特征工程](./synthesis/03_feature_engineering.md)
- [工业排序论文](./synthesis/05_industry_ranking_papers.md)
- [排序深度](./synthesis/05_ranking_deep.md)
- [工业召回论文](./synthesis/06_industry_recall_papers.md)
- [因果推断](./synthesis/07_causal_inference.md)
- [重排多样性](./synthesis/08_rerank_diversity.md)
- [特征存储实践](./synthesis/09_feature_store_practice.md)

---

## 💡 使用指南

### 学习路径

**初级工程师** (1-2年)
1. 理解推荐系统架构
2. 学习基础召回与排序
3. 实现简单的推荐系统

**中级工程师** (2-5年)
1. 深入理解多目标排序
2. 掌握在线学习与强化学习
3. 优化系统性能

**高级工程师** (5年+)
1. LLM 时代的推荐范式
2. 系统架构与规模化
3. 创新研究

### 按应用场景

**电商推荐**
- 性能最优
- 转化率优先
- 库存约束

**内容推荐**
- 多样性优先
- 点击率优先
- 用户粘性

**社交推荐**
- 关系图优先
- 新颖性约束
- 隐私保护

---

## 🔗 知识体系

```
推荐系统
├── 🔍 召回层
│   ├── 协同过滤
│   ├── 基于内容
│   ├── 向量检索
│   └── 多路召回
│
├── 🔄 排序层
│   ├── 学习排序 (LTR)
│   ├── 神经排序
│   ├── 多目标排序
│   └── 实时排序
│
├── 🎭 重排层
│   ├── 多样性控制
│   ├── 序列优化
│   └── 应用策略
│
└── 🤖 LLM 时代
    ├── LLM 排序
    ├── LLM 召回
    ├── 生成式推荐
    └── 多模态推荐
```

---

## 📊 论文统计

- **总数**: 69 篇
- **类型分布**: 
  - 算法论文: ~40 篇
  - 系统论文: ~15 篇
  - 应用论文: ~14 篇
- **年份分布**: 主要为 2024-2026 年最新论文

---

## 💼 相关资源

### 开源项目
- [fun-rec](https://github.com/datawhalechina/fun-rec) - 推荐算法入门
- [AlgoNotes](https://github.com/shenweichen/AlgoNotes) - 算法工程笔记

### 学习社区
- 推荐系统论坛
- GitHub 推荐相关项目
- 顶会论文 (RecSys, SIGIR, WWW)

---

## 📝 最后更新

- **最后更新**: 2026-03-24
- **总文档数**: 80 篇
- **近期更新**: 补充 LLM 在推荐中的应用

> 💡 **提示**: 推荐系统与搜索广告密切相关。建议对比学习相邻领域的论文！
