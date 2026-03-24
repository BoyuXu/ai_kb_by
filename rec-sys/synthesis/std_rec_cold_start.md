# 推荐系统冷启动：新用户、新物品、新场景的破冰

> 📚 参考文献
> - [Recommendation-Cold-Start](../../rec-sys/papers/20260317_recommendation-cold-start.md) — 推荐系统冷启动（Cold Start）


> 创建：2026-03-24 | 领域：推荐系统 | 类型：综合分析
> 来源：DropoutNet, VARK, IDProxy, Meta-Learning, Content-based 系列

---

## 🎯 核心洞察（4条）

1. **冷启动分三类**：新用户（无行为历史）→ 实时行为捕获 + Explore；新物品（无交互数据）→ 内容特征 + 相似物品迁移；新场景（新业务线）→ 跨域迁移学习
2. **内容特征是冷启动最可靠的信号**：物品标题/图片/视频的预训练编码（BERT/CLIP），即使没有任何用户交互也能提供语义表示
3. **Meta-Learning 是冷启动的理论最优方案**：MAML 学习"快速适应"的初始化参数，few-shot 学习新用户/新物品，但工程复杂度高
4. **DropoutNet 思想简单有效**：训练时随机丢弃 ID Embedding，迫使模型学习用内容特征做预测，部署时对冷启动物品自然有效

---

## 🎓 面试考点（4条）

### Q1: 新用户冷启动怎么做？
**30秒答案**：①注册信息画像（年龄/性别 → 人群热门推荐）；②实时行为捕获（前 5 次点击快速更新用户 Embedding）；③Bandit 探索（UCB/TS 快速试探用户兴趣）；④跨平台数据（Device ID 关联其他 App 的行为）。

### Q2: 新物品冷启动怎么做？
**30秒答案**：①内容特征（BERT 编码标题 + CLIP 编码图片）代替 ID Embedding；②相似物品迁移（找内容最相似的热门物品，用其 CTR 做初始估计）；③流量倾斜（给新物品额外曝光机会积累数据）。

### Q3: DropoutNet 的原理？
**30秒答案**：训练时以概率 p 将物品 ID Embedding 置零，模型被迫从内容特征中学习有效表示。部署时，热门物品用 ID+内容联合表示，冷启动物品只用内容特征，模型已经学会了如何利用内容特征。

### Q4: 跨域推荐怎么做冷启动？
**30秒答案**：①共享用户 Embedding（同一用户在不同业务线的 ID Embedding 共享/对齐）；②知识蒸馏（成熟业务的模型蒸馏知识到新业务模型）；③预训练+微调（在所有域数据上预训练，在新域上微调）。

---

## 🌐 知识体系连接

- **上游依赖**：内容编码（BERT/CLIP）、Meta-Learning、Bandit 算法
- **下游应用**：新品推广、用户增长、跨域推荐
- **相关 synthesis**：std_ads_cold_start.md, std_rec_recall_evolution.md, std_rec_feature_engineering.md
