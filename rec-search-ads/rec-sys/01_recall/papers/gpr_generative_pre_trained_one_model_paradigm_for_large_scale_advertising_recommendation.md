# GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation
> 来源：arxiv/2311.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
广告推荐系统中存在多个独立子系统（召回、粗排、精排、重排），各自优化目标不一致：
- 多阶段漏斗存在目标不对齐（召回高 recall，精排高 CTR）
- 子系统参数无法共享，维护成本高
- 无法利用大规模预训练的语义理解能力
- 跨场景、跨任务迁移困难

## 核心方法与创新点
**GPR (Generative Pre-trained Recommendation)**：统一多阶段推荐为单一生成式模型。

**核心思路：**
将推荐任务统一为条件生成问题：
```
P(item_id | user_context, task_prefix) = Π P(token_t | token_{<t}, context)
```

**Semantic ID 设计：**
- 用分层量化（RQ-VAE）将 item 编码为离散 token 序列
- 每个 item = [c_1, c_2, ..., c_k]，k 层码本
- 生成过程：逐 token 自回归解码

**预训练 + 微调框架：**
```
预训练目标：L_pretrain = -Σ log P(item_t | history_{<t})
微调目标：  L_finetune = -Σ λ_task · log P(item | context, task)
```

**任务前缀（Task-Conditioned）：**
- [RECALL]: 全库检索
- [RANKING]: 候选集重排
- [CTR]: 点击率预估

## 实验结论
- 统一模型 vs 独立多阶段：整体 GMV +3.2%
- 召回阶段 Recall@100：+5.8%
- 精排 AUC：持平（减少了 stage gap 损失）
- 模型参数量：比多套独立模型总和少 40%

## 工程落地要点
1. **Semantic ID 生成**：离线 RQ-VAE 训练，每日/每周更新码本
2. **Beam Search 检索**：生成式召回用 beam search，k=100 取 top 候选
3. **前缀路由**：线上按任务动态切换 prefix token，共享主干参数
4. **增量更新**：新 item 需及时生成 Semantic ID 并更新码本映射
5. **延迟控制**：生成式解码比向量检索慢，需 speculative decoding 加速

## 面试考点
**Q1: GPR 为什么能统一召回和排序？**
A: 通过任务前缀条件化，同一模型根据不同前缀 token 切换行为模式：召回时做全库 beam search 生成，排序时做候选集 scoring，共享底层语义表示。

**Q2: Semantic ID 相比 item ID 的优势？**
A: Semantic ID 捕获 item 语义（相似 item 有相近码字），天然支持跨 item 泛化，缓解长尾和冷启动问题。传统 ID 是独立哈希，无语义关联。

**Q3: 生成式召回的主要挑战？**
A: ①解码延迟高（需自回归多步）②新 item 的 Semantic ID 更新滞后 ③beam search 难以覆盖全库多样性。

**Q4: GPR 训练时如何避免"召回-排序"目标冲突？**
A: 多任务损失加权：L = λ_recall · L_recall + λ_rank · L_rank，不同任务有独立的训练样本（召回用全量曝光，排序用精选候选集）。

**Q5: 如何评估统一模型效果？**
A: 不能只看单阶段指标，需看端到端 GMV/RPM；同时对比各阶段独立模型的 stage gap（上游输出在下游的落差）。
