# MTFM: A Scalable and Alignment-free Foundation Model for Industrial Multi-Task Recommendation
> 来源：arXiv:2602.11235 | 领域：rec-sys | 学习日期：20260330

## 问题定义
工业推荐系统通常同时优化 CTR、CVR、时长、满意度等多个目标。现有多任务方法（MMOE/PLE）需要针对每对任务设计对齐策略，复杂度随任务数 O(T²) 增长。MTFM 提出无需任务对齐（alignment-free）的统一基础模型，支持任意多任务扩展。

## 核心方法与创新点
1. **Task-Agnostic Backbone**：统一的 Transformer backbone 学习与任务无关的用户/物品表征，所有任务共享该表征。
2. **Task-Specific Prompt**：每个任务用可学习的 prompt token（task token）注入 backbone，无需设计 task-specific 结构。
3. **Alignment-Free 训练**：抛弃 MMOE/PLE 的门控对齐机制，用 task token 的注意力自然路由，避免手工设计 alignment。
4. **Continual Task Addition**：新任务只需添加新 task token 并微调，backbone 不变，支持 O(1) 任务扩展。
5. **Pre-training + Fine-tuning**：在海量曝光数据上预训练基础表征，下游任务 fine-tune task token，提升冷启动和低数据任务效果。

## 实验结论
- 对比 PLE baseline，8 任务场景 CTR AUC +0.8%，CVR AUC +1.2%，任务平均 AUC +0.9%
- 任务数从 4 扩展到 16，MTFM 性能持续提升，PLE 出现明显负迁移
- 新任务 fine-tune 只需 10% 数据达到 PLE 全量训练水平（pre-training 带来的迁移优势）

## 工程落地要点
- Task token 维度建议与 backbone hidden size 一致（512/768），过小欠拟合
- Prompt 注入位置影响效果：建议注入每一层（深层 prompt）而非仅输入层
- Backbone 预训练数据选择：用全量曝光（不区分任务）最优
- 推理时所有任务共享一次 backbone forward，只多跑 task head，延迟友好

## 面试考点
- Q: MMOE 和 PLE 的区别？
  - A: MMOE 所有任务共享专家池 + 各任务门控选专家；PLE 在此基础上增加 task-specific experts，减少任务间干扰
- Q: Foundation Model 在推荐系统中的主要挑战？
  - A: ID 特征的泛化（物品/用户在不同场景 ID 不通用）；低延迟推理（Transformer 比 DNN 慢）；数据分布 shift
- Q: Prompt tuning 在推荐系统里为什么有效？
  - A: prompt token 作为 soft task descriptor，引导 backbone 激活与该任务相关的特征路径，比 hard-coded gate 更灵活
