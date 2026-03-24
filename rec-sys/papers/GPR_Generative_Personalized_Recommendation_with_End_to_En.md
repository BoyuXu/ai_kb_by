# GPR: Generative Personalized Recommendation with End-to-End Advertising System Deployment
> 来源：https://arxiv.org/search/?query=GPR+generative+personalized+recommendation+advertising&searchtype=all | 领域：rec-sys | 日期：20260323

## 问题定义
传统推荐系统将召回和排序割裂，广告系统部署时存在多阶段不一致问题。GPR提出端到端生成式个性化推荐，统一广告召回与排序，直接在广告投放系统中部署。

## 核心方法与创新点
- 生成式框架：将推荐视为序列生成任务，自回归生成物品ID序列
- 端到端训练：召回与排序联合优化，消除多阶段误差积累
- 个性化tokenization：基于用户历史行为构建个性化物品token
- 广告系统集成：解决生成式模型在实际广告投放的工程落地问题
- beam search解码：保证生成多样性与质量的平衡

## 实验结论
在大规模广告系统上，GPR相比传统两阶段系统在CTR和RPM指标上均有显著提升，端到端训练比分阶段训练在AUC上提升约1%。

## 工程落地要点
- 物品tokenization需要离线预计算，实时serving依赖token索引
- beam search宽度是效果与延迟的关键权衡参数（通常beam=10-50）
- 需要建立token-to-item的反查表，支持实时广告竞价

## 面试考点
1. **Q: 生成式推荐与传统召回+排序的本质区别？** A: 生成式直接输出物品序列，联合优化召回排序；传统方式分阶段，存在目标不一致
2. **Q: 物品tokenization的常见方案？** A: RQ-VAE分层量化、语义哈希、层级分类树编码
3. **Q: 生成式推荐在工业落地的主要挑战？** A: 百万级物品的beam search效率、token空间的冷启动、实时latency约束
4. **Q: 端到端训练如何处理召回和排序的目标差异？** A: 多任务学习或统一loss设计，用listwise ranking loss
5. **Q: GPR如何保证广告的商业收益目标？** A: 在生成过程中融入bid信号，用RPM而非仅CTR作为优化目标
