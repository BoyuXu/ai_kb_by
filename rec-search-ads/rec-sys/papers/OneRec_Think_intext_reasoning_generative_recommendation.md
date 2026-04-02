# OneRec-Think: In-text Reasoning for Generative Recommendation
> 来源：arXiv:2510.11639 | 领域：rec-sys | 学习日期：20260330

## 问题定义
生成式推荐系统（Generative Recommendation）直接生成物品 ID 序列，但缺乏对用户意图的显式推理过程。OneRec-Think 在生成文本内部嵌入推理步骤（in-text reasoning），使模型在生成推荐序列前先"思考"用户需求，提升推荐精度和可解释性。

## 核心方法与创新点
1. **In-text Reasoning**：在自回归生成 token 流中插入 `<think>...</think>` 推理块，模型先生成推理再生成推荐 ID，无需额外推理模块。
2. **训练策略**：两阶段训练——第一阶段 SFT 教会格式；第二阶段用 GRPO（Group Relative Policy Optimization）强化学习提升推理质量，以命中率作为 reward。
3. **与 OneRec 一脉相承**：继承 Decoder-only 生成式推荐架构，在同一模型内完成召回+排序+生成。
4. **推理蒸馏**：大模型生成推理链后蒸馏到小模型，降低线上推理成本。

## 实验结论
- 在快手工业数据集上 NDCG@10 提升 4.2%，Hit@10 提升 3.8%（对比 OneRec baseline）
- 强化学习阶段（GRPO）相比纯 SFT 多提升 1.5% NDCG
- 推理链蒸馏后小模型性能损失 <1%，延迟降低 60%

## 工程落地要点
- 推理块 token 数控制（建议 ≤128），过长影响整体延迟
- GRPO 训练需要在线采样，计算成本高，建议 offline batch reward 近似
- 工业部署可用 Speculative Decoding 提速，推理 token 由草稿模型生成
- 推理质量对 reward 信号设计敏感，需精心设计 hit@k / diversity 组合

## 常见考点
- Q: 生成式推荐和传统 CTR 排序的核心区别？
  - A: 传统 CTR 逐物品打分（pointwise/listwise），生成式直接 autoregressive 生成物品序列，天然支持长程依赖
- Q: GRPO 和 PPO 的区别？
  - A: GRPO 用组内相对奖励归一化，无需 critic 网络，训练更稳定，适合推荐这种离散奖励场景
- Q: In-text reasoning 推理链如何验证质量？
  - A: 除命中率外，可用 diversity、novelty 等二级指标；也可用 LLM-as-Judge 评判推理逻辑一致性
