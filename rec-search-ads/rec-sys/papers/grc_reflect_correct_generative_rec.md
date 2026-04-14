# GRC: Learning to Reflect and Correct for Generative Recommendation

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2602.23639

## 核心贡献

生成-反思-纠正(GRC)框架，使生成式推荐模型能自我检测并修正解码错误。

## 关键技术

- **三阶段解码**: Generation → Reflection → Correction
- **多粒度反思信号**: 检测初稿中的偏差
- **GRPO-RL**: 优化反思-纠正决策
- **EGRS**: 熵引导奖励调度，平衡延迟约束

## 核心结果

- 基线提升15.74%
- A/B测试收入提升1.79%
- 可与beam search集成

## 面试考点

生成式推荐解码策略，自纠正机制，RL在推荐系统中的应用。
