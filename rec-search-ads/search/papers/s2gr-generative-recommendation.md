# S2GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation

- **Date**: 2026-01
- **Domain**: Search/Recommendation
- **URL**: https://arxiv.org/html/2601.18664v1

## 核心贡献

S2GR提出在生成式推荐的潜在空间中使用逐步语义引导推理。thinking tokens作为过渡模块，对相关粗粒度语义簇进行推理，指导后续Semantic ID (SID)代码生成。

## 关键技术

- Thinking tokens直接对应粗粒度语义分布，具有可解释的物理意义
- 训练时使用ground-truth粗粒度SID语义约束thinking tokens，确保推理路径可靠性
- 每个SID code在经过验证的语义推理后生成，显著提高输出质量

## 与现有工作对比

- 对比STREAM-Rec和GPR：这些方法直接用相似item表示填充推理链或从意图embedding合成潜在推理状态
- S2GR通过可解释的语义引导克服了这些局限

## 面试考点

- 生成式推荐中Semantic ID的设计与编码策略
- Chain-of-thought推理在推荐中的应用范式
- 粗粒度到细粒度的层次化生成策略
