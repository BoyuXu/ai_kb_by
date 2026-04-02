# PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations
> 来源：https://arxiv.org/abs/2510.07784 | 领域：ads | 日期：20260323

## 问题定义
如何将预训练语言模型（PLM）高效适配到工业级别（数十亿用户/物品）的生成式推荐系统。PLUM研究大规模PLM在推荐场景的适配策略，解决语言模态与行为模态的domain gap。

## 核心方法与创新点
- 轻量化适配：仅微调少量参数（LoRA/Adapter），保留PLM通用知识
- 行为-语言对齐：用对比学习将用户行为序列与文本描述对齐
- 分层特征提取：浅层捕获语法，深层捕获语义，不同层对推荐任务贡献不同
- 工业Scale适配：批归一化、梯度裁剪等工业训练技巧

## 实验结论
在工业广告推荐系统，PLUM相比从头训练的推荐模型，AUC提升约0.8%；仅微调10%参数就能获得全量微调95%的效果；新广告冷启动场景提升最显著（+2% AUC）。

## 工程落地要点
- LoRA rank建议设为16-64，在效果和效率之间平衡
- PLM的tokenizer需要扩展物品ID词表，支持ID-based推荐
- 混合精度训练（BF16）在保证精度的同时降低显存需求

## 常见考点
1. **Q: LoRA（低秩适配）的原理？** A: 将参数矩阵的更新分解为低秩矩阵乘积ΔW=BA，只训练B和A，大幅减少参数
2. **Q: 为什么PLM需要适配（adaptation）才能用于推荐？** A: 语言预训练目标（Next Token Prediction）与推荐目标（CTR/CVR）不同，直接用效果差
3. **Q: 行为-语言对齐的意义？** A: 让模型理解"用户买了耐克鞋"与"Nike sneakers"是相同的概念
4. **Q: LoRA vs 全量微调的权衡？** A: LoRA：快、少参数、防过拟合；全量：效果上限更高，需要更多计算
5. **Q: 工业PLM适配的主要工程挑战？** A: 模型大（7B+参数）导致训练成本高、serving latency长、多副本部署成本
