# Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation for Speculative Decoding
> 来源：https://arxiv.org/search/?query=aligning+draft+models+parameter+efficient+speculative+decoding&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
Speculative Decoding（投机解码）需要draft model（小模型）和target model（大模型）的分布高度对齐，但现有draft model与target model往往分布不匹配，导致接受率低。本文提出高效对齐draft model的方法。

## 核心方法与创新点
- 参数高效适配：用LoRA/Adapter以少量参数对齐draft model分布
- 数据高效：仅用target model生成的少量数据微调draft model
- 对齐目标：最小化draft model和target model的token分布KL散度
- 动态适配：根据不同任务类型动态调整draft model参数

## 实验结论
通过参数高效对齐，draft model的平均接受率从约60%提升至约80%；推理加速比从约1.8x提升至约2.5x；仅需target model约1%的参数量用于LoRA适配即可获得显著提升。

## 工程落地要点
- 不同任务（代码/对话/推理）需要分别训练draft model适配器
- draft model和target model必须共享词表，否则token分布无法对齐
- 生产系统中，draft model更新比target model更频繁（适应新任务）

## 面试考点
1. **Q: Speculative Decoding的工作原理？** A: 小draft model先生成多个token，大target model并行验证并接受/拒绝，接受则加速
2. **Q: draft model接受率如何计算？** A: 接受率α = draft分布与target分布的最小值之比，α高则加速比好
3. **Q: LoRA用于draft model对齐的具体实现？** A: 在draft model的注意力层添加LoRA矩阵，用target model的logits做KD训练
4. **Q: Speculative Decoding的加速比与哪些因素相关？** A: 接受率（越高越好）、draft步数（通常3-7步）、draft/target模型速度比
5. **Q: 什么场景下Speculative Decoding最有效？** A: 语言模型推理（分布稳定）、相对简单的任务（接受率高）、draft比target小10x以上
