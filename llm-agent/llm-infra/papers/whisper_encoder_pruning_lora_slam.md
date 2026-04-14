# Encoder Depth Pruning: Whisper + LoRA in SLAM-ASR

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2603.27981
- **Date**: 2026-04

## 核心贡献

分析Whisper编码器层剪枝对SLAM-ASR的影响，以及LoRA微调恢复性能退化的能力。

## 关键技术

- **渐进式层剪枝**: 对Small/Medium/Large-v2逐层剪枝
- **LoRA适应**: 在不同资源水平下用LoRA恢复性能
- **跨语言评估**: 丹麦语(4.2h)/荷兰语(50h)/英语(100h)

## 核心洞察

- 编码器深度与ASR性能的非线性关系
- LoRA在低资源语言中恢复效果更显著
- 模型压缩与性能保持的平衡点

## 面试考点

模型剪枝策略，LoRA原理与应用，语音-LLM系统架构。
