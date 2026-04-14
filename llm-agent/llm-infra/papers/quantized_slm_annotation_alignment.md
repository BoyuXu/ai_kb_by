# More Human, More Efficient: Aligning Annotations with Quantized SLMs

- **Type**: Research Paper
- **URL**: https://arxiv.org/abs/2604.00586
- **Date**: 2026-04

## 核心贡献

证明微调后的1.7B量化小模型可作为确定性评估器，一致性超越商用LLM。

## 关键技术

- **多维评分框架**: 自定义rubric + 数据增强 + 正则化
- **量化SLM微调**: 在有限人工标注数据上微调量化小模型
- **本地部署**: 解决成本、隐私、偏差和可复现性问题

## 核心结果

- Krippendorff's α比最佳商用LLM高0.23
- 完全本地运行，成本极低
- 确定性输出，可复现

## 面试考点

模型量化技术，标注质量评估指标，小模型vs大模型trade-off。
