# ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT Models

> 来源：arXiv 2026 | 领域：search | 学习日期：20260408

## 问题定义

多向量检索模型（如 ColBERT）是否需要专门的预训练？

## 核心方法与创新点

1. **大规模多向量预训练**：
   - 使用 PyLate 框架
   - 对比学习 + 蒸馏

2. **关键发现**：
   - 仅蒸馏不够，需要对齐的监督微调
   - 预训练和微调之间的 prompt 对齐至关重要

3. **开源贡献**：
   - 所有模型和脚本 Apache 2.0 开源

## 关键结果

- 55.43 nDCG@10
- 仅用公开数据超越 GTE-ModernColBERT

## 面试考点

- 多向量检索的预训练策略
- 预训练-微调对齐问题
