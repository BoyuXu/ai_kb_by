# Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning
> ACL 2025 | Date: 20260409

## Core Contribution
Training-free, attention-based data selection method that leverages few-shot ICL to efficiently identify optimal training subsets for LLM fine-tuning. Achieves comparable performance using only 10% of data with 7.4x speedup.

## Key Techniques
- **Attention-based scoring**: Uses attention weights from ICL demonstrations to rank data importance
- **Weak-to-strong ICL integration**: Enables cross-model-family data selection
- **Attention-weighted demonstration refinement**: Accounts for contextual significance of each sample

## Technical Details
- Data selection is training-free — no gradient computation needed
- Leverages the model's own attention patterns as a proxy for data utility
- Achieves 7.4x speedup with 10% data while maintaining or improving task performance

## Industrial Implications
- Dramatically reduces fine-tuning costs for domain-specific LLM adaptation
- Enables faster iteration cycles in production ML pipelines
- Applicable to any task-specific fine-tuning workflow

## Interview Points
- Q: How to efficiently select fine-tuning data? A: Attention-based scoring from ICL, no training needed
- Q: Trade-off between data quantity and quality? A: 10% carefully selected data ≈ full dataset performance
