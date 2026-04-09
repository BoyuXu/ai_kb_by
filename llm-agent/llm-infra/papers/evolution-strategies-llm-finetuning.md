# Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning

- **Date**: 2025-09 (revised 2026-02)
- **Domain**: LLM-Infra/Training
- **URL**: https://arxiv.org/abs/2509.24372
- **Code**: https://github.com/VsonicV/es-fine-tuning-paper

## 核心贡献

首次成功将进化策略(ES)应用于数十亿参数LLM的全参数微调，无需降维。ES不仅是RL的替代方案，而是一种根本不同的无反向传播后训练范式。

## ES vs RL优势

1. **样本效率**: 更高的样本利用率
2. **长期奖励容忍度**: 对长horizon奖励更鲁棒
3. **基座模型鲁棒性**: 跨不同base LLM表现稳定
4. **Reward Hacking**: 更少的奖励黑客倾向
5. **稳定性**: 跨运行更稳定

## 面试考点

- ES与RL在LLM后训练中的对比
- 无梯度优化方法在大模型中的可行性
- 后训练范式的演进：SFT → RLHF → ES
