# DGenCTR: Universal Generative Paradigm for CTR Prediction via Discrete Diffusion

> 来源：https://arxiv.org/abs/2508.14500 | 领域：计算广告 | 学习日期：20260331

## 问题定义

如何将离散扩散模型高效应用到CTR预测的工业场景中？重点解决推理效率问题。

## 核心方法与创新点

1. **Absorbing State Diffusion**：使用吸收态扩散模型处理离散特征
2. **掩码预测训练**：随机掩码特征tokens，训练预测被掩码内容
3. **并行解码**：多个被掩码位置同时预测

$$
\mathcal{L} = \mathbb{E}_{t, x_0} [-\log p_\theta(x_0 | x_t)]
$$

4. **自适应步数控制**：根据输入复杂度动态调整去噪步数

## 实验结论

自适应步数机制使平均推理步数从5降到2.3，延迟减少54%而AUC仅降0.02%。

## 工程落地要点

- 并行解码利用GPU并行能力
- 自适应步数在简单样本上更快
- 可与传统CTR模型ensemble
- 掩码预测支持缺失特征自然处理

## 面试考点

1. **Absorbing State vs Gaussian扩散？** 前者适合离散数据
2. **掩码预测与BERT的关系？** 类似MLM但在特征空间
3. **自适应步数判断标准？** 基于模型置信度动态决定
