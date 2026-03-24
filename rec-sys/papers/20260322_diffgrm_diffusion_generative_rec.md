# DiffGRM: Diffusion-based Generative Recommendation Model

> 来源：arxiv (https://arxiv.org/abs/2510.21805) | 日期：20260322 | 领域：推荐系统

## 问题定义

生成式推荐主流范式（自回归序列生成）存在两个问题：
1. **曝光偏差（Exposure Bias）**：训练时用真实 token，推理时用预测 token，分布不匹配
2. **单向生成限制**：自回归模型只能从左到右生成，无法利用未来 token 的上下文

DiffGRM 引入扩散模型范式解决上述问题。

## 核心方法与创新点

- **连续空间扩散**：在 Semantic ID 的连续嵌入空间（而非离散 token）进行扩散，避免离散扩散的不稳定性
- **条件扩散生成**：
  - 条件：用户历史序列的编码向量
  - 目标：从高斯噪声去噪恢复目标 item 的 Semantic ID 嵌入
- **双向 Attention**：扩散过程使用 DDPM 中的双向 Transformer，充分利用上下文
- **离散化检索**：去噪得到连续嵌入后，通过 ANN 检索最近邻 Semantic ID，得到候选 item
- **课程扩散（Curriculum Diffusion）**：训练时按难度递增安排噪声水平，先学易区分的 item 再学难区分的

## 实验结论

- 在 ML-1M、Amazon Beauty、Yelp 数据集上，HR@10 平均提升 8.3%（vs 自回归 baseline TIGER）
- 多样性指标（Intra-List Diversity）提升 12%：扩散模型的随机性带来推荐多样化
- 训练收敛速度比自回归慢 2×，但推理时无需逐 token 生成，延迟相当
- 消融：课程扩散 vs 随机噪声水平，前者 +3.2% HR@10

## 工程落地要点

- **扩散步数选择**：训练用 T=1000 步，推理用 DDIM 加速到 50 步，精度损失<1%
- **条件注入方式**：将用户历史编码作为 cross-attention key/value 注入扩散 Transformer
- **冷启动友好**：扩散模型的去噪过程可融合多种条件（内容特征、用户画像），冷启动表现好
- **延迟控制**：DDIM 50 步 + batch size 优化，p99 延迟控制在 80ms 内可接受

## 面试考点

1. **Q：扩散模型用于推荐相比自回归生成的核心优势？**
   A：①无曝光偏差，训练推理过程一致；②双向 attention 利用全局上下文；③随机性提升推荐多样性；④并行去噪（非逐 token 自回归）

2. **Q：为什么在连续空间而非离散 token 空间扩散？**
   A：离散扩散（如 Discrete Diffusion）数学不稳定、梯度计算困难；连续空间扩散理论完备，DDPM/DDIM 框架直接可用

3. **Q：扩散推荐在工业部署的主要挑战？**
   A：多步去噪延迟（需 DDIM/DDPM 加速）；扩散随机性导致结果不确定（线上 A/B 控制困难）；模型规模通常比自回归大
