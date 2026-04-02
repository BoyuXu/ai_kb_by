# ETEGRec: Generative Recommender with End-to-End Learnable Item Tokenization
> 来源：https://arxiv.org/abs/2409.05546 | 领域：rec-sys | 日期：20260323

## 问题定义
现有生成式推荐中，物品tokenization（如RQ-VAE）与推荐模型分离训练，导致token不能最优化地服务推荐任务。ETEGRec提出端到端可学习的物品tokenization，让token学习与推荐目标联合优化。

## 核心方法与创新点
- 端到端联合训练：tokenizer和推荐模型同时优化，token学习推荐感知的表示
- 软量化：用Gumbel-Softmax实现可微分的离散量化，支持梯度回传
- 两阶段训练策略：预训练tokenizer + 联合微调，避免训练不稳定
- 推荐感知码本：码本的更新受推荐loss的直接监督

## 实验结论
在Amazon和MovieLens等标准benchmark上，ETEGRec比SASRec等传统模型提升5-12%的NDCG；比TIGER（分离训练tokenizer）提升约3%。

## 工程落地要点
- 端到端训练计算量大，需要更长训练时间和更大内存
- Gumbel temperature需要退火调度，初期高温探索，后期低温收敛
- 工业场景中物品动态增减需要支持tokenizer的在线更新

## 常见考点
1. **Q: 为什么分离训练tokenizer会导致次优？** A: tokenizer优化重建loss，不直接优化推荐目标，存在目标gap
2. **Q: Gumbel-Softmax如何实现可微分离散化？** A: 通过温度参数τ控制softmax的sharp程度，τ→0时近似argmax
3. **Q: 端到端训练的主要困难是什么？** A: 离散化操作不可微、梯度消失、码本崩溃（codebook collapse）
4. **Q: 码本崩溃的原因和解决方案？** A: 部分code未被使用导致退化；解法：EMA更新、random restart、commitment loss
5. **Q: ETEGRec在实际系统中的部署难点？** A: tokenizer更新需重建全量物品token，影响在线serving稳定性
