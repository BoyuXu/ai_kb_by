# OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment
> 来源：https://arxiv.org/abs/2502.18965 | 领域：rec-sys | 日期：20260323

## 问题定义
快手提出OneRec系统，解决工业级推荐系统多阶段pipeline的信息损失和目标不一致问题，用单一生成式模型实现统一召回排序，并引入迭代偏好对齐机制。

## 核心方法与创新点
- 统一Encoder-Decoder：用Transformer统一处理召回和排序
- 迭代偏好对齐（IPA）：多轮RLHF-like训练，逐步对齐用户偏好
- 物品树状编码：RQ-VAE生成分层token，控制生成空间
- 在线部署优化：prefix cache、batch prefill等工程优化
- 冷启动策略：新物品通过内容特征映射到token空间

## 实验结论
快手线上A/B实验：OneRec相比传统两阶段系统，用户观看时长+1.8%，用户留存率+0.5%；迭代偏好对齐每轮平均带来+0.3%的留存提升。

## 工程落地要点
- RQ-VAE的codebook需要定期重建，保持token空间的语义一致性
- prefix cache大幅降低生成推理的首token延迟
- 新物品冷启动需要内容tower和行为tower的异步更新机制

## 常见考点
1. **Q: OneRec中IPA（迭代偏好对齐）的具体实现？** A: 收集用户正负反馈→构建偏好对→DPO/PPO训练→更新策略模型→循环
2. **Q: RQ-VAE如何用于物品tokenization？** A: 残差量化VAE，将物品embedding量化为多层离散码本，形成层级token
3. **Q: 为什么快手选择OneRec而不是传统pipeline？** A: 行为多样性高（短视频），多阶段pipeline的目标不一致更明显
4. **Q: prefix cache在生成式推荐中如何工作？** A: 缓存用户历史context的KV cache，避免重复计算，降低首token延迟
5. **Q: 生成式推荐的物品覆盖率问题如何解决？** A: beam search宽度+约束解码+多样性采样+长尾物品的专项召回通道
