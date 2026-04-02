# Act-With-Think: Chunk Auto-Regressive Modeling for Generative Recommendation
> 来源：https://arxiv.org/abs/2506.23643 | 领域：rec-sys | 日期：20260323

## 问题定义
现有生成式推荐要么纯粹基于行为（act），要么先推理再行动（CoT），两者割裂。Act-With-Think提出分块自回归建模，在生成推荐时交织思考（reasoning）和行动（recommendation），实现更智能的推荐。

## 核心方法与创新点
- Chunk自回归：将生成序列分为"思考chunk"和"行动chunk"，交替生成
- 思考-行动交织：每生成一批物品前先生成用户意图分析文本
- 轻量推理：相比完整CoT，仅在关键决策点插入思考步骤
- 端到端训练：思考和推荐共同优化，推理过程对推荐有直接监督

## 实验结论
在多个推荐benchmark上，Act-With-Think相比纯行为生成提升NDCG约5%，相比完整CoT推理效率提升3x（更少思考token），质量接近。

## 工程落地要点
- 思考chunk的长度需要控制（通常10-50 token），避免过多推理增加latency
- 可以对思考过程进行缓存，相同用户profile复用思考结果
- 思考内容可以用于推荐解释（explainability），提升用户信任

## 常见考点
1. **Q: 为什么推荐系统需要"思考"步骤？** A: 捕获隐式用户意图，处理复杂多意图场景，提升长尾场景的推荐质量
2. **Q: Chain-of-Thought在推荐中的挑战？** A: 推理token增加latency、工业场景无监督思考数据、思考质量难以评估
3. **Q: Chunk自回归相比token自回归的优势？** A: 并行处理chunk内的token（speculative decoding），降低推理延迟
4. **Q: 推荐解释性（explainability）的意义？** A: 提升用户信任、帮助用户校正推荐、满足监管合规要求
5. **Q: Act-With-Think如何处理思考内容的监督？** A: 弱监督（仅监督行动结果，思考过程自由）或半监督（用LLM生成伪思考标签）
