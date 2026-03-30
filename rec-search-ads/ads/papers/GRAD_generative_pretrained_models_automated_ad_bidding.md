# GRAD: Generative Large-Scale Pre-trained Models for Automated Ad Bidding Optimization
> 来源：arXiv:2508.02002 | 领域：ads | 学习日期：20260330

## 问题定义
自动出价（Auto-bidding）通常针对单广告主独立优化，忽略了跨广告主的市场共性规律。GRAD 提出用大规模预训练生成模型学习广告竞价市场的通用动态，通过 prompt 条件化适配各广告主策略，实现"一个模型服务所有广告主"。

## 核心方法与创新点
1. **Generative Bidding Model**：将出价决策建模为序列生成：给定广告主历史 budget 消耗、竞价结果序列，模型自回归预测下一时刻最优出价乘数。
2. **Market Context Pre-training**：在全量广告主数据上预训练，学习市场动态（竞价压力、时段规律、物料质量分布）等通用知识。
3. **Per-Advertiser Prompt**：每个广告主用其 KPI 目标、历史表现作为 prompt condition，无需为每个广告主单独训练模型。
4. **Diffusion-based 出价生成**：用 Diffusion Model（而非 AR 模型）在连续出价空间建模，支持多峰分布和不确定性估计。
5. **预算感知采样**：在 Diffusion 采样过程中注入预算约束（guidance），保证生成的出价序列满足约束。

## 实验结论
- 某平台 1000+ 广告主测试：平均 GMV +4.2%，ROI +3.1%（对比人均独立 RL 出价）
- 新广告主（<7 天历史）收益提升 +9.8%（预训练市场知识迁移优势）
- 单一 GRAD 模型 vs 1000 个独立模型：维护成本降低 99%，效果仍优

## 工程落地要点
- Diffusion 模型推理步数（denoising steps）需控制（建议 DDIM 10-20 步），满足实时出价延迟
- Prompt 设计：广告主 KPI（ROI 目标/日预算/行业类别）需标准化编码
- 预训练数据需做隐私保护（不同广告主数据聚合时用差分隐私或联邦学习）
- 上线采用 shadow mode 对比，人工出价作为 fallback

## 面试考点
- Q: 为什么用生成模型做出价，而不是传统 RL？
  - A: 生成模型（特别是 Diffusion）能建模出价的多峰分布，处理不确定性更好；预训练共享市场知识，样本效率更高；RL 需要大量在线探索，成本高
- Q: Diffusion Model 如何保证出价满足预算约束？
  - A: Classifier-free guidance：训练时同时学习 conditional（有预算）和 unconditional 分布；推理时 guided diffusion 向满足约束的方向移动
- Q: 广告竞价中的探索-利用权衡（Exploration-Exploitation）？
  - A: UCB/Thompson Sampling 在出价策略上做探索；预算约束限制探索空间；在线 A/B 实验是最终探索机制
