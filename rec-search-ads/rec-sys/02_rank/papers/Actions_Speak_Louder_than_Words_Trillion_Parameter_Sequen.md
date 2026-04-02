# Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
> 来源：https://arxiv.org/abs/2402.17152 | 领域：rec-sys | 日期：20260323

## 问题定义
Meta提出的工业级大规模生成式推荐系统，探索万亿参数规模的序列转换器（Sequential Transducer）在推荐系统中的应用，证明scaling law在推荐领域同样有效。

## 核心方法与创新点
- 万亿参数模型：将推荐模型规模扩展到1T+参数，验证scaling law
- 行为序列优先：强调用户行为序列（actions）比文本描述更重要
- Sequential Transducer：借鉴ASR中的transducer结构，流式处理用户行为
- 分布式训练：针对超大规模模型的工程优化（模型并行、流水线并行）
- 多任务联合训练：CTR、时长、完播率等多目标统一建模

## 实验结论
在Meta视频推荐场景，万亿参数模型比百亿参数模型在线上NE（Normalized Entropy）提升0.5%，用户视频观看时长提升显著；scaling law在推荐领域有效但增益边际递减。

## 工程落地要点
- 万亿参数需要数百GPU，训练成本极高，非头部公司难以复现
- 推理serving需要模型蒸馏或量化，将大模型知识迁移到小模型
- 分布式训练的通信效率是关键瓶颈，需要专门优化

## 常见考点
1. **Q: Scaling Law在推荐系统中是否成立？** A: 成立，但边际收益递减，且推荐的scaling受数据质量限制更大
2. **Q: Sequential Transducer相比Transformer的优势？** A: 流式处理，latency更低；可以实时更新用户状态
3. **Q: 工业推荐模型为何选择行为序列而非用户画像？** A: 行为反映实时意图，画像是静态的；行为数据量大，监督信号丰富
4. **Q: 万亿参数推荐模型的主要工程挑战？** A: 内存墙（单机放不下）、通信开销、推理latency、训练不稳定
5. **Q: 如何将超大模型的能力迁移到小模型？** A: 知识蒸馏（KD）、逐层蒸馏、行为对齐蒸馏
