# DiffGRM: Diffusion-based Generative Recommendation Model

> 来源：arxiv | 日期：20260321 | 领域：rec-sys

## 问题定义

当前生成式推荐主要基于自回归语言模型（Autoregressive LM），存在以下局限：
1. **单向依赖**：自回归生成只能从左到右，无法建模 Semantic ID token 之间的双向依赖关系（物品的不同粒度特征是相互约束的，非单向序列关系）
2. **生成多样性不足**：贪心/束搜索倾向于高频物品，长尾探索性差
3. **训练-推理不一致**：训练用 teacher forcing，推理时误差累积（exposure bias）

DiffGRM 将扩散模型（Diffusion Model）引入生成式推荐，通过去噪过程生成物品的 Semantic ID，克服上述自回归方法的局限。

## 核心方法与创新点

1. **离散扩散用于 Semantic ID 生成**：
   - Semantic ID 是离散 token 序列，使用**离散扩散过程**（Discrete Diffusion，如 MDLM/D3PM）
   - 前向过程：对真实物品的 Semantic ID 逐步加 mask 噪声
   - 反向过程：从完全 mask 的状态，条件于用户历史，迭代去噪恢复物品 Semantic ID
   
2. **条件去噪网络**：
   - 用户历史 embedding 作为条件信号，通过 Cross-Attention 注入去噪网络
   - 去噪网络为 Transformer，可同时预测所有 token 位置（非自回归），天然支持双向依赖
   
3. **多步采样换多样性**：
   - 扩散采样本质是随机过程，同一用户历史可采样出不同的物品 Semantic ID
   - 通过控制噪声强度（Temperature）调节探索-利用平衡
   - 实践中 10-20 步去噪即可，无需完整的 1000 步

4. **约束采样**（与 Constrained Decoding 类比）：
   - 每步去噪后，将概率分布约束在合法 Semantic ID 路径上（借助 Trie）
   - 保证最终生成结果对应真实存在的物品

## 实验结论

- 在 Amazon 数据集（Books/Movies/Electronics）上，DiffGRM 相比 TIGER（自回归生成式推荐）：Recall@10 **+6.8%**，NDCG@10 **+5.9%**
- 推荐多样性（Intra-List Diversity）提升 **+22%**，覆盖率（Catalog Coverage）提升 **+18%**
- 在隐式反馈噪声较多的场景下表现更稳健（扩散去噪对噪声有天然鲁棒性）
- 推理延迟：20 步去噪约 80ms（P95），比自回归 beam search 高约 30%，但多样性收益明显

## 工程落地要点

1. **推理步数与质量权衡**：10步 vs 20步性能差距约 1-2%，但延迟差 50%；工业上通常用 10步作为平衡点
2. **并行采样**：扩散模型可以并行采样多个候选（多路召回），用于后续排序模型的多样性 candidates 集合
3. **与自回归融合**：DiffGRM 擅长探索和多样性，自回归擅长精确热门推荐；多路召回融合效果优于单一方法
4. **GPU 利用率**：非自回归推理（全 token 并行预测）比自回归更适合 GPU 批处理，吞吐量更高

## 常见考点

- Q: 扩散模型在推荐系统中相比自回归模型的核心优势是什么？
  A: (1) 双向依赖建模：所有 token 位置同时预测，能捕获 Semantic ID token 间的约束关系；(2) 多样性：随机采样过程天然产生多样化候选；(3) 对噪声鲁棒：去噪训练目标本身对输入噪声有抵抗力。代价是推理需要多步迭代，延迟略高。

- Q: 离散扩散和连续扩散的区别？在推荐场景为什么用离散扩散？
  A: 连续扩散（DDPM）在实值空间（图像像素、embedding 向量）加高斯噪声。离散扩散在离散符号空间（token ID）做 masking/replacing 操作。推荐的 Semantic ID 是离散 token，直接用离散扩散更自然；若用连续扩散需要 embed-denoise-quantize 多步转换，误差更大。

- Q: 如何评估推荐系统的多样性？
  A: 常用指标：(1) Intra-List Diversity（ILD）：推荐列表内物品间的平均距离；(2) Coverage：推荐结果覆盖的物品占总物品库的比例；(3) Serendipity：惊喜度，推荐了用户没预期但喜欢的物品；(4) Novelty：推荐新颖/冷门物品的比例。工业上通常结合准确性指标一起看（准确性-多样性帕累托前沿）。
