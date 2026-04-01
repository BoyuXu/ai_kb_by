# Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning
> 来源：arXiv:2503.06034 | 领域：search | 学习日期：20260330

## 问题定义
LLM 用于文档重排（Document Reranking）时，通常直接输出相关性分数或排列，缺乏对排序决策的推理过程。Rank-R1 受 DeepSeek-R1 启发，将强化学习（RL）引入 LLM 重排器，让模型在排序前生成显式推理链，提升复杂查询的重排质量。

## 核心方法与创新点
1. **Reasoning-before-Ranking**：重排器先生成 `<think>这个文档讨论了...，与查询的关系是...</think>`，再输出相关性判断，显式推理使决策更可靠。
2. **RL 训练（GRPO）**：以 NDCG/MRR 作为 reward，用 Group Relative Policy Optimization 训练推理链质量，无需人工标注推理过程。
3. **Pointwise with Reasoning**：每次对单个文档推理（Pointwise），而非对整个候选集 Listwise，降低上下文长度，支持更多候选。
4. **蒸馏到小模型**：大模型（70B）生成高质量推理链作为教师，蒸馏到 7B 模型，推理延迟可接受。
5. **查询分解**：复杂多意图查询先分解为子查询，每个子查询独立推理后合并，提升长尾复杂查询效果。

## 实验结论
- BEIR benchmark：NDCG@10 平均提升 3.8%（对比 RankGPT baseline）
- 复杂查询（多跳、多意图）提升最显著 +7.2%
- 7B 蒸馏模型 vs 70B 教师模型：NDCG@10 差距 <1%，延迟降低 10x

## 工程落地要点
- 推理链长度控制（128-256 token），过长影响重排吞吐量
- Pointwise 模式支持并行处理多个文档（无文档间依赖），适合工业部署
- RL 训练需要多样化 query-doc 对，BEIR 数据集覆盖不同领域
- 生产环境可用异步重排：快速初排 + 异步 LLM 重排更新结果

## 面试考点
- Q: Pointwise、Pairwise、Listwise 重排的区别？
  - A: Pointwise：独立对每个文档打分（可并行）；Pairwise：文档对比较（O(n²) 候选）；Listwise：对整个排列打分（最难但理论最优）
- Q: 为什么 RL 训练比 SFT 更适合重排任务？
  - A: 重排的 ground truth 是 ranked list，SFT 需要标注推理过程（昂贵）；RL 可以直接以排序指标（NDCG）为 reward，无需推理标注
- Q: LLM 重排的主要延迟来源？
  - A: ① 文档 tokenization（长文档慢）；② LLM forward pass（每次 attention 全量）；③ 推理链生成（额外 decode 步骤）。解法：PrefixCache、批量处理、4-bit 量化

## 数学公式

$$
r_i = \text{NDCG}(\sigma_i) - \text{NDCG}(\bar{\sigma}), \quad \bar{\sigma} = \text{group average}
$$

$$
\mathcal{L}_\text{GRPO} = -\mathbb{E}\left[\frac{\pi_\theta(\sigma)}{\pi_{\theta_\text{old}}(\sigma)} \cdot \hat{r}\right]
$$
