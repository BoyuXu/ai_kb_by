# GR2: Generative Reasoning Re-ranker
> 来源：arXiv:2602.07774 | 领域：rec-sys | 学习日期：20260330

## 问题定义
重排阶段（Re-ranking）需要在已有粗排候选基础上，考虑物品间上下文依赖（context-aware）做最终排序。传统重排模型（SetRank/PRM）用 Transformer 对候选集做 listwise 打分，但缺乏对用户需求的深层推理。GR2 将 LLM 推理能力引入重排，生成推理链后输出最终排序。

## 核心方法与创新点
1. **Generative Reranking Paradigm**：将重排定义为：给定候选集 $\mathcal{C} = \{i_1,...,i_K\}$ 和用户 context，LLM 生成 reasoning + 有序排列。
2. **Chain-of-Thought Reranking**：模型先输出 `<think>用户最近对XX感兴趣，候选中YY与之最相关...</think>`，再输出排序序列。
3. **Permutation 输出**：直接输出物品 ID 的排列（`[3, 1, 4, 2, ...]`），作为最终排序，比逐项打分保留更多 listwise 信息。
4. **训练数据**：用大模型生成推理链（教师模型），小模型蒸馏（学生模型），结合人工标注排序结果 SFT。
5. **RL 精炼**：以 NDCG 作为 reward，GRPO 强化学习进一步提升排序质量。

## 实验结论
- 工业视频推荐：NDCG@10 提升 5.1%，相比 PRM baseline
- 用户满意度（主观评分）提升 8%（推理链使推荐更贴合当前意图）
- 小模型蒸馏后，推理延迟 ~200ms，满足重排阶段 latency 要求

## 工程落地要点
- 重排候选数 K 通常 50-200，LLM 处理 K 个候选的上下文长度需控制
- 推理链长度需限制（128 token），避免超出 context window
- 输出排列的解码需专门 constrained decoding（只允许候选 ID 出现）
- 可与传统重排模型（PRM）并联，A/B 实验逐步替换

## 常见考点
- Q: 重排（Re-ranking）的核心挑战是什么？
  - A: ① 候选间上下文依赖（互补/竞争关系）；② 多样性 vs 相关性权衡；③ 实时性（latency <100ms）
- Q: Listwise 排序和 Pointwise/Pairwise 的区别？
  - A: Pointwise 逐项打分（CTR 预估）；Pairwise 比较物品对（RankNet）；Listwise 对整个列表优化（ListMLE/ApproxNDCG）
- Q: GR2 如何保证输出排列的完整性（不重复/不遗漏候选）？
  - A: Constrained Beam Search：decode 时屏蔽已生成 token，强制从剩余候选中选择
