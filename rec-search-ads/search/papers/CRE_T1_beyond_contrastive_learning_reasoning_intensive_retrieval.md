# CRE-T1 Preview Technical Report: Beyond Contrastive Learning for Reasoning-Intensive Retrieval
> 来源：arXiv:2603.17387 | 领域：search | 学习日期：20260330

## 问题定义
传统 embedding 模型训练主要依赖对比学习（Contrastive Learning，InfoNCE loss），但对比学习存在局限：(1) 依赖大量高质量正负样本对；(2) 对推理密集型检索效果有限（浅层语义匹配）。CRE-T1 探索超越对比学习的检索模型训练范式，重点解决推理密集型检索。

## 核心方法与创新点
1. **Generative Retrieval Training（GRT）**：将检索任务建模为生成任务——给定 query，直接生成 relevant document 的 semantic ID（而非做 embedding 相似度匹配）。模型学习的是"生成正确文档"而非"区分正负样本"。
2. **Reasoning Chain as Bridge**：在 query 和 document 之间显式生成推理链（intermediate reasoning steps），使模型理解 query→reasoning→document 的逻辑路径。
3. **RL from Retrieval Outcomes（RLRO）**：用检索成功率（top-k recall）作为 reward，GRPO 训练推理-检索策略，超越传统监督学习上限。
4. **Think-then-Retrieve**：检索前先生成推理（think token），推理内容作为 query 增强（query expansion），再做 embedding 检索，而非直接用原始 query。
5. **混合训练目标**：对比损失（基础语义对齐）+ 生成损失（推理-文档联合）+ RL 损失（检索成功率），三者联合优化。

## 实验结论
- BEIR 零样本检索：NDCG@10 67.8%，新 SOTA（超越 Qwen3-Embedding-8B +2.6%）
- 数学推理检索（MATH-RAG）：Recall@5 +18.2%（推理引导带来巨大提升）
- 法律推理检索（LegalBench-RAG）：Recall@5 +12.7%

## 工程落地要点
- Think token 生成增加约 50-150ms latency，在精排（重排）阶段使用，不用于一次召回
- RL 训练需要 retrieval simulator（高效计算 recall reward），工程实现复杂
- Generative Retrieval 需要预先建立 document semantic ID 索引，新文档入库需实时分配 ID
- 适合专业知识库检索（法律/医疗/学术），通用网页搜索提升有限

## 面试考点
- Q: 为什么对比学习在推理检索上有局限？
  - A: 对比学习优化 embedding 空间相似度，但推理相关性 ≠ 语义相似性。数学题和其解题文档的 embedding 可能很不相似，但推理上高度相关
- Q: "Think-then-Retrieve" vs 标准 RAG 的区别？
  - A: 标准 RAG：直接用 query 检索。Think-then-Retrieve：先推理（理解 query 意图/分解问题），用推理产物做增强 query 检索，更接近人类信息搜索过程
- Q: RL 如何应用于检索模型训练？
  - A: 将 top-k recall 作为 reward signal，用 policy gradient 更新检索模型参数。挑战：reward 不可微（离散检索结果），需要 REINFORCE/GRPO 等无梯度方法

## 数学公式
$$\mathcal{L} = \mathcal{L}_\text{contrast} + \alpha \mathcal{L}_\text{generate} + \beta \mathcal{L}_\text{RL}$$

$$\mathcal{L}_\text{RL} = -\mathbb{E}_\pi[\text{Recall@k}(\text{retrieved docs})]$$
