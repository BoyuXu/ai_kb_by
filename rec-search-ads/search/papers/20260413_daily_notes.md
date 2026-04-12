# 搜索系统论文/资源笔记 — 2026-04-13

## 1. OpenMatch: 信息检索开源工具包

**来源：** https://github.com/OpenMatch/OpenMatch
**领域：** 神经信息检索（Neu-IR）
**核心定位：** 基于 PyTorch 的全功能信息检索研究工具包

**关键能力：**
- 20+ 稀疏检索方法
- 稠密检索器：DPR、ANCE（基于预训练语言模型的语义匹配）
- 神经 IR 模型：K-NRM、Conv-KNRM、CEDR、BERT
- OpenMatch-v2：跨模态模型支持 + 增强域适应技术 + 优化基础设施

**面试考点：** 稀疏检索 vs 稠密检索、双塔模型 vs 交互模型、BERT 在 IR 中的应用

---

## 2. Rank-R1: 基于强化学习的 LLM 文档重排序

**来源：** arXiv:2503.06034 (Zhuang et al., 2025)
**领域：** LLM + 信息检索排序
**核心问题：** LLM 作为重排序器时的推理能力提升

**核心方法：**
- 使用 Group Relative Policy Optimization (GRPO) 训练 LLM 重排序器
- 仅用少量相关性标注 + 无推理监督的 RL 算法增强推理能力
- 在排序前对 query 和候选文档进行推理

**关键结果：**
- 在 TREC DL 和 BRIGHT 数据集上高度有效，尤其适合复杂查询
- 仅用 18% 训练数据即可达到与监督微调方法同等效果

**面试考点：** GRPO 算法原理、RL 在 IR 中的应用、LLM 重排序 vs 传统重排序

---

## 3. ReasonIR: 面向推理任务的检索器训练

**来源：** arXiv:2504.20595 (Meta AI, 2025)
**领域：** 推理密集型信息检索
**核心问题：** 现有检索器在推理任务上收益有限，因训练数据仅关注短事实性查询

**核心贡献：**
- 合成数据生成管道：为每个文档生成需要推理才能匹配的挑战性查询 + 看似相关但无用的困难负例
- ReasonIR-8B：首个专门为通用推理任务训练的检索器
- 在 BRIGHT 基准上达 SOTA：无重排 29.9 nDCG@10，有重排 36.9 nDCG@10
- RAG 场景：MMLU 提升 6.4%，GPQA 提升 22.6%

**面试考点：** 推理型检索 vs 事实型检索、合成数据在 IR 中的应用、test-time compute 优化

---

## 4. From Validity to Inter-Subjectivity: Reliability Signals in Search

**来源：** arXiv:2604.01186 (2026)
**领域：** 搜索系统认知论
**核心观点：** 以有效性为中心的框架不足以应对搜索环境的认知挑战，提出可靠性信号作为替代

---

## 5. AgentSearch: 开源代理搜索框架

**来源：** https://github.com/SciPhi-AI/agent-search
**领域：** 代理搜索 / RAG
**核心定位：** 为搜索代理提供动力并实现可定制本地搜索的框架

**关键特性：**
- 搜索代理集成：连接搜索专用 LLM（如 Sensei-7B）与搜索引擎
- 可定制本地搜索：使用 AgentSearch 数据集部署本地搜索引擎
- 多 API 集成：Bing、SERP API 等
- RAG 能力：搜索结果摘要、查询生成、下游详细检索

**面试考点：** Agentic Search 架构、搜索代理 vs 传统搜索的差异
