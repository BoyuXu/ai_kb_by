# 推荐系统论文笔记 — 2026-04-16

## 1. A Survey of Foundation Model-Powered Recommender Systems: From Feature-Based, Generative to Agentic Paradigms

**来源：** https://arxiv.org/abs/2504.16420
**领域：** 基础模型 × 推荐系统综述
**核心定位：** 首个系统性梳理 FM（GPT/LLaMA/CLIP 等）赋能推荐系统三大范式的综合综述

**三大范式：**
1. **Feature-Based（特征增强）：** FM 作为特征提取器增强用户/物品表征，如用 CLIP 提取多模态特征输入传统推荐模型
2. **Generative（生成式推荐）：** FM 直接生成推荐结果，包括 prompt-based（提示词驱动）、token-based（Semantic ID 生成）、embedding-based（如 MoRec 用视觉/文本编码器生成 item embedding 输入 DSSM/SASRec）
3. **Agentic（智能体推荐）：** LLM 作为 Agent 核心，配合记忆、工具使用和规划能力，实现交互式、上下文感知的推荐

**覆盖任务：** Top-N 推荐、序列推荐、零/少样本推荐、对话推荐、内容生成推荐

**关键洞察：**
- Feature-Based 成熟稳定但上限有限
- Generative 灵活但训练成本高、幻觉问题待解
- Agentic 最前沿但系统复杂度和延迟是挑战

**面试考点：** FM4RecSys 三范式区别与适用场景、Semantic ID 的生成式检索原理、Agentic RS 的系统设计

---

## 2. Rethinking Recommendation Paradigms: From Pipelines to Agentic Recommender Systems

**来源：** https://arxiv.org/abs/2603.26100 (Alibaba International, Mar 2026)
**领域：** Agentic 推荐系统架构
**核心问题：** 传统多阶段流水线（召回→粗排→精排→重排）本质是静态的，模型改进依赖人工假设和工程试错，难以规模化应对异构数据和多目标业务约束

**核心贡献：**
- 提出 AgenticRS 框架，将推荐系统关键模块重组为 Agent
- **Agent 晋升条件：** 功能闭环、可独立评估、具备可演化的决策空间
- **自演化机制：**
  - RL 风格优化：在定义良好的动作空间中通过强化学习优化
  - LLM 辅助架构搜索：在开放设计空间中生成和选择新架构/训练方案

**与传统流水线对比：**
- 传统：人→假设→实验→上线，反馈回路长
- AgenticRS：Agent 自主探索→评估→迭代，形成闭环自优化

**面试考点：** 推荐流水线的局限性分析、Agent 化改造的条件判断、RL vs LLM 驱动的系统自演化

---

## 3. AgenticTagger: Structured Item Representation for Recommendation with LLM Agents

**来源：** https://arxiv.org/abs/2602.05945 (Google, Feb 2026)
**领域：** LLM Agent × 物品表征
**核心问题：** 开放式 LLM 生成的物品描述基数高、质量不稳定，难以在下游推荐模型中有效使用

**核心方法（两阶段）：**
1. **词表构建（Vocabulary Building）：** 架构师 LLM 生成层级化、低基数的描述词集合；标注器 LLM 并行反馈，通过多 Agent 反思机制迭代优化词表
2. **词表分配（Vocabulary Assignment）：** LLM 将词表内描述词分配给每个物品，生成结构化表征

**下游应用：** 生成式检索（Semantic ID）、基于 term 的检索、排序、critique-based 推荐

**关键创新：** 将非结构化到结构化的转换过程 Agent 化，通过多 Agent 协作保证词表质量

**面试考点：** 物品表征的结构化 vs 非结构化权衡、多 Agent 反思机制、Semantic ID 的构建方法

---

## 4. RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation

**来源：** https://arxiv.org/abs/2603.00638 (Mar 2026)
**领域：** LLM 推荐 × 持续学习
**核心问题：** 用户偏好随时间漂移，全局微调影响无关行为，逐点编辑无法捕捉全局偏好迁移，重复编辑导致灾难性遗忘

**核心方法 RAIE（插件式框架）：**
1. **知识区域构建：** 在表征空间中用球面 k-means 聚类构建语义一致的偏好区域
2. **区域感知编辑：** 三种局部编辑操作
   - **Update：** 更新已有区域的 LoRA 参数
   - **Expand：** 扩展区域边界
   - **Add：** 新增区域
3. **区域感知路由：** 置信度感知门控，将输入序列路由到正确区域

**核心公式：**
```
每个区域 R_k 配备独立 LoRA 模块：W = W_0 + B_k A_k
路由函数：g(x) = argmax_k confidence(x, R_k)
```

**关键优势：** 冻结主干模型，仅更新受影响区域的 LoRA，有效缓解遗忘

**面试考点：** 增量学习三大挑战（遗忘/干扰/粒度）、LoRA 的区域化应用、球面 k-means 的选择原因

---

## 5. Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models

**来源：** https://arxiv.org/abs/2503.16734 (Mar 2026)
**领域：** MLLM × Agentic 推荐
**核心观点：** MLLM（多模态大语言模型）为推荐系统注入推理能力，能发现复杂的用户-物品关系，生成可解释且语义丰富的推荐

**Agentic RS 能力框架：**
- **交互性：** 多轮对话理解用户意图
- **上下文感知：** 融合多模态信号（文本/图像/视频/音频）
- **主动推荐：** 不等用户请求，主动推送个性化建议
- **工具使用：** 调用搜索、数据库、外部 API 等工具增强推荐

**三范式对比（承接 2504.16420 综述）：**
- Feature-Based → 增强表征
- Generative → 直接生成
- Agentic → 自主决策 + 工具调用 + 记忆

**面试考点：** MLLM 与传统推荐模型的互补性、多模态特征融合策略、Agentic 推荐的延迟控制

---

## 6. GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models

**来源：** https://arxiv.org/abs/2507.06507 (2025, v2 2026)
**领域：** 生成式推荐综述
**核心定位：** 系统梳理 LLM 驱动的生成式推荐最新进展

**分类框架：**
1. **Prompt-Based 召回：** 直接用 prompt 让 LLM 生成推荐列表
2. **Token-Based 召回：** 将物品映射为 Semantic ID token，训练 LLM 生成 ID 序列
3. **Embedding-Based 召回：** LLM 作为编码器生成 item embedding，输入经典推荐架构（DSSM/SASRec）

**关键技术点：**
- Semantic ID 构建：层级聚类 → 离散化 → token 序列
- 训练范式：预训练 + 指令微调 + RLHF 对齐
- 评估挑战：传统指标（NDCG/Recall）vs 生成质量评估

**面试考点：** 三种召回方式的优劣对比、Semantic ID 的编码-解码流程、生成式推荐的在线部署挑战

---

## 7. Generative Recommendation with Semantic IDs: A Practitioner's Handbook

**来源：** https://arxiv.org/abs/2507.22224 (Google, 2025)
**领域：** Semantic ID × 生成式推荐工程实践
**核心定位：** 面向工程落地的 Semantic ID 生成式推荐实操手册

**Semantic ID 构建流程：**
1. 用预训练模态编码器（CLIP/BERT）计算物品语义嵌入
2. 层级聚类（RQ-VAE / 树状 k-means）将嵌入映射为离散 ID 序列
3. 训练 seq2seq 模型（如 T5）从用户行为序列生成 Semantic ID

**工程要点：**
- **码本大小选择：** 平衡表达能力与生成难度
- **层级数设计：** 通常 3-5 层，过深增加解码延迟
- **Beam Search 调优：** beam size 影响召回率和推理成本
- **在线服务：** Semantic ID 解码 + 后续精排 pipeline 集成

**面试考点：** RQ-VAE 原理与 Semantic ID 的关系、树状 vs 平坦 Semantic ID 的选型、生产环境的延迟优化

---

## 8. RecThinker: An Agentic Framework for Tool-Augmented Reasoning in Recommendation

**来源：** https://arxiv.org/abs/2603.09843 (Mar 2026)
**领域：** 工具增强推理 × 推荐
**核心贡献：** 提出 Agent 框架，让推荐 LLM 在推理过程中调用外部工具（搜索引擎、知识图谱、计算器等）增强推荐决策

**工具增强推理流程：**
1. 用户请求 → LLM 分析意图
2. 规划工具调用序列
3. 执行工具获取外部知识
4. 综合推理生成推荐

**面试考点：** 工具选择策略、推理链与推荐质量的关系、Agent 推荐的实时性约束

---

## 9. Internalizing Multi-Agent Reasoning for Accurate LLM-based Recommendation

**来源：** https://arxiv.org/abs/2602.09829 (Feb 2026)
**领域：** 多 Agent 推理 × 推荐
**核心思想：** 将多 Agent 协作推理的能力内化到单个 LLM 中，避免多 Agent 系统的通信开销和延迟问题

**面试考点：** 多 Agent → 单模型蒸馏的方法论、推理质量 vs 推理效率的权衡

---

## 10. AgenticRS-Architecture: System Design for Agentic Recommender Systems

**来源：** https://arxiv.org/abs/2603.26085 (Alibaba, Mar 2026)
**领域：** Agentic 推荐系统架构设计
**核心贡献：** 配合 AgenticRS 论文的系统架构设计方案，详细阐述如何在工业级推荐系统中实现 Agent 化改造

**架构要素：**
- Agent 编排层：管理多个功能 Agent 的协作
- 记忆管理：短期（会话内）+ 长期（跨会话）记忆
- 工具注册表：可插拔的工具接口设计
- 评估框架：Agent 行为的自动化评估

**面试考点：** 工业推荐系统的 Agent 化改造路径、记忆系统设计、Agent 间通信协议

---

**今日 rec-sys 总结：** 10 篇论文聚焦于推荐系统的 Agentic 化转型趋势，从综述（FM4RecSys 三范式）、架构设计（AgenticRS）、核心技术（AgenticTagger/RAIE/Semantic ID）到工程实践（Practitioner's Handbook），形成了完整的知识链条。关键趋势：推荐系统正从静态流水线向自主 Agent 系统演进。
