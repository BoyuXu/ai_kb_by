# Scaling Law + Long Sequence + Multi-Task: Rec-Sys 前沿论文精读 (2025-2026)

> 10 篇论文覆盖三大趋势：推荐系统 Scaling Law、长序列用户建模、多任务学习高效扩展

---

## 1. Scaling Law 与大规模推荐模型

### ULTRA-HSTU: Bending the Scaling Law Curve in Large-Scale Recommendation Systems (arxiv 2602.16986)
- **Problem**: 大规模推荐系统中，如何在模型质量和效率之间实现 scaling law 的突破？标准 self-attention 在超长序列上计算代价过高。
- **Method**: 基于 HSTU 的下一代模型，借鉴 DeepSeek-V2 进行端到端 model-system co-design，在输入序列构造、稀疏注意力机制和模型拓扑三方面创新。部署 18 层 self-attention 处理 16K 用户行为序列，多百张 H100 训练。
- **Key Innovation**: 实现 5x 训练 scaling 加速和 21x 推理 scaling 加速，首次在工业推荐系统中"弯曲" scaling law 曲线。
- **Results**: 服务数十亿用户的生产环境部署，显著提升模型质量与效率。
- **Industry**: Meta (生产环境大规模部署)
- **Keywords**: Scaling Law, Sparse Attention, HSTU, System Co-design, Long Sequence

### LUM: Unlocking Scaling Law in Industrial Recommendation (arxiv 2502.08309)
- **Problem**: 传统 DLRM 无法有效 scale up，生成式建模与工业推荐需求之间存在鸿沟。
- **Method**: 三步范式——(1) Knowledge Construction: Transformer + 生成式学习预训练，捕获用户兴趣与物品协同关系；(2) Knowledge Querying: 预定义问题查询 LUM 获取用户特定信息；(3) Knowledge Utilization: 输出作为补充特征注入传统 DLRM。
- **Key Innovation**: 将生成式预训练引入推荐，使模型可 scale 到 7B 参数并持续获得性能提升，验证了推荐领域的 scaling law。
- **Results**: A/B 测试显著提升，scaling 到 7B 参数仍有收益。
- **Industry**: 工业部署验证 (具体公司未公开)
- **Keywords**: Scaling Law, Generative Pre-training, Large User Model, Three-step Paradigm

---

## 2. 长序列用户行为建模

### SparseCTR: Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR (arxiv 2601.17836) — WWW 2026
- **Problem**: 标准 self-attention 计算复杂度限制了长序列建模，而 NLP/CV 领域的稀疏注意力方案不适配推荐场景的时序特性。
- **Method**: 由 SparseBlock 堆叠构成，核心是 EvoAttention (Evolutionary Sparse Self-attention)：先用 TimeChunking 按时间间隔分段，再设计全局/过渡/局部三分支注意力分别建模长期兴趣、兴趣转移和短期兴趣。引入 RelTemporal 相对时间编码。
- **Key Innovation**: 针对推荐场景设计的稀疏注意力，展现明显 scaling law 现象（跨三个数量级 FLOPs 持续提升）。
- **Results**: 在线 A/B 测试 CTR +1.72%, CPM +1.41%。
- **Industry**: 工业部署 (具体平台未明确)
- **Keywords**: Sparse Attention, Long-term Behavior, TimeChunking, Scaling Law, CTR

### HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR (arxiv 2601.12681)
- **Problem**: 现有架构将长序列建模（LONGER）和特征交互（RankMixer）解耦为流水线，限制了表达能力和交互灵活性。
- **Method**: 混合 Transformer 架构，交替执行 Query Decoding（将非序列特征扩展为 Global Tokens，对行为序列层级 KV 做解码）和 Query Boosting（高效 token mixing 增强跨 query/跨序列异构交互）。
- **Key Innovation**: 首次将长序列建模与特征交互统一到单一 backbone，在相同参数/FLOPs 预算下优于 LONGER+RankMixer 基线，且展现更优 scaling 行为。
- **Results**: 十亿级工业数据集一致优于基线，在 ByteDance 全量部署服务数十亿用户。
- **Industry**: ByteDance (全量部署)
- **Keywords**: Hybrid Transformer, Sequence-Feature Unification, Query Decoding, Scaling

### TransAct V2: Lifelong User Action Sequence Modeling on Pinterest (arxiv 2506.02267)
- **Problem**: 工业 CTR 模型通常只用短序列（~100 actions），无法捕获用户长期行为模式；且缺少高效服务超长序列的基础设施方案。
- **Method**: 三大创新：(1) 利用超长用户序列 O(10^4) actions（V1 的 160x）；(2) 引入 Next Action Loss (NAL) 增强动作预测；(3) 可扩展低延迟部署方案。基于 point-wise MTL 架构。
- **Key Innovation**: 从 100 actions 扩展到 16K+ actions 的 lifelong 建模，同时解决了推理延迟问题。
- **Results**: Pinterest Homefeed 排序系统显著提升。
- **Industry**: Pinterest (生产部署)
- **Keywords**: Lifelong Sequence, Long-term User Modeling, Next Action Loss, Low-latency Serving

### DAIAN: Deep Adaptive Intent-Aware Network for Trigger-Induced Rec (arxiv 2602.13971)
- **Problem**: Trigger-Induced Recommendation (TIR) 中存在"intent myopia"问题——过度强调 trigger item 而忽略用户多样化偏好；基于稀疏 ID 的协同模式限制了泛化。
- **Method**: 三个核心模块：User Intent Modeling (UIM) 分析多相似度层级点击概率建模意图分布、Diverse Intent Extraction (DIE) 从子序列提取显式意图并挖掘潜在多样意图、Similarity-Enhanced Intent Network (SEIN)。引入三阶段训练策略解决融合不收敛问题。
- **Key Innovation**: 将用户意图建模为 preference distribution，通过多层次相似度分析动态适应用户意图偏好。
- **Results**: 离线和在线工业电商平台均优于 SOTA。
- **Industry**: 电商平台 (在线部署)
- **Keywords**: Trigger-Induced Recommendation, Intent Modeling, Multi-granularity, Three-stage Training

---

## 3. 统一架构与多任务学习

### OneTrans: Unified Feature Interaction and Sequence Modeling (arxiv 2510.26104)
- **Problem**: 特征交互模块和序列建模模块分别扩展，阻碍双向信息交换，无法统一优化和 scaling。
- **Method**: 统一 Tokenizer 将序列和非序列属性转为单一 token 序列；OneTrans blocks 对相似序列 token 共享参数，对非序列 token 分配 token-specific 参数；causal attention + cross-request KV caching 实现预计算与缓存。
- **Key Innovation**: 单一 Transformer 同时完成特征交互和序列建模，通过 KV caching 大幅降低训练和推理成本。
- **Results**: 在线 A/B 测试 per-user GMV +5.68%，随参数增长持续 scaling。
- **Industry**: ByteDance (生产部署)
- **Keywords**: Unified Transformer, Feature Interaction, KV Caching, Sequence Modeling

### SRP4CTR: Sequential Recommendation Pre-training for CTR (arxiv 2407.19658)
- **Problem**: 预训练模型引入推荐时忽视额外推理开销，且未充分考虑如何将预训练信息有效迁移到特定候选物品的 CTR 预测。
- **Method**: 两阶段框架——预训练阶段用 Fine-Grained BERT (FG-BERT) 进行多属性 masking 预测，编码全部 side information；微调阶段用 Uni Cross-Attention 建立候选物品到预训练模型的单向注意力连接，配合 folded inference 减少开销。
- **Key Innovation**: Uni Cross-Attention 机制在预训练与 CTR 之间建立低成本桥梁，仅增加少量推理开销即可获得预训练收益。
- **Results**: 在多种预训练模型上均能提升 CTR 任务表现，保持低推理开销。
- **Industry**: 未明确具体部署
- **Keywords**: Pre-training, Sequential Recommendation, Cross-Attention, Fine-Grained BERT, Low Inference Cost

### SMES: Scalable Multi-Task Recommendation via Expert Sparsity (arxiv 2602.09386) — Kuaishou
- **Problem**: 将 Sparse MoE 应用于多任务推荐面临两大障碍：expert activation 爆炸破坏 instance-level sparsity、独立 task-wise routing 导致 expert load skew。
- **Method**: 将 expert activation 分解为 task-shared expert subset (跨任务联合选择) 和 task-adaptive private experts，显式约束每 instance expert 执行数量；引入 global multi-gate load-balancing regularizer 稳定训练。
- **Key Innovation**: Progressive expert routing + 全局负载均衡，在保持任务特异性的同时实现实例级稀疏，解决了 MoE 在多任务推荐中的两大核心难题。
- **Results**: 在线部署提升 0.31% watch time，服务 4 亿+ DAU。
- **Industry**: Kuaishou (4亿+日活短视频)
- **Keywords**: Sparse MoE, Multi-Task Learning, Expert Routing, Load Balancing, Kuaishou

### GRec: Efficient Multi-Task Learning via Generalist Recommender (arxiv 2504.05318)
- **Problem**: 现有 MTL 实现随任务数增加导致训练和推理性能退化，限制了生产环境中的可扩展性。
- **Method**: 端到端 Generalist Recommender，利用 NLP heads、并行 Transformers 和 wide-and-deep 结构处理多模态输入，通过新颖的 task-sentence level routing 机制扩展多任务能力。
- **Key Innovation**: 借鉴 LLM 的 Sparse MoE 架构，提出 task-sentence routing 策略，在跨任务泛化的同时保持推理效率。
- **Results**: 显著优于先前方案，成功部署于大型电信网站/App，处理大量在线流量。
- **Industry**: 大型电信平台 (生产部署)
- **Keywords**: Generalist Recommender, Sparse MoE, Task-Sentence Routing, Multi-Task, Multi-Modal

---

## 综合趋势分析

这 10 篇论文揭示了 2025-2026 年推荐系统研究的三大核心趋势：

**1. Scaling Law 成为推荐系统的新范式。** ULTRA-HSTU、LUM、SparseCTR 等工作明确验证了推荐模型也存在 scaling law，且通过 system co-design（ULTRA-HSTU 的 21x 推理加速）和生成式预训练（LUM 的 7B 参数）可以有效释放规模效应。这标志着推荐系统从"feature engineering + shallow model"向"大模型 scaling"的范式转移。

**2. 长序列建模从 O(100) 跃迁到 O(10^4)。** TransAct V2（16K actions）、ULTRA-HSTU（16K sequences）、SparseCTR 等工作表明，工业界已将用户行为建模从短期（100 actions）扩展到 lifelong（10^4+ actions）。核心技术路线是稀疏注意力（SparseCTR 的三分支 EvoAttention）和高效推理（KV caching、预计算）。

**3. 统一架构取代模块化流水线。** HyFormer、OneTrans 将特征交互与序列建模统一进单一 Transformer backbone，GRec/SMES 将多任务学习与 Sparse MoE 结合实现可扩展的 generalist 模型。"一个模型做所有事"的 unified architecture 正在取代传统的"序列模块 + 特征交互模块 + 多任务头"的解耦架构。

这些趋势的交汇点在于：**以 Transformer 为基础骨架，通过稀疏化（Sparse Attention + Sparse MoE）实现效率可控的大规模扩展，在统一架构内同时处理长序列建模、特征交互和多任务学习。**

---

## 相关概念页链接

- [[concepts/attention_in_recsys]] — Sparse Attention、EvoAttention、Cross-request KV Caching
- [[concepts/sequence_modeling_evolution]] — Lifelong Sequence、HSTU 演进、TransAct 系列
- [[concepts/multi_objective_optimization]] — Multi-Task MoE、Expert Routing、Task-Sentence Routing
- [[concepts/embedding_everywhere]] — Unified Tokenization、Fine-Grained BERT

---

*Created: 2026-05-03 | Papers: 10 | Focus: Scaling + Sequence + Multi-Task*
