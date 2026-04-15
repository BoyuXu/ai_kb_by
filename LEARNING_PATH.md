# 搜广推从零到精通学习路径

> Karpathy 原则：从简单到复杂，每步都可验证，深度优先不要广度优先。
> 每个节点标注：前置知识 | 核心论文 | 对应 synthesis | 预计学习时间

---

## 路径一：推荐系统 🎯

```
Level 0: 基础概念
    │
    ├─ [1] 协同过滤 & MF ────────────── 2h
    │   前置：线性代数基础
    │   核心：Matrix Factorization (Koren 2009)
    │   验证：手推 SVD 分解，理解 user-item 矩阵
    │
    ├─ [2] LR → FM → DeepFM ─────────── 3h ⭐ 必学
    │   前置：逻辑回归、梯度下降
    │   核心：FM (Rendle 2010), DeepFM (Guo 2017)
    │   📄 fundamentals/ctr_calibration.md
    │   📄 rec-sys/02_rank/synthesis/CTR模型深度解析.md
    │   验证：能画出 DeepFM 网络结构，解释 FM 二阶交叉
    │
    ├─ [3] Wide & Deep → DCN-V2 ──────── 2h
    │   前置：[2] DeepFM
    │   核心：Wide&Deep (Google 2016), DCN-V2 (Google 2021)
    │   验证：解释 Wide 记忆 vs Deep 泛化的区别
    │
Level 1: 序列建模（推荐的核心差异化）
    │
    ├─ [4] DIN: Target Attention ──────── 3h ⭐ 必学
    │   前置：[2] + Attention 基础
    │   核心：DIN (Zhou 2018)
    │   📄 fundamentals/din_sequence_modeling.md
    │   📄 concepts/attention_in_recsys.md
    │   验证：手推 Target Attention 公式，解释为什么比 Avg Pooling 好
    │
    ├─ [5] DIEN: 兴趣演化 ────────────── 3h
    │   前置：[4] + GRU 基础
    │   核心：DIEN (Zhou 2019)
    │   📄 rec-sys/02_rank/synthesis/用户行为序列建模.md
    │   验证：画出 AUGRU 结构，解释辅助损失的作用
    │
    ├─ [6] SIM → ETA → SparseCTR: 长序列 ── 4h
    │   前置：[5]
    │   核心：SIM (Pi 2020), ETA (Chen 2021), SparseCTR (WWW 2026)
    │   📄 rec-sys/long-sequence/synthesis/长序列用户行为建模技术演进.md
    │   📄 concepts/sequence_modeling_evolution.md
    │   📄 concepts/attention_in_recsys.md §4-B
    │   验证：解释 SIM 两阶段 vs SparseCTR 三分支稀疏注意力的区别
    │
Level 2: 多任务 & 全链路
    │
    ├─ [7] MMoE → PLE ────────────────── 3h ⭐ 必学
    │   前置：[2] 基础 CTR 模型
    │   核心：MMoE (Google 2018), PLE (腾讯 2020)
    │   📄 fundamentals/mmoe_multitask.md
    │   📄 rec-sys/04_multi-task/synthesis/推荐广告系统多任务学习与MoE专家混合.md
    │   📄 concepts/multi_objective_optimization.md
    │   验证：画 MMoE vs PLE 结构图，解释跷跷板效应
    │
    ├─ [8] 召回系统 ──────────────────── 4h
    │   前置：[2] Embedding 基础
    │   核心：DSSM, YouTube DNN
    │   📄 rec-sys/01_recall/synthesis/推荐系统召回范式演进.md
    │   📄 rec-sys/01_recall/synthesis/召回系统工业界最佳实践.md
    │   📄 concepts/embedding_everywhere.md
    │   验证：解释双塔负样本策略，ANN 索引选型
    │
    ├─ [9] 重排与多样性 ──────────────── 2h
    │   前置：[8]
    │   📄 rec-sys/03_rerank/synthesis/推荐系统重排与多样性.md
    │   验证：手推 MMR 公式，解释 DPP
    │
    ├─ [10] 全链路串联 ─────────────────── 2h
    │   前置：[7][8][9]
    │   📄 rec-sys/02_rank/synthesis/推荐系统全链路架构概览.md
    │   验证：能画出完整推荐链路，说清每层输入输出和量级
    │
    ├─ [10.5] 粗排蒸馏 ─────────────────── 3h ⭐ 面试高频
    │   前置：[10] 全链路 + 蒸馏基础
    │   核心：COLD (阿里 2020), Rocket Launching, RankDistil
    │   📄 rec-sys/02_rank/synthesis/粗排蒸馏模型专题.md
    │   验证：说清 6 种蒸馏方法差异，解释为什么排序蒸馏比 logit 蒸馏更适合粗排
    │
Level 3: 前沿（面试加分项）
    │
    ├─ [11] 生成式推荐 ─────────────────── 4h
    │   前置：[8] + Transformer 基础
    │   核心：TIGER, HSTU, UniGRec, Mender (TMLR 2025)
    │   📄 concepts/generative_recsys.md
    │   📄 rec-sys/01_recall/synthesis/生成式推荐系统技术全景_2026.md
    │   📄 concepts/vector_quantization_methods.md
    │   验证：解释 Semantic ID 的 RQ-VAE，对比判别式 vs 生成式
    │   进阶：四种量化方法选型（RQ-VAE/FSQ/LFQ），Mender 的 Preference Discerning
    │
    ├─ [12] Scaling Law & LLM×推荐 ────── 3h
    │   前置：[11] + LLM 基础
    │   📄 rec-sys/02_rank/synthesis/推荐系统ScalingLaw_Wukong.md
    │   📄 rec-sys/synthesis/2026-04-09_llm_for_recsys_landscape.md
    │
    └─ [13] 因果推断 & AB测试 ──────────── 3h
        前置：统计学基础
        📄 rec-sys/04_multi-task/synthesis/推荐系统因果推断.md
        📄 rec-sys/04_multi-task/synthesis/推荐广告AB测试与在线实验.md
```

**推荐系统总学习时间：~38h（核心路径 [1-10] ~26h）**

---

## 路径二：广告系统 📊

```
Level 0: CTR 基础（与推荐共享）
    │
    ├─ [A1] 广告系统架构 & RTB ────────── 2h ⭐ 必学
    │   前置：推荐 [2]
    │   📄 ads/04_bidding/synthesis/广告系统RTB架构全景.md
    │   验证：画出 DSP-ADX-SSP 架构，说清一次竞价流程
    │
    ├─ [A2] 广告 CTR/CVR ──────────────── 3h
    │   前置：[A1] + 推荐 [2]
    │   📄 ads/02_rank/synthesis/广告CTR_CVR预估与校准.md
    │   📄 ads/02_rank/synthesis/CTR预估模型工业级实践进展.md
    │   验证：解释 CTR 校准为什么对广告至关重要（影响出价）
    │
Level 1: 竞价与出价
    │
    ├─ [A3] ESMM → CVR 偏差治理 ────────── 3h ⭐ 必学
    │   前置：[A2]
    │   📄 ads/02_rank/synthesis/ESMM系列CVR估计演进_从整体空间到因果推断.md
    │   📄 ads/02_rank/synthesis/广告系统偏差治理三部曲.md
    │   验证：画 ESMM 整体空间图，解释 IPW 纠偏
    │
    ├─ [A4] oCPC/oCPA → AutoBidding ───── 4h ⭐ 必学
    │   前置：[A2] + 优化理论基础
    │   📄 ads/04_bidding/synthesis/AutoBidding技术演进_从规则到RL.md
    │   📄 ads/04_bidding/synthesis/广告预算Pacing算法全景.md
    │   验证：推导 KKT 最优出价公式 b*=v/(1+λ)
    │
    ├─ [A4.5] 延迟转化预估 ──────────────── 3h ⭐ 面试高频
    │   前置：[A3] ESMM + [A4] oCPC
    │   核心：DFM (Chapelle 2014), ESDF, DRSA
    │   📄 ads/02_rank/synthesis/延迟转化预估处理方案.md
    │   📄 concepts/multi_objective_optimization.md
    │   验证：推导 DFM 的 EM 算法 w_i 公式，解释生存分析 vs 分类模型选择
    │
    ├─ [A5] 多目标优化 ────────────────── 3h
    │   前置：[A3] + 推荐 [7]
    │   📄 ads/02_rank/synthesis/广告系统多目标优化.md
    │   📄 concepts/multi_objective_optimization.md
    │   验证：对比加权求和 vs Pareto vs MMoE 在广告中的取舍
    │
Level 2: 进阶
    │
    ├─ [A6] LTV 预测 ──────────────────── 2h
    │   前置：[A4]
    │   📄 ads/02_rank/synthesis/LTV预测技术演进与工业实践.md
    │   验证：解释 ZILN 零膨胀建模
    │
    ├─ [A7] 创意生成 ──────────────────── 2h
    │   前置：LLM 基础
    │   📄 ads/05_creative/synthesis/广告创意优化.md
    │
    └─ [A8] Uplift & 因果 ─────────────── 3h
        前置：统计学基础
        📄 ads/uplift/synthesis/Uplift建模技术演进与工业实践.md
        📄 fundamentals/uplift_modeling.md
```

**广告系统总学习时间：~22h（核心路径 [A1-A5] ~15h）**

---

## 路径三：搜索 🔍

```
Level 0: 检索基础
    │
    ├─ [S1] BM25 & 倒排索引 ─────────── 2h ⭐ 必学
    │   前置：信息论基础
    │   📄 fundamentals/bm25_sparse_retrieval.md
    │   验证：手算一个 BM25 分数
    │
    ├─ [S2] 稠密检索 (DPR) ────────────── 3h ⭐ 必学
    │   前置：[S1] + Embedding 基础
    │   📄 fundamentals/embedding_ann.md
    │   📄 search/01_recall/synthesis/检索三角_Dense_Sparse_LateInteraction.md
    │   验证：对比 BM25 vs DPR 优劣，说出各自失败场景
    │
    ├─ [S3] ColBERT & 混合检索 ──────── 3h
    │   前置：[S2]
    │   📄 search/01_recall/synthesis/混合检索的工业化演进.md
    │   📄 concepts/embedding_everywhere.md
    │   验证：解释延迟交互 vs 双编码器的 trade-off
    │
Level 1: 排序与重排
    │
    ├─ [S4] 多阶段排序 ────────────────── 3h
    │   前置：[S3]
    │   📄 search/03_rerank/synthesis/搜索Reranker演进.md
    │   📄 search/03_rerank/synthesis/LearningToRank搜索排序三大范式.md
    │   验证：解释 pointwise/pairwise/listwise 三范式
    │
    ├─ [S5] Query 理解 ────────────────── 2h
    │   前置：NLP 基础
    │   📄 search/04_query/synthesis/搜索Query理解.md
    │   验证：画出 query 理解全链路（分词→意图→改写→扩展）
    │
Level 2: LLM 增强搜索（前沿）
    │
    ├─ [S6] LLM 增强重排 ──────────────── 3h
    │   前置：[S4] + LLM 基础
    │   📄 search/03_rerank/synthesis/LLM增强信息检索与推理重排序综合总结.md
    │   📄 concepts/attention_in_recsys.md
    │   验证：对比 DEAR 蒸馏 vs 直接 LLM 重排的延迟/精度
    │
    ├─ [S7] RAG 系统 ──────────────────── 4h ⭐ 必学
    │   前置：[S3] + LLM 基础
    │   📄 llm-agent/llm-infra/synthesis/RAG系统全景.md
    │   📄 search/synthesis/2026-04-09_rag_systems_evolution.md
    │   验证：搭一个 naive RAG，然后逐步优化
    │
    └─ [S8] 生成式检索 ────────────────── 3h
        前置：[S7]
        📄 search/synthesis/2026-04-09_generative_retrieval_evolution.md
        📄 concepts/generative_recsys.md
```

**搜索总学习时间：~23h（核心路径 [S1-S5,S7] ~17h）**

---

## 路径四：LLM 基础设施 🧠

```
Level 0: 基础
    │
    ├─ [L1] Transformer & Attention ──── 3h ⭐ 必学
    │   前置：深度学习基础
    │   📄 fundamentals/attention_transformer.md
    │   📄 concepts/attention_in_recsys.md
    │   验证：手推 self-attention 公式，解释 Q/K/V
    │
    ├─ [L2] KV Cache & 推理优化 ───────── 3h ⭐ 必学
    │   前置：[L1]
    │   📄 fundamentals/kv_cache_inference.md
    │   📄 llm-infra/synthesis/KVCache与LLM推理优化全景.md
    │   验证：解释为什么解码阶段是 memory-bound
    │
    ├─ [L3] LoRA & PEFT ──────────────── 2h
    │   前置：[L1]
    │   📄 fundamentals/lora_peft.md
    │   📄 llm-infra/synthesis/LoRA与PEFT高效微调技术进展.md
    │   验证：手推 LoRA 低秩分解，解释为什么能省 99% 参数
    │
Level 1: 系统优化
    │
    ├─ [L4] FlashAttention ────────────── 2h
    │   前置：[L2] + GPU 内存层次基础
    │   📄 llm-infra/synthesis/FlashAttention3与LLM推理基础设施.md
    │   验证：解释 Tiling + Online Softmax 如何避免 HBM 瓶颈
    │
    ├─ [L5] 量化 (GPTQ/AWQ/NF4) ──────── 3h
    │   前置：[L2]
    │   📄 llm-infra/synthesis/LLM_quantization_evolution_20260408.md
    │   验证：对比 INT8/INT4/NF4 精度-速度 trade-off
    │
    ├─ [L5.5] 知识蒸馏 10 大模式 ──────── 4h ⭐ 必学
    │   前置：[L5] + 推荐排序基础
    │   📄 llm-infra/synthesis/知识蒸馏技术整体总结.md
    │   覆盖：经典KD/自蒸馏/多教师/在线蒸馏/渐进式/黑盒/推理链/投机解码/蒸馏+对齐/无数据
    │   验证：解释 on-policy vs off-policy 蒸馏信息密度差异，画出 Qwen3 两阶段蒸馏
    │
    ├─ [L6] MoE 架构 ─────────────────── 3h
    │   前置：[L1] + 推荐 [7] MMoE
    │   📄 llm-infra/synthesis/MoE架构设计与推理优化.md
    │   验证：解释 expert 路由、负载均衡损失
    │
Level 2: 对齐与 Agent
    │
    ├─ [L7] RLHF → DPO → GRPO ────────── 4h ⭐ 必学
    │   前置：[L3] + RL 基础
    │   📄 fundamentals/rlhf_reward_modeling.md
    │   📄 llm-infra/synthesis/LLM对齐方法演进.md
    │   📄 llm-infra/synthesis/GRPO大模型推理RL算法.md
    │   验证：推导 DPO 损失，解释为什么不需要 Reward Model
    │
    ├─ [L8] RAG 系统 ──────────────────── 3h
    │   前置：[L3] + 搜索 [S3]
    │   📄 llm-infra/synthesis/RAG系统全景.md
    │   📄 llm-infra/synthesis/RAG_vs_Finetune决策框架.md
    │
    └─ [L9] Agent 架构 ────────────────── 3h
        前置：[L8]
        📄 llm-infra/synthesis/agent-frameworks-landscape-2025-2026.md
        📄 llm-infra/synthesis/agentic_AI_systems_20260408.md
```

**LLM 基础设施总学习时间：~26h（核心路径 [L1-L3,L7] ~12h）**

---

## 面试冲刺路径 🚀（7天计划）

| 天数 | 主题 | 学习内容 | 时间 |
|------|------|---------|------|
| Day 1 | CTR 全链路 | [2][3][4] + CORE_TECH_DIGEST §1 | 8h |
| Day 2 | 序列 & 多任务 | [5][6][7] + concepts/attention + sequence | 7h |
| Day 3 | 召回 & 全链路 | [8][9][10] + concepts/embedding | 8h |
| Day 4 | 广告核心 | [A1][A2][A3][A4] | 8h |
| Day 5 | 搜索核心 | [S1][S2][S3][S4][S7] | 8h |
| Day 6 | LLM & 前沿 | [L1][L2][L7] + [11] 生成式推荐 | 8h |
| Day 7 | 面试模拟 | interview/ 题库 + CORE_TECH_DIGEST 速览 | 8h |

---

## 跨领域概念导航

学到某个节点时，跳转到横切概念页加深理解：

| 当你学到... | 去读这个概念页 |
|------------|--------------|
| DIN/DIEN/SIM | → [concepts/attention_in_recsys.md](concepts/attention_in_recsys.md) |
| 双塔/DPR/ColBERT | → [concepts/embedding_everywhere.md](concepts/embedding_everywhere.md) |
| MMoE/PLE/ESMM | → [concepts/multi_objective_optimization.md](concepts/multi_objective_optimization.md) |
| GRU/Transformer/Mamba | → [concepts/sequence_modeling_evolution.md](concepts/sequence_modeling_evolution.md) |
| TIGER/HSTU/生成式检索 | → [concepts/generative_recsys.md](concepts/generative_recsys.md) |
| RQ-VAE/FSQ/LFQ/量化 | → [concepts/vector_quantization_methods.md](concepts/vector_quantization_methods.md) |
