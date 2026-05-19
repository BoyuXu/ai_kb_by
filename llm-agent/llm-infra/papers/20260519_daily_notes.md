# LLM 基础设施论文笔记 — 2026-05-19

> 来源：MelonEgg 每日学习（automated daily-cron）
> 范围：llm-infra 领域 5 篇

---

## 1. GR-LLMs: Recent Advances in Generative Recommendation Based on LLMs

**来源：** https://arxiv.org/abs/2507.06507 （Jul 2025）
**领域：** Generative Recommendation × LLM 综述

**论点：** GR-LLM 已构成一个新范式，与传统判别式推荐显著不同，并有取代后者的潜力

**综述结构：**
- **Preliminaries：** Semantic ID、Tokenization、Generative Retrieval 等基础概念
- **应用案例：** 召回、排序、重排、多目标、对话式推荐的 LLM 化
- **工业落地考量：** Latency、cold-start、长尾、可解释性、合规
- **未来方向：** Multi-modal GR、Agentic GR、Self-evolving GR

**核心 take-away：** GR-LLM 的工业化瓶颈不在效果而在 serving cost — 推理延迟、KV cache 占用、batch 效率均为关键

**面试考点：** Generative vs Discriminative Rec 的边界、Semantic ID 构建（RQ-VAE / TIGER）、LLM 推荐的工业 latency 控制手段

---

## 2. When Text Embedding Meets Large Language Model: A Comprehensive Survey

**来源：** https://arxiv.org/abs/2412.09165 （Dec 2024）
**领域：** Text Embedding × LLM 综述

**三大主题划分：**
1. **LLM-augmented Text Embedding：** 用 LLM 增强传统 embedding 方法（hard negative mining、query/document 改写、伪标注）
2. **LLMs as Text Embedders：** 直接把 LLM 当作 encoder（Mean pooling / EOS token / instruction-aware embedding，如 E5-Mistral、bge-en-icl）
3. **Text Embedding Understanding with LLMs：** 用 LLM 解释 / 分析 embedding（embedding inversion、可解释性探针）

**覆盖任务：**
- 传统：STS（语义相似度）、IR（信息检索）、Text Clustering
- 新生：Long context compression、Embedding inversion attack/defense

**Insight：** LLM-as-embedder 已经在 MTEB 等 benchmark 上反超 BERT 类 encoder，趋势是 instruction-aware 与 task-conditioned embedding

**面试考点：** Bi-encoder 与 Cross-encoder 训练 loss（InfoNCE / Margin / Hard negative）、E5-Mistral 等 LLM-embedder 的 fine-tuning recipe、Embedding inversion 的安全风险

---

## 3. TaiChi: Unifying Aggregation and Disaggregation for LLM Serving

**来源：** https://arxiv.org/abs/2508.01989 （Aug 2025）
**领域：** LLM Serving × P-D 调度

**背景：**
- **PD Aggregation（chunked prefill + decode 同 GPU）：** TTFT 紧、TPOT 宽时最优
- **PD Disaggregation（prefill / decode 分卡）：** TPOT 紧、TTFT 宽时最优
- 中间 balanced SLO 区，两者都次优

**TaiChi 核心：**
- **统一 disaggregation-aggregation 架构：** GPU 实例区分能力 — prefill-heavy（快 prefill、decode 易受扰）、decode-heavy（decode 干扰小、prefill 慢）
- **Latency Shifting：** 把"已达标"请求的 GPU 资源动态调给"濒临违 SLO"的请求
- **两类调度：**
  - Flowing decode scheduling 控 TPOT
  - Length-aware prefill scheduling 控 TTFT

**效果：** balanced SLO 区 goodput 比 SOTA 高 **+77%**

**面试考点：** Goodput vs Throughput 的差异、TTFT/TPOT/TBT 三大 latency 指标、Chunked prefill 与 Disaggregation 的取舍

---

## 4. Disaggregated P-D Inference on Multi-Vendor GPUs

**来源：** https://arxiv.org/abs/2509.17542 （Sep 2025）
**领域：** Heterogeneous GPU × P-D Disaggregation

**问题背景：** 现实集群常是异构（A100 + H100 + 国产卡），不同卡 compute / memory 能力差异大

**架构：**
- **Prefill 实例：** 部署在强 compute GPU（H100 / GH200）
- **Decode 实例：** 部署在强 memory bandwidth GPU（MI250 / A100-80G / 国产 HBM 卡）
- **Heterogeneous-Compatible Transmission Module：** 解决多厂商 GPU 间 KV cache 传输的数据兼容性（精度对齐、layout 转换）

**优化算法：**
- 联合优化 parallel strategy（TP/PP/EP）与各实例数量
- 给定异构资源池 → 输出最优部署方案

**意义：** 让"昂贵 prefill 卡 + 便宜 decode 卡"组合成为可能，降低单 token 成本

**面试考点：** TP/PP/EP 的差异、KV cache 跨卡传输的实现（NCCL / NVLink / RDMA）、异构卡精度对齐（FP16/BF16/FP8）

---

## 5. Nexus: Proactive Intra-GPU P-D Disaggregation

**来源：** https://arxiv.org/abs/2507.06608 （Jul 2025）
**领域：** Single-GPU 内部 P-D Disaggregation × SM Partitioning

**与 #3 / #4 的区别：** Nexus 不跨 GPU 分 P-D，而是在 **单 GPU 内部** 用 SM 分区实现 P-D 隔离

**核心机制：**
- **Proactive SM Partitioning：** 用 contention-aware cost model 预测 per-phase 延迟，提前调整 SM 配额（而不是出现 SLO violation 后才反应）
- **Phase-Aware Scheduling：** 根据当前 batch 的 prefill / decode 比例动态调度
- 求解：双目标优化（min TTFT + min TPOT），greedy search

**实测：**
- vs vLLM：吞吐 +2.2×、TTFT −20×、TBT −2.5×
- vs SGLang：吞吐 +2×、TTFT −2×、TBT −1.7×
- vs vLLM-disaggregation：用一半 GPU 数达到 +1.4× 吞吐

**意义：** 中小集群（资源不足以做物理分卡）也能享受 P-D Disaggregation 的红利

**面试考点：** GPU SM Partitioning 的实现（CUDA Streams + MIG / MPS）、Chunked prefill 的干扰来源、TTFT 与 TBT 的 SLO 设计

---

## 当日小结

- **主线：P-D 分离的全谱演进** — TaiChi（#3）解 SLO 灵活配置、Multi-Vendor（#4）解异构资源、Nexus（#5）解单 GPU 内部，三篇覆盖了 P-D Disaggregation 的"宏观调度 → 异构部署 → 微观分区"完整谱系
- **副线：表征 + 应用** — GR-LLMs（#1）回顾生成式推荐生态、Text Embedding × LLM（#2）回顾 embedding 范式演进；两者共同描绘了 LLM 在"召回 / 推荐 / 检索"侧的全栈影响
