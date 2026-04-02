# RGAlign-Rec: Ranking-Guided Alignment for Latent Query Reasoning in Recommendation Systems

> 来源：https://arxiv.org/abs/2602.12968 | 领域：rec-sys | 学习日期：20260401

## 问题定义

**场景**：现代电商聊天机器人（Chatbot）中的**主动意图预测（Proactive Intent Prediction）**能力。与被动推荐不同，主动推荐在用户未发出显式查询的情况下（"zero-query"），通过感知用户行为和上下文信号来预判需求，实现个性化推荐。

**两个核心挑战：**

1. **语义鸿沟（Semantic Gap）**：用户特征是离散的行为数据（点击、购买、浏览历史），而 chatbot 知识库（Knowledge Base）中的意图是自然语言语义。如何跨越行为特征空间与语义意图空间之间的表示鸿沟？

2. **目标错位（Objective Misalignment）**：通用 LLM 的预训练目标（语言建模）和推荐系统的排序目标（最大化 CTR/CVR）存在根本性差异。LLM 输出的意图查询可能语义完整但排序效果差。

**工业背景**：来自 Shopee（东南亚头部电商）的大规模工业数据集，挑战来自真实生产环境。

## 核心方法与创新点

本文提出 **RGAlign-Rec**，一个将 LLM 语义推理与排序模型紧密耦合的**闭环对齐框架（Closed-Loop Alignment Framework）**。

### 1. 系统架构：两模块闭环

```
用户行为特征（离散）
        ↓
[LLM 语义推理器]  ← Ranking-Guided Alignment ←┐
        ↓                                      │
   意图查询（Query）                            │
        ↓                                      │
[QE 排序模型（Query-Enhanced Ranker）] ──────────┘
        ↓
   排序结果 → A/B 在线测试
```

### 2. Query-Enhanced（QE）排序模型

首先设计了一个**查询增强排序模型（QE-Rec）**，将 LLM 生成的语义查询作为额外特征注入传统排序模型：

$$
\text{Score}(u, item) = f(\mathbf{e}_u, \mathbf{e}_{item}, \mathbf{e}_q)
$$

其中 $\mathbf{e}_q$ 是 LLM 生成查询的文本 embedding，通过特征拼接与用户 embedding、物品 embedding 融合。

### 3. 排序引导对齐（Ranking-Guided Alignment, RGA）

**核心创新**：利用下游排序信号作为反馈来精调 LLM 的潜在推理，实现端到端对齐。

**多阶段训练范式：**

**Stage 1：预训练 LLM 推理器**
- 用用户行为序列和知识库意图对训练 LLM
- 损失：语义相似度损失（Contrastive Learning）
- 目的：建立行为→意图的基本映射能力

**Stage 2：训练 QE-Rec**
- 固定 LLM，用生成的查询训练排序模型
- 损失：LTR（Learning-to-Rank）损失

**Stage 3：RGA 对齐**
- 用排序模型的排名信号反馈优化 LLM：

$$
\mathcal{L}_{RGA} = -\mathbb{E}_{q \sim \pi_{LLM}}\left[R(q) \cdot \log \pi_{LLM}(q|x)\right]
$$

其中 $R(q)$ 是查询 $q$ 在排序模型上的排名奖励（如 NDCG 或 MAP），$\pi_{LLM}$ 是 LLM 策略。类似于 RLHF 的思路，但奖励信号来自排序模型而非人类。

### 4. 实验规模与在线验证

- **工业数据集**：来自 Shopee 的大规模真实数据
- **双阶段在线 A/B 测试**：分别验证 QE-Rec 和 RGA 的独立贡献

## 实验结论

**离线指标（大规模工业数据集）：**

| 方法 | GAUC 提升 | Error Rate | Recall@3 |
|------|-----------|------------|---------|
| Base（无查询增强） | - | - | - |
| QE-Rec（查询增强排序） | - | - | - |
| **RGAlign-Rec（QE+RGA）** | **+0.12%** | **-3.52% (relative)** | **+0.56%** |

**在线 A/B 测试结果：**

| 阶段 | 指标 | 提升 |
|------|------|------|
| QE-Rec（Stage 1） | CTR | +0.98% |
| RGA（Stage 2 额外） | CTR | +0.13% |
| **累计总提升** | **CTR** | **+1.11%** |

**关键结论：**
- GAUC 0.12% 的提升在工业系统中非常显著（通常 >0.1% 即可上线）
- 错误率相对降低 3.52%，反映查询质量的实质提升
- RGA 对齐阶段在 QE-Rec 基础上额外贡献 0.13% CTR，说明排序对齐的价值

## 工程落地要点

### 意图预测系统架构

```
实时用户行为流（Kafka）
        ↓
  特征实时计算服务
        ↓
  LLM 意图推理服务（离线批量+在线实时缓存）
        ↓
  查询 Embedding 存储（向量数据库）
        ↓
  排序服务（集成 QE-Rec）
        ↓
  Chatbot 推荐卡片
```

### 关键工程考量

1. **LLM 推理延迟**：意图推理可异步进行（非实时路径），将结果缓存后供排序服务使用
2. **RGA 训练频率**：排序信号反馈循环，建议每日/每周批量重训 LLM，而非实时更新
3. **知识库（KB）维护**：意图类别需人工维护和定期更新，避免 KV 漂移
4. **奖励设计**：$R(q)$ 的设计对 RGA 至关重要，可用 GAUC/MAP/CTR 等，需根据业务目标选择

### 对比传统方法的优势
| 方面 | 传统规则/检索 | RGAlign-Rec |
|------|-------------|------------|
| 语义理解 | 弱（关键词匹配） | 强（LLM 理解） |
| 排序对齐 | 人工设计特征 | 数据驱动闭环 |
| 冷启动 | 规则覆盖 | LLM 泛化 |
| 维护成本 | 高（规则维护） | 低（数据驱动） |

## 常见考点

**Q1: RGAlign-Rec 的"闭环对齐"与 RLHF 有何相似之处？**
A: 非常相似。RLHF 中，人类偏好作为奖励信号训练 Reward Model，再用 RL 优化 LLM；RGAlign-Rec 中，下游排序模型的排名效果作为奖励信号（Ranking Reward），通过类 Policy Gradient 方法优化 LLM 生成的意图查询。本质都是"用下游任务信号指导上游生成"。

**Q2: 为什么 GAUC 比普通 AUC 更适合推荐系统评估？**
A: GAUC（Group AUC）是在每个用户内部分别计算 AUC 后取平均，消除了不同用户间绝对偏好差异的影响。普通 AUC 会受热门物品和活跃用户主导，而 GAUC 更公平地反映模型对每个用户的个性化排序能力。

**Q3: 意图预测中的"零查询推荐（zero-query recommendation）"有哪些实际挑战？**
A: (1) 用户未表达明确意图，意图推断本身存在不确定性；(2) 行为信号稀疏（session 刚开始时数据少）；(3) 主动推荐的时机选择（何时介入不打扰用户）；(4) 用户接受度——主动推荐可能被视为侵入性。

**Q4: 如何设计 RGA 中的排序奖励函数 R(q)？**
A: 奖励设计有多种选择：(1) 离散奖励：查询生成的候选集中有点击则 R=1，无点击则 R=0；(2) 连续奖励：用排序指标 NDCG@K 作为奖励信号；(3) 在线奖励：直接用 A/B 实验的 CTR 提升作为奖励（但延迟高，需要 reward shaping）。实践中通常用 NDCG 或 Recall@K 作为代理奖励。

**Q5: 对于电商 Chatbot 的意图预测，除 RGAlign-Rec 外还有哪些典型方法？**
A: (1) 基于规则的意图匹配（Intent Classification）——快速但覆盖有限；(2) BERT/RoBERTa 意图分类——端到端训练但语义泛化性一般；(3) 基于 RAG 的检索增强生成——从知识库检索相关意图再生成推荐；(4) 多轮对话推荐系统（CRS）——通过多轮问答澄清意图。RGAlign-Rec 的优势在于将 LLM 推理和排序对齐融为一体。
