# 搜索 Agent 中的重排策略演进：从启发式到 LLM + Agentic Search 中的重排定位

> **覆盖论文**：RGS (2509.07163) | Rerank Before You Reason (2601.14224) | RAG Survey (2506.00054) | Reranking Evolution Survey (2512.16236) | KG-RAG E-commerce (2602.22219)
>
> **相关概念页**：[[attention_in_recsys|Attention in RecSys]] | [[embedding_everywhere|Embedding 技术全景]] | [[generative_recsys|生成式推荐]]
>
> **关联 synthesis**：[[20260420_retrieval_reranking_distillation|检索-重排-蒸馏前沿]] | [[20260411_dense_retrieval_and_reranking_advances|密集检索与重排进展]]

---

## 0. 全文导读

本文围绕一个核心问题：**在 Agentic Search / Deep Research 场景下，重排（Reranking）的角色正在从"后处理滤波器"演变为"搜索引导器"和"推理预算分配器"**。我们沿三条线索展开：

1. **技术演进线**：启发式 → 神经网络 → LLM 重排（Survey 视角）
2. **架构创新线**：从顺序流水线到图引导搜索（RGS）
3. **系统设计线**：Agentic Search 中重排的最优定位与预算分配

---

## 1. 重排技术演进全景（基于 2512.16236）

### 1.1 三代重排模型

| 代际 | 代表方法 | 核心机制 | 优势 | 局限 |
|------|---------|---------|------|------|
| **第一代：启发式** | BM25 re-score, TF-IDF 加权 | 词频/逆文档频率统计 | 速度快、可解释 | 无语义理解 |
| **第二代：神经网络** | BERT Cross-Encoder, ColBERT, MonoT5 | 深度交互/生成式打分 | 语义匹配强 | 计算开销大 |
| **第三代：LLM 重排** | RankGPT, RankVicuna, RankZephyr | 提示工程 + zero-shot 排序 | 泛化能力强、无需标注 | Token 开销高、延迟大 |

### 1.2 关键神经重排架构

**Cross-Encoder**：联合编码 query-document 对，通过自注意力实现 token 级深度交互，输出相关性分数。效果最好但 $O(n)$ 次前向传播。

**ColBERT Late Interaction**：

$$S(q, d) = \sum_{i=1}^{|q|} \max_{j \in [1,|d|]} \mathbf{q}_i^\top \mathbf{d}_j$$

文档表示可预计算，检索时只需 MaxSim 运算，兼顾效果与效率。

**MonoT5 生成式重排**：将 query-doc 对输入 T5，生成 "true"/"false"，用 $P(\text{"true"})$ 作为相关性分数。ListT5 进一步用 FiD 架构实现 listwise 重排。

### 1.3 LLM 重排三种范式

| 范式 | 方法 | 机制 | 代表 |
|------|------|------|------|
| **Pointwise** | 逐条打分 | LLM 对每个文档独立评分 | GPT-4 relevance scoring |
| **Pairwise** | 两两比较 | LLM 比较两个文档哪个更相关，汇总排序 | Pairwise prompting |
| **Listwise** | 整列排序 | LLM 一次看多个文档，输出排列 | RankGPT（滑动窗口） |

**RankGPT 滑动窗口策略**：将候选文档分成窗口，每个窗口内 LLM 输出排列，再合并结果。解决 LLM 上下文长度限制。

### 1.4 知识蒸馏加速

- **KARD**：250M T5 通过知识增强+重排训练两阶段蒸馏，超越 fine-tuned 3B T5
- **RADIO**：用对比学习解决 teacher-student 偏好不对齐问题
- **温度蒸馏**：通过 softmax 温度参数 $T$ 产生软标签，修正交叉熵损失传递知识

---

## 2. 突破顺序流水线：Reranker-Guided Search（RGS, 2509.07163）

### 2.1 问题：传统 Retrieve-then-Rerank 的天花板

传统流水线的两大瓶颈：
1. **召回天花板**：重排只能在 top-k 文档中选，如果正确文档未被初始检索命中，重排无力回天
2. **计算预算限制**：LLM-based reranker 的推理成本限制了可处理文档数（通常 100-500 篇）

### 2.2 RGS 核心思想：让重排器引导搜索方向

RGS 不再按顺序执行"先检索后重排"，而是**在近似最近邻（ANN）搜索的邻近图（Proximity Graph）上进行贪心搜索，由重排器偏好决定探索方向**。

**算法流程**：
1. 构建 ANN 邻近图（如 HNSW）
2. 从入口点开始贪心搜索
3. 每步探索当前节点的邻居时，**用重排器而非嵌入相似度决定优先级**
4. 在固定重排预算内（如 100 次 reranker 调用），选择性地将预算分配给最有前景的候选

**核心洞见**：重排器不再是"事后评判者"，而是"搜索导航员"。

### 2.3 实验结果

在 100 文档重排预算下的提升（vs 传统 retrieve-then-rerank）：

| 基准 | 提升 |
|------|------|
| **BRIGHT**（推理密集检索） | **+3.5** |
| **FollowIR**（指令遵从检索） | **+2.9** |
| **M-BEIR**（多模态检索） | **+5.1** |

**关键发现**：当重排预算较大（500）时，最终检索精度更依赖 reranker 与真实标签的对齐程度，而非嵌入模型能力。这说明 **reranker 质量是上限，embedding 只是起点**。

---

## 3. Agentic Search 中的重排预算分配（2601.14224）

### 3.1 问题：Deep Search Agent 的推理预算如何分配？

Deep Research Agent（如 OpenAI Deep Research）通过迭代检索+推理回答复杂查询。核心问题：**把推理 token 花在重排上，还是花在最终推理上？**

### 3.2 Effective Token Cost（ETC）指标

$$\text{ETC} = \text{Input}_{nc} + \alpha \cdot \text{Input}_c + \beta \cdot \text{Output}_t$$

| 参数 | 含义 | 取值范围 |
|------|------|---------|
| $\text{Input}_{nc}$ | 非缓存输入 token | — |
| $\text{Input}_c$ | 缓存输入 token | — |
| $\text{Output}_t$ | 总输出 token（含推理） | — |
| $\alpha$ | 缓存折扣因子 | 0.1, 0.3, 0.5 |
| $\beta$ | 输出溢价因子 | 3, 5, 7 |

ETC 使不同硬件/API 定价下的效率-效果权衡具有可比性。

### 3.3 核心实验发现（BrowseComp-Plus, 830 查询）

**实验配置**：
- 模型：gpt-oss-20b / gpt-oss-120b
- 推理预算：Low（2k）/ Medium（8k）/ High（16k）
- 重排深度：d ∈ {10, 20, 50}

**关键结果**：

| 配置 | NDCG@5 | Recall@5 |
|------|--------|----------|
| 无重排（baseline） | 19.72 | 14.91% |
| top-50 重排 + medium 推理（oss-120b） | **46.05** | **32.17%** |

**四大洞见**：

1. **重排 > 推理**：中等重排（d=20）+ 低推理预算 ≈ 高推理预算无重排的效果，但 token 成本大幅降低
2. **边际递减**：d=20 → d=50 的提升远小于 d=0 → d=20
3. **模型规模交互**：大模型（120b）在有重排时从额外推理中获益更多；小模型（20b）的推理增益更大
4. **成本最优策略**：优先分配中等重排预算（d=10-20），再将剩余预算给最终推理

### 3.4 对 Agentic Search 的设计建议

```
                    ┌─────────────────┐
                    │  Query Analysis │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Initial Retrieval│ (dense/sparse/hybrid)
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Moderate Reranking (d=10-20)│ ← 性价比最高的环节
              │  低推理预算即可              │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Reasoning │ ← 剩余推理预算集中在此
                    │  (answer synthesis)│
                    └─────────────────┘
```

---

## 4. RAG 系统中的重排定位（2506.00054）

### 4.1 四类 RAG 架构中的重排角色

| 架构类型 | 重排定位 | 代表系统 |
|---------|---------|---------|
| **Retriever-centric** | 重排作为检索后过滤器 | RQ-RAG, RAG-Fusion |
| **Generator-centric** | 重排融入解码控制 | SELF-RAG, xRAG |
| **Hybrid** | 重排作为迭代循环组件 | IM-RAG, CRAG, FLARE |
| **Robustness-oriented** | 重排负责噪声过滤 | RAAT, BadRAG defense |

### 4.2 三种重排增强策略

1. **自适应重排**：根据查询复杂度动态调整重排深度（RLT, ToolRerank）
2. **统一流水线**：将排序与生成统一在单一架构中（RankRAG, uRAG）
3. **融合策略**：多路查询改写 + 倒数排名融合（RAG-Fusion, R2AG）

### 4.3 开放挑战

- **检索自适应**：根据任务难度动态校准检索策略
- **对抗鲁棒性**：防御语料级投毒攻击
- **多跳推理**：跨证据文档的组合推理
- **可解释性**：检索决策和生成溯源的透明化

---

## 5. 知识图谱上的重排实践（2602.22219）

### 5.1 电商 KG-RAG 场景挑战

- 结构化 KG 上的自然语言查询检索，需保留实体间关系
- 图结构信息在重排时容易丢失

### 5.2 三种流水线配置

| 流水线 | 检索策略 | 重排策略 | 特点 |
|--------|---------|---------|------|
| **FRWSR** | 全量检索 + 弱监督 | 浅层重排 | 速度快 |
| **FRMR** | 全量检索 | 多轮重排 | 精度高 |
| **BARMR** | BM25 + 稠密混合 | 多轮重排 | 平衡 |

### 5.3 实验结果（STaRK SKB 电商数据集）

| 指标 | 提升（vs 已有基准） |
|------|-------------------|
| **Hit@1** | **+20.4%** |
| **MRR** | **+14.5%** |

**关键发现**：Cross-Encoder 重排在 KG 结构化检索中提升显著，因为初始检索往往难以准确捕获实体关系语义。

---

## 6. 横向对比：五篇论文的核心对照

| 维度 | RGS | Rerank Before Reason | RAG Survey | Reranking Evolution | KG-RAG |
|------|-----|---------------------|------------|-------------------|--------|
| **核心贡献** | 重排引导搜索方向 | ETC 预算分配框架 | RAG 架构分类 | 重排技术全景 | KG 上重排实践 |
| **重排角色** | 搜索导航员 | 预算分配器 | 架构组件 | 独立模块 | 精度增强器 |
| **关键洞见** | reranker > embedder | reranking > reasoning | 自适应重排 | 蒸馏可逼近 LLM | Cross-Encoder + KG |
| **适用场景** | 推理密集检索 | Deep Research Agent | 通用 RAG | 信息检索全场景 | 电商结构化查询 |
| **公式/指标** | ANN graph search | ETC 公式 | — | 蒸馏损失函数 | Hit@1, MRR |

---

## 7. 面试考点

### Q1: 传统 retrieve-then-rerank 的核心瓶颈是什么？RGS 如何解决？

**答**：两大瓶颈：(1) 召回天花板——正确文档未进入 top-k 则重排无力回天；(2) 计算预算——LLM reranker 只能处理有限文档数。RGS 通过在 ANN 邻近图上贪心搜索，让 reranker 偏好引导探索方向，在固定预算内突破召回天花板。本质是从"顺序流水线"变为"交互式搜索"。

### Q2: 在 Deep Search Agent 中，推理预算应优先分配给重排还是最终推理？

**答**：根据 ETC 分析，中等重排（d=10-20）+ 低推理预算的性价比最高。具体来说，top-50 重排可将 NDCG@5 从 19.72 提升到 46.05。重排的边际效益在 d=20 后快速递减，因此最优策略是先保证中等重排深度，再将剩余预算分配给最终推理。

### Q3: LLM 重排有哪三种范式？各自优劣？

**答**：(1) Pointwise——逐条打分，简单但忽略文档间相对关系；(2) Pairwise——两两比较后汇总，精度较高但 $O(n^2)$ 复杂度；(3) Listwise（如 RankGPT）——一次排列多文档，效率和效果较好但受上下文长度限制，需滑动窗口。实践中 listwise 最常用。

### Q4: 知识蒸馏如何让小模型逼近 LLM reranker？

**答**：KARD 证明 250M T5 通过两阶段蒸馏（知识增强 + 重排训练）可超越 fine-tuned 3B T5。核心技巧包括：(1) 温度蒸馏传递软标签；(2) RADIO 用对比学习解决 teacher-student 偏好不对齐；(3) 理念是"用 LLM 生成训练信号，用小模型上线服务"。

### Q5: RAG 系统中重排有哪些增强策略？

**答**：三类：(1) 自适应重排——根据查询复杂度动态调整深度（简单查询浅排，复杂查询深排）；(2) 统一流水线——将排序与生成整合（如 RankRAG 在单一模型内完成排序+生成）；(3) 融合策略——多路查询改写 + 倒数排名融合（RAG-Fusion），提高召回多样性。

---

## 8. 技术趋势总结

```
传统 IR                     Agentic Search / Deep Research
─────────                   ──────────────────────────────

Retrieve → Rerank → Return   Retrieve ⇄ Rerank ⇄ Reason ⇄ Iterate
    ↓                              ↓
重排是后处理                  重排是搜索策略的核心组件
    ↓                              ↓
固定 top-k                    动态预算分配（ETC 框架）
    ↓                              ↓
Embedding 决定上限            Reranker 决定上限（RGS 发现）
```

**三个确定性趋势**：
1. **重排从被动到主动**：RGS 证明重排器可以引导搜索方向，不再受初始检索限制
2. **推理预算精细化分配**：ETC 框架证明"中等重排 + 低推理"是最优性价比配置
3. **蒸馏民主化**：KARD/RADIO 等方法让 250M 模型逼近 3B+ 效果，使 LLM-quality 重排可上线
