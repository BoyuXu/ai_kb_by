# LLM 赋能搜索系统前沿综述
> 综合学习笔记 | 领域：search | 日期：20260329

---

## 📌 概述

本综述基于5篇2025-2026年前沿论文，系统梳理 LLM 在现代搜索系统中的应用全链路：从查询理解（QU）、语义检索（Retrieval）、多向量模型（Late Interaction）到文档重排序（Reranking）的技术演进与工程实践。

**核心主题**：LLM 正在重构搜索系统的每一个环节，但工业落地的核心挑战始终是**效率与效果的权衡**。

---

## 📐 核心公式

### 公式 1：ColBERT MaxSim 相似度计算
Late Interaction 模型的核心评分函数，保留 token 级交互：

$$
S(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} (q_i \cdot d_j)
$$

其中 $q_i$ 为 query 第 $i$ 个 token 的向量，$d_j$ 为 document 第 $j$ 个 token 的向量。
- 计算复杂度：$O(|Q| \cdot |D|)$
- 相比 single-vector 的 $O(d)$ 更贵，但保留了更精细的 token-level 匹配

### 公式 2：GRPO 强化学习目标（Rank-R1）
将推理能力注入 reranker 的强化学习目标：

$$
\mathcal{L}_{GRPO} = -\mathbb{E}\left[\sum_t \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right] + \beta \cdot KL(\pi_\theta \| \pi_{ref})
$$

其中：
- $\pi_\theta$：当前策略（正在优化的 reranker）
- $\pi_{ref}$：参考策略（初始 instruction-tuned LLM）
- $\hat{A}_t$：advantage（由 rule-based reward 计算：format correct + correct answer → 1，否则 0）
- $\beta$：KL 惩罚系数，防止策略偏离过远

### 公式 3：知识蒸馏目标（KL 散度）
将大 teacher 模型（cross-encoder）的知识迁移到小 student 模型（ColBERT/SLM）：

$$
\mathcal{L}_{KD} = KL(P_{teacher} \| P_{student}) = \sum_i P_{teacher}(i) \log \frac{P_{teacher}(i)}{P_{student}(i)}
$$

其中 $P_{teacher}$ 和 $P_{student}$ 分别是 teacher 和 student 在候选文档上的相关性分数分布。
- LinkedIn 用此将 8B oracle judge 蒸馏为 SLM
- PyLate (GTE-ModernColBERT) 用此将 bge-reranker-v2-gemma 蒸馏为 ColBERT

### 公式 4：Hybrid Retrieval 插值
Dense 检索 + BM25 的线性插值（BRIGHT 上最有效的简单 trick）：

$$
S_{hybrid}(q, d) = \alpha \cdot S_{dense}(q, d) + (1-\alpha) \cdot S_{BM25}(q, d)
$$

其中 $\alpha = 0.5$ 在 BRIGHT benchmark 上使 nDCG@10 从 0.289 提升到 **0.372**（+28.4% 相对提升）。

---

## 🗺️ 搜索系统全链路技术图谱

```
用户查询（文本/图片）
        ↓
[Query Understanding]
 - 多模态 LLM 查询理解（图文融合）
 - 品类路由 / 属性抽取 / 意图识别
 - LinkedIn SAGE: LLM 评估框架
        ↓
[Retrieval]
 - BM25（稀疏，关键词匹配）
 - Dense Bi-encoder（单向量，MSMARCO 主流）
 - Late Interaction ColBERT（多向量，BRIGHT SOTA）
 - Hybrid Dense + BM25（互补，简单有效）
        ↓
[Reranking]
 - Cross-encoder（精度最高，速度最慢）
 - SLM（LinkedIn 方案：75x 吞吐提升）
 - Rank-R1（GRPO 推理增强，BRIGHT SOTA）
        ↓
最终排序结果
```

---

## 📚 论文速览与关键贡献

| 论文 | 机构 | 核心贡献 | 关键指标 |
|------|------|---------|---------|
| Query Understanding with Multi-Modal LLMs | 电商（占位） | 图文融合查询理解，视觉属性提取 | GMV +1-2%（典型工业值） |
| Semantic Search At LinkedIn | LinkedIn (2026) | 75x 推理吞吐提升；多教师蒸馏 SLM | 75x 吞吐；SAGE kappa=0.77 |
| PyLate (2025) | LightOn AI | Late Interaction 工具库；GTE-ModernColBERT SOTA | BEIR avg 54.89（SOTA） |
| BRIGHT (2025) | 多机构 | 推理密集型检索 benchmark；揭示现有模型盲区 | SOTA ~0.37 nDCG@10 |
| Rank-R1 (2025) | CSIRO/UWaterloo | GRPO 强化学习 reranker；无推理标注 | BRIGHT avg .205（超 GPT-4） |

---

## 🔑 核心技术对比

### 检索架构三角

| 维度 | Sparse (BM25) | Dense (Bi-Encoder) | Late Interaction (ColBERT) | Cross-Encoder |
|------|--------------|-------------------|--------------------------|--------------|
| 文档预计算 | ✅ 倒排索引 | ✅ 单向量 | ✅ 多向量 | ❌ 实时计算 |
| 语义理解 | ❌ 词汇匹配 | ✅ 语义 | ✅✅ Token 级语义 | ✅✅✅ 最强 |
| 推理能力 | ❌ | ❌ | 部分 | 部分 |
| 延迟（召回阶段）| 低 | 低 | 中 | 极高（不用于召回）|
| 索引大小 | 小 | 中 | 大（多向量）| 不需要索引 |

### LLM Ranking 效率技术对比（LinkedIn 方案）

| 技术 | 作用 | 效果 |
|------|------|------|
| Prefill-Only 推理 | 消除 autoregressive decode 开销 | 核心吞吐提升来源 |
| 结构化剪枝 | 减小模型参数量 | 配合其他技术共同实现 75x |
| 离线摘要压缩 | 减少在线 context 长度 | 降低 prefill 计算量 |
| 共享前缀 KV Cache | 相同 query 候选共享 context | 避免重复计算 |
| Text-Embedding 混合 | 融合稠密向量与文本交互 | 提升模型能力 |

---

## 🎓 Q&A（10道面试核心题）

**Q1：什么是 Late Interaction？它与 Dense Retrieval 和 Cross-Encoder 有何本质区别？**
> Dense Retrieval 压缩到单向量（lossy），Cross-Encoder 实时全交互（无法预计算文档），Late Interaction（ColBERT）保留每个 token 向量并预计算文档，通过 MaxSim 实现 token-level 匹配，平衡了两者的优缺点。

**Q2：LinkedIn 如何实现 75x 的 LLM 推理吞吐提升？**
> 核心是 Prefill-Only 执行（排序只需要打分，不需要生成文本，消除 decode 阶段）+ 共享前缀 KV Cache + 结构化剪枝 + 离线摘要压缩上下文 + Text-Embedding 混合交互，在固定延迟约束下综合实现 75x 吞吐提升。

**Q3：Rank-R1 如何在没有推理标注数据的情况下学会推理？**
> 使用 GRPO 强化学习：Reward 为纯规则化（格式正确 + 答案正确 → 1，否则 0），模型通过探索自发学习推理链，无需任何人工 reasoning annotation。仅使用 18% MSMARCO 数据，in-domain 效果与全量 SFT 相当，out-of-domain 远超 SFT。

**Q4：为什么 BRIGHT 上 SFT 14B 模型比 Zero-shot 14B 表现更差？**
> SFT 在 MSMARCO（浅层关键词匹配数据）上过拟合，学到了"依赖词汇重叠判断相关性"的 shortcut。14B 模型参数越多，这种过拟合越严重（记忆能力更强），导致域外泛化更差。GRPO 训练的推理能力因为是通用的"现场分析"能力，不依赖词汇记忆，泛化更好。

**Q5：什么是 PLAID 索引？为什么它是 ColBERT 生产部署的 de facto 标准？**
> PLAID（Product-quantized Late Interaction Approximate for Documents）使用 centroid-based 聚类和 bitmap 过滤来近似 MaxSim 计算，相比 exhaustive MaxSim 大幅降低延迟和内存。PyLate 的 PLAID 实现在 embedding 层操作，与建模解耦，兼容任何 late interaction 架构（包括多模态 ColPali）。

**Q6：为什么对比学习训练 retrieval 模型需要大 batch size？如何在不增加显存的情况下扩大 batch？**
> 对比学习依赖 in-batch negatives，batch 越大，hard negative 越多，训练越难，效果越好。PyLate 用 GradCache（解耦 forward/backward，真正等价大 batch）+ Multi-GPU Embeddings Gathering（跨 GPU 聚合 embedding 扩大 effective batch），在 8× H100 上实现 16k-32k 有效 batch size 而无 OOM。

**Q7：LLM 评估框架（如 LinkedIn SAGE）的作用是什么？如何保证其与人工标准对齐？**
> SAGE 解决"什么叫相关性"的治理问题：用 8B oracle 模型作为 LLM judge（0-4 分档 + 自然语言理由），通过显式产品政策 + 人工标注先例数据进行对齐。线性 kappa = 0.77（人工）/ 0.81（teacher），支持每日数千万次评估，保证实验和上线标准的一致性。

**Q8：电商搜索中多模态 QU 的核心技术挑战和解决方案是什么？**
> 挑战：1) 图文融合表示学习；2) 在线延迟要求（<50ms）vs LLM 推理慢；3) 电商专有知识不在通用 LLM 中。解决方案：1) 多模态 LLM（视觉-语言对齐）；2) 知识蒸馏 + 查询分级处理（简单用小模型，复杂用 LLM）；3) 电商 KG 注入 + 用户行为作为弱监督。

**Q9：Hybrid Retrieval（Dense + BM25）在什么情况下特别有效？如何调参？**
> 在推理密集型（BRIGHT）和专业领域检索中特别有效：dense 捕捉语义，BM25 捕捉精确术语（专业词汇往往关键）。调参：$\alpha$ 通常在 0.3-0.7 之间，在开发集上网格搜索；也可以用 learned linear interpolation 根据查询特征动态调整 $\alpha$。BRIGHT 上 $\alpha=0.5$ 实现 nDCG@10 从 0.289 → 0.372。

**Q10：如何设计一个从召回到排序的全链路 LLM 搜索系统，平衡效果和成本？**
> 分层设计：1) **召回**：BM25 + GPU 加速向量检索（穷举或 ANN），召回 top-100-200；2) **粗排**：双塔 or 轻量 Cross-Encoder（300M 参数），筛选 top-20-50；3) **精排**：SLM/Rank-R1（Prefill-Only + batching），复杂查询路由大模型；4) **QU 层**：多模态 LLM 蒸馏小模型，分级处理；5) **评估**：LLM judge（SAGE 模式）统一评估标准；6) **降级**：大模型超时自动 fallback 到传统 LTR。

---

## 📚 参考文献

1. **Semantic Search At LinkedIn** (Borisyuk et al., 2026). arXiv:2602.07309. LinkedIn 的 LLM 语义搜索框架，多教师蒸馏 SLM + 75x 推理吞吐提升。

2. **PyLate: Flexible Training and Retrieval for Late Interaction Models** (Clavié et al., 2025). arXiv:2508.03555. 基于 Sentence Transformers 的 ColBERT 训练/检索工具库，GTE-ModernColBERT SOTA。

3. **BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval** (Su et al., 2025). OpenReview:us-kxq531b. 推理密集型检索 benchmark，揭示现有模型在复杂查询上的盲区。

4. **Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning** (Zhuang et al., 2025). arXiv:2503.06034. GRPO 强化学习 reranker，无推理标注，BRIGHT 上超越 GPT-4。

5. **Query Understanding with Multi-Modal LLMs for E-Commerce Search** (2026). arXiv:2601.xxxx（占位）. 电商多模态查询理解，图文融合，视觉属性提取。

6. **ColBERTv2** (Santhanam et al., 2022). NAACL 2022. Late Interaction 检索模型的里程碑工作，引入 PLAID 索引和知识蒸馏训练。

7. **DeepSeek-R1** (Guo et al., 2025). 使用 GRPO 强化学习训练推理能力的 LLM，启发了 Rank-R1 的设计。

8. **RankZephyr** (Pradeep et al., 2023). GPT-4 distillation 训练的 Listwise reranker，是 TREC DL 的强基线。

9. **BEIR Benchmark** (Thakur et al., 2021). 信息检索域外泛化评估 benchmark，是 PyLate 等工作的主要评估场景。

10. **SGLang** (open-source). LinkedIn 将 Prefill-Only 推理栈开源的项目，支持高吞吐 LLM scoring 推理。
## 参考文献

- [distillation](../../papers/distillation.md)
- [Sparse](../../papers/sparse.md)
- [clip](../../papers/clip.md)
- [GPT-4](../../papers/gpt_4.md)

## 📐 核心公式直观理解

### Query2Doc

$$
d_{\text{pseudo}} = \text{LLM}(\text{"生成一段回答以下问题的文档："} + q)
$$

**直观理解**：让 LLM 先"想象"一篇理想的答案文档，再用这篇伪文档增强检索。伪文档虽然可能有错误，但在 embedding 空间中和真相关文档更近（都是"答案风格"的文本），比直接用 query（问题风格）检索效果好 10-20%。

### 检索增强生成的 FiD（Fusion-in-Decoder）

$$
P(y | q, d_1, ..., d_K) = \text{Decoder}(\text{Enc}(q, d_1) \oplus ... \oplus \text{Enc}(q, d_K))
$$

**直观理解**：每个检索到的文档和 query 分别编码（encoder 内文档间独立），然后在 decoder 中融合所有编码。这比把所有文档拼接后输入更高效——encoder 部分可并行，融合推理只在 decoder 做。

### 搜索 Agent 的迭代检索

$$
q_{t+1} = \text{LLM}(q_t, d_{1:K}^{(t)}, \text{"哪些信息还缺？请改写 query"})
$$

**直观理解**：第一轮检索可能找不到完整答案，LLM 分析返回结果后自动改写 query 做第二轮检索——就像研究者在图书馆"越查越精确"。通常 2-3 轮迭代就能找到全面的答案。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
