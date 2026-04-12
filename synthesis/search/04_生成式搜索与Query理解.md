# 生成式搜索与 Query 理解

> 综合文档 | 领域：搜索 | 合并自 5 篇 synthesis | 更新：2026-04-13
> 来源：generative_retrieval_evolution、generative_search_synthesis、端到端生成式搜索前沿、搜索Query理解.md、语义搜索与推理检索前沿

---

## 一、生成式搜索：从 Semantic ID 到端到端

### 1.1 技术演进

```
传统搜索：BM25 + 倒排索引
  → Dense Retrieval (DPR, FAISS)
    → 生成式文档检索 (DSI, TIGER)
      → 工业生成搜索 (OneSearch)
        → OneSearch-V2: CoT + 自蒸馏 + 行为对齐
  → Faceted Search: 规则 → 学习 → GenFacet（生成式端到端）
  → 用户控制: CTRL-Rec（自然语言实时控制推荐）
```

### 1.2 端到端生成式搜索的核心优势

传统级联 $N$ 阶段，每阶段保留率 $r_i$，全链路保留率：

$$
R_{\text{cascade}} = \prod_{i=1}^{N} r_i
$$

$N=5$, $r_i=0.9$ 时仅 0.59 —— 超过 40% 有效信息在级联中丢失。端到端消除中间截断：$R_{\text{e2e}} = 1$。这解释了 OneSearch 在长尾 query 上 Recall@50 提升 +11.3%。

### 1.3 Semantic Product Identifier (SPI)

```
SPI = [Cat_L1][Cat_L2][Cat_L3][RQ_Code_1][RQ_Code_2]
```

- 类目树提供语义骨架 + RQ-VAE 量化编码提供细粒度区分
- 前缀天然形成粗到细的语义聚类 → beam search 更高效
- 新商品只需类目+RQ-VAE即可获得 SPI → 解决冷启动

### 1.4 关键方法对比

| 系统 | 核心创新 | 关键指标 |
|------|---------|---------|
| FORGE (淘宝) | 多模态 SID + collision mitigation | 250M items, +0.35% transactions |
| OneSearch | SPI + Constrained Beam Search | CTR +2.3%, GMV +1.8% |
| OneSearch-V2 | CoT 增强 + 自蒸馏 | 长尾查询显著改善 |
| GenFacet | 联合生成 + GRPO 对齐 | Facet CTR +42%, UCVR +2% |
| IntRR | Recursive-Assignment Network | 单 token 预测, lowest latency |
| OpenOneRec | Itemic Tokens + Qwen3 | +26.8% Recall@10 |
| OneSug | 统一查询建议生成 | CTR +3.5%, P99 35ms |

### 1.5 SID 设计关键挑战

1. **Collision**：不同 item 映射到相同 SID → FORGE 的 multimodal + collision mitigation
2. **长度**：multi-token SID 导致 autoregressive 延迟 → IntRR 单 token RAN
3. **亿级 SKU 的 Trie 索引**：内存 10-20GB，Trie 查找 O(L) 与 SKU 数无关，但需 copy-on-write 支持高频上下架

---

## 二、Query 理解：从分词到 LLM 改写

### 2.1 QU 四层架构

```
规则分词 + 同义词词典 (~2010)
  → 统计方法 (CRF NER + SVM 意图分类, 2010-2016)
    → 深度模型 (BERT NER + Seq2Seq 改写, 2018-2022)
      → LLM 统一 QU (GPT/Claude 一个 prompt 完成全部, 2023+)
```

模块：(1) 拼写纠错 → (2) 分词+NER → (3) 意图分类（导航/信息/事务/多意图）→ (4) 同义词扩展 → (5) Query 改写

### 2.2 核心公式

**意图分类**：

$$
P(\text{intent} | q) = \text{softmax}(W \cdot \text{BERT}_{CLS}(q) + b)
$$

**Query Autocomplete**：

$$
P(q | q_{1:k}) = \prod_{t=k+1}^{T} P(q_t | q_{1:t-1})
$$

**HyDE（假想文档 Embedding）**：

$$
\text{Retrieve}(q) = \text{ANN}(\text{Encode}(\text{LLM}(q \to \hat{d})))
$$

**ThinkQE 迭代查询扩展**：

$$
E_t = f_\theta(Q, E_{t-1}, \text{TopK}(R(E_{t-1}))), \quad \text{converge if Jaccard}(E_t, E_{t-1}) > 0.85
$$

通常 $t=2$ 即收敛。

**推理增强查询分解**：

$$
\{q_1, ..., q_K\} = \text{LLM}(\text{"分解为子问题："} + q), \quad \text{result} = \text{Merge}(\text{Retrieve}(q_k))
$$

**Omni-RAG 融合评分**：

$$
\text{score}(d) = 0.3 \cdot s(q_{\text{orig}}, d) + 0.7 \cdot \max_{i} s(q_i^{\text{rewrite}}, d)
$$

LLM 改写 query 权重 0.7 >> 原始 query 0.3 —— 用户原始查询大多数情况下不是最优检索 query。

### 2.3 Query 理解的难度示例

| Query | 真实意图 | 挑战 |
|-------|---------|------|
| "苹果" | 80% 手机，15% 水果，5% 公司 | 词义消歧 |
| "便宜手机" | 阈值因人而异 | 个性化理解 |
| "怎么选显卡" | 实际是知识类 query | 意图识别 |
| "推荐适合去日本的行李箱" | 购买+场景意图 | 上下文推断 |

### 2.4 LLM QU vs 传统 NLU 实战对比

| 维度 | 传统 NLU (BERT) | LLM (7B+) |
|------|----------------|---------|
| 延迟 | < 5ms | 50-200ms |
| 标准意图准确率 | 92% | 95% |
| 新兴 query 准确率 | 60%（需重训） | 85%（zero-shot） |
| 维护成本 | 高（每季度重标注） | 低（prompt 调整） |
| 适用策略 | P50 高频 query | P99 长尾/新 query |

**实践结论**：LLM 做兜底，传统 NLU 做主力（80/20 分工）。

---

## 三、Embedding 训练效率革命

### KaLM-V2 Focal Contrastive Loss

$$
\mathcal{L}_{\text{focal}} = -\log \frac{e^{s(q, d^+)/\tau}}{\sum_{i=1}^{N} (1 - p_i)^{\gamma} \cdot e^{s(q, d_i)/\tau}}
$$

- $\gamma=2$ 时，easy negative（$p_i=0.9$）的权重仅 $0.1^2=0.01$
- 0.5B 模型击败 3B-26B，推理吞吐 10-15x
- 核心结论：**训练策略的重要性远超参数规模**

### 三阶段训练框架

1. **Stage 1 弱监督预训练**：200M+ 标题-正文对/问答对 → 建立广泛语义基础
2. **Stage 2 监督精调**：50M 高质量标注数据
3. **Stage 3 任务适配**：目标任务微调

跳过 Stage 1 性能下降 3-4%。

---

## 四、搜索系统架构三大演进方向

1. **统一化**：OneSearch/OneSug 验证单模型替代多阶段级联
2. **小型化**：KaLM-V2（0.5B）击败 7B，Omni-RAG 用 1.5B 做 QU
3. **生成化**："生成"正在取代"匹配"成为搜索核心操作

---

## 五、面试 Q&A

### Q1: 生成式搜索 vs 传统搜索？
优势：端到端优化、更好语义理解、易整合 LLM。挑战：生成质量控制、推理延迟、新 item 处理。

### Q2: SPI 与传统 Document Identifier 的区别？
传统（DSI/GENRE）用原子化 docid，缺层级语义。SPI 结合类目树（语义骨架）+ RQ-VAE（细粒度区分），前缀形成粗到细聚类。

### Q3: Query 理解包含哪些模块？
拼写纠错 → 分词+NER → 意图分类（导航/信息/事务） → 同义词扩展 → Query 改写。

### Q4: HyDE 为什么有效？
Query（疑问句）和 Document（陈述句）语言风格不同，假设文档 embedding 空间更接近真实文档。

### Q5: Query 改写的收益与风险？
收益：解决长尾 query 零召回。风险：改写过度语义偏移（"白色连衣裙"→"连衣裙"丢失属性）。规则：泛化不超过 1 层。

### Q6: LLM 做 QU 的优劣势？
优势：一个模型全部 QU 任务，零样本强。劣势：延迟高、成本高、不可控。平衡：简单 query 传统方法，复杂/模糊 query 调 LLM。

### Q7: GenFacet 的相关性如何定义？
传统是 query-doc 相关性；GenFacet 额外考虑 facet-to-intent 对齐和 facet-to-query-rewrite 完整性。

### Q8: OneSearch 的 Constrained Beam Search 亿级可行性？
Trie 内存 10-20GB，查找 O(L) 与 SKU 数无关。挑战：Trie 并发读写（copy-on-write）、跨分片 beam search、beam_size 需 50-100。

### Q9: 电商多模态 QU 的落地挑战？
(1) 延迟：MM-LLM 需蒸馏+量化；(2) 隐私：图片脱敏；(3) 冷启动：降级纯文本；(4) 改写风险：置信度过滤。

### Q10: ThinkQE 的 Corpus-Interaction 工程挑战？
每轮 +300-500ms；需实时访问索引；Jaccard 判断收敛；热门 query 预计算缓存。

---

## 参考文献

1. OneSearch / OneSearch-V2 — 电商统一生成搜索
2. OneSug — 统一生成式查询建议
3. FORGE (淘宝) — 多模态 SID 250M items
4. GenFacet — 生成式分面搜索
5. IntRR — Recursive-Assignment Network
6. OpenOneRec — Itemic Tokens + Qwen3
7. KaLM-V2 — Focal Contrastive Loss 0.5B
8. Omni-RAG — LLM 辅助 RAG 查询理解
9. SmartSearch — 排序击败结构化
10. ThinkQE — 迭代思维链查询扩展

---

## 相关概念

- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[01_检索范式_稀疏到混合到稠密|检索范式]]
- [[02_LLM增强检索与RAG|LLM增强检索与RAG]]
