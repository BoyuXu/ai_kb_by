# RAG 系统全景与决策框架

> 综合自 5+ 篇 synthesis | 更新：2026-04-13 | 领域：RAG/检索增强生成
> 关联：[[concepts/embedding_everywhere]] | [[04_Agent系统完整指南]]

---

## RAG 演进三代

| 维度 | Naive RAG | Advanced RAG | Agentic RAG |
|------|----------|-------------|-------------|
| 检索策略 | 单次 top-K | 查询重写 + 混合检索 | **多步迭代 + 自适应触发** |
| 文档处理 | 固定 chunk | 语义分块 + 重叠 | 分层索引 + 知识图谱 |
| 生成控制 | 拼接 context | Re-ranking + 压缩 | **反思 + 工具调用** |
| 幻觉控制 | 无 | Faithfulness 约束 | **Self-RAG 自我验证** |
| Query 增强 | 无 | HyDE（假设文档） | 子问题分解 + CoT |

---

## 架构总览

```
离线索引: 原始文档 → Chunking → Embedding → 向量数据库(FAISS/Milvus)
在线检索: Query → Query改写 → ANN检索 + BM25 → RRF混合融合 → Reranker精排
生成:     精排 top-K + Query → Prompt 组装 → LLM 生成 → 回答
```

---

## 一、核心公式

### BM25 稀疏检索

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t,d) \cdot (k_1 + 1)}{\text{TF}(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}
$$

**直觉**：稀有词(IDF高)在某文档频繁出现(TF高) → 该文档与 query 高度相关。$k_1=1.2$ 防 TF 无限增长，$b=0.75$ 修正长文档偏差。

### Hybrid Retrieval 融合

$$
\text{score}_{hybrid} = \alpha \cdot \text{score}_{dense} + (1-\alpha) \cdot \text{score}_{sparse}
$$

Dense 擅长语义匹配（"汽车"≈"轿车"），Sparse 擅长精确匹配（型号、代码）。实践中 hybrid 几乎总优于单一方法。

### RRF（Reciprocal Rank Fusion）

$$
\text{RRF}(d) = \sum_{q'} \frac{1}{k + \text{rank}_{q'}(d)}, \quad k=60 \text{ (经典值)}
$$

### Reranker（交叉编码器）

$$
\text{score}(q, d) = \text{MLP}(\text{CLS}(\text{BERT}([q; \text{SEP}; d])))
$$

**工业流程**：双塔/BM25 粗检索 top-100 → Reranker 精排 top-10 → LLM 生成。

---

## 二、Chunk 策略

| 策略 | 特点 | 适用 |
|------|------|------|
| 固定长度(512 tok + 128 overlap) | 最简单 | 通用 |
| 语义段落切分 | 按标题/段落边界 | 结构化文档 |
| 递归切分 | 先段落、太长再按句子 | 兼顾两者 |
| Agentic Chunking | LLM 判断切分边界 | 最灵活但最慢 |

**最佳实践**：512-1024 tokens + 20% overlap + 按语义段落切分。

---

## 三、Query 改写（最高杠杆）

- **HyDE**：先生成假设答案，用假设答案去检索（答案和文档语义更接近）
- **Multi-Query**：将 query 从多角度改写，合并检索结果
- **子问题分解**：复杂 query 拆成子问题分别检索
- 可提升 Recall 10-20%

---

## 四、自适应 RAG

### Self-RAG
模型输出特殊 token 自主决定：
1. [Retrieve]：是否需要检索
2. [IsRel]：检索结果是否相关
3. [IsSup]：生成的回答是否被检索内容支持

### CRAG（Corrective RAG）
检索后质量评估 → 低质量时回退到 Web 搜索或直接生成。

### Doctor-RAG
失败恢复机制：检测检索失败 → 自动修正策略 → 重试。

### CoCR-RAG
概念蒸馏：将检索到的长文档蒸馏为核心概念，再基于概念生成。

---

## 五、RAG vs Fine-tuning 决策框架

### 决策树

```
知识频繁更新？
├── 是（周/日级）→ RAG
├── 否 → 格式/风格问题？
│   ├── 是 → Fine-tuning
│   └── 否 → 数据量？
│       ├── < 100条 → RAG + Few-shot
│       ├── 100-10000条 → LoRA Fine-tuning
│       └── > 10000条 → Full FT / Continue Pretraining
```

### 场景选型

| 场景 | 推荐 | 原因 |
|------|------|------|
| 客服(产品知识) | RAG | 产品手册频繁更新 |
| 代码补全(特定风格) | Fine-tuning | 格式固定 |
| 医疗问答 | RAG + FT | 知识实时性 + 专业理解 |
| 广告文案 | Fine-tuning | 风格一致性 |
| 数学推理 | RLVR FT | 有可验证标准答案 |

### 成本对比（7B 模型）

| 方案 | 一次性成本 | 运行成本 | 更新成本 |
|------|----------|---------|---------|
| Prompt Engineering | 低 | 高（长 prompt） | 低 |
| RAG | 中（向量库） | 中 | 低（更新文档即可） |
| LoRA FT | 中 | 低（短 prompt） | 中（重训） |
| Full FT | 高 | 低 | 高 |

### 常见误区

1. **"FT 一定比 RAG 好"** → FT 无法解决知识实时性
2. **"RAG 就够了"** → 模型不理解领域术语时检索会失败
3. **"两者选一个"** → 最优解常常是组合：FT 理解领域 + RAG 提供实时知识

---

## 六、前沿方向

### GraphRAG
GNN 作为 KG 检索器 + LLM 作为推理器。GNN-RAG 在 WebQSP 达到 85.7% Hits@1，输入 Token 减少 93%。

### RL 驱动的检索推理
- **Search-R1**：GRPO 训练 LLM 自主决定何时搜索、搜索什么
- **TongsearchQR**：PPO + NDCG 奖励，查询改写与检索系统端到端对齐

### Collab-RAG
多 Agent 协作检索，不同 Agent 负责不同检索路径。

---

## 七、减少 RAG 幻觉

1. **引用标注**：LLM 明确标注信息来自哪个 chunk
2. **Self-consistency**：多次生成取一致的回答
3. **Faithfulness 检查**：用另一个 LLM 验证回答是否忠于检索内容
4. **Grounding Score**：计算回答与检索内容的语义重叠度

---

## 面试高频 Q&A

### Q1: RAG 基本架构？
**30秒**：离线：文档切分→编码→向量库。在线：Query→(改写)→检索 top-K→(Reranker)→检索结果+Query 拼接→LLM 生成。

### Q2: RAG 检索效果差怎么优化？
**30秒**：① Chunk 调整（大小/overlap/语义切分）② Query 改写（HyDE/Multi-Query）③ 领域 Embedding（BGE-M3）④ 加 Reranker ⑤ Metadata filter。

### Q3: RAG vs FT vs 长上下文？
**30秒**：RAG—知识常更新/量大(>100K tokens)/需引用；FT—特定格式风格/知识固定；长上下文—知识量适中(<128K)/需全局理解。三者可组合。

### Q4: HyDE 原理？
**30秒**：不直接搜原始 query，先让 LLM 生成假设答案，用假设答案去检索（因为答案和文档语义更接近）。

### Q5: Self-RAG 怎么工作？
**30秒**：训练 LLM 输出特殊 token 决定是否检索、检索是否相关、回答是否被支持。模型自主决定何时检索、如何使用结果。

---

## 记忆助手

- **RAG = 开卷考试**：模型不需要记住所有知识，考试时翻书
- **Chunking = 拆书**：百科全书拆成词条，方便精准检索
- **HyDE = 以身试法**：先生成假设答案，用假设答案去检索
- **Reranker = 精筛**：粗检索 top-100，Cross-Encoder 精排挑 top-5
- **混合检索金律**：BM25 + Dense + RRF 是最鲁棒方案

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
