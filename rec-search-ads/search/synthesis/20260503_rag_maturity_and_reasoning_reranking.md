# RAG 成熟化与推理增强检索（2025-2026）

> 5 篇论文综合：RAG Survey + RAG Evaluation + Rank1 + QAC-RAG + RAG Security

---

## 核心趋势

RAG 在 2025-2026 沿三个轴线同步成熟：

1. **架构统一**：从 pipeline（检索→生成分离）走向端到端集成
2. **推理增强**：Test-time compute 进入 IR 组件（Rank1 蒸馏 R1 推理链）
3. **安全与评估体系补齐**：安全分类学 + 评估框架

---

## 论文精读

### 1. RAG Comprehensive Survey (arxiv 2506.00054)

**定位**：首个同时覆盖架构设计 + 鲁棒性前沿的 RAG 综述

**分类框架**：
- Retriever-centric / Generator-centric / Hybrid / Robustness-oriented
- 增强维度：检索优化 / 上下文过滤 / 解码控制 / 效率

**核心洞察**：
- 检索精度 vs 生成灵活性的权衡是 RAG 的根本张力
- 自适应检索（何时检索、检索多少）和多跳推理仍是开放问题
- 效率 vs 忠实性的 trade-off 在长文档场景尤为突出

**面试考点**：RAG 四类架构各适用什么场景？

---

### 2. RAG Evaluation Survey (arxiv 2504.14891)

**定位**：最全面的 RAG 评估综述，桥接传统 IR 指标与 LLM 时代方法

**评估维度**：
| 维度 | 传统方法 | LLM 时代方法 |
|------|---------|-------------|
| 检索质量 | Recall@K, MRR, nDCG | LLM-as-judge relevance |
| 生成质量 | ROUGE, BLEU | Factual grounding score |
| 端到端 | EM, F1 | 多维综合评分 |
| 安全性 | N/A | Hallucination rate, toxicity |

**核心洞察**：
- 多数 RAG 论文只评单一维度，缺乏系统性
- LLM-as-judge 正在成为主流但自身有偏差
- 需要标准化的 RAG 评估 benchmark

---

### 3. Rank1: Test-Time Compute for Reranking (arxiv 2502.18418)

**核心创新**：首个将 test-time reasoning 引入 IR reranking 的工作

**方法**：
- 从 DeepSeek-R1 蒸馏 600K+ 推理链到 MS MARCO queries
- 训练 reranker 先生成 reasoning chain，再输出 relevance judgment
- 支持 user-input prompt 控制推理行为

**结果**：
- BEIR: Rank1-7B 78.6 nDCG@10 vs RankLLaMA-7B 76.8
- BRIGHT (推理密集): SOTA
- NevIR (否定理解): SOTA
- mFollowIR (多语言指令跟随): SOTA

**面试考点**：
- 为什么 reranking 需要 reasoning？→ 复杂查询需要多步推理判断相关性
- Rank1 vs 传统 cross-encoder 的本质区别？→ 显式推理 vs 隐式打分
- 参考：[[concepts/attention_in_recsys.md]]

---

### 4. QAC-RAG: Query Auto-Completion via RAG + DPO (arxiv 2602.01023)

**核心创新**：将 QAC 重新定义为列表生成 + 多目标对齐

**方法**：
- RAG 检索候选 + LLM 生成完整列表
- 多目标 DPO 对齐：相关性 + 安全性 + 多样性
- 迭代 critique-revision 生成训练数据

**结果**：
- 线上 A/B: 击键减少 5.44%，采纳率提升 3.46%
- 来自 Apple 大规模商业搜索

**面试考点**：QAC 为什么适合用生成式？→ 长尾覆盖 + 多目标可控

---

### 5. Securing RAG: Attack Taxonomy (arxiv 2604.08304)

**核心创新**：首个系统化 RAG 攻击面分类学

**安全框架**：
| 攻击面 | 类型 | 风险等级 |
|--------|------|---------|
| 预检索知识污染 | 知识库投毒 | 高 |
| 检索时访问操控 | Query injection | 中 |
| 下游上下文利用 | Prompt injection via context | 高 |
| 知识窃取 | 通过生成结果推断知识库 | 中 |

**面试考点**：RAG 相比纯 LLM 多了哪些安全风险？→ 外部知识注入扩大了攻击面

---

## 趋势总结

RAG 正从"检索后阅读"的 pipeline 进化为**推理集成系统**：
- **Rank1** 说明检索组件本身需要推理能力（System 2）
- **Apple QAC** 说明生成与检索的界限在模糊
- **安全/评估**体系成熟意味着 RAG 进入工业化阶段

→ 参考：[[synthesis/llm/03_RAG系统全景与决策框架.md]] | [[synthesis/search/02_LLM增强检索与RAG.md]]
