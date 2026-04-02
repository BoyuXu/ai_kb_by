# Semantic Search At LinkedIn: LLM-based Semantic Search Framework
> 来源：https://arxiv.org/abs/2602.07309 | 领域：search | 学习日期：20260329

## 问题定义

LinkedIn Search 服务于 People Search（人才发现）和 Job Search（职位搜索）两大核心场景，每秒处理数十万次查询。传统关键词检索无法捕捉自然语言语义意图，且 LLM 推理开销大，难以在严格延迟约束下实现生产级 LLM 排序。

核心挑战：
1. 语义检索：如何用 LLM 理解查询意图而非关键词匹配
2. 推理效率：LLM cross-encoder 推理代价随上下文长度线性增长，难以达到生产 QPS
3. 多目标优化：需同时优化 relevance（相关性）和 engagement（用户参与度）

## 核心方法与创新点

### 系统架构
三段式流水线：
1. **Query Understanding (QU) 层**：将模糊短查询转为确定性机器可解析信号（路由决策、归一化属性、查询重写）
2. **GPU加速穷举检索**：Embedding-based retrieval，不用 ANN 近似，支持亿级索引全量扫描+属性过滤
3. **SLM Reranker**：对 Top 250 候选重排，联合预测 relevance 和 engagement

### SLM（Small Language Model）Reranking
**多阶段训练框架（Multi-Teacher Distillation, MTD）：**
1. 训练专门的 teacher 模型（relevance teacher + engagement teacher）
2. 多教师蒸馏为单一 student SLM，使用分布式监督 + relevance-aware 暖启动 + 不平衡感知 loss masking
3. 应用 LLM 特征工程（MixLM 等）

**Prompt 结构：** `system_prefix + context(query+searcher features) + document + suffix`，固定查询时 context 在所有候选间共享

### SAGE（语义评估框架）
- 8B oracle 模型作为 relevance judge，输出 0-4 档相关性分数 + 自然语言理由
- 与人工标注线性 kappa = **0.77**，与 teacher 对齐 kappa = **0.81**
- 支持每天数千万次评估

### 推理优化（75x 吞吐提升）
- **结构化剪枝（Structured Pruning）**：减小模型体积
- **离线摘要（Context Compression）**：候选文档离线预生成摘要，压缩上下文长度
- **Text-Embedding 混合交互（Hybrid Interactions）**：结合稠密向量与文本交互
- **Prefill-Only 执行**：评分任务只需 prefill 阶段，无需 decode，共享前缀 KV cache
- 推理栈开源为 SGLang 项目的一部分

## 实验结论

| 指标 | 数值 |
|------|------|
| 推理吞吐提升 | **75x**（固定延迟约束下） |
| NDCG 保留 | 接近 teacher 水平 |
| SAGE 人工对齐 | kappa = 0.77 |
| SAGE vs teacher | kappa = 0.81 |
| 生产 QPS | 数十万/秒（支持 LinkedIn 全量搜索） |
| SLM Reranking 候选数 | Top 250 |

定性结论：
- SLM 联合排序（相关性 + 参与度）优于 DLRM-style baseline
- 首批在生产中实现 LLM 排序效率与传统方法相当的系统之一

## 工程落地要点

1. **Prefill-Only 推理架构**：排序任务只评分不生成，消除 decode 开销，是高 QPS LLM ranking 的核心突破
2. **共享前缀 KV Cache**：相同查询的候选共享 context prefix，大幅降低重复计算
3. **离线摘要预计算**：候选文档文本离线压缩，减少 online context 长度
4. **穷举检索代替 ANN**：GPU 加速全量扫描，避免 ANN 的"liquidity"损失（特别适合有复杂属性过滤的场景）
5. **多教师蒸馏**：单个 student 模型同时继承 relevance + engagement 两个专家的知识，避免部署多模型
6. **SAGE 评估治理**：建立标准化 LLM 评估框架，保证实验和上线决策的一致性

## 常见考点

**Q1：LinkedIn 语义搜索为什么选择 Prefill-Only 推理架构？**
> A：LLM 排序任务只需要对 query-document pair 打分（pointwise 或 listwise），不需要生成文本。Prefill 阶段可以计算整个输入序列的 representation，然后直接从最后 token 的 hidden state 提取分数，无需 autoregressive decode。这将推理吞吐提升了 75x，是 LLM ranking 生产化的关键。

**Q2：什么是多教师蒸馏（MTD），在该场景中如何应用？**
> A：MTD 同时从多个专门化 teacher 学习。LinkedIn 有 relevance teacher（相关性）和 engagement teacher（参与度），两者优化目标不同。通过 distributional supervision + relevance-aware 暖启动 + 不平衡感知 loss masking，将两者知识蒸馏进单一 SLM。好处是生产只需部署一个模型，同时保留多目标能力。

**Q3：SAGE 框架解决了什么问题？**
> A：解决了工业级搜索中"什么叫相关性"的治理问题。LLM 作为评估标准本身也需要对齐人类标准。SAGE 使用显式产品策略 + 人工标注先例数据 + LLM surrogate judge 构成闭环，线性 kappa 0.77（人工对齐），0.81（与 teacher 对齐），保证实验可复现、上线标准一致。

**Q4：为什么 LinkedIn 选择穷举检索而非 ANN？**
> A：LinkedIn Job/People Search 有大量属性过滤（位置、公司、学历等），ANN 索引难以高效支持复杂过滤（即"liquidity"问题）。GPU 加速全量扫描可以在低延迟下对十亿级索引进行全量检索并叠加任意属性过滤，准确率更高，不存在近似误差。

**Q5：SLM Reranker 的 Prompt 结构如何设计，为什么这样设计？**
> A：结构为 `system_prefix + context(query + searcher features) + document + suffix`。关键点在于：相同查询的所有候选共享 context 部分，这样可以共享 KV Cache。system_prefix 定义任务指令和 chat template；context 包含查询和搜索者画像（实现个性化）；document 是候选文档的结构化表示；suffix 触发评分输出。
