# LLM推理增强搜索重排与查询扩展进展综合总结

> 综合日期：20260401 | 领域：Information Retrieval | 方向：LLM Reasoning for Search

## 概述

2024-2025 年，搜索领域最重要的范式转变之一是**将 LLM 推理能力（Reasoning）深度融入检索重排和查询扩展**。本综合总结覆盖五篇代表性论文，分析以下四个核心方向：

1. **Test-Time Compute 重排**：让重排模型"思考得更多"（Rank1）
2. **RL 驱动的推理重排代理**：用强化学习训练推理重排（REARANK、ERank）
3. **查询扩展的推理化**：思维链驱动的迭代式查询扩展（ThinkQE）
4. **指令跟随能力评测与训练**：让 IR 模型真正理解用户指令（FollowIR）

---

## 📚 参考文献

1. **Rank1**: Weller et al., "Test-Time Compute for Reranking in Information Retrieval", arXiv:2502.18418, CoLM 2025. https://arxiv.org/abs/2502.18418

2. **REARANK**: Zhang et al., "REARANK: Reasoning Re-ranking Agent via Reinforcement Learning", arXiv:2505.20046, 2025. https://arxiv.org/abs/2505.20046

3. **FollowIR**: Weller et al., "FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions", arXiv:2403.15246, NAACL 2024. https://arxiv.org/abs/2403.15246

4. **ThinkQE**: Lei et al., "ThinkQE: Query Expansion via an Evolving Thinking Process", arXiv:2506.09260, EMNLP 2025 Findings. https://arxiv.org/abs/2506.09260

5. **ERank**: Cai et al., "ERank: Fusing SFT and RL for Efficient Reasoning Reranking", arXiv:2509.00520, 2025. https://arxiv.org/abs/2509.00520

---

## 技术脉络与关联

```
推理能力的引入路径：
                    ┌─────────────────────────────────────┐
                    │     大型推理模型 (o1/R1/QwQ)           │
                    │  能力：CoT、多步推理、自我修正         │
                    └─────────────┬───────────────────────┘
                                  │ 蒸馏/直接使用
              ┌───────────────────┼───────────────────────┐
              ▼                   ▼                       ▼
    [检索前：QueryExp]    [重排：Reranking]         [指令理解]
         ThinkQE          Rank1 / REARANK          FollowIR
    (思维链扩展查询)    ERank(高效推理重排)     (指令跟随评测)
              │                   │
              │               效率优化
              │            SFT+RL两段式
              └───────────────────┘
                       ERank
              (统一高效 + 高准确 方案)
```

---

## 📐 核心公式

### 公式1：nDCG@k（所有重排方法的核心评测指标）

$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

其中：

$$
\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

- $rel_i$：位置 $i$ 处文档的相关性分数（0-3 或 0-5）
- $\text{IDCG@k}$：理想排序的 DCG（所有相关文档按相关性降序排列）
- 位置折损因子 $\log_2(i+1)$ 体现了"越靠前越重要"的用户行为规律

### 公式2：ERank 的 Listwise RL 奖励函数

$$
R_{listwise}(\mathbf{S}, \pi^*) = \text{nDCG@10}\left(\text{argsort}(-\mathbf{S}),\ \pi^*\right)
$$

其中 $\mathbf{S} = (S_1, S_2, ..., S_k)$ 为 pointwise 打分向量，$\pi^*$ 为金标准排序。

GRPO 更新梯度：

$$
\nabla_\theta \mathcal{L}_{RL} = \mathbb{E}_{q \sim \mathcal{D}} \left[ \sum_{i=1}^{k} (R_i - b) \nabla_\theta \log \pi_\theta(S_i | q, d_i) \right]
$$

其中 $b$ 是 batch 内奖励均值（baseline，降低方差）。

### 公式3：ThinkQE 迭代查询扩展

$$
E_t = f_\theta\left(Q,\ E_{t-1},\ \text{TopK}(R(E_{t-1}))\right)
$$

- $E_t$：第 $t$ 轮扩展词集合
- $Q$：原始查询
- $R(E_{t-1})$：用 $E_{t-1}$ 检索到的 Top-K 文档
- $f_\theta$：带有思维链的 LLM

收敛条件：

$$
\text{Jaccard}(E_t, E_{t-1}) = \frac{|E_t \cap E_{t-1}|}{|E_t \cup E_{t-1}|} > \tau
$$

通常 $\tau = 0.85$，实践中 $t = 2$ 即收敛。

### 公式4：FollowIR 的 p-MRR（pairwise 指令跟随评测）

$$
p\text{-}MRR = \frac{1}{|Q|} \sum_{q \in Q} \left[ \frac{1}{2}\text{MRR}(q, A) + \frac{1}{2}\text{MRR}(q, B) \right]_\text{contrast}
$$

核心思想：对于同一 query，在指令 A 下文档 $d_A$ 应排前，在指令 B 下 $d_B$ 应排前。p-MRR 测量模型正确响应指令对比的能力，而非绝对排序质量。

### 公式5：Rank1 的 Test-Time Compute 扩展

$$
\text{Score}(q, d) = P_\theta(\text{Relevant} | q, d, \text{CoT_{1:T}})
$$

其中 CoT 长度 $T$ 是可控变量：
- $T$ 小 → 快速低成本推理
- $T$ 大 → 深度推理，更高准确率

性能随 $T$ 的近似关系（log-linear）：

$$
\text{nDCG@10} \approx \alpha \cdot \log(T) + \beta
$$

---

## 各方法性能对比（BRIGHT 基准，nDCG@10）

| 方法 | BRIGHT | 方法类型 | 效率 | 训练需求 |
|------|--------|---------|------|---------|
| BM25 | 4.0 | 词汇检索 | ✅✅✅ | 无 |
| E5-Mistral-7B | 22.5 | Dense Retrieval | ✅✅ | 有 |
| ThinkQE + BM25 | 26.7 | 查询扩展 | ✅✅ | 无 |
| Rank1-7B | ~32 | 推理重排 | ⚠️ | 有（蒸馏） |
| REARANK-7B | ~30 | RL 推理重排 | ⚠️ | 有（RL） |
| GPT-4 重排 | 28.4 | 闭源推理重排 | ⚠️ | 无 |
| **ERank-4B** | **38.7** | **RL 精确重排** | **✅✅** | **有** |
| **ERank-32B** | **40.2** | **RL 精确重排** | **✅** | **有** |

---

## 🎓 常见Q&A（≥10条）

### Q1：什么是 Test-Time Compute？它在搜索重排中有什么意义？

**A**：Test-Time Compute（推理时计算扩展）是指允许模型在推理阶段使用更多计算（更长的思维链、更多采样）来提升性能，而不仅仅依赖参数量。在搜索重排中（Rank1），这意味着模型可以为每个 query-document 对生成详细的推理链，分析多个相关性维度，最终给出更准确的排序判断。类比：不是让学生靠背书（更多参数），而是让他们有更多时间在考场"思考"（推理时间）。

### Q2：Rank1 如何用 DeepSeek-R1 进行知识蒸馏？蒸馏的目标是什么？

**A**：Rank1 的蒸馏流程：
1. 用 R1 对 60 万条 (query, passage) 对生成完整推理轨迹（思维链 + 相关性结论）
2. 这些推理轨迹作为监督信号，对小模型（如 7B）进行 SFT
3. 小模型不仅学"答案"（Relevant/Not Relevant），还学"推理过程"

蒸馏目标：让小模型在重排任务上复现大模型的推理质量，同时保持可部署的计算成本。

### Q3：REARANK 只用 179 个标注样本，为什么 RL 训练比 SFT 更数据高效？

**A**：核心原因在于 RL 的"奖励驱动探索"机制：
- SFT 只学习"已标注"的排序，泛化依赖标注覆盖率
- RL 通过奖励（nDCG）探索不同排序策略，自动发现"好的排序"而不需要逐条标注
- 奖励函数（nDCG）直接对齐评测目标，消除了 SFT 标注与评测指标之间的 gap
- 数据增强（排列扩增）进一步放大了少量标注的价值

### Q4：FollowIR 发现现有 IR 模型在指令跟随上有什么具体问题？

**A**：FollowIR 实验揭示：
1. **关键词提取偏差**：模型从"请返回关于气候变化对极地熊影响的文章，不要包含政策讨论"中只提取"气候变化、极地熊"，忽略"不要包含政策"
2. **否定条件失败**：无法正确处理"不相关的文档类型"指令
3. **语义浮浅理解**：将指令当查询扩展，不理解其约束含义
4. **长指令截断**：超过 512 tokens 的叙述性指令被截断处理

### Q5：ThinkQE 的 Corpus-Interaction 策略在工程上有什么挑战？

**A**：主要挑战：
1. **延迟增加**：每轮迭代需要额外的检索 + LLM 推理，增加 ~300-500ms
2. **语料库访问依赖**：需要在查询处理时能实时访问索引（适合在线场景，不适合离线预计算）
3. **迭代控制**：需要判断收敛条件，避免无限循环
4. **缓存策略**：相同查询的不同迭代结果可缓存，但需要考虑语料库更新的失效问题

解决方案：生产中限制 T=2，用 Jaccard 相似度快速判断收敛，对热门查询预计算并缓存。

### Q6：ERank 的两阶段训练（SFT + RL）各自解决什么问题？能否只用其中一个？

**A**：
- 仅 SFT（二分类）：判别力不足（binary），nDCG = 29.3
- 仅 SFT（整数分数）：比二分类好，但仍缺全局排序感知，nDCG = 33.1
- SFT + RL（pointwise reward）：RL 有帮助但奖励信号局部，nDCG = 35.8
- SFT + RL（listwise reward）：最优，nDCG = 38.7

结论：两阶段缺一不可。SFT 提供好的初始化和细粒度判别力；RL 注入全局排序感知。

### Q7：listwise 和 pointwise 重排在实际系统中如何选择？

**A**：
- **Listwise**（Rank1, REARANK）：适合高质量、可接受高延迟（>500ms）的场景，如文档检索、企业知识库搜索
- **Pointwise**（ERank）：适合实时搜索、移动端、延迟要求 <100ms 的场景
- **Hybrid**：先用 BM25/Dense 粗排，再用 ERank pointwise 精排（推荐的生产方案）

### Q8：BRIGHT 基准为什么是评测 LLM 推理重排的最佳基准？

**A**：BRIGHT（Benchmark for Reasoning-Intensive Generative Tasks）的特点：
- 来源：StackExchange、LeetCode、科学问答等专业领域
- 特征：查询需要深度理解才能匹配文档（如"这段代码的时间复杂度优化建议"）
- 传统模型（BM25）仅 nDCG@10 ≈ 4.0，接近随机
- 真正考验模型的推理理解能力，而非词汇匹配
- 这就是为什么 Rank1、ERank 等推理模型在 BRIGHT 上优势最明显

### Q9：如何在 RAG 系统中整合这些新型重排方法？

**A**：推荐架构：
```
用户 Query + 意图描述 (instruction)
    ↓
ThinkQE 扩展 Query（提升召回）
    ↓
Dense Retrieval Top-100
    ↓
ERank 重排 Top-10（高效 + FollowIR-style 指令理解）
    ↓
Rank1 / REARANK 生成推理链（注入 LLM 生成 context）
    ↓
LLM 生成（利用推理链作为参考）
```

### Q10：推理链（CoT）在重排中如何提升性能？有什么理论解释？

**A**：理论解释：
1. **注意力引导**：CoT 强制模型逐步分析文档的不同方面（时效性、权威性、相关性），防止"过早下结论"
2. **多角度评估**：推理过程天然考虑多个维度，而不是单一相关性分数
3. **否定处理**：推理步骤中可以显式处理"为什么不相关"，提升精确率
4. **指令理解**：复杂指令可以在 CoT 中逐步解析，正确识别约束条件

实验证据：所有论文的消融实验都显示，去除推理链后性能显著下降（3-8% nDCG），在复杂查询（BRIGHT）上下降更明显。

### Q11：查询扩展 vs 重排：这两个阶段如何协同工作？

**A**：
- **查询扩展**（ThinkQE）：作用在检索阶段，提升**召回率**，让更多相关文档进入候选集
- **重排**（Rank1/REARANK/ERank）：作用在排序阶段，提升**精确率**，让最相关的文档排在最前
- **协同效果**：两者互补，叠加使用时性能提升显著
- **成本分配**：查询扩展增加检索时间，重排增加排序时间，需要根据系统延迟预算分配

### Q12：FollowIR 的 p-MRR 指标与传统 nDCG 的根本区别是什么？

**A**：
- **nDCG**：测量"能否找到相关文档"（绝对排序质量）
- **p-MRR**：测量"能否根据指令调整排序"（指令响应能力）

关键区别：模型可能在两个互相矛盾的指令下给出几乎相同的排序（nDCG 都高，但 p-MRR 低），说明模型没有真正理解指令，只是找到了"普遍相关"的文档。p-MRR 专门检测这种"伪理解"。

---

## 趋势总结与展望

### 2024-2025 核心趋势

1. **推理模型全面进入 IR**：R1/o1 范式成功迁移到搜索重排，性能大幅提升
2. **效率问题是主要障碍**：listwise 方法虽准确，但工业部署仍以 pointwise 为主
3. **ERank 是目前最佳实践**：SFT+RL 两段式，解决了效率-准确度困境
4. **指令跟随成为新评测维度**：FollowIR 揭示了 IR 模型的重要能力缺口
5. **迭代检索-扩展循环**：ThinkQE 展示了"先检索再改进查询"的价值

### 未来方向

- **多模态指令跟随**：图文混合查询的指令理解
- **动态推理预算**：根据查询复杂度自动调整推理深度
- **在线学习**：利用用户反馈实时优化推理策略
- **端到端推理检索**：将查询扩展和重排融合为单一推理步骤

## 📐 核心公式直观理解

### Pointwise LLM Scoring

$$
\text{score}(q, d) = \text{LLM}(\text{"1-5分评估相关性："} + q + d)
$$

**直观理解**：直接让 LLM 打分。简单但不稳定——LLM 倾向于给高分（positivity bias），且对评分尺度不敏感（3 分和 4 分的区别不如"A 比 B 好"直观）。适合快速原型验证，生产环境需要更结构化的方法。

### Pairwise LLM Comparison

$$
P(d_i \succ d_j) = \text{LLM}(\text{"以下两篇文档哪个更相关？"} + q + d_i + d_j)
$$

**直观理解**：让 LLM 做两两比较而非绝对打分——更符合 LLM 的能力特点（判断相对好坏比打绝对分数稳定）。缺点是 $O(n^2)$ 次调用，优化方案：先用传统 reranker 粗排 top-20，再对 top-20 做 LLM pairwise rerank。

### Setwise LLM Reranking（折中方案）

$$
\text{best}(d_1, ..., d_m) = \text{LLM}(\text{"以下} m \text{篇文档中最相关的是？"} + q + d_{1:m})
$$

**直观理解**：每次给 LLM 看 $m$ 篇文档（如 5 篇），选最好的一篇。比 pairwise 快（$O(n/m)$ 次调用），比 listwise 稳定（context window 压力小）。通过 heap sort 策略可以 $O(n \log n / m)$ 次调用完成排序。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
