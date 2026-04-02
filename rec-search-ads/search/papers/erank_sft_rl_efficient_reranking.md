# ERank: Fusing SFT and RL for Efficient Reasoning Reranking
> 来源：https://arxiv.org/abs/2509.00520 | 领域：search | 学习日期：20260401

## 问题定义

LLM-based 重排方法面临一个根本性的两难困境（dilemma）：

### 困境1：Pointwise SFT 方法
- **做法**：将相关性判断视为二分类任务（Relevant/Not Relevant），逐文档独立打分
- **优点**：推理效率高，延迟低（O(n)）
- **缺点**：
  - **判别力不足**：二元标签无法区分"非常相关"vs"有点相关"
  - **全局排序感知缺失**：独立打分，不考虑文档间相对关系
  - 基于推理 LLM 构建时，困境更严重（推理能力与二分类任务不匹配）

### 困境2：Listwise 推理方法（如 Rank1、REARANK）
- **做法**：整体处理候选文档列表，生成完整排序
- **优点**：考虑文档间关系，准确度高
- **缺点**：
  - **效率问题**：O(n²) 或更高的复杂度
  - **延迟不可接受**：生产环境中难以部署
  - 对低延迟应用（移动搜索、实时推荐）完全不适用

**ERank** 的目标：打破这个困境，构建一个**高效的 pointwise 架构**，同时具备**listwise 的排序感知能力**。

关键结果：
- ERank-4B: BRIGHT nDCG@10 = **38.7**
- ERank-32B: BRIGHT nDCG@10 = **40.2**（新 SOTA）
- 在 FollowIR、TREC DL、BEIR 上全面领先

## 核心方法与创新点

### 两阶段训练 Pipeline

**阶段1：SFT with Fine-grained Integer Scores（细粒度整数评分监督微调）**

不同于传统二分类（0/1），ERank 在 SFT 阶段训练模型输出**细粒度整数相关性分数**：

```
传统 SFT：
输入：[Query] + [Document]
输出：Relevant / Not Relevant

ERank SFT：
输入：[Query] + [Document]  
输出：Score ∈ {0, 1, 2, 3, 4, 5}
（0=完全不相关，5=高度相关且完整回答查询）
```

**训练目标**（生成式，非分类）：

$$
\mathcal{L}_{SFT} = -\sum_t \log P(score_t | query, doc, score_{<t})
$$

通过生成式输出整数分数，推理 LLM 的 Chain-of-Thought 能力被充分利用，模型可以"解释"为什么给出该分数，从而提升判别力。

**SFT 数据构建**：
- 从 TREC、MSMARCO、BEIR 等数据集中提取分级相关性标注
- 将 TREC 的 4 级（0/1/2/3）映射到 6 级整数分数
- 使用 GPT-4 对边界案例进行重标注

**阶段2：RL with Listwise-Derived Reward（列表感知奖励强化学习）**

这是 ERank 的核心创新——**将 listwise 的全局排序感知注入 pointwise 架构**：

```
训练时：
  对同一 query 的 k 个候选文档同时进行 pointwise 打分
  用打分结果生成排序 π
  计算 listwise 奖励 R = nDCG@10(π, π*)
  
推理时：
  仍然 pointwise 逐文档打分（高效）
  但由于 RL 训练时见过文档间排序关系，打分具备全局一致性
```

**奖励函数**：

$$
R_{listwise} = \text{nDCG@10}\left(\text{rank}(S_1, S_2, ..., S_k), \pi^*\right)
$$

其中 $S_i$ 是第 $i$ 个文档的 pointwise 分数，$\pi^*$ 是金标准排序。

**RL 算法**：GRPO（Group Relative Policy Optimization）
- 对同一 query 采样多个排序结果
- 用相对奖励更新策略（减少方差）

### 架构图

```
[查询 Q] + [文档 Di]
        ↓
   推理 LLM
   (先思考相关性)
        ↓
  生成整数分数 Si ∈ {0,1,2,3,4,5}
        ↓
  ← pointwise（高效）
  ↑ 训练时有 listwise reward（排序感知）
  
最终排序：按 Si 降序排列所有候选文档
```

### 与其他方法的对比

| 方法 | 推理范式 | 推理时效率 | 排序感知 | BRIGHT nDCG@10 |
|------|---------|-----------|---------|---------------|
| MonoT5 | Pointwise-SFT | O(n) ✅ | 无 | ~20 |
| Rank1-7B | Listwise | O(n·L) ⚠️ | 有 | ~32 |
| REARANK-7B | Listwise | O(n·L) ⚠️ | 有 | ~30 |
| **ERank-4B** | **Pointwise-RL** | **O(n) ✅** | **有** | **38.7** |
| **ERank-32B** | **Pointwise-RL** | **O(n) ✅** | **有** | **40.2** |

（L = 推理链长度，通常 500-2000 tokens）

## 实验结论

### 全面基准评测

**BRIGHT（推理密集型，最具挑战性）**：
- ERank-4B: 38.7 → 超越所有 7B-13B 量级模型
- ERank-32B: 40.2 → **新 SOTA**（超越 GPT-4o、Claude-3.5）

**FollowIR（指令跟随）**：
- 由于细粒度打分和 RL 训练，对指令的理解能力显著提升
- ERank 在 FollowIR 的 p-MRR 领先 RankZephyr 约 8%

**TREC DL 19/20（标准段落重排）**：
- ERank nDCG@10 ≈ 75+，与最佳 listwise 模型持平

**BEIR（泛化能力）**：
- 平均 nDCG@10 超越 MonoT5 约 5%，验证泛化性

### 消融实验关键结论

| 变体 | BRIGHT nDCG@10 | 分析 |
|------|---------------|------|
| 仅 SFT（二分类） | 29.3 | 基础 SFT 困境 |
| 仅 SFT（整数分数） | 33.1 | 细粒度分数有效 |
| SFT + RL（pointwise reward） | 35.8 | RL 有效但不够 |
| **SFT + RL（listwise reward）** | **38.7** | **最优** |

结论：**listwise reward 是关键**，单纯 RL 或单纯 SFT 都不足以获得 SOTA 性能。

## 工程落地要点

### 部署优势分析

**ERank vs Listwise 方法的延迟对比（Top-20 重排）**：

```
MonoT5-3B (pointwise SFT):  ~20ms
Rank1-7B (listwise + CoT):  ~800ms
REARANK-7B (listwise):      ~600ms  
ERank-4B (pointwise + RL):  ~80ms    ← 5-10x 优于 listwise
ERank-32B:                  ~250ms
```

对实时搜索系统，ERank-4B 的 ~80ms 延迟完全可接受。

### 服务架构

```
L1: 初始检索 (ANN) → Top-200
L2: 轻量粗排 (BM25 + 规则) → Top-50
L3: ERank-4B pointwise 精排 → Top-10
L4: 返回结果（可选：附带相关性得分用于 RAG）
```

### 训练数据准备

```python
# ERank 训练数据格式
{
  "query": "explain transformer attention mechanism",
  "passages": [
    {"text": "...", "relevance_score": 4},  # 4分：高度相关
    {"text": "...", "relevance_score": 2},  # 2分：部分相关
    {"text": "...", "relevance_score": 0},  # 0分：不相关
  ]
}
# 关键：需要分级标注（≥4级），不能只有二分类
```

**数据来源**：
- TREC 系列（天然有分级标注）
- MSMARCO v2（扩展版，有细粒度标注）
- 内部标注（需要专门培训标注员理解细粒度相关性）

### RL 训练技巧

1. **SFT 热身**：先用整数分数 SFT 训练至少 1 epoch，提供好的 policy 初始点
2. **GRPO vs PPO**：GRPO 在小批量数据上更稳定，推荐用 GRPO
3. **Listwise 构建**：每个 query 采样 4-8 个文档组成 listwise 批次
4. **奖励归一化**：不同 query 的 nDCG 方差不同，需做 batch 内归一化

### 细粒度分数的工程价值

整数分数（0-5）在工程上有额外价值：
- **多路融合**：不同召回路的分数可线性加权合并
- **过滤阈值**：分数 < 2 的文档可直接过滤，节省下游 LLM 推理成本
- **调试可视性**：相比二分类，细粒度分数更容易分析错误

## 常见考点

**Q1：ERank 如何解决 pointwise 重排的"无全局排序感知"问题？**
A：通过 RL 训练时使用 listwise 奖励（nDCG）。虽然推理时仍然 pointwise（每个文档独立打分），但训练过程中模型同时看到多个文档的相对关系并优化排序质量，使得学到的打分函数具有全局一致性。

**Q2：为什么用生成式输出整数分数，而不是简单分类器？**
A：推理 LLM 的优势在于 Chain-of-Thought 推理。生成式输出整数分数时，模型可以先"思考"文档与查询的关联，再给出细粒度分数，充分利用推理能力。分类器头会截断这个推理过程。

**Q3：ERank 的两阶段训练各自解决什么问题？**
A：SFT 阶段用细粒度整数标注解决"判别力不足"（binary SFT 太粗糙）；RL 阶段用 listwise reward 解决"全局排序感知缺失"（pointwise 独立打分忽略文档间关系）。两者组合，用 pointwise 架构的效率获得 listwise 方法的排序质量。

**Q4：ERank 的主要竞争优势是什么？适合什么场景？**
A：核心优势是**高效 + 高性能**：O(n) pointwise 推理延迟（~80ms for 4B），同时 BRIGHT nDCG@10 达到 40.2（SOTA）。适合所有需要低延迟的实时重排场景，是目前工业部署性价比最高的推理重排方案。

**Q5：如何构建 ERank 所需的细粒度相关性标注数据？**
A：可以从三个来源获取：(1) TREC 系列数据集（天然 4 级标注）；(2) 用强 LLM（GPT-4）自动标注（生成 1-5 分 + 理由，再人工抽查）；(3) 将 clickthrough 数据转化为弱监督分级标注（点击 + 停留时长 → 分级分数）。
