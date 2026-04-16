# LLM 驱动广告系统：从增强到原生

> 创建：2026-04-13 | 领域：广告系统 | 类型：综合分析
> 来源文件：
> - `ads/synthesis/20260411_LLM驱动推荐推理_生成式召回_工业基础设施.md`
> - `ads/synthesis/20260411_llm_native_recsys_and_industrial_infra.md`
> - `ads/synthesis/工业广告系统生成式革命_20260403.md`

---

## 一、LLM 在广告系统中的定位演变

```
第一阶段（2023）：LLM 作为辅助 → 特征增强（语义 embedding）
第二阶段（2024）：LLM 作为模块 → 重排/创意生成
第三阶段（2025）：LLM 原生 → 统一建模（排序+推理+生成一体）
第四阶段（2026+）：LLM Agent → 自主投放决策
```

### 核心洞察

LLM 不是替代传统广告模型，而是在不同层次融合：
- **底层**：语义特征增强（LEARN, DGenCTR）
- **排序层**：推理增强排序（OneRanker, GRR）
- **召回层**：生成式召回（TBGRecall, R2ec）
- **决策层**：Agent 化自主出价（RTBAgent）

---

## 二、LLM 特征增强

### 2.1 LEARN（知识适配）

$$
\hat{y} = f_{\text{CTR}}(e_{\text{user}}, e_{\text{item}}, \text{LLM}_{\text{Emb}}(\text{title, desc}))
$$

用 LLM 提取广告标题/描述的语义 embedding 作为额外特征。对新广告（冷启动）效果显著——LLM 能理解"限时特惠"和"促销优惠"是同义。

### 2.2 DGenCTR（生成式 CTR）

将 CTR 预估重构为文本生成任务：

$$
P(\text{click}) = P_{\text{LLM}}(\text{"yes"} \mid \text{prompt}(u, i, \text{context}))
$$

优势：利用 LLM 的世界知识理解广告语义。劣势：推理延迟高，工业上常用蒸馏后的小模型。

### 2.3 CADET（因果去偏 + LLM 增强）

LLM 提供因果先验知识，识别广告特征间的因果关系（如"价格降低→CVR 提升"不等于"标注降价→CVR 提升"），辅助去偏。

---

## 三、LLM 推理增强排序

### 3.1 OneRanker

统一排序框架：一个模型同时处理多种排序任务（广告排序、搜索排序、推荐排序）。

$$
\text{score} = \text{OneRanker}(\text{query/context}, \text{candidates}, \text{taskPrompt})
$$

通过 task prompt 区分不同排序场景，共享底层表示学习。

### 3.2 GRR（Generative Reasoning Re-ranker）

$$
P(\text{rank} \mid q, u, \mathcal{C}) = P(\text{reasoning} \mid q, u) \cdot P(\text{rank} \mid \text{reasoning}, \mathcal{C})
$$

显式推理链（CoT）迫使模型先理解用户意图再排序。训练数据：GPT-4 蒸馏生成推理链 → 小模型学习。

### 3.3 ThinkRec（推荐推理）

将推荐决策分解为显式推理步骤：

```
Step 1: 用户意图识别 → "用户最近在看运动鞋，可能有运动需求"
Step 2: 候选评估 → "广告A是运动鞋促销，匹配度高"
Step 3: 排序决策 → "广告A > 广告B（品牌广告，弱相关）"
```

推理链提升可解释性，便于 debug 和优化。

---

## 四、生成式召回

### 4.1 TBGRecall（Token-Based Generative Recall）

将广告 ID 编码为语义 token 序列（Semantic ID），用自回归模型逐 token 生成：

$$
P(\text{ad}) = \prod_{t=1}^{T} P(\text{token}_t \mid \text{token}_{<t}, \text{userContext})
$$

优势：无需维护显式候选集索引，新广告自然融入生成空间。

### 4.2 R2ec（Retrieval-Augmented Recommendation）

检索增强推荐：先检索相关广告作为上下文，再用 LLM 精排：

$$
\text{candidates} = \text{Retrieve}(\text{user}, \text{context}) \quad \rightarrow \quad \text{rank} = \text{LLM}(\text{candidates}, \text{user})
$$

结合传统召回的效率和 LLM 的语义理解能力。

### 4.3 QuaSID（量化语义 ID）

用向量量化（VQ）将广告 embedding 离散化为语义 ID：

$$
\text{SID} = \text{VQ}(e_{\text{ad}}) = [c_1, c_2, ..., c_L]
$$

关键改进：RQ-VAE → RQ-GMM（GMM 软赋值量化），提升 codebook 利用率和语义判别性。

---

## 五、生成式革命覆盖全链路

### 5.1 生成式出价

将出价策略建模为条件生成：

$$
\text{bid} \sim P_\theta(\text{bid} \mid s_t, \text{constraints})
$$

用 Diffusion Model 生成出价分布，捕捉多模态最优策略。

### 5.2 生成式创意

LLM 驱动的创意生产：

```
广告主 Brief → [LLM Creative Director] → 策略
    → [LLM Copywriter] → 文案（10-50 条候选）
    → [多模态模型] → 配图
    → [CTR 预估模型] → 离线筛选
    → [Thompson Sampling] → 在线收敛到最优组合
```

### 5.3 生成式 Foundation Model

OneRec 类统一框架：一个模型同时处理召回、排序、重排：

$$
\text{output} = \text{FoundationModel}(\text{user}, \text{context}, \text{taskToken})
$$

task\_token 切换任务模式（召回/排序/解释）。

---

## 六、LLM-Native vs LLM-Augmented 对比

| 维度 | LLM-Augmented | LLM-Native |
|------|--------------|------------|
| LLM 角色 | 特征提取器/辅助模块 | 核心排序/生成引擎 |
| 传统模型 | 仍是主体 | 被替代或简化 |
| 推理延迟 | 低（LLM 离线预计算） | 高（在线推理） |
| 协同过滤 | 传统模型提供 | 需要 Semantic ID 编码 |
| 冷启动 | LLM 语义缓解 | 天然优势（语义理解） |
| 成熟度 | 已大规模落地 | 部分落地，快速发展 |

### 工业落地建议

1. **短期**（2025）：LLM-Augmented 为主，语义特征增强 + LLM 重排
2. **中期**（2026）：混合架构，传统召回 + LLM 精排/重排
3. **长期**（2027+）：LLM-Native 统一模型，端到端优化

---

## 七、LLM 出价 Agent

### 7.1 RTBAgent

LLM 作为实时竞价决策 Agent：

```
输入：竞价请求（用户特征、广告信息、预算状态）
LLM CoT：
  1. "剩余预算 60%，时间过半，消耗节奏正常"
  2. "该用户历史高转化，CTR 预估 0.08"
  3. "竞争激烈度中等，建议出价 1.2x base"
输出：bid = 1.2 × base_bid
```

### 7.2 工程挑战

- **延迟**：LLM 推理 >100ms，不满足实时竞价要求
- **解决方案**："离线策略 + 在线执行"两阶段架构
  - 离线：LLM 生成策略 lookup table（用户群×广告类型×预算状态→出价倍率）
  - 在线：查表执行，延迟 <5ms

---

## 八、面试高频 Q&A（12 题）

**Q1: LLM 在广告系统中的三种使用方式？**
特征增强（语义 embedding 作为辅助特征）、排序增强（CoT 推理链排序）、原生替代（端到端生成式排序/召回）。

**Q2: DGenCTR 的优劣？**
优势：利用 LLM 世界知识理解广告语义。劣势：推理延迟高（>100ms），工业上需蒸馏到小模型。

**Q3: Semantic ID 如何解决生成式召回的 ID 空间问题？**
用 VQ 将广告 embedding 离散化为 token 序列，自回归模型逐 token 生成。RQ-GMM 比 RQ-VAE 的 codebook 利用率更高。

**Q4: GRR 的推理链训练数据怎么获取？**
GPT-4 蒸馏（大模型生成推理链→小模型学习）、规则+弱监督生成+RL 过滤、结构化解释的程序生成。

**Q5: LLM-Native 推荐的最大挑战？**
推理延迟（>100ms vs 广告要求 <50ms 精排）和协同过滤信号缺失（LLM 只理解语义，不理解行为统计）。

**Q6: RTBAgent 的延迟问题怎么解决？**
离线策略生成（LLM 生成 lookup table）+ 在线查表执行（<5ms）。本质是将 LLM 的推理能力"缓存"到高效数据结构中。

**Q7: 生成式排序 vs 判别式排序的工业选择？**
判别式做粗排（效率高），生成式做精排/重排（质量高）。两者互补而非替代。

**Q8: R2ec 检索增强推荐的优势？**
结合传统召回的效率（毫秒级 ANN 检索）和 LLM 的语义理解（精排质量）。比纯 LLM 推荐延迟低，比纯传统模型语义理解强。

**Q9: LLM 创意生成的质量控制？**
CTR 预估模型离线评分筛选 + 合规审核（NSFW/竞品词/资质检查）+ 线上 A/B 测试（Thompson Sampling 收敛最优）。

**Q10: Foundation Model 统一推荐广告的挑战？**
目标冲突（用户满意度 vs 商业收入）、出价引入分布偏移（高出价广告偏多）、需要用户体验保护机制（兴趣阈值+混排比例）。

**Q11: LEARN 知识适配如何避免 LLM 特征"覆盖"协同过滤信号？**
LLM embedding 作为额外特征（concatenate/gate），不替代 ID embedding。门控网络动态学习 LLM 特征和 ID 特征的权重。

**Q12: 生成式出价（Diffusion Model）比 RL 出价好在哪？**
输出出价分布而非单点值，捕捉多模态最优策略。可同时考虑多约束下的 Pareto 最优解集。但训练和推理成本更高。

---

## 参考文献

1. LEARN: Knowledge Adaptation from LLM to Recommendation (2024)
2. DGenCTR: Generative CTR Prediction (2025)
3. OneRanker: Unified Ranking Framework (2025)
4. GRR: Generative Reasoning Re-ranker (2025)
5. ThinkRec: In-Text Reasoning for Recommendation (2025)
6. TBGRecall: Token-Based Generative Recall (2025)
7. R2ec: Retrieval-Augmented Recommendation (2025)
8. RTBAgent: LLM as Real-Time Bidding Agent (2025)
9. One Model Two Markets (2025)
10. 工业广告系统生成式革命综合 (2026)

---

## 相关概念

- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[synthesis/ads/01_CTR_CVR预估与校准全景|CTR/CVR 预估与校准]]
- [[synthesis/ads/02_广告排序系统演进|广告排序系统演进]]
- [[synthesis/ads/08_广告创意优化|广告创意优化]]
