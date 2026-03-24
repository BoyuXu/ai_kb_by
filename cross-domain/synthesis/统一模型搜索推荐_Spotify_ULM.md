# 统一模型做搜索+推荐：Spotify ULM 的工程哲学

> 📚 参考文献
> - [Counterfactual-Learning-For-Unbiased-Ad-Ranking...](../../ads/papers/Counterfactual_Learning_for_Unbiased_Ad_Ranking_in_Indust.md) — Counterfactual Learning for Unbiased Ad Ranking in Indust...
> - [Spotify Unified Lm Search Rec](../../rec-sys/papers/A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/Linear_Item_Item_Model_with_Neural_Knowledge_for_Session.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/Query_as_Anchor_Scenario_Adaptive_User_Representation_via.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Intent-Aware-Neural-Query-Reformulation-For-Beh...](../../search/papers/Intent_Aware_Neural_Query_Reformulation_for_Behavior_Alig.md) — Intent-Aware Neural Query Reformulation for Behavior-Alig...
> - [Generative Query Expansion For E-Commerce Search A](../../search/papers/Generative_Query_Expansion_for_E_Commerce_Search_at_Scale.md) — Generative Query Expansion for E-Commerce Search at Scale


**一句话**：搜索和推荐本质上都是「给定用户意图，找最相关内容」——用一个模型同时服务两者，知识可以互相迁移，参数量减少 60%，效果反而更好。

**类比**：一个公司的销售培训。以前搜索团队培训「如何回答精确问题」，推荐团队培训「如何猜测客户需求」，各练各的。统一培训就是：两组学员一起上课，训练「理解客户意图」——搜索客户的精确意图能让推荐更准，推荐的行为理解能让搜索更懂上下文。


## 📐 核心公式与原理

### 1. 多目标优化
$$\min_{\theta} \sum_k \lambda_k L_k(\theta)$$
- Scalarization 方法，λ 控制任务权重

### 2. Pareto 最优
$$x^* \text{ is Pareto optimal } \iff \nexists x: f_i(x) \leq f_i(x^*) \forall i$$
- 不存在在所有目标上都更好的解

### 3. 偏差校正 (IPW)
$$\hat{R} = \frac{1}{n}\sum_i \frac{r_i}{P(O=1|x_i)}$$
- 逆倾向加权消除选择偏差

---

## Spotify ULM 设计哲学

### 为什么三任务可以统一？

| 任务 | 输入 | 输出 | 共性 |
|------|------|------|------|
| 搜索 | query text | 相关 item | 用户意图 → 内容 |
| 推荐 | 用户历史序列 | next item | 用户行为 → 内容 |
| 内容推理 | 自然语言问题 | 答案/item | 语义理解 → 内容 |

**共性**：都需要理解用户意图（显式 query 或隐式行为），都需要内容语义理解（item 是什么）。底层语义表示可以共享。

### 关键设计决策

**1. 统一文本序列格式**
```
搜索: [SEARCH] query: "好的跑步播客" → item_1, item_2, ...
推荐: [REC] history: <pod_1> <pod_2> <pod_3> → next_pod
推理: [REASON] question: "今天跑10km配什么音乐?" → playlist
```
任务前缀 token 让模型切换任务，共享底层理解能力。

**2. 共享 backbone，独立轻量 head**
```
共享 Transformer (1B params)
    ├── 搜索 head (5M params)
    ├── 推荐 head (5M params)  
    └── 推理 head (5M params)
```
总参数：1B + 15M（vs 原来 3.6B 独立模型）→ 节省 61%

**3. 异构数据采样策略**
- 搜索日志：推荐日志：播客文本 = 3:2:1
- 不按数据量等比采样（搜索日志 >> 推荐日志）
- 防止高频任务「淹没」低频任务的梯度

---

## 工程落地核心挑战（论文没讲够的）

| 挑战 | 具体问题 | 解决方案 |
|------|---------|---------|
| 版本管理 | 三个系统共用一个模型，任何需求变化都触发重训 | Task head 级别灰度发布，head 可独立回滚 |
| 延迟 SLA 隔离 | 搜索需要 <100ms，推理可以慢一些 | 同一 ULM 服务，按任务类型设不同超时 |
| 特征对齐 | 歌名 vs 播客标题 format 不同 | 统一 normalization pipeline（小写/截断/去符号） |
| 增量学习 | 新功能（如 Audiobook）上线 | Adapter 层微调，不重训 backbone |
| 多语言 | 葡语/印尼语搜索量少，独立模型效果差 | ULM 跨语言迁移，低资源语言 +9.7% |

---

## 迁移学习的隐藏价值

**最重要的收益**：不是效率，而是**互相增强**。

- 搜索数据（query → 精确 item）教会模型：item 描述的关键词模式
- 推荐数据（用户序列 → item）教会模型：用户偏好的长期规律  
- 推理数据（自然语言 → 内容）教会模型：深层语义理解

这三类知识在统一模型里互相迁移：搜索因为懂了用户行为变得更准，推荐因为懂了精确意图变得更相关。

---

## 技术演进脉络

```
2018 各系统独立 Embedding（搜索用 BERT，推荐用 CF）
    ↓ 维护成本高，知识孤岛
2020 多任务学习（Hard/Soft Parameter Sharing）
    ↓ 任务数多时 sharing 不够灵活
2022 Prompt-based 统一（T5 格式统一多任务）
    ↓ LLM 规模化
2024 ULM（Spotify）：生产级统一搜推推理
    ↓ 方向
2025+ Agent + 搜推统一（用工具调用搜索，推荐用 LLM 推理）
```

---

## 面试考点

**Q：搜索和推荐统一建模的最大挑战是什么？**  
答：（1）数据不平衡：搜索日志远比推荐日志多，需要精心设计采样策略防止推荐任务欠优化；（2）延迟 SLA 不同：搜索要求实时 <100ms，推荐 <200ms，统一模型需要隔离不同任务的服务超时；（3）负迁移：某些任务数据分布差异大时，联合训练反而会互相干扰，需要监控各任务的独立指标。

**Q：为什么 Spotify ULM 在低资源语言上收益更大？**  
答：独立搜索模型在葡语/印尼语等低资源语言上训练数据少，模型质量差。ULM 的预训练包含跨语言内容（多语言播客文本），底层语义表示具有跨语言对齐能力，迁移到低资源语言搜索效果更好。本质是 transfer learning 的经典优势。

### Q1: 搜广推三个领域的技术共性？
**30秒答案**：①都需要召回+排序架构；②都用 CTR/CVR 预估模型；③都面临冷启动问题；④都需要实时特征系统；⑤都可以用 LLM 增强。差异主要在约束条件和评估指标。

### Q2: 多目标优化在三个领域的应用？
**30秒答案**：广告：收入+用户体验+广告主 ROI；推荐：CTR+时长+多样性+留存；搜索：相关性+新鲜度+权威性+多样性。方法共通：Pareto/MMoE/PLE/Scalarization。

### Q3: 偏差问题在三个领域的表现？
**30秒答案**：广告：位置偏差+样本选择偏差；推荐：流行度偏差+曝光偏差；搜索：位置偏差+呈现偏差。解决方法类似：IPW/因果推断/去偏训练。

### Q4: 端到端学习的趋势和挑战？
**30秒答案**：趋势：统一模型替代分层管道（OneRec 统一召排）。挑战：①推理效率（一个大模型 vs 多个小模型）；②可控性差（难以插入业务规则）；③调试困难（黑盒）。

### Q5: 面试中如何体现跨领域理解？
**30秒答案**：①用类比说明（如广告出价≈搜索 LTR）；②指出技术迁移（如 DIN 从推荐到广告）；③提出统一视角（如多目标在三领域的共通框架）；④结合实际经验说明如何借鉴。

### Q6: 如何向面试官展示技术深度？
**30秒答案**：①先总后分：先说整体架构，追问时展开细节；②对比分析：主动比较 2-3 种方案的优劣；③数字说话：「AUC 从 0.72 提升到 0.74」而非「效果变好了」；④边界意识：说清楚方案的局限和适用条件。

### Q7: 跨领域知识迁移的实际案例？
**30秒答案**：①DIN（推荐→广告）：注意力机制从推荐 CTR 迁移到广告 CTR；②BERT（NLP→搜索）：预训练语言模型用于搜索排序；③Semantic ID（搜索→推荐）：从搜索的 doc ID 到推荐的 item ID 统一表示。

### Q8: 大规模系统的性能优化通用方法？
**30秒答案**：①缓存（特征缓存、结果缓存）；②异步（特征获取异步化）；③预计算（user embedding 离线算好）；④分层（粗排+精排降低计算量）；⑤模型优化（蒸馏/量化/剪枝）。

### Q9: 线上事故排查的思路？
**30秒答案**：①看监控：指标异常时间点→②查变更：最近上线了什么→③回滚验证：回滚后指标恢复说明是变更导致→④深入分析：看特征分布、样本分布、模型输出分布有无异常。

### Q10: 算法工程师的核心竞争力？
**30秒答案**：①业务理解（指标 → 技术方案的转化能力）；②工程能力（模型能上线、能调优、能排查问题）；③论文能力（快速读懂并判断论文的实用价值）；④系统思维（全链路优化而非单点优化）。
