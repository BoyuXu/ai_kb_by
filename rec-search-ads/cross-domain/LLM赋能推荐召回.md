# 知识卡片 #009：LLM 赋能推荐召回——从多路专家到通用检索

> 📚 参考文献
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/Dense_Retrieval_vs_Sparse_Retrieval_A_Unified_Evaluation.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/Multi_Objective_Optimization_for_Online_Advertising_Balan.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/Linear_Item_Item_Model_with_Neural_Knowledge_for_Session.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Llm-Universal-Retriever](../../rec-sys/papers/Large_Language_Model_as_Universal_Retriever_in_Industrial.md) — Large Language Model as Universal Retriever in Industrial...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Web-Scale-Llm-Recsys](../../rec-sys/papers/Towards_Web_scale_Recommendations_with_LLMs_From_Quality.md) — Towards Web-scale Recommendations with LLMs: From Quality...
> - [Dense Passage Retrieval For Open-Domain Questio...](../../search/papers/Dense_Passage_Retrieval_for_Open_Domain_Question_Answerin.md) — Dense Passage Retrieval for Open-Domain Question Answerin...

> 创建：2026-03-20 | 领域：推荐系统·召回 | 难度：⭐⭐⭐⭐
> 来源：LLM Universal Retriever (2502.03041，广告平台)、Web-scale LLM Rec (3690624，Bing)

## 📐 核心公式与原理

### 1. 多目标优化

$$
\min_{\theta} \sum_k \lambda_k L_k(\theta)
$$

- Scalarization 方法，λ 控制任务权重

### 2. Pareto 最优

$$
x^* \text{ is Pareto optimal } \iff \nexists x: f_i(x) \leq f_i(x^*) \forall i
$$

- 不存在在所有目标上都更好的解

### 3. 偏差校正 (IPW)

$$
\hat{R} = \frac{1}{n}\sum_i \frac{r_i}{P(O=1|x_i)}
$$

- 逆倾向加权消除选择偏差

---

## 🌟 一句话解释

传统推荐召回需要多个专家模型分别处理不同目标（相关性、多样性、新颖度），**LLM 通用召回器（URM）用单一 LLM + 生成式框架统一替代，同时引入 LLM 生成全新候选（Q' Recall），找到传统召回"永远找不到"的好内容**。

---

## 🎭 生活类比

**传统多路召回** = 公司里每类招聘需求都有专职 HR（协同过滤召回、向量召回、热门召回）：职责清晰但人手多、沟通成本高。

**URM** = 一个超级 HR，精通所有招聘类型，且能根据任意 JD 自适应调整策略。

**Q' Recall（LLM 生成候选）** = 不只靠简历库（历史数据）找人，还能主动挖掘"潜力股"——那些简历库里根本没有的候选人，相当于猎头主动出击。

---

## ⚙️ 技术演进脉络

```
【时代一：规则召回（Popularity / Category）】
  热门内容 + 类目匹配 → 覆盖面差，个性化不足

【时代二：协同过滤（CF）+ 矩阵分解（MF）】
  用户行为相似度 → 依赖稀疏度，冷启动弱

【时代三：双塔 Dense Retrieval（DSSM / FAISS）】
  用户/物品 embedding 近似最近邻
  → 语义丰富，但多目标需要多套模型

【时代四：多路召回（工业标准）】
  CF + Dense + 热门 + 实时 + 规则 并行
  → 召回丰富，但路数多，维护成本高

【时代五：LLM 赋能召回（2024-2025）】
  方向一（URM）：单 LLM 统一替代多专家，生成式框架
  方向二（Q' Recall / Bing）：LLM 生成全新候选补充传统召回
  → 减少模型维护成本 + 召回多样性质的提升
```

---

## 🔬 两种 LLM 召回范式对比

### 范式一：URM（替代专家）

```
用户历史行为 + 检索目标
    ↓
LLM（多查询表示 + 矩阵分解）
    ↓
生成候选 item ID 序列
    ↓
概率采样（控制计算成本）
    ↓
候选集（数百 → 下游排序）
```

**关键技术**：
- 多查询表示：一个 user context 生成多个查询向量，覆盖多个意图维度
- 矩阵分解：处理亿级候选集，降低 LLM 生成难度
- 概率采样：平衡探索与效率，满足数十毫秒延迟

### 范式二：Q' Recall（补充专家）

```
传统多路召回（CF、Dense、热门...）  +  LLM 生成召回
          ↓                              ↓
       候选集 A                       候选集 B（新颖、互补）
                    ↓ 合并
              粗排 → 精排
```

**优势**：LLM 生成能发现"语义互补但表面不相似"的内容，增加惊喜度

---

## 🏭 工业落地关键

| 挑战 | 解决方案 |
|------|---------|
| LLM 推理延迟（100ms+） | 矩阵分解 + 概率采样压缩到数十 ms；蒸馏轻量模型 |
| 亿级候选集 | 生成式检索 + 分层索引；不直接生成 ID 而是生成特征向量 |
| LLM 输出不稳定 | 后处理校验 + 回退到传统召回 |
| 多目标切换 | Instruction-tuning 使 LLM 理解不同检索目标描述 |
| 线上 A/B 实验 | Q' Recall 作为独立召回路，不影响现有路，便于隔离实验 |

---

## 🆚 和已有知识的对比

**LLM 召回 vs 双塔召回**：
- 双塔：离线训练 embedding，online ANN 检索，延迟极低（<5ms）
- LLM 召回：online 生成，延迟高，但语义理解和任务泛化更强
- 实践：LLM 召回作为**补充路**而非替代，保留双塔作为主路

**URM vs 多任务双塔（MMOE 召回）**：
- MMOE 召回：共享底层 + 多任务 head，仍是表示学习范式
- URM：生成式，自然语言描述目标 → 更灵活，新目标无需重训

---

## 🎯 常见考点

**Q1：生成式检索和向量检索的核心区别是什么？**
A：向量检索将物品表示为固定向量，通过 ANN 检索语义相近内容（检索已有）。生成式检索将检索建模为序列生成问题，LLM 直接"生成"候选 ID 或特征，可以生成训练数据中不常见的组合（创造可能）。

**Q2：URM 如何处理亿级候选集？**
A：通过矩阵分解降低候选空间维度，将 item ID 空间分解为多层级的离散编码（类似 RQ-VAE），LLM 逐级生成编码，最终映射回 item ID。这使生成空间从亿级降到数百级别。

**Q3：Q' Recall 相比传统召回的核心增量价值在哪里？**
A：传统召回基于行为共现或语义相似，容易产生同质化推荐（用户历史 → 相似历史）。Q' Recall 利用 LLM 的世界知识生成"互补性"候选，例如看完《哈利波特》可能推荐《魔法师的外甥》（叙事互补）而非只推荐同系列书。Bing 的实验显示核心指标 +3%。

**Q4：LLM 推理延迟是召回阶段的最大挑战，工业界有哪些应对策略？**
A：① 模型蒸馏：将 70B LLM 蒸馏到 7B 专用召回模型；② 推测解码：用小模型草稿 + 大模型校验；③ 异步预计算：部分用户特征可提前 LLM 处理并缓存；④ 批处理：多用户共享一次 LLM forward；⑤ 量化：INT4/INT8 加速推理。

---

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
