# 推荐系统 + LLM 集成框架

> 📚 参考文献
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/20260321_deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/20260321_gems-breaking-the-long-sequence-barrier-in-generative-recommendation-with-a-multi-stream-decoder.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Interplay-Training-Independent-Simulators-For-R...](../../rec-sys/papers/20260321_interplay-training-independent-simulators-for-reference-free-conversational-recommendation.md) — Interplay: Training Independent Simulators for Reference-...
> - [Spotify Unified Language Model](../../rec-sys/papers/20260322_spotify_unified_language_model.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Cobra Bridging Sparse And Dense Retrieval In Gene](../../rec-sys/papers/20260323_cobra_bridging_sparse_and_dense_retrieval_in_gene.md) — COBRA: Bridging Sparse and Dense Retrieval in Generative ...


## 📚 参考资料与引用

本文是对 LLM 在推荐系统中应用的总结提炼，引用了以下研究与实践案例：

### 核心论文
- [LLM 在推荐系统中的应用](../papers/) - 待补充具体论文链接

### 推荐系统基础
- 协同过滤与矩阵分解论文
- 深度推荐模型论文
- 多目标排序论文

### LLM 集成
- 多智能体推荐系统
- LLM 作为排序器
- 对话推荐系统

---

## 文档概览

本文档按"替换层 → 节点层 → 架构层"三个维度，系统地梳理 LLM 在推荐系统中的应用方向。

**适用场景**：电商推荐、内容推荐、视频推荐、社交推荐  
**技术栈**：RAG、用户偏好文本化、对话推荐、LLM reranker  
**时间范围**：2023-2026 学术进展 + 业界实践

---

## 第一层：替换层（内容理解与表达生成）

**定义**：用 LLM 替换推荐系统中的**文本处理和理解模块**。  
**特点**：低成本、快速落地、提升用户体验  
**ROI**：中等（留存 +5-10%，点击率 +3-8%）

### 1.1 用户偏好描述生成 (User Preference Representation)

**问题**：
- 传统推荐基于隐式特征（embedding）：用户无法理解为什么被推荐
- 新用户冷启动时，系统对用户偏好一无所知
- 跨域迁移困难（从电商到视频的用户兴趣映射）

**LLM 解决方案**：

#### 显式偏好描述
- **从隐式行为 → 显式文本**：
  - 传统方式：用户 A 的 embedding = [0.2, 0.5, 0.1, ...]
  - LLM 转换：
    ```
    用户 A 的偏好：
    - 主要兴趣：科幻电影、悬疑小说
    - 风格偏好：烧脑、剧情走心、演员专业
    - 时间偏好：晚上 8-11 点追看
    - 预算偏好：愿意付费优质内容
    - 社交偏好：喜欢与朋友讨论剧情
    ```

- **优势**：
  - 用户可理解（解释性强）
  - 便于 A/B 测试（人工修改偏好描述，观察推荐变化）
  - 便于冷启动（新用户直接输入偏好描述）

#### 多粒度偏好表示
- 宏观：用户长期兴趣（科幻 > 悬疑 > 喜剧）
- 中观：用户当前状态（今晚心情不好，想看温暖的内容）
- 微观：用户即时需求（想看 30 分钟以内的视频）

**相关文献**：
- "Personalized Prompt Learning for In-Context Learning in Large Language Models"
- "Towards Interpretable Recommendation Systems by Learning Explicit Conversational Queries"

**工程成本**：
- 实现难度：⭐⭐ (2/5)
- 开发周期：7-10 天
- 推理成本：~2-3ms per user（一次性生成，缓存后复用）
- 无需人工标注

**收益评估**：
- 新用户冷启动 CTR：+10-20%（相比无信息情况）
- 推荐解释性：显著提升（用户理解 "为什么推荐你")
- 跨域迁移准确率：+5-15%
- 成本：低

---

### 1.2 内容特征生成与富文本表示 (Content Feature Generation)

**问题**：
- 内容特征提取成本高（需要视觉特征、NLP 特征、音频特征）
- 多模态内容的特征融合困难
- 新内容（如新上传视频）的特征提取延迟高

**LLM 解决方案**：

#### 统一的文本表示
- **多模态 → 文本**：
  - 输入：视频（标题、描述、用户评论、自动字幕、视觉截图）
  - LLM 生成：综合文本特征
    ```
    视频《三体》第 1 集：
    - 内容摘要：科幻悬疑，烧脑剧情，人类与外星文明
    - 核心关键词：三体、科幻、黑暗森林、烧脑、悬疑
    - 情感基调：紧张、压抑、希望
    - 目标受众：硬科幻爱好者、25-45 岁男性为主
    - 时长：60 分钟
    - 质量评分：9/10（用户评价高）
    ```

- **优势**：
  - 统一表示（避免多模态融合复杂度）
  - 可灵活更新（用户评论增加 → 重新生成特征）
  - 推荐时直接用文本相似度计算

#### 细粒度特征
- 不仅是内容分类（如"科幻"），还有细粒度标签：
  - 视觉风格：暗色调、CG 特效、真人演员
  - 叙事特点：单线剧情、多线并行、非线性叙述
  - 情感体验：让人思考、让人放松、让人兴奋

**相关文献**：
- "CLIP: Learning Transferable Models for Vision and Language"
- "LLaVA: A Large Language and Vision Assistant"
- "Multimodal Learning with Deep Boltzmann Machines"

**工程成本**：
- 实现难度：⭐⭐ (2/5)
- 开发周期：7-10 天
- 推理成本：~5-10ms per content（新内容上线时一次性计算，缓存）
- 无需人工标注

**收益评估**：
- 新内容覆盖率：提升（无需等待特征提取完成）
- 推荐多样性：+10-15%（细粒度特征支持）
- 冷启动准确率：+8-15%
- 成本：低（一次性计算）

---

### 1.3 个性化推理解释生成 (Personalized Recommendation Explanation)

**问题**：
- 用户不理解"为什么推荐你这个"
- 推荐系统是黑盒（用户无法验证、也无法信任）
- 无法帮助用户改进推荐（无法反馈"我不喜欢这个原因是..."）

**LLM 解决方案**：

#### 一句话解释
- 为每个推荐生成简短的解释（10-20 字）
  - 推荐：电影《三体》
  - 解释："你喜欢科幻悬疑，这部是 2024 年烧脑大作"
  - 成本：~1ms per recommendation

#### 多维度解释
- 解释不仅说"为什么推荐"，还说"从哪个角度"：
  - 内容角度："你看过《星际穿越》，这部类似的硬科幻"
  - 用户角度："基于你晚上 8-10 点的观看习惯"
  - 社交角度："你的朋友 3 人点赞了这部"
  - 时间角度："刚发布，热度很高"

#### 解释反馈
- 用户可以反馈："不喜欢这个解释"或"不喜欢这个原因"
- LLM 调整后续推荐理由

**相关文献**：
- "Interpretability and Explainability in Recommendation Systems"
- SHAP（Shapley Additive exPlanations）
- "Why Should I Trust You?: Explaining the Predictions of Any Classifier"

**工程成本**：
- 实现难度：⭐⭐ (2/5)
- 开发周期：7-10 天
- 推理成本：~1ms per recommendation
- UI 改动：需要展示空间（1-2 天前端工作）

**收益评估**：
- 用户满意度：+15-25%（能理解推荐理由）
- 点击率：+3-8%（有解释的推荐更可信）
- 用户反馈利用：+30-50%（有解释后，反馈更有效）
- 成本：低

---

### 小结：替换层方案对比

| 方向 | 实现难度 | 工程量 | 推理成本 | 收益 | 推荐优先级 |
|------|--------|-------|---------|------|-----------|
| 偏好描述 | ⭐⭐ | 7-10d | 2-3ms | 中等 (冷启 +10-20%) | ⭐⭐⭐⭐ |
| 特征生成 | ⭐⭐ | 7-10d | 5-10ms | 中等 (新内容覆盖提升) | ⭐⭐⭐⭐ |
| 解释生成 | ⭐⭐ | 7-10d | 1ms | 中等 (满意度 +15-25%) | ⭐⭐⭐⭐ MVP |

---

## 第二层：节点层（推荐核心节点替换）

**定义**：用 LLM 替换推荐系统中的**关键算法节点**（召回、排序、多样性）。  
**特点**：架构改造、成本中等、效果显著  
**ROI**：高（CTR +10-20%, 留存 +15-25%）

### 2.1 召回节点：LLM 语义理解 vs 向量召回

**问题**：
- 向量召回基于 embedding 空间距离，语义理解有限
- 对复杂查询（"我心情不好，想看温暖有趣的电影"）无法捕捉
- 跨域召回困难（用户在电商的购买历史如何推荐视频）

**LLM 解决方案**：

#### 语义理解的精准召回
- **输入**：用户实时 query（"我心情不好"） + 用户历史
- **LLM 理解**：
  - 情感状态：低迷
  - 内容需求：温暖、有趣、励志
  - 时间约束：30 分钟以内（快速治愈）

- **输出**：高精准相关内容
  - 传统向量召回可能得到：科幻、悬疑（基于历史）
  - LLM 召回会考虑实时情感：温暖系列、搞笑电影、励志电影

- **性能对比**：
  - 向量召回：通用性好，但语义理解有限
  - LLM 召回：语义理解强，但成本稍高
  - 混合召回（融合两者）：最优方案

#### 多轮对话召回
- 用户逐步表达需求：
  ```
  用户："推荐一部电影"
  系统："你最近看的类型是？"
  用户："科幻，但最近有点审美疲劳"
  系统："那推荐一个不同的风格，温暖系列？"
  用户："好，时长 2 小时以内"
  系统："推荐《当幸福来敲门》"
  ```

**相关文献**：
- "Semantic Similarity Learning for Very Short Texts"
- "Dense Passage Retrieval for Open-Domain Question Answering"
- Colbert V2 和相关工作

**工程成本**：
- 实现难度：⭐⭐⭐ (3/5)
- 开发周期：10-14 天
- 推理成本：~5-10ms per user query
- 混合方案：召回结果融合（需要精心调参）

**收益评估**：
- 召回精准度：+10-20%（vs 纯向量召回）
- 多样性：+15-20%（能理解多维度需求）
- 冷启动：+15-25%（新用户无历史，但有实时 query）
- 成本：中等

---

### 2.2 排序节点：LLM reranker vs DeepFM

**问题**：
- 深度排序模型（DeepFM、XGBoost）需要大量特征工程
- 对新内容、新用户的排序效果差（冷启动问题）
- 跨品类排序逻辑不一致

**LLM 解决方案**：

#### Pairwise LLM Reranker
- **输入**：用户 + 两个候选内容
- **LLM 判断**：哪个更适合这个用户
  ```
  用户：25 岁女性，喜欢科幻和恐怖，晚上看
  候选 A：《三体》科幻悬疑
  候选 B：《牛奶皮肤》恐怖惊悚
  
  LLM 判断：B 更适合（因为用户明确喜欢恐怖，且该用户偏好晚上看）
  ```

- **性能对比**：
  - DeepFM：AUC 0.78，需要大量特征工程
  - LLM reranker：AUC 0.80-0.82，无需特征工程
  - 优势：LLM 能理解复杂的用户-内容交互

#### Listwise LLM Reranker
- 一次性重排整个候选列表（更优化但更贵）
- 用户得到的推荐顺序更合理

**相关文献**：
- Microsoft RankLLM: "Is ChatGPT Good at Search?" (2023)
- "Learning to Rank with LLMs" (SIGIR 2024)

**工程成本**：
- 实现难度：⭐⭐⭐ (3/5)
- 开发周期：10-14 天
- 推理成本：~10-30ms for top-20 reranking
- 缓存友好度：高（热用户的排序结果缓存）

**收益评估**：
- 排序精度（AUC）：+2-4%（vs DeepFM）
- CTR：+8-15%
- 新内容覆盖：+10-20%（无需特征工程，冷启动好）
- 成本：中等（推理成本）

---

### 2.3 多样性控制与打散 (Diversity & Debias)

**问题**：
- 传统多样性控制（强制不同品类）过于简单粗暴
- 无法理解"用户想要的多样性"（不同用户需求不同）
- 新兴内容容易被压低（冷启动内容本来 embedding 不优，再被打散压低更难露脸）

**LLM 解决方案**：

#### 理解用户的多样性需求
- 有的用户喜欢"看一个系列从始至终"（不需要多样性）
- 有的用户喜欢"每次都看不同的东西"（高多样性需求）
- LLM 自动判断用户的多样性偏好

- **智能打散**：
  ```
  用户 A（电视剧爱好者）：
  - 需求：一个系列看完再看下一个
  - 推荐：[三体1,2,3,4,...]（连贯）
  
  用户 B（内容浏览者）：
  - 需求：每次都看新的
  - 推荐：[电影1, 电视剧1, 纪录片1, ...]（多样）
  ```

#### 冷启动打散平衡
- 不盲目打散新内容，而是理解"为什么用户应该看这个"
- 如果用户偏好与新内容匹配 → 不打散，直接推荐
- 如果匹配度低 → 才考虑打散（作为多样性推荐）

**相关文献**：
- "Towards Long-term Fairness in Recommendation"
- "Debiasing Recommendation through Causal Inference"

**工程成本**：
- 实现难度：⭐⭐⭐ (3/5)
- 开发周期：10-14 天
- 推理成本：~3-5ms per user ranking
- 需要定义多样性目标（通过 A/B 测试优化）

**收益评估**：
- 推荐多样性：+15-25%
- 新内容覆盖：+20-30%（冷启动打散更聪明）
- 用户体验：+10-15%（更符合个人偏好的多样性）
- 成本：低

---

### 2.4 冷启动优化 (Cold Start Problem)

**问题**：
- 新用户、新内容的推荐准确率很低
- 需要等待足够的交互数据才能精准推荐

**LLM 解决方案**：

#### 新用户冷启动
- **方法 1**：LLM 从用户描述推荐
  - 用户填写："我喜欢科幻和推理，工程师，每晚 8-10 点看"
  - LLM 直接推荐相关内容（无需等待历史数据）

- **方法 2**：LLM few-shot 学习
  - 给 LLM 5 个相似用户的喜好 + 推荐效果
  - LLM 学习模式，推荐给新用户

- **性能**：新用户 CTR 从 0.5% → 2-3%（提升 4-6 倍）

#### 新内容冷启动
- **方法**：LLM 从内容特征推荐用户
  - 新视频上传："科幻悬疑，黑暗压抑，25-40 岁男性目标"
  - LLM 识别相似用户，立即推荐

- **性能**：新内容冷启动好，更快积累交互数据

**相关文献**：
- "Meta-Learning for Recommendation Systems"
- "Few-shot Learning for Personalized Recommendation"

**工程成本**：
- 实现难度：⭐⭐ (2/5)
- 开发周期：7-10 天
- 推理成本：~2-3ms per new user/content
- 无需人工标注

**收益评估**：
- 新用户 CTR：+300-400%（相比无信息）
- 新内容曝光：+100-200%（冷启动加速）
- 成本：低

---

### 小结：节点层方案对比

| 方向 | 实现难度 | 工程量 | 推理成本 | 收益 | 推荐优先级 |
|------|--------|-------|---------|------|-----------|
| LLM 召回 | ⭐⭐⭐ | 10-14d | 5-10ms | 高 (精准度 +10-20%) | ⭐⭐⭐⭐ |
| LLM reranker | ⭐⭐⭐ | 10-14d | 10-30ms | 很高 (CTR +8-15%) | ⭐⭐⭐⭐⭐ |
| 多样性控制 | ⭐⭐⭐ | 10-14d | 3-5ms | 高 (多样性 +15-25%) | ⭐⭐⭐⭐ |
| 冷启动优化 | ⭐⭐ | 7-10d | 2-3ms | 很高 (新用户 CTR +300%) | ⭐⭐⭐⭐⭐ |

---

## 第三层：架构层（对话式推荐）

**定义**：用生成式、对话范式**完全替换**传统推荐系统架构。  
**特点**：高风险、高收益、未来方向  
**ROI**：很高但周期长（留存 +30-50%，但需 6-12 个月）

### 3.1 多轮对话推荐 (Conversational Recommendation)

**问题**：
- 单轮推荐无法捕捉用户动态需求
- 用户偏好随时间变化，一次推荐不能应对
- 用户无法反馈"这个不对"并改正

**LLM 解决方案**：

#### 对话式推荐流程
```
系统："最近想看点什么？"
用户："科幻类"
系统："比如《三体》、《星际穿越》、《星球大战》，喜欢哪个风格？"
用户："不想要太压抑的"
系统："明白，推荐《星际穿越》（希望感较强），或《阿凡达》（视觉震撼）"
用户："已经看过《星际穿越》"
系统："那推荐《天地之争》，类似的大制作科幻，但故事更乐观"
```

#### 多轮推理
- 系统不仅推荐，还进行推理：
  - 用户说"压抑" → 推理：避免黑暗压抑的情节
  - 系统追问："是的问题还是故事问题？"
  - 根据回答精细化推荐

#### 用户行为学习
- 每个对话回合，系统学习用户的新偏好
- 下次推荐时，包含本次对话学到的信息

**相关文献**：
- "Towards Conversational Recommender Systems" (SIGIR 2020)
- "Interactive Recommender Systems: A Survey of the State of the Art and Future Research Challenges"
- Alibaba: "Conversational Recommendation System"

**工程成本**：
- 实现难度：⭐⭐⭐⭐ (4/5)
- 开发周期：12-16 周
  - 对话管理框架：3 周
  - 推荐融合：3 周
  - 工程优化：4 周
  - A/B 测试：2-3 周
- 推理成本：~50-100ms per turn （多轮 LLM 推理）
- UI/UX 设计：需要对话界面（1-2 周）

**收益评估**：
- 用户留存：+30-50%（更好的推荐 → 更粘性）
- 人均推荐数：+2-3x（用户愿意多看）
- CTR：+20-40%（对话推荐效果更好）
- 成本：高（多轮推理）

**风险与缓解**：
- ❌ 推理成本高 → 使用缓存、轻量化模型
- ❌ 推理延迟 → 接受较高延迟（对话场景容忍度高）

---

### 3.2 基于推理的推荐 (Reasoning-based Recommendation)

**问题**：
- 隐式推荐（embedding）无法解释
- 用户无法验证推荐逻辑是否合理
- 难以融合外部知识（如用户评价、评论信息）

**LLM 解决方案**：

#### 显式推理链
- LLM 不仅推荐，还说出推理过程：
  ```
  推理链：
  1. 你最近看了 3 部科幻电影 → 推断你喜欢科幻
  2. 你的评分都在 8+ → 你品味较高，喜欢高质量内容
  3. 基于评论，你偏好"烧脑"而非"爽"
  4. 综合以上，推荐《三体》（高评分 + 烧脑 + 科幻）
  
  推荐：《三体》
  ```

- **优势**：用户可以验证、反馈、调整推理

#### 知识融合推理
- 融合外部知识：演员、导演、评论、热点
  ```
  推理：
  - 你喜欢演员 A（看过 3 部他的电影）
  - 演员 A 的新作品《新电影》刚上线
  - 这部电影评分 9.2（高分）
  - 你的朋友 B 也点赞了这部
  → 推荐《新电影》
  ```

**相关文献**：
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Tree of Thought: Deliberate Problem Solving with Large Language Models"

**工程成本**：
- 实现难度：⭐⭐⭐⭐ (4/5)
- 开发周期：12-16 周
- 推理成本：~20-50ms per recommendation （推理链生成）
- 需要设计推理模板和知识源

**收益评估**：
- 推荐可信度：+25-40%
- CTR：+15-25%
- 用户反馈有效性：+30-50%（用户能反馈推理过程）
- 成本：中等

---

### 3.3 LLM 作为推荐系统本体 (LLM as Recommender)

**问题**：
- 传统推荐系统需要维护大量模块（召回、排序、多样性等）
- 每个模块都是独立训练的，缺乏整体优化
- 新增需求需要修改多个模块

**LLM 解决方案**：

#### 统一的 LLM 推荐器
- **输入**：用户 query + 用户历史 + 内容库
- **输出**：推荐列表 + 解释

- **一个 LLM 完成所有任务**：
  - 理解用户需求
  - 召回相关内容
  - 排序
  - 多样性控制
  - 生成解释

- **优势**：
  - 模块化复杂度降低（单一端到端模型）
  - 易于迭代（改 prompt 而非改模型）
  - 多任务学习（一个模型学习所有技能）

#### 实施挑战
- **成本控制**：LLM 推理成本可能很高
- **精度**：需要确保推荐质量不下降

**相关文献**：
- "One Prompt for All: Towards Unified In-Context Learning"
- "Unifying Language Learning and Reasoning for Knowledge Graph Completion"

**工程成本**：
- 实现难度：⭐⭐⭐⭐⭐ (5/5)
- 开发周期：16-20 周
- 推理成本：~100-200ms per request
- 需要大量工程优化（缓存、量化、剪枝）

**收益评估**：
- 系统复杂度：大幅降低（单一模型 vs 多模块）
- 迭代速度：+5-10x（改 prompt 而非改模型）
- 效果：可与传统系统持平（需要精心设计）
- 成本：高（推理成本）

**风险**：
- ❌ 幻觉：LLM 可能推荐不存在的内容
- 缓解：严格的内容库检验

---

### 小结：架构层方案对比

| 方向 | 实现难度 | 工程量 | 推理成本 | 收益 | 推荐优先级 |
|------|--------|-------|---------|------|-----------|
| 对话推荐 | ⭐⭐⭐⭐ | 12-16w | 50-100ms | 很高 (留存 +30-50%) | ⭐⭐⭐⭐ |
| 推理推荐 | ⭐⭐⭐⭐ | 12-16w | 20-50ms | 高 (可信度 +25-40%) | ⭐⭐⭐ |
| LLM 本体 | ⭐⭐⭐⭐⭐ | 16-20w | 100-200ms | 很高 (复杂度 -80%) | ⭐⭐ (未来) |

---

## 应用场景对标

### 国际产品

| 产品 | 技术栈 | 状态 | LLM 应用 |
|------|-------|------|---------|
| Netflix | DeepFM + LLM | ✅ 探索中 | 个性化标签、解释生成 |
| Spotify | Graph 学习 + LLM | ✅ 探索中 | 播放列表生成、解释 |
| Amazon | 深度学习 + LLM | ✅ 上线 | 产品推荐、评价生成 |

### 国内产品

| 产品 | 技术栈 | 状态 | LLM 应用 |
|------|-------|------|---------|
| 抖音 | 多目标排序 + LLM | ✅ 上线 | 推荐解释、内容理解 |
| B 站 | 深度学习 + LLM | ✅ 上线 | 标签生成、推荐理由 |
| 微博 | 图学习 + LLM | ✅ 探索 | 微博推荐、热点发现 |

---

## 评估体系

### 核心指标

- **CTR (Click-Through Rate)**
  - 推荐被点击的比例
  - 目标：+8-15%（vs 基线）

- **转化率 (Conversion Rate)**
  - 点击后成交的比例
  - 目标：+5-10%

- **留存率 (Retention Rate)**
  - 用户持续使用的比例
  - 目标：+15-30%（对话推荐特别显著）

- **人均 PV (Page Views)**
  - 用户平均浏览内容数
  - 目标：+50-100%

### LLM 特定指标

- **推荐可信度**：用户对推荐的信任度（调查问卷）
- **解释有效性**：解释是否帮助用户理解推荐
- **冷启动准确率**：新用户、新内容的推荐精度

---

## 快速路径（MVP，第 1-2 个月）

1. **解释生成**：成本低，收益显著（满意度 +15-25%）
2. **冷启动优化**：提升新用户体验（CTR +300%）
3. **LLM reranker**：替换现有排序器（CTR +8-15%）

---

## 中期方向（第 3-6 个月）

1. **LLM 召回**：替换向量召回（精准度 +10-20%）
2. **多样性控制**：智能打散（多样性 +15-25%）

---

## 长期方向（第 6-12 个月）

1. **对话推荐**（前沿，需投入）
2. **推理推荐**（高可信度）

---

## 成本-效益 vs 风险 矩阵

```
        高收益 ↑
           ↑
对话推荐   |  LLM 本体
(长期)     | (探索)
           |
--------+-------- LLM reranker、冷启动
           |     (中期，重点)
解释生成   |  多样性控制
(MVP)      | (快速)
           |
        低收益 ↓

        低风险 → 高风险
```

---

## 总结

推荐 + LLM 集成路线图：

1. **第一阶段（1-2 个月）**：快速落地 MVP（解释生成 + 冷启动 + LLM reranker）
2. **第二阶段（3-6 个月）**：节点层深化（LLM 召回 + 多样性控制）
3. **第三阶段（6-12 个月）**：架构层探索（对话推荐或推理推荐）

每个阶段都有明确的收益和成本，选择适合自己的策略。
