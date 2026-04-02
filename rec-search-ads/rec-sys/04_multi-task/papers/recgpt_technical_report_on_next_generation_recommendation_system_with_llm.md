# RecGPT: Technical Report on Next-Generation Recommendation System with LLM
> 来源：arxiv/2405.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
传统推荐系统依赖 ID-based 协同过滤，面临：
- 冷启动：新用户/物品无历史交互数据
- 语义鸿沟：item ID 无语义，难以理解内容和用户意图
- 跨域迁移：不同场景（视频/电商/音乐）无法共享知识
- 自然语言交互：用户无法用语言表达偏好

## 核心方法与创新点
**RecGPT**：将 LLM 作为推荐系统的核心引擎。

**三阶段训练框架：**
```
Stage 1 - 预训练：
  LLM 在通用语料上预训练（或使用 GPT/LLaMA 底座）

Stage 2 - 推荐领域微调（SFT）：
  Input:  "用户历史：[item1描述, item2描述, ...] 用户查询：[query]"
  Output: "推荐：[itemN描述]"
  L_sft = -Σ log P(output_t | input, output_{<t})

Stage 3 - 偏好对齐（RLHF）：
  奖励：CTR/评分信号
  L_rl = E[r(item)] - β·KL(π_rl || π_sft)
```

**Semantic ID 生成：**
- 用 LLM 的 item 文本表示替代传统 Embedding Table
- item_emb = LLM_encode(title + category + description)

**混合检索：**
```
score = α·semantic_sim(user_emb, item_emb) + β·collaborative_score(CF)
```

## 实验结论
- 电商推荐数据集 HitRate@10：+12.3%（vs SASRec）
- 冷启动场景（新 item < 10次交互）：HR@10 提升 45%
- 跨域迁移（源域→目标域）：+18% vs ID-based 方法
- 模型：LLaMA-7B 微调版本

## 工程落地要点
1. **LLM 推理加速**：vLLM + PagedAttention 服务化，保证 P99 延迟 <100ms
2. **离线 Embedding 缓存**：item 文本 embedding 离线计算并索引（FAISS）
3. **增量更新**：新 item 加入时只需计算其 embedding，无需重训
4. **两阶段架构**：LLM 负责语义理解，轻量 MLP 负责最终打分
5. **成本控制**：仅对 top-k 候选调用 LLM 重排，召回阶段用向量检索

## 常见考点
**Q1: RecGPT 相比传统协同过滤最大的优势？**
A: 语义理解能力：能理解 item 文本内容和用户意图，天然解决冷启动；支持自然语言交互；跨域知识迁移（语言模型的通用知识）。

**Q2: LLM 在推荐系统中的主要瓶颈？**
A: ①推理延迟：自回归解码比 embedding lookup 慢 100x ②成本：大模型 GPU 成本高 ③幻觉：LLM 可能生成不存在的 item ④个性化弱：预训练不包含用户协同信号。

**Q3: 如何将 RLHF 应用于推荐系统？**
A: 奖励模型用点击/购买/评分等真实反馈；用 PPO 微调 LLM，使其生成更符合用户行为偏好的推荐；KL 散度惩罚防止偏离 SFT 模型太远。

**Q4: Semantic ID vs Collaborative ID，如何选择？**
A: 冷启动/内容理解/跨域：Semantic ID 优；有丰富交互数据/高频物品：Collaborative ID 优。最佳实践是融合：CF embedding + semantic embedding 拼接。

**Q5: LLM 推荐系统的幻觉问题怎么解决？**
A: ①约束解码：解码时只允许生成已知 item 的 token（constrained beam search）②生成后验证：用 item 库做后处理过滤 ③Retrieval-Augmented：给 LLM 提供候选集，只做重排。
