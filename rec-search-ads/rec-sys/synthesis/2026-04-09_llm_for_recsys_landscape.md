# Synthesis: LLM for Recommendation Systems — 技术全景
> Date: 2026-04-09 | Papers/Projects: ThinkRec, RecAI, OpenOneRec, MiniOneRec, TRL, Awesome-LLM-for-RecSys

## 1. 技术演进 (Technical Evolution)

### Phase 1: LLM as Feature Extractor
- 用 LLM 生成 text embeddings 作为推荐特征
- 代表：RecAI 的 RecLM-emb

### Phase 2: LLM as Ranker/Agent
- LLM 直接参与排序或作为 agent 调用传统模型
- 代表：RecAI 的 InteRecAgent (LLM orchestrates traditional models as tools)

### Phase 3: LLM as Generative Recommender
- 端到端生成推荐结果，SID 表示 item
- 代表：MiniOneRec (SFT + GRPO), OpenOneRec (Itemic Tokens + Foundation Model)

### Phase 4: LLM with Reasoning
- Chain-of-thought reasoning 提升推荐质量和可解释性
- 代表：ThinkRec (System 2 reasoning, +56.54% explanation quality)

## 2. 核心架构模式

### Pattern A: LLM-as-Agent (RecAI)
```
User Query → LLM Agent → [Tool: Recall Model] → [Tool: Rank Model] → Response
优点：复用现有模型，可解释
缺点：延迟高，依赖 tool quality
```

### Pattern B: LLM-as-Generator (MiniOneRec/OpenOneRec)
```
User History → LLM → SID Tokens → Item
优点：端到端优化，跨域迁移
缺点：SID 设计复杂，训练成本高
```

### Pattern C: LLM-with-Reasoning (ThinkRec)
```
User + Item → Reasoning Trace → Expert Routing → Recommendation + Explanation
优点：可解释性强，准确度高
缺点：推理延迟，reasoning trace 构建成本
```

## 3. 工业实践对比

| Approach | Latency | Accuracy | Interpretability | Training Cost |
|----------|---------|----------|------------------|---------------|
| LLM-as-Agent | High | Depends on tools | High | Low |
| LLM-as-Generator | Medium | High (+26.8% Recall) | Low | Very High |
| LLM-with-Reasoning | High | High (+56.54% explain) | Very High | Medium |

### 选型建议
- **已有成熟推荐系统**: RecAI agent 模式，渐进式 LLM 增强
- **新建推荐系统**: OpenOneRec/MiniOneRec generative 模式
- **可解释性要求高**: ThinkRec reasoning 模式

## 4. 面试考点 (Interview Points)

**Q1: LLM 在推荐中的三种角色？**
A: (1) Feature extractor: 生成 embeddings; (2) Agent: 调用传统模型; (3) Generator: 端到端生成推荐。趋势从 (1) → (3) 演进。

**Q2: Generative Recommendation 的关键技术？**
A: SID 构建 (RQ-VAE), SFT 训练 (next-token prediction), RL 对齐 (GRPO)。OpenOneRec 用 Itemic Tokens 将 item 视为独立模态。

**Q3: 如何用 RLHF/GRPO 优化推荐？**
A: TRL 提供 SFT→RLHF pipeline。MiniOneRec 用 GRPO 做 recommendation-oriented RL。关键：reward 定义（CTR, conversion, satisfaction）。

**Q4: LLM 推荐的可解释性？**
A: ThinkRec: System 2 reasoning + dynamic expert routing。+56.54% explanation quality。关键：synthetic reasoning traces + personalized reasoning paths。
