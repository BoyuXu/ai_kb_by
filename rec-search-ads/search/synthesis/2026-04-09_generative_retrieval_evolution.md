# Synthesis: Generative Retrieval — 技术演进与工业实践
> Date: 2026-04-09 | Papers: IntRR, FORGE, MiniOneRec, OpenOneRec

## 1. 技术演进 (Technical Evolution)

### Phase 1: Semantic ID Construction
- 传统方法：Atomic IDs, hierarchical clustering
- FORGE: 多模态特征 + collision mitigation，250M items scale
- MiniOneRec: RQ-VAE quantization → compact SID sequences

### Phase 2: Autoregressive Generation Optimization  
- 瓶颈：Multi-token SID 需要多步 autoregressive generation
- IntRR: Recursive-Assignment Network (RAN) → 单 token 预测
- 效果：Consistent lowest latency across search spaces

### Phase 3: Foundation Model Integration
- OpenOneRec: Itemic Tokens 将 item 视为独立模态
- Qwen3-based 1.7B/8B 模型，+26.8% Recall@10
- Zero-shot cross-domain transfer

## 2. 核心公式 / Core Formulations

### SID Construction (FORGE)
```
SID(item) = Quantize(f_text(item) ⊕ f_image(item) ⊕ f_behavior(item))
Quality(SID) ∝ Coverage × Discriminability × Collision_Avoidance
```

### Recursive Assignment (IntRR)
```
h_item = RAN(SID_hierarchy, UID_anchor)
P(item | query) = softmax(W · h_item)  // Single-token prediction
```

### Generative Recommendation (MiniOneRec)
```
Loss = L_SFT(next_token) + α · L_GRPO(reward)
Inference: Constrained beam search over valid SID space
```

## 3. 工业实践 (Industrial Practices)

| System | Scale | Key Metric |
|--------|-------|------------|
| FORGE (Taobao) | 250M items, 14B interactions | +0.35% transactions |
| OpenOneRec (Kuaishou) | 200K users, multi-domain | +26.8% Recall@10 |
| IntRR | Variable | Lowest latency |

### 部署要点
- SID collision 在 billion-scale 是关键问题 → FORGE 的 mitigation 策略
- Serving latency: IntRR 的 single-token 方案 vs traditional multi-step
- Foundation model: 需要 1.7B+ 参数才能 capture 足够语义

## 4. 面试考点 (Interview Points)

**Q1: Generative Retrieval vs Traditional Two-Tower?**
A: GR 将 retrieval 转化为 sequence generation，避免 ANN 索引维护。优势：端到端优化，语义理解更深。劣势：推理延迟、SID 设计复杂。

**Q2: SID 设计的关键挑战？**
A: (1) Collision: 不同 item 映射到相同 SID → FORGE 的 multimodal + collision mitigation; (2) 长度：multi-token SID 导致 autoregressive 延迟 → IntRR 的 single-token RAN; (3) 评估：训练 GR 代价大 → FORGE 的 training-free metrics

**Q3: Foundation Model 如何用于推荐？**
A: OpenOneRec 用 Itemic Tokens 将 item 视为模态，像处理图片一样处理 item。MiniOneRec 用 GRPO 对齐推荐目标。关键：需要足够规模 (1.7B+) 和推荐专用 RL。

**Q4: Generative Retrieval 的 serving 优化？**
A: IntRR: RAN single-token prediction 消除 autoregressive 瓶颈。MiniOneRec: constrained beam search 确保生成有效 SID。实践中需 SID → item 的高效映射表。

---

## 相关概念

- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/vector_quantization_methods|向量量化方法]]
