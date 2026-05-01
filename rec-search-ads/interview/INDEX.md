# 面试准备知识库导航

## 📊 领域概览

| 分类 | 文档数 | 描述 |
|------|--------|------|
| **Synthesis** | 6篇 | 提炼总结 |
| **Cards** | 8篇 | 知识卡片 |
| **总计** | 14篇 | - |

---

## 📝 Synthesis 总结文档

- [知识卡片 #004：Query Rewriting & RAG 优化](./synthesis/QueryRewriting与RAG优化.md)
- [知识卡片 #003：oCPC/oCPA 智能出价系统](./synthesis/oCPC_oCPA智能出价系统.md)
- [知识卡片 #002：双塔召回模型](./synthesis/双塔召回模型.md)
- [知识卡片 #001：多任务学习（MMoE / PLE）](./synthesis/多任务学习_MMoE_PLE.md)
- [知识卡片 #005：推荐系统全链路串联](./synthesis/推荐系统全链路串联.md)
- [知识卡片：面试项目讲故事 - 30个精选案例的核心方法论](./synthesis/面试项目讲故事_30个案例.md)

## 🃏 知识卡片

- [面试知识卡片库 索引](./cards/INDEX.md)
- [A/B实验 分层知识卡片](./cards/ab_test_cards.md)
- [广告系统 分层知识卡片](./cards/ads_system_cards.md)
- [Auto Bidding 分层知识卡片](./cards/auto_bidding_cards.md)
- [CTR预估 分层知识卡片](./cards/ctr_prediction_cards.md)
- [LLM工程 分层知识卡片](./cards/llm_engineering_cards.md)
- [推荐系统 分层知识卡片](./cards/rec_sys_cards.md)
- [搜索算法 分层知识卡片](./cards/search_cards.md)

---

## 🆕 2026 Q1 更新（2026-04-04）

### 新增文件
- [热题榜单 2026-04-04](./热题榜单_20260404.md) — 25道最新精选题，TOP 10 + 系统设计题 + 开放高分题
- [面试官最想听到什么](./面试官最想听到什么.md) — 高分答案框架（Before/After+工程约束+踩坑）

### 题库更新
- `qa-bank.md` — 追加 2026-03-28 至 2026-04-03 共 25 道题（5278 → 6579 行）

### 知识卡片更新
- `cards/rec_sys_cards.md` — 新增：Mamba SSM、HSTU ReLU Attention、MTGR双流融合、Semantic ID演进、PROMISE Test-Time Compute、O1 Embedder
- `cards/ads_system_cards.md` — 新增：生存分析延迟CVR、Causal Bidding、BundleNet、FPA+GBS出价分布、Shapley归因

### 面经收录
- [面经：月之暗面（Kimi）模型算法工程师](./面经-月之暗面-模型算法工程师.md) — LLM 推理优化方向 5 题（Prefill/Decode、OOM 排查、FLOPs 计算、PagedAttention、FlashAttention）

### 模拟面试更新
- `mock-interview-llm.md` — 新增 Q13-Q15：推理两阶段、OOM 排查、MHA+MLP FLOPs 计算

### 核心主题覆盖（新增）
- **序列建模**：Mamba SSM、HSTU、Differentiable Semantic ID、UniGRec Soft ID
- **延迟CVR**：生存分析、软标签重加权
- **生成式广告**：EGA端到端框架
- **因果出价**：Causal Bidding、GBS分布出价、FPA Bid Shading
- **Test-Time Compute**：PROMISE、PRM、Beam Search推荐
- **推理增强检索**：O1 Embedder、Rank1、REARANK
