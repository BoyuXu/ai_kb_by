# Semantic Search At LinkedIn: LLM-based Semantic Search Framework

> 来源：arxiv | 领域：search | 学习日期：20260328
> 论文：https://arxiv.org/abs/2602.07309

## 问题定义

LinkedIn 搜索（AI Job Search + AI People Search）需要解决三个核心矛盾：
1. **语义 vs 效率**：LLM 语义理解强但推理慢，传统关键词检索快但语义弱
2. **相关性 vs 参与度**：搜索结果需同时优化 NDCG（相关性）和用户点击/互动（参与度）
3. **规模约束**：LinkedIn 规模每日亿级搜索请求，P99 延迟严格约束

## 核心方法与创新点

### 系统架构三层
```
用户查询
   ↓
[Embedding-based Retrieval]  ← 向量召回
   ↓
[LLM Relevance Judge]        ← 大模型相关性打分（Teacher）
   ↓
[Small LM Re-ranker]         ← 多教师蒸馏的小模型重排
   ↓
最终排序结果
```

### 关键技术
1. **LLM 相关性法官（Relevance Judge）**：大 LLM 作为离线 Teacher，对 query-doc pair 打相关性分
2. **多教师蒸馏（Multi-teacher Distillation）**：同时学习相关性 Teacher + 参与度 Teacher，SLM 联合优化
3. **Prefill-oriented 推理架构**：
   - 将排序中的共享前缀（query 侧）做 KV cache 预计算
   - 文档侧复用 prefill，减少重复计算
4. **模型剪枝 + 上下文压缩**：在固定延迟预算下最大化 NDCG
5. **Text-Embedding 混合交互**：稀疏 + 稠密特征联合建模

### 核心指标
- 排序吞吐量提升 **75×**（相比朴素 LLM 排序）
- 延迟与传统方法 comparable
- NDCG 达到 Teacher LLM 的 near-teacher 水平

## 实验结论

- LLM-based 语义排序相比关键词排序：NDCG 显著提升
- 多教师蒸馏（相关性 + 参与度）优于单一目标蒸馏
- Prefill 优化 + 模型剪枝后：75× 吞吐提升，P99 延迟满足 SLA
- 线上 A/B 实验：Job Search CTR 和 People Search 质量均有显著提升
- 这是业界**首个大规模生产 LLM 排序系统**效率可比传统方案的案例

## 工程落地要点

1. **KV Cache 复用**：Query 侧 prefill 缓存是吞吐优化的核心，减少 O(N) 次重复计算到 O(1)
2. **多教师蒸馏流程**：
   - 离线：大 LLM 生成相关性软标签 + 用户行为信号生成参与度标签
   - 在线：SLM 联合学习两类信号
3. **上下文压缩**：对候选文档做摘要/截断，降低 token 数量，但需保留关键匹配信号
4. **延迟预算分配**：召回 < 5ms，重排 < 20ms，整体搜索 < 100ms
5. **灰度发布**：先在 Job Search 验证，再推 People Search，避免全量风险

## 面试考点

**Q1: LinkedIn 语义搜索的整体架构是什么？**
A: Embedding 召回 → LLM 相关性法官（Teacher）→ 多教师蒸馏 SLM 重排，核心是用小模型学大模型的语义判断能力

**Q2: 多教师蒸馏如何平衡相关性和参与度？**
A: 两个 Teacher 分别输出 soft labels，SLM 用加权 KL 散度联合学习，超参控制两个目标权重

**Q3: Prefill-oriented 推理为什么能大幅提升吞吐？**
A: LLM 排序中 query 对所有候选文档相同，传统逐个推理重复计算 query 侧 KV；prefill 优化将 query KV cache 预计算并复用，吞吐从 O(N) 降至 O(1) + O(N×doc_len)

**Q4: 如何评估语义搜索效果？**
A: 离线：NDCG@K、MAP；在线：CTR、Apply Rate（岗位申请率）、搜索成功率；需对齐离线-在线一致性
