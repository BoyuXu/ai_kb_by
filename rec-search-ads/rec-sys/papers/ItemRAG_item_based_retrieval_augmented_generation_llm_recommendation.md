# ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

LLM 做推荐时面临两大问题：
1. **知识截止**：LLM 不了解最新上架物品
2. **幻觉问题**：LLM 可能推荐不存在的物品

传统 RAG 是文档级检索，推荐场景需要 **物品级** 精准检索，且物品数量可达亿级。

$$\text{Recommend}(u) = \text{LLM}(\text{query}(u), \text{Retrieve}(u, \mathcal{I}))$$

## 核心方法与创新点

**ItemRAG** 将 RAG 范式适配到推荐系统：

1. **物品感知检索器（Item-Aware Retriever）**：
   - 构建物品 dense index，用对比学习训练检索器
   - 检索信号：用户历史 + 当前意图 → 召回 Top-K 候选物品
   
$$s(u, i) = \text{sim}(f_u(\text{history}_u), f_i(\text{item}_i))$$

2. **结构化物品上下文（Structured Item Context）**：
   - 不直接将物品 raw 文本塞入 prompt
   - 构造结构化卡片：`{title, category, price_range, rating, key_features}`
   - 压缩 token 用量，保留关键信息

3. **推荐感知生成（Recommendation-Aware Generation）**：
   - 训练 LLM 识别物品卡片格式
   - 输出格式化推荐列表而非自由文本
   - 加入 **物品 ID 锚点**，防止幻觉

4. **迭代精化**：
   - 第一轮：宽泛检索 + 粗粒度推荐
   - 第二轮：用粗推荐结果重检索 + 精细化

## 实验结论

- NDCG@10: +8.7% vs 传统 LLM 推荐（无 RAG）
- 幻觉率（推荐不存在物品）从 23% 降至 **<1%**
- 检索 Recall@50 达 78.3%，覆盖 95% 最终推荐物品
- 推理延迟：~200ms（含检索），生产可用

## 工程落地要点

- 物品 Index 需支持增量更新（新品秒级入库）
- 结构化卡片模板需按业务场景定制（电商 vs 内容 vs 本地生活）
- 检索器与 LLM 联合微调效果优于分开训练
- Token 预算控制：Top-K 物品卡片总 token ≤ 2048

## 面试考点

1. **Q**: LLM 推荐的主要问题是什么？ItemRAG 如何解决？  
   **A**: 知识截止 + 幻觉。ItemRAG 通过物品级 RAG：实时检索候选 + 结构化 prompt + ID 锚点防幻觉。

2. **Q**: 物品 RAG 与文档 RAG 的主要区别？  
   **A**: 物品 RAG 需要亿级 Index、结构化卡片格式、推荐排序输出（而非文本生成）、ID 锚点防幻觉。

3. **Q**: 如何评估 RAG 质量？  
   **A**: 检索层：Recall@K；推荐层：NDCG/Hit Rate；幻觉：推荐物品存在率；端到端：在线 CTR/转化。
